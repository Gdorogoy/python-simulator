"""RTL handoff trainer: PPO + PID teacher distillation + controller->policy handoff stage. See app/training/rtl-handoff.md."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow_isaac.db")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn

from app.control.los import velocity_to_action
from app.environmental.base_drone_env import VEL_SCALE  # obs-normalisation constant, needed outside run() too
from app.guidance.train import ActorCritic, compute_gae, load_bc_checkpoint
from app.training.config import WEIGHT_DECAY

PHASES = ("settle", "critic_warmup", "head_only", "full")
GAIN_NAMES = ("kp_pos", "kd_pos", "ki_pos", "kp_att", "kd_att", "ki_att", "kp_yaw", "kd_yaw", "ki_yaw")
NAN = float("nan")


# ===== Model =====

class RTLActorCritic(ActorCritic):
    """ActorCritic with a separate critic MLP. forward() returns the usual (mean, std, value) triple, so
    diagnostics/ONNX code that only needs the actor works unchanged. The inherited critic_head is unused."""

    def __init__(self, obs_dim, action_dim, action_low, action_high, hidden=64, num_hidden_layers=4,
                 critic_hidden=128, critic_layers=2, log_std_min=-3.5, log_std_max=-2.0):
        super().__init__(obs_dim, action_dim, action_low, action_high, hidden=hidden,
                         num_hidden_layers=num_hidden_layers, dropout=0.0,
                         log_std_min=log_std_min, log_std_max=log_std_max)
        layers, d = [], obs_dim
        for _ in range(critic_layers):
            layers += [nn.Linear(d, critic_hidden), nn.Tanh()]
            d = critic_hidden
        layers.append(nn.Linear(d, 1))
        self.value_net = nn.Sequential(*layers)
        with torch.no_grad():
            self.value_net[-1].weight.mul_(0.1)
            self.value_net[-1].bias.zero_()

    def effective_std(self):
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(self.actor_log_std) + 1)
        return torch.exp(log_std)

    def forward(self, obs):
        x = torch.relu(self.shared(obs))
        return self.actor_mean(x), self.effective_std(), self.value_net(obs).squeeze(-1)

    def actor_state_dict(self):
        """Plain-ActorCritic-compatible state dict (everything except the separate critic)."""
        return {k: v.detach().clone() for k, v in self.state_dict().items() if not k.startswith("value_net.")}

    def actor_params(self):
        return list(self.shared.parameters()) + list(self.actor_mean.parameters()) + [self.actor_log_std]

    def critic_params(self):
        return list(self.value_net.parameters())


def set_initial_std(model: RTLActorCritic, std: float):
    lo, hi = model.log_std_min, model.log_std_max
    target = min(max(math.log(std), lo + 1e-3), hi - 1e-3)
    if abs(target - math.log(std)) > 1e-6:
        print(f"[rtl] --init-std {std} is outside [{math.exp(lo):.4f}, {math.exp(hi):.4f}] -> using {math.exp(target):.4f}")
    p = math.atanh(min(max(2.0 * (target - lo) / (hi - lo) - 1.0, -0.999), 0.999))
    with torch.no_grad():
        model.actor_log_std.fill_(p)


def log_prob_raw(model, mean, std, raw):
    """Density of the action actually sent to the env (tanh + affine rescale), identical to
    ActorCritic.get_action_and_value, so rollout and update log-probs are consistent."""
    logp = (-(raw - mean).pow(2) / (2 * std.pow(2)) - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
    squashed = torch.tanh(raw)
    half = 0.5 * (model.action_high - model.action_low)
    return logp - torch.log(half * (1 - squashed.pow(2)) + 1e-6).sum(-1)


def gaussian_entropy(std):
    return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum()


# ===== Teacher =====

DEFAULT_BC_CHECKPOINT = "app/control/pretrained_bc_dagger.pt"

class PidTeacher:
    """Stateful batched per-distance PID, queried on the states the STUDENT visits (DAgger-style labels).
    Gain selection is vectorised (one table lookup per reset batch) instead of per-env python loops."""

    def __init__(self, env, gains_by_dist: dict):
        from app.control.torch_pid import TorchPIDController, wrap_angle
        from app.training.base_training_isaac import _read_kinematics

        self._read_kinematics = _read_kinematics
        self.env = env
        u = env.unwrapped
        self.pid = TorchPIDController(u.num_envs, u.device, **next(iter(gains_by_dist.values())))
        self.key_dists = torch.tensor([float(k) for k in gains_by_dist], device=u.device)
        rows = []
        for g in gains_by_dist.values():
            rows.append([float(g[n]) if n in g else float(getattr(self.pid, n)[0].item()) for n in GAIN_NAMES])
        self.table = torch.tensor(rows, dtype=torch.float32, device=u.device)
        self.low = torch.as_tensor(u.single_action_space.low, device=u.device)
        self.high = torch.as_tensor(u.single_action_space.high, device=u.device)
        self.reset(None)

    def reset(self, env_ids):
        u = self.env.unwrapped
        ids = torch.arange(u.num_envs, device=u.device) if env_ids is None else env_ids
        sd = u._start_dist[ids].unsqueeze(-1)
        idx = torch.argmin((sd - self.key_dists.unsqueeze(0)).abs(), dim=-1)
        gains = self.table[idx]
        for j, name in enumerate(GAIN_NAMES):
            getattr(self.pid, name)[ids] = gains[:, j]
        self.pid.reset(ids)

    @torch.no_grad()
    def label(self):
        """Teacher action for the env's CURRENT state, normalised to [-1, 1] (same space as tanh(mean))."""
        pos, vel, ang_vel, roll, pitch, yaw, target = self._read_kinematics(self.env)
        a = self.pid.compute_action(pos, vel, ang_vel, roll, pitch, yaw, target,
                                    self.env.unwrapped._desired_yaw_w, dt=1 / 240)
        return torch.clamp(2.0 * (a - self.low) / (self.high - self.low) - 1.0, -1.0, 1.0)


def make_los_label(env, teacher: PidTeacher, max_speed: float, accel: float | None = None, cruise_gain: float = 1.0,
                   stop_speed: float = 0.5, max_tilt: float | None = None):
    """--teacher los: obs -> action label. Fly the straight line of sight at the braking-limited speed
    min(max_speed, sqrt(2*accel*dist)): fast while far, slowing so it arrives near the target at ~stop_speed instead of
    passing through. Velocity tracking on all three axes (xy -> tilt, z -> thrust / cos(tilt)) with FIXED attitude gains:
    the short-range PID set, not the per-start-distance set PidTeacher picks (with the handoff that is the 150-250 m set,
    kp_att ~0.005: no attitude authority at all). Same output space as PidTeacher.label(). The obs argument is ignored;
    the state is read from the sim like the PID does."""
    u = env.unwrapped
    pid = teacher.pid
    max_tilt = pid.max_tilt_rad if max_tilt is None else max_tilt   # the PID's own cap (~17 deg) is too gentle to stop from speed
    if accel is None:
        if not 0.0 < max_tilt <= math.pi / 2:       # tan() < 0 past 90 deg: sqrt(2 * accel * dist) is NaN and so is every label
            raise SystemExit(f"LOS teacher: tilt cap {math.degrees(max_tilt):.0f} deg needs 0 < cap <= 90 deg (the braking accel is "
                             f"0.5*g*tan(cap)); pass --teacher-accel to use a larger cap")
        accel = 0.5 * 9.81 * math.tan(max_tilt)
    g = dict(zip(GAIN_NAMES, teacher.table[int(torch.argmin(teacher.key_dists))].tolist()))
    kp_att, kd_att, kp_yaw, kd_yaw = g["kp_att"], g["kd_att"], g["kp_yaw"], g["kd_yaw"]
    hover = float(u._hover_thrust)
    print(f"[rtl] LOS teacher attitude gains (short-range set): kp_att={kp_att:.3f} kd_att={kd_att:.3f} "
          f"kp_yaw={kp_yaw:.3f} kd_yaw={kd_yaw:.3f}; velocity gain {cruise_gain}, tilt cap {math.degrees(max_tilt):.0f} deg")

    @torch.no_grad()
    def los_label(obs):
        pos, vel, ang_vel, roll, pitch, yaw, target = teacher._read_kinematics(env)
        to_target = target - pos
        dist = torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)
        v_cmd = torch.sqrt(2 * accel * dist).clamp(max=max_speed).clamp_min(min(stop_speed, max_speed))   # stop_speed >= max_speed: never brake
        v_des = to_target / dist * v_cmd
        if getattr(u, "_runner", False):
            # moving target (acro runner): brake in the TARGET's frame, i.e. add its velocity. Without this the braking profile settles
            # at the distance where sqrt(2*accel*d) = runner speed and trails it forever. Total capped at max_speed.
            v_des = v_des + u._runner_vel_w
            v_des = v_des * torch.clamp(max_speed / torch.linalg.norm(v_des, dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)
        raw, des_roll, des_pitch = velocity_to_action(                  # shared maths: app/control/los.py
            v_des, vel, ang_vel, roll, pitch, yaw, u._desired_yaw_w, hover_thrust=hover, kp_att=kp_att, kd_att=kd_att,
            kp_yaw=kp_yaw, kd_yaw=kd_yaw, max_tilt=max_tilt, cruise_gain=cruise_gain)
        los_label.last_des = (des_roll, des_pitch)

        a = torch.clamp(raw, pid.action_low, pid.action_high)
        return torch.clamp(2.0 * (a - teacher.low) / (teacher.high - teacher.low) - 1.0, -1.0, 1.0)
    return los_label


# ===== Handoff stage (controller -> policy) =====

SPEED_BUCKETS = ((0.10, 0.25), (0.25, 0.75), (0.75, 1.00))   # fractions of max speed
SPEED_BUCKET_PROBS = (0.20, 0.40, 0.40)


def install_handoff_stage(u, *, max_speed, prefix_low, prefix_high, lateral_frac, tilt_deg, dist_low, dist_high,
                          spawn_low=None, spawn_high=None, gains_by_dist=None,
                          teacher_accel=None, teacher_cruise_gain=1.0, leg_max_elev_deg=15.0, recenter=True):
    """Wrap u._reset_idx so every episode starts like a real controller->policy handoff. See rtl-handoff.md "Handoff stage"."""
    import isaaclab.utils.math as math_utils
    from app.control.torch_pid import TorchPIDController, wrap_angle

    dev = u.device
    # handoff fires when the real distance to target <= h (h ~ U(dist_low, dist_high)); _prefix_left is only a fallback timer
    u._prefix_left = torch.zeros(u.num_envs, dtype=torch.long, device=dev)
    u._handed_off = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
    u._stagger_next = False        # set True by the training loop around its own full resets (see _reset_idx)
    u._handoff_h = torch.zeros(u.num_envs, device=dev)
    u._handoff_snap_enabled = spawn_low is not None and spawn_high is not None
    edges = torch.tensor([b[0] for b in SPEED_BUCKETS] + [SPEED_BUCKETS[-1][1]], device=dev)
    cum = torch.tensor([SPEED_BUCKET_PROBS[0], SPEED_BUCKET_PROBS[0] + SPEED_BUCKET_PROBS[1]], device=dev)
    orig_reset = getattr(u, "_handoff_orig_reset", None) or u._reset_idx      # re-installing replaces the wrapper, it does not stack
    u._handoff_orig_reset = orig_reset
    u._handoff_cfg = dict(max_speed=max_speed, prefix_low=prefix_low, prefix_high=prefix_high,
                          lateral_frac=lateral_frac, tilt_deg=tilt_deg, spawn_low=spawn_low, spawn_high=spawn_high)

    carrot_pid = None
    carrot_key_dists = carrot_gain_table = None
    if u._handoff_snap_enabled:
        assert gains_by_dist, "install_handoff_stage: gains_by_dist is required when spawn_low/spawn_high are set"
        carrot_pid = TorchPIDController(u.num_envs, dev, **next(iter(gains_by_dist.values())))
        if teacher_accel is None:
            # plan the ramp at half the tilt-limited accel (g*tan(max_tilt)); the velocity loop lags
            teacher_accel = 0.5 * 9.81 * math.tan(carrot_pid.max_tilt_rad)
        u._handoff_cfg["teacher_accel"] = teacher_accel
        # carrot gains re-picked every step from the live remaining distance (fixed close-range gains caused roll crashes)
        carrot_key_dists = torch.tensor([float(k) for k in gains_by_dist], device=dev)
        rows = []
        for g in gains_by_dist.values():
            rows.append([float(g[n]) if n in g else float(getattr(carrot_pid, n)[0].item()) for n in GAIN_NAMES])
        carrot_gain_table = torch.tensor(rows, dtype=torch.float32, device=dev)
        u._prof_vpeak = torch.zeros(u.num_envs, device=dev)
        u._prof_dup = torch.zeros(u.num_envs, device=dev)
        u._prof_dcruise = torch.zeros(u.num_envs, device=dev)
        u._prof_prefix_m = torch.zeros(u.num_envs, device=dev)
        u._prof_spawn_local = torch.zeros(u.num_envs, 3, device=dev)
        u._prof_unit = torch.zeros(u.num_envs, 3, device=dev)

    def _reset_idx(self, env_ids):
        orig_reset(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)
        origins = self._terrain.env_origins[env_ids]
        spawn_local = self._prev_position[env_ids]                       # set by the original reset
        to_target = (self._desired_pos_w[env_ids] - origins) - spawn_local
        unit = to_target / torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)

        # move the target so the distance left at handoff is U(dist_low, dist_high)
        handoff_dist = torch.empty(n, device=dev).uniform_(dist_low, dist_high)
        if u._handoff_snap_enabled:
            spawn_dist = torch.empty(n, device=dev).uniform_(spawn_low, spawn_high)
            spawn_dist = torch.maximum(spawn_dist, handoff_dist + 0.5)   # PID always flies >= 0.5 m
            prefix_m = spawn_dist - handoff_dist
        else:
            prefix_m = torch.empty(n, device=dev).uniform_(prefix_low, prefix_high)
        total = handoff_dist + prefix_m
        if u._handoff_snap_enabled:
            # clamp the leg's elevation: target stays above the 0.5 m floor and climbs stay followable
            uz_max = math.sin(math.radians(leg_max_elev_deg))
            uz_min = -(spawn_local[:, 2] - 0.5) / total
            uz = torch.minimum(torch.maximum(unit[:, 2], uz_min), torch.full_like(uz_min, uz_max))
            xy = unit[:, :2] / torch.linalg.norm(unit[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
            unit = torch.cat([xy * torch.sqrt((1.0 - uz ** 2).clamp_min(0.0)).unsqueeze(-1), uz.unsqueeze(-1)], dim=-1)
        if u._handoff_snap_enabled and recenter:
            # full-mode obs has absolute position: shift the spawn so the handoff point lands at x=y=0
            spawn_local = spawn_local.clone()
            spawn_local[:, :2] = -unit[:, :2] * prefix_m.unsqueeze(-1)
        target_local = spawn_local + unit * total.unsqueeze(-1)
        target_local[:, 2] = target_local[:, 2].clamp(min=0.5)           # same floor as the env's own sampler
        self._desired_pos_w[env_ids] = target_local + origins
        self._start_dist[env_ids] = total
        self._prev_distance[env_ids] = total
        self._handoff_h[env_ids] = handoff_dist
        self._handed_off[env_ids] = False

        b = (torch.rand(n, device=dev).unsqueeze(-1) >= cum).sum(-1)     # bucket index 0..2
        frac = edges[b] + (edges[b + 1] - edges[b]) * torch.rand(n, device=dev)
        speed = frac * max_speed
        self._spawn_speed[env_ids] = speed

        pos0 = spawn_local                                              # staggered starts move this (below)
        if u._handoff_snap_enabled:
            # spawn at rest and level; bucket speed/tilt are imposed at the handoff snap
            vel = torch.zeros(n, 3, device=dev)
            quat = torch.zeros(n, 4, device=dev)
            quat[:, 0] = 1.0  # identity (w,x,y,z)

            vpeak_tri = torch.sqrt(torch.clamp(teacher_accel * prefix_m + 0.5 * speed ** 2, min=0.0))
            vpeak = torch.clamp(vpeak_tri, max=max_speed)
            d_up = vpeak ** 2 / (2 * teacher_accel)
            d_down = torch.clamp(vpeak ** 2 - speed ** 2, min=0.0) / (2 * teacher_accel)
            d_cruise = torch.clamp(prefix_m - d_up - d_down, min=0.0)
            self._prof_vpeak[env_ids] = vpeak
            self._prof_dup[env_ids] = d_up
            self._prof_dcruise[env_ids] = d_cruise
            self._prof_prefix_m[env_ids] = prefix_m
            self._prof_spawn_local[env_ids] = spawn_local
            self._prof_unit[env_ids] = unit
            carrot_pid.reset(env_ids)
            # fallback timer from the ramp/cruise/ramp times (not prefix_m / handoff speed)
            t_up = vpeak / teacher_accel
            t_cruise = d_cruise / vpeak.clamp_min(1e-6)
            t_down = torch.clamp(vpeak - speed, min=0.0) / teacher_accel
            # only a FALLBACK now (the position trigger normally fires first): 2.5x the ideal time + 10 s
            self._prefix_left[env_ids] = torch.ceil((2.5 * (t_up + t_cruise + t_down) + 10.0) / self.step_dt).long()
            if self._stagger_next and n == self.num_envs:
                # staggered start (training, full resets only): random point in TIME along the leg, see rtl-handoff.md
                tau = torch.rand(n, device=dev) * (t_up + t_cruise + t_down)
                in_up = tau < t_up
                in_cr = (~in_up) & (tau < t_up + t_cruise)
                u_dn = (tau - t_up - t_cruise).clamp_min(0.0)
                s0 = torch.where(in_up, 0.5 * teacher_accel * tau ** 2,
                                 torch.where(in_cr, d_up + vpeak * (tau - t_up),
                                             d_up + d_cruise + vpeak * u_dn - 0.5 * teacher_accel * u_dn ** 2))
                s0 = torch.minimum(s0, prefix_m)
                v0 = torch.where(in_up, teacher_accel * tau,
                                 torch.where(in_cr, vpeak, torch.maximum(vpeak - teacher_accel * u_dn, speed)))
                pos0 = spawn_local + unit * s0.unsqueeze(-1)
                vel = unit * v0.unsqueeze(-1)
                self._prev_distance[env_ids] = torch.linalg.norm(target_local - pos0, dim=-1)
        else:
            side = torch.randn(n, 3, device=dev)
            side = side - (side * unit).sum(-1, keepdim=True) * unit
            side = side / torch.linalg.norm(side, dim=-1, keepdim=True).clamp_min(1e-6)
            vel = unit * speed.unsqueeze(-1) + side * (lateral_frac * speed * torch.rand(n, device=dev)).unsqueeze(-1)

            tilt = math.radians(tilt_deg)
            roll = (torch.rand(n, device=dev) * 2 - 1) * tilt
            pitch = (torch.rand(n, device=dev) * 2 - 1) * tilt
            quat = math_utils.quat_from_euler_xyz(roll, pitch, torch.zeros(n, device=dev))

            self._prefix_left[env_ids] = torch.ceil(prefix_m / (speed * self.step_dt).clamp_min(1e-3)).long()

        self._prev_position[env_ids] = pos0   # the true start (recentred / staggered); per-axis rewards read it
        self._robot.write_root_pose_to_sim(torch.cat([pos0 + origins, quat], dim=-1), env_ids)
        self._robot.write_root_velocity_to_sim(torch.cat([vel, torch.zeros(n, 3, device=dev)], dim=-1), env_ids)

    def _refresh_handoff(self):
        """Latch `_handed_off` for envs whose control should now belong to the policy: real distance to the target
        <= this env's sampled h (spawn-leg mode), or the fallback timer ran out. Call after every env.step()."""
        handed = self._handed_off | (self._prefix_left <= 0)
        if self._handoff_snap_enabled:
            pos_w = self._robot.data.root_pos_w
            remaining = torch.linalg.norm(self._desired_pos_w - pos_w, dim=-1)
            # 2nd trigger: flew past the handoff point along the line without entering the h sphere
            s_along = ((pos_w - self._terrain.env_origins - self._prof_spawn_local) * self._prof_unit).sum(-1)
            handed = handed | (remaining <= self._handoff_h) | (s_along >= self._prof_prefix_m)
        self._handed_off = handed

    def _handoff_snap_obs(self, obs, ids):
        """At handoff: snap velocity + attitude to the sampled bucket (random lateral/tilt) and patch `obs` in place."""
        if ids.numel() == 0:
            return None
        dv = self.device
        origins = self._terrain.env_origins[ids]
        pos = self._robot.data.root_pos_w[ids] - origins
        target_local = self._desired_pos_w[ids] - origins
        to_target = target_local - pos
        unit = to_target / torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)
        speed = self._spawn_speed[ids]
        n = ids.numel()
        side = torch.randn(n, 3, device=dv)
        side = side - (side * unit).sum(-1, keepdim=True) * unit
        side = side / torch.linalg.norm(side, dim=-1, keepdim=True).clamp_min(1e-6)
        vel = unit * speed.unsqueeze(-1) + side * (lateral_frac * speed * torch.rand(n, device=dv)).unsqueeze(-1)
        self._robot.write_root_velocity_to_sim(torch.cat([vel, torch.zeros(n, 3, device=dv)], dim=-1), ids)

        tilt = math.radians(tilt_deg)
        roll = (torch.rand(n, device=dv) * 2 - 1) * tilt
        pitch = (torch.rand(n, device=dv) * 2 - 1) * tilt
        quat = math_utils.quat_from_euler_xyz(roll, pitch, torch.zeros(n, device=dv))
        pose = torch.cat([self._robot.data.root_pos_w[ids], quat], dim=-1)
        self._robot.write_root_pose_to_sim(pose, ids)

        # keep the observation consistent with the state we just wrote (layout: see _get_observations)
        obs[ids, 3:6] = vel / VEL_SCALE
        obs[ids, 6:10] = torch.cat([quat[:, 1:], quat[:, 0:1]], dim=-1)      # (w,x,y,z) -> (x,y,z,w)
        obs[ids, 10:13] = 0.0
        yaw_err = (self._desired_yaw_w[ids] + math.pi) % (2 * math.pi) - math.pi   # new yaw is 0
        obs[ids, 21] = torch.sin(yaw_err)
        obs[ids, 22] = torch.cos(yaw_err)
        if getattr(self, "_runner", False):                                  # runner inputs are the last 3: target vel - drone vel
            obs[ids, -3:] = (self._runner_vel_w[ids] - vel) / VEL_SCALE
        return vel

    def _carrot_action(self, pos, vel, ang_vel, roll, pitch, yaw):
        """Raw carrot-leg action (full batch): velocity tracking along the spawn->target trapezoid, z position hold, per-distance attitude gains."""
        spawn, unit = self._prof_spawn_local, self._prof_unit
        prefix_m, vpeak = self._prof_prefix_m, self._prof_vpeak
        d_up, d_cruise = self._prof_dup, self._prof_dcruise
        vh = self._spawn_speed
        s = ((pos - spawn) * unit).sum(-1).clamp_min(0.0).clamp(max=prefix_m)  # clamp() can't mix scalar+Tensor bounds
        v_cmd = torch.where(
            s < d_up, torch.sqrt(torch.clamp(2 * teacher_accel * s, min=0.0)),
            torch.where(s < d_up + d_cruise, vpeak,
                        torch.sqrt(torch.clamp(vpeak ** 2 - 2 * teacher_accel * (s - d_up - d_cruise),
                                                min=torch.clamp(vh, min=0.0) ** 2))))
        v_cmd = v_cmd.clamp_min(0.1)  # zero command at s=0 would leave v=0 forever -- nothing starts the drone moving

        remaining = torch.linalg.norm((self._desired_pos_w - self._terrain.env_origins) - pos, dim=-1)
        nearest = torch.argmin((remaining.unsqueeze(-1) - carrot_key_dists.unsqueeze(0)).abs(), dim=-1)
        gains = carrot_gain_table[nearest]
        for j, name in enumerate(GAIN_NAMES):
            getattr(carrot_pid, name).copy_(gains[:, j])

        accel_xy = teacher_cruise_gain * ((unit * v_cmd.unsqueeze(-1))[:, :2] - vel[:, :2])
        z_err = (self._desired_pos_w[:, 2] - self._terrain.env_origins[:, 2]) - pos[:, 2]
        thrust_delta = carrot_pid.kp_pos * z_err - carrot_pid.kd_pos * vel[:, 2]

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        accel_body_x = accel_xy[:, 0] * cos_yaw + accel_xy[:, 1] * sin_yaw
        accel_body_y = -accel_xy[:, 0] * sin_yaw + accel_xy[:, 1] * cos_yaw
        max_tilt = carrot_pid.max_tilt_rad
        des_roll = torch.clamp(-accel_body_y / 9.81, -max_tilt, max_tilt)
        des_pitch = torch.clamp(accel_body_x / 9.81, -max_tilt, max_tilt)

        roll_torque = carrot_pid.kp_att * (des_roll - roll) - carrot_pid.kd_att * ang_vel[:, 0]
        pitch_torque = carrot_pid.kp_att * (des_pitch - pitch) - carrot_pid.kd_att * ang_vel[:, 1]
        yaw_err = wrap_angle(self._desired_yaw_w - yaw)
        yaw_torque = carrot_pid.kp_yaw * yaw_err - carrot_pid.kd_yaw * ang_vel[:, 2]

        action = torch.stack([thrust_delta, roll_torque, pitch_torque, yaw_torque], dim=-1)
        return torch.clamp(action, carrot_pid.action_low, carrot_pid.action_high)

    u._spawn_speed = torch.zeros(u.num_envs, device=dev)
    u._reset_idx = types.MethodType(_reset_idx, u)
    u._refresh_handoff = types.MethodType(_refresh_handoff, u)
    u._handoff_snap_obs = types.MethodType(_handoff_snap_obs, u)
    if u._handoff_snap_enabled:
        u._carrot_action = types.MethodType(_carrot_action, u)


# ===== Isaac evaluation =====

EVAL_CODES = {"hit": 1, "hover": 2, "oob": 3, "roll": 4, "pitch": 5, "timeout": 6, "caught": 7}


@torch.no_grad()
def isaac_eval(env, model, teacher, steps, use_pid=False, actor=None, recorder=None):
    """Deterministic Isaac eval on the real task; first episode per env only. use_pid=True gives the PID baseline.
    actor (obs -> label in [-1, 1], e.g. the LOS / policy teacher) flies the policy's part of the episode in place of the
    student, through the same handoff, so the teacher is scored on exactly the states the student sees. Reset env afterwards.
    recorder(i, u, handed, terminated_or_truncated, outcome, reward): optional, called after every step (the viewer records tracks with it)."""
    u = env.unwrapped
    N, dev = u.num_envs, u.device
    obs = env.reset()[0]["policy"]
    u.episode_length_buf.zero_()   # a full reset randomises the elapsed-time counter (desync) -> fake early timeouts
    teacher.reset(None)
    u._qv2_stall.zero_()
    speed0 = getattr(u, "_spawn_speed", torch.zeros(N, device=dev)).clone()
    outcome = torch.zeros(N, dtype=torch.long, device=dev)    # 0 = still running
    geom_ep = torch.full((N,), -1, dtype=torch.long, device=dev)   # geometry spawn: the first threat's geometry in each env's first episode (2 = no chaser that episode)
    prefix = getattr(u, "_prefix_left", None)
    # handoff check (first episode of each env): where/how fast the drone REALLY is when control passes to the policy
    import isaaclab.utils.math as math_utils
    ho_rec = torch.zeros(N, dtype=torch.bool, device=dev)
    ho_dist, ho_h, ho_speed, ho_tilt, ho_steps = (torch.zeros(N, device=dev) for _ in range(5))
    # per-episode time budget from the start distance (steps_for_dist)
    from app.control.step_budget import steps_for_dist
    budget = torch.ceil(torch.tensor([steps_for_dist(float(d)) for d in u._start_dist.tolist()], device=dev)
                        * (u.step_dt / (1.0 / 240.0)) ** -1).long()   # physics steps (1/240 s) -> policy steps
    steps = min(steps, int(budget.max().item())) if steps > 0 else int(budget.max().item())
    min_d = torch.full((N,), 1e9, device=dev)         # actor diagnostics: closest approach / speed while the episode is running
    spd_sum, spd_n = torch.zeros((), device=dev), torch.zeros((), device=dev)
    trace = []

    for i in range(steps):
        teach = teacher.label()
        teacher_action = teacher.low + (teach + 1.0) * 0.5 * (teacher.high - teacher.low)
        if use_pid:
            action = teacher_action
        else:
            if actor is None:
                action = model.scale_action(model.forward(obs)[0])
            else:
                action = teacher.low + (actor(obs) + 1.0) * 0.5 * (teacher.high - teacher.low)
            if prefix is not None:
                ctrl0 = ~u._handed_off
                if getattr(u, "_handoff_snap_enabled", False):
                    pos_k, vel_k, ang_vel_k, roll_k, pitch_k, yaw_k, _ = teacher._read_kinematics(env)
                    teacher_action = u._carrot_action(pos_k, vel_k, ang_vel_k, roll_k, pitch_k, yaw_k)
                action = torch.where(ctrl0.unsqueeze(-1), teacher_action, action)
        ctrl = ~u._handed_off if prefix is not None else None
        if prefix is not None:
            prefix.sub_(1).clamp_(min=0)
        nd, _r, _term, truncated, extras = env.step(action)
        obs = nd["policy"]
        if u._K and u._threat_spawn == "geometry":
            geom_ep = torch.where((geom_ep < 0) & (outcome == 0), u.threat_bucket(), geom_ep)
        if prefix is not None:
            u._refresh_handoff()
            if not use_pid and getattr(u, "_handoff_snap_enabled", False):
                # handoff this step: record the real state, then apply the perturbation (skip envs that just terminated)
                js = torch.nonzero(ctrl & u._handed_off & ~(_term | truncated) & ~ho_rec, as_tuple=False).squeeze(-1)
                if js.numel() > 0:
                    pos_w = u._robot.data.root_pos_w[js]
                    ho_dist[js] = torch.linalg.norm(u._desired_pos_w[js] - pos_w, dim=-1)
                    ho_h[js] = u._handoff_h[js]
                    ho_speed[js] = torch.linalg.norm(u._robot.data.root_lin_vel_w[js], dim=-1)
                    r_k, p_k, _y = math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w[js])
                    ho_tilt[js] = torch.rad2deg(torch.maximum(r_k.abs(), p_k.abs()))
                    ho_steps[js] = float(i + 1)
                    ho_rec[js] = True
                js_all = torch.nonzero(ctrl & u._handed_off & ~(_term | truncated), as_tuple=False).squeeze(-1)
                u._handoff_snap_obs(obs, js_all)
        if actor is not None:
            running = (outcome == 0) & ~(_term | truncated) & u._handed_off if prefix is not None else (outcome == 0)
            d_now = torch.linalg.norm(u._desired_pos_w - u._robot.data.root_pos_w, dim=-1)
            min_d = torch.where(running, torch.minimum(min_d, d_now), min_d)
            spd_sum += torch.linalg.norm(u._robot.data.root_lin_vel_w, dim=-1)[running].sum()
            spd_n += running.sum()
            if i % 30 == 0 and bool(running[:3].any()):
                # flight trace of the first 3 envs, every 0.5 s while handed off: dist, speed, closing, lateral, dz, roll/pitch
                rel = (u._desired_pos_w - u._robot.data.root_pos_w)[:3]
                v = u._robot.data.root_lin_vel_w[:3]
                unit = rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
                close = (v * unit).sum(-1)
                lat = torch.linalg.norm(v - close.unsqueeze(-1) * unit, dim=-1)
                r_t, p_t, _y = math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w[:3])
                des = getattr(actor, "last_des", None)    # commanded roll/pitch of the step just flown (LOS teacher only)
                for e in range(3):
                    if bool(running[e]):
                        cmd = (math.degrees(des[0][e].item()), math.degrees(des[1][e].item())) if des is not None else (NAN, NAN)
                        trace.append((e, i, d_now[e].item(), v[e].norm().item(), close[e].item(), lat[e].item(),
                                      -rel[e, 2].item(), math.degrees(r_t[e].item()), math.degrees(p_t[e].item()), *cmd))
        tr = extras["term_reasons"]
        zero = torch.zeros(N, dtype=torch.long, device=dev)
        code = torch.where(tr["hit"], zero + EVAL_CODES["hit"],
               torch.where(tr["hover"], zero + EVAL_CODES["hover"],
               torch.where(tr["oob"], zero + EVAL_CODES["oob"],
               torch.where(tr.get("intercepted", zero.bool()), zero + EVAL_CODES["caught"],
               torch.where(tr["attitude_roll"], zero + EVAL_CODES["roll"],
               torch.where(tr["attitude_pitch"], zero + EVAL_CODES["pitch"],
               torch.where(truncated, zero + EVAL_CODES["timeout"], zero)))))))
        first = (outcome == 0) & (code > 0)
        outcome = torch.where(first, code, outcome)
        out_of_time = (outcome == 0) & (i + 1 >= budget)           # per-distance budget used up, never hit
        outcome = torch.where(out_of_time, torch.full_like(outcome, EVAL_CODES["timeout"]), outcome)
        if recorder is not None:
            recorder(i, u, u._handed_off if prefix is not None else torch.ones_like(outcome, dtype=torch.bool), _term | truncated, outcome, _r)
        if i == steps - 1 or bool((outcome > 0).all()):
            break
        done_ids = torch.nonzero((_term | truncated), as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            teacher.reset(done_ids)

    res = {"hit": (outcome == 1).float().mean().item()}
    for name, c in EVAL_CODES.items():
        if name != "hit":
            res[name] = (outcome == c).float().mean().item()
    res["unfinished"] = (outcome == 0).float().mean().item()
    if u._K and u._threat_spawn == "geometry":        # hit / caught rate per geometry (the first threat's) and without a chaser, first episodes only
        for gi in range(len(u.BUCKET_NAMES)):
            m = geom_ep == gi
            n_g = max(int(m.sum().item()), 1)
            res[f"n_g{gi}"] = float(m.sum().item())
            res[f"hit_g{gi}"] = ((outcome == EVAL_CODES["hit"]) & m).sum().item() / n_g
            res[f"caught_g{gi}"] = ((outcome == EVAL_CODES["caught"]) & m).sum().item() / n_g
            res[f"att_g{gi}"] = (((outcome == EVAL_CODES["roll"]) | (outcome == EVAL_CODES["pitch"])) & m).sum().item() / n_g
            for nm in ("oob", "timeout", "hover"):
                res[f"{nm}_g{gi}"] = ((outcome == EVAL_CODES[nm]) & m).sum().item() / n_g
    if actor is not None:
        seen = min_d < 1e8
        if bool(seen.any()):
            q = np.percentile(min_d[seen].cpu().numpy(), [10, 50, 90])
            res["min_dist_p10"], res["min_dist_p50"], res["min_dist_p90"] = (float(x) for x in q)
        res["mean_speed"] = float((spd_sum / spd_n.clamp_min(1)).item())
        res["trace"] = trace
    if bool(ho_rec.any()):
        m = ho_rec
        derr = (ho_dist - ho_h)[m].cpu().numpy()
        serr = (ho_speed - speed0)[m].cpu().numpy()
        res["ho_n"] = float(m.sum().item())
        res["ho_dist_err_mean"] = float(derr.mean())
        res["ho_dist_err_p10"] = float(np.percentile(derr, 10))
        res["ho_dist_err_p90"] = float(np.percentile(derr, 90))
        res["ho_speed_err_mean"] = float(serr.mean())
        res["ho_speed_err_p10"] = float(np.percentile(serr, 10))
        res["ho_speed_err_p90"] = float(np.percentile(serr, 90))
        res["ho_speed_mean"] = float(ho_speed[m].mean().item())
        res["ho_tilt_mean"] = float(ho_tilt[m].mean().item())
        res["ho_off_frac"] = float((derr > 0.5).mean())       # handed off farther than h (passed the handoff point off-axis, or timer)
        res["ho_leg_s"] = float((ho_steps[m] * float(u.step_dt)).mean().item())
    cfg = getattr(u, "_handoff_cfg", None)
    if cfg is not None:
        frac = speed0 / cfg["max_speed"]
        lo = [b[0] for b in SPEED_BUCKETS]
        hi = [b[1] for b in SPEED_BUCKETS]
        for i, (a, b) in enumerate(zip(lo, hi)):
            m = (frac >= a) & (frac < b if i < len(lo) - 1 else frac <= b)
            res[f"hit_b{i}"] = ((outcome == 1) & m).sum().item() / max(m.sum().item(), 1)
            res[f"crash_b{i}"] = (((outcome >= 3) & (outcome <= 5)) & m).sum().item() / max(m.sum().item(), 1)
    return res


# ===== Rollout + update =====

@torch.no_grad()
def collect_rollout(env, model, teacher, obs, num_steps, gamma, bootstrap_timeouts, ep_ret, anchor=None,
                    expert_beta=0.0, deterministic=False):
    """One chunk of on-policy data. Returns (batch dict, last_obs, rollout stats).
    expert_beta / deterministic are for the teacher-BC stage only (the logp is then NOT the behaviour policy's):
    each step, a fraction expert_beta of the envs fly the anchor's action instead of the student's (DAgger mixing);
    deterministic flies tanh(mean) without exploration noise."""
    u = env.unwrapped
    N, dev = u.num_envs, u.device
    obs_dim = obs.shape[-1]
    a_dim = model.actor_mean.out_features
    b_obs = torch.zeros(num_steps, N, obs_dim, device=dev)
    b_raw = torch.zeros(num_steps, N, a_dim, device=dev)
    b_logp = torch.zeros(num_steps, N, device=dev)
    b_rew = torch.zeros(num_steps, N, device=dev)
    b_val = torch.zeros(num_steps, N, device=dev)
    b_done = torch.zeros(num_steps, N, device=dev)
    b_teach = torch.zeros(num_steps, N, a_dim, device=dev)
    b_pm = torch.ones(num_steps, N, device=dev)   # 1 = the policy chose this action (counts for PPO), 0 = controller
    prefix = getattr(u, "_prefix_left", None)     # handoff stage: steps the controller still flies per env
    ret_sum = torch.zeros((), device=dev)
    ret_n = torch.zeros((), device=dev)

    for t in range(num_steps):
        mean, std, value = model.forward(obs)
        raw = mean if deterministic else mean + std * torch.randn_like(mean)
        logp = log_prob_raw(model, mean, std, raw)
        action = model.scale_action(raw)
        pid_teach = teacher.label()        # always queried: keeps the PID's integrators in step, and (legacy short
                                           # prefix mode) it IS the controller action during the prefix
        teach = anchor(obs) if anchor is not None else pid_teach   # label used by the distillation loss
        b_teach[t] = teach
        if expert_beta > 0.0 and anchor is not None:
            expert = torch.rand(N, device=dev) < expert_beta
            action = torch.where(expert.unsqueeze(-1), teacher.low + (teach + 1.0) * 0.5 * (teacher.high - teacher.low), action)
        if prefix is not None:
            ctrl = ~u._handed_off          # controller flies until the real distance reaches h (see _refresh_handoff)
            teacher_action = teacher.low + (pid_teach + 1.0) * 0.5 * (teacher.high - teacher.low)
            if getattr(u, "_handoff_snap_enabled", False):
                # spawn leg: the env action comes from the carrot controller (full batch, per-env tensors)
                pos_k, vel_k, ang_vel_k, roll_k, pitch_k, yaw_k, _ = teacher._read_kinematics(env)
                teacher_action = u._carrot_action(pos_k, vel_k, ang_vel_k, roll_k, pitch_k, yaw_k)
            action = torch.where(ctrl.unsqueeze(-1), teacher_action, action)
            b_pm[t] = (~ctrl).float()
            prefix.sub_(1).clamp_(min=0)   # a reset inside env.step() below re-arms it for the new episode

        next_dict, reward, terminated, truncated, extras = env.step(action)
        next_obs = next_dict["policy"]
        done = terminated | truncated

        if prefix is not None:
            u._refresh_handoff()
            if getattr(u, "_handoff_snap_enabled", False):
                # handoff this step: apply the perturbation (also patches next_obs), skip envs that just terminated
                js = torch.nonzero(ctrl & u._handed_off & ~done, as_tuple=False).squeeze(-1)
                u._handoff_snap_obs(next_obs, js)

        if bootstrap_timeouts:
            trunc_only = truncated & ~terminated
            if trunc_only.any():
                boot_obs = torch.where(trunc_only.unsqueeze(-1), extras["terminal_observation"], next_obs)
                reward = reward + gamma * model.forward(boot_obs)[2] * trunc_only.float()

        b_obs[t], b_raw[t], b_logp[t], b_rew[t], b_val[t], b_done[t] = obs, raw, logp, reward, value, done.float()

        ep_ret += reward
        ret_sum += (ep_ret * done.float()).sum()
        ret_n += done.sum()
        ep_ret *= (~done).float()

        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            teacher.reset(done_ids)
        obs = next_obs

    last_value = model.forward(obs)[2]
    batch = dict(obs=b_obs, raw=b_raw, logp=b_logp, rew=b_rew, val=b_val, done=b_done, teach=b_teach,
                 pm=b_pm, last_value=last_value)
    stats = {"ep_return_mean": (ret_sum / ret_n.clamp(min=1)).item() if ret_n.item() > 0 else NAN,
             "controller_step_frac": 1.0 - b_pm.mean().item()}
    return batch, obs, stats


def explained_variance(pred, target):
    var = target.var()
    return NAN if var.item() < 1e-12 else (1.0 - (target - pred).var() / var).item()


def ppo_distill_update(model, optimizer, batch, *, gamma, lam, clip_eps, vf_coef, ent_coef, distill_coef,
                       epochs, num_minibatches, target_kl, max_grad_norm, train_actor, train_critic):
    """PPO on the actor (+ teacher distillation term) and plain regression on the critic. When train_actor is
    False the actor receives no gradient at all (critic warmup)."""
    adv, ret = compute_gae(batch["rew"], batch["val"], batch["done"], batch["last_value"], gamma, lam)
    T, N = batch["rew"].shape
    obs = batch["obs"].reshape(T * N, -1)
    raw = batch["raw"].reshape(T * N, -1)
    logp_old = batch["logp"].reshape(-1)
    teach = batch["teach"].reshape(T * N, -1)
    tw = batch["tw"].reshape(-1) if "tw" in batch else torch.ones(T * N, device=batch["obs"].device)
    pm = batch["pm"].reshape(-1) if "pm" in batch else torch.ones(T * N, device=batch["obs"].device)
    val_old = batch["val"].reshape(-1)
    adv = adv.reshape(-1)
    ret = ret.reshape(-1)
    ev = explained_variance(val_old, ret)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = obs.shape[0]
    mb = max(1, n // num_minibatches)
    dev = obs.device
    actor_params, critic_params = model.actor_params(), model.critic_params()
    agg = {k: torch.zeros((), device=dev) for k in
           ("policy_loss", "value_loss", "entropy", "kl", "clip_frac", "distill_loss", "gn_actor", "gn_critic", "n")}
    epochs_run, early = 0, False

    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        ep_kl, ep_nb = torch.zeros((), device=dev), 0
        for s in range(0, n, mb):
            b = perm[s:s + mb]
            mean, std, v = model.forward(obs[b])
            value_loss = (v - ret[b]).pow(2).mean()
            loss = vf_coef * value_loss if train_critic else torch.zeros((), device=dev)
            if train_actor:
                logp = log_prob_raw(model, mean, std, raw[b])
                log_ratio = logp - logp_old[b]
                ratio = log_ratio.exp()
                # controller-flown handoff steps (pm == 0) were not sampled from the policy: no policy gradient
                pg_each = -torch.min(ratio * adv_n[b], torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_n[b])
                pg = (pg_each * pm[b]).sum() / pm[b].sum().clamp(min=1.0)
                ent = gaussian_entropy(std)
                distill = ((torch.tanh(mean) - teach[b]).pow(2).mean(-1) * tw[b]).mean()
                loss = loss + pg - ent_coef * ent + distill_coef * distill
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    agg["policy_loss"] += pg.detach()
                    agg["entropy"] += ent.detach()
                    agg["distill_loss"] += distill.detach()
                    agg["kl"] += kl
                    agg["clip_frac"] += ((ratio - 1).abs() > clip_eps).float().mean()
                    ep_kl += kl
                    ep_nb += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn_a = torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
            gn_c = torch.nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
            optimizer.step()
            agg["value_loss"] += value_loss.detach()
            agg["gn_actor"] += gn_a.detach() if torch.is_tensor(gn_a) else torch.tensor(float(gn_a), device=dev)
            agg["gn_critic"] += gn_c.detach() if torch.is_tensor(gn_c) else torch.tensor(float(gn_c), device=dev)
            agg["n"] += 1
        epochs_run += 1
        if train_actor and ep_nb > 0 and (ep_kl / ep_nb).item() > 1.5 * target_kl:
            early = True
            break

    k = agg["n"].clamp(min=1)
    out = {name: (agg[name] / k).item() for name in
           ("policy_loss", "value_loss", "entropy", "kl", "clip_frac", "distill_loss", "gn_actor", "gn_critic")}
    out = {("approx_kl" if k_ == "kl" else k_): v for k_, v in out.items()}
    out.update(epochs_run=epochs_run, early_stopped=early, explained_var=ev,
               adv_abs_mean=adv.abs().mean().item(), ret_mean=ret.mean().item(), val_mean=val_old.mean().item())
    if not train_actor:
        for key in ("policy_loss", "entropy", "approx_kl", "clip_frac", "distill_loss"):
            out[key] = NAN
    return out


def teacher_weight(obs, fade_speed, vel_scale, vel_slice=slice(3, 6)):
    """Per-sample weight in [0, 1] for the teacher loss: 1 at rest, 0 at >= fade_speed m/s. Speed is read back
    from the observation (obs[3:6] = velocity / VEL_SCALE, see BaseDroneEnvIsaac._get_observations)."""
    if fade_speed <= 0:
        return torch.ones(obs.shape[:-1], device=obs.device)
    speed = torch.linalg.norm(obs[..., vel_slice], dim=-1) * vel_scale
    return torch.clamp(1.0 - speed / fade_speed, 0.0, 1.0)


@torch.no_grad()
def teacher_agreement(model, batch, chunk_rows=262144):
    """Deterministic student vs teacher on the rollout states (before this chunk's update)."""
    obs = batch["obs"].reshape(-1, batch["obs"].shape[-1])
    teach = batch["teach"].reshape(-1, batch["teach"].shape[-1])
    diffs, sat = [], []
    for s in range(0, obs.shape[0], chunk_rows):
        mean = model.forward(obs[s:s + chunk_rows])[0]
        a = torch.tanh(mean)
        diffs.append((a - teach[s:s + chunk_rows]).abs().sum(0))
        sat.append((a.abs() > 0.98).float().sum(0))
    n = obs.shape[0]
    mae = (torch.stack(diffs).sum(0) / n).tolist()
    sat_f = (torch.stack(sat).sum(0) / n).tolist()
    return {"agree_mae": float(np.mean(mae)),
            **{f"agree_mae_d{i}": v for i, v in enumerate(mae)},
            "sat_frac": float(np.mean(sat_f)),
            **{f"sat_frac_d{i}": v for i, v in enumerate(sat_f)},
            "teacher_abs_mean": batch["teach"].abs().mean().item()}


# ===== Schedules / phases =====

def phase_of(chunk: int, settle: int, warmup: int, head: int) -> str:
    if chunk < settle:
        return "settle"
    if chunk < settle + warmup:
        return "critic_warmup"
    if chunk < settle + warmup + head:
        return "head_only"
    return "full"


def apply_phase(model: RTLActorCritic, phase: str):
    trunk = list(model.shared.parameters())
    head = list(model.actor_mean.parameters()) + [model.actor_log_std]
    for p in trunk:
        p.requires_grad_(phase == "full")
    for p in head:
        p.requires_grad_(phase in ("head_only", "full"))
    for p in model.critic_params():
        p.requires_grad_(phase != "settle")
    for p in model.critic_head.parameters():
        p.requires_grad_(False)  # inherited, unused


def cosine_factor(progress: float, min_ratio: float) -> float:
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def distill_scheduled(idx: int, start: float, end: float, decay_chunks: int) -> float:
    return start + (end - start) * min(1.0, idx / max(1, decay_chunks))


# ===== Logging helpers =====

def _py(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return float(v)


class RunLog:
    """Keeps every row in memory and rewrites the CSV with the union of all keys, so a metric that only
    exists in some phases never breaks the file."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.rows = []

    def add(self, row: dict):
        self.rows.append({k: _py(v) for k, v in row.items()})
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        tmp = self.csv_path + ".tmp"
        try:
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, restval="")
                w.writeheader()
                w.writerows(self.rows)
            # Windows: os.replace fails while another process (Excel, an IDE CSV viewer, the web viewer, antivirus) holds the
            # CSV open. Retry briefly; if it is still locked, skip this write -- every row stays in memory, so the next
            # chunk's write includes it. A log write must never kill a training run.
            for attempt in range(10):
                try:
                    os.replace(tmp, self.csv_path)
                    return
                except PermissionError:
                    time.sleep(0.2 * (attempt + 1))
            print(f"[rtl] WARNING: {self.csv_path} is locked by another program; metrics not written this chunk "
                  f"(kept in memory, retried next chunk)")
        except OSError as e:
            print(f"[rtl] WARNING: metrics write failed ({e}); continuing")


def plot_summary(rows: list, out_path: str, pid_success=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(name):
        return np.array([float(r.get(name, NAN) if r.get(name, "") != "" else NAN) for r in rows])

    t = col("timesteps") / 1e6
    phase_change = [t[i] for i in range(1, len(rows)) if rows[i]["phase"] != rows[i - 1]["phase"]]
    fig, axs = plt.subplots(3, 2, figsize=(14, 11))

    def mark(ax):
        for x in phase_change:
            ax.axvline(x, color="k", ls=":", alpha=0.4)
        ax.set_xlabel("M env-steps")

    ax = axs[0, 0]
    ax.plot(t, col("train_hit_rate"), label="train hit rate (noisy policy, all envs)")
    ax.plot(t, col("success_rate"), label="eval success (deterministic, Isaac)")
    if pid_success is not None:
        ax.axhline(pid_success, color="g", ls="--", label="PID eval success")
    ax.set_ylim(-0.02, 1.02); ax.set_title("hit rate"); ax.legend(fontsize=8); mark(ax)

    ax = axs[0, 1]
    for k in ("train_timeout_rate", "train_hover_rate", "train_oob_rate", "train_roll_rate", "train_pitch_rate"):
        ax.plot(t, col(k), label=k.replace("train_", ""))
    ax.set_title("how training episodes end (besides hit)"); ax.legend(fontsize=8); mark(ax)

    ax = axs[1, 0]
    ax.plot(t, col("distill_coef_eff"), label="distill coef (effective)")
    ax.plot(t, col("agree_mae"), label="|student - teacher| (mean, normalised)")
    ax.set_title("teacher anchor / agreement"); ax.legend(fontsize=8); mark(ax)

    ax = axs[1, 1]
    ax.plot(t, col("train_stall_frac"), label="stall fraction (0.25-0.75 m, v<0.15)")
    ax.plot(t, col("train_near_frac"), label="fraction of steps within 1 m")
    ax.set_title("lingering near the target"); ax.legend(fontsize=8); mark(ax)

    ax = axs[2, 0]
    for k in ("rew_progress", "rew_step", "rew_tumble", "rew_tilt", "rew_hit", "rew_crash", "rew_timeout", "rew_hover"):
        ax.plot(t, col(k), label=k.replace("rew_", ""))
    ax.set_title("mean reward per step by component"); ax.legend(fontsize=7, ncol=2); mark(ax)

    ax = axs[2, 1]
    ax.semilogy(t, np.maximum(col("value_loss"), 1e-8), label="value_loss")
    ax.semilogy(t, np.maximum(col("gn_actor"), 1e-8), label="grad norm actor")
    ax.semilogy(t, np.maximum(col("approx_kl"), 1e-8), label="approx_kl")
    ax.set_title("optimisation"); ax.legend(fontsize=8); mark(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ===== Main loop =====

def build_parser():
    p = argparse.ArgumentParser(description="RTL handoff training in Isaac (PPO + PID teacher)")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--total_timesteps", type=int, default=128_000_000)
    p.add_argument("--distance-low", type=float, default=3.0)
    p.add_argument("--distance-high", type=float, default=10.0)
    p.add_argument("--checkpoint-dir", default="runs/rtl_isaac")
    p.add_argument("--bc-checkpoint", dest="bc_checkpoint_path", default=None,
                   help=f"student init weights (default {DEFAULT_BC_CHECKPOINT}); passing it explicitly also makes "
                        "--teacher auto pick the policy teacher")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-steps-per-chunk", type=int, default=256)
    p.add_argument("--spawn-speed-low", type=float, default=8.0,
                   help="Every episode (training AND eval) starts already moving at Uniform(low, high) m/s straight "
                        "toward its own target. The repo defines no 'full speed', 8-12 m/s is an assumption "
                        "(VEL_SCALE, the obs normaliser, is 10 m/s).")
    p.add_argument("--spawn-speed-high", type=float, default=12.0)
    p.add_argument("--from-rest", action="store_true", help="ignore --spawn-speed-*: spawn at rest like the old runs")
    p.add_argument("--handoff", action="store_true",
                   help="Handoff-stage training (reward_qv2 + varied-speed handoff spawn + controller prefix). "
                        "Overrides --spawn-speed-*: max speed = --max-speed, speed buckets 10-25 %% (20 %% of "
                        "episodes) / 25-75 %% (40 %%) / 75-100 %% (40 %%). Use with --distance-low 3 --distance-high 50.")
    p.add_argument("--max-speed", type=float, default=12.0, help="--handoff: 100 %% of the speed distribution, m/s")
    p.add_argument("--prefix-low", type=float, default=2.0, help="--handoff: controller flies at least this many m first")
    p.add_argument("--prefix-high", type=float, default=5.0, help="--handoff: ... and at most this many m")
    p.add_argument("--spawn-dist-low", type=float, default=None,
                   help="--handoff: spawn the drone this far from the target (overrides --prefix-low/--prefix-high); "
                        "the PID actually flies it from the spawn point down to the handoff distance "
                        "(~U(--distance-low, --distance-high)), and at the moment control passes to the policy "
                        "its velocity is snapped to the sampled speed bucket (b1 10-25%%, b2 25-75%%, b3 75-100%% "
                        "of --max-speed) regardless of what the PID's own flight converged to. Requires "
                        "--spawn-dist-high too; omit both to keep the short legacy prefix leg.")
    p.add_argument("--spawn-dist-high", type=float, default=None, help="--spawn-dist-low: ... and at most this far")
    p.add_argument("--teacher-accel", type=float, default=None,
                   help="--spawn-dist-low: ramp steepness (m/s^2) of the scripted 0 -> max-speed -> handoff-speed "
                        "carrot-chase profile flown on the long spawn leg (see install_handoff_stage). Default: derived "
                        "from the tilt limit, 0.5 * g * tan(max_tilt) (~1.5 m/s^2) -- the drone must be able to fly it")
    p.add_argument("--leg-max-elev-deg", type=float, default=15.0,
                   help="--spawn-dist-low: the spawn->target line is clamped to at most this elevation angle (and never "
                        "so far downward that the target would end underground). The carrot controller's altitude "
                        "channel is a plain hold, so steep long climbs do not work.")
    p.add_argument("--no-recenter", dest="recenter", action="store_false",
                   help="--spawn-dist-low: by default the spawn is shifted back along the leg so the handoff POINT is at "
                        "x=y=0 (the observation has absolute position and the policy trained near the origin)")
    p.add_argument("--teacher-cruise-gain", type=float, default=1.0,
                   help="--spawn-dist-low: horizontal velocity-tracking gain (m/s^2 of accel per m/s of error) "
                        "for the scripted ramp/cruise on the spawn leg")
    p.add_argument("--lateral-frac", type=float, default=0.3,
                   help="--handoff: sideways velocity up to this fraction of the speed")
    p.add_argument("--handoff-tilt-deg", type=float, default=15.0, help="--handoff: random roll/pitch at handoff")
    p.add_argument("--distill-fade-speed", type=float, default=6.0,
                   help="Teacher loss weight = clamp(1 - speed/this, 0, 1) per sample: a hover PID commands BRAKING "
                        "at high speed, which fights 'hit the target'. 0 disables the gating (weight 1).")
    p.add_argument("--imitation-fraction", type=float, default=0.0,
                   help="Fraction of --total_timesteps spent on the on-policy DAgger imitation stage first. "
                        "0 = trust --bc-checkpoint as is (auto 0.08 if that file does not exist).")
    # phases
    p.add_argument("--settle-chunks", type=int, default=2, help="rollout-only chunks (baseline measurement)")
    p.add_argument("--warmup-chunks", type=int, default=16, help="critic-only chunks, actor frozen")
    p.add_argument("--head-chunks", type=int, default=8, help="actor_mean+log_std trainable, trunk frozen")
    p.add_argument("--lr-ramp-chunks", type=int, default=3)
    # optimisation
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--warmup-epochs", type=int, default=8)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr-actor", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW decay, PPO and teacher BC")
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--critic-layers", type=int, default=2)
    p.add_argument("--init-std", type=float, default=0.06, help="initial exploration std in tanh-normalised action space")
    p.add_argument("--log-std-min", type=float, default=-3.5)
    p.add_argument("--log-std-max", type=float, default=-2.0)
    p.add_argument("--bootstrap-timeouts", action="store_true")
    # distillation teacher
    p.add_argument("--teacher", choices=("auto", "pid", "policy", "los"), default="auto",
                   help="who labels the distillation loss. pid: the per-distance PID (speed-faded). policy: a FROZEN actor "
                        "(--anchor-checkpoint, else --bc-checkpoint). los: straight line of sight to the target at a "
                        "braking-limited speed. auto: policy if --anchor-checkpoint or --bc-checkpoint was passed, else los. "
                        "The PID still flies the handoff leg and the PID eval baseline in every mode.")
    p.add_argument("--scratch", action="store_true",
                   help="random student: ignore every --bc-checkpoint (and the pretrained PID-imitation default) and skip the forced "
                        "PID imitation stage (only --imitation-fraction applies). Pair it with a teacher that is not the PID.")
    p.add_argument("--teacher-bc-chunks", type=int, default=0,
                   help="before PPO: this many chunks of supervised DAgger onto the los/policy teacher (student flies, teacher "
                        "labels, actor regresses with no KL limit). Use 10-20 with --scratch")
    p.add_argument("--teacher-bc-epochs", type=int, default=4,
                   help="passes over the aggregated teacher-BC buffer per chunk (batch 8192, at least 200 steps per chunk)")
    p.add_argument("--teacher-bc-lr", type=float, default=1e-3)
    p.add_argument("--teacher-check-only", action="store_true",
                   help="fly the non-PID teacher (los / policy) once through the handoff, print its outcomes and closest approach, and exit")
    p.add_argument("--los-stop-speed", type=float, default=0.5,
                   help="--teacher los: the speed it brakes down to at the target. 0.5 = arrive and stop; >= --max-speed = "
                        "never brake, just fly the line at full speed straight through the target")
    p.add_argument("--los-max-tilt-deg", type=float, default=35.0,
                   help="--teacher los: tilt cap of the LOS controller (the PID's is 0.3 rad = 17 deg, which can't stop from "
                        "20 m/s inside 50 m; the reward ends an episode at 60 deg)")
    p.add_argument("--anchor-checkpoint", default=None,
                   help="actor weights (e.g. a previous best.pt) used as the FROZEN teacher for --teacher policy "
                        "(default: the --bc-checkpoint weights). The student is pulled toward that policy's action "
                        "(tanh(mean)) on the states it visits, at the --distill-* weight, with no speed fade.")
    p.add_argument("--distill-start", type=float, default=1.0)
    p.add_argument("--distill-end", type=float, default=0.05)
    p.add_argument("--distill-decay-chunks", type=int, default=60)
    p.add_argument("--distill-max", type=float, default=5.0)
    p.add_argument("--guard-margin", type=float, default=0.05)
    p.add_argument("--guard-gain", type=float, default=2.0)
    p.add_argument("--guard-boost-max", type=float, default=20.0)
    p.add_argument("--obs-position-mode", choices=("full", "height"), default="full",
                   help="observation obs[0:3]: 'full' = symlog(absolute x,y,z) (original; the policy depends on where the handoff "
                        "is, so deployment must re-centre it); 'height' = (0, 0, symlog(clip(z, 0, 30 m))) -- no absolute horizontal "
                        "position at all, nothing to re-centre at deployment. Same 23-dim layout, so a 'full' checkpoint still loads as "
                        "the start. Recorded in run_config.json; replay tools read it from there.")
    p.add_argument("--guard-min-episodes", type=int, default=None,
                   help="a chunk only feeds the guard / baseline / abort check if at least this many episodes ended in it "
                        "(default: 4%% of --num_envs, at least 50)")
    p.add_argument("--abort-hit-rate", type=float, default=0.25)
    p.add_argument("--abort-window", type=int, default=6)
    # eval / output
    p.add_argument("--eval-steps", type=int, default=0,
                   help="Isaac eval: hard cap on policy steps per eval (0 = none). Each env's FIRST episode after a "
                        "full reset is scored and gets its own per-distance time budget (steps_for_dist: ~1 s/m, "
                        "7.5 s floor); exceeding it counts as a timeout")
    p.add_argument("--eval-every", type=int, default=5, help="Isaac eval every N chunks (it resets all envs)")
    p.add_argument("--eval-burnin", type=int, default=600,
                   help="policy steps run (and discarded) after each eval so the reset envs desynchronise; "
                        "without it the following chunk's train stats are biased (spikes). 0 = off")
    p.add_argument("--pid-eval-every", type=int, default=10, help="also run the PID baseline every N chunks")
    p.add_argument("--no-onnx", dest="onnx", action="store_false")
    p.add_argument("--reward", choices=("qv2", "rc"), default="qv2",
                   help="qv2 = current reward; rc = remote-controlled drone reward (log-potential closing, +30 hit, -15 "
                        "caught/crash, hover + timeout penalties; angle_reward_rc.py). Overrides via --reward-json.")
    p.add_argument("--num-threats", type=int, default=0,
                   help="Threat slots appended to the observation (7 each, after the 23 base inputs). Allocate the MAXIMUM "
                        "you will ever use (e.g. 5) and raise --active-threats per stage, so checkpoints keep loading. "
                        "Needs --reward rc to matter.")
    p.add_argument("--no-timer-obs", action="store_true",
                   help="--reward rc appends 2 inputs after the threat slots (in_zone, zone_count / steps needed for the hit); "
                        "this keeps the old observation size. Older checkpoints load either way (new inputs start at zero weight).")
    p.add_argument("--active-threats", type=int, default=None, help="threats switched on (<= --num-threats); default = all slots")
    p.add_argument("--threat-speed", "--threat-speed-max", type=float, default=10.0,
                   help="threat chase speed, m/s: the commanded speed of the limit-LOS controller, and its ONLY limit (the drone "
                        "tilt clamp is --los-max-tilt-deg); the upper end when --threat-speed-min is set")
    p.add_argument("--threat-speed-min", type=float, default=None,
                   help="lower end of the chase speed: each threat draws U(min, max) per episode. Default: same as the max (fixed)")
    p.add_argument("--threat-radius", type=float, default=None,
                   help="collision-sphere radius of a threat drone, m (e.g. 1.0). Default: the drone's own body (0.4 m cuboid)")
    p.add_argument("--threat-spread", "--threat-spawn-max", type=float, default=30.0,
                   help="uniform spawn: outer radius of the ball around the base, m")
    p.add_argument("--threat-spawn-min", type=float, default=15.0,
                   help="uniform spawn: inner radius, m. Threats start between --threat-spawn-min and --threat-spawn-max from the base "
                        "(0 = anywhere in the ball)")
    p.add_argument("--threat-spawn", choices=("uniform", "base", "sphere", "geometry"), default="uniform",
                   help="uniform (alias sphere): each threat starts at a random point of the ball around the base, new every episode; "
                        "base: all start at the base (3 m height), slot 0 on it, the rest on a ring with --threat-buffer clearance; "
                        "geometry: each threat appears AT THE HANDOFF relative to the drone -- "
                        "head-on (ahead, coming at it) or tail (behind, chasing) -- already moving and released at once "
                        "(--threat-release-after is ignored). See --threat-geometry-weights")
    p.add_argument("--threat-geometry-weights", type=float, nargs=2, metavar=("HEAD_ON", "TAIL"), default=(1.0, 1.0),
                   help="geometry spawn: relative weights of the two geometries, drawn per threat and episode (1 1 = half each)")
    p.add_argument("--threat-reaction-s", type=float, default=1.5,
                   help="geometry spawn, head-on: spawn at least this many seconds of closing time ahead, i.e. max(20 m, T * (drone speed + "
                        "threat speed)), so a dodge is physically possible")
    p.add_argument("--threat-tail-gap", type=float, nargs=2, metavar=("LO", "HI"), default=(15.0, 40.0),
                   help="geometry spawn, tail: distance behind the drone, m")
    p.add_argument("--threat-jitter", type=float, default=2.0,
                   help="geometry spawn, head-on: random offset from the drone's line of travel, m (sideways and vertical)")
    p.add_argument("--threat-prob", type=float, default=1.0,
                   help="chance that an episode has its chasers at all (drawn per env at reset). 0.5 = half the episodes fly with no chaser "
                        "(every slot dead, observation zeros there) and half with --active-threats of them. Train and eval both use it; "
                        "the logs split hit / caught by head-on / tail / none")
    p.add_argument("--threat-buffer", type=float, default=0.5,
                   help="base spawn: clearance between neighbouring threat bodies on top of their diameter, m")
    p.add_argument("--threat-release-after", type=float, nargs=2, metavar=("LO", "HI"), default=(0.33, 0.99),
                   help="each threat waits (hovering) until the policy-flown drone has covered a fraction U(LO, HI) of the distance it had "
                        "left at the handoff; 0.33 0.99 = released between 33%% and 99%% of the way. LO=HI gives a fixed value")
    p.add_argument("--threat-release-frac", type=float, default=None,
                   help="old fixed form, as the fraction of the handoff distance still LEFT (0.7 = after covering 30%%); "
                        "if given it replaces --threat-release-after")
    p.add_argument("--obs-layout", choices=("v1", "v2"), default="v1",
                   help="v1 (default): 23 base + 7 per threat + timer. v2: 23 base + timer + 4 per opponent (position + status) + 5 per ally + "
                        "3 flags (link, jammed, autonomous); see environmental/docs.md. A 25-input checkpoint (23 + timer) loads into v2 as is; "
                        "a v1 checkpoint WITH threat slots does not")
    p.add_argument("--num-allies", type=int, default=0, help="v2: ally slots (x, y, z, status, id); 0 = 1v1, no inputs")
    p.add_argument("--reward-json", default=None, help="path to a JSON file or an inline JSON object with "
                   "RewardQV1Config overrides, e.g. '{\"r_hit\": 20}'")
    return p


def resolve_teacher(args) -> str:
    if args.teacher != "auto":
        return args.teacher
    return "policy" if (args.anchor_checkpoint or (args.bc_explicit and not args.scratch)) else "los"


def load_anchor(args, env, device, teacher):
    """Distillation teacher other than the PID, as a callable obs -> label in [-1, 1] (the space of the student's
    tanh(mean)), or None for --teacher pid. policy: a FROZEN copy of a previous actor; los: see make_los_label."""
    mode = resolve_teacher(args)
    if mode == "pid":
        print("[rtl] TEACHER: pid")
        return None
    if mode == "los":
        print(f"[rtl] TEACHER: los (straight line of sight, braking-limited, max {args.max_speed} m/s); "
              f"the PID is no longer the distillation teacher (speed fade off)")
        return make_los_label(env, teacher, args.max_speed, args.teacher_accel, args.teacher_cruise_gain,
                              stop_speed=args.los_stop_speed, max_tilt=math.radians(args.los_max_tilt_deg))
    ckpt = args.anchor_checkpoint or args.bc_checkpoint_path
    assert os.path.exists(ckpt), f"--teacher policy needs weights, {ckpt} not found"
    from app.guidance.export_onnx import infer_architecture
    u = env.unwrapped
    obs_dim = u.single_observation_space["policy"].shape[0]
    a_dim = u.single_action_space.shape[0]
    low, high = u.single_action_space.low, u.single_action_space.high
    raw = torch.load(ckpt, map_location="cpu")
    ck_obs, hidden, layers, ck_a = infer_architecture(raw)
    assert ck_obs <= obs_dim and ck_a == a_dim, f"anchor dims {(ck_obs, ck_a)} != env {(obs_dim, a_dim)}"
    net = ActorCritic(obs_dim, a_dim, low, high, hidden=hidden, num_hidden_layers=layers, dropout=0.0).to(device)
    load_bc_checkpoint(net, ckpt, map_location=device)
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    print(f"[rtl] TEACHER: policy, a frozen copy of {ckpt} (hidden={hidden}, layers={layers}); "
          f"the PID is no longer the distillation teacher (speed fade off)")

    @torch.no_grad()
    def anchor_label(obs):
        return torch.tanh(net.forward(obs)[0])
    return anchor_label


def load_student(args, env, device):
    u = env.unwrapped
    obs_dim = u.single_observation_space["policy"].shape[0]
    a_dim = u.single_action_space.shape[0]
    low, high = u.single_action_space.low, u.single_action_space.high
    hidden, layers = 64, 4
    have_ckpt = not args.scratch and os.path.exists(args.bc_checkpoint_path)
    if have_ckpt:
        from app.guidance.export_onnx import infer_architecture
        raw = torch.load(args.bc_checkpoint_path, map_location="cpu")
        ck_obs, hidden, layers, ck_a = infer_architecture(raw)
        assert ck_obs <= obs_dim and ck_a == a_dim, f"checkpoint dims {(ck_obs, ck_a)} != env {(obs_dim, a_dim)}"
    model = RTLActorCritic(obs_dim, a_dim, low, high, hidden=hidden, num_hidden_layers=layers,
                           critic_hidden=args.critic_hidden, critic_layers=args.critic_layers,
                           log_std_min=args.log_std_min, log_std_max=args.log_std_max).to(device)
    if have_ckpt:
        tmp = ActorCritic(obs_dim, a_dim, low, high, hidden=hidden, num_hidden_layers=layers, dropout=0.0)
        load_bc_checkpoint(tmp, args.bc_checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(tmp.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("value_net.") for k in missing), (missing, unexpected)
        print(f"[rtl] student initialised from {args.bc_checkpoint_path} (hidden={hidden}, layers={layers})")
    else:
        print(f"[rtl] no checkpoint at {args.bc_checkpoint_path}: student starts random; imitation stage required")
    return model, have_ckpt, (obs_dim, a_dim, low, high, hidden, layers)


def configure_env(u, args):
    """Everything between "env created" and "model built": reward, threats, observation mode, target range, handoff stage. Shared by
    run() and the web viewer's Isaac worker (guidance/isaac_worker.py), so a flight in the viewer is set up exactly like training.
    Returns (reward config, PID gains by distance, oob radius)."""
    reward_overrides = None
    if args.reward_json:
        reward_overrides = json.load(open(args.reward_json)) if os.path.exists(args.reward_json) \
            else json.loads(args.reward_json)
    if args.reward == "rc":
        from app.reward_functions.angle_reward_rc import RewardRCConfig, install_reward_rc
        rcfg = install_reward_rc(u, RewardRCConfig.from_overrides(reward_overrides))
        _k = args.num_threats if args.active_threats is None else args.active_threats
        if args.num_threats:
            rel_after = tuple(args.threat_release_after)
            fixed = args.threat_release_frac
            u.set_threats(_k, max_speed=args.threat_speed, spread=args.threat_spread, spawn=args.threat_spawn,
                          spawn_min=args.threat_spawn_min, min_speed=args.threat_speed_min, buffer=args.threat_buffer,
                          release_frac=1.0 if fixed is None else fixed,
                          release_frac_range=None if fixed is not None else (1.0 - rel_after[1], 1.0 - rel_after[0]),
                          geometry_weights=tuple(args.threat_geometry_weights), reaction_s=args.threat_reaction_s,
                          tail_gap=tuple(args.threat_tail_gap), jitter=args.threat_jitter,
                          threat_prob=args.threat_prob)
            if args.threat_spawn == "geometry":
                print(f"[rtl] THREAT GEOMETRY: spawned at the handoff relative to the drone, released at once; weights (head-on, tail) = "
                      f"{tuple(args.threat_geometry_weights)}, head-on >= {args.threat_reaction_s} s of closing time ahead, "
                      f"tail {tuple(args.threat_tail_gap)} m behind")
            rel_txt = (f"at {fixed:.0%} of the handoff distance left" if fixed is not None
                       else f"after {rel_after[0]:.0%}-{rel_after[1]:.0%} of the way from the handoff (uniform per threat)")
            print(f"[rtl] THREATS: {_k} active of {args.num_threats} drone slots, limit-LOS chase at "
                  f"{args.threat_speed if args.threat_speed_min is None else args.threat_speed_min}-{args.threat_speed} m/s, "
                  f"body {'drone' if args.threat_radius is None else f'sphere r={args.threat_radius} m'}, spawn {args.threat_spawn} "
                  f"({'r=%g-%g m' % (args.threat_spawn_min, args.threat_spread) if args.threat_spawn != 'base' else 'buffer %g m' % args.threat_buffer}), "
                  f"released {rel_txt}; observation = 23 + 7*{args.num_threats}")
    else:
        from app.reward_functions.angle_reward_qv2 import RewardQV2Config, install_reward_qv2
        rcfg = install_reward_qv2(u, RewardQV2Config.from_overrides(reward_overrides))

    oob_radius = max(20.0, args.distance_high * 3.0)
    # no finite target pool: every reset draws a fresh random target
    u.set_obs_position_mode(args.obs_position_mode)
    print(f"[rtl] observation position mode: {args.obs_position_mode}")
    u.set_target_pairs(None)
    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)
    if args.reward == "rc" and args.num_threats:
        u.set_threat_gains(gains_by_dist, math.radians(args.los_max_tilt_deg))     # threat drones fly the LOS controller (control/los.py)
    prefix_hi = args.prefix_high if args.handoff else 0.0
    # in --handoff mode the env distance is the TOTAL one; the controller prefix eats up to prefix_hi of it
    u.set_target_distance_range(args.distance_low + prefix_hi, args.distance_high + prefix_hi)
    if args.handoff:
        # The env's own constant-speed spawn is replaced by the handoff wrapper.
        spawn_range = (SPEED_BUCKETS[0][0] * args.max_speed, args.max_speed)
        u.set_spawn_speed_range(None, None)
        install_handoff_stage(u, max_speed=args.max_speed, prefix_low=args.prefix_low, prefix_high=args.prefix_high,
                              lateral_frac=args.lateral_frac, tilt_deg=args.handoff_tilt_deg,
                              dist_low=args.distance_low, dist_high=args.distance_high,
                              spawn_low=args.spawn_dist_low, spawn_high=args.spawn_dist_high,
                              gains_by_dist=gains_by_dist, teacher_accel=args.teacher_accel,
                              teacher_cruise_gain=args.teacher_cruise_gain, leg_max_elev_deg=args.leg_max_elev_deg, recenter=args.recenter)
        if args.spawn_dist_low is not None:
            print(f"[rtl] HANDOFF stage: speed buckets {SPEED_BUCKETS} x {args.max_speed} m/s, probs {SPEED_BUCKET_PROBS}; "
                  f"spawned at rest {args.spawn_dist_low}-{args.spawn_dist_high} m from target, carrot-chase PID ramps "
                  f"0 -> min(max_speed, leg-limited peak) -> down to the sampled bucket speed (accel {u._handoff_cfg['teacher_accel']:.2f} "
                  f"m/s^2, velocity-tracking gain {args.teacher_cruise_gain}) flying down to the handoff point ~ "
                  f"U({args.distance_low}, {args.distance_high}) m (the MODEL's range, not the spawn range): control passes "
                  f"when the REAL distance reaches it, then velocity+tilt get the random handoff perturbation")
        else:
            print(f"[rtl] HANDOFF stage: speed buckets {SPEED_BUCKETS} x {args.max_speed} m/s, probs {SPEED_BUCKET_PROBS}; "
                  f"controller flies {args.prefix_low}-{args.prefix_high} m first; lateral<= {args.lateral_frac:.0%} of speed, "
                  f"tilt<= {args.handoff_tilt_deg} deg; remaining dist at handoff ~ U({args.distance_low}, {args.distance_high})")
    else:
        spawn_range = None if args.from_rest else (args.spawn_speed_low, args.spawn_speed_high)
        u.set_spawn_speed_range(*(spawn_range or (None, None)))
        print(f"[rtl] spawn: {'from rest' if spawn_range is None else f'already moving at U{spawn_range} m/s toward the target'}")
    return rcfg, gains_by_dist, oob_radius


def run(env, args):
    import mlflow
    from app.guidance.export_onnx import export_onnx_model
    from app.guidance.mlflow_utils import log_metrics_safe, log_params_safe, start_run
    from app.guidance.plotting import plot_training_run
    from app.reward_functions.angle_reward_qv2 import BUCKET_LABELS, RewardQV2Config, install_reward_qv2, pop_reward_stats
    from app.training.base_training_isaac import METRICS_JSON_PATH, _append_json_row, _run_imitation_stage_isaac

    run_start = datetime.now()
    u = env.unwrapped
    device = u.device
    N = u.num_envs
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    rcfg, gains_by_dist, oob_radius = configure_env(u, args)

    model, have_ckpt, arch = load_student(args, env, device)
    obs_dim, a_dim, low, high, hidden, layers = arch
    chunk_steps = args.num_steps_per_chunk * N

    imit_frac = args.imitation_fraction if (have_ckpt or args.scratch) else max(args.imitation_fraction, 0.08)
    if imit_frac > 0:
        imit_steps_per_env = max(1, int(imit_frac * args.total_timesteps) // N)
        print(f"[rtl] on-policy imitation stage: {imit_steps_per_env} steps/env")
        model = _run_imitation_stage_isaac(env, model, imit_steps_per_env, gains_by_dist)
    set_initial_std(model, args.init_std)

    n_chunks = max(1, (args.total_timesteps - int(imit_frac * args.total_timesteps)) // chunk_steps)
    first_trainable = args.settle_chunks + args.warmup_chunks
    if n_chunks <= first_trainable:
        print(f"[rtl] WARNING: n_chunks={n_chunks} <= settle+warmup={first_trainable}; the actor never trains")

    groups = [
        {"name": "trunk", "params": list(model.shared.parameters()), "lr": args.lr_actor},
        {"name": "head", "params": list(model.actor_mean.parameters()) + [model.actor_log_std], "lr": args.lr_actor},
        {"name": "critic", "params": model.critic_params(), "lr": args.lr_critic},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lr = {g["name"]: g["lr"] for g in groups}

    teacher = PidTeacher(env, gains_by_dist)
    anchor = load_anchor(args, env, device, teacher)
    from app.environmental.base_drone_env import VEL_SCALE   # only the obs-normalisation constant

    plain =ActorCritic(obs_dim, a_dim, low, high, hidden=hidden, num_hidden_layers=layers, dropout=0.0).to(device)

    run_name = f"rtl_isaac_{args.distance_low:g}_{args.distance_high:g}"
    run_key = f"{run_name}_{run_start.strftime('%Y%m%d_%H%M%S')}"
    start_run(experiment_name="rtl-isaac", run_name=run_name)
    cfg_dump = {**vars(args), "reward_qv2": rcfg.to_dict(), "arch": {"hidden": hidden, "layers": layers},
                "chunk_steps": chunk_steps, "n_chunks": n_chunks, "run_start": run_start.isoformat(timespec="seconds")}
    with open(os.path.join(args.checkpoint_dir, "run_config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=2, default=str)
    log_params_safe({**vars(args), **{f"rew_{k}": v for k, v in rcfg.to_dict().items()}, "n_chunks": n_chunks})

    log = RunLog(os.path.join(args.checkpoint_dir, "metrics.csv"))
    if anchor is not None:
        # fly the distillation teacher itself through the handoff: if IT crashes, no student can learn to fly from its labels
        tv = isaac_eval(env, model, teacher, args.eval_steps, actor=anchor)
        print(f"[rtl] TEACHER CHECK ({resolve_teacher(args)} flying the policy's part, n={N}): "
              + " ".join(f"{k}={tv[k]:.3f}" for k in ("hit", "roll", "pitch", "oob", "hover", "timeout", "caught", "unfinished")))
        print("[rtl]   teacher flight (while handed off): "
              + " ".join(f"{k}={tv[k]:.2f}" for k in ("mean_speed", "min_dist_p10", "min_dist_p50", "min_dist_p90", "ho_speed_mean", "ho_leg_s") if k in tv))
        for e, i, d, sp, cl, la, dz, ro, pi_, cro, cpi in tv.get("trace", []):
            print(f"[rtl]   trace env{e} t={i * float(u.step_dt):6.1f}s dist={d:6.2f} speed={sp:5.2f} closing={cl:+6.2f} "
                  f"lateral={la:5.2f} above_target={dz:+6.2f} roll/pitch={ro:+5.1f}/{pi_:+5.1f} cmd={cro:+5.1f}/{cpi:+5.1f}")
        if tv["hit"] < 0.5:
            print("[rtl] WARNING: the teacher itself hits < 50% -- distilling from it cannot work; fix the teacher first")
        if args.teacher_check_only:
            print("[rtl] --teacher-check-only: done")
            return
    u._stagger_next = True
    obs = env.reset()[0]["policy"]
    u._stagger_next = False
    teacher.reset(None)
    pop_reward_stats(u)  # drop anything accumulated by reset/imitation
    ep_ret = torch.zeros(N, device=device)

    if args.teacher_bc_chunks > 0 and anchor is not None:
        # supervised DAgger on the teacher labels before PPO: the student flies, the teacher labels its states, the actor
        # regresses onto them. Inside PPO the same pull is throttled by --target-kl, far too slow from a random start.
        # DAgger: the teacher flies a fraction beta of the envs (1 -> 0 over the first 70% of the chunks) so the data starts
        # on normal flight instead of the random student's crashes; labels aggregate across chunks.
        bc_opt = torch.optim.AdamW(model.actor_params(), lr=args.teacher_bc_lr, weight_decay=args.weight_decay)
        agg_X, agg_Y, cap, mb = None, None, 4_000_000, 8192
        for c in range(args.teacher_bc_chunks):
            beta = max(0.0, 1.0 - c / max(1.0, 0.7 * args.teacher_bc_chunks))
            bb, obs, _s = collect_rollout(env, model, teacher, obs, args.num_steps_per_chunk, args.gamma, False, ep_ret,
                                          anchor=anchor, expert_beta=beta, deterministic=True)
            keep = bb["pm"].reshape(-1) > 0               # only the policy's part of the flight
            X = bb["obs"].reshape(-1, bb["obs"].shape[-1])[keep]
            Y = bb["teach"].reshape(-1, bb["teach"].shape[-1])[keep]
            del bb
            agg_X = X if agg_X is None else torch.cat([agg_X, X])[-cap:]
            agg_Y = Y if agg_Y is None else torch.cat([agg_Y, Y])[-cap:]
            n = agg_X.shape[0]
            n_steps = max(200, args.teacher_bc_epochs * -(-n // mb))
            tot = 0.0
            for _k in range(n_steps):
                b = torch.randint(0, n, (mb,), device=device)
                loss = (torch.tanh(model.forward(agg_X[b])[0]) - agg_Y[b]).pow(2).mean()
                bc_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor_params(), args.max_grad_norm)
                bc_opt.step()
                tot += loss.item() / n_steps
            st = pop_reward_stats(u)
            print(f"[rtl] teacher BC {c + 1}/{args.teacher_bc_chunks}: beta={beta:.2f} mse={tot:.4f} new={X.shape[0]} "
                  f"buffer={n} grad_steps={n_steps} hit={st.get('train_hit_rate', NAN):.3f} roll={st.get('train_roll_rate', NAN):.3f}")
        del bc_opt, agg_X, agg_Y
        ev_bc = isaac_eval(env, model, teacher, args.eval_steps)
        print(f"[rtl] after teacher BC, student eval: hit={ev_bc['hit']:.3f} roll={ev_bc['roll']:.3f} "
              f"timeout={ev_bc['timeout']:.3f} oob={ev_bc['oob']:.3f}")
        u._stagger_next = True
        obs = env.reset()[0]["policy"]
        u._stagger_next = False
        u.episode_length_buf.zero_()
        teacher.reset(None)
        pop_reward_stats(u)
        ep_ret.zero_()

    N_EVAL = N   # every env contributes one first-episode outcome per eval
    pid_row, pid_success = {}, None
    eval_row, success, caught = {}, NAN, NAN
    baseline_hits, hit_hist = [], []
    best_ma, boost, best_score, best_info = -1.0, 1.0, -1e9, None
    actor_idx = 0          # chunks in which the actor was trainable so far
    group_idx = {"trunk": 0, "head": 0, "critic": 0}
    prev_phase, aborted = None, None
    t0 = time.time()
    print(f"[rtl] {n_chunks} chunks x {chunk_steps} steps | settle {args.settle_chunks}, warmup {args.warmup_chunks}, "
          f"head {args.head_chunks}, then full | reward_qv2 {rcfg.to_dict()}")

    for chunk in range(n_chunks):
        phase = phase_of(chunk, args.settle_chunks, args.warmup_chunks, args.head_chunks)
        if phase != prev_phase:
            print(f"[rtl] ---- chunk {chunk}: phase -> {phase}")
            prev_phase = phase
        apply_phase(model, phase)
        train_actor = phase in ("head_only", "full")
        train_critic = phase != "settle"
        active = {"trunk": phase == "full", "head": train_actor, "critic": train_critic}

        lrs = {}
        for g in optimizer.param_groups:
            nme = g["name"]
            lr = base_lr[nme]
            if nme != "critic" and active[nme]:
                lr *= cosine_factor((chunk - first_trainable) / max(1, n_chunks - first_trainable), args.lr_min_ratio)
            if active[nme]:
                lr *= min(1.0, (group_idx[nme] + 1) / max(1, args.lr_ramp_chunks))
            g["lr"] = lr
            lrs[nme] = lr

        ds = distill_scheduled(actor_idx, args.distill_start, args.distill_end, args.distill_decay_chunks)
        coef_eff = min(args.distill_max, ds * boost) if train_actor else 0.0

        tc = time.time()
        batch, obs, roll_stats = collect_rollout(env, model, teacher, obs, args.num_steps_per_chunk,
                                                 args.gamma, args.bootstrap_timeouts, ep_ret, anchor=anchor)
        t_roll = time.time() - tc
        # a frozen-model anchor is good at high speed too (unlike a hover PID), so no speed fade for it
        batch["tw"] = (torch.ones(batch["obs"].shape[:-1], device=device) if anchor is not None
                       else teacher_weight(batch["obs"], args.distill_fade_speed, VEL_SCALE))
        train_stats = pop_reward_stats(u)
        agree = teacher_agreement(model, batch)

        if phase == "settle":
            upd = {k: NAN for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "distill_loss",
                                    "gn_actor", "gn_critic", "epochs_run", "explained_var", "adv_abs_mean",
                                    "ret_mean", "val_mean")}
            upd["early_stopped"] = False
        else:
            tu = time.time()
            upd = ppo_distill_update(
                model, optimizer, batch, gamma=args.gamma, lam=args.lam, clip_eps=args.clip_eps,
                vf_coef=args.vf_coef, ent_coef=args.ent_coef, distill_coef=coef_eff,
                epochs=args.epochs if train_actor else args.warmup_epochs, num_minibatches=args.num_minibatches,
                target_kl=args.target_kl, max_grad_norm=args.max_grad_norm,
                train_actor=train_actor, train_critic=train_critic)
            upd["t_update_s"] = time.time() - tu
        for nme in group_idx:
            if active[nme]:
                group_idx[nme] += 1
        if train_actor:
            actor_idx += 1
        timesteps = (chunk + 1) * chunk_steps

        # ---- Isaac eval: deterministic student (and every pid_eval_every chunks the PID) on freshly reset envs ----
        did_eval = chunk % max(1, args.eval_every) == 0 or chunk == n_chunks - 1
        eval_row = {}
        if did_eval:
            te = time.time()
            ev = isaac_eval(env, model, teacher, args.eval_steps)
            success = ev["hit"]
            caught = ev.get("caught", NAN)
            eval_row = {f"eval_{k}": v for k, v in ev.items()}
            eval_row["success_rate"] = success
            if chunk % max(1, args.pid_eval_every) == 0 or not pid_row:
                pv = isaac_eval(env, model, teacher, args.eval_steps, use_pid=True)
                pid_success = pv["hit"]
                pid_row = {f"pid_{k}": v for k, v in pv.items()}
                pid_row["pid_success_rate"] = pid_success
            # eval reset every env and ran with other actions: drop its stats and start training from fresh envs
            pop_reward_stats(u)
            u._stagger_next = True
            obs = env.reset()[0]["policy"]
            u._stagger_next = False
            u.episode_length_buf.zero_()   # else random early truncations = fake -r_timeout hits in training
            teacher.reset(None)
            u._qv2_stall.zero_()
            # eval resets put all envs in lock-step: burn in with the noisy policy (data discarded) to desync
            if args.eval_burnin > 0:
                _b, obs, _s = collect_rollout(env, model, teacher, obs, args.eval_burnin, args.gamma, False, ep_ret, anchor=anchor)
                del _b
            ep_ret.zero_()
            pop_reward_stats(u)
            eval_row["t_eval_s"] = time.time() - te

        # ---- teacher-anchor guard + abort criterion ----
        hit_rate = train_stats["train_hit_rate"]
        # chunks with too few finished episodes don't count (they used to read as train_hit=0 and trip the guard)
        enough_eps = train_stats["train_episodes"] >= (args.guard_min_episodes if args.guard_min_episodes is not None
                                                         else max(50, int(0.04 * N)))
        if phase == "critic_warmup" and enough_eps:
            baseline_hits.append(hit_rate)
        baseline = float(np.mean(baseline_hits)) if baseline_hits else NAN
        guard_note = ""
        # the guard uses the unbiased EVAL hit rate, judged at eval chunks only
        if did_eval and phase == "settle":
            best_ma = max(best_ma, success)                  # the untouched start policy sets the bar
        if train_actor and did_eval:
            if success < best_ma - args.guard_margin:
                boost = min(args.guard_boost_max, boost * args.guard_gain)
                guard_note = f" GUARD: eval {success:.3f} < best {best_ma:.3f}-{args.guard_margin} -> boost x{boost:.1f}"
            else:
                best_ma = max(best_ma, success)
                boost = max(1.0, boost * 0.9)
        if train_actor and enough_eps:
            hit_hist.append(hit_rate)
            if len(hit_hist) >= args.abort_window and float(np.mean(hit_hist[-args.abort_window:])) < args.abort_hit_rate:
                aborted = (f"train hit rate {np.mean(hit_hist[-args.abort_window:]):.3f} < {args.abort_hit_rate} "
                           f"over the last {args.abort_window} trainable chunks")

        std = model.effective_std().detach().cpu().tolist()
        row = {
            "chunk": chunk, "phase": phase, "stage": phase, "timesteps": timesteps,
            "wall_s": time.time() - t0, "steps_per_s": chunk_steps / max(t_roll, 1e-9),
            "lr_trunk": lrs["trunk"], "lr_head": lrs["head"], "lr_critic": lrs["critic"],
            "teacher_weight_mean": batch["tw"].mean().item(),
            "distill_coef_sched": ds if train_actor else 0.0, "distill_boost": boost, "distill_coef_eff": coef_eff,
            "ent_coef": args.ent_coef if train_actor else 0.0,
            **upd, **roll_stats, **agree, **train_stats,
            "baseline_train_hit_rate": baseline, "best_eval_hit": best_ma,
            "std_mean": float(np.mean(std)), **{f"std_d{i}": s for i, s in enumerate(std)},
            **eval_row, **(pid_row if did_eval else {}),
            "total_param_norm": sum(p.data.norm().item() for p in model.actor_params()),
        }

        ckpt = os.path.join(args.checkpoint_dir, f"model_{timesteps}.pt")
        sd = model.actor_state_dict()
        torch.save(sd, ckpt)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "chunk": chunk,
                    "boost": boost, "args": vars(args)}, os.path.join(args.checkpoint_dir, "latest_full.pt"))
        if args.onnx:
            try:
                plain.load_state_dict(sd)
                export_onnx_model(plain, obs_dim, ckpt[:-3] + ".onnx")
            except Exception as e:  # never let an export problem kill a long run
                print(f"[rtl] onnx export failed (continuing): {e}")

        # best checkpoint = highest Isaac eval hit rate (deterministic policy, handoff task), only on eval chunks
        score = success if did_eval else NAN
        row["score"] = score
        if train_actor and did_eval and score > best_score:
            best_score = score
            best_info = {"checkpoint": ckpt, "chunk": chunk, "phase": phase, "score": score,
                         "train_hit_rate": hit_rate, "eval_hit_rate": success, "timesteps": timesteps}
            torch.save(sd, os.path.join(args.checkpoint_dir, "best.pt"))
            with open(os.path.join(args.checkpoint_dir, "BEST.json"), "w") as f:
                json.dump(best_info, f, indent=2)
        row["best_score"] = best_score if best_score > -1e8 else NAN

        log.add(row)
        try:
            _append_json_row({k: _py(v) for k, v in log.rows[-1].items()}, METRICS_JSON_PATH, run_key)
        except Exception as e:
            print(f"[rtl] json log failed (continuing): {e}")
        log_metrics_safe({k: v for k, v in log.rows[-1].items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                          and k not in ("stage",)}, step=timesteps)
        bucket_str = ""
        if did_eval and "eval_hit_b0" in eval_row:
            bucket_str = " [b0/b1/b2 " + "/".join("%.2f" % eval_row["eval_hit_b%d" % i] for i in range(3)) + "]"
        pid_txt = "%.2f" % pid_success if pid_success is not None else "nan"
        if did_eval:   # how the deterministic eval episodes ended (every env's first episode, Isaac only)
            ends = " ".join("%s=%.3f" % (k, eval_row["eval_" + k]) for k in
                            ("hit", "caught", "hover", "oob", "roll", "pitch", "timeout", "unfinished"))
            print(f"[rtl]   eval endings: {ends}" + (f" | crash by bucket "
                  + "/".join("%.2f" % eval_row["eval_crash_b%d" % i] for i in range(3)) if "eval_crash_b0" in eval_row else ""))
            if "eval_hit_g0" in eval_row:
                geo = " | ".join("%s n=%.0f h=%.2f c=%.2f a=%.2f o=%.2f t=%.2f hv=%.2f" % (
                    nm, *(eval_row[f"eval_{k}_g{i}"] for k in ("n", "hit", "caught", "att", "oob", "timeout", "hover")))
                    for i, nm in enumerate(u.BUCKET_NAMES))
                print(f"[rtl]   by geometry (h hit, c caught, a attitude, o oob, t timeout, hv hover): {geo}")
            if "eval_ho_n" in eval_row:
                e = eval_row
                print(f"[rtl]   handoff check (n={e['eval_ho_n']:.0f}, before perturbation): "
                      f"dist actual-h mean {e['eval_ho_dist_err_mean']:+.2f} m (p10 {e['eval_ho_dist_err_p10']:+.2f}, "
                      f"p90 {e['eval_ho_dist_err_p90']:+.2f}) | speed actual-bucket mean {e['eval_ho_speed_err_mean']:+.2f} "
                      f"m/s (p10 {e['eval_ho_speed_err_p10']:+.2f}, p90 {e['eval_ho_speed_err_p90']:+.2f}) | "
                      f"speed {e['eval_ho_speed_mean']:.1f} m/s tilt {e['eval_ho_tilt_mean']:.1f} deg | "
                      f"leg {e['eval_ho_leg_s']:.1f} s | handed off >0.5 m farther than h: {e['eval_ho_off_frac']:.1%}")
        if args.num_threats and sum(train_stats.get(f"train_n_{lb}", 0) for lb in BUCKET_LABELS) > 0:   # train episodes by chaser geometry
            tg = " | ".join("%s n=%.0f h=%.2f c=%.2f a=%.2f o=%.2f t=%.2f hv=%.2f" % (
                lb, *(train_stats[f"train_{k}_{lb}"] for k in ("n", "hit", "caught", "att", "oob", "timeout", "hover")))
                for lb in BUCKET_LABELS)
            print(f"[rtl]   train by geometry (h hit, c caught, a attitude, o oob, t timeout, hv hover): {tg}")
        print(f"[rtl] {chunk + 1}/{n_chunks} {phase:13s} train_hit={hit_rate:.3f} (base {baseline:.3f}) "
              f"eval={success:.2f}{bucket_str} caught={caught:.3f} pid={pid_txt} timeout={train_stats['train_timeout_rate']:.3f} "
              f"hover={train_stats['train_hover_rate']:.3f} "
              f"stall={train_stats['train_stall_frac']:.3f} agree={agree['agree_mae']:.3f} "
              f"d_coef={coef_eff:.3f} kl={upd.get('approx_kl', NAN):.4f} vloss={upd.get('value_loss', NAN):.2f} "
              f"ev={upd.get('explained_var', NAN):.2f} {steps_str(row['steps_per_s'])}{guard_note}")

        bad = [k for k in ("policy_loss", "value_loss") if upd.get(k) is not None and upd[k] == upd[k] and
               not math.isfinite(upd[k])]
        if bad:
            aborted = f"non-finite {bad}"
        if aborted:
            print(f"[rtl] ABORT at chunk {chunk}: {aborted}")
            break

    # ---- wrap-up ----
    summary = {"aborted": aborted, "best": best_info, "chunks_run": len(log.rows),
               "baseline_train_hit_rate": float(np.mean(baseline_hits)) if baseline_hits else None,
               "pid_success_rate": pid_success, "duration_s": time.time() - t0}
    with open(os.path.join(args.checkpoint_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    try:
        plot_summary(log.rows, os.path.join(args.checkpoint_dir, "summary.png"), pid_success)
    except Exception as e:
        print(f"[rtl] summary plot failed: {e}")
    try:
        plot_training_run(log.csv_path, output_dir=os.path.join(args.checkpoint_dir, "plots"),
                          hover_success_steps=200, n_diag_episodes=N_EVAL, hit_threshold=rcfg.hit_threshold,
                          max_steps=int(u.max_episode_length), oob_radius=oob_radius, hit_reward=rcfg.r_hit,
                          attitude_penalty=-rcfg.r_crash, oob_penalty=-rcfg.r_crash)
    except Exception as e:
        print(f"[rtl] legacy plot_training_run failed (summary.png is the primary plot): {e}")
    mlflow.set_tag("aborted", str(aborted))
    if best_info:
        mlflow.set_tag("best_checkpoint", best_info["checkpoint"])
    mlflow.log_artifact(log.csv_path)
    mlflow.end_run()
    print(f"[rtl] done. {json.dumps(summary, default=str)}")
    return model


def steps_str(sps):
    return f"{sps / 1e3:.0f}k step/s"


def make_env_cfg(args, num_envs=None):
    """The env cfg a run uses (observation layout, threat slots, timer): shared by main() and the web viewer's Isaac worker."""
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args.num_envs if num_envs is None else num_envs)
    from app.environmental.base_drone_env_isaac import THREAT_FEATURES, TIMER_FEATURES
    env_cfg.max_threats = args.num_threats
    env_cfg.threat_radius = args.threat_radius
    env_cfg.timer_obs = args.reward == "rc" and not args.no_timer_obs
    env_cfg.observation_space = 23 + THREAT_FEATURES * args.num_threats + TIMER_FEATURES * env_cfg.timer_obs
    env_cfg.obs_layout = args.obs_layout
    env_cfg.max_allies = args.num_allies
    if args.obs_layout == "v2":
        from app.environmental.perception import ALLY_FEATURES, FLAG_FEATURES, OPPONENT_FEATURES
        env_cfg.observation_space = (23 + TIMER_FEATURES * env_cfg.timer_obs + OPPONENT_FEATURES * args.num_threats
                                     + ALLY_FEATURES * args.num_allies + FLAG_FEATURES)
        print(f"[rtl] OBSERVATION v2: 23 base + {TIMER_FEATURES * env_cfg.timer_obs} timer + {OPPONENT_FEATURES}*{args.num_threats} opponents "
              f"+ {ALLY_FEATURES}*{args.num_allies} allies + {FLAG_FEATURES} flags = {env_cfg.observation_space}")
    return env_cfg


def main():
    parser = build_parser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.bc_explicit = args.bc_checkpoint_path is not None
    if not args.bc_explicit:
        args.bc_checkpoint_path = DEFAULT_BC_CHECKPOINT
    sim_app = AppLauncher(args).app

    import gymnasium as gym
    import app.environmental.base_drone_env_isaac  # noqa: F401  (registers the gym id)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = make_env_cfg(args)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)
    try:
        run(env, args)
    except BaseException:
        # print now: SimulationApp.close() can hard-exit and swallow the traceback
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        env.close()
        sim_app.close()


if __name__ == "__main__":
    main()
