"""Acro-style trainer: DAgger-BC from a teacher, then PPO + distillation with the quaternion tilt-limit reward.
Teacher (--teacher): los = the angle trainer's straight-line-of-sight controller (make_los_label), its angle-style torques
turned into body-rate commands; checkpoint = a frozen angle-style policy (--teacher-checkpoint); acro = a frozen acro-style policy (e.g. the previous stage best.pt),
labels used as-is; none.
Staged use: stage 1 = the los teacher (BC + PPO), later stages = --bc-checkpoint <previous best.pt> with a wider
--tilt-limit and no teacher. See app/training/docs.md "acro_rtl_train_isaac"."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow_isaac.db")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from app.guidance.train import ActorCritic, load_bc_checkpoint
from app.training.config import WEIGHT_DECAY
from app.training.distance_buckets import DEFAULT_BUCKETS, describe_buckets, parse_buckets
# angle trainer pieces that do not depend on the angle env / PID teacher (Isaac is only imported inside its functions)
from app.training.angle_rtl_train_isaac import (
    NAN, PidTeacher, RTLActorCritic, RunLog, _py, apply_phase, cosine_factor, distill_scheduled, log_prob_raw,
    make_los_label, phase_of, ppo_distill_update, set_initial_std, steps_str, teacher_agreement,
    SPEED_BUCKETS, SPEED_BUCKET_PROBS, install_handoff_stage,
)

EVAL_CODES = {"hit": 1, "hover": 2, "oob": 3, "tilt": 4, "timeout": 5, "near_miss": 6, "escape": 7}   # near_miss: acro_reward_v2 timer;
# escape: --runner, the target left the sphere of --runner-escape-dist around its spawn point


# ===== Teacher: frozen angle-style policy, relabelled into acro actions =====

class AngleTeacher:
    """Frozen angle-style actor. Its action [thrust_delta, tau_x, tau_y, tau_z] is turned into the acro action
    [thrust_delta, p, q, r] whose first-substep torque is the same: the rate loop outputs I*kp*(cmd - w), so
    cmd = w + tau / (I*kp). Queried on the states the STUDENT visits (DAgger labels)."""

    def __init__(self, path, env, device):
        from app.guidance.export_onnx import infer_architecture
        raw = torch.load(path, map_location="cpu")
        obs_dim, hidden, layers, a_dim = infer_architecture(raw)
        u = env.unwrapped
        env_obs = u.single_observation_space["policy"].shape[0]
        # an angle teacher trained with the zone timer has 2 extra trailing inputs (in_zone, zone_count/needed) that this env
        # does not provide; they are 0 away from the zone, so pad with zeros. Anything else that does not fit is still an error.
        assert a_dim == 4 and (obs_dim <= env_obs or obs_dim - env_obs == 2), \
            f"teacher dims {(obs_dim, a_dim)} do not fit the env ({env_obs} inputs)"     # threat slots are appended after its inputs
        self.obs_dim = obs_dim
        self.env_obs = env_obs
        self.u = u
        self.low = torch.as_tensor(u.single_action_space.low, device=device)
        self.high = torch.as_tensor(u.single_action_space.high, device=device)
        angle_low = raw["action_low"] if "action_low" in raw else self.low.cpu()
        angle_high = raw["action_high"] if "action_high" in raw else self.high.cpu()
        self.net = ActorCritic(obs_dim, a_dim, angle_low, angle_high, hidden=hidden, num_hidden_layers=layers,
                               dropout=0.0).to(device)
        load_bc_checkpoint(self.net, path, map_location=device)
        self.net.eval()
        for q in self.net.parameters():
            q.requires_grad_(False)
        self.arch = (hidden, layers)
        print(f"[rtl] TEACHER: frozen angle-style policy {path} (hidden={hidden}, layers={layers}, "
              f"angle action bounds {self.net.action_low.tolist()} .. {self.net.action_high.tolist()})")

    @torch.no_grad()
    def label(self, obs):
        """Returns (normalised label in [-1, 1], action in acro units)."""
        o = obs[:, :self.obs_dim]
        if o.shape[-1] < self.obs_dim:   # missing zone-timer inputs -> zeros
            o = torch.cat([o, torch.zeros(o.shape[0], self.obs_dim - o.shape[-1], device=o.device)], dim=-1)
        a = self.net.scale_action(self.net.forward(o)[0])
        act = angle_to_acro(self.u, a, self.low, self.high)
        return torch.clamp(2.0 * (act - self.low) / (self.high - self.low) - 1.0, -1.0, 1.0), act


class AcroTeacher:
    """Frozen ACRO-style actor (e.g. the previous stage's best.pt). Its action is already [thrust_delta, p, q, r] in this
    env's units: no angle_to_acro conversion. Same interface as AngleTeacher: label(obs) -> (normalised label, action)."""

    def __init__(self, path, env, device):
        from app.guidance.export_onnx import infer_architecture
        raw = torch.load(path, map_location="cpu")
        obs_dim, hidden, layers, a_dim = infer_architecture(raw)
        u = env.unwrapped
        env_obs = u.single_observation_space["policy"].shape[0]
        assert a_dim == 4 and obs_dim <= env_obs, f"teacher dims {(obs_dim, a_dim)} do not fit the env ({env_obs} inputs)"
        self.obs_dim = obs_dim
        self.low = torch.as_tensor(u.single_action_space.low, device=device)
        self.high = torch.as_tensor(u.single_action_space.high, device=device)
        self.net = ActorCritic(obs_dim, a_dim, self.low.cpu(), self.high.cpu(), hidden=hidden, num_hidden_layers=layers,
                               dropout=0.0).to(device)
        load_bc_checkpoint(self.net, path, map_location=device)
        ck_low, ck_high = self.net.action_low.clone(), self.net.action_high.clone()
        if not (torch.allclose(ck_low, self.low) and torch.allclose(ck_high, self.high)):
            raise SystemExit(f"--teacher acro: {path!r} has action bounds {ck_low.tolist()} .. {ck_high.tolist()}, env uses "
                             f"{self.low.tolist()} .. {self.high.tolist()}. Is it an angle-style policy? Use --teacher checkpoint for those.")
        self.net.eval()
        for q in self.net.parameters():
            q.requires_grad_(False)
        self.arch = (hidden, layers)
        print(f"[rtl] TEACHER: frozen ACRO policy {path} (hidden={hidden}, layers={layers})")

    @torch.no_grad()
    def label(self, obs):
        mean = self.net.forward(obs[:, :self.obs_dim])[0]
        return torch.tanh(mean), self.net.scale_action(mean)


class LosTeacher:
    """The angle trainer's LOS teacher (fly the line of sight at the braking-limited speed) as an acro teacher. make_los_label gives
    an angle-style action [thrust_delta, tau_x, tau_y, tau_z]; angle_to_acro turns it into the rate command with the same
    first-substep torque. Same interface as AngleTeacher: label(obs) -> (normalised label, action in acro units)."""

    def __init__(self, env, gains_by_dist, args):
        import types
        pt = PidTeacher(env, gains_by_dist)      # only for its short-range gain table and the angle-bounds PID the labels are made with
        self.u = env.unwrapped
        self.low = torch.as_tensor(self.u.single_action_space.low, device=self.u.device)
        self.high = torch.as_tensor(self.u.single_action_space.high, device=self.u.device)
        self.angle_low, self.angle_high = pt.pid.action_low, pt.pid.action_high
        # make_los_label normalises with teacher.low / high: here those are the ANGLE action bounds, not this env's rate bounds
        angle_side = types.SimpleNamespace(pid=pt.pid, table=pt.table, key_dists=pt.key_dists, low=self.angle_low,
                                           high=self.angle_high, _read_kinematics=pt._read_kinematics)
        print(f"[rtl] TEACHER: los (straight line of sight, braking-limited, max {args.max_speed} m/s) -> body-rate commands")
        self._angle_label = make_los_label(env, angle_side, args.max_speed, args.teacher_accel, args.teacher_cruise_gain,
                                           stop_speed=args.los_stop_speed, max_tilt=math.radians(args.los_max_tilt_deg))

    @torch.no_grad()
    def label(self, obs):
        a = self.angle_low + (self._angle_label(obs) + 1.0) * 0.5 * (self.angle_high - self.angle_low)
        act = angle_to_acro(self.u, a, self.low, self.high)
        return torch.clamp(2.0 * (act - self.low) / (self.high - self.low) - 1.0, -1.0, 1.0), act


def angle_to_acro(u, a, low, high):
    """Angle-style action [thrust_delta, tau_x, tau_y, tau_z] -> acro action [thrust_delta, p, q, r] whose first-substep torque is
    the same: the rate loop outputs I*kp*(cmd - w), so cmd = w + tau/(I*kp). Clamped to the acro bounds."""
    rc = u._rate_ctrl
    cmd = u._robot.data.root_ang_vel_b + a[:, 1:4] / (rc.inertia * rc.kp)
    return torch.max(torch.min(torch.cat([a[:, :1], cmd], dim=-1), high), low)


def controller_action(env, u, low, high):
    """Handoff stage: the acro action the scripted carrot-chase controller wants for the envs it still flies (full batch)."""
    from app.training.base_training_isaac import _read_kinematics
    pos, vel, ang_vel, roll, pitch, yaw, _ = _read_kinematics(env)
    return angle_to_acro(u, u._carrot_action(pos, vel, ang_vel, roll, pitch, yaw), low, high)


# ===== Rollout / BC =====

@torch.no_grad()
def collect_rollout(env, model, teacher, obs, num_steps, gamma, bootstrap_timeouts, ep_ret, beta=0.0):
    """One chunk of on-policy data. With beta > 0 a random fraction of envs fly the teacher action (DAgger mixing,
    pm = 0 so PPO ignores those steps). Returns (batch, last_obs, stats)."""
    u = env.unwrapped
    N, dev = u.num_envs, u.device
    obs_dim, a_dim = obs.shape[-1], model.actor_mean.out_features
    b_obs = torch.zeros(num_steps, N, obs_dim, device=dev)
    b_raw = torch.zeros(num_steps, N, a_dim, device=dev)
    b_logp, b_rew, b_val, b_done, b_pm = (torch.zeros(num_steps, N, device=dev) for _ in range(5))
    b_teach = torch.zeros(num_steps, N, a_dim, device=dev)
    ret_sum, ret_n = torch.zeros((), device=dev), torch.zeros((), device=dev)
    handoff = getattr(u, "_handoff_snap_enabled", False)   # two-stage episodes: the carrot controller flies until the handoff
    if handoff:
        low_a, high_a = torch.as_tensor(u.single_action_space.low, device=dev), torch.as_tensor(u.single_action_space.high, device=dev)

    for t in range(num_steps):
        mean, std, value = model.forward(obs)
        raw = mean + std * torch.randn_like(mean)
        logp = log_prob_raw(model, mean, std, raw)
        action = model.scale_action(raw)
        pm = torch.ones(N, device=dev)
        if teacher is not None:
            label, t_action = teacher.label(obs)
            b_teach[t] = label
            if beta > 0:
                use_t = torch.rand(N, device=dev) < beta
                action = torch.where(use_t.unsqueeze(-1), t_action, action)
                pm = (~use_t).float()
        if handoff:
            ctrl = ~u._handed_off                          # controller still flies this env: PPO ignores the step (pm = 0)
            action = torch.where(ctrl.unsqueeze(-1), controller_action(env, u, low_a, high_a), action)
            pm = pm * (~ctrl).float()
            u._prefix_left.sub_(1).clamp_(min=0)
        next_dict, reward, terminated, truncated, extras = env.step(action)
        next_obs = next_dict["policy"]
        done = terminated | truncated
        if handoff:
            u._refresh_handoff()
            js = torch.nonzero(ctrl & u._handed_off & ~done, as_tuple=False).squeeze(-1)
            u._handoff_snap_obs(next_obs, js)              # random sideways velocity + tilt at the takeover (patches next_obs)

        if bootstrap_timeouts:
            trunc_only = truncated & ~terminated
            if trunc_only.any():
                boot_obs = torch.where(trunc_only.unsqueeze(-1), extras["terminal_observation"], next_obs)
                reward = reward + gamma * model.forward(boot_obs)[2] * trunc_only.float()

        b_obs[t], b_raw[t], b_logp[t], b_rew[t], b_val[t], b_done[t], b_pm[t] = \
            obs, raw, logp, reward, value, done.float(), pm
        ep_ret += reward
        ret_sum += (ep_ret * done.float()).sum()
        ret_n += done.sum()
        ep_ret *= (~done).float()
        obs = next_obs

    batch = dict(obs=b_obs, raw=b_raw, logp=b_logp, rew=b_rew, val=b_val, done=b_done, teach=b_teach, pm=b_pm,
                 last_value=model.forward(obs)[2])
    stats = {"ep_return_mean": (ret_sum / ret_n.clamp(min=1)).item() if ret_n.item() > 0 else NAN,
             "teacher_step_frac": 1.0 - b_pm.mean().item()}
    return batch, obs, stats


def install_stagger(u, speed):
    """Staggered start, like the angle trainer's: on a FULL reset with u._stagger_next set, every env starts at a random
    point (0-70% of the leg) on the spawn->target line, already moving toward the target at U(0, speed) m/s."""
    import types
    u._stagger_next = False
    u._stagger_speed = float(speed)
    orig_reset = u._reset_idx

    def _reset_idx(self, env_ids):
        orig_reset(env_ids)
        if not self._stagger_next or not (env_ids is None or len(env_ids) == self.num_envs):
            return
        ids, org, n = self._robot._ALL_INDICES, self._terrain.env_origins, self.num_envs
        tgt = self._desired_pos_w - org
        pos = self._prev_position.clone()   # spawn point written by the base reset
        frac = 0.7 * torch.rand(n, device=self.device)   # never closer than 30% of the leg
        pos0 = pos + (tgt - pos) * frac.unsqueeze(-1)
        unit = (tgt - pos) / torch.linalg.norm(tgt - pos, dim=-1, keepdim=True).clamp_min(1e-6)
        vel = unit * (torch.rand(n, device=self.device) * self._stagger_speed).unsqueeze(-1)
        self._robot.write_root_pose_to_sim(torch.cat([pos0 + org, self._robot.data.default_root_state[:, 3:7]], dim=-1), ids)
        self._robot.write_root_velocity_to_sim(torch.cat([vel, torch.zeros_like(vel)], dim=-1), ids)
        self._prev_distance[ids] = torch.linalg.norm(tgt - pos0, dim=-1)
        self._prev_position[ids] = pos0

    u._reset_idx = types.MethodType(_reset_idx, u)


def reset_and_desync(env, model, teacher, args, ep_ret):
    """Fresh reset after an eval: keep the env's staggered episode clocks (do NOT zero them: that is what put every env in
    lock-step and made the stats saw-tooth), clear the stall/away counters, optionally burn in (data discarded). Returns obs."""
    from app.reward_functions.acro_reward_v1 import pop_acro_stats
    u = env.unwrapped
    u._stagger_next = True      # random start point on the spawn->target line, like the angle trainer
    obs = env.reset()[0]["policy"]   # the env staggers episode_length_buf on a full reset; the reward treats the first (shortened) episode as partial
    u._stagger_next = False
    u._acro_stall.zero_()
    u._acro_away.zero_()
    if args.eval_burnin > 0:
        _b, obs, _s = collect_rollout(env, model, teacher, obs, args.eval_burnin, args.gamma, False, ep_ret)
        del _b
    if args.desync_steps > 0:
        # The staggered clock only spreads TIMEOUTS. Fast endings (hit ~3.4 s, crash) still happen at the same moment in every
        # env because they all restart from rest together, so the stats came in waves. Fix: force-reset each env once, at a
        # random step inside the window, so episode STARTS are spread out. The data of this phase is discarded.
        kill_at = torch.randint(0, args.desync_steps, (u.num_envs,), device=u.device)
        for t in range(args.desync_steps):
            _b, obs, _s = collect_rollout(env, model, teacher, obs, 1, args.gamma, False, ep_ret)
            ids = torch.nonzero(kill_at == t, as_tuple=False).squeeze(-1)
            if ids.numel() > 0:
                u._reset_idx(ids)
                u._acro_stall[ids] = 0
                u._acro_away[ids] = 0
        del _b
    ep_ret.zero_()
    pop_acro_stats(u)
    return obs


class DaggerBuffer:
    """Aggregated (obs, teacher label) pairs from every BC chunk so far (the D in DAgger). Each chunk adds a random
    subsample; when over `cap` the oldest chunks are dropped. States the student reached by drifting stay in the
    set, so later updates cannot forget them."""

    def __init__(self, cap):
        self.cap, self.chunks, self.n, self._cat = cap, [], 0, None

    def add(self, batch, per_chunk):
        T, N = batch["rew"].shape
        obs = batch["obs"].reshape(T * N, -1)
        teach = batch["teach"].reshape(T * N, -1)
        idx = torch.randperm(T * N, device=obs.device)[:per_chunk]
        self.chunks.append((obs[idx].clone(), teach[idx].clone()))
        self.n += idx.numel()
        while self.n > self.cap and len(self.chunks) > 1:
            self.n -= self.chunks.pop(0)[0].shape[0]
        self._cat = None

    def tensors(self):
        """(obs, labels) of the whole buffer; concatenated once per add, not once per bc_train call."""
        if self._cat is None:
            self._cat = (torch.cat([c[0] for c in self.chunks]), torch.cat([c[1] for c in self.chunks]))
        return self._cat


def bc_train(model, optimizer, buf, steps, minibatch):
    """`steps` minibatches of MSE(tanh(mean), teacher label), drawn uniformly from the whole aggregated buffer.
    Returns (mean loss, per-dimension loss [thrust, p, q, r])."""
    obs, teach = buf.tensors()
    loss_sum, dim_sum = 0.0, torch.zeros(teach.shape[1], device=obs.device)
    for _ in range(steps):
        idx = torch.randint(0, obs.shape[0], (minibatch,), device=obs.device)
        err = (torch.tanh(model.forward(obs[idx])[0]) - teach[idx]).pow(2)
        loss = err.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor_params(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        dim_sum += err.detach().mean(0)
    return loss_sum / max(steps, 1), (dim_sum / max(steps, 1)).tolist()   # (mean loss, per action dimension)


# ===== Eval =====

@torch.no_grad()
def isaac_eval(env, model, steps, tilt_limit_deg, noisy=False, teacher=None, dist_buckets=None):
    """Eval on the real task; first episode per env only, with a per-distance time budget. noisy=False: tanh(mean);
    noisy=True: the exploration noise of training (to tell a policy that needs its noise from one that does not).

    `dist_buckets`: None pins the eval to U(--distance-low, --distance-high), even when the trainer samples a distance
    curriculum (the saved curriculum is restored right after the reset that samples it). Pass the bucket list to run the
    pass on the curriculum instead: the returned `hit_d*/crash_d*/n_d*` rows are grouped by the bucket each env was
    spawned in.
    """
    from app.control.step_budget import steps_for_dist
    u = env.unwrapped
    N, dev = u.num_envs, u.device
    from app.reward_functions.acro_reward_v1 import zero_acro_clock
    saved_buckets = getattr(u, "_dist_buckets", None)
    u.set_target_distance_buckets(dist_buckets)   # None = uniform --distance-low/high
    obs = env.reset()[0]["policy"]
    bucket0 = getattr(u, "_bucket_id", None)
    bucket0 = bucket0.clone() if bucket0 is not None else None
    u._dist_buckets = saved_buckets               # the sampler only runs on reset; training keeps its curriculum
    zero_acro_clock(u)   # eval scores true first episodes: every clock back to 0
    speed0 = getattr(u, "_spawn_speed", torch.zeros(N, device=dev)).clone()   # handoff: the sampled takeover speed
    outcome = torch.zeros(N, dtype=torch.long, device=dev)
    hit_step = torch.zeros(N, device=dev)
    tilt_max = torch.zeros(N, device=dev)
    d_min = torch.full((N,), float("inf"), device=dev)     # closest approach per episode
    d_best = torch.full((N,), float("inf"), device=dev)    # distance at the closest SAMPLE (perp is read there)
    perp_at_min = torch.zeros(N, device=dev)               # perpendicular offset of that sample's velocity line
    d_last = torch.zeros(N, device=dev)
    speed_sum, speed_n = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    budget = torch.ceil(torch.tensor([steps_for_dist(float(d)) for d in u._start_dist.tolist()], device=dev)
                        / (u.step_dt * 240.0)).long()
    if getattr(u, "_runner", False):
        # a fleeing target has no per-distance budget: the episode ends by hit, crash, escape (gap > escape dist) or the episode cap
        budget = torch.full_like(budget, int(u.max_episode_length))
    steps = min(steps, int(budget.max().item())) if steps > 0 else int(budget.max().item())
    zero = torch.zeros(N, dtype=torch.long, device=dev)
    trace = []
    handoff = getattr(u, "_handoff_snap_enabled", False)
    if handoff:
        import isaaclab.utils.math as math_utils
        low_a, high_a = torch.as_tensor(u.single_action_space.low, device=dev), torch.as_tensor(u.single_action_space.high, device=dev)
        ho_rec = torch.zeros(N, dtype=torch.bool, device=dev)
        ho_dist, ho_h, ho_speed, ho_tilt, ho_steps = (torch.zeros(N, device=dev) for _ in range(5))

    for i in range(steps):
        if teacher is not None:   # teacher-alone baseline: is the converted label even a flyable target in this env?
            action = teacher.label(obs)[1]
        else:
            mean, std, _v = model.forward(obs)
            action = model.scale_action(mean + std * torch.randn_like(mean) if noisy else mean)
        if handoff:
            ctrl = ~u._handed_off
            action = torch.where(ctrl.unsqueeze(-1), controller_action(env, u, low_a, high_a), action)
            u._prefix_left.sub_(1).clamp_(min=0)
        nd, _r, term, trunc, extras = env.step(action)
        obs = nd["policy"]
        if handoff:
            u._refresh_handoff()
            js = torch.nonzero(ctrl & u._handed_off & ~(term | trunc) & ~ho_rec, as_tuple=False).squeeze(-1)
            if js.numel() > 0:                             # the real state at takeover, before the perturbation
                ho_dist[js] = torch.linalg.norm(u._desired_pos_w[js] - u._robot.data.root_pos_w[js], dim=-1)
                ho_h[js] = u._handoff_h[js]
                ho_speed[js] = torch.linalg.norm(u._robot.data.root_lin_vel_w[js], dim=-1)
                r_k, p_k, _y = math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w[js])
                ho_tilt[js] = torch.rad2deg(torch.maximum(r_k.abs(), p_k.abs()))
                ho_steps[js] = float(i + 1)
                ho_rec[js] = True
            js_all = torch.nonzero(ctrl & u._handed_off & ~(term | trunc), as_tuple=False).squeeze(-1)
            u._handoff_snap_obs(obs, js_all)
        alive = outcome == 0
        if teacher is not None and i % 30 == 0:   # a few teacher flights, printed by the caller like the angle trainer's trace
            import isaaclab.utils.math as math_utils
            for e in range(min(3, N)):
                if bool(alive[e]) and (not handoff or bool(u._handed_off[e])):
                    rel = u._desired_pos_w[e] - u._robot.data.root_pos_w[e]
                    vel = u._robot.data.root_lin_vel_w[e]
                    d = rel.norm().clamp_min(1e-6)
                    closing = (vel * rel).sum() / d
                    lateral = (vel - closing * rel / d).norm()
                    r_e, p_e, _ = math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w[e:e + 1])
                    trace.append((e, i, d.item(), vel.norm().item(), closing.item(), lateral.item(), -rel[2].item(),
                                  math.degrees(r_e.item()), math.degrees(p_e.item())))
        st = extras["acro_step"]   # pre-reset values, so the step on which an episode ends is counted too
        policy_flies = alive & (u._handed_off | (term | trunc)) if handoff else alive   # miss stats only after the takeover
        d_min = torch.where(policy_flies, torch.minimum(d_min, st["dist"]), d_min)
        closer = policy_flies & (st["dist"] < d_best)
        d_best = torch.where(closer, st["dist"], d_best)
        perp_at_min = torch.where(closer, st["perp"], perp_at_min)
        d_last = torch.where(alive, st["dist"], d_last)
        speed_sum += torch.where(alive, st["speed"], torch.zeros_like(d_last))
        speed_n += alive.float()
        tilt_max = torch.where(alive, torch.maximum(tilt_max, st["tilt"]), tilt_max)
        tr = extras["term_reasons"]
        code = torch.where(tr["hit"], zero + EVAL_CODES["hit"],
               torch.where(tr["hover"], zero + EVAL_CODES["hover"],
               torch.where(tr["oob"], zero + EVAL_CODES["oob"],
               torch.where(tr["attitude_tilt"], zero + EVAL_CODES["tilt"],
               torch.where(tr["near_miss"], zero + EVAL_CODES["near_miss"],
               torch.where(tr.get("escape", trunc & False), zero + EVAL_CODES["escape"],
               torch.where(trunc, zero + EVAL_CODES["timeout"], zero)))))))
        first = alive & (code > 0)
        hit_step = torch.where(first & (code == EVAL_CODES["hit"]), torch.full_like(hit_step, i + 1.0), hit_step)
        outcome = torch.where(first, code, outcome)
        out_of_time = (outcome == 0) & (i + 1 >= budget)
        outcome = torch.where(out_of_time, torch.full_like(outcome, EVAL_CODES["timeout"]), outcome)
        if bool((outcome > 0).all()):
            break

    res = {"hit": (outcome == EVAL_CODES["hit"]).float().mean().item()}
    for name, c in EVAL_CODES.items():
        if name != "hit":
            res[name] = (outcome == c).float().mean().item()
    res["unfinished"] = (outcome == 0).float().mean().item()
    if handoff and bool(ho_rec.any()):
        m = ho_rec
        derr = (ho_dist - ho_h)[m].cpu().numpy()
        serr = (ho_speed - speed0)[m].cpu().numpy()
        res["ho_n"] = float(m.sum().item())
        res["ho_dist_err_mean"] = float(derr.mean())
        res["ho_dist_err_p10"], res["ho_dist_err_p90"] = (float(x) for x in np.percentile(derr, [10, 90]))
        res["ho_speed_err_mean"] = float(serr.mean())
        res["ho_speed_err_p10"], res["ho_speed_err_p90"] = (float(x) for x in np.percentile(serr, [10, 90]))
        res["ho_speed_mean"] = ho_speed[m].mean().item()
        res["ho_tilt_mean"] = ho_tilt[m].mean().item()
        res["ho_off_frac"] = float((derr > 0.5).mean())       # handed off farther than h (passed the handoff point off-axis, or timer)
        res["ho_leg_s"] = float((ho_steps[m] * float(u.step_dt)).mean().item())
    cfg_h = getattr(u, "_handoff_cfg", None)
    if handoff and cfg_h is not None:       # hit / crash per takeover-speed bucket, as in the angle eval
        frac = speed0 / cfg_h["max_speed"]
        for bi, (b_lo, b_hi) in enumerate(SPEED_BUCKETS):
            m = (frac >= b_lo) & (frac < b_hi if bi < len(SPEED_BUCKETS) - 1 else frac <= b_hi)
            n_b = max(int(m.sum().item()), 1)
            res[f"hit_b{bi}"] = ((outcome == EVAL_CODES["hit"]) & m).sum().item() / n_b
            res[f"crash_b{bi}"] = (((outcome == EVAL_CODES["oob"]) | (outcome == EVAL_CODES["tilt"])) & m).sum().item() / n_b
            res[f"n_b{bi}"] = float(m.sum().item())
    if dist_buckets and bucket0 is not None and bool((bucket0 >= 0).any()):
        for bi in range(len(dist_buckets)):       # distance-curriculum eval pass: one row per spawn bucket
            m = bucket0 == bi
            n_b = max(int(m.sum().item()), 1)
            res[f"hit_d{bi}"] = ((outcome == EVAL_CODES["hit"]) & m).sum().item() / n_b
            res[f"crash_d{bi}"] = (((outcome == EVAL_CODES["oob"]) | (outcome == EVAL_CODES["tilt"])) & m).sum().item() / n_b
            res[f"n_d{bi}"] = float(m.sum().item())
    hit = outcome == EVAL_CODES["hit"]
    res["hit_time_s"] = (hit_step[hit].mean() * float(u.step_dt)).item() if bool(hit.any()) else NAN
    res["tilt_max_mean_deg"] = math.degrees(tilt_max.mean().item())
    res["over_limit_frac"] = (tilt_max > math.radians(tilt_limit_deg)).float().mean().item()
    res["mean_speed"] = (speed_sum / speed_n.clamp_min(1.0)).mean().item()
    if bool(hit.any()):
        res["hit_speed"] = (speed_sum[hit] / speed_n[hit].clamp_min(1.0)).mean().item()
    fin = d_min[torch.isfinite(d_min)]
    if fin.numel() > 0:
        res["min_dist_p10"], res["min_dist_p50"], res["min_dist_p90"] = (float(x) for x in np.percentile(fin.cpu().numpy(), [10, 50, 90]))
    if trace:
        res["trace"] = trace
    flown = torch.isfinite(d_best)
    if bool(flown.any()):
        # perpendicular distance from the target to the velocity line at the closest sample. It separates the two ways
        # of missing: large perp = flew past on a parallel line; small perp + large d_min = stopped short.
        pv = perp_at_min[flown].cpu().numpy()
        res["perp_miss_mean"] = float(pv.mean())
        res["perp_miss_p50"], res["perp_miss_p90"] = (float(x) for x in np.percentile(pv, [50, 90]))
    miss = ~hit
    if bool(miss.any()):   # what the non-hits looked like: near misses (small d_min) or never close
        res["miss_dmin_mean"] = d_min[miss & torch.isfinite(d_min)].mean().item()
        res["miss_dlast_mean"] = d_last[miss].mean().item()
        res["miss_speed_mean"] = (speed_sum[miss] / speed_n[miss].clamp_min(1.0)).mean().item()
    return res


# ===== Plot =====

def plot_summary(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(name):
        return np.array([float(r.get(name, NAN) if r.get(name, "") != "" else NAN) for r in rows])

    t = col("timesteps") / 1e6
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    ax = axs[0, 0]
    ax.plot(t, col("train_hit_rate"), label="train hit")
    ax.plot(t, col("success_rate"), label="eval hit (deterministic)")
    ax.set_ylim(-0.02, 1.02); ax.set_title("hit rate"); ax.legend(fontsize=8)
    ax = axs[0, 1]
    for k in ("train_timeout_rate", "train_hover_rate", "train_oob_rate", "train_tilt_rate", "train_escape_rate"):
        ax.plot(t, col(k), label=k.replace("train_", ""))
    ax.set_title("how training episodes end (besides hit)"); ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.plot(t, col("train_mean_tilt_deg"), label="mean tilt")
    ax.plot(t, col("train_max_tilt_deg"), label="max tilt")
    ax.plot(t, col("eval_tilt_max_mean_deg"), label="eval mean of per-episode max")
    ax.set_title("tilt [deg]"); ax.legend(fontsize=8)
    ax = axs[1, 1]
    for k in ("rew_progress", "rew_vclose", "rew_step", "rew_decel", "rew_tumble", "rew_tilt", "rew_away", "rew_los", "rew_speed", "rew_line", "rew_att", "rew_path", "rew_hit", "rew_crash", "rew_timeout", "rew_hover"):
        ax.plot(t, col(k), label=k.replace("rew_", ""))
    ax.set_title("mean reward per step by component"); ax.legend(fontsize=7, ncol=2)
    for ax in axs.flat:
        ax.set_xlabel("M env-steps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ===== Student =====

def load_student(args, env, teacher, device):
    from app.guidance.export_onnx import infer_architecture
    u = env.unwrapped
    obs_dim = u.single_observation_space["policy"].shape[0]
    a_dim = u.single_action_space.shape[0]
    low, high = u.single_action_space.low, u.single_action_space.high
    hidden, layers = 64, 4
    if args.bc_checkpoint_path and not os.path.exists(args.bc_checkpoint_path):
        raise SystemExit(f"--bc-checkpoint {args.bc_checkpoint_path!r} does not exist")   # never fall back silently
    src = args.bc_checkpoint_path or None
    from_teacher = False
    if src is None and isinstance(teacher, AcroTeacher):
        src = args.teacher_checkpoint      # same action space: start as an exact copy of the teacher (head kept)
    elif src is None and teacher is not None and args.teacher_checkpoint:
        src, from_teacher = args.teacher_checkpoint, True      # same 23 -> 4 network: its trunk is a better start than random
    if src:
        raw = torch.load(src, map_location="cpu")
        ck_obs, hidden, layers, ck_a = infer_architecture(raw)
        if from_teacher and ck_obs > obs_dim:   # e.g. a teacher trained with the zone timer: its trunk does not fit, start from scratch
            print(f"[rtl] teacher has {ck_obs} inputs, env {obs_dim}: not inheriting its trunk, student starts from scratch (hover head)")
            src, from_teacher, hidden, layers = None, False, 64, 4
        else:
            assert ck_obs <= obs_dim and ck_a == a_dim, f"checkpoint dims {(ck_obs, ck_a)} != env {(obs_dim, a_dim)}"
    model = RTLActorCritic(obs_dim, a_dim, low, high, hidden=hidden, num_hidden_layers=layers,
                           critic_hidden=args.critic_hidden, critic_layers=args.critic_layers,
                           log_std_min=args.log_std_min, log_std_max=args.log_std_max).to(device)
    if src:
        tmp = ActorCritic(obs_dim, a_dim, low, high, hidden=hidden, num_hidden_layers=layers, dropout=0.0)
        load_bc_checkpoint(tmp, src, map_location="cpu")
        missing, unexpected = model.load_state_dict(tmp.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("value_net.") for k in missing), (missing, unexpected)
        ck_low, ck_high = model.action_low.clone(), model.action_high.clone()
        env_low, env_high = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
        differs = not (torch.allclose(ck_low, env_low) and torch.allclose(ck_high, env_high))
        if differs and not from_teacher and not args.keep_ckpt_bounds:
            raise SystemExit(
                f"--bc-checkpoint {src!r} was trained with action bounds {ck_low.tolist()} .. {ck_high.tolist()} but this env uses "
                f"{env_low.tolist()} .. {env_high.tolist()}. Its weights only make sense with its own scale (an earlier bug saved the "
                f"angle-style +-0.5 rate bounds into acro checkpoints). Pass --keep-ckpt-bounds to continue it as it was trained, "
                f"or retrain from the teacher / from scratch to get the real rate range.")
        if not (differs and args.keep_ckpt_bounds):   # the teacher's angle-style bounds must never be used: always this env's
            with torch.no_grad():
                model.action_low.copy_(env_low)
                model.action_high.copy_(env_high)
        elif differs:
            print(f"[rtl] WARNING: keeping the checkpoint's action bounds {ck_low.tolist()} .. {ck_high.tolist()} (env: "
                  f"{env_low.tolist()} .. {env_high.tolist()}); the policy was trained with that scale")
        if from_teacher:
            # the teacher's head outputs angle-style torques in the angle action range; the same normalised numbers
            # mean different physical commands here. Keep the trunk features, restart the head as hover (BC fits it).
            with torch.no_grad():
                model.actor_mean.weight.zero_()
                model.actor_mean.bias.zero_()
        print(f"[rtl] student initialised from {'the TEACHER trunk (actor head zeroed = hover) ' if from_teacher else ''}{src} "
              f"(hidden={hidden}, layers={layers}); critic is fresh")
    else:
        with torch.no_grad():   # tanh(0) is the middle of every action range: thrust delta 0, rate 0 = hover
            model.actor_mean.weight.zero_()
            model.actor_mean.bias.zero_()
        print("[rtl] no checkpoint: student starts as a HOVER policy (actor head zero-initialised); "
              "pass --teacher-checkpoint or --bc-checkpoint to start from weights")
    return model, from_teacher, (obs_dim, a_dim, low, high, hidden, layers)


# ===== Main loop =====

def build_parser():
    p = argparse.ArgumentParser(description="Acro-style drone training in Isaac (DAgger-BC from an angle-style teacher + PPO)")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--total_timesteps", type=int, default=128_000_000, help="PPO samples (the BC stage is extra)")
    p.add_argument("--distance-low", type=float, default=3.0)
    p.add_argument("--distance-high", type=float, default=50.0)
    p.add_argument("--buckets", nargs="?", const=DEFAULT_BUCKETS, default=None, metavar="SPEC",
                   help="distance curriculum: draw each training episode's spawn distance from weighted ranges instead of "
                        f"U(--distance-low, --distance-high). Pass --buckets alone for the default '{DEFAULT_BUCKETS}', or give a "
                        "spec like '200-400:0.5,800-1200:0.5'. --distance-low/high stay the eval range. Also switches the time "
                        "bonus to a distance-dependent clock (budget = start_distance / 30 m/s, clamped to the leg bounds); an "
                        "explicit time_bonus_speed in --reward-json wins, and a fixed time_bonus_budget_s is replaced. "
                        "Ignored with --handoff.")
    p.add_argument("--checkpoint-dir", default="runs/acro_isaac")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-steps-per-chunk", type=int, default=256)
    p.add_argument("--spawn-speed-low", type=float, default=None, help="spawn already moving at U(low, high) m/s toward the target; omit both = from rest")
    p.add_argument("--spawn-speed-high", type=float, default=None)
    p.add_argument("--obs-position-mode", choices=("full", "height"), default="height")
    p.add_argument("--episode-s", type=float, default=None,
                   help="shorten training episodes to this many seconds (default: the env's 62.5 s). A policy that circles the "
                        "target for 20 s finishes almost no episodes, so the train stats rest on a few dozen of them")
    # tilt curriculum (the stage knob)
    p.add_argument("--tilt-limit", type=float, default=120.0, help="soft tilt limit L [deg] (angle between body-z and up); 180 = none")
    p.add_argument("--tilt-margin", type=float, default=30.0, help="hard crash at L + margin")
    p.add_argument("--hit-radius", "--hit-zone", type=float, default=None,
                   help="hit zone: the episode is a hit when the drone is within this many metres of the target (default: the reward's 0.25). "
                        "Same as --reward-json '{\"hit_threshold\": R}'; the JSON wins if both are given")
    p.add_argument("--zone-time", type=float, default=None,
                   help="seconds the drone must stay inside the hit radius (consecutive policy steps) before the hit counts; default 0 = the first "
                        "step inside. Same as --reward-json '{\"zone_time\": T}'; the JSON wins if both are given")
    p.add_argument("--reward", choices=("v1", "v2", "v3", "v4"), default="v1",
                   help="v1 = acro_reward_v1; v2 = v1 + the line-of-sight penalty of the angle rc reward (acro_reward_v2.py, los_* fields); "
                        "v3 = v2 + the speed-profile / cross-track / soft-attitude terms that pay for flying fast and straight (acro_reward_v3.py, "
                        "speed_*, line_*, attitude_* fields); v4 = v3 + the swept hit test, cross-track per METRE with a look-ahead, and the "
                        "terminal straightness bonus (acro_reward_v4.py, hit_swept, line_path_leg, line_lead_s, path_leg)")
    p.add_argument("--reward-json", default=None, help="path or inline JSON with AcroRewardConfig (v2: AcroRewardV2Config, v3: AcroRewardV3Config, v4: AcroRewardV4Config) overrides, e.g. '{\"c_tumble\": 2e-4}'")
    # weights
    p.add_argument("--teacher", choices=("auto", "los", "checkpoint", "acro", "none"), default="auto",
                   help="DAgger-BC / distillation teacher. los: the angle trainer's line-of-sight controller. checkpoint: the frozen angle-style "
                        "--teacher-checkpoint. acro: the frozen ACRO-style --teacher-checkpoint (e.g. the previous stage's best.pt), labels used as-is. "
                        "auto: checkpoint if --teacher-checkpoint is given, none when continuing from --bc-checkpoint, else los")
    p.add_argument("--teacher-checkpoint", default=None, help="frozen actor (.pt) used as DAgger-BC / distillation teacher: ANGLE-style for "
                                                              "--teacher checkpoint, ACRO-style for --teacher acro")
    p.add_argument("--los-stop-speed", type=float, default=0.5,
                   help="--teacher los: the speed it brakes down to at the target. 0.5 = arrive and stop; >= --max-speed = never brake")
    p.add_argument("--los-max-tilt-deg", type=float, default=35.0,
                   help="--teacher los: tilt cap of the LOS controller, 0 < cap <= 90 (its braking accel is 0.5*g*tan(cap) unless --teacher-accel)")
    p.add_argument("--teacher-check-only", action="store_true", help="fly the teacher alone through the handoff, print TEACHER CHECK, and exit")
    p.add_argument("--pid-eval-every", type=int, default=10, help="also fly the teacher alone (shown as pid= in the log line) every N chunks")
    p.add_argument("--bc-checkpoint", dest="bc_checkpoint_path", default=None,
                   help="start weights (e.g. the previous stage's best.pt). Without it, the student starts from the teacher's weights")
    p.add_argument("--bc-chunks", type=int, default=None, help="MAX DAgger-BC chunks before PPO (default 6 with a teacher when the student does not start from --bc-checkpoint, else 0)")
    p.add_argument("--bc-beta-start", type=float, default=0.5, help="fraction of envs flown by the teacher in the first BC chunk")
    p.add_argument("--bc-beta-chunks", type=int, default=4, help="beta decays linearly to 0 over this many chunks; after that the student flies alone and BC continues on its own mistakes")
    p.add_argument("--bc-target-hit", type=float, default=0.9, help="end BC early once the deterministic eval hit rate (beta = 0) reaches this")
    p.add_argument("--bc-eval-every", type=int, default=3)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--bc-steps", type=int, default=600, help="gradient steps per BC chunk, sampled from the aggregated buffer")
    p.add_argument("--bc-minibatch", type=int, default=8192)
    p.add_argument("--bc-buffer", type=int, default=10_000_000, help="max aggregated (obs, label) pairs")
    p.add_argument("--bc-per-chunk", type=int, default=250_000, help="pairs kept from each BC chunk")
    # phases
    p.add_argument("--settle-chunks", type=int, default=2)
    p.add_argument("--warmup-chunks", type=int, default=8, help="critic-only chunks, actor frozen")
    p.add_argument("--head-chunks", type=int, default=8)
    p.add_argument("--lr-ramp-chunks", type=int, default=3)
    # PPO
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
    p.add_argument("--lr-actor", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW decay, PPO and BC")
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--critic-layers", type=int, default=2)
    p.add_argument("--init-std", type=float, default=0.07, help="initial std in tanh-normalised action space (max is exp(log-std-max) = 0.135)")
    p.add_argument("--log-std-min", type=float, default=-3.5)
    p.add_argument("--log-std-max", type=float, default=-2.0)
    p.add_argument("--bootstrap-timeouts", action="store_true")
    # teacher distillation during PPO (only with --teacher-checkpoint)
    p.add_argument("--distill-start", type=float, default=0.05)
    p.add_argument("--distill-end", type=float, default=0.0)
    p.add_argument("--distill-decay-chunks", type=int, default=30)
    p.add_argument("--distill-max", type=float, default=5.0)
    p.add_argument("--guard-margin", type=float, default=0.05)
    p.add_argument("--guard-gain", type=float, default=2.0)
    p.add_argument("--guard-boost-max", type=float, default=20.0)
    p.add_argument("--guard-min-episodes", type=int, default=None)
    p.add_argument("--abort-hit-rate", type=float, default=0.0, help="abort if train hit rate stays below this (0 = never)")
    p.add_argument("--abort-window", type=int, default=6)
    # stage gate
    p.add_argument("--stop-hit", type=float, default=0.0, help="stop the run (saving best.pt) once eval hit >= this for --stop-patience evals in a row; 0 = off")
    p.add_argument("--stop-patience", type=int, default=2)
    p.add_argument("--success-window", type=int, default=30,
                   help="the logged success rate is taken over the most recent chunks holding at least max(this, 25%% of num_envs) "
                        "finished training episodes (a chunk that finished 12 episodes is never shown as a rate on its own)")
    # eval / output
    p.add_argument("--eval-steps", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-burnin", type=int, default=0, help="noisy steps run and discarded after each eval; not needed now that the env staggers episode clocks (0 = off)")
    p.add_argument("--handoff-speed", type=float, default=1.0,
                   help="staggered start (trainer's full resets only): each env starts at a random point on the spawn->target "
                        "line, already moving toward the target at U(0, this) m/s (0 = start at rest)")
    # two-stage episodes (same as angle_rtl_train_isaac.py): a scripted controller flies spawn -> handoff point, then the policy takes over
    p.add_argument("--handoff", action="store_true",
                   help="two-stage episodes like the angle trainer: the carrot-chase controller flies from the spawn point (spawn-dist "
                        "from the target) at --max-speed-bucket speeds down to the handoff point ~ U(--distance-low, --distance-high) "
                        "from the target; there the policy takes over, with a random sideways velocity and tilt. Needs "
                        "--spawn-dist-low/high. --spawn-speed-* and --handoff-speed are ignored.")
    p.add_argument("--max-speed", type=float, default=18.0, help="--handoff: 100 %% of the handoff speed distribution, m/s "
                                                              "(buckets 10-25 %% / 25-75 %% / 75-100 %% with probs .2/.4/.4)")
    p.add_argument("--spawn-dist-low", type=float, default=None, help="--handoff: spawn at least this far from the target")
    p.add_argument("--spawn-dist-high", type=float, default=None, help="--handoff: ... and at most this far")
    p.add_argument("--teacher-accel", type=float, default=None, help="--handoff: ramp accel (m/s^2) of the controller leg (default 0.5*g*tan(max tilt))")
    p.add_argument("--teacher-cruise-gain", type=float, default=1.0, help="--handoff: velocity-tracking gain of the controller leg")
    p.add_argument("--leg-max-elev-deg", type=float, default=15.0, help="--handoff: clamp the leg's elevation angle")
    p.add_argument("--no-recenter", dest="recenter", action="store_false", help="--handoff: do not shift the spawn so the handoff point is at x=y=0")
    p.add_argument("--lateral-frac", type=float, default=0.3, help="--handoff: sideways velocity up to this fraction of the speed")
    p.add_argument("--handoff-tilt-deg", type=float, default=15.0, help="--handoff: random roll/pitch at the takeover")
    p.add_argument("--keep-ckpt-bounds", action="store_true",
                   help="--bc-checkpoint: keep the action bounds stored in the checkpoint instead of this env's (needed to continue "
                        "acro_v4/v5-style checkpoints that were saved with the +-0.5 rate bug)")
    p.add_argument("--desync-steps", type=int, default=0,
                   help="after every full reset, force-reset each env once at a random step in this window (data discarded) so "
                        "episode starts are spread out and the train stats are not in waves; 0 = off. Should cover a typical "
                        "fast ending (a hit takes about 200 steps)")
    p.add_argument("--runner", action="store_true",
                   help="moving target: the target is a drone (same physics) that hovers on the target point until the handoff (from the "
                        "start without --handoff), then flees straight away from the drone at its speed cap. Adds 3 inputs (its velocity "
                        "relative to the drone); smaller checkpoints load with those columns at zero weight")
    p.add_argument("--runner-speed", type=float, default=8.0, help="--runner: max flee speed cap, m/s (keep it below --max-speed)")
    p.add_argument("--runner-speed-min", type=float, default=None, help="--runner: min flee speed cap, drawn U(min, max) per episode (default = max)")
    p.add_argument("--runner-escape-dist", type=float, default=100.0,
                   help="--runner: the drone LOSES (episode ends, -r_escape) once the runner is farther than this from its OWN spawn point "
                        "(the target point it hovered on until the policy took over), m. The drone's position does not matter")
    p.add_argument("--runner-min-alt", type=float, default=2.0, help="--runner: the runner never flees below this height, m")
    p.add_argument("--no-onnx", dest="onnx", action="store_false")
    return p


def run(env, args):
    import mlflow
    from app.guidance.export_onnx import export_onnx_model
    from app.guidance.mlflow_utils import log_metrics_safe, log_params_safe, start_run
    from app.reward_functions.acro_reward_v1 import AcroRewardConfig, install_acro_reward, pop_acro_stats
    from app.training.base_training_isaac import METRICS_JSON_PATH, _append_json_row

    run_start = datetime.now()
    u = env.unwrapped
    device, N = u.device, u.num_envs
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    overrides = {}
    if args.reward_json:
        overrides = json.load(open(args.reward_json)) if os.path.exists(args.reward_json) else json.loads(args.reward_json)
    if args.hit_radius is not None:
        if args.hit_radius <= 0:
            raise SystemExit("--hit-radius must be above 0 m")
        overrides.setdefault("hit_threshold", args.hit_radius)
    if args.zone_time is not None:
        if args.zone_time < 0:
            raise SystemExit("--zone-time must be 0 or more seconds")
        overrides.setdefault("zone_time", args.zone_time)
    overrides.setdefault("tilt_limit_deg", args.tilt_limit)
    overrides.setdefault("tilt_hard_margin_deg", args.tilt_margin)
    dist_buckets = args.dist_buckets
    if args.handoff and dist_buckets is not None:
        print("[rtl] --buckets ignored with --handoff: the handoff stage sets every episode's distance")
        dist_buckets = None
    if dist_buckets is not None and "time_bonus_speed" not in overrides:
        # A fixed clock saturates on the short buckets (a 150 m episode leaves ~max bonus no matter how it flies), so
        # the budget follows the episode's own distance. 30 m/s reproduces the v4 run's fixed 40 s at ~1200 m.
        # An explicit "time_bonus_speed": 0 in --reward-json keeps a fixed clock (set budget_s as well).
        if "time_bonus_budget_s" in overrides:
            print("[rtl] --buckets: replacing the fixed time_bonus_budget_s with a distance-dependent clock")
            overrides.pop("time_bonus_budget_s")
        overrides["time_bonus_speed"] = 30.0
    if args.reward == "v4":
        from app.reward_functions.acro_reward_v4 import AcroRewardV4Config, install_acro_reward_v4
        rcfg = install_acro_reward_v4(u, AcroRewardV4Config.from_overrides(overrides))
    elif args.reward == "v3":
        from app.reward_functions.acro_reward_v3 import AcroRewardV3Config, install_acro_reward_v3
        rcfg = install_acro_reward_v3(u, AcroRewardV3Config.from_overrides(overrides))
    elif args.reward == "v2":
        from app.reward_functions.acro_reward_v2 import AcroRewardV2Config, install_acro_reward_v2
        rcfg = install_acro_reward_v2(u, AcroRewardV2Config.from_overrides(overrides))
    else:
        rcfg = install_acro_reward(u, AcroRewardConfig.from_overrides(overrides))
    teacher_mode = args.teacher
    if teacher_mode == "auto":
        teacher_mode = "checkpoint" if args.teacher_checkpoint else ("none" if args.bc_checkpoint_path else "los")
    if teacher_mode in ("checkpoint", "acro") and not args.teacher_checkpoint:
        raise SystemExit(f"--teacher {teacher_mode} needs --teacher-checkpoint")
    gains_by_dist = None
    if args.handoff or teacher_mode == "los" or args.runner:
        with open("app/control/best_pid_gains_per_dist.json") as f:
            gains_by_dist = json.load(f)
    if args.runner:
        u.set_runner_gains(gains_by_dist, max_tilt=math.radians(args.los_max_tilt_deg))
        u.set_runner(args.runner_speed, min_speed=args.runner_speed_min, escape_dist=args.runner_escape_dist, min_alt=args.runner_min_alt)
        lo, hi = u._runner_speed
        print(f"[rtl] RUNNER: the target flees from the drone at U({lo:g}, {hi:g}) m/s once the policy takes over; escape (loss) when it is "
              f"{args.runner_escape_dist:g} m from its spawn point, hit at {rcfg.hit_threshold:g} m; +3 inputs (target velocity rel. to the drone)")
        if hi >= args.max_speed:
            print(f"[rtl] WARNING: runner speed {hi:g} m/s >= --max-speed {args.max_speed:g}: the LOS teacher (capped at max-speed) cannot close")
    if args.handoff:
        if args.spawn_dist_low is None or args.spawn_dist_high is None:
            raise SystemExit("--handoff needs --spawn-dist-low and --spawn-dist-high (how far from the target the controller starts)")
        # the env's own sampler is replaced per reset by the handoff wrapper (it moves the target so h ~ U(distance-low, -high))
        u.set_target_distance_range(args.distance_low, args.distance_high)
        u.set_spawn_speed_range(None, None)
        install_handoff_stage(u, max_speed=args.max_speed, prefix_low=0.0, prefix_high=0.0, lateral_frac=args.lateral_frac,
                              tilt_deg=args.handoff_tilt_deg, dist_low=args.distance_low, dist_high=args.distance_high,
                              spawn_low=args.spawn_dist_low, spawn_high=args.spawn_dist_high, gains_by_dist=gains_by_dist,
                              teacher_accel=args.teacher_accel, teacher_cruise_gain=args.teacher_cruise_gain,
                              leg_max_elev_deg=args.leg_max_elev_deg, recenter=args.recenter)
        print(f"[rtl] HANDOFF stage: carrot controller flies from {args.spawn_dist_low}-{args.spawn_dist_high} m out to a handoff point "
              f"~ U({args.distance_low}, {args.distance_high}) m from the target at speed buckets {SPEED_BUCKETS} x {args.max_speed} m/s "
              f"(probs {SPEED_BUCKET_PROBS}); then the policy takes over with sideways <= {args.lateral_frac:.0%} of the speed and tilt <= "
              f"{args.handoff_tilt_deg} deg")
    else:
        install_stagger(u, args.handoff_speed)

    if args.episode_s is not None:
        u.cfg.episode_length_s = args.episode_s
        if hasattr(u, "max_episode_length_s"):
            try:
                u.max_episode_length_s = args.episode_s
            except AttributeError:
                pass
        want = math.ceil(args.episode_s / u.step_dt)
        print(f"[rtl] episode length: asked {args.episode_s}s = {want} steps, env now reports {u.max_episode_length} steps"
              + ("" if int(u.max_episode_length) == want else "NOT APPLIED"))
    u.set_obs_position_mode(args.obs_position_mode)
    u.set_target_pairs(None)
    u.set_target_distance_range(args.distance_low, args.distance_high)
    u.set_target_distance_buckets(dist_buckets)
    if dist_buckets is not None:
        clock = (f"start_distance / {rcfg.time_bonus_speed:g} m/s, clamped to {rcfg.min_leg_s:g}-{rcfg.max_leg_s:g} s"
                 if getattr(rcfg, "time_bonus_speed", 0.0) > 0 else "off")
        print(f"[rtl] distance buckets: {describe_buckets(dist_buckets)} | eval stays on U({args.distance_low:g}, "
              f"{args.distance_high:g}) m | time bonus clock {clock}")
    if args.spawn_speed_low is not None and args.spawn_speed_high is not None:
        u.set_spawn_speed_range(args.spawn_speed_low, args.spawn_speed_high)
        print(f"[rtl] spawn: moving at U({args.spawn_speed_low}, {args.spawn_speed_high}) m/s toward the target")
    else:
        u.set_spawn_speed_range(None, None)
        print("[rtl] spawn: from rest")
    dist_txt = (f"distance buckets {describe_buckets(dist_buckets)}" if dist_buckets is not None
                else f"distance U({args.distance_low:g}, {args.distance_high:g}) m")
    print(f"[rtl] {dist_txt} | hit radius {rcfg.hit_threshold:g} m for {rcfg.zone_time:g} s | tilt limit {rcfg.tilt_limit_deg} deg "
          f"(hard cap {rcfg.tilt_limit_deg + rcfg.tilt_hard_margin_deg} deg) | obs position mode {args.obs_position_mode}")

    teacher = (LosTeacher(env, gains_by_dist, args) if teacher_mode == "los"
               else AngleTeacher(args.teacher_checkpoint, env, device) if teacher_mode == "checkpoint"
               else AcroTeacher(args.teacher_checkpoint, env, device) if teacher_mode == "acro" else None)
    model, from_teacher, (obs_dim, a_dim, low, high, hidden, layers) = load_student(args, env, teacher, device)
    chunk_steps = args.num_steps_per_chunk * N
    bc_chunks = args.bc_chunks if args.bc_chunks is not None else (6 if (teacher is not None and not args.bc_checkpoint_path) else 0)
    if bc_chunks > 0 and teacher is None:
        raise SystemExit("--bc-chunks needs a teacher (--teacher los, or --teacher-checkpoint)")

    ep_ret = torch.zeros(N, device=device)
    obs = env.reset()[0]["policy"]
    pop_acro_stats(u)

    # ---- teacher check (same lines as the angle trainer): fly the teacher itself through the handoff. If IT fails, no student can
    # learn from its labels; this is the ceiling for BC. Labels go through angle_to_acro, so it can differ from the angle teacher. ----
    if teacher is not None:
        tv = isaac_eval(env, model, args.eval_steps, rcfg.tilt_limit_deg, teacher=teacher)
        print(f"[rtl] TEACHER CHECK ({teacher_mode} flying the policy's part, n={N}): "
              + " ".join(f"{k}={tv.get(k, 0.0):.3f}" for k in ("hit", "tilt", "oob", "hover", "timeout", "near_miss", "escape", "unfinished"))
              + f" hit_time={tv['hit_time_s']:.1f}s")
        print("[rtl]   teacher flight (while handed off): "
              + " ".join(f"{k}={tv[k]:.2f}" for k in ("mean_speed", "min_dist_p10", "min_dist_p50", "min_dist_p90") if k in tv)
              + (f" ho_speed_mean={tv['ho_speed_mean']:.2f} ho_leg_s={tv['ho_leg_s']:.2f}" if "ho_speed_mean" in tv else ""))
        for e, i, d, sp, cl, la, dz, ro, pi_ in tv.get("trace", []):
            print(f"[rtl]   trace env{e} t={i * float(u.step_dt):6.1f}s dist={d:6.2f} speed={sp:5.2f} closing={cl:+6.2f} "
                  f"lateral={la:5.2f} above_target={dz:+6.2f} roll/pitch={ro:+5.1f}/{pi_:+5.1f}")
        if tv["hit"] < 0.5:
            print("[rtl] WARNING: the teacher itself hits < 50% -- distilling from it cannot work; fix the teacher first")
        if args.teacher_check_only:
            print("[rtl] --teacher-check-only: done")
            return
        obs = reset_and_desync(env, model, teacher, args, ep_ret)

    # ---- stage A: DAgger-BC (student acts, teacher relabels, all labelled states are kept) ----
    if bc_chunks > 0:
        bc_opt = torch.optim.AdamW(model.actor_params(), lr=args.bc_lr, weight_decay=args.weight_decay)
        apply_phase(model, "full")
        set_initial_std(model, args.init_std)
        buf = DaggerBuffer(args.bc_buffer)
        bc_best, bc_best_sd = -1.0, None
        for c in range(bc_chunks):
            beta = args.bc_beta_start * max(0.0, 1.0 - c / max(1, args.bc_beta_chunks))
            tb = time.time()
            batch, obs, _ = collect_rollout(env, model, teacher, obs, args.num_steps_per_chunk, args.gamma, False, ep_ret, beta=beta)
            buf.add(batch, args.bc_per_chunk)
            del batch
            loss, loss_dims = bc_train(model, bc_opt, buf, args.bc_steps, args.bc_minibatch)
            st = pop_acro_stats(u)
            note = ""
            if (c + 1) % max(1, args.bc_eval_every) == 0 or c == bc_chunks - 1:
                ev = isaac_eval(env, model, args.eval_steps, rcfg.tilt_limit_deg)
                note = (f" | EVAL hit={ev['hit']:.3f} hover={ev['hover']:.3f} oob={ev['oob']:.3f} timeout={ev['timeout']:.3f} near_miss={ev['near_miss']:.3f} escape={ev['escape']:.3f} unfinished={ev['unfinished']:.3f} "
                        f"tilt={ev['tilt']:.3f} miss: closest {ev.get('miss_dmin_mean', NAN):.2f} m, speed {ev.get('miss_speed_mean', NAN):.2f} m/s")
                if ev["hit"] > bc_best:           # keep the best student: a later chunk may regress
                    bc_best, bc_best_sd = ev["hit"], model.actor_state_dict()
                    torch.save(bc_best_sd, os.path.join(args.checkpoint_dir, "bc_best.pt"))
                    note += " (best, saved)"
                obs = reset_and_desync(env, model, teacher, args, ep_ret)
            print(f"[rtl] BC {c + 1}/{bc_chunks} beta={beta:.2f} buf={buf.n / 1e6:.1f}M loss={loss:.5f} "
                  f"[T,p,q,r {' '.join(f'{x:.5f}' for x in loss_dims)}] train_hit={st['train_hit_rate']:.3f} tilt_crash={st['train_tilt_rate']:.3f} "
                  f"oob={st['train_oob_rate']:.3f} hover={st['train_hover_rate']:.3f} timeout={st['train_timeout_rate']:.3f} "
                  f"ended/env={st['train_ended_per_env']:.2f} ({time.time() - tb:.0f}s){note}")
            if note and beta == 0.0 and bc_best >= args.bc_target_hit:
                print(f"[rtl] BC target reached (eval hit {bc_best:.3f} >= {args.bc_target_hit}) with the student flying alone")
                break
        if bc_best_sd is not None:
            model.load_state_dict(bc_best_sd, strict=False)   # actor only; the critic is untouched
            print(f"[rtl] BC done: using the best student (eval hit {bc_best:.3f}), saved as {args.checkpoint_dir}/bc_best.pt")
        del buf

    set_initial_std(model, args.init_std)
    # PPO starts from a fresh reset (lock-step envs): burn in with the noisy policy so the baseline / first chunks are not
    # made of fast-ending episodes only. Runs for BC and non-BC starts alike.
    obs = reset_and_desync(env, model, teacher, args, ep_ret)
    n_chunks = max(1, args.total_timesteps // chunk_steps)
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
    plain = ActorCritic(obs_dim, a_dim, low, high, hidden=hidden, num_hidden_layers=layers, dropout=0.0).to(device)

    dist_tag = f"bk{len(dist_buckets)}" if dist_buckets is not None else f"{args.distance_low:g}_{args.distance_high:g}"
    run_name = f"acro_isaac_{dist_tag}_L{rcfg.tilt_limit_deg:g}" + (f"_run{args.runner_speed:g}" if args.runner else "")
    run_key = f"{run_name}_{run_start.strftime('%Y%m%d_%H%M%S')}"
    start_run(experiment_name="acro-isaac", run_name=run_name)
    with open(os.path.join(args.checkpoint_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "acro_reward": rcfg.to_dict(), "arch": {"hidden": hidden, "layers": layers},
                   "chunk_steps": chunk_steps, "n_chunks": n_chunks, "bc_chunks": bc_chunks,
                   "run_start": run_start.isoformat(timespec="seconds")}, f, indent=2, default=str)
    log_params_safe({**vars(args), **{f"rew_{k}": v for k, v in rcfg.to_dict().items()}, "n_chunks": n_chunks})

    log = RunLog(os.path.join(args.checkpoint_dir, "metrics.csv"))
    success, baseline_hits, hit_hist, stop_streak = NAN, [], [], 0
    pid_success, pid_row = None, {}               # the teacher flying alone, printed as pid= like the angle trainer
    recent_eps = []   # (hits, finished episodes) per chunk, newest last
    best_ma, boost, best_score, best_info, aborted, stopped = -1.0, 1.0, -1e9, None, None, None
    actor_idx, group_idx, prev_phase = 0, {"trunk": 0, "head": 0, "critic": 0}, None
    t0 = time.time()
    print(f"[rtl] {n_chunks} PPO chunks x {chunk_steps} steps | settle {args.settle_chunks}, warmup {args.warmup_chunks}, "
          f"head {args.head_chunks}, then full | teacher {teacher_mode} | reward_acro_{args.reward} {rcfg.to_dict()}")

    for chunk in range(n_chunks):
        phase = phase_of(chunk, args.settle_chunks, args.warmup_chunks, args.head_chunks)
        if phase != prev_phase:
            print(f"[rtl] ---- chunk {chunk}: phase  {phase}")
            prev_phase = phase
        apply_phase(model, phase)
        train_actor, train_critic = phase in ("head_only", "full"), phase != "settle"
        active = {"trunk": phase == "full", "head": train_actor, "critic": train_critic}
        lrs = {}
        for g in optimizer.param_groups:
            nme, lr = g["name"], base_lr[g["name"]]
            if nme != "critic" and active[nme]:
                lr *= cosine_factor((chunk - first_trainable) / max(1, n_chunks - first_trainable), args.lr_min_ratio)
            if active[nme]:
                lr *= min(1.0, (group_idx[nme] + 1) / max(1, args.lr_ramp_chunks))
            g["lr"] = lr
            lrs[nme] = lr

        ds = distill_scheduled(actor_idx, args.distill_start, args.distill_end, args.distill_decay_chunks) if teacher else 0.0
        coef_eff = min(args.distill_max, ds * boost) if (train_actor and teacher) else 0.0

        tc = time.time()
        batch, obs, roll_stats = collect_rollout(env, model, teacher, obs, args.num_steps_per_chunk, args.gamma,
                                                 args.bootstrap_timeouts, ep_ret)
        t_roll = time.time() - tc
        batch["tw"] = torch.ones(batch["obs"].shape[:-1], device=device)
        train_stats = pop_acro_stats(u)
        agree = teacher_agreement(model, batch) if teacher else {}

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

        did_eval = chunk % max(1, args.eval_every) == 0 or chunk == n_chunks - 1
        eval_row = {}
        if did_eval:
            te = time.time()
            ev = isaac_eval(env, model, args.eval_steps, rcfg.tilt_limit_deg)
            success = ev["hit"]
            eval_row = {f"eval_{k}": v for k, v in ev.items()}
            eval_row["success_rate"] = success
            evn = isaac_eval(env, model, args.eval_steps, rcfg.tilt_limit_deg, noisy=True)
            eval_row.update({f"evalnoisy_{k}": v for k, v in evn.items()})
            if dist_buckets is not None:
                # extra deterministic pass on the training distribution: the aggregate eval above stays on
                # --distance-low/high, so this is what tells a long-range regression apart from a short-bucket gain
                evd = isaac_eval(env, model, args.eval_steps, rcfg.tilt_limit_deg, dist_buckets=dist_buckets)
                eval_row.update({f"eval_{k}": v for k, v in evd.items()
                                 if k.startswith(("hit_d", "crash_d", "n_d"))})
            if teacher is not None and (chunk % max(1, args.pid_eval_every) == 0 or not pid_row):
                pv = isaac_eval(env, model, args.eval_steps, rcfg.tilt_limit_deg, teacher=teacher)
                pid_success = pv["hit"]
                pv.pop("trace", None)
                pid_row = {f"pid_{k}": v for k, v in pv.items()}
                pid_row["pid_success_rate"] = pid_success
            obs = reset_and_desync(env, model, teacher, args, ep_ret)
            eval_row["t_eval_s"] = time.time() - te

        hit_rate = train_stats["train_hit_rate"]
        n_eps = train_stats["train_episodes"]
        recent_eps.append((0.0 if n_eps == 0 else hit_rate * n_eps, n_eps))
        win_hits = win_eps = 0.0
        need_eps = max(args.success_window, int(0.25 * N))   # a rate over 30 of 4096 envs is the easy episodes only
        for h_, e_ in reversed(recent_eps):
            win_hits, win_eps = win_hits + h_, win_eps + e_
            if win_eps >= need_eps:
                break
        succ_win = win_hits / win_eps if win_eps >= need_eps else NAN
        enough_eps = train_stats["train_episodes"] >= (args.guard_min_episodes if args.guard_min_episodes is not None
                                                         else max(50, int(0.04 * N)))
        if phase == "critic_warmup" and enough_eps:
            baseline_hits.append(hit_rate)
        baseline = float(np.mean(baseline_hits)) if baseline_hits else NAN
        guard_note = ""
        if did_eval and phase == "settle":
            best_ma = max(best_ma, success)
        if train_actor and did_eval and teacher is not None:
            if success < best_ma - args.guard_margin:
                boost = min(args.guard_boost_max, boost * args.guard_gain)
                guard_note = f" GUARD: eval {success:.3f} < best {best_ma:.3f}-{args.guard_margin} -> boost x{boost:.1f}"
            else:
                best_ma = max(best_ma, success)
                boost = max(1.0, boost * 0.9)
        if train_actor and enough_eps:
            hit_hist.append(hit_rate)
            if args.abort_hit_rate > 0 and len(hit_hist) >= args.abort_window \
                    and float(np.mean(hit_hist[-args.abort_window:])) < args.abort_hit_rate:
                aborted = f"train hit rate {np.mean(hit_hist[-args.abort_window:]):.3f} < {args.abort_hit_rate} over {args.abort_window} chunks"

        std = model.effective_std().detach().cpu().tolist()
        row = {
            "chunk": chunk, "phase": phase, "stage": phase, "timesteps": timesteps, "wall_s": time.time() - t0,
            "steps_per_s": chunk_steps / max(t_roll, 1e-9), "lr_trunk": lrs["trunk"], "lr_head": lrs["head"],
            "lr_critic": lrs["critic"], "teacher_weight_mean": batch["tw"].mean().item(), "distill_coef_sched": ds if train_actor else 0.0, "distill_boost": boost,
            "distill_coef_eff": coef_eff, "ent_coef": args.ent_coef if train_actor else 0.0,
            "tilt_limit_deg": rcfg.tilt_limit_deg, "success_window": succ_win, "success_window_eps": win_eps,
            **upd, **roll_stats, **agree, **train_stats, "baseline_train_hit_rate": baseline, "best_eval_hit": best_ma,
            "std_mean": float(np.mean(std)), **{f"std_d{i}": s for i, s in enumerate(std)}, **eval_row, **(pid_row if did_eval else {}),
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
            except Exception as e:
                print(f"[rtl] onnx export failed (continuing): {e}")

        # best = highest eval hit rate, ties broken toward faster hits
        ht = eval_row.get("eval_hit_time_s", NAN)
        score = success - 1e-4 * (ht if ht == ht else 0.0) if did_eval else NAN
        row["score"] = score
        if train_actor and did_eval and score > best_score:
            best_score = score
            best_info = {"checkpoint": ckpt, "chunk": chunk, "phase": phase, "score": score, "eval_hit_rate": success,
                         "eval_hit_time_s": ht, "tilt_limit_deg": rcfg.tilt_limit_deg, "timesteps": timesteps}
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
        if did_eval:   # how the deterministic eval episodes ended (every env's first episode, Isaac only); same layout as the angle trainer
            ends = " ".join("%s=%.3f" % (k, eval_row.get("eval_" + k, 0.0)) for k in
                            ("hit", "caught", "hover", "oob", "tilt", "timeout", "near_miss", "escape", "unfinished"))   # no chasers here: caught is always 0
            print(f"[rtl]   eval endings: {ends}" + (f" | crash by bucket "
                  + "/".join("%.2f" % eval_row["eval_crash_b%d" % i] for i in range(3)) if "eval_crash_b0" in eval_row else ""))
            if "eval_hit_d0" in eval_row:
                btxt = " | ".join(
                    f"d{i} hit {eval_row['eval_hit_d%d' % i]:.2f} crash {eval_row['eval_crash_d%d' % i]:.2f} "
                    f"(n={eval_row['eval_n_d%d' % i]:.0f})"
                    for i in range(len(dist_buckets)))
                print(f"[rtl]   distance buckets (training distribution): {btxt}")
            if "eval_ho_n" in eval_row:
                e = eval_row
                print(f"[rtl]   handoff check (n={e['eval_ho_n']:.0f}, before perturbation): "
                      f"dist actual-h mean {e['eval_ho_dist_err_mean']:+.2f} m (p10 {e['eval_ho_dist_err_p10']:+.2f}, "
                      f"p90 {e['eval_ho_dist_err_p90']:+.2f}) | speed actual-bucket mean {e['eval_ho_speed_err_mean']:+.2f} "
                      f"m/s (p10 {e['eval_ho_speed_err_p10']:+.2f}, p90 {e['eval_ho_speed_err_p90']:+.2f}) | "
                      f"speed {e['eval_ho_speed_mean']:.1f} m/s tilt {e['eval_ho_tilt_mean']:.1f} deg | "
                      f"leg {e['eval_ho_leg_s']:.1f} s | handed off >0.5 m farther than h: {e['eval_ho_off_frac']:.1%}")
            if "eval_perp_miss_mean" in eval_row:
                e = eval_row
                print(f"[rtl]   closest approach: perpendicular miss mean {e['eval_perp_miss_mean']:.2f} m "
                      f"(p50 {e['eval_perp_miss_p50']:.2f}, p90 {e['eval_perp_miss_p90']:.2f}) | "
                      f"hit-flight mean speed {e.get('eval_hit_speed', NAN):.1f} m/s")
        print(f"[rtl] {chunk + 1}/{n_chunks} {phase:13s} train_hit={hit_rate:.3f} (base {baseline:.3f}) "
              f"eval={success:.2f}{bucket_str} caught={0.0:.3f} pid={pid_txt} timeout={train_stats['train_timeout_rate']:.3f} "
              f"hover={train_stats['train_hover_rate']:.3f} "
              + (f"escape={train_stats['train_escape_rate']:.3f} " if args.runner else "")
              # cut = episodes that ran out of a shortened clock (not in train_hit's denominator); hit_all counts them as misses
              + f"cut={train_stats['train_cut_rate']:.3f} hit_all={train_stats['train_hit_all']:.3f} "
              + f"stall={train_stats['train_stall_frac']:.3f} agree={agree.get('agree_mae', NAN):.3f} "
              # v3: per-step speed-profile / line cost, plus the same two numbers in physical units (lateral metres off
              # the straight line, fraction of the target speed the drone is short of)
              + (f"spd={train_stats['rew_speed']:+.4f} line={train_stats['rew_line']:+.4f} "
                 f"lat={train_stats['train_mean_lateral_m']:.1f}m short={train_stats['train_speed_short_frac']:.2f} "
                 + (f"perp={train_stats['train_perp_miss_m']:.2f}m path={train_stats['rew_path']:+.3f} "
                    f"hitv={train_stats['train_hit_speed']:.1f} " if args.reward == "v4" else "")
                 if args.reward in ("v3", "v4") else "")
              + f"d_coef={coef_eff:.3f} kl={upd.get('approx_kl', NAN):.4f} vloss={upd.get('value_loss', NAN):.2f} "
              f"ev={upd.get('explained_var', NAN):.2f} {steps_str(row['steps_per_s'])}{guard_note}")

        if args.stop_hit > 0 and did_eval and train_actor:
            stop_streak = stop_streak + 1 if success >= args.stop_hit else 0
            if stop_streak >= args.stop_patience:
                stopped = f"eval hit >= {args.stop_hit} for {stop_streak} evals in a row at tilt limit {rcfg.tilt_limit_deg}"
        bad = [k for k in ("policy_loss", "value_loss") if upd.get(k) is not None and upd[k] == upd[k] and not math.isfinite(upd[k])]
        if bad:
            aborted = f"non-finite {bad}"
        if aborted:
            print(f"[rtl] ABORT at chunk {chunk}: {aborted}")
            break
        if stopped:
            print(f"[rtl] STAGE DONE at chunk {chunk}: {stopped}. Next stage: --bc-checkpoint {args.checkpoint_dir}/best.pt "
                  f"with a wider --tilt-limit")
            break

    summary = {"aborted": aborted, "stage_done": stopped, "best": best_info, "chunks_run": len(log.rows),
               "pid_success_rate": pid_success,
               "tilt_limit_deg": rcfg.tilt_limit_deg, "duration_s": time.time() - t0}
    with open(os.path.join(args.checkpoint_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    try:
        plot_summary(log.rows, os.path.join(args.checkpoint_dir, "summary.png"))
    except Exception as e:
        print(f"[rtl] summary plot failed: {e}")
    mlflow.set_tag("aborted", str(aborted))
    if best_info:
        mlflow.set_tag("best_checkpoint", best_info["checkpoint"])
    mlflow.log_artifact(log.csv_path)
    mlflow.end_run()
    print(f"[rtl] done. {json.dumps(summary, default=str)}")
    return model


def main():
    parser = build_parser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        args.dist_buckets = parse_buckets(args.buckets)   # fail before the Isaac app starts
    except ValueError as e:
        parser.error(str(e))
    sim_app = AppLauncher(args).app

    import gymnasium as gym
    import app.environmental.base_drone_env_isaac  # noqa: F401  (registers the gym ids)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("Isaac-Acro-Drone-Direct-v0", num_envs=args.num_envs)
    if args.runner:
        from app.environmental.base_drone_env_isaac import RUNNER_FEATURES
        env_cfg.runner = True
        env_cfg.runner_radius = args.hit_radius if args.hit_radius is not None else env_cfg.runner_radius   # what you see = the hit zone
        env_cfg.observation_space = env_cfg.observation_space + RUNNER_FEATURES
    env = gym.make("Isaac-Acro-Drone-Direct-v0", cfg=env_cfg)
    try:
        run(env, args)
    except BaseException:
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
