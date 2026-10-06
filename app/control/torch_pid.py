"""Batched torch mirror of PIDController with per-env gains; no Isaac imports. See docs.md "PID"."""

from __future__ import annotations

import math

import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Matches app.control.pid.wrap_angle exactly."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class TorchPIDController:
    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        kp_pos, kd_pos, kp_att, kd_att, kp_yaw, kd_yaw,
        max_tilt_rad: float = 0.3,
        ki_pos=0.0, ki_att=0.0, ki_yaw=0.0,
        integral_limit_pos: float = 0.3, integral_limit_att: float = 0.3,
        action_low=(-1.5 * 9.81, -0.5, -0.5, -0.5),
        action_high=(1.5 * 9.81, 0.5, 0.5, 0.5),
    ):
        """Every gain arg accepts either a python scalar (broadcast to all envs)
        or a (num_envs,) tensor/array (per-env gains) -- see assign_gains_by_distance."""
        self.num_envs = num_envs
        self.device = device

        def _bcast(v):
            return torch.as_tensor(v, dtype=torch.float32, device=device).expand(num_envs).clone()

        self.kp_pos = _bcast(kp_pos)
        self.kd_pos = _bcast(kd_pos)
        self.ki_pos = _bcast(ki_pos)
        self.kp_att = _bcast(kp_att)
        self.kd_att = _bcast(kd_att)
        self.ki_att = _bcast(ki_att)
        self.kp_yaw = _bcast(kp_yaw)
        self.kd_yaw = _bcast(kd_yaw)
        self.ki_yaw = _bcast(ki_yaw)
        self.max_tilt_rad = max_tilt_rad
        self.integral_limit_pos = integral_limit_pos
        self.integral_limit_att = integral_limit_att

        self.integral_pos = torch.zeros(num_envs, 3, device=device)
        self.integral_att = torch.zeros(num_envs, 2, device=device)
        self.integral_yaw = torch.zeros(num_envs, device=device)

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.integral_pos.zero_()
            self.integral_att.zero_()
            self.integral_yaw.zero_()
        else:
            self.integral_pos[env_ids] = 0.0
            self.integral_att[env_ids] = 0.0
            self.integral_yaw[env_ids] = 0.0

    def set_gains(self, env_ids: torch.Tensor, gains: dict):
        """Overwrite gains for env_ids only; keys match PIDController kwargs."""
        for key in ("kp_pos", "kd_pos", "ki_pos", "kp_att", "kd_att", "ki_att", "kp_yaw", "kd_yaw", "ki_yaw"):
            if key in gains:
                getattr(self, key)[env_ids] = float(gains[key])

    def compute_action(
        self,
        pos: torch.Tensor, vel: torch.Tensor, ang_vel: torch.Tensor,
        roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor,
        target_pos: torch.Tensor, target_yaw: torch.Tensor, dt: float = 1 / 240,
    ) -> torch.Tensor:
        """(N,3) pos/vel (world), ang_vel (body), (N,) roll/pitch/yaw -> (N,4) action, same as PIDController."""
        pos_err = target_pos - pos

        self.integral_pos = torch.clamp(
            self.integral_pos + pos_err * dt, -self.integral_limit_pos, self.integral_limit_pos)

        accel_cmd = (self.kp_pos.unsqueeze(-1) * pos_err - self.kd_pos.unsqueeze(-1) * vel
                     + self.ki_pos.unsqueeze(-1) * self.integral_pos)

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        accel_body_x = accel_cmd[:, 0] * cos_yaw + accel_cmd[:, 1] * sin_yaw
        accel_body_y = -accel_cmd[:, 0] * sin_yaw + accel_cmd[:, 1] * cos_yaw

        des_roll = torch.clamp(-accel_body_y / 9.81, -self.max_tilt_rad, self.max_tilt_rad)
        des_pitch = torch.clamp(accel_body_x / 9.81, -self.max_tilt_rad, self.max_tilt_rad)

        att_err_roll = des_roll - roll
        att_err_pitch = des_pitch - pitch
        att_err = torch.stack([att_err_roll, att_err_pitch], dim=-1)
        self.integral_att = torch.clamp(
            self.integral_att + att_err * dt, -self.integral_limit_att, self.integral_limit_att)

        yaw_err = wrap_angle(target_yaw - yaw)
        self.integral_yaw = torch.clamp(
            self.integral_yaw + yaw_err * dt, -self.integral_limit_att, self.integral_limit_att)

        roll_torque = self.kp_att * att_err[:, 0] - self.kd_att * ang_vel[:, 0] + self.ki_att * self.integral_att[:, 0]
        pitch_torque = self.kp_att * att_err[:, 1] - self.kd_att * ang_vel[:, 1] + self.ki_att * self.integral_att[:, 1]
        yaw_torque = self.kp_yaw * yaw_err - self.kd_yaw * ang_vel[:, 2] + self.ki_yaw * self.integral_yaw
        thrust_delta = self.kp_pos * pos_err[:, 2] - self.kd_pos * vel[:, 2] + self.ki_pos * self.integral_pos[:, 2]

        action = torch.stack([thrust_delta, roll_torque, pitch_torque, yaw_torque], dim=-1)
        return torch.clamp(action, self.action_low, self.action_high)


def assign_gains_by_distance(pid: TorchPIDController, start_dist: torch.Tensor, gains_by_dist: dict,
                              env_ids: torch.Tensor | None = None):
    """Per env, pick the gain set whose distance key is nearest its start_dist; call after every reset."""
    if env_ids is None:
        env_ids = torch.arange(pid.num_envs, device=pid.device)
    keys = list(gains_by_dist.keys())
    key_dists = torch.tensor([float(k) for k in keys], device=pid.device)
    dists = start_dist[env_ids].unsqueeze(-1)  # (n,1)
    nearest_idx = torch.argmin((dists - key_dists.unsqueeze(0)).abs(), dim=-1)  # (n,)
    for i, key_idx in enumerate(nearest_idx.tolist()):
        pid.set_gains(env_ids[i:i + 1], gains_by_dist[keys[key_idx]])
