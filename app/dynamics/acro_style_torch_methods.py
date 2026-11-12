"""Batched torch mirror of app.dynamics.acro_style_methods (the numpy oracle); no Isaac imports. See docs.md "Drone styles"."""

from __future__ import annotations

import torch

from app.dynamics.acro_style_methods import RATE_ZETA, TORQUE_MARGIN, rate_kp_from_plant
from app.dynamics.angle_style_torch_methods import QuadConfigTorch, mixer_inversion


def hover_torque_authority(cfg: QuadConfigTorch) -> torch.Tensor:
    """(3,) matches app.dynamics.acro_style_methods.hover_torque_authority."""
    hover = torch.zeros(1, 4, device=cfg.device, dtype=cfg.k_f.dtype)
    hover[0, 0] = cfg.mass * 9.81
    w_sq = mixer_inversion(cfg, hover)[0] ** 2
    f = cfg.k_f * w_sq
    return torch.stack([
        (cfg.rotor_xy[:, 1].abs() * f).sum(),
        (cfg.rotor_xy[:, 0].abs() * f).sum(),
        (cfg.k_m * w_sq).sum(),
    ])


class TorchRateController:
    """Batched RateController; gains are shared by all envs."""

    def __init__(self, cfg: QuadConfigTorch, num_envs: int, kp=None, ki=0.0, integral_limit: float = 1.0,
                 torque_margin: float = TORQUE_MARGIN, gyro_ff: bool = False, zeta: float = RATE_ZETA):
        dtype = cfg.k_f.dtype
        self.inertia = cfg.inertia
        if kp is None:
            kp = rate_kp_from_plant(float(cfg.motor_tau[0]), zeta)
        self.kp = torch.as_tensor(kp, device=cfg.device, dtype=dtype).expand(3).clone()
        self.ki = torch.as_tensor(ki, device=cfg.device, dtype=dtype).expand(3).clone()
        self.integral_limit = integral_limit
        self.torque_limit = torque_margin * hover_torque_authority(cfg)
        self.gyro_ff = gyro_ff
        self.integral = torch.zeros(num_envs, 3, device=cfg.device, dtype=dtype)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.integral.zero_()
        else:
            self.integral[env_ids] = 0.0

    def compute_torque(self, ang_vel_b: torch.Tensor, rate_cmd: torch.Tensor, dt: float) -> torch.Tensor:
        """(N,3) body rates + (N,3) rate setpoints -> (N,3) body torque."""
        err = rate_cmd - ang_vel_b
        self.integral = torch.clamp(self.integral + err * dt, -self.integral_limit, self.integral_limit)

        tau = self.inertia * (self.kp * err + self.ki * self.integral)
        if self.gyro_ff:
            tau = tau + torch.cross(ang_vel_b, self.inertia * ang_vel_b, dim=-1)
        return torch.maximum(torch.minimum(tau, self.torque_limit), -self.torque_limit)
