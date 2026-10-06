"""Batched torch mirror of app.dynamics.methods (the numpy oracle); no Isaac imports. See docs.md "Torch mirror"."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuadConfigTorch:
    mass: float
    inertia: torch.Tensor  # (3,) Ixx, Iyy, Izz
    rotor_xy: torch.Tensor  # (4, 2) rotor x,y position rel. to COM (z assumed 0, matches oracle)
    spin_dir: torch.Tensor  # (4,) +1/-1
    k_f: torch.Tensor  # (4,) thrust coeff
    k_m: torch.Tensor  # (4,) torque coeff
    motor_tau: torch.Tensor  # (4,) spin-up/down time constant (s)
    max_rpm: float
    drag_coeff: float
    mixer_inv: torch.Tensor  # (4, 4) precomputed inv(M), M as in app.dynamics.methods.mixer_inversion
    device: torch.device


def build_quad_config(
    mass: float,
    inertia: tuple[float, float, float],
    arm_length: float,
    drag_coeff: float,
    max_rpm: float,
    motor_tau: float,
    hover_rpm_fraction: float = 0.5,
    kf_km_ratio: float = 0.02,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> QuadConfigTorch:
    """Mirrors drone.create_quad_rotors/create_quad_config (float64 in diff tests, float32 in the env)."""
    angles_deg = torch.tensor([45.0, 135.0, 225.0, 315.0], device=device, dtype=dtype)
    spin_dir = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device, dtype=dtype)

    k_f_scalar = (mass * 9.81 / 4) / (max_rpm * hover_rpm_fraction) ** 2
    k_f = torch.full((4,), k_f_scalar, device=device, dtype=dtype)
    k_m = k_f * kf_km_ratio

    angles_rad = torch.deg2rad(angles_deg)
    rotor_xy = torch.stack(
        [arm_length * torch.cos(angles_rad), arm_length * torch.sin(angles_rad)], dim=-1
    )
    motor_tau_t = torch.full((4,), motor_tau, device=device, dtype=dtype)

    M = torch.zeros((4, 4), device=device, dtype=dtype)
    M[0, :] = k_f
    M[1, :] = rotor_xy[:, 1] * k_f
    M[2, :] = -rotor_xy[:, 0] * k_f
    M[3, :] = k_m * spin_dir
    mixer_inv = torch.linalg.inv(M)

    return QuadConfigTorch(
        mass=mass,
        inertia=torch.tensor(inertia, device=device, dtype=dtype),
        rotor_xy=rotor_xy,
        spin_dir=spin_dir,
        k_f=k_f,
        k_m=k_m,
        motor_tau=motor_tau_t,
        max_rpm=max_rpm,
        drag_coeff=drag_coeff,
        mixer_inv=mixer_inv,
        device=torch.device(device),
    )


def mixer_inversion(cfg: QuadConfigTorch, desired: torch.Tensor) -> torch.Tensor:
    """(N, 4) [thrust, roll, pitch, yaw] -> (N, 4) rotor speeds; unrealizable w^2 < 0 clipped to 0."""
    speeds_sq = desired @ cfg.mixer_inv.T
    speeds_sq = torch.clamp(speeds_sq, min=0.0)
    return torch.sqrt(speeds_sq)


def motor_lag(w_current: torch.Tensor, w_target: torch.Tensor, motor_tau: torch.Tensor, dt: float) -> torch.Tensor:
    """First-order lag toward w_target, per rotor. Matches app.dynamics.methods.motor_lag."""
    w_dot = (w_target - w_current) / motor_tau
    return w_current + w_dot * dt


def net_combining_thrust(cfg: QuadConfigTorch, w: torch.Tensor) -> torch.Tensor:
    """w: (num_envs, 4) rotor speeds -> (num_envs,) net body-z thrust."""
    return (cfg.k_f * w**2).sum(dim=-1)


def net_combining_torque(cfg: QuadConfigTorch, w: torch.Tensor) -> torch.Tensor:
    """(N, 4) rotor speeds -> (N, 3) body torque: moment arms [y*F, -x*F] + reaction k_m*w^2*spin_dir."""
    F = cfg.k_f * w**2  # (num_envs, 4)
    torque_x = (cfg.rotor_xy[:, 1] * F).sum(dim=-1)
    torque_y = (-cfg.rotor_xy[:, 0] * F).sum(dim=-1)
    torque_z = (cfg.k_m * w**2 * cfg.spin_dir).sum(dim=-1)
    return torch.stack([torque_x, torque_y, torque_z], dim=-1)


def drag_force(velocity: torch.Tensor, drag_coeff: float, cross_sec_area: float, air_dens: float) -> torch.Tensor:
    """(N, 3) world velocity -> (N, 3) opposing drag; zero below 0.01 m/s like the oracle."""
    speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
    f_drag_mag = 0.5 * air_dens * speed**2 * drag_coeff * cross_sec_area
    direction = torch.where(speed < 0.01, torch.zeros_like(velocity), -velocity / speed.clamp_min(1e-8))
    return direction * f_drag_mag


def wind_force(wind_vec: torch.Tensor, mass: float, k_wind_coeff: float) -> torch.Tensor:
    """wind_vec: (num_envs, 3) -> (num_envs, 3). Matches app.dynamics.methods.wind."""
    return wind_vec * mass * k_wind_coeff
