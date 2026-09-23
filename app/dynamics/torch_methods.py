"""
Batched torch mirror of app.dynamics.methods (the numpy oracle) -- mixer
inversion, motor lag, thrust/torque, quadratic drag, wind. Pure torch, NO
isaaclab/omni/pxr imports, so this module (and its diff-test) runs without
booting Isaac Sim -- see scripts/isaac_lab/diff_test_physics.py (migration step 4).

Not per-env: BaseDroneEnv builds one identical QuadConfig every episode (no
domain randomization currently -- mass_scale/wind sampling are present but
disabled in base_drone_env.py), so rotor geometry/coeffs are plain (4,)
tensors broadcast against (num_envs, 4) state, matching that behavior.

Frame convention matches the oracle exactly: torque and angular_velocity are
body-frame throughout (rotor positions are body-frame, inertia is diagonal in
body-frame); linear force/velocity are world-frame. Caller
(app/environmental/base_drone_env_isaac.py) is responsible for rotating into
whatever frame the RigidObject wrench API expects.
"""

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
    """Mirrors app.dynamics.drone.create_quad_rotors/create_quad_config exactly:
    4 rotors at 45/135/225/315deg, alternating spin_dir, k_f solved so
    hover_rpm_fraction*max_rpm balances gravity. dtype defaults to float32 (the
    env's dtype); diff_test_physics.py passes float64 for tight-tolerance checks."""
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
    """desired: (num_envs, 4) [thrust, roll, pitch, yaw] -> (num_envs, 4) rotor speeds (rad/s).
    Matches app.dynamics.methods.mixer_inversion: solves for w^2 via inv(M), clips
    negative w^2 to 0 (mirrors a torque/thrust combo the mixer can't realize), sqrt."""
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
    """w: (num_envs, 4) rotor speeds -> (num_envs, 3) body-frame torque.
    Matches app.dynamics.methods.net_combining_torque: cross([x,y,0],[0,0,F]) = [y*F,-x*F,0]
    (moment-arm contribution) + [0,0,k_m*w^2*spin_dir] (reaction torque), summed over rotors."""
    F = cfg.k_f * w**2  # (num_envs, 4)
    torque_x = (cfg.rotor_xy[:, 1] * F).sum(dim=-1)
    torque_y = (-cfg.rotor_xy[:, 0] * F).sum(dim=-1)
    torque_z = (cfg.k_m * w**2 * cfg.spin_dir).sum(dim=-1)
    return torch.stack([torque_x, torque_y, torque_z], dim=-1)


def drag_force(velocity: torch.Tensor, drag_coeff: float, cross_sec_area: float, air_dens: float) -> torch.Tensor:
    """velocity: (num_envs, 3) world-frame -> (num_envs, 3) drag force opposing it.
    Matches app.dynamics.methods.drag_force, including the <0.01 m/s zero-force floor."""
    speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
    f_drag_mag = 0.5 * air_dens * speed**2 * drag_coeff * cross_sec_area
    direction = torch.where(speed < 0.01, torch.zeros_like(velocity), -velocity / speed.clamp_min(1e-8))
    return direction * f_drag_mag


def wind_force(wind_vec: torch.Tensor, mass: float, k_wind_coeff: float) -> torch.Tensor:
    """wind_vec: (num_envs, 3) -> (num_envs, 3). Matches app.dynamics.methods.wind."""
    return wind_vec * mass * k_wind_coeff
