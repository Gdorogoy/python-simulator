"""Tilt limit is a CONE, not a box: every controller (PID, PID teacher, carrot, LOS) must share app/control/tilt.py.
Regression for the old per-axis clamp: on a diagonal its corners reached sqrt(2) x tan(max_tilt) (~1.41x the tilt) and,
when only one axis saturated first, the commanded direction rotated off the line of sight (diagonal undershoot)."""
import math

import numpy as np
import torch

from app.control import pid
from app.control.tilt import tilt_cone_np, tilt_cone_torch
from app.control.torch_pid import TorchPIDController
from app.dynamics.drone import Quaternion, QuadState, Vector3D

MAX_TILT = 0.3
CAP = math.tan(MAX_TILT)


def test_numpy_and_torch_agree():
    rng = np.random.default_rng(0)
    r, p = rng.normal(size=64) * 2, rng.normal(size=64) * 2
    r_np, p_np = tilt_cone_np(r, p, MAX_TILT)
    r_t, p_t = tilt_cone_torch(torch.as_tensor(r), torch.as_tensor(p), MAX_TILT)
    assert np.allclose(r_np, r_t.numpy(), atol=1e-12) and np.allclose(p_np, p_t.numpy(), atol=1e-12)


def test_diagonal_saturates_the_cone_not_each_axis():
    # the old box clamp produced (cap, cap) -> |v| = sqrt(2) * cap, i.e. 1.41x the tilt cap
    assert np.hypot(*np.clip([2.0, 2.0], -CAP, CAP)) > 1.41 * CAP - 1e-9     # guards the regression: the box overshoots
    r, p = tilt_cone_np(2.0, 2.0, MAX_TILT)
    assert abs(np.hypot(r, p) - CAP) < 1e-12                    # cone: exactly at the cap
    assert abs(abs(r) - abs(p)) < 1e-12                         # direction preserved
    assert math.atan(np.hypot(r, p)) <= MAX_TILT + 1e-9


def test_off_diagonal_direction_is_preserved():
    # one axis saturates first; the old clamp rotated the command toward the minor axis
    raw_r, raw_p = -2.0, 0.5                                    # roll axis is the one over the cap
    r, p = tilt_cone_np(raw_r, raw_p, MAX_TILT)
    assert abs(np.hypot(r, p) - CAP) < 1e-12
    assert abs(r / p - raw_r / raw_p) < 1e-9                    # same heading, only the magnitude changed


def test_inside_the_cone_nothing_changes():
    r, p = tilt_cone_np(0.1, -0.05, MAX_TILT)
    assert abs(r - 0.1) < 1e-15 and abs(p + 0.05) < 1e-15


def _state():
    return QuadState(position=Vector3D(0.0, 0.0, 0.0), velocity=Vector3D(0.0, 0.0, 0.0),
                     orientation=Quaternion(0.0, 0.0, 0.0, 1.0), angular_velocity=Vector3D(0.0, 0.0, 0.0),
                     rotor_rpm=[6000.0, 6000.0, 6000.0, 6000.0])


def test_numpy_pid_uses_the_cone():
    # kp_pos = g makes the raw (roll, pitch) command exactly the position error, so a 45 deg target stays 45 deg
    c = pid.PIDController(kp_pos=9.81, kd_pos=0.0, kp_att=0.8, kd_att=0.2, kp_yaw=0.3, kd_yaw=0.2,
                          max_tilt_rad=MAX_TILT)
    action = c.compute_action(_state(), target_pos=np.array([10.0, 10.0, 0.0]))
    des_roll, des_pitch = action[1] / 0.8, action[2] / 0.8      # att_err = des - 0, no rate feedback at rest
    assert abs(np.hypot(des_roll, des_pitch) - CAP) < 1e-9      # at the cone, not 1.41x over it
    assert abs(des_roll / des_pitch + 1.0) < 1e-9               # 45 deg diagonal preserved


def test_torch_pid_uses_the_cone():
    # the torch teacher is the BC label source; it must clamp identically
    c = TorchPIDController(num_envs=1, device="cpu", kp_pos=9.81, kd_pos=0.0, kp_att=0.8, kd_att=0.2,
                           kp_yaw=0.3, kd_yaw=0.2, max_tilt_rad=MAX_TILT)
    z3 = torch.zeros(1, 3)
    action = c.compute_action(pos=z3, vel=z3, ang_vel=z3, roll=torch.zeros(1), pitch=torch.zeros(1),
                              yaw=torch.zeros(1), target_pos=torch.tensor([[10.0, 10.0, 0.0]]),
                              target_yaw=torch.zeros(1))
    des_roll, des_pitch = action[:, 1] / 0.8, action[:, 2] / 0.8
    assert abs(float(torch.hypot(des_roll, des_pitch)[0]) - CAP) < 1e-6
    assert abs(float(des_roll[0] / des_pitch[0]) + 1.0) < 1e-6
