"""The numpy LOS controller (viewer sim) must equal the torch one (Isaac env, LOS teacher): same maths, two implementations."""
import math

import numpy as np
import torch

from app.control import los, los_np


def _case(rng, n=64):
    return dict(
        v_des=rng.normal(size=(n, 3)) * 10, vel=rng.normal(size=(n, 3)) * 10, ang_vel=rng.normal(size=(n, 3)),
        roll=rng.uniform(-1.2, 1.2, n), pitch=rng.uniform(-1.2, 1.2, n), yaw=rng.uniform(-3.0, 3.0, n),
        yaw_des=rng.uniform(-3.0, 3.0, n))


def test_velocity_to_action_matches_torch():
    rng = np.random.default_rng(0)
    kw = dict(hover_thrust=14.715, kp_att=0.82, kd_att=0.22, kp_yaw=0.26, kd_yaw=0.18, max_tilt=0.6, cruise_gain=2.0)
    for _ in range(5):
        c = _case(rng)
        a_np, r_np, p_np = los_np.velocity_to_action(**c, **kw)
        t = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in c.items()}
        a_t, r_t, p_t = los.velocity_to_action(**t, **kw)
        assert np.allclose(a_np, a_t.numpy(), atol=1e-9)
        assert np.allclose(r_np, r_t.numpy(), atol=1e-12) and np.allclose(p_np, p_t.numpy(), atol=1e-12)


def test_tilt_clamp_is_isotropic():
    # a 45 deg diagonal command must saturate the TOTAL tilt cone (|roll, pitch| = tan(max_tilt)), not each axis:
    # the old per-axis clip let both axes reach max_tilt, i.e. a 40 deg tilt out of a 35 deg cap.
    z = np.zeros((1, 3))
    _, r, p = los_np.velocity_to_action(np.array([[30.0, 30.0, 0.0]]), z, z, np.zeros(1), np.zeros(1), np.zeros(1),
                                        np.zeros(1), hover_thrust=14.715, kp_att=0.82, kd_att=0.22, kp_yaw=0.26,
                                        kd_yaw=0.18, max_tilt=0.6, cruise_gain=1.0)
    assert np.hypot(r[0], p[0]) <= math.tan(0.6) + 1e-9
    assert np.hypot(r[0], p[0]) > 0.99 * math.tan(0.6)          # still saturated, not under-driven
    assert abs(abs(r[0]) - abs(p[0])) < 1e-9                    # the diagonal direction is preserved
    # below the cap nothing is scaled
    _, r2, p2 = los_np.velocity_to_action(np.array([[1.0, 0.0, 0.0]]), z, z, np.zeros(1), np.zeros(1), np.zeros(1),
                                          np.zeros(1), hover_thrust=14.715, kp_att=0.82, kd_att=0.22, kp_yaw=0.26,
                                          kd_yaw=0.18, max_tilt=0.6, cruise_gain=1.0)
    assert abs(p2[0] - 1.0 / 9.81) < 1e-12 and r2[0] == 0.0


def test_chase_and_hold_match_torch():
    rng = np.random.default_rng(1)
    rel = rng.normal(size=(32, 3)) * 20
    cap = rng.uniform(1, 15, size=(32, 1))
    assert np.allclose(los_np.chase_velocity(rel, cap), los.chase_velocity(torch.as_tensor(rel), torch.as_tensor(cap)).numpy())
    assert np.allclose(los_np.hold_velocity(rel), los.hold_velocity(torch.as_tensor(rel)).numpy())
    assert np.allclose(np.linalg.norm(los_np.chase_velocity(rel, cap), axis=1), cap[:, 0])     # chase speed is exactly the cap
