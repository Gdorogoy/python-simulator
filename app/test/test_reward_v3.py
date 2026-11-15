"""Acro reward v3 (app/reward_functions/acro_reward_v3.py): the speed profile, the cross-track line and the soft attitude
wall. The point of every test is "one leg pays exactly the `*_leg` number, and 0 means off", the same contract as
proportion.py. Plain asserts, like the rest of app/test (no pytest needed)."""

import math

import torch

from app.reward_functions.acro_reward_v3 import (AcroRewardV3Config, compute_attitude_term, compute_line_term,
                                                 compute_speed_track_term, speed_profile)


def _close(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


def _cfg(**kw):
    return AcroRewardV3Config.from_overrides(kw)


# ---- the target-speed profile ------------------------------------------------------------------------------------

def test_profile_is_brake_limited_and_capped_at_the_target():
    cfg = _cfg(speed_target=50.0, speed_accel=16.0)
    assert speed_profile(torch.tensor([0.0]), cfg).item() == 0.0
    _close(speed_profile(torch.tensor([50.0]), cfg).item(), math.sqrt(2 * 16 * 50))     # 40 m/s, still accelerating
    _close(speed_profile(torch.tensor([100.0]), cfg).item(), 50.0)                     # sqrt(3200) = 56.6 -> capped
    _close(speed_profile(torch.tensor([1000.0]), cfg).item(), 50.0)                    # far away: cruise at the cap


def test_profile_defaults_are_physically_flyable():
    cfg = AcroRewardV3Config()
    assert cfg.speed_target <= 52.0                    # 52 m/s is the airframe limit at ~70 deg tilt, full thrust
    assert speed_profile(torch.tensor([5.0]), cfg).item() < 15.0                      # it brakes near the target


def test_profile_can_brake_to_a_crossing_speed_not_a_stop():
    # speed_end_speed: the hit sphere is a fly-through, so braking to 0 at the target left the last ~80 m unpoliced
    cfg = _cfg(speed_target=50.0, speed_accel=16.0, speed_end_speed=20.0)
    _close(speed_profile(torch.tensor([0.0]), cfg).item(), 20.0)                      # 20 m/s AT the target
    _close(speed_profile(torch.tensor([10.0]), cfg).item(), math.sqrt(400 + 2 * 16 * 10))
    _close(speed_profile(torch.tensor([66.0]), cfg).item(), 50.0)                     # (50^2 - 20^2) / (2 * 16) = 65.6 m
    _close(speed_profile(torch.tensor([500.0]), cfg).item(), 50.0)
    # end speed above the cruise target cannot make the profile exceed the target
    _close(speed_profile(torch.tensor([0.0]), _cfg(speed_end_speed=80.0)).item(), 50.0)


# ---- the speed term ----------------------------------------------------------------------------------------------

def _speed(vel, rel, dist=None, active=True, n_steps=100, **kw):
    v = torch.tensor(vel, dtype=torch.float32).reshape(1, 3)
    r = torch.tensor(rel, dtype=torch.float32).reshape(1, 3)
    d = torch.linalg.norm(r, dim=-1) if dist is None else torch.tensor([dist])
    return compute_speed_track_term(vel=v, rel=r, dist=d, active=torch.tensor([active]), cfg=_cfg(**kw),
                                    n_steps=n_steps).item()


def test_speed_charges_the_full_leg_when_standing_still():
    # far away, target = 50, v_toward = 0 -> the whole per-step coefficient, i.e. speed_leg over one leg
    _close(_speed([0, 0, 0], [1000, 0, 0], speed_leg=25.0), -25.0 / 100)
    # and N such steps total exactly speed_leg
    total = sum(_speed([0, 0, 0], [1000, 0, 0], speed_leg=25.0) for _ in range(100))
    _close(total, -25.0, tol=1e-4)


def test_speed_is_paid_off_by_flying_at_the_target():
    assert _speed([50, 0, 0], [1000, 0, 0], speed_leg=25.0) == 0.0
    assert _speed([60, 0, 0], [1000, 0, 0], speed_leg=25.0) == 0.0          # faster than the cap is not punished
    half = _speed([25, 0, 0], [1000, 0, 0], speed_leg=25.0)
    _close(half, -12.5 / 100)                                               # half the target: half the charge


def test_speed_only_counts_the_line_of_sight_component():
    # 50 m/s straight sideways earns nothing toward the target: same charge as standing still
    _close(_speed([0, 50, 0], [1000, 0, 0], speed_leg=25.0), -25.0 / 100)
    # flying away is a bigger shortfall than standing still, but it clamps at one leg's worth
    _close(_speed([-50, 0, 0], [1000, 0, 0], speed_leg=25.0), -25.0 / 100)


def test_speed_brakes_near_the_target():
    # 20 m out the profile is sqrt(2*16*20) = 25.3 m/s, so 20 m/s is nearly on target (unlike the 50 it needs far away)
    near = _speed([20, 0, 0], [20, 0, 0], speed_leg=25.0)
    far = _speed([20, 0, 0], [1000, 0, 0], speed_leg=25.0)
    assert near > far and near < 0.0


def test_speed_end_speed_charges_the_final_creep():
    # the term only punishes being SLOWER than the profile, so with end speed 0 the last ~80 m is unpoliced and the
    # policy is free to creep in at 5 m/s. end speed 20 makes that creep cost something: it is paid where it happens.
    creep_end0 = _speed([5, 0, 0], [1, 0, 0], speed_leg=25.0)
    creep_end20 = _speed([5, 0, 0], [1, 0, 0], speed_leg=25.0, speed_end_speed=20.0)
    assert creep_end0 > -0.01 and creep_end20 < -0.05 and creep_end20 < 5 * creep_end0
    # crossing the target at the end speed (or faster) is essentially free: sqrt(20^2 + 2*16*1) = 20.8 at 1 m
    assert -0.005 < _speed([20, 0, 0], [1, 0, 0], speed_leg=25.0, speed_end_speed=20.0) < 0.0
    assert _speed([30, 0, 0], [1, 0, 0], speed_leg=25.0, speed_end_speed=20.0) == 0.0


def test_speed_gate_and_off_switch():
    assert _speed([0, 0, 0], [5, 0, 0], speed_leg=25.0, speed_min_dist=10.0) == 0.0
    assert _speed([0, 0, 0], [1000, 0, 0], speed_leg=0.0) == 0.0             # 0 = off
    assert _speed([0, 0, 0], [1000, 0, 0], speed_leg=25.0, active=False) == 0.0


def test_speed_is_per_env_and_one_dimensional():
    cfg = _cfg(speed_leg=25.0)
    vel = torch.tensor([[50.0, 0, 0], [0.0, 0, 0]], dtype=torch.float32)
    rel = torch.tensor([[1000.0, 0, 0], [1000.0, 0, 0]], dtype=torch.float32)
    out = compute_speed_track_term(vel=vel, rel=rel, dist=torch.linalg.norm(rel, dim=-1),
                                   active=torch.tensor([True, True]), cfg=cfg, n_steps=100)
    assert out.shape == (2,)
    assert out[0].item() == 0.0
    _close(out[1].item(), -0.25)


# ---- the cross-track (straightness) term -------------------------------------------------------------------------

def _line(rel, rel0, active=True, n_steps=100, **kw):
    return compute_line_term(rel=torch.tensor(rel, dtype=torch.float32).reshape(1, 3),
                             rel0=torch.tensor(rel0, dtype=torch.float32).reshape(1, 3),
                             active=torch.tensor([active]), cfg=_cfg(**kw), n_steps=n_steps).item()


def test_line_is_off_by_default_and_free_on_the_line():
    assert _line([900, 50, 0], [1000, 0, 0]) == 0.0                          # line_leg defaults to 0
    assert _line([900, 0, 0], [1000, 0, 0], line_leg=8.0) == 0.0             # perfectly on the line
    assert _line([900, 0, 0], [1000, 0, 0], line_leg=8.0, active=False) == 0.0


def test_line_charges_cross_track_and_saturates_at_line_ref():
    # 50 m off a 1000 m leg with line_ref 50: the whole per-step charge
    _close(_line([900, 50, 0], [1000, 0, 0], line_leg=8.0), -8.0 / 100)
    _close(_line([900, 250, 0], [1000, 0, 0], line_leg=8.0), -8.0 / 100)     # clamped at 1
    _close(_line([900, 25, 0], [1000, 0, 0], line_leg=8.0), -4.0 / 100)            # half way: half the charge


def test_line_counts_only_the_perpendicular_offset():
    # overshooting along the line is not cross-track; it is the distance terms' job
    assert _line([-200, 0, 0], [1000, 0, 0], line_leg=8.0) == 0.0
    # and the term is per env / 1-D
    cfg = _cfg(line_leg=8.0)
    rel = torch.tensor([[900.0, 0, 0], [900.0, 50, 0]], dtype=torch.float32)
    rel0 = torch.tensor([[1000.0, 0, 0], [1000.0, 0, 0]], dtype=torch.float32)
    out = compute_line_term(rel=rel, rel0=rel0, active=torch.tensor([True, True]), cfg=cfg, n_steps=100)
    assert out.shape == (2,)
    assert out[0].item() == 0.0 and abs(out[1].item() + 0.08) < 1e-6


# ---- the soft attitude wall --------------------------------------------------------------------------------------

def _att(deg, n_steps=100, **kw):
    return compute_attitude_term(tilt=torch.tensor([math.radians(deg)]), cfg=_cfg(**kw), n_steps=n_steps).item()


def test_attitude_is_off_by_default_and_below_the_soft_angle():
    assert _att(100.0) == 0.0                                                # attitude_soft_deg defaults to 0 -> off
    assert _att(100.0, attitude_soft_deg=85.0) == 0.0                         # attitude_leg defaults to 0 -> off
    assert _att(80.0, attitude_soft_deg=85.0, attitude_leg=8.0) == 0.0        # below the wall


def test_attitude_ramps_from_soft_to_the_hard_cap():
    # hard cap = tilt_limit_deg + tilt_hard_margin_deg = 80 + 45 = 125 deg at the run's settings
    kw = dict(attitude_soft_deg=85.0, attitude_leg=8.0, tilt_limit_deg=80.0, tilt_hard_margin_deg=45.0)
    assert _att(85.0, **kw) == 0.0
    _close(_att(125.0, **kw), -8.0 / 100)                                     # full charge at the crash angle
    _close(_att(105.0, **kw), -4.0 / 100)                                     # half way: half the charge
    _close(_att(180.0, **kw), -8.0 / 100)                                     # past it: clamped, not unbounded
    # 0 tilt_limit_deg = no limit, so the soft wall has nothing to ramp to and must be a no-op, not a divide by zero
    flat = dict(attitude_soft_deg=85.0, attitude_leg=8.0, tilt_limit_deg=180.0, tilt_hard_margin_deg=0.0)
    assert math.isfinite(_att(90.0, **flat))


# ---- config ------------------------------------------------------------------------------------------------------

def test_overrides_accept_every_new_field():
    cfg = _cfg(speed_target=48.0, speed_accel=14.0, speed_leg=20.0, speed_min_dist=2.0,
               line_leg=6.0, line_ref=40.0, attitude_soft_deg=80.0, attitude_leg=5.0,
               time_bonus_budget_s=62.5, time_bonus_speed=30.0)
    assert (cfg.speed_target, cfg.speed_leg, cfg.line_leg, cfg.attitude_soft_deg) == (48.0, 20.0, 6.0, 80.0)
    assert cfg.time_bonus_budget_s == 62.5                      # from proportion.py: restores the time bonus
    assert cfg.time_bonus_speed == 30.0                         # ... and the distance-dependent version


def test_unknown_override_is_refused():
    try:
        _cfg(speed_taarget=50.0)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


# ---- installer: the glue that would only fail inside Isaac ------------------------------------------------------

def _fake_env(n=2, vel=None):
    """The handful of attributes the v1/v2/v3 wrappers touch, so the installer can be exercised without Isaac."""
    import types

    u = types.SimpleNamespace()
    u.device = torch.device("cpu")
    u.num_envs, u.step_dt, u.max_episode_length = n, 1.0 / 60.0, 3750
    u.episode_length_buf = torch.zeros(n, dtype=torch.long)
    data = types.SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.zeros(n, 4),
                                 root_lin_vel_w=torch.zeros(n, 3) if vel is None else torch.tensor(vel, dtype=torch.float32),
                                 root_ang_vel_b=torch.zeros(n, 3))
    data.root_quat_w[:, 0] = 1.0
    u._robot = types.SimpleNamespace(data=data)
    u._terrain = types.SimpleNamespace(env_origins=torch.zeros(n, 3))
    u._desired_pos_w = torch.tensor([[1000.0, 0, 0], [1000.0, 0, 0]][:n], dtype=torch.float32)
    u._start_dist = torch.full((n,), 1000.0)
    u._prev_position = torch.zeros(n, 3)
    u._prev_distance = torch.full((n,), 1000.0)
    u._prev_target_w = u._desired_pos_w.clone()
    u._cached_reward = torch.zeros(n)
    u._episode_sums = {"reward": torch.zeros(n), "distance_to_goal": torch.zeros(n)}
    u.extras = {}
    u._reset_idx = lambda env_ids: None
    return u


def test_installer_runs_and_charges_the_slow_env_only():
    from app.reward_functions.acro_reward_v3 import install_acro_reward_v3
    u = _fake_env(vel=[[0.0, 0, 0], [50.0, 0, 0]])
    cfg = install_acro_reward_v3(u, _cfg(speed_leg=25.0, line_leg=8.0, attitude_soft_deg=85.0, attitude_leg=8.0,
                                         tilt_limit_deg=80.0, tilt_hard_margin_deg=45.0))
    assert u._acro_cfg is cfg and cfg.speed_leg == 25.0
    out = u._compute_dones_and_reward()
    assert out.shape == (2,)
    assert torch.isfinite(u._cached_reward).all() and u._cached_reward.shape == (2,)
    n_steps = u._start_dist[0].item() / (cfg.reference_speed * u.step_dt)
    assert abs(u._acro_acc["r_speed"].item() + 25.0 / n_steps) < 1e-4      # env 0 stands still, env 1 is at the target
    assert u._acro_acc["r_line"].item() == 0.0                             # both are exactly on the line
    assert u._acro_acc["r_att"].item() == 0.0                              # level
    # drift 50 m sideways on the next step: the cross-track term fires, and only for that env
    u._robot.data.root_pos_w[0, 1] = 50.0
    u._compute_dones_and_reward()
    assert abs(u._acro_acc["r_line"].item() + 8.0 / n_steps) < 1e-4
    # the reset hook clears the captured line for the reset env only
    u._reset_idx(torch.tensor([0]))
    assert not u._v3_rel0_valid[0] and u._v3_rel0_valid[1]


def test_stats_include_the_three_new_keys():
    from app.reward_functions.acro_reward_v1 import pop_acro_stats
    from app.reward_functions.acro_reward_v3 import install_acro_reward_v3
    u = _fake_env(vel=[[0.0, 0, 0], [50.0, 0, 0]])
    install_acro_reward_v3(u, _cfg(speed_leg=25.0))
    u._compute_dones_and_reward()
    s = pop_acro_stats(u)
    assert s["rew_speed"] < 0.0 and s["rew_line"] == 0.0 and s["rew_att"] == 0.0
    assert s["rew_speed"] != 0.0 and "rew_total" in s
    # diagnostics in physical units: env 0 stands still (shortfall 1), env 1 is at the target speed (0)
    assert abs(s["train_speed_short_frac"] - 0.5) < 1e-6
    assert s["train_mean_lateral_m"] == 0.0                    # both start on the line


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")


if __name__ == "__main__":
    run_all()
