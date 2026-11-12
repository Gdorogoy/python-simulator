"""Acro reward v4 (app/reward_functions/acro_reward_v4.py): the swept hit test, the per-METRE cross-track term with its
look-ahead, and the terminal straightness bonus. Same contract as v3: one leg pays exactly the `*_leg` number, and
0 means off. Plain asserts, like the rest of app/test (no pytest needed)."""

import torch

from app.reward_functions.acro_reward_v1 import compute_acro_reward, pop_acro_stats, segment_min_dist
from app.reward_functions.acro_reward_v4 import (AcroRewardV4Config, compute_line_path_term, compute_path_bonus,
                                                 cross_track_lead, path_ratio)


def _close(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


def _cfg(**kw):
    return AcroRewardV4Config.from_overrides(kw)


# ---- the swept hit test --------------------------------------------------------------------------------------------

def _hit(prev_err, err, *, prev_speed=50.0, zone_count=0, hit_swept=True, threshold=0.25):
    """One env, one step of compute_acro_reward with everything else neutral."""
    cfg = _cfg(hit_swept=hit_swept, hit_threshold=threshold, tilt_limit_deg=180.0)
    a = torch.tensor(prev_err, dtype=torch.float32).reshape(1, 3)
    b = torch.tensor(err, dtype=torch.float32).reshape(1, 3)
    out = compute_acro_reward(
        pos=torch.zeros(1, 3), dist=torch.linalg.norm(b, dim=-1), start_dist=torch.tensor([1000.0]),
        quat_wxyz=torch.tensor([[1.0, 0, 0, 0]]), ang_vel=torch.zeros(1, 3),
        timed_out=torch.tensor([False]), time_left_frac=torch.tensor([1.0]),
        stall_count=torch.zeros(1, dtype=torch.long), away_count=torch.zeros(1, dtype=torch.long),
        grace_steps=60, limit_steps=180, away_grace_steps=120, cfg=cfg,
        err=b, prev_err=a, speed=torch.tensor([abs(prev_speed)]), prev_speed=torch.tensor([prev_speed]),
        ref_dist=torch.tensor([1000.0]), zone_count=torch.tensor([zone_count], dtype=torch.long),
        limit_steps_hit=1)
    return bool(out.hit.item())


def test_segment_min_dist_is_the_closest_point_on_the_step():
    _close(segment_min_dist(torch.tensor([[5.0, 0, 0]]), torch.tensor([[-5.0, 0, 0]])).item(), 0.0)
    _close(segment_min_dist(torch.tensor([[10.0, 3, 0]]), torch.tensor([[-2.0, 3, 0]])).item(), 3.0)
    # the foot of the perpendicular is outside the segment: the endpoint is the closest point (5 m, not 0)
    _close(segment_min_dist(torch.tensor([[-5.0, 0, 0]]), torch.tensor([[-5.0, 3, 0]])).item(), 5.0)


def test_swept_hit_scores_a_sphere_crossed_between_two_samples():
    # the two samples are 5 m from the target on either side: the point distance never enters the 0.25 m sphere
    assert _hit([-5.0, 0, 0], [5.0, 0, 0], hit_swept=True)
    assert not _hit([-5.0, 0, 0], [5.0, 0, 0], hit_swept=False)          # 0 = off (v1/v2/v3 behaviour)
    # a segment that misses by more than the threshold still misses
    assert not _hit([-5.0, 0.5, 0], [5.0, 0.5, 0], hit_swept=True)


def test_swept_hit_is_ignored_on_the_first_step_after_a_reset():
    # prev_speed < 0 means prev_err belongs to the previous episode / spawn: no segment, no free hit
    assert not _hit([-5.0, 0, 0], [5.0, 0, 0], prev_speed=-1.0)
    # ... and the zone timer path still works when the sample itself is inside
    assert _hit([0.1, 0, 0], [0.1, 0, 0], prev_speed=-1.0, zone_count=1)


def test_swept_hit_respects_the_zone_timer_when_asked():
    # zone_time > 0 needs that many consecutive samples INSIDE; a swept crossing is one step, so it does not satisfy it
    cfg = _cfg(hit_swept=True, hit_threshold=1.0, tilt_limit_deg=180.0)
    out = compute_acro_reward(
        pos=torch.zeros(1, 3), dist=torch.tensor([5.0]), start_dist=torch.tensor([1000.0]),
        quat_wxyz=torch.tensor([[1.0, 0, 0, 0]]), ang_vel=torch.zeros(1, 3),
        timed_out=torch.tensor([False]), time_left_frac=torch.tensor([1.0]),
        stall_count=torch.zeros(1, dtype=torch.long), away_count=torch.zeros(1, dtype=torch.long),
        grace_steps=60, limit_steps=180, away_grace_steps=120, cfg=cfg,
        err=torch.tensor([[5.0, 0, 0]]), prev_err=torch.tensor([[-5.0, 0, 0]]),
        speed=torch.tensor([50.0]), prev_speed=torch.tensor([50.0]), ref_dist=torch.tensor([1000.0]),
        zone_count=torch.tensor([3], dtype=torch.long), limit_steps_hit=6)
    assert not out.hit.item()


# ---- cross-track per metre ----------------------------------------------------------------------------------------

def _line(rel, vel, *, rel0=(1000.0, 0, 0), lead_s=0.0, active=True, ref_dist=1000.0, **kw):
    return compute_line_path_term(
        rel=torch.tensor(rel, dtype=torch.float32).reshape(1, 3),
        vel=torch.tensor(vel, dtype=torch.float32).reshape(1, 3),
        rel0=torch.tensor(rel0, dtype=torch.float32).reshape(1, 3),
        active=torch.tensor([active]), cfg=_cfg(**kw),
        ref_dist=torch.tensor([ref_dist]), step_dt=1.0 / 60.0).item()


def test_line_path_is_off_by_default_and_free_on_the_line():
    assert _line([900, 50, 0], [50, 0, 0]) == 0.0                     # line_path_leg defaults to 0
    assert _line([900, 0, 0], [50, 0, 0], line_path_leg=10.0) == 0.0  # on the line
    assert _line([900, 50, 0], [50, 0, 0], line_path_leg=10.0, active=False) == 0.0


def test_line_path_pays_the_same_for_the_same_detour_at_any_speed():
    # 50 m off a 1000 m leg with line_ref 50: fly the whole leg at 10 m/s or at 50 m/s, the total is line_path_leg
    for speed in (10.0, 25.0, 50.0):
        n = int(round(1000.0 / (speed / 60.0)))
        one = _line([900, 50, 0], [speed, 0, 0], line_path_leg=10.0)
        _close(one * n, -10.0, tol=1e-3)
    # and it is linear in the offset up to line_ref, flat beyond it
    _close(_line([900, 25, 0], [50, 0, 0], line_path_leg=10.0) * 2, _line([900, 50, 0], [50, 0, 0], line_path_leg=10.0))
    _close(_line([900, 250, 0], [50, 0, 0], line_path_leg=10.0), _line([900, 50, 0], [50, 0, 0], line_path_leg=10.0))


def test_line_path_is_per_metre_not_per_step():
    # twice the speed covers twice the metres in the same step, so the same detour pays twice as much per STEP ...
    slow = _line([900, 50, 0], [25, 0, 0], line_path_leg=10.0)
    fast = _line([900, 50, 0], [50, 0, 0], line_path_leg=10.0)
    _close(fast, 2 * slow)
    # ... and the same total over the leg. A per-step penalty (v3's line_leg) would instead be halved by speed.
    assert _line([900, 50, 0], [25, 0, 0], line_leg=10.0) == 0.0     # v4 ignores line_leg entirely


def test_line_lead_charges_drifting_off_the_line_before_the_offset_exists():
    # on the line (lateral 0) but moving 50 m/s sideways: lead 0 is free, lead 0.5 s anticipates 25 m
    assert cross_track_lead(torch.tensor([[900.0, 0, 0]]), torch.tensor([[0.0, 50, 0]]),
                            torch.tensor([[1000.0, 0, 0]]), 0.0).item() == 0.0
    _close(cross_track_lead(torch.tensor([[900.0, 0, 0]]), torch.tensor([[0.0, 50, 0]]),
                            torch.tensor([[1000.0, 0, 0]]), 0.5).item(), 25.0)
    assert _line([900, 0, 0], [0, 50, 0], line_path_leg=10.0, line_lead_s=0.0) == 0.0
    assert _line([900, 0, 0], [0, 50, 0], line_path_leg=10.0, line_lead_s=0.5) < 0.0
    # the lead does not change a stationary offset: hovering 50 m off the line is charged the same
    _close(_line([900, 50, 0], [0, 0, 0], line_path_leg=10.0, line_lead_s=0.5),
           _line([900, 50, 0], [0, 0, 0], line_path_leg=10.0, line_lead_s=0.0))


# ---- the terminal path bonus --------------------------------------------------------------------------------------

def _bonus(path_len, *, d0=1000.0, hit=True, active=True, r_path_bonus=10.0, **kw):
    return compute_path_bonus(rel0=torch.tensor([[d0, 0, 0]]), path_len=torch.tensor([float(path_len)]),
                              hit=torch.tensor([hit]), active=torch.tensor([active]),
                              cfg=_cfg(**kw), r_path_bonus=r_path_bonus).item()


def test_path_cost_makes_attempting_versus_giving_up_explicit():
    # nothing here changes the attempt/give-up balance; the bonus only separates a hit flown straight from a hit flown
    # the long way round, so it can never make a hit worse than a failed attempt (it is >= 0).
    assert _bonus(2000.0) == 0.0
    assert _bonus(1000.0) == 10.0


def test_path_bonus_is_full_on_the_straight_line_and_zero_at_slack():
    _close(_bonus(1000.0), 10.0)
    _close(_bonus(1050.0), 5.0)                     # 5 % extra with path_slack 0.10: half the bonus
    assert _bonus(1100.0) == 0.0                    # exactly at the slack
    assert _bonus(1500.0) == 0.0                    # and beyond
    assert _bonus(1000.0, r_path_bonus=0.0) == 0.0  # 0 = off
    assert _bonus(1000.0, hit=False) == 0.0         # only a hit pays
    assert _bonus(1000.0, active=False) == 0.0


def test_path_ratio_is_one_for_the_straight_line():
    _close(path_ratio(torch.tensor([[1000.0, 0, 0]]), torch.tensor([1000.0])).item(), 1.0)
    _close(path_ratio(torch.tensor([[1000.0, 0, 0]]), torch.tensor([1250.0])).item(), 0.8)
    assert path_ratio(torch.tensor([[1000.0, 0, 0]]), torch.tensor([900.0])).item() == 1.0    # clamped


# ---- config -------------------------------------------------------------------------------------------------------

def test_v4_defaults_turn_the_new_fixes_on_and_keep_v3_basics():
    cfg = _cfg()
    assert cfg.hit_swept is True                    # the swept hit is v4's default
    assert cfg.line_path_leg == 0.0 and cfg.line_lead_s == 0.0 and cfg.path_slack == 0.10
    assert cfg.speed_target == 50.0 and cfg.line_ref == 50.0     # inherited from v3


def test_overrides_accept_every_new_field_and_refuse_unknown_ones():
    cfg = _cfg(hit_swept=False, line_path_leg=12.0, line_lead_s=0.4, path_slack=0.2, path_leg=20.0)
    assert (cfg.hit_swept, cfg.line_path_leg, cfg.line_lead_s, cfg.path_slack, cfg.path_leg) == (False, 12.0, 0.4, 0.2, 20.0)
    try:
        _cfg(line_path_lag=1.0)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


# ---- installer: the glue that would only fail inside Isaac --------------------------------------------------------

def _fake_env(n=2):
    import types

    u = types.SimpleNamespace()
    u.device = torch.device("cpu")
    u.num_envs, u.step_dt, u.max_episode_length = n, 1.0 / 60.0, 3750
    u.episode_length_buf = torch.zeros(n, dtype=torch.long)
    data = types.SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.zeros(n, 4),
                                 root_lin_vel_w=torch.tensor([[-50.0, 0, 0]] * n), root_ang_vel_b=torch.zeros(n, 3))
    data.root_quat_w[:, 0] = 1.0
    u._robot = types.SimpleNamespace(data=data)
    u._terrain = types.SimpleNamespace(env_origins=torch.zeros(n, 3))
    u._desired_pos_w = torch.zeros(n, 3)
    u._start_dist = torch.full((n,), 1000.0)
    u._prev_position = torch.zeros(n, 3)
    u._prev_distance = torch.full((n,), 1000.0)
    u._prev_target_w = u._desired_pos_w.clone()
    u._cached_reward = torch.zeros(n)
    u._episode_sums = {"reward": torch.zeros(n), "distance_to_goal": torch.zeros(n)}
    u.extras = {}
    u._reset_idx = lambda env_ids: None
    return u


def test_installer_charges_the_detour_per_metre_and_forces_line_leg_off():
    from app.reward_functions.acro_reward_v4 import install_acro_reward_v4
    u = _fake_env()
    u._desired_pos_w = torch.tensor([[1000.0, 0, 0], [1000.0, 0, 0]])
    u._prev_position = torch.zeros(2, 3)
    cfg = install_acro_reward_v4(u, _cfg(line_leg=10.0, line_path_leg=10.0, speed_leg=25.0))
    assert cfg.line_leg == 0.0                                    # per-step line term replaced, with a warning
    out = u._compute_dones_and_reward()
    assert out.shape == (2,) and torch.isfinite(u._cached_reward).all()
    # one step at 50 m/s = 0.833 m flown; on the line -> no line charge
    _close(u._v4_path[0].item(), 50.0 / 60.0)
    assert u._acro_acc["r_line"].item() == 0.0
    # drift both 50 m sideways: both are charged, and the charge is per metre flown
    u._robot.data.root_pos_w[:, 1] = 50.0
    u._compute_dones_and_reward()
    n_steps = 1000.0
    one = 10.0 / n_steps * 1.0 * 50.0 / 60.0                      # (line_path_leg / D) * pen * |v| dt
    _close(u._acro_acc["r_line"].item(), -2 * one, tol=1e-6)
    # the reset hook restarts the odometer for the reset env only
    u._reset_idx(torch.tensor([0]))
    assert u._v4_path[0].item() == 0.0 and u._v4_path[1].item() > 0.0


def test_installer_pays_the_path_bonus_on_a_swept_hit():
    from app.reward_functions.acro_reward_v4 import install_acro_reward_v4
    u = _fake_env()
    u._prev_position = torch.full((2, 3), 10.0)
    u._prev_position[:, 1:] = 0.0
    u._robot.data.root_pos_w = torch.full((2, 3), 10.0)
    u._robot.data.root_pos_w[:, 1:] = 0.0
    cfg = install_acro_reward_v4(u, _cfg(hit_threshold=0.25, path_leg=20.0, tilt_limit_deg=180.0))
    u._compute_dones_and_reward()                                 # captures the leg's line at 10 m, no hit
    assert u._v3_rel0_valid.all() and u._acro_acc["r_path"].item() == 0.0
    # next step: the segment from +5 m to -5 m crosses the target, which neither sample is inside
    u._prev_position = torch.tensor([[5.0, 0, 0], [5.0, 0, 0]])
    u._robot.data.root_pos_w = torch.tensor([[-5.0, 0, 0], [-5.0, 0, 0]])
    u._compute_dones_and_reward()
    assert u.extras["term_reasons"]["hit"].all()
    _close(u._acro_acc["r_path"].item(), 2 * 20.0)                # path_leg for both: the path was shorter than the line
    s = pop_acro_stats(u)
    assert s["rew_path"] > 0.0 and s["train_hit_speed"] == 50.0


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")


if __name__ == "__main__":
    run_all()
