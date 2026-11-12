import math

import torch

from app.reward_functions.acro_reward_v1 import AcroRewardConfig, compute_acro_reward, tilt_from_quat


def _quat_about_x(deg):
    h = math.radians(deg) / 2
    return torch.tensor([[math.cos(h), math.sin(h), 0.0, 0.0]])


def _run(tilt_deg, cfg, dist=5.0, prev=5.1, left=0.5, away=0, speed=3.0, prev_speed=3.0, ref=10.0):
    return compute_acro_reward(
        pos=torch.tensor([[0.0, 0.0, 5.0]]), dist=torch.tensor([dist]),
        start_dist=torch.tensor([10.0]), quat_wxyz=_quat_about_x(tilt_deg), ang_vel=torch.zeros(1, 3),
        timed_out=torch.tensor([False]), time_left_frac=torch.tensor([left]), stall_count=torch.zeros(1, dtype=torch.long),
        away_count=torch.tensor([away]), away_grace_steps=120,
        grace_steps=60, limit_steps=180, cfg=cfg,
        err=torch.tensor([[dist, 0.0, 0.0]]), prev_err=torch.tensor([[prev, 0.0, 0.0]]),
        speed=torch.tensor([speed]), prev_speed=torch.tensor([prev_speed]), ref_dist=torch.tensor([ref]))


def test_tilt_from_quat_matches_rotation():
    for deg in (0, 45, 90, 120, 179):
        assert abs(math.degrees(tilt_from_quat(_quat_about_x(deg)).item()) - deg) < 0.1


def test_tilt_zones():
    cfg = AcroRewardConfig(tilt_limit_deg=120.0, tilt_hard_margin_deg=30.0)
    assert _run(100, cfg).parts["tilt"].item() == 0.0
    over = _run(130, cfg)
    assert over.parts["tilt"].item() < 0.0 and not over.terminated.item()
    assert _run(155, cfg).tilt_bad.item()


def test_no_limit_at_180():
    cfg = AcroRewardConfig(tilt_limit_deg=180.0)
    assert not _run(179.9, cfg).terminated.item()


def test_fast_hit_pays_more():
    cfg = AcroRewardConfig()
    fast, slow = _run(0, cfg, dist=0.1, left=0.9), _run(0, cfg, dist=0.1, left=0.1)
    assert fast.hit.item() and fast.reward.item() > slow.reward.item()


def test_away_penalty_only_after_grace():
    cfg = AcroRewardConfig()
    assert _run(0, cfg, away=100).parts["away"].item() == 0.0
    assert abs(_run(0, cfg, away=121).parts["away"].item() + cfg.c_away) < 1e-6


def test_braking_penalty_zone_and_direction():
    cfg = AcroRewardConfig(c_decel=0.1, decel_zone_frac=0.33, decel_deadband=0.05)
    # ref 10 m -> zone is dist < 3.3 m. Closing, losing 10 m/s in one step inside the zone -> 0.1 * (10 - 0.05)
    braking = _run(0, cfg, dist=3.0, prev=3.2, speed=5.0, prev_speed=15.0)
    assert abs(braking.parts["decel"].item() - (-0.1 * 9.95)) < 1e-5
    # same drop outside the zone, speeding up, a drop inside the deadband, receding (overshoot), unknown previous speed: all free
    assert _run(0, cfg, dist=6.0, prev=6.2, speed=5.0, prev_speed=15.0).parts["decel"].item() == 0.0
    assert _run(0, cfg, dist=3.0, prev=3.2, speed=15.0, prev_speed=5.0).parts["decel"].item() == 0.0
    assert _run(0, cfg, dist=3.0, prev=3.2, speed=4.97, prev_speed=5.0).parts["decel"].item() == 0.0
    assert _run(0, cfg, dist=3.2, prev=3.0, speed=5.0, prev_speed=15.0).parts["decel"].item() == 0.0
    assert _run(0, cfg, dist=3.0, prev=3.2, speed=5.0, prev_speed=-1.0).parts["decel"].item() == 0.0


def test_braking_penalty_ignores_attitude():
    cfg = AcroRewardConfig()
    flip = _run(100, cfg, dist=3.0, prev=3.2, speed=15.0, prev_speed=15.0)   # a 100 deg flip at constant speed costs nothing extra
    assert flip.parts["decel"].item() == 0.0


def _zone(count, limit, dist=0.1):
    return compute_acro_reward(
        pos=torch.tensor([[0.0, 0.0, 5.0]]), dist=torch.tensor([dist]),
        start_dist=torch.tensor([10.0]), quat_wxyz=_quat_about_x(0), ang_vel=torch.zeros(1, 3),
        timed_out=torch.tensor([False]), time_left_frac=torch.tensor([0.5]), stall_count=torch.zeros(1, dtype=torch.long),
        away_count=torch.zeros(1, dtype=torch.long), away_grace_steps=120, grace_steps=60, limit_steps=180, cfg=AcroRewardConfig(),
        err=torch.tensor([[dist, 0.0, 0.0]]), prev_err=torch.tensor([[dist + 0.1, 0.0, 0.0]]),
        speed=torch.tensor([1.0]), prev_speed=torch.tensor([1.0]), ref_dist=torch.tensor([10.0]),
        zone_count=torch.tensor([count]), limit_steps_hit=limit)


def test_zone_time_needs_consecutive_steps_inside():
    assert not _zone(count=2, limit=5).hit.item() and not _zone(count=2, limit=5).terminated.item()
    assert _zone(count=5, limit=5).hit.item() and _zone(count=5, limit=5).terminated.item()


def test_zone_time_zero_is_the_first_step_inside():
    cfg = AcroRewardConfig()
    assert cfg.zone_time == 0.0
    assert _run(0, cfg, dist=0.1).hit.item()          # no zone_count given: instant hit, like before
    assert not _run(0, cfg, dist=0.5).hit.item()
