"""Numpy rc terminal parity with angle_reward_rc.compute_reward_rc (plain asserts, like the rest of app/test).

The Angle tab used to end a flight on the old instant 0.25 m hit with no tilt check. It now uses the run's rc
config at the 60 Hz policy rate: zone dwell -> hit, post-hit window, tilt/roll/pitch crash, OOB and hover.
"""
import math

from app.guidance.record_run import _RCTerminalChecker, _segment_min_dist, _tilt_from_quat_deg
from app.reward_functions.angle_reward_rc import RewardRCConfig

DT = 1.0 / 60.0
CFG = RewardRCConfig(
    hit_radius=0.75, zone_time=0.05, post_hit_steps=6,
    tilt_limit_deg=85.0, roll_limit_deg=360.0, pitch_limit_deg=360.0,
    stall_speed=0.5, hover_grace_s=1.0, hover_limit_s=2.5,
    oob_min_radius=50.0, oob_start_dist_mult=3.0,
)


def _step(rc, dist=10.0, speed=10.0, tilt=0.0, roll=0.0, pitch=0.0, pos=(0.0, 0.0, 10.0),
          target=(0.0, 0.0, 10.0)):
    return rc.step(pos, dist, speed, tilt, roll, pitch, target_pos=target)


def test_dwell_hit_then_post_hit_window():
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)
    out = [_step(rc, dist=0.5) for _ in range(4)]
    assert [hit for _, hit in out] == [False, False, True, False]      # 3 policy steps = zone_time at 60 Hz
    assert out[2][0] is None                                           # the hit does not end the flight
    # post_hit_steps=6: the 4th call above was the first post-hit step, so 5 more end the window
    # (isaac_eval still scores the episode as Hit)
    terms = [_step(rc, dist=10.0)[0] for _ in range(5)]
    assert terms == [None] * 4 + ["Hit"]
    assert rc.hit_scored and rc.hit_step == 3


def test_dwell_resets_when_leaving_the_zone():
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)
    for _ in range(2):
        assert _step(rc, dist=0.5)[1] is False
    assert _step(rc, dist=10.0)[1] is False
    for _ in range(2):
        assert _step(rc, dist=0.5)[1] is False
    assert _step(rc, dist=0.5)[1] is True                              # 3 fresh consecutive steps


def test_zero_post_hit_window_ends_on_the_hit():
    rc = _RCTerminalChecker(RewardRCConfig(**{**CFG.to_dict(), "post_hit_steps": 0}), DT)
    rc.reset(210.0)
    assert _step(rc, dist=0.5)[0] is None
    assert _step(rc, dist=0.5)[0] is None
    term, hit = _step(rc, dist=0.5)
    assert hit and term == "Hit"


def test_tilt_maps_to_the_trainers_roll_bucket():
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)
    assert _step(rc, tilt=85.0)[0] is None                              # exactly at the limit is still alive
    term, hit = _step(rc, tilt=85.1)
    assert term == "attitude-ROLL" and not hit                         # isaac_eval buckets tilt as roll
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)
    assert _step(rc, pitch=360.1)[0] == "attitude-PITCH"
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)
    assert _step(rc, roll=360.1)[0] == "attitude-ROLL"


def test_crash_during_the_post_hit_window_keeps_the_hit_scored():
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)
    for _ in range(3):
        _step(rc, dist=0.5)
    term, hit = _step(rc, dist=10.0, tilt=90.0)
    assert term == "attitude-ROLL" and not hit
    assert rc.hit_scored


def test_hover_terminates_after_the_limit():
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)
    terms = [_step(rc, dist=10.0, speed=0.0)[0] for _ in range(150)]   # 2.5 s at 60 Hz
    assert terms[:149] == [None] * 149
    assert terms[149] == "hover"


def test_oob_uses_the_trained_radius_and_floor():
    rc = _RCTerminalChecker(CFG, DT)
    rc.reset(210.0)                                                    # 3 * 210 m, above the 50 m floor
    assert _step(rc, pos=(0.0, 0.0, 629.0))[0] is None
    assert _step(rc, pos=(0.0, 0.0, 631.0))[0] == "oob"
    assert _step(rc, pos=(0.0, 0.0, -0.1))[0] == "oob"
    rc.reset(10.0)                                                     # 50 m floor wins
    assert _step(rc, pos=(0.0, 0.0, 49.0))[0] is None
    assert _step(rc, pos=(0.0, 0.0, 51.0))[0] == "oob"


def test_zone_timer_progress_uses_the_trained_radius():
    rc = _RCTerminalChecker(CFG, DT, timer=(0.75, 0.05))
    rc.reset(210.0)
    _step(rc, dist=0.5)
    _step(rc, dist=0.5)
    assert abs(rc.timer_progress() - 2.0 / 3.0) < 1e-9
    _step(rc, dist=2.0)
    assert rc.timer_progress() == 0.0


def test_tilt_from_quat():
    assert abs(_tilt_from_quat_deg((1.0, 0.0, 0.0, 0.0))) < 1e-9
    s = math.sin(math.pi / 4.0)
    assert abs(_tilt_from_quat_deg((s, s, 0.0, 0.0)) - 90.0) < 1e-6     # 90 deg roll: body z is horizontal


def test_segment_min_dist():
    assert abs(_segment_min_dist((0.0, 2.0, 0.0), (0.0, -2.0, 0.0))) < 1e-9      # segment crosses the origin
    assert abs(_segment_min_dist((0.0, 3.0, 0.0), (0.0, 5.0, 0.0)) - 3.0) < 1e-9  # closest is the near end
    assert abs(_segment_min_dist((1.0, 0.0, 0.0), (1.0, 5.0, 0.0)) - 1.0) < 1e-9  # parallel, closest is the line


def test_acro_rule_counts_the_first_step_inside():
    rc = _RCTerminalChecker(CFG, DT, hit_rule="acro")
    assert rc.hit_dwell_s == 0.0 and rc.limit_steps_hit == 1
    rc.reset(210.0)
    term, hit = _step(rc, dist=0.5)
    assert hit and term is None                                        # the post-hit window keeps the flight alive
    assert rc.hit_step == 1
    terms = [_step(rc, dist=10.0)[0] for _ in range(6)]
    assert terms == [None] * 5 + ["Hit"]


def test_acro_rule_counts_a_swept_crossing():
    rc = _RCTerminalChecker(CFG, DT, hit_rule="acro")
    rc.reset(210.0)
    target = (0.0, 10.0, 0.0)
    # both samples are 2 m from the target, outside the 0.75 m zone, but the path between them crosses it
    assert _step(rc, dist=2.0, speed=40.0, pos=(0.0, 8.0, 0.0), target=target) == (None, False)
    term, hit = _step(rc, dist=2.0, speed=40.0, pos=(0.0, 12.0, 0.0), target=target)
    assert hit and term is None
    assert rc.hit_step == 2


def test_acro_rule_skips_the_swept_test_on_the_first_step():
    rc = _RCTerminalChecker(CFG, DT, hit_rule="acro")
    rc.reset(210.0)
    # no previous position: only the point distance can score
    term, hit = _step(rc, dist=2.0, speed=40.0, pos=(0.0, 12.0, 0.0), target=(0.0, 10.0, 0.0))
    assert not hit and term is None


def test_acro_rule_crash_beats_a_hit_on_the_same_step():
    rc = _RCTerminalChecker(CFG, DT, hit_rule="acro")
    rc.reset(210.0)
    term, hit = _step(rc, dist=0.5, tilt=85.1)
    assert term == "attitude-ROLL" and not hit
    assert not rc.hit_scored


def test_trained_rule_still_needs_the_dwell():
    rc = _RCTerminalChecker(CFG, DT)                                    # default: "trained"
    assert rc.hit_rule == "trained" and rc.limit_steps_hit == 3
    rc.reset(210.0)
    assert _step(rc, dist=0.5)[1] is False                             # one step is not enough
