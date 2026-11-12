"""Per-leg reward proportions (app/reward_functions/proportion.py): one leg = one unit of reward at any distance.
Plain asserts, like the rest of app/test (no pytest needed)."""
import torch

from app.reward_functions.acro_reward_v1 import AcroRewardConfig
from app.reward_functions.acro_reward_v2 import AcroRewardV2Config
from app.reward_functions.angle_reward_qv2 import RewardQV2Config
from app.reward_functions.angle_reward_rc import RewardRCConfig
from app.reward_functions.proportion import _fill, _is_zero, leg_steps, resolve_proportion

DT = 1.0 / 60.0
DISTS = (250.0, 500.0, 1000.0, 1500.0)


def _close(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_leg_steps_is_the_physical_flight_time():
    cfg = AcroRewardConfig()
    for d in DISTS:
        raw = d / (cfg.reference_speed * DT)
        expected = min(max(raw, cfg.min_leg_s / DT), cfg.max_leg_s / DT)
        _close(leg_steps(d, DT, cfg), expected)
    _close(leg_steps(1000.0, DT, cfg), 1000.0 / (55.0 * DT))                   # 55 m/s, the physical limit


def test_leg_steps_clamps_both_ends():
    cfg = AcroRewardConfig()
    _close(leg_steps(1.0, DT, cfg), cfg.min_leg_s / DT)
    _close(leg_steps(1e6, DT, cfg), cfg.max_leg_s / DT)
    t = leg_steps(torch.tensor([1.0, 1000.0, 1e6]), DT, cfg)
    assert t.shape == (3,)
    _close(t[0].item(), cfg.min_leg_s / DT)
    _close(t[2].item(), cfg.max_leg_s / DT)


def test_per_step_terms_total_one_leg_at_any_distance():
    for cfg in (AcroRewardV2Config(), RewardRCConfig(), RewardQV2Config()):
        for d in DISTS:
            c = resolve_proportion(cfg, d, DT)
            n = leg_steps(d, DT, cfg)
            _close(c.c_step * n, cfg.step_leg)
            _close(c.c_hover * n, cfg.hover_accrual_leg)
            if hasattr(cfg, "los_k") and not _is_zero(cfg.los_k):
                _close(c.los_k * n, cfg.los_leg)
            if hasattr(cfg, "closing_k") and not _is_zero(cfg.closing_k):
                _close(c.closing_k * n, cfg.closing_leg)


def test_terminals_are_paid_once_so_they_are_the_leg_value():
    for cfg in (AcroRewardV2Config(), RewardRCConfig(), RewardQV2Config()):
        for d in DISTS:
            c = resolve_proportion(cfg, d, DT)
            assert c.r_hit == cfg.hit_leg
            assert c.r_crash == cfg.crash_leg
            assert c.r_hover == cfg.hover_leg
            assert c.r_timeout == cfg.timeout_leg
    rc = resolve_proportion(RewardRCConfig(), 500.0, DT)
    assert rc.r_zone_entry == RewardRCConfig().zone_entry_leg
    assert rc.r_intercepted == RewardRCConfig().intercepted_leg


def test_miss_costs_the_leg_value_at_the_edge_of_the_radius():
    cfg = AcroRewardV2Config()
    c = resolve_proportion(cfg, 1000.0, DT)
    _close(c.k_miss * cfg.miss_radius, cfg.miss_leg)
    assert c.k_miss * 0.0 == 0.0                                               # a perfect pass is free


def test_attempt_beats_giving_up_by_default():
    cfg = AcroRewardV2Config()
    attempt = cfg.step_leg + cfg.miss_leg
    assert cfg.hit_leg >= attempt
    for leg in (cfg.hover_leg, cfg.crash_leg, cfg.timeout_leg, cfg.near_leg, cfg.escape_leg):
        assert leg > attempt
    assert cfg.hit_leg > cfg.hover_leg                                        # a hit must beat parking
    rc = RewardRCConfig()
    assert rc.hit_leg >= rc.step_leg + rc.miss_leg
    assert rc.intercepted_leg > rc.step_leg + rc.miss_leg


def test_zero_means_off_and_stays_off():
    v2 = resolve_proportion(AcroRewardV2Config(), 1000.0, DT)
    assert v2.progress_k == 0.0                                               # acro_v2 turns v1's progress term off
    assert v2.closing_speed_k == 0.0                                          # optional term off by default
    rc = RewardRCConfig.from_overrides({"axis_k": 0.0, "los_k": 0.0})
    rc = resolve_proportion(rc, 1000.0, DT)
    assert rc.axis_k == 0.0 and rc.los_k == 0.0


def test_auto_scale_false_returns_the_config_untouched():
    cfg = AcroRewardV2Config.from_overrides({"auto_scale": False})
    out = resolve_proportion(cfg, 1500.0, DT)
    assert out is cfg
    assert out.c_step == AcroRewardV2Config().c_step
    assert out.r_hit == AcroRewardV2Config().r_hit


def test_from_overrides_accepts_the_new_fields():
    cfg = AcroRewardV2Config.from_overrides({"hit_leg": 80.0, "reference_speed": 52.0, "auto_scale": False})
    assert cfg.hit_leg == 80.0 and cfg.reference_speed == 52.0 and cfg.auto_scale is False
    assert RewardRCConfig.from_overrides({"intercepted_leg": 60.0}).intercepted_leg == 60.0
    assert RewardQV2Config.from_overrides({"timeout_leg": 55.0}).timeout_leg == 55.0


def test_fill_accepts_scalars_and_per_env_tensors():
    dist = torch.zeros(3)
    assert torch.equal(_fill(dist, 2.0), torch.full_like(dist, 2.0))
    per_env = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(_fill(dist, per_env), per_env)
    assert _is_zero(0.0) and not _is_zero(0.1) and not _is_zero(per_env)


def test_compute_functions_stay_pure():
    """The scaling lives in the install_* glue: a direct compute call with auto_scale=True must use the raw fields."""
    from app.reward_functions.angle_reward_qv2 import compute_reward_qv2
    cfg = RewardQV2Config()                                                   # auto_scale True, but compute ignores it
    n = 1
    out = compute_reward_qv2(pos=torch.zeros(n, 3), dist=torch.tensor([10.0]), prev_dist=torch.tensor([10.0]),
                             start_dist=torch.tensor([10.0]), roll=torch.zeros(n), pitch=torch.zeros(n),
                             ang_vel=torch.zeros(n, 3), timed_out=torch.zeros(n, dtype=torch.bool),
                             stall_count=torch.zeros(n, dtype=torch.long), grace_steps=60, limit_steps=180, cfg=cfg)
    _close(out.parts["step"][0].item(), -cfg.c_step)


def run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    run_all()
