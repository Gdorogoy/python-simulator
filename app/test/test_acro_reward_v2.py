import torch

from app.reward_functions.acro_reward_v2 import AcroRewardV2Config, compute_closing_speed_term, compute_los_term


def _los(vel, rel, gate=None, **kw):
    cfg = AcroRewardV2Config.from_overrides(kw)
    return compute_los_term(vel=torch.tensor([vel], dtype=torch.float32), rel=torch.tensor([rel], dtype=torch.float32),
                            gate=None if gate is None else torch.tensor([gate]), cfg=cfg).item()


def test_aligned_pays_the_flat_cost():
    assert abs(_los([5, 0, 0], [10, 0, 0]) - (-0.02)) < 1e-6


def test_sideways_costs_double_and_away_five_times():
    assert abs(_los([0, 5, 0], [10, 0, 0]) - (-0.04)) < 1e-6
    assert abs(_los([-5, 0, 0], [10, 0, 0]) - (-0.10)) < 1e-6


def test_slow_is_free_and_gate_switches_it_off():
    assert _los([0.1, 0, 0], [-10, 0, 0]) == 0.0
    assert _los([-5, 0, 0], [10, 0, 0], gate=False) == 0.0
    assert _los([-5, 0, 0], [10, 0, 0], gate=True) < 0.0


def test_los_k_zero_is_off_and_cos_mode_rewards_alignment():
    assert _los([-5, 0, 0], [10, 0, 0], los_k=0.0) == 0.0
    assert _los([5, 0, 0], [10, 0, 0], los_mode="cos") > 0.0
    assert _los([-5, 0, 0], [10, 0, 0], los_mode="cos") < 0.0


def test_unknown_override_is_refused():
    try:
        AcroRewardV2Config.from_overrides({"los_kk": 1.0})
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def _progress(rel, prev_rel, **kw):
    from app.reward_functions.acro_reward_v2 import compute_progress_terms
    cfg = AcroRewardV2Config.from_overrides(kw)
    return compute_progress_terms(rel=torch.tensor([rel], dtype=torch.float32), prev_rel=torch.tensor([prev_rel], dtype=torch.float32), cfg=cfg).item()


def test_progress_pays_closing_and_retreating_is_weighted_by_lambda():
    assert _progress([9, 0, 0], [10, 0, 0]) > 0.0
    back = _progress([10, 0, 0], [9, 0, 0])
    assert back < 0.0
    assert abs(_progress([10, 0, 0], [9, 0, 0], closing_lambda=3.0) - 3.0 * back) < 1e-5


def test_progress_axis_term_punishes_drifting_off_an_aligned_axis():
    # distance barely changes but y moves off the line: the axis term is what notices
    assert _progress([10, 1, 0], [10, 0, 0], closing_k=0.0) < 0.0
    assert _progress([10, 1, 0], [10, 0, 0], closing_k=0.0, axis_k=0.0) == 0.0


def _fly(dists, dt=1.0 / 60.0, active=True, **kw):
    """Run the missed-pass detector over a 1-env distance sequence; returns (total penalty, number of passes)."""
    from app.reward_functions.acro_reward_v2 import update_pass_state
    cfg = AcroRewardV2Config.from_overrides(kw)
    app_ = torch.zeros(1, dtype=torch.bool)
    dmin = torch.full((1,), float("inf"))
    total, n = 0.0, 0
    for prev, d in zip(dists[:-1], dists[1:]):
        pen, passed, app_, dmin = update_pass_state(
            dist=torch.tensor([d]), closing_speed=torch.tensor([(prev - d) / dt]), active=torch.tensor([active]),
            approaching=app_, d_min=dmin, cfg=cfg)
        total += pen.item(); n += int(passed.item())
    return total, n


def test_fly_by_costs_its_closest_distance_once():
    dists = [10 - 0.1 * i for i in range(76)] + [2.5 + 0.1 * i for i in range(1, 76)]   # in to 2.5 m, back out
    total, n = _fly(dists)
    assert n == 1 and abs(total - (-2.5)) < 1e-4


def test_two_laps_two_passes_and_far_pass_does_not_count():
    lap = [10 - 0.1 * i for i in range(76)] + [2.5 + 0.1 * i for i in range(1, 76)]
    assert _fly(lap + lap[1:])[1] == 2
    far = [30 - 0.1 * i for i in range(51)] + [25 + 0.1 * i for i in range(1, 51)]    # closest 25 m > miss_radius 15
    assert _fly(far) == (0.0, 0)


def test_slow_wobble_and_inactive_are_free():
    wobble = [5.0, 5.001, 5.0, 5.001, 5.0]          # closing speed ~0.06 m/s, inside the deadband
    assert _fly(wobble) == (0.0, 0)
    lap = [10 - 0.1 * i for i in range(76)] + [2.5 + 0.1 * i for i in range(1, 76)]
    assert _fly(lap, active=False) == (0.0, 0)


def _vc(vel, rel, active=True, **kw):
    cfg = AcroRewardV2Config.from_overrides(kw)
    return compute_closing_speed_term(vel=torch.tensor([vel], dtype=torch.float32), rel=torch.tensor([rel], dtype=torch.float32),
                                      active=torch.tensor([active]), cfg=cfg).item()


def test_closing_speed_is_off_by_default():
    assert _vc([20, 0, 0], [100, 0, 0]) == 0.0


def test_closing_speed_is_linear_in_speed_toward_the_target():
    assert abs(_vc([20, 0, 0], [100, 0, 0], closing_speed_k=0.02) - 0.02) < 1e-6
    assert abs(_vc([40, 0, 0], [400, 0, 0], closing_speed_k=0.02) - 0.04) < 1e-6      # same pay far away (log term: ~0)
    assert abs(_vc([20, 30, 0], [100, 0, 0], closing_speed_k=0.02) - 0.02) < 1e-6     # sideways part does not count


def test_closing_speed_away_costs_lambda_and_gates():
    assert abs(_vc([-20, 0, 0], [100, 0, 0], closing_speed_k=0.02, closing_lambda=2.5) - (-0.05)) < 1e-6
    assert _vc([20, 0, 0], [100, 0, 0], active=False, closing_speed_k=0.02) == 0.0
    assert _vc([20, 0, 0], [5, 0, 0], closing_speed_k=0.02, closing_speed_min_dist=10.0) == 0.0
