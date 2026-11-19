"""--hit-radius on the angle trainer.

The two angle rewards name the same thing differently: `--reward rc` has `hit_radius` (drone inside this of the base),
`--reward qv2` has `hit_threshold`. Until now the only way to set it was `--reward-json`, and using the wrong key there
raises `KeyError: unknown reward_rc field 'hit_threshold'`. One flag maps to whichever key the chosen reward uses; an
explicit --reward-json entry still wins.
"""
import types
from pathlib import Path

import pytest

from app.training import angle_rtl_train_isaac as angle

SRC = Path(angle.__file__).read_text()


def _args(reward, radius):
    return types.SimpleNamespace(reward=reward, hit_radius=radius)


def test_flag_exists_and_defaults_to_none():
    parser = angle.build_parser()
    assert parser.parse_args([]).hit_radius is None
    assert parser.parse_args(["--hit-radius", "1.5"]).hit_radius == 1.5
    assert parser.parse_args(["--hit-zone", "2"]).hit_radius == 2.0      # the acro trainer's spelling


def test_maps_to_each_rewards_own_key():
    assert angle.apply_hit_radius(_args("rc", 1.25), None) == {"hit_radius": 1.25}
    assert angle.apply_hit_radius(_args("qv2", 1.25), None) == {"hit_threshold": 1.25}


def test_reward_json_wins_over_the_flag():
    assert angle.apply_hit_radius(_args("rc", 1.25), {"hit_radius": 3.0}) == {"hit_radius": 3.0}
    assert angle.apply_hit_radius(_args("qv2", 1.25), {"hit_threshold": 3.0}) == {"hit_threshold": 3.0}


def test_other_json_keys_are_kept():
    assert angle.apply_hit_radius(_args("rc", 2.0), {"r_hit": 30}) == {"r_hit": 30, "hit_radius": 2.0}


def test_non_positive_radius_is_rejected():
    with pytest.raises(SystemExit):
        angle.apply_hit_radius(_args("rc", 0.0), None)
    with pytest.raises(SystemExit):
        angle.apply_hit_radius(_args("qv2", -1.0), None)


def test_the_mapped_keys_are_real_config_fields():
    from app.reward_functions.angle_reward_qv2 import RewardQV2Config
    from app.reward_functions.angle_reward_rc import RewardRCConfig

    for reward, cfg_cls, key in (("rc", RewardRCConfig, "hit_radius"), ("qv2", RewardQV2Config, "hit_threshold")):
        assert angle.HIT_RADIUS_KEY[reward] == key
        assert getattr(cfg_cls.from_overrides({key: 1.5}), key) == 1.5


def test_legacy_plot_no_longer_requires_hit_threshold():
    # plot_training_run is called with rcfg.hit_threshold, which rc does not have (it was a cosmetic crash)
    assert 'getattr(rcfg, "hit_threshold"' in SRC
