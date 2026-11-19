"""Distance buckets on the angle trainer + the distance-dependent time budget they turn on.

Acro's behaviour is the reference: `--buckets` samples the spawn distance from weighted ranges instead of one uniform
range, and switches the hit time bonus to a per-episode clock (budget = start_distance / time_bonus_speed, clamped to
min_leg_s..max_leg_s) instead of one fixed budget. The eval stays on U(--distance-low, --distance-high).
"""
import types

import torch

from app.reward_functions.angle_reward_qv2 import RewardQV2Config, compute_reward_qv2
from app.reward_functions.angle_reward_rc import RewardRCConfig, compute_reward_rc
from app.reward_functions.proportion import time_bonus_frac
from app.training import angle_rtl_train_isaac as angle
from app.training.distance_buckets import DEFAULT_BUCKETS


def _args(reward="rc", radius=None, buckets=None):
    return types.SimpleNamespace(reward=reward, hit_radius=radius, dist_buckets=buckets)


# ---- the flag -------------------------------------------------------------------------------------

def test_buckets_flag_parses_specs_like_acro():
    parser = angle.build_parser()
    assert parser.parse_args([]).buckets is None
    assert parser.parse_args(["--buckets"]).buckets == DEFAULT_BUCKETS
    assert parser.parse_args(["--buckets", "200-400:0.5,800-1200:0.5"]).buckets == "200-400:0.5,800-1200:0.5"


# ---- the clock ------------------------------------------------------------------------------------

def test_buckets_switch_the_bonus_to_a_distance_clock():
    folded = angle.apply_bucket_clock(None)
    assert folded["time_bonus_speed"] == angle.BUCKET_TIME_BONUS_SPEED
    assert folded["r_time_bonus"] == angle.BUCKET_TIME_BONUS


def test_an_explicit_clock_or_bonus_in_the_json_wins():
    assert angle.apply_bucket_clock({"time_bonus_speed": 45.0})["time_bonus_speed"] == 45.0
    assert angle.apply_bucket_clock({"r_time_bonus": 0.0})["r_time_bonus"] == 0.0


def test_a_fixed_budget_is_replaced_by_the_distance_clock():
    assert "time_bonus_budget_s" not in angle.apply_bucket_clock({"time_bonus_budget_s": 40.0, "r_hit": 30})


def test_apply_bucket_clock_leaves_other_keys_alone():
    folded = angle.apply_bucket_clock({"r_hit": 30, "hit_radius": 1.0})
    assert folded["r_hit"] == 30 and folded["hit_radius"] == 1.0


def test_time_bonus_frac_counts_down_the_per_episode_budget():
    cfg = RewardRCConfig(auto_scale=False, time_bonus_speed=30.0)
    ref = torch.tensor([300.0])          # 300 m / 30 m/s = 10 s = 600 policy steps at 1/60
    step_dt = 1 / 60
    assert time_bonus_frac(cfg, ref, torch.tensor([0]), step_dt).item() == 1.0
    assert abs(time_bonus_frac(cfg, ref, torch.tensor([300]), step_dt).item() - 0.5) < 1e-6
    assert time_bonus_frac(cfg, ref, torch.tensor([900]), step_dt).item() == 0.0      # over budget -> never negative


def test_time_bonus_frac_is_off_without_a_speed():
    cfg = RewardRCConfig(auto_scale=False)
    frac = time_bonus_frac(cfg, torch.tensor([300.0]), torch.tensor([0]), 1 / 60)
    assert frac.item() == 1.0


# ---- the bonus is actually paid by both rewards ---------------------------------------------------

def test_rc_hit_pays_the_bonus_scaled_by_the_clock():
    cfg = RewardRCConfig(auto_scale=False, r_time_bonus=5.0, post_hit_steps=0)

    def hit_part(frac):
        n = 1
        out = compute_reward_rc(
            pos=torch.zeros(n, 3), dist=torch.full((n,), 0.5), prev_dist=torch.full((n,), 1.0),
            start_dist=torch.full((n,), 300.0), tilt=torch.zeros(n), roll=torch.zeros(n), pitch=torch.zeros(n),
            interceptor_rel=None, timed_out=torch.zeros(n, dtype=torch.bool),
            stall_count=torch.zeros(n, dtype=torch.long), zone_count=torch.tensor([5], dtype=torch.long),
            first_entry=torch.zeros(n, dtype=torch.bool), grace_steps=60, limit_steps=150, limit_steps_hit=5,
            cfg=cfg, time_left_frac=torch.full((n,), frac),
        )
        return out.parts["hit"].item()

    base = cfg.r_hit - cfg.r_zone_entry
    assert abs(hit_part(0.0) - base) < 1e-4
    assert abs(hit_part(1.0) - (base + 5.0)) < 1e-4


def test_qv2_hit_pays_the_bonus_scaled_by_the_clock():
    cfg = RewardQV2Config(auto_scale=False, r_time_bonus=5.0)

    def hit_part(frac):
        n = 1
        out = compute_reward_qv2(
            pos=torch.zeros(n, 3), dist=torch.full((n,), 0.1), prev_dist=torch.full((n,), 1.0),
            start_dist=torch.full((n,), 300.0), roll=torch.zeros(n), pitch=torch.zeros(n), ang_vel=torch.zeros(n, 3),
            timed_out=torch.zeros(n, dtype=torch.bool), stall_count=torch.zeros(n, dtype=torch.long),
            grace_steps=60, limit_steps=150, cfg=cfg, time_left_frac=torch.full((n,), frac),
        )
        return out.parts["hit"].item()

    assert abs(hit_part(0.0) - cfg.r_hit) < 1e-4
    assert abs(hit_part(1.0) - (cfg.r_hit + 5.0)) < 1e-4
