"""Isaac reward for the acro-style drone: qv2's task terms (hit / progress / timeout / crash / stall) with the
roll/pitch cliffs replaced by a quaternion tilt limit that is widened stage by stage. See docs.md "acro_v1"."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, asdict, fields
from typing import NamedTuple

import torch

from app.control.step_budget import MIN_STEPS, STEPS_PER_METER
from app.reward_functions.proportion import RewardProportionMixin, _fill, leg_steps, resolve_proportion
from app.reward_functions.rewards import HIT_THRESHOLD, OOB_RADIUS


@dataclass
class AcroRewardConfig(RewardProportionMixin):
    hit_threshold: float = HIT_THRESHOLD
    zone_time: float = 0.0                 # s the drone must stay inside hit_threshold, in consecutive policy steps, to score the hit (0 = the first step inside; rc uses 5/60)
    hit_swept: bool = False                # True = also score a hit when the SEGMENT flown in one step passes through the sphere (see acro_v4)
    r_hit: float = 10.0
    r_time_bonus: float = 5.0              # extra on a hit, scaled by the fraction of the per-distance time budget left (0 after it)
    r_crash: float = 7.0                  # must not be below the whole stall cost (r_hover + the c_hover accrual), else flying off is the cheapest way out
    r_timeout: float = 3.0
    r_escape: float = 7.0                  # runner target only: the runner left the escape sphere around its spawn point (the drone lost);
                                           # = r_crash so letting it escape is never a cheaper ending than crashing
    progress_k: float = 2.0
    progress_eps: float = 0.5
    # braking penalty: inside the zone near the target, every m/s of speed LOST between two steps while still closing on the target costs
    # c_decel. Only drops count (accelerating is free, and so is being slow), only on the way in (after an overshoot the drone may brake and
    # turn around), and only inside decel_zone_frac * ref (ref = the handoff distance h in --handoff mode, else the start distance).
    # Attitude is NOT limited here: a flip or a fast drift to line up the target costs nothing.
    c_decel: float = 0.1
    decel_zone_frac: float = 0.33
    decel_deadband: float = 0.05           # m/s per step ignored, so measurement noise does not add up
    c_step: float = 0.002
    # ---- tilt curriculum: tilt = angle between body-z and world-up, from the quaternion (no Euler singularity) ----
    tilt_limit_deg: float = 120.0          # soft limit L; widen between stages (180 = no limit)
    tilt_hard_margin_deg: float = 30.0     # crash at L + margin (never trips once that reaches 180)
    c_tilt: float = 5.0                    # -c_tilt * relu(tilt - L)^2  [rad]
    c_tumble: float = 5e-4                 # -min(c_tumble*|w|^2, cap); lower it as L grows
    tumble_cap: float = 0.05
    oob_min_radius: float = OOB_RADIUS
    oob_start_dist_mult: float = 3.0
    # ---- miss + hover (same as qv2) ----
    stall_speed: float = 0.3
    hover_grace_s: float = 1.0
    hover_limit_s: float = 3.0
    c_hover: float = 0.01
    r_hover: float = 18.8                  # + c_hover * (hover_limit - hover_grace) * 60 = 1.2 -> a full stall costs 20.0 = r_crash: parking is exactly as bad as crashing
    # ---- stayed far from the target (new) ----
    away_factor: float = 1.3               # "away" = distance > away_factor * start distance
    away_grace_s: float = 2.0              # away this long before the per-step penalty starts
    c_away: float = 0.02                   # per policy step while still away (0.02 * 60 Hz = 1.2/s); stops the moment it is back
    stall_inner: float = 0.25
    stall_outer: float = 0.75
    diag_stall_speed: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_overrides(overrides: dict | None) -> "AcroRewardConfig":
        cfg = AcroRewardConfig()
        known = {f.name for f in fields(AcroRewardConfig)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown acro reward field {k!r}; known: {sorted(known)}")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


class AcroRewardOut(NamedTuple):
    reward: torch.Tensor
    terminated: torch.Tensor    # crash | hit | hover (timeouts are reported by the env separately)
    hit: torch.Tensor
    oob: torch.Tensor
    tilt_bad: torch.Tensor      # tilt > L + margin
    hover_term: torch.Tensor
    escape: torch.Tensor        # runner target left the escape sphere around its spawn point (all False without a runner)
    tilt: torch.Tensor          # (N,) rad
    parts: dict


def tilt_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Angle (rad, 0..pi) between body-z and world-up. World-z component of body-z is 1 - 2(x^2 + y^2)."""
    cos_t = 1.0 - 2.0 * (quat_wxyz[:, 1].pow(2) + quat_wxyz[:, 2].pow(2))
    return torch.acos(torch.clamp(cos_t, -1.0, 1.0))


def potential(dist: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log1p(dist / eps)


def segment_min_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N,) distance from the origin to the segment a->b: the closest approach of the target to the path flown in one
    step (a = target - pos one step ago, b = target - pos now). At 50 m/s one 60 Hz step is 0.83 m, so a 1 m sphere
    can be crossed between two samples; the point distance alone would miss that hit."""
    ab = b - a
    denom = ab.pow(2).sum(dim=-1)
    t = -(a * ab).sum(dim=-1) / denom.clamp_min(1e-12)
    closest = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
    return torch.linalg.norm(closest, dim=-1)


def compute_acro_reward(*, pos: torch.Tensor, dist: torch.Tensor, start_dist: torch.Tensor,
                        quat_wxyz: torch.Tensor, ang_vel: torch.Tensor, timed_out: torch.Tensor,
                        time_left_frac: torch.Tensor, stall_count: torch.Tensor, away_count: torch.Tensor, grace_steps: int,
                        limit_steps: int, away_grace_steps: int, cfg: AcroRewardConfig,
                        err: torch.Tensor, prev_err: torch.Tensor, speed: torch.Tensor, prev_speed: torch.Tensor,
                        ref_dist: torch.Tensor, zone_count: torch.Tensor | None = None, limit_steps_hit: int = 1,
                        escaped: torch.Tensor | None = None, oob_min: torch.Tensor | None = None) -> AcroRewardOut:
    """pos: (N,3); dist/start_dist/time_left_frac: (N,); quat_wxyz: (N,4); ang_vel: (N,3) body frame;
    timed_out: (N,) bool; stall_count / away_count: (N,) consecutive stalled / far-from-target steps INCLUDING this one.
    err / prev_err: (N,3) target - position now and one step ago (the progress term works per axis).
    speed / prev_speed: (N,) |velocity| now and one step ago (prev_speed < 0 = unknown, e.g. first step); ref_dist: (N,) zone reference distance.
    hit_swept: cfg.hit_swept widens the hit test to the whole segment prev_err -> err, so a sphere crossed between two
    60 Hz samples still counts (prev_speed < 0 disables it: prev_err would be stale).
    zone_count: (N,) consecutive steps inside hit_threshold INCLUDING this one; the hit needs limit_steps_hit of them (None = the first step inside).
    escaped: (N,) bool, runner left its escape sphere (None = no runner); oob_min: (N,) lower bound on the out-of-bounds radius (a runner
    drags the chase far from the start, so the bound follows the target)."""

    oob_radius = torch.clamp(start_dist * cfg.oob_start_dist_mult, min=cfg.oob_min_radius)
    if oob_min is not None:
        oob_radius = torch.maximum(oob_radius, oob_min)
    oob = torch.isnan(pos).any(dim=-1) | (pos[:, 2] < 0.0) | (torch.linalg.norm(pos, dim=-1) > oob_radius)
    tilt = tilt_from_quat(quat_wxyz)
    limit = math.radians(cfg.tilt_limit_deg)
    tilt_bad = tilt > math.radians(cfg.tilt_limit_deg + cfg.tilt_hard_margin_deg)
    crash = oob | tilt_bad
    inside = (dist < cfg.hit_threshold) if zone_count is None else (zone_count >= limit_steps_hit)
    if getattr(cfg, "hit_swept", False) and limit_steps_hit <= 1:
        # swept hit: the target was crossed BETWEEN the two samples. prev_speed < 0 marks the first step after a reset,
        # where prev_err is stale (it belongs to another episode / spawn) and the segment is not a real path.
        # With a zone timer the drone must be INSIDE for that many samples, so a one-step crossing cannot satisfy it.
        swept = (segment_min_dist(prev_err, err) < cfg.hit_threshold) & (prev_speed >= 0.0)
        inside = inside | swept
    hit = inside & ~crash
    escape = torch.zeros_like(hit) if escaped is None else (escaped & ~crash & ~hit)
    hover_term = (stall_count >= limit_steps) & ~crash & ~hit & ~escape

    zero = torch.zeros_like(dist)
    # per-axis potential: a small error on one axis keeps its own steep log gradient even when another axis dominates the
    # distance. k/3 per axis keeps the total from 10 m below r_hit.
    progress = (cfg.progress_k / 3.0) * (potential(prev_err.abs(), cfg.progress_eps)
                                         - potential(err.abs(), cfg.progress_eps)).sum(dim=-1)
    closing = torch.linalg.norm(prev_err, dim=-1) - dist                     # > 0: getting closer
    in_zone = dist < cfg.decel_zone_frac * ref_dist
    drop = torch.relu(prev_speed - speed - cfg.decel_deadband) * (prev_speed >= 0.0).float()
    decel_pen = -cfg.c_decel * drop * (in_zone & (closing > 0.0)).float()
    tumble = -torch.clamp(cfg.c_tumble * ang_vel.pow(2).sum(dim=-1), max=cfg.tumble_cap)
    tilt_pen = -cfg.c_tilt * torch.relu(tilt - limit).pow(2)
    time_bonus = cfg.r_time_bonus * torch.clamp(time_left_frac, 0.0, 1.0)
    timeout_pen = torch.where(timed_out & ~crash & ~hit & ~hover_term & ~escape, -_fill(dist, cfg.r_timeout), zero)
    hover_pen = torch.where(stall_count > grace_steps, -_fill(dist, cfg.c_hover), zero)
    away_pen = torch.where(away_count > away_grace_steps, -_fill(dist, cfg.c_away), zero)
    hover_end = torch.where(hover_term, -_fill(dist, cfg.r_hover), zero)

    live = ~crash
    parts = {
        "progress": torch.where(live, progress, zero),
        "step": torch.where(live & ~hit, -_fill(dist, cfg.c_step), zero),
        "decel": torch.where(live & ~hit, decel_pen, zero),
        "tumble": torch.where(live, tumble, zero),
        "tilt": torch.where(live, tilt_pen, zero),
        "hit": torch.where(hit, cfg.r_hit + time_bonus, zero),
        "crash": torch.where(crash, -_fill(dist, cfg.r_crash), zero),
        "timeout": timeout_pen,
        "hover": torch.where(live & ~hit, hover_pen, zero) + hover_end,
        "away": torch.where(live & ~hit, away_pen, zero),
        "escape": torch.where(escape, -_fill(dist, cfg.r_escape), zero),
    }
    reward = sum(parts.values())
    return AcroRewardOut(reward=reward, terminated=crash | hit | hover_term | escape, hit=hit, oob=oob, tilt_bad=tilt_bad,
                         hover_term=hover_term, escape=escape, tilt=tilt, parts=parts)


def zero_acro_clock(unwrapped_env) -> None:
    """Make every env's episode clock start at 0 (used by eval: it must score true first episodes with full time budgets)."""
    unwrapped_env.episode_length_buf.zero_()
    unwrapped_env._acro_age_off.zero_()
    unwrapped_env._acro_partial.zero_()
    unwrapped_env._acro_stall.zero_()
    unwrapped_env._acro_away.zero_()
    unwrapped_env._acro_zone.zero_()


def set_tilt_limit(unwrapped_env, limit_deg: float, margin_deg: float | None = None) -> None:
    """Change the tilt limit of an installed env (the reward reads the config every step)."""
    unwrapped_env._acro_cfg.tilt_limit_deg = float(limit_deg)
    if margin_deg is not None:
        unwrapped_env._acro_cfg.tilt_hard_margin_deg = float(margin_deg)


# --------------------------------------------------------------------------------------------------
# Isaac glue
# --------------------------------------------------------------------------------------------------

_STAT_KEYS = (
    "steps", "episodes", "hits", "oob", "tilt_crash", "timeouts", "hovers", "hit_steps_sum", "final_dist_sum",
    "speed_sum", "tilt_sum", "over_limit_steps", "tilt_max", "near_steps", "near_speed_sum", "stall_steps",
    "r_progress", "r_step", "r_decel", "r_tumble", "r_tilt", "r_hit", "r_crash", "r_timeout", "r_hover", "r_away", "r_los", "r_total",
    "passes", "near_misses", "r_miss", "r_near", "escapes", "r_escape", "cuts", "r_vclose",
    "r_speed", "r_line", "r_att", "lateral_sum", "lateral_n", "short_sum", "short_n",
    "r_path", "perp_sum", "perp_n", "hit_speed_sum",
)                                                                  # r_los / passes / near_misses / r_miss / r_near stay 0 here: v2 fills them; r_speed / r_line / r_att / lateral_* / short_*: v3; r_path / perp_* / hit_speed_sum: v4


def _new_acc(device) -> dict:
    return {k: torch.zeros((), device=device) for k in _STAT_KEYS}


def pop_acro_stats(unwrapped_env) -> dict:
    """Per-chunk training statistics since the last call (python floats); resets them. One GPU sync per call."""
    acc = unwrapped_env._acro_acc
    vals = torch.stack([acc[k] for k in _STAT_KEYS]).tolist()
    for k in _STAT_KEYS:
        acc[k].zero_()
    s = dict(zip(_STAT_KEYS, vals))
    steps = max(s["steps"], 1.0)
    eps = s["episodes"] if s["episodes"] > 0 else float("nan")
    dt = float(getattr(unwrapped_env, "step_dt", 1.0 / 60.0))
    n_envs = float(unwrapped_env.num_envs)
    out = {
        "train_episodes": s["episodes"],
        # hit_rate above is hits / FINISHED episodes: if the policy stops finishing episodes (loiters until the timeout)
        # the few that do finish are the easy hits and it reads ~1.0. These two are per ENV, so that cannot happen:
        "train_ended_per_env": s["episodes"] / n_envs,
        "train_hit_per_env": s["hits"] / n_envs,
        "train_hit_rate": s["hits"] / eps,
        "train_oob_rate": s["oob"] / eps,
        "train_tilt_rate": s["tilt_crash"] / eps,
        "train_timeout_rate": s["timeouts"] / eps,
        "train_hover_rate": s["hovers"] / eps,
        "train_hit_time_s": (s["hit_steps_sum"] / s["hits"] * dt) if s["hits"] > 0 else float("nan"),
        "train_final_dist": s["final_dist_sum"] / eps,
        "train_mean_speed": s["speed_sum"] / steps,
        "train_mean_tilt_deg": math.degrees(s["tilt_sum"] / steps),
        "train_max_tilt_deg": math.degrees(s["tilt_max"]),
        "train_over_limit_frac": s["over_limit_steps"] / steps,
        "train_near_frac": s["near_steps"] / steps,
        "train_near_speed": (s["near_speed_sum"] / s["near_steps"]) if s["near_steps"] > 0 else float("nan"),
        "train_stall_frac": s["stall_steps"] / steps,
        # v3, in physical units (independent of the coefficients): mean metres off the start->target line, and mean
        # fraction of the target speed the drone is short of, measured along the line of sight.
        "train_mean_lateral_m": s["lateral_sum"] / s["lateral_n"] if s["lateral_n"] > 0 else float("nan"),
        "train_speed_short_frac": s["short_sum"] / s["short_n"] if s["short_n"] > 0 else float("nan"),
        # v4, in physical units: mean perpendicular distance from the target to the velocity line (how parallel the
        # approach is, independent of when it is sampled) and the mean speed of the hits (a swept hit can be fast).
        "train_perp_miss_m": s["perp_sum"] / s["perp_n"] if s["perp_n"] > 0 else float("nan"),
        "train_hit_speed": s["hit_speed_sum"] / s["hits"] if s["hits"] > 0 else float("nan"),
        "train_near_miss_rate": s["near_misses"] / eps,           # v2: episodes ended by the near-miss timer
        "train_passes_per_ep": s["passes"] / eps,                 # v2: missed passes (fly-bys without a hit) per episode
        "train_escape_rate": s["escapes"] / eps,                  # runner target: left the escape sphere around its spawn point
        # The rates above leave out "cut" episodes: the first episode after a full reset (every eval) runs on a randomly shortened clock,
        # and when that clock runs out it is not counted at all. With evals every few chunks (21 s of sim at the defaults) no episode ever
        # reaches the real 62.5 s cap, so train_timeout_rate stays 0 and slow chases vanish from the stats. These two count them:
        "train_cut_rate": s["cuts"] / (s["episodes"] + s["cuts"]) if s["episodes"] + s["cuts"] > 0 else float("nan"),
        "train_hit_all": s["hits"] / (s["episodes"] + s["cuts"]) if s["episodes"] + s["cuts"] > 0 else float("nan"),
    }
    for name in ("progress", "step", "decel", "tumble", "tilt", "away", "hit", "crash", "timeout", "hover", "los", "miss", "near",
                 "escape", "vclose", "speed", "line", "att", "path", "total"):
        out[f"rew_{name}"] = s[f"r_{name}"] / steps
    return out


def install_acro_reward(unwrapped_env, cfg: AcroRewardConfig | None = None) -> AcroRewardConfig:
    """Swap this env instance's _compute_dones_and_reward for the acro reward; adds the `_acro_stall` / `_acro_away` counters."""
    cfg = cfg or AcroRewardConfig()

    unwrapped_env._acro_cfg = cfg
    unwrapped_env._acro_acc = _new_acc(unwrapped_env.device)
    unwrapped_env._acro_stall = torch.zeros(unwrapped_env.num_envs, dtype=torch.long, device=unwrapped_env.device)
    unwrapped_env._acro_away = torch.zeros_like(unwrapped_env._acro_stall)
    unwrapped_env._acro_zone = torch.zeros_like(unwrapped_env._acro_stall)       # consecutive steps inside the hit zone
    # The env randomises episode_length_buf on a FULL reset so envs time out at different steps (desync). That counter is
    # also this reward's episode age, so the age is buf - _acro_age_off. The first episode after a full reset is shortened
    # on purpose: a timeout there is "partial" (no timeout penalty, not counted in the episode statistics).
    unwrapped_env._acro_age_off = torch.zeros_like(unwrapped_env._acro_stall)
    unwrapped_env._acro_prev_speed = torch.full((unwrapped_env.num_envs,), -1.0, device=unwrapped_env.device)   # -1 = unknown
    unwrapped_env._acro_partial = torch.zeros(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device)
    if not getattr(unwrapped_env, "_acro_reset_wrapped", False):
        orig_reset = unwrapped_env._reset_idx

        def _reset_idx(self, env_ids):
            orig_reset(env_ids)
            self._acro_zone.zero_() if (env_ids is None or len(env_ids) == self.num_envs) else self._acro_zone.index_fill_(0, env_ids, 0)
            self._acro_prev_speed.fill_(-1.0) if (env_ids is None or len(env_ids) == self.num_envs) else self._acro_prev_speed.index_fill_(0, env_ids, -1.0)
            if env_ids is None or len(env_ids) == self.num_envs:
                self._acro_age_off.copy_(self.episode_length_buf)
                self._acro_partial.fill_(True)
            else:
                self._acro_age_off[env_ids] = 0
                self._acro_partial[env_ids] = False

        unwrapped_env._reset_idx = types.MethodType(_reset_idx, unwrapped_env)
        unwrapped_env._acro_reset_wrapped = True
    step_dt = float(getattr(unwrapped_env, "step_dt", 1.0 / 60.0))
    if cfg.auto_scale:
        # attempt must beat giving up: a failed attempt costs step_leg + miss_leg, every give-up ending must cost more
        attempt = cfg.step_leg + cfg.miss_leg
        give_up = {"crash": cfg.crash_leg, "hover": cfg.hover_leg, "timeout": cfg.timeout_leg}
        for name, leg in (("escape", cfg.escape_leg), ("near", cfg.near_leg)):
            if hasattr(cfg, f"r_{name}"):
                give_up[name] = leg
        cheap = {name: leg for name, leg in give_up.items() if leg < attempt - 1e-6}
        if cheap:
            print(f"[acro reward] WARNING: auto_scale per-leg totals {cheap} are below a failed attempt "
                  f"(step {cfg.step_leg} + miss {cfg.miss_leg} = {attempt}): giving up would beat trying. Raise them.")
        if cfg.hit_leg < attempt - 1e-6:
            print(f"[acro reward] WARNING: hit_leg {cfg.hit_leg} < step_leg + miss_leg ({attempt}): a hit would pay less "
                  f"than a failed attempt. Raise hit_leg.")
        print(f"[acro reward] auto_scale: one leg = one unit ({cfg.hit_leg:.0f} hit, {cfg.hover_leg:.0f} hover, "
              f"{cfg.crash_leg:.0f} crash, {cfg.timeout_leg:.0f} timeout, {cfg.step_leg:g} step, {cfg.miss_leg:g} miss); "
              f"N(D) = clamp(D / {cfg.reference_speed:g} m/s / dt, {cfg.min_leg_s:g} s, {cfg.max_leg_s:g} s)")
    else:
        stall_cost = cfg.r_hover + cfg.c_hover * (cfg.hover_limit_s - cfg.hover_grace_s) / step_dt
        if cfg.r_crash < stall_cost - 1e-6:   # equal is allowed: parking costs the same as crashing
            print(f"[acro reward] WARNING: r_crash {cfg.r_crash} < full stall cost {stall_cost:.1f}: a policy that cannot hit yet "
                  f"learns to fly off and crash instead of hovering. Raise r_crash or lower r_hover / c_hover.")
    away_grace_steps = max(1, int(round(cfg.away_grace_s / step_dt)))
    grace_steps = max(1, int(round(cfg.hover_grace_s / step_dt)))
    limit_steps = max(grace_steps + 1, int(round(cfg.hover_limit_s / step_dt)))
    limit_steps_hit = max(1, int(round(cfg.zone_time / step_dt)))

    def _compute_dones_and_reward(self):
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        quat = self._robot.data.root_quat_w
        target_local = self._desired_pos_w - self._terrain.env_origins
        rel = target_local - pos
        dist = torch.linalg.norm(rel, dim=-1)
        vel = self._robot.data.root_lin_vel_w
        speed = torch.linalg.norm(vel, dim=-1)
        ang_vel = self._robot.data.root_ang_vel_b
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        age = (self.episode_length_buf - self._acro_age_off).clamp_min(0)
        partial_timeout = timed_out & self._acro_partial
        # zone reference for the braking penalty, "away" and the per-leg proportions: the handoff distance h when a two-stage
        # episode is installed, else the start distance. In a two-stage episode _start_dist is the whole leg from the spawn
        # point (400-600 m), so "away" measured against it (> 1.3 x that) could never happen after the policy takes over.
        handoff_h = getattr(self, "_handoff_h", None)
        if getattr(self, "_handoff_snap_enabled", False) and handoff_h is not None:
            ref_dist = torch.where(handoff_h > 0, handoff_h, self._start_dist)
        else:
            ref_dist = self._start_dist
        cfg_eff = resolve_proportion(self._acro_cfg, ref_dist, step_dt)      # one leg = one unit of reward
        # time bonus: with auto_scale it is measured against the physical per-leg flight time; otherwise against the same
        # per-distance budget the eval uses (control.step_budget.steps_for_dist, in policy steps), NOT the 62 s episode.
        # `time_bonus_speed` / `time_bonus_budget_s` override both: the former with a per-episode distance / speed
        # clock (distance curricula), the latter with one fixed clock. The per-leg budget is the flight time at the
        # REFERENCE speed, and no real flight beats it (it starts at rest), so the bonus was 0 on every hit.
        if getattr(cfg, "time_bonus_speed", 0.0) > 0:
            # distance-dependent clock: one budget per episode, from the leg this env actually has to fly
            budget_steps = torch.clamp(ref_dist / float(cfg.time_bonus_speed),
                                       min=float(cfg.min_leg_s), max=float(cfg.max_leg_s)) / step_dt
        elif cfg.time_bonus_budget_s > 0:
            budget_steps = float(cfg.time_bonus_budget_s) / step_dt
        elif cfg.auto_scale:
            budget_steps = leg_steps(ref_dist, step_dt, cfg)
        else:
            budget_steps = torch.clamp(STEPS_PER_METER * self._start_dist, min=float(MIN_STEPS)) / (step_dt * 240.0)
        time_left = 1.0 - age.float() / budget_steps

        stalled = (dist > cfg_eff.hit_threshold) & (speed < cfg_eff.stall_speed)
        handed = getattr(self, "_handed_off", None)
        if handed is not None:
            policy_on = handed.clone()
        else:
            prefix = getattr(self, "_prefix_left", None)
            policy_on = (prefix <= 0) if prefix is not None else torch.ones_like(dist, dtype=torch.bool)
        stalled = stalled & policy_on
        self._acro_stall = (self._acro_stall + 1) * stalled.long()
        self._acro_away = (self._acro_away + 1) * (dist > cfg_eff.away_factor * ref_dist).long()
        self._acro_zone = (self._acro_zone + 1) * (dist < cfg_eff.hit_threshold).long()
        # moving target (runner): progress is the TRUE change of distance, so it needs where the target was one step ago
        prev_target_local = getattr(self, "_prev_target_w", self._desired_pos_w) - self._terrain.env_origins
        escaped = self.runner_escaped() if hasattr(self, "runner_escaped") else None
        oob_min = None
        if escaped is not None:
            # the runner stays inside escape_dist of its spawn point, and a chasing drone within about the handoff distance of the runner
            spawn_local = self._runner_hold_w - self._terrain.env_origins
            oob_min = torch.linalg.norm(spawn_local, dim=-1) + self._runner_escape_dist + ref_dist
        out = compute_acro_reward(pos=pos, dist=dist, start_dist=self._start_dist,
                                  quat_wxyz=quat, ang_vel=ang_vel, timed_out=timed_out & ~partial_timeout, time_left_frac=time_left,
                                  stall_count=self._acro_stall, away_count=self._acro_away, grace_steps=grace_steps,
                                  limit_steps=limit_steps, away_grace_steps=away_grace_steps,
                                  cfg=cfg_eff, err=rel, prev_err=prev_target_local - self._prev_position, speed=speed,
                                  prev_speed=self._acro_prev_speed, ref_dist=ref_dist,
                                  zone_count=self._acro_zone, limit_steps_hit=limit_steps_hit, escaped=escaped, oob_min=oob_min)
        self._acro_prev_speed = speed.clone()          # a reset in this env.step() puts it back to -1 (see _reset_idx)

        self.extras["term_reasons"] = {"hit": out.hit, "oob": out.oob, "attitude_tilt": out.tilt_bad,
                                       "hover": out.hover_term, "near_miss": torch.zeros_like(out.hit),   # v2 sets near_miss
                                       "escape": out.escape}
        # state of THIS step before _reset_idx overwrites it (the terminal step of an ending episode included).
        # perp = perpendicular distance from the target to the step's velocity line (|rel x vel| / |vel|): small perp + a
        # large closest approach means the drone stopped short; a large perp means it flew past on a parallel line.
        perp = torch.linalg.norm(torch.cross(rel, vel, dim=-1), dim=-1) / speed.clamp_min(1e-6)
        self.extras["acro_step"] = {"tilt": out.tilt, "dist": dist, "speed": speed, "perp": perp}

        ended = out.terminated | timed_out
        self._acro_stall = torch.where(ended, torch.zeros_like(self._acro_stall), self._acro_stall)
        self._acro_away = torch.where(ended, torch.zeros_like(self._acro_away), self._acro_away)

        continuing = ~out.terminated
        self._prev_position = torch.where(continuing.unsqueeze(-1), pos, self._prev_position)
        self._prev_distance = torch.where(continuing, dist, self._prev_distance)

        self._cached_reward = out.reward
        self._episode_sums["reward"] += out.reward
        self._episode_sums["distance_to_goal"] += dist

        acc = self._acro_acc
        timeout_only = timed_out & ~out.terminated & ~partial_timeout
        near = dist < 1.0
        # window just outside the hit zone: stall_inner..stall_outer, shifted up when the hit radius is larger than stall_inner
        # (inside the hit zone the episode has already ended, so a window there never counts anything)
        stall_lo = max(cfg.stall_inner, cfg.hit_threshold)
        diag_stall = (dist > stall_lo) & (dist < stall_lo + (cfg.stall_outer - cfg.stall_inner)) & (speed < cfg.diag_stall_speed)
        acc["steps"] += dist.numel()
        acc["episodes"] += (out.terminated | (timed_out & ~partial_timeout)).sum()
        acc["hits"] += out.hit.sum()
        acc["oob"] += out.oob.sum()
        acc["tilt_crash"] += (out.tilt_bad & ~out.oob).sum()
        acc["timeouts"] += timeout_only.sum()
        acc["hovers"] += out.hover_term.sum()
        acc["escapes"] += out.escape.sum()
        acc["cuts"] += (partial_timeout & ~out.terminated).sum()      # shortened first episode ran out of time: not in "episodes" above
        acc["hit_steps_sum"] += (age.float() * out.hit.float()).sum()
        acc["final_dist_sum"] += (dist * (out.terminated | (timed_out & ~partial_timeout)).float()).sum()
        acc["speed_sum"] += speed.sum()
        acc["tilt_sum"] += out.tilt.sum()
        acc["over_limit_steps"] += (out.tilt > math.radians(cfg.tilt_limit_deg)).sum()
        acc["tilt_max"] = torch.maximum(acc["tilt_max"], out.tilt.max())
        acc["near_steps"] += near.sum()
        acc["near_speed_sum"] += (speed * near.float()).sum()
        acc["stall_steps"] += diag_stall.sum()
        acc["perp_sum"] += (perp * policy_on.float()).sum()
        acc["perp_n"] += policy_on.float().sum()
        acc["hit_speed_sum"] += (speed * out.hit.float()).sum()
        for name, part in out.parts.items():
            acc[f"r_{name}"] += part.sum()
        acc["r_total"] += out.reward.sum()

        return out.terminated

    unwrapped_env._compute_dones_and_reward = types.MethodType(_compute_dones_and_reward, unwrapped_env)
    return cfg
