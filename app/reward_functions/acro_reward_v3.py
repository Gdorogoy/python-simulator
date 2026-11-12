"""Acro reward v3 = v2 plus the terms that make the drone fly FAST and STRAIGHT.

Why v2 could not. Every shaping term in v1/v2 is the integral of a function of position, so it pays for the PATH and
not for the PACE. The log potentials (`closing_k`, `axis_k`, `progress_k`) telescope to a log of the distances, and the
linear closing-speed term is even more explicit:

    sum_t closing_speed_k * v_toward / closing_speed_ref = close_speed_leg * reference_speed / closing_speed_ref

A 26 m/s flight and a 50 m/s flight over the same path earn exactly the same `vclose`. Measured on a 1000-1500 m run:
`rew_vclose` +0.0118/step against `rew_step` -0.0037/step, so doubling the speed paid ~8 units per leg while a crash
cost 45 and the hit (50) sat 2500 steps away (0.995 ** 2500 = 3e-6). Flying at 26 m/s was the risk-optimal choice.

A per-step reward can only reward speed in three ways, and v3 adds the two that are safe:

1. a constant per-step cost (`c_step`, already in v1) - a slow flight pays it for more steps;
2. a NON-LINEAR function of speed (`speed_leg` below) - the drone is charged for falling short of a brake-limited
   target speed, so the charge is paid on every step it is slow instead of once at the end;
3. a terminal bonus (`r_hit` + `r_time_bonus`) - reachable only with a working time budget, see
   `time_bonus_budget_s` in `proportion.py` (with the reference-speed budget the bonus was always 0).

A reward for `dv/dt` (an "acceleration bonus") is deliberately NOT added: a reward for accelerating pays for
oscillating, and the target-speed profile below already pays for getting up to speed quickly.

`speed_leg` also carries the path, because it measures the velocity component ALONG the line of sight only: a metre
flown sideways earns nothing here and still pays `c_step` and `los`. `line_leg` adds the cross-track term for a
curved-but-fast approach, and `attitude_leg` charges for rolling past `attitude_soft_deg` - with `tilt_limit_deg`
180 the policy flew an eval max tilt of 100 deg, i.e. inverted, for free.

Everything else (hit, hover, tilt crash, away, near-miss, miss, los, closing) is v2 unchanged.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, fields

import torch

from app.reward_functions.acro_reward_v1 import tilt_from_quat
from app.reward_functions.acro_reward_v2 import AcroRewardV2Config, install_acro_reward_v2
from app.reward_functions.proportion import _is_zero, leg_steps


@dataclass
class AcroRewardV3Config(AcroRewardV2Config):
    # ---- speed: a brake-limited target speed. The only structurally sound way to pay for speed. ----
    speed_target: float = 50.0        # m/s cruise target (52 is the physical limit at ~70 deg tilt / full thrust)
    speed_accel: float = 16.0         # m/s^2 braking budget -> v* = min(target, sqrt(end^2 + 2 * a * dist))
    # the speed the profile brakes to AT the target. 0 = brake to a stop (sqrt(2ad), the LOS teacher's landing profile),
    # which makes the last ~80 m a creep and is what made the policy arrive at 5 m/s. A hit only needs to pass through
    # the hit sphere, so braking to the crossing speed (e.g. 20-25 m/s) removes the creep WITHOUT costing accuracy: the
    # miss distance is set by the straightness of the path hundreds of metres earlier, not by the last 20 m.
    speed_end_speed: float = 0.0
    speed_leg: float = 25.0           # per-leg cost when v_toward == 0 for the whole leg (0 = off)
    speed_min_dist: float = 0.0       # m; only while farther than this
    # ---- straight line: cross-track error from the episode's start->target line ----
    line_leg: float = 0.0             # per-leg cost at line_ref metres of cross-track (0 = off)
    # NOTE: the term is linear up to `line_ref` and FLAT beyond it (no gradient), so line_ref must be larger than any
    # offset you actually want corrected. At 50 the per-step charge was only -0.0038/step on the first reward-v3 run,
    # small next to the speed term.
    line_ref: float = 50.0
    # ---- soft attitude wall: a roll past 90 deg is not free ----
    attitude_soft_deg: float = 0.0    # 0 = off; the charge starts above this tilt
    attitude_leg: float = 0.0         # cost at tilt_limit_deg + tilt_hard_margin_deg

    @staticmethod
    def from_overrides(overrides: dict | None) -> "AcroRewardV3Config":
        cfg = AcroRewardV3Config()
        known = {f.name for f in fields(AcroRewardV3Config)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown acro reward field {k!r}; known: {sorted(known)}")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


def speed_profile(dist: torch.Tensor, cfg: AcroRewardV3Config) -> torch.Tensor:
    """(N,) the brake-limited target speed at this distance: fast while far, slowing so it arrives at `speed_end_speed`.
    `sqrt(end^2 + 2 * a * dist)` is the LOS teacher's landing profile with a non-zero touchdown speed: `end = 0` gives
    `sqrt(2 * a * dist)` (brake to a stop at the target), `end = 22` brakes only to 22 m/s, ~63 m out at 50 m/s."""
    target = float(cfg.speed_target)
    end = min(float(cfg.speed_end_speed), target)
    v = torch.sqrt((end * end + 2.0 * float(cfg.speed_accel) * dist).clamp_min(0.0))
    return torch.clamp(v, min=end, max=target)


def speed_shortfall(vel: torch.Tensor, rel: torch.Tensor, dist: torch.Tensor, cfg: AcroRewardV3Config) -> torch.Tensor:
    """(N,) in [0, 1]: how far below the target speed the drone is, measured along the line of sight only.
    `v_toward = v . rel_hat`, so flying sideways (or away) is 'slow' even at 50 m/s airspeed. Used by the reward and by
    the `train_speed_short_frac` diagnostic (0 = it is tracking the profile, 1 = standing still / flying away)."""
    v_toward = (vel * rel).sum(dim=-1) / (dist + 1e-6)
    ref = max(float(cfg.speed_target), 1e-6)
    return ((speed_profile(dist, cfg) - v_toward) / ref).clamp(0.0, 1.0)


def compute_speed_track_term(*, vel: torch.Tensor, rel: torch.Tensor, dist: torch.Tensor, active: torch.Tensor,
                             cfg: AcroRewardV3Config, n_steps) -> torch.Tensor:
    """(N,) per-step cost of falling short of the brake-limited target speed, normalised to `speed_leg` per leg:
    -(speed_leg / N) * clamp((v* - v_toward) / speed_target, 0, 1), 0 where inactive or inside `speed_min_dist`.

    v_toward is the velocity along the line of sight, so flying sideways earns nothing (path and speed in one term).
    Flying away gives a bigger shortfall than standing still, and it clamps at one leg's worth. 0 where speed_leg is 0.
    """
    if _is_zero(cfg.speed_leg):
        return torch.zeros_like(dist)
    short = speed_shortfall(vel, rel, dist, cfg)
    on = active & (dist > float(cfg.speed_min_dist))
    return -(float(cfg.speed_leg) / n_steps) * short * on.float()


def cross_track(rel: torch.Tensor, rel0: torch.Tensor) -> torch.Tensor:
    """(N,) perpendicular distance from the straight line along rel0 through the target: the 'how far off the optimal
    path am I' number, in metres. rel0 need not be unit length (it is the direction to the target on the first step)."""
    rel0_hat = rel0 / (torch.linalg.norm(rel0, dim=-1, keepdim=True) + 1e-6)
    along = (rel * rel0_hat).sum(dim=-1, keepdim=True)
    return torch.linalg.norm(rel - along * rel0_hat, dim=-1)


def compute_line_term(*, rel: torch.Tensor, rel0: torch.Tensor, active: torch.Tensor, cfg: AcroRewardV3Config,
                      n_steps) -> torch.Tensor:
    """(N,) cross-track penalty against the episode's start->target line:
    -(line_leg / N) * clamp(lateral / line_ref, 0, 1), 0 where inactive or line_leg is 0.
    rel0 is the direction to the target on the first policy step (start of the leg), rel the same now.
    Linear up to `line_ref` metres, flat beyond it, so `line_ref` must cover the offsets you want corrected."""
    if _is_zero(cfg.line_leg):
        return torch.zeros_like(torch.linalg.norm(rel, dim=-1))
    pen = (cross_track(rel, rel0) / max(float(cfg.line_ref), 1e-6)).clamp(0.0, 1.0)
    return -(float(cfg.line_leg) / n_steps) * pen * active.float()


def compute_attitude_term(*, tilt: torch.Tensor, cfg: AcroRewardV3Config, n_steps) -> torch.Tensor:
    """(N,) linear ramp from `attitude_soft_deg` to the hard tilt cap, normalised to `attitude_leg` per leg:
    -(attitude_leg / N) * clamp((tilt - soft) / (hard - soft), 0, 1). 0 below the soft angle or where it is off."""
    if _is_zero(cfg.attitude_leg) or float(cfg.attitude_soft_deg) <= 0.0:
        return torch.zeros_like(tilt)
    soft = math.radians(float(cfg.attitude_soft_deg))
    hard = math.radians(float(cfg.tilt_limit_deg) + float(cfg.tilt_hard_margin_deg))
    span = max(hard - soft, 1e-3)
    return -(float(cfg.attitude_leg) / n_steps) * ((tilt - soft) / span).clamp(0.0, 1.0)


def install_acro_reward_v3(unwrapped_env, cfg: AcroRewardV3Config | None = None) -> AcroRewardV3Config:
    """Install v2, then add the speed / cross-track / attitude parts on top of its reward, episode sums and statistics
    (`rew_speed`, `rew_line`, `rew_att`). The new fields are per-leg totals by construction, divided by the live N
    every step, so they scale with the stage distance like the rest of the reward (see proportion.py)."""
    cfg = cfg or AcroRewardV3Config()
    install_acro_reward_v2(unwrapped_env, cfg)
    v2_compute = unwrapped_env._compute_dones_and_reward
    u = unwrapped_env
    step_dt = float(getattr(u, "step_dt", 1.0 / 60.0))
    u._v3_rel0 = torch.zeros((u.num_envs, 3), device=u.device)
    u._v3_rel0_valid = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    if not getattr(u, "_acro_v3_reset_wrapped", False):
        prev_reset = u._reset_idx

        def _reset_idx(self, env_ids):
            prev_reset(env_ids)
            if env_ids is None or len(env_ids) == self.num_envs:
                self._v3_rel0.zero_()
                self._v3_rel0_valid.fill_(False)
            else:
                self._v3_rel0_valid[env_ids] = False

        u._reset_idx = types.MethodType(_reset_idx, u)
        u._acro_v3_reset_wrapped = True

    def _compute_dones_and_reward(self):
        terminated = v2_compute()                      # v2 sets the reward, the episode sums and the v1/v2 statistics
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        rel = (self._desired_pos_w - self._terrain.env_origins) - pos
        dist = torch.linalg.norm(rel, dim=-1)
        tilt = tilt_from_quat(self._robot.data.root_quat_w)
        handoff_h = getattr(self, "_handoff_h", None)
        if getattr(self, "_handoff_snap_enabled", False) and handoff_h is not None:
            ref_dist = torch.where(handoff_h > 0, handoff_h, self._start_dist)
        else:
            ref_dist = self._start_dist
        n_steps = leg_steps(ref_dist, step_dt, self._acro_cfg)          # one leg = one unit, same N as the rest
        handed = getattr(self, "_handed_off", None)
        if handed is not None:
            policy_on = handed.clone()
        else:
            prefix = getattr(self, "_prefix_left", None)
            policy_on = (prefix <= 0) if prefix is not None else torch.ones_like(dist, dtype=torch.bool)
        alive = policy_on & ~terminated
        # the straight line of THIS leg: the direction to the target on its first policy step (reset clears it)
        capture = alive & ~self._v3_rel0_valid & (dist > 1.0)
        self._v3_rel0 = torch.where(capture.unsqueeze(-1), rel, self._v3_rel0)
        self._v3_rel0_valid = self._v3_rel0_valid | capture
        on_line = alive & self._v3_rel0_valid
        speed = compute_speed_track_term(vel=self._robot.data.root_lin_vel_w, rel=rel, dist=dist, active=alive,
                                         cfg=self._acro_cfg, n_steps=n_steps)
        line = compute_line_term(rel=rel, rel0=self._v3_rel0, active=on_line,
                                 cfg=self._acro_cfg, n_steps=n_steps)
        att = compute_attitude_term(tilt=tilt, cfg=self._acro_cfg, n_steps=n_steps)
        extra = speed + line + att
        self._cached_reward = self._cached_reward + extra
        self._episode_sums["reward"] += extra
        acc = self._acro_acc
        acc["r_speed"] += speed.sum()
        acc["r_line"] += line.sum()
        acc["r_att"] += att.sum()
        acc["r_total"] += extra.sum()
        # diagnostics in physical units, independent of the coefficients: how far off the line (m) and how far below the
        # target speed (fraction). A mean lateral of ~33 m and a shortfall of 0.31 is what the tuned reward-v3 run
        # measured (a flight mean; the terminal aim is ~0.5 m).
        short = speed_shortfall(self._robot.data.root_lin_vel_w, rel, dist, self._acro_cfg)
        acc["lateral_sum"] += (cross_track(rel, self._v3_rel0) * on_line.float()).sum()
        acc["lateral_n"] += on_line.float().sum()
        acc["short_sum"] += (short * alive.float()).sum()
        acc["short_n"] += alive.float().sum()
        return terminated

    unwrapped_env._compute_dones_and_reward = types.MethodType(_compute_dones_and_reward, unwrapped_env)
    return cfg
