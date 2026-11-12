"""Per-leg reward proportions: one leg = one unit of reward, whatever the distance.

The absolute reward numbers of every reward in this package (c_step 0.002, r_hit 10, r_crash 7, ...) were tuned on
one curriculum stage and silently change meaning at another: a 250 m leg and a 1500 m leg both paid r_hit 10 and
r_crash 7, so crashing was 30x cheaper in "per-metre" terms on the long leg, and the step cost either dominated
(long leg) or vanished (short leg). This module rescales the config from per-leg totals instead:

    N(D) = clamp(D / (reference_speed * dt), min_leg_s / dt, max_leg_s / dt)   # steps in one leg at the reference speed

* per-step coefficients (c_step, c_hover, los_k, closing_k, ...) become ``*_leg / N``, so a term that pays its
  coefficient on every step of a flown leg totals exactly ``*_leg``;
* terminal / one-off terms (r_hit, r_crash, r_hover, r_timeout, r_near, ...) become the ``*_leg`` value itself,
  because they are paid once;
* a coefficient that is 0 on the config stays 0 ("0 = off"), so optional shaping terms stay optional.

"Attempt beats giving up" is enforced by the defaults: a hit pays ``hit_leg`` (50), a failed attempt costs
``step_leg + miss_leg`` (5 + 30 = 35), and every way of giving up (hover, timeout, crash, near-miss, caught) costs
at least 45. Attempting therefore always beats parking, and a hit beats everything.

The scaling runs in the ``install_*`` glue only (once per env step, from the live reference distance), never inside
the pure ``compute_*`` functions: direct calls keep the exact numbers in the config, which is also what the unit
tests rely on. ``auto_scale=False`` returns the config untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class RewardProportionMixin:
    """Fields shared by every config in this package. See the module docstring for the method."""

    auto_scale: bool = True               # install_* derives the absolute fields below from the *_leg values
    reference_speed: float = 55.0         # m/s, the physical limit of the acro drone (52-55); sets the length of a leg
    min_leg_s: float = 7.5                # floor, same as step_budget.MIN_EPISODE_SECONDS (settling time, not travel)
    max_leg_s: float = 62.5               # ceiling, the 62.5 s episode at the default step budget
    # the time bonus is paid on a hit and scaled by the fraction of its budget left. 0 = the per-leg budget (N steps at
    # reference_speed), which no real flight ever beats (it starts at rest), so the bonus was always 0; set this to the
    # episode cap (e.g. 62.5) to pay for arriving early instead.
    time_bonus_budget_s: float = 0.0
    # ---- per-step accruals: total over one leg flown at the reference speed ----
    step_leg: float = 5.0                 # c_step: living cost of a whole leg
    hover_accrual_leg: float = 5.0        # c_hover: extra cost of stalling for the whole leg
    away_leg: float = 3.0                 # c_away
    decel_leg: float = 3.0                # c_decel (per m/s of speed lost, so this is a scale, not a hard total)
    tumble_leg: float = 2.0               # c_tumble
    tilt_leg: float = 5.0                 # c_tilt
    progress_leg: float = 10.0            # progress_k (acro_v1 per-axis log potential)
    closing_leg: float = 15.0             # closing_k
    axis_leg: float = 15.0                # axis_k
    los_leg: float = 3.0                  # los_k
    close_speed_leg: float = 10.0         # closing_speed_k
    # ---- one-off terminals and events: paid once, so the value is the leg value ----
    hit_leg: float = 50.0                 # r_hit (+ rc's zone entry, total = hit_leg)
    hover_leg: float = 45.0               # r_hover
    crash_leg: float = 45.0               # r_crash
    timeout_leg: float = 45.0             # r_timeout
    escape_leg: float = 45.0              # r_escape (runner target left its escape sphere = loss)
    near_leg: float = 45.0                # r_near (near-miss timer)
    intercepted_leg: float = 45.0         # r_intercepted (caught by a threat)
    zone_entry_leg: float = 7.0           # r_zone_entry (rc; hit_leg - zone_entry_leg on completion)
    path_leg: float = 0.0                 # r_path_bonus (acro_v4; paid once on a hit, 0 = off)
    # ---- one-off per metre: a pass at the edge of miss_radius costs miss_leg ----
    miss_leg: float = 30.0                # k_miss * d_min; k_miss = miss_leg / miss_radius


_TERMINAL_FIELDS = {
    "r_hit": "hit_leg",
    "r_hover": "hover_leg",
    "r_crash": "crash_leg",
    "r_timeout": "timeout_leg",
    "r_escape": "escape_leg",
    "r_near": "near_leg",
    "r_intercepted": "intercepted_leg",
    "r_zone_entry": "zone_entry_leg",
    "r_path_bonus": "path_leg",
}

_PER_STEP_FIELDS = {
    "c_step": "step_leg",
    "c_hover": "hover_accrual_leg",
    "c_away": "away_leg",
    "c_decel": "decel_leg",
    "c_tumble": "tumble_leg",
    "c_tilt": "tilt_leg",
    "progress_k": "progress_leg",
    "closing_k": "closing_leg",
    "axis_k": "axis_leg",
    "los_k": "los_leg",
    "closing_speed_k": "close_speed_leg",
}


def _is_zero(value) -> bool:
    """True only for a scalar zero. Never a tensor: the guards that use this must not sync the GPU, and a resolved
    tensor that happens to be zero is harmless (multiplying by it still gives 0)."""
    return not torch.is_tensor(value) and float(value) == 0.0


def _fill(dist: torch.Tensor, value) -> torch.Tensor:
    """``torch.full_like(dist, value)`` that also accepts a per-env tensor (the resolved per-step coefficients)."""
    if torch.is_tensor(value):
        return value
    return torch.full_like(dist, float(value))


def _col(value):
    """(N,) -> (N,1) for a per-env coefficient multiplied against an (N,3) tensor. Scalars pass through."""
    return value.unsqueeze(-1) if torch.is_tensor(value) else value


def leg_steps(dist, step_dt: float, cfg: RewardProportionMixin):
    """Steps in one leg of length ``dist`` (metres) at ``cfg.reference_speed``, clamped to the leg floor / ceiling.
    ``dist`` may be a float or an (N,) tensor; the return type follows it."""
    speed = float(cfg.reference_speed) * float(step_dt)
    lo = max(1.0, float(cfg.min_leg_s) / float(step_dt))
    hi = max(lo, float(cfg.max_leg_s) / float(step_dt))
    raw = dist / speed
    if torch.is_tensor(raw):
        return torch.clamp(raw, min=lo, max=hi)
    return min(max(float(raw), lo), hi)


def resolve_proportion(cfg: RewardProportionMixin, ref_dist, step_dt: float) -> RewardProportionMixin:
    """Return a copy of ``cfg`` with the absolute reward fields derived from the per-leg totals at ``ref_dist``.
    ``auto_scale=False`` returns ``cfg`` itself (same object). Fields absent from a config are skipped, and a field
    that is 0 on ``cfg`` stays 0 (a disabled term is never switched on)."""
    if not cfg.auto_scale:
        return cfg
    n = leg_steps(ref_dist, step_dt, cfg)

    changes = {}
    for field, leg_field in _PER_STEP_FIELDS.items():
        if not hasattr(cfg, field):
            continue
        base = getattr(cfg, field)
        if _is_zero(base):
            changes[field] = 0.0
        else:
            changes[field] = getattr(cfg, leg_field) / n
    for field, leg_field in _TERMINAL_FIELDS.items():
        if not hasattr(cfg, field):
            continue
        base = getattr(cfg, field)
        changes[field] = 0.0 if _is_zero(base) else float(getattr(cfg, leg_field))
    if hasattr(cfg, "k_miss") and not _is_zero(cfg.k_miss):
        changes["k_miss"] = float(cfg.miss_leg) / max(float(cfg.miss_radius), 1e-6)
    return replace(cfg, **changes)
