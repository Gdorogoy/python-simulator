"""Distance curriculum for the acro trainer: parse `--buckets` and size the per-episode time-bonus clock.

Pure Python (no torch, no Isaac), so it can be tested in the uv env. Used by
`app/training/acro_rtl_train_isaac.py` for the `--buckets` spec and the run header; the reward applies the
same clock from `proportion.time_bonus_speed` (`start_distance / speed`, clamped to the leg window), so the
two stay in sync without importing each other.

The sampler lives in `app/environmental/base_drone_env_isaac.py` (`set_target_distance_buckets`), because there is
one Isaac env for both trainers and that is the only place a spawn distance is drawn. The default curriculum is
applied by the acro trainer, not by the env, so the angle trainer keeps its single `--distance-low/--distance-high`
range unless it opts in.
"""

# 25 % short, 35 % medium, 40 % long. The long bucket is the task the speed run was tuned on; the short ones give
# the chase (and the evader) enough episodes per chunk to learn from without a full-length episode every time.
DEFAULT_BUCKETS = "100-200:0.25,500-800:0.35,1000-1500:0.40"


def parse_buckets(spec):
    """'lo-hi:weight,lo-hi:weight,...' -> [(lo, hi, weight), ...]; None for 'none' / ''.

    Weights are optional (`lo-hi` means weight 1) and are normalised by the caller. Raises ValueError with the
    offending item so a typo in a long command fails before Isaac starts.
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if not text or text.lower() in ("none", "off", "0"):
        return None
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            span, _, weight = item.partition(":")
            low_s, _, high_s = span.partition("-")
            lo, hi = float(low_s), float(high_s)
            w = float(weight) if weight else 1.0
        except ValueError:
            raise ValueError(f"bad --buckets item {item!r}; expected lo-hi:weight, e.g. 100-200:0.25") from None
        if not (0.0 < lo < hi):
            raise ValueError(f"bad --buckets item {item!r}: need 0 < low < high")
        if w <= 0.0:
            raise ValueError(f"bad --buckets item {item!r}: weight must be > 0")
        out.append((lo, hi, w))
    if not out:
        raise ValueError(f"--buckets {spec!r} has no ranges")
    return out


def describe_buckets(buckets):
    """One-line human summary for the run header, e.g. '100-200 m 25% | 500-800 m 35% | ...'."""
    total = sum(w for _lo, _hi, w in buckets)
    return " | ".join(f"{lo:g}-{hi:g} m {100.0 * w / total:.0f}%" for lo, hi, w in buckets)


def time_bonus_budget_s(distance, speed, min_s, max_s):
    """Per-episode clock the time bonus is measured against: distance / speed, clamped to [min_s, max_s].

    `speed` is the average speed a hit should beat to earn bonus, not the airframe limit: 30 m/s means a 1200 m
    leg has 40 s of budget (the v4 policy flies it in ~35 s) while a 150 m leg has 5 s, floored at `min_s`.
    With `speed <= 0` the caller's fixed `time_bonus_budget_s` is used instead, so old runs are unchanged.
    """
    if speed <= 0.0:
        return None
    return min(max(float(distance) / float(speed), float(min_s)), float(max_s))
