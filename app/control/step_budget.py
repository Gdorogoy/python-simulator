"""Per-distance episode step budget, shared by tune_pid.py (settling-time target
for gain derivation) and rewards.py's reward_func (step_penalty scaling)."""

DT = 1 / 240
MIN_EPISODE_SECONDS = 7.5  # floor is settling time, not travel time
MIN_STEPS = int(MIN_EPISODE_SECONDS * 240)
STEPS_PER_METER = 750 / 3   # physics steps (240 Hz) per metre of start distance


def steps_for_dist(target_dist):
    """Episode step budget: grows with distance, floored at MIN_STEPS for final settling."""
    return max(MIN_STEPS, int(STEPS_PER_METER * target_dist))
