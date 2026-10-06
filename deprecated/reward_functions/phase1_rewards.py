"""
Phase 1.0-1.3 reward methods
for more info training-goals.md
"""

import numpy as np

from app.reward_functions.rewards import RewardConfig, _kinematics, _terminal_checks, chain_reward_fns


class Phase1Config(RewardConfig):
    def __init__(self, *args,
                 axis_pos_coef=1.15,
                 axis_penalty_coef=0.1,
                 hit_streak_target=None,
                 hit_streak_bonus=2.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.axis_pos_coef = axis_pos_coef
        self.axis_penalty_coef = axis_penalty_coef
        self.hit_streak_target = hit_streak_target
        self.hit_streak_bonus = hit_streak_bonus


def make_axis_fn(cfg: Phase1Config, axis: int, sign: int, name: str):
    """
    Builds a single reward fn that rewards moving along `axis` (0=x, 1=y, 2=z) in the
    `sign` direction (+1/-1) towards env.target_pos. Same terminal-check / streak-penalty
    skeleton as phase_0_fn / base_fn in rewards.py, just axis-scoped instead of full 3D dist.

    Direction itself comes from whichever env.target_pos the training script sets for this
    phase (e.g. (-3,0,0.5) for left). `axis`/`sign` are used to scope TWO things to the axis
    actually being trained:
      - the off-axis drift penalty (drifting on the other two axes costs extra)
      - the approach/moving-away progress signal itself, via axis-scoped distance rather
        than full 3D dist - otherwise off-axis drift gets double-counted as "moving away"
        (it already costs axis_penalty) and can trip moving_away_streak/streak_cap even
        while the drone is making real progress on the trained axis, killing episodes
        before the policy gets credit for the direction it's actually supposed to learn.
    """

    other_axes = [i for i in range(3) if i != axis]

    def axis_fn(env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(cfg, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        env.hit_streak = getattr(env, "hit_streak", 0)

        # off-axis drift: distance from target on the two axes NOT being trained this phase
        off_axis_error = np.linalg.norm([env.target_pos[i] - pos[i] for i in other_axes])
        axis_penalty = cfg.axis_penalty_coef * off_axis_error

        # progress measured along the trained axis only (env.steps_elapsed == 1 marks the
        # first reward call of a fresh episode, since InterceptorDroneEnv.step() increments
        # it before calling the reward fn - re-seed prev_axis_dist there instead of relying
        # on env.reset() to know about this phase-1-only field)
        axis_dist = abs(env.target_pos[axis] - pos[axis])
        if env.steps_elapsed <= 1:
            env.prev_axis_dist = axis_dist
        diff = env.prev_axis_dist - axis_dist
        env.prev_axis_dist = axis_dist

        # if the drone is close
        if dist < cfg.outer_dist:
            env.moving_away_streak = 0
            approach_term = diff * cfg.axis_pos_coef - 0.01 if diff < 0 else diff - 0.01

            if dist < cfg.hit_threshold:
                env.hit_streak += 1
                bonus = (cfg.hit_streak_bonus
                         if cfg.hit_streak_target is not None and env.hit_streak >= cfg.hit_streak_target
                         else 0.0)
                env.prev_distance = dist
                return cfg.hit_reward + bonus, True, "target_hit"

        # if the drone is moving away (along the trained axis)
        else:
            env.hit_streak = 0
            env.moving_away_streak = env.moving_away_streak + 1 if diff < 0 else 0
            approach_term = diff * cfg.axis_pos_coef - 0.01 if diff < 0 else diff - 0.01

        closer_bonus = 0.1 if diff > 0 else 0.0
        streak_penalty = cfg.streak_penalty_coef * env.moving_away_streak
        env.prev_distance = dist

        if env.moving_away_streak >= cfg.streak_cap:
            return cfg.oob_penalty, True, "moving_away_cap"

        return approach_term + streak_penalty + closer_bonus - axis_penalty, False, f"running-{name}"

    return axis_fn


def make_phase1_reward_fn(cfg: Phase1Config, axis_duration_steps=None):
    """
    Chains left -> right -> forward -> back, each active for axis_duration_steps
    (or indefinitely if None, e.g. for manual/curriculum-controlled advancement), same
    chaining pattern as make_reward_fn in rewards.py.
    """
    left = make_axis_fn(cfg, axis=1, sign=-1, name="left")
    right = make_axis_fn(cfg, axis=1, sign=1, name="right")
    forward = make_axis_fn(cfg, axis=0, sign=1, name="forward")
    back = make_axis_fn(cfg, axis=0, sign=-1, name="back")

    phases = [
        (left, axis_duration_steps),
        (right, axis_duration_steps),
        (forward, axis_duration_steps),
        (back, None),
    ]

    return chain_reward_fns(phases)
