from app.reward_functions.rewards import RewardConfig, _kinematics, _terminal_checks, _check_hit, chain_reward_fns, base_reward_fn
import numpy as np
"""
Phase-1 curriculum reward. Inherits RewardConfig for all the shared tuning knobs
(oob/drift/attitude thresholds, streak settings, etc.) without needing a separate
config class, and doesn't have to touch knobs it has no use for (e.g. imitation_coef).
"""


class RewardFnPhase1(RewardConfig):
    """
    Approach + hover-thrust shaping, same shape as rewards.py's phase_0_fn but as a
    chainable object: an instance is directly usable as InterceptorDroneEnv's reward_fn
    (env -> (reward, terminated, reason)), and can be chained into a curriculum via .chain(...).
    """


    def __init__(self,hit_steps_streak,phase1_pos_coef,hit_reward,
                 warmup_duration_steps=10_000,phase1_duration_steps=200_000,**kwargs):
        super().__init__(**kwargs)
        self.hit_steps_streak=hit_steps_streak
        self.phase1_pos_coef=phase1_pos_coef
        self.hit_reward=hit_reward
        self.warmup_duration_steps=warmup_duration_steps
        self.phase1_duration_steps=phase1_duration_steps


    # shared hit check so every stage of the roadmap stops the instant the drone
    # gets within hit_threshold of the target, not just the stages that happen
    # to be active when it does
    def _check_hit(self, dist):
        return _check_hit(self, dist)

    # helper method to claculate the approach term and its bonus/penalty
    def _approach_shaping(self, env, dist, pos_coef):
        diff = env.prev_distance - dist
        # if getting closer
        if dist < self.outer_dist:
            env.moving_away_streak = 0
            approach_term = 0.2 - (1.15 * pos_coef * dist)
        else:
            # if moving away
            env.moving_away_streak = env.moving_away_streak + 1 if diff < 0 else 0
            approach_term = (diff * 1.25 - 0.01) if diff < 0 else (diff - 0.01)

        closer_bonus = 0.01 if diff > 0 else 0.0
        streak_penalty = self.streak_penalty_coef * env.moving_away_streak
        env.prev_distance = dist

        return approach_term, closer_bonus, streak_penalty



    # should run for 100-10k +- steps
    def warmup(self, env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(self, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        hit = self._check_hit(dist)
        if hit is not None:
            return hit

        approach_term, closer_bonus, streak_penalty = self._approach_shaping(env, dist, self.phase1_pos_coef)

        if env.moving_away_streak >= self.streak_cap:
            return self.oob_penalty, True, "moving_away_cap"

        return approach_term + streak_penalty + closer_bonus, False, "running"



    """
    Method for rewarding based on how good it is doing good on the PID imitation (aka on the action imitation)
    """
    def phase_1_imitation(self, env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)
        terminal = _terminal_checks(self, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        hit = self._check_hit(dist)
        if hit is not None:
            env.moving_away_streak = 0
            return hit

        approach_term, closer_bonus, streak_penalty = self._approach_shaping(env, dist, self.phase1_pos_coef)

        if env.moving_away_streak >= self.streak_cap:
            return self.oob_penalty, True, "moving_away_cap"

        teacher_action = env.pid_teacher.compute_action(env.drone_state, env.target_pos)
        action_range = env.action_space.high - env.action_space.low
        normalized_diff = np.linalg.norm((env.last_raw_action - teacher_action) / action_range)
        imitation_term = -self.imitation_coef * normalized_diff

        return approach_term + streak_penalty + closer_bonus + imitation_term, False, "running"

    """
    Method for rewarding on hitting the target(or getting close),
    If the model hit the target > 1500times one time bonus,
    The distance for hitting the drone is now 0.05m
    approach term rewards more velocity then the approach
    """
    def phase_1_fn(self, env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(self, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        hit = self._check_hit(dist)
        if hit is not None:
            env.moving_away_streak = 0
            return hit

        approach_term, closer_bonus, streak_penalty = self._approach_shaping(env, dist, self.phase1_pos_coef)

        # reward closing velocity (speed toward the target) rather than just position
        to_target = env.target_pos - pos
        closing_speed = float(np.dot(vel, to_target) / max(dist, 1e-6))
        velocity_term = 0.1 * max(closing_speed, 0.0)

        if env.moving_away_streak >= self.streak_cap:
            return self.oob_penalty, True, "moving_away_cap"

        return approach_term + streak_penalty + closer_bonus + velocity_term, False, "running"


    def as_roadmap(self):
        """
        Builds the full phase-1 roadmap: warmup -> phase_1_imitation -> phase_1_fn -> base_fn,
        each stage active for its n_duration_steps before permanently advancing to the next
        (base_fn, from rewards.py, runs indefinitely once reached). Usage:
        """
        return chain_reward_fns([
            (self.warmup, self.warmup_duration_steps),
            (self.phase_1_imitation, self.imitation_duration_steps),
            (self.phase_1_fn, self.phase1_duration_steps),
            (lambda env: base_reward_fn(self, env), None),
        ])

    def chain(self, *phases, duration_steps=None):
        """
        Chains this phases roadmap in front of subsequent (reward_fn, duration_steps) phases, e.g.:
            reward_fn = RewardFnPhase1(oob_radius=7, hit_steps_streak=1500, phase1_pos_coef=0.5).chain((other_fn, None), duration_steps=500_000)
            env = InterceptorDroneEnv(reward_fn)
        duration_steps defaults to self.phase0_duration_steps if not given explicitly.
        """
        own_duration = duration_steps if duration_steps is not None else self.phase0_duration_steps
        return chain_reward_fns([(self.as_roadmap(), own_duration), *phases])
