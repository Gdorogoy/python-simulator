from app.reward_functions.rewards import RewardConfig, _kinematics, _terminal_checks, _check_hit, chain_reward_fns, base_reward_fn
import numpy as np
"""Phase-1 curriculum reward. Inherits RewardConfig for shared tuning knobs
(thresholds, streak settings, shaping coefficients) instead of a separate config class."""


class RewardFnPhase1(RewardConfig):
    """Approach + hover-thrust shaping, chainable: usable directly as
    InterceptorDroneEnv's reward_fn, or chained into a curriculum via .chain(...)."""


    def __init__(self,hit_steps_streak,phase1_pos_coef,hit_reward,
                 warmup_duration_steps=10_000,phase1_duration_steps=200_000,
                 velocity_gain=0.1,**kwargs):
        super().__init__(**kwargs)
        self.hit_steps_streak=hit_steps_streak
        self.phase1_pos_coef=phase1_pos_coef
        self.hit_reward=hit_reward
        self.warmup_duration_steps=warmup_duration_steps
        self.phase1_duration_steps=phase1_duration_steps
        self.velocity_gain=velocity_gain


    def _check_hit(self, dist):
        return _check_hit(self, dist)

    def _approach_shaping(self, env, dist, pos_coef):
        diff = env.prev_distance - dist
        if dist < self.outer_dist:
            env.moving_away_streak = 0
            approach_term = self.zone_bonus - (self.dist_penalty_coef * pos_coef * dist)
        else:
            env.moving_away_streak = env.moving_away_streak + 1 if diff < 0 else 0
            approach_term = (diff * self.approach_gain - self.step_penalty) if diff < 0 else (diff - self.step_penalty)

        closer_bonus = self.closer_bonus_val if diff > 0 else 0.0
        streak_penalty = self.streak_penalty_coef * env.moving_away_streak
        env.prev_distance = dist

        return approach_term, closer_bonus, streak_penalty

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



    def phase_1_imitation(self, env):
        """Rewards matching the PID teacher's action on top of approach shaping."""
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

    def phase_1_fn(self, env):
        """Approach shaping plus a bonus for closing velocity toward the target."""
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(self, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        hit = self._check_hit(dist)
        if hit is not None:
            env.moving_away_streak = 0
            return hit

        approach_term, closer_bonus, streak_penalty = self._approach_shaping(env, dist, self.phase1_pos_coef)

        to_target = env.target_pos - pos
        closing_speed = float(np.dot(vel, to_target) / max(dist, 1e-6))
        velocity_term = self.velocity_gain * max(closing_speed, 0.0)

        if env.moving_away_streak >= self.streak_cap:
            return self.oob_penalty, True, "moving_away_cap"

        return approach_term + streak_penalty + closer_bonus + velocity_term, False, "running"


    def as_roadmap(self):
        """Chains warmup -> phase_1_imitation -> phase_1_fn -> base_fn, each active
        for its own duration_steps before permanently advancing to the next."""
        return chain_reward_fns([
            (self.warmup, self.warmup_duration_steps),
            (self.phase_1_imitation, self.imitation_duration_steps),
            (self.phase_1_fn, self.phase1_duration_steps),
            (lambda env: base_reward_fn(self, env), None),
        ])

    def chain(self, *phases, duration_steps=None):
        """Chains this roadmap in front of subsequent (reward_fn, duration_steps) phases.
        duration_steps defaults to self.phase0_duration_steps if not given."""
        own_duration = duration_steps if duration_steps is not None else self.phase0_duration_steps
        return chain_reward_fns([(self.as_roadmap(), own_duration), *phases])
