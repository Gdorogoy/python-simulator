import numpy as np
from scipy.spatial.transform import Rotation

from app.dynamics.methods import mixer_inversion
from app.control.pid_hover import PIDHoverController

"""Reward functions and curriculum chaining (phase_0 -> base) for the hover/approach task."""

class RewardConfig:
    def __init__(self, oob_radius, drift_radius=None, hit_threshold=0.3,
                 streak_cap=30, attitude_roll_deg=65, attitude_pitch_deg=80,
                 hit_reward=5, attitude_penalty=-5, oob_penalty=-6,
                 streak_penalty_coef=-0.02, hover_success_steps=None,
                 inner_dist=0.15,
                 outer_dist=0.3,
                 tilt_penalty_coef=0.1,
                 ang_vel_penalty_coef=0.1,
                 phase0_pos_coef=0.5,
                 phase0_duration_steps=None,
                 rpm_penalty_coef=0.3,
                 imitation_coef=0.5,
                 imitation_duration_steps=None,
                 zone_bonus=0.2,
                 vel_penalty_coef=0.1,
                 vel_penalty_cap=4.0,
                 ang_vel_penalty_cap=4.0,
                 dist_penalty_coef=1.15,
                 pos_term_dist_coef=0.5,
                 approach_gain=1.25,
                 step_penalty=0.01,
                 closer_bonus_val=0.01,
                 ):

        self.oob_radius = oob_radius
        self.drift_radius = drift_radius
        self.hit_threshold = hit_threshold
        self.streak_cap = streak_cap
        self.attitude_roll_deg = attitude_roll_deg
        self.attitude_pitch_deg = attitude_pitch_deg
        self.hit_reward = hit_reward
        self.attitude_penalty = attitude_penalty
        self.oob_penalty = oob_penalty
        self.streak_penalty_coef = streak_penalty_coef
        self.hover_success_steps = hover_success_steps
        self.hover_steps_in_zone = 0
        self.inner_dist = inner_dist
        self.outer_dist = outer_dist
        self.tilt_penalty_coef = tilt_penalty_coef
        self.ang_vel_penalty_coef = ang_vel_penalty_coef
        self.phase0_pos_coef = phase0_pos_coef
        self.phase0_duration_steps = phase0_duration_steps
        self.rpm_penalty_coef = rpm_penalty_coef

        self.imitation_coef=imitation_coef
        self.imitation_duration_steps=imitation_duration_steps

        # Reward-shaping magic numbers, now tunable instead of hardcoded in the fns below.
        self.zone_bonus = zone_bonus
        self.vel_penalty_coef = vel_penalty_coef
        self.vel_penalty_cap = vel_penalty_cap
        self.ang_vel_penalty_cap = ang_vel_penalty_cap
        self.dist_penalty_coef = dist_penalty_coef
        self.pos_term_dist_coef = pos_term_dist_coef
        self.approach_gain = approach_gain
        self.step_penalty = step_penalty
        self.closer_bonus_val = closer_bonus_val

        self.hit_streak=0






def _kinematics(env):
    pos = np.array([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
    vel = np.array([env.drone_state.velocity.x, env.drone_state.velocity.y, env.drone_state.velocity.z])
    ang_vel = np.array([env.drone_state.angular_velocity.x, env.drone_state.angular_velocity.y,
                         env.drone_state.angular_velocity.z])
    rot = Rotation.from_quat([env.drone_state.orientation.x, env.drone_state.orientation.y,
                               env.drone_state.orientation.z, env.drone_state.orientation.w])

    roll, pitch, yaw = rot.as_euler("xyz")
    dist = np.linalg.norm(env.target_pos - pos)

    return pos, vel, ang_vel, roll, pitch, yaw, dist


def _check_hit(cfg, dist):
    """Returns (hit_reward, True, "Hit") once dist closes under cfg.hit_threshold, else None.
    Shared by every stage/phase so "Hit" stays reachable once curricula advance past
    the stage that would otherwise be the only one checking it."""
    if cfg.hit_threshold is not None and dist < cfg.hit_threshold:
        return cfg.hit_reward, True, "Hit"
    return None


def _terminal_checks(cfg, env, pos, roll, pitch):
    """Returns (reward, terminated, reason) if a hard-failure condition is hit, else None."""
    if np.any(np.isnan(pos)) or pos[2] < 0.0 or np.linalg.norm(pos) > cfg.oob_radius:
        return cfg.oob_penalty, True, "oob"

    if cfg.drift_radius is not None and np.linalg.norm(pos) < cfg.drift_radius:
        return cfg.oob_penalty, True, "drift"

    if abs(roll) > np.radians(cfg.attitude_roll_deg):
        return cfg.attitude_penalty, True, "attitude-ROLL"

    if abs(pitch) > np.radians(cfg.attitude_pitch_deg):
        return cfg.attitude_penalty, True, "attitude-PITCH"

    return None


def chain_reward_fns(phases:list[object]):
    """phases is an ordered list of (reward_fn, duration_steps) tuples; duration_steps
    counts env steps before permanently advancing to the next phase. Last phase's
    duration should be None (runs indefinitely)."""
    state = {"step": 0, "idx": 0}

    def chained_fn(env):
        state["step"] += 1
        fn, duration = phases[state["idx"]]
        if duration is not None and state["step"] > duration and state["idx"] < len(phases) - 1:
            state["idx"] += 1
            fn, duration = phases[state["idx"]]

        return fn(env)

    return chained_fn


def base_reward_fn(cfg, env):
    """Base hover/approach reward. Factored out of make_reward_fn so it can also
    serve as the terminal stage of other curricula (e.g. RewardFnPhase1)."""
    pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

    bonus=0.0

    terminal = _terminal_checks(cfg, env, pos, roll, pitch)
    if terminal is not None:
        return terminal

    hit = _check_hit(cfg, dist)
    if hit is not None:
        return hit

    if dist < cfg.outer_dist:
        env.moving_away_streak = 0
        env.hover_steps_in_zone += 1

        if cfg.hover_success_steps is not None and env.hover_steps_in_zone == cfg.hover_success_steps:
            env.hover_success_achieved = True
            bonus = cfg.hit_reward  # one-time bonus, doesn't terminate
        else:
            bonus = 0.0

        tilt = abs(roll) + abs(pitch)

        # Per-step hover-quality score: reward for being in-zone, penalized by
        # speed, distance, tilt, and angular velocity (each capped to limit
        # how negative a single term can push the reward).
        stability_term = (cfg.zone_bonus - cfg.vel_penalty_coef * min(np.linalg.norm(vel), cfg.vel_penalty_cap)
                           - cfg.dist_penalty_coef * dist
                           - cfg.tilt_penalty_coef * tilt
                           - cfg.ang_vel_penalty_coef * min(np.linalg.norm(ang_vel), cfg.ang_vel_penalty_cap))

        pos_term = cfg.zone_bonus - cfg.pos_term_dist_coef * dist

        diff = env.prev_distance - dist
        approach_term = diff * cfg.approach_gain - cfg.step_penalty if diff < 0 else diff - cfg.step_penalty

        # Weighted blend of pos_term/approach_term, deeper into the zone favoring pos_term.
        blend = np.clip((cfg.outer_dist - dist) / (cfg.outer_dist - cfg.inner_dist), 0.0, 1.0)
        progress = blend * pos_term + (1 - blend) * approach_term + stability_term

        closer_bonus = cfg.closer_bonus_val if diff > 0 else 0.0

    else:
        env.hover_steps_in_zone = 0
        diff = env.prev_distance - dist
        if diff < 0:
            env.moving_away_streak += 1
            progress = diff * cfg.approach_gain - cfg.step_penalty
        else:
            env.moving_away_streak = 0
            progress = diff - cfg.step_penalty
        closer_bonus = cfg.closer_bonus_val if diff > 0 else 0.0

    streak_penalty = cfg.streak_penalty_coef * env.moving_away_streak
    env.prev_distance = dist

    if env.moving_away_streak >= cfg.streak_cap:
        return cfg.oob_penalty, True, "moving_away_cap"

    return progress + streak_penalty + closer_bonus+bonus , False, "running"


def make_reward_fn(cfg: RewardConfig):

    def base_fn(env):
        return base_reward_fn(cfg, env)

    def phase_0_fn(env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(cfg, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        diff = env.prev_distance - dist

        # Known-correct hover RPMs for this episode's mass, used to reward matching them.
        hover_thrust = env.config.mass * 9.81
        hover_rpm = np.array(mixer_inversion(env.config, [hover_thrust, 0.0, 0.0, 0.0]))
        current_rpm = np.array(env.drone_state.rotor_rpm)
        rpm_deviation = np.linalg.norm(current_rpm - hover_rpm) / env.config.rotors[0].max_rpm  # normalized ~[0,1]

        if dist < cfg.outer_dist:
            env.moving_away_streak = 0
            approach_term = cfg.zone_bonus - (cfg.dist_penalty_coef * cfg.phase0_pos_coef * dist) - cfg.rpm_penalty_coef * rpm_deviation
        else:
            env.moving_away_streak = env.moving_away_streak + 1 if diff < 0 else 0
            approach_term = (diff * cfg.approach_gain - cfg.step_penalty) if diff < 0 else (diff - cfg.step_penalty)
            approach_term -= cfg.rpm_penalty_coef * 0.5 * rpm_deviation

        closer_bonus = cfg.closer_bonus_val if diff > 0 else 0.0
        streak_penalty = cfg.streak_penalty_coef * env.moving_away_streak
        env.prev_distance = dist

        if env.moving_away_streak >= cfg.streak_cap:
            return cfg.oob_penalty, True, "moving_away_cap"

        return approach_term + streak_penalty + closer_bonus, False, "running"

    def phase_imitation_fn(env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(cfg, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        diff = env.prev_distance - dist
        if dist < cfg.outer_dist:
            env.moving_away_streak = 0
            approach_term = cfg.zone_bonus - (cfg.dist_penalty_coef * cfg.phase0_pos_coef * dist)
        else:
            env.moving_away_streak = env.moving_away_streak + 1 if diff < 0 else 0
            approach_term = (diff * cfg.approach_gain - cfg.step_penalty) if diff < 0 else (diff - cfg.step_penalty)

        closer_bonus = cfg.closer_bonus_val if diff > 0 else 0.0
        streak_penalty = cfg.streak_penalty_coef * env.moving_away_streak
        env.prev_distance = dist

        if env.moving_away_streak >= cfg.streak_cap:
            return cfg.oob_penalty, True, "moving_away_cap"

        teacher_action = env.pid_teacher.compute_action(env.drone_state, env.target_pos)
        action_range = env.action_space.high - env.action_space.low
        normalized_diff = np.linalg.norm((env.last_raw_action - teacher_action) / action_range)
        imitation_term = -cfg.imitation_coef * normalized_diff

        return approach_term + streak_penalty + closer_bonus + imitation_term, False, "running"

    phases = []
    if cfg.imitation_duration_steps is not None:
        phases.append((phase_imitation_fn, cfg.imitation_duration_steps))
    if cfg.phase0_duration_steps is not None:
        phases.append((phase_0_fn, cfg.phase0_duration_steps))
    phases.append((base_fn, None))

    if len(phases) == 1:
        return phases[0][0]
    return chain_reward_fns(phases)







