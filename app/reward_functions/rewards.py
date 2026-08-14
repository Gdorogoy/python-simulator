import numpy as np
from scipy.spatial.transform import Rotation

from app.dynamics.methods import mixer_inversion
from app.control.pid_hover import PIDHoverController

"""
Added on 09/08/26 new class for rewards functions, chaining added on 12/08/26 for
curriculum-style training phases (phase_0 -> base).
"""

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
                 imitation_duration_steps=None
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




def _kinematics(env):

    # getting the position and velocity of the drone
    pos = np.array([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
    vel = np.array([env.drone_state.velocity.x, env.drone_state.velocity.y, env.drone_state.velocity.z])
    # getting the angular velocity and rotation of the drone
    ang_vel = np.array([env.drone_state.angular_velocity.x, env.drone_state.angular_velocity.y,
                         env.drone_state.angular_velocity.z])
    rot = Rotation.from_quat([env.drone_state.orientation.x, env.drone_state.orientation.y,
                               env.drone_state.orientation.z, env.drone_state.orientation.w])

    # getting individual forces of the drone and distance
    roll, pitch, yaw = rot.as_euler("xyz")
    dist = np.linalg.norm(env.target_pos - pos)

    return pos, vel, ang_vel, roll, pitch, yaw, dist


def _terminal_checks(cfg, env, pos, roll, pitch):
    """Returns (reward, terminated, reason) if a hard-failure condition is hit, else None."""
    if np.any(np.isnan(pos)) or pos[2] < 0.0 or np.linalg.norm(pos) > cfg.oob_radius:
        return cfg.oob_penalty, True, "oob"


    # check for drift value
    if cfg.drift_radius is not None and np.linalg.norm(pos) < cfg.drift_radius:
        return cfg.oob_penalty, True, "drift"

    # checking what the attitude crash reason was
    if abs(roll) > np.radians(cfg.attitude_roll_deg):
        return cfg.attitude_penalty, True, "attitude-ROLL"

    if abs(pitch) > np.radians(cfg.attitude_pitch_deg):
        return cfg.attitude_penalty, True, "attitude-PITCH"

    return None


def chain_reward_fns(phases:list[object]):
    """
    chaining: phases is a list of (reward_fn, duration_steps) tuples, in order.
    duration_steps counts env steps across episode resets (total calls made to the chained
    function) that a phase stays active before permanently advancing to the next one.
    The last phase's duration should be None (runs indefinitely).
    """
    state = {"step": 0, "idx": 0}

    def chained_fn(env):
        state["step"] += 1
        fn, duration = phases[state["idx"]]
        if duration is not None and state["step"] > duration and state["idx"] < len(phases) - 1:
            state["idx"] += 1
            fn, duration = phases[state["idx"]]

        return fn(env)

    return chained_fn


def make_reward_fn(cfg: RewardConfig):

    # base reward function check the basic things
    def base_fn(env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(cfg, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        if cfg.hover_success_steps is not None and env.hover_steps_in_zone >= cfg.hover_success_steps:
            return cfg.hit_reward, True, "hover_success"
        # ----- single in-zone / outside-zone branch -----

        if dist < cfg.outer_dist:
            # resetting the moving away streak each time the drone is back in the zone and adding to the in zone streak
            env.moving_away_streak = 0
            env.hover_steps_in_zone += 1

            if cfg.hover_success_steps is not None and env.hover_steps_in_zone >= cfg.hover_success_steps:
                env.hover_success_achieved = True   # telemetry only no reward bonus


            # calculating tilt to find the stable state
            tilt = abs(roll) + abs(pitch)

            """
                per step reward that scores how well the drone is hovering 
                0.2 - reward for begin in the zone
                -0.1 * min(||vel||) , 4.0) - penalizes speed because we need no speed to hover , set to min of the vel and 4 to not get the model in too much negative reward
                -1.15 * dist - penalizes the distance from target the until 0.3 (because then the calculation isnt reached) 
                -cfg.tilt_penalty_coef * tilt - penalizes the tilt so the drone wont hover at an angle
                -cfg.ang_vel_penatly * min(||ang_vel||,4.0) - same as tilt and speed penalty discourages the drone from spinning 
                
            """

            stability_term = (0.2 - 0.1 * min(np.linalg.norm(vel), 4.0) - 1.15 * dist
                               - cfg.tilt_penalty_coef * tilt
                               - cfg.ang_vel_penalty_coef * min(np.linalg.norm(ang_vel), 4.0))

            pos_term=0.2-0.5*dist

            diff = env.prev_distance - dist
            approach_term = diff * 1.25 - 0.01 if diff < 0 else diff - 0.01

            """
                blend is just an wighted average between two formulas of approach term and stability term and they always add up to 1
                and it dosent work so well because it dosent get more stability handed because it only understands stability when its deep in the zone
            """


            blend = np.clip((cfg.outer_dist - dist) / (cfg.outer_dist - cfg.inner_dist), 0.0, 1.0)

            # progress = blend * stability_term + (1 - blend) * approach_term
            progress= blend* pos_term + (1-blend) * approach_term + stability_term

            closer_bonus = 0.01 if diff > 0 else 0.0

        else:
            env.hover_steps_in_zone = 0
            diff = env.prev_distance - dist
            if diff < 0:
                env.moving_away_streak += 1
                progress = diff * 1.25 - 0.01
            else:
                env.moving_away_streak = 0
                progress = diff - 0.01
            closer_bonus = 0.01 if diff > 0 else 0.0

        streak_penalty = cfg.streak_penalty_coef * env.moving_away_streak
        env.prev_distance = dist

        if env.moving_away_streak >= cfg.streak_cap:
            return cfg.oob_penalty, True, "moving_away_cap"

        return progress + streak_penalty + closer_bonus , False, "running"

    def phase_0_fn(env):
        pos, vel, ang_vel, roll, pitch, yaw, dist = _kinematics(env)

        terminal = _terminal_checks(cfg, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        diff = env.prev_distance - dist

        # compute the known-correct hover RPMs for THIS episode's mass
        # (mixer_inversion needs importing into rewards.py, or pass hover_rpm in via env)
        hover_thrust = env.config.mass * 9.81
        hover_rpm = np.array(mixer_inversion(env.config, [hover_thrust, 0.0, 0.0, 0.0]))
        current_rpm = np.array(env.drone_state.rotor_rpm)
        rpm_deviation = np.linalg.norm(current_rpm - hover_rpm) / env.config.rotors[0].max_rpm  # normalized, ~[0, ~1]

        if dist < cfg.outer_dist:
            env.moving_away_streak = 0
            # reward for being close AND for rotors matching hover exactly
            approach_term = 0.2 - (1.15 * cfg.phase0_pos_coef * dist) - cfg.rpm_penalty_coef * rpm_deviation
        else:
            env.moving_away_streak = env.moving_away_streak + 1 if diff < 0 else 0
            approach_term = (diff * 1.25 - 0.01) if diff < 0 else (diff - 0.01)
            # even outside the zone, mildly reward moving toward correct rotor output
            approach_term -= cfg.rpm_penalty_coef * 0.5 * rpm_deviation

        closer_bonus = 0.01 if diff > 0 else 0.0
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
            approach_term = 0.2 - (1.15 * cfg.phase0_pos_coef * dist)
        else:
            env.moving_away_streak = env.moving_away_streak + 1 if diff < 0 else 0
            approach_term = (diff * 1.25 - 0.01) if diff < 0 else (diff - 0.01)

        closer_bonus = 0.01 if diff > 0 else 0.0
        streak_penalty = cfg.streak_penalty_coef * env.moving_away_streak
        env.prev_distance = dist

        if env.moving_away_streak >= cfg.streak_cap:
            return cfg.oob_penalty, True, "moving_away_cap"

        # --- imitation term ---
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







