<<<<<<< HEAD
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
                 streak_cap=30, attitude_roll_deg=120, attitude_pitch_deg=120,
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

        bonus=0.0

        terminal = _terminal_checks(cfg, env, pos, roll, pitch)
        if terminal is not None:
            return terminal

        # if cfg.hover_success_steps is not None and env.hover_steps_in_zone >= cfg.hover_success_steps:
        #     return cfg.hit_reward, True, "hover_success"

        # ----- single in-zone / outside-zone branch -----

        if dist < cfg.outer_dist:
            # resetting the moving away streak each time the drone is back in the zone and adding to the in zone streak
            env.moving_away_streak = 0
            env.hover_steps_in_zone += 1

            if cfg.hover_success_steps is not None and env.hover_steps_in_zone == cfg.hover_success_steps:
                env.hover_success_achieved = True
                bonus = cfg.hit_reward  # one-time bonus, doesn't terminate
            else:
                bonus = 0.0



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

        return progress + streak_penalty + closer_bonus+bonus , False, "running"

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







=======
import numpy as np
from scipy.spatial.transform import Rotation

from app.control.step_budget import steps_for_dist

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


def chain_reward_fns(fns: list, n: int):
    """fns is an ordered list of plain reward functions (each fn(env) ->
    (reward, terminated, reason)); each one runs for exactly n env-steps
    before permanently advancing to the next -- the LAST function in the
    list runs indefinitely once reached (no duration to advance past).
    state["step"] resets on every advance so each function gets its own
    full n steps, not n minus whatever the global step count already
    accumulated on prior functions."""
    state = {"step": 0, "idx": 0}

    def chained_fn(env):
        state["step"] += 1
        if state["step"] > n and state["idx"] < len(fns) - 1:
            state["idx"] += 1
            state["step"] = 1

        return fns[state["idx"]](env)

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

        teacher_action = env.pid_teacher.compute_action(env.drone_state, env.target_pos, env.target_yaw)
        action_range = env.action_space.high - env.action_space.low
        normalized_diff = np.linalg.norm((env.last_raw_action - teacher_action) / action_range)
        imitation_term = -cfg.imitation_coef * normalized_diff

        return approach_term + streak_penalty + closer_bonus + imitation_term , False, "running"

    fns = []
    if cfg.imitation_duration_steps is not None:
        fns.append(phase_imitation_fn)
    fns.append(base_fn)

    if len(fns) == 1:
        return fns[0]
    return chain_reward_fns(fns, cfg.imitation_duration_steps)


# reward_func -- flat, no phases, no config class, every knob a module constant.
# Potential-based approach shaping (Ng et al.): F(s,s')=gamma*phi(s')-phi(s),
# phi = negative L1 distance to target normalized by start_dist.
OOB_RADIUS = 30.0
ATTITUDE_ROLL_DEG = 65
ATTITUDE_PITCH_DEG = 80
ATTITUDE_PENALTY = -1.0
OOB_PENALTY = -1.5
HIT_THRESHOLD = 0.25

# Was 1000 -- 100-1000x every other term, which produced heavy-tailed advantages
# that blew up KL divergence and collapsed training. See PROJECT_DEFENSE_GUIDE.md Part 2.
HIT_REWARD = 50

TARGET_FRACTION = 0.25
# Measured via the tuned PID at dist=3,10 -- rerun app.control.tune_pid.calibrate_approach_milestone_budget()
# and update this if best_pid_gains_per_dist.json, HIT_REWARD, or this reward changes.
APPROACH_MILESTONE_BUDGET = 96.72

MILESTONE_FRACS = (0.25, 0.5, 0.75)
MILESTONE_BONUSES = (10.0, 15.0, 20.0)

# Anti-oscillation terms (2026-09-19) -- see PROJECT_DEFENSE_GUIDE.md Part 2.4.1
# for the full derivation and worked best/avg/worst-case numbers.
OUTER_ZONE_RADIUS = 1.0
INNER_ZONE_RADIUS = 0.5
OUTER_ZONE_EXIT_PENALTY = -1.0
INNER_ZONE_EXIT_PENALTY = -2.0
STABILITY_COEF = 0.1  # applied inside OUTER_ZONE_RADIUS: -STABILITY_COEF*(|vel|+|roll|+|pitch|)

# PPO's own discount factor, exported so base_training.PARAMS can't drift from it.
GAMMA = 0.97


def terminal_checks(pos, roll, pitch, oob_radius=OOB_RADIUS):
    """Evaluates every hard-failure condition (not just the first one tripped)
    and returns (penalty_sum, term_reason, active_reasons): term_reason/
    active_reasons are empty/None while non-terminal (episode continues);
    otherwise every condition that tripped THIS step is summed into
    penalty_sum and listed (e.g. an oob position that's also over the
    attitude limit reports both, instead of only whichever check happened
    to be evaluated first).

    `oob_radius` defaults to the flat module constant (the original Uniform(3,10)
    task) but reward_func passes a distance-scaled value instead -- a fixed 30m
    radius makes any target past ~10m structurally unreachable (the drone gets
    flagged oob leaving the sphere long before reaching the target), which isn't
    a PID/policy failure, just a mismatched radius."""
    penalty_sum = 0.0
    active_reasons = []

    if np.any(np.isnan(pos)) or pos[2] < 0.0 or np.linalg.norm(pos) > oob_radius:
        penalty_sum += OOB_PENALTY
        active_reasons.append("oob")
    if abs(roll) > np.radians(ATTITUDE_ROLL_DEG):
        penalty_sum += ATTITUDE_PENALTY
        active_reasons.append("attitude-ROLL")
    if abs(pitch) > np.radians(ATTITUDE_PITCH_DEG):
        penalty_sum += ATTITUDE_PENALTY
        active_reasons.append("attitude-PITCH")

    term_reason = "+".join(active_reasons) if active_reasons else None
    return penalty_sum, term_reason, active_reasons


def hit_target(dist):
    return dist < HIT_THRESHOLD


def milestone_bonus(env, dist, start_dist):
    """Fires each of MILESTONE_FRACS' bonuses once per episode, the first
    time progress (fraction of start_dist closed) crosses it. env.milestones_hit
    (a set, reset in BaseDroneEnv.reset()) tracks which have already fired."""
    progress = 1.0 - dist / start_dist
    bonus = 0.0
    for frac, val in zip(MILESTONE_FRACS, MILESTONE_BONUSES):
        if progress >= frac and frac not in env.milestones_hit:
            env.milestones_hit.add(frac)
            bonus += val
    return bonus


def reward_func(env):
    pos, vel, _ang_vel, roll, pitch, _yaw, dist = _kinematics(env)

    # oob_radius scales with this episode's start_dist so a far target isn't structurally unreachable.
    oob_radius = max(OOB_RADIUS, env.start_dist * 3.0)
    term_sum, term_reason, _term_list = terminal_checks(pos, roll, pitch, oob_radius=oob_radius)
    if term_reason:
        return term_sum, True, term_reason

    if hit_target(dist):
        return HIT_REWARD, True, "Hit"

    # Plain phi_now-phi_prev, not GAMMA*phi_now-phi_prev -- the gamma-scaled form leaves
    # a small positive reward for standing still, undermining step_penalty (see rewards.py history).
    diff = env.target_pos - pos
    prev_diff = env.target_pos - env.prev_position
    phi_now = -np.sum(np.abs(diff)) / env.start_dist
    phi_prev = -np.sum(np.abs(prev_diff)) / env.start_dist
    reward = phi_now - phi_prev

    step_penalty = -(TARGET_FRACTION * APPROACH_MILESTONE_BUDGET) / steps_for_dist(env.start_dist)
    reward += step_penalty

    reward += milestone_bonus(env, dist, env.start_dist)

    # Stability term: costs more to fly fast/tilted once inside the outer zone.
    if dist < OUTER_ZONE_RADIUS:
        tilt = abs(roll) + abs(pitch)
        reward -= STABILITY_COEF * (np.linalg.norm(vel) + tilt)

    # Zone-exit penalty: retreating out of a zone entered last step is strictly worse than neutral.
    prev_dist = env.prev_distance
    was_in_inner = prev_dist < INNER_ZONE_RADIUS
    was_in_outer = prev_dist < OUTER_ZONE_RADIUS
    now_in_inner = dist < INNER_ZONE_RADIUS
    now_in_outer = dist < OUTER_ZONE_RADIUS
    if was_in_inner and not now_in_inner:
        reward += INNER_ZONE_EXIT_PENALTY
    elif was_in_outer and not now_in_outer:
        reward += OUTER_ZONE_EXIT_PENALTY

    env.prev_position = pos.copy()
    env.prev_distance = dist

    return reward, False, None







>>>>>>> later_to_remove
