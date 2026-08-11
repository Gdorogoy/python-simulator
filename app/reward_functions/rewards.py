import numpy as np
from scipy.spatial.transform import Rotation


"""
Added on 09/08/26 new class for rewards functions ,later will add chaining for more advanced training phases
"""

class RewardConfig:
    def __init__(self, oob_radius, drift_radius=None, hit_threshold=0.3,
                 streak_cap=30, attitude_roll_deg=65, attitude_pitch_deg=80,
                 hit_reward=5, attitude_penalty=-5, oob_penalty=-6,
                 streak_penalty_coef=-0.02,hover_success_steps=None,
                 inner_dist=0.15,
                 outer_dist=0.3
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
        self.inner_dist=inner_dist
        self.outer_dist=outer_dist


def make_reward_fn(cfg: RewardConfig):
    def reward_fn(env):
        pos = np.array([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        dist = np.linalg.norm(env.target_pos - pos)

        # ----- base checks -----
        if np.any(np.isnan(pos)) or pos[2] < 0.0 or np.linalg.norm(pos) > cfg.oob_radius:
            return cfg.oob_penalty, True, "oob"

        if cfg.drift_radius is not None and np.linalg.norm(pos) < cfg.drift_radius:
            return cfg.oob_penalty, True, "drift"

        rot = Rotation.from_quat([env.drone_state.orientation.x, env.drone_state.orientation.y,
                                   env.drone_state.orientation.z, env.drone_state.orientation.w])
        roll, pitch, yaw = rot.as_euler("xyz")

        if abs(roll) > np.radians(cfg.attitude_roll_deg) :
            return cfg.attitude_penalty, True, "attitude-ROLL"

        if abs(pitch) > np.radians(cfg.attitude_pitch_deg):
            return cfg.attitude_penalty, True, "attitude-PITCH"

        # ----- single in-zone / outside-zone branch -----

        
        if dist < cfg.outer_dist:
            env.moving_away_streak=0
            env.hover_steps_in_zone+=1

            if cfg.hover_success_steps is not None and env.hover_steps_in_zone >= cfg.hover_success_steps:
                env.hover_success_achieved = True   # telemetry only, no reward bonus


            vel=np.array([env.drone_state.velocity.x,env.drone_state.velocity.y,env.drone_state.velocity.z,])
            stability_term= 0.2 - 0.1 * min (np.linalg.norm(vel),4.0) - 0.5 *dist

            diff= env.prev_distance- dist
            approach_term= diff * 1.25-0.01 if diff < 0 else diff - 0.01

            blend = np.clip((cfg.outer_dist - dist ) / (cfg.outer_dist - cfg.inner_dist), 0.0,1.0)

            progress= blend* stability_term + ( 1 - blend) * approach_term

            closer_bonus =0.01 if diff > 0 else 0.0

        else:
            env.hover_steps_in_zone=0
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


        return progress + streak_penalty + closer_bonus,  False, "running"

    return reward_fn