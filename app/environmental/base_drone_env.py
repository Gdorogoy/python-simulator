import json
from math import inf
from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from scipy.spatial.transform import Rotation

from app.control.pid import PIDController
from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion, QuadConfig
from app.dynamics.methods import mixer_inversion, timestamp_update
from app.environmental.enviorment import sample_wind_conditions

from app.reward_functions.rewards import RewardConfig, make_reward_fn

#normolization values
POS_SCALE = 15.0 
VEL_SCALE = 10.0 
ANG_VEL_SCALE =20.0
MAX_RPM = 12000.0 
DIST_SCALE = 15.0 

# obs[0:3] layout: "full" = symlog(x, y, z), "height" = (0, 0, symlog(clip(z))). See docs.md "Position modes".
POSITION_MODES = ("full", "height")
HEIGHT_CLIP = 30.0




def _symlog_scale(x, linthresh):
    """Symmetric log scaling: linear near zero, logarithmic beyond linthresh."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    linear = x / linthresh
    log_part = np.sign(x) * (1.0 + np.log(np.maximum(ax / linthresh, 1e-8)))
    return np.where(ax <= linthresh, linear, log_part)


def build_observation(state: QuadState, target_pos: np.ndarray, target_yaw: float = 0.0,
                      position_mode: str = "full") -> np.ndarray:
    rel = target_pos - np.array([state.position.x, state.position.y, state.position.z])
    dist = np.linalg.norm(rel)

    # sin/cos of the yaw error so the +/-pi wraparound is continuous
    _, _, yaw = Rotation.from_quat(
        [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w]
    ).as_euler("xyz")
    yaw_err = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

    if position_mode == "height":
        z_clipped = float(np.clip(state.position.z, 0.0, HEIGHT_CLIP))
        pos_feat = np.array([0.0, 0.0, float(_symlog_scale(z_clipped, POS_SCALE))])
    elif position_mode == "full":
        pos_feat = _symlog_scale([state.position.x, state.position.y, state.position.z], POS_SCALE)
    else:
        raise ValueError(f"position_mode must be one of {POSITION_MODES}, got {position_mode!r}")

    return np.concatenate([
        pos_feat,
        np.array([state.velocity.x, state.velocity.y, state.velocity.z]) / VEL_SCALE,
        [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w],
        np.array([state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]) / ANG_VEL_SCALE,
        np.array(state.rotor_rpm) / MAX_RPM,
        _symlog_scale(rel, DIST_SCALE),
        [_symlog_scale(dist, DIST_SCALE)],
        [np.sin(yaw_err), np.cos(yaw_err)],
    ]).astype(np.float32)


register(
    id='base_drone_env_v0',
    entry_point='app.environmental.base_drone_env:BaseDroneEnv',
)

class BaseDroneEnv(gym.Env):
    # render_mode is accepted for compatibility but ignored (no PyBullet GUI any more)
    metadata = {'render_modes': [], 'render_fps': 30}

    def _get_obs(self):
        return build_observation(self.drone_state, self.target_pos, self.target_yaw, self.position_mode)


    def __init__(self, custom_reward, render_mode=None, pid_gains_path="app/control/best_pid_gains.json",
                 spawn_offset_range=None, target_offset_range=None, target_pairs=None,
                 max_steps=15_000, dt=1 / 240, position_mode="full"):

        self.dt = dt
        self.max_steps = max_steps
        if position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}, got {position_mode!r}")
        self.position_mode = position_mode
        self.spawn_offset_range = spawn_offset_range
        self.target_offset_range = target_offset_range
        self.target_pairs = target_pairs


        if custom_reward is None:
            raise ValueError("No custom_reward provided")

        self.reward_method= custom_reward

        self.render_mode = render_mode  # accepted, unused -- see class docstring note above

        self.reset()

        hover_thrust=self.config.mass*9.81


        # +-0.5 N*m torque bounds are a deliberate comfort limit, not physics (docs.md "Action space")
        self.action_space= spaces.Box(
            low=np.array([-hover_thrust, -0.5, -0.5, -0.5], dtype=np.float32),
            high=np.array([hover_thrust,0.5, 0.5, 0.5], dtype=np.float32),
        )

        self.observation_space= spaces.Box(
            low=-inf,
            high=inf,
            shape=(23,),
            dtype=np.float32,

        )

        self.pid_teacher=None

        if pid_gains_path  is not None:
            try:
                with open(pid_gains_path) as f:
                    gains = json.load(f)
                self.pid_teacher = PIDController(**gains)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"[warn] could not load PID gains from {pid_gains_path} pid_teacher unavailable")

    def reset(self, start_pos=None, target_pos=None, target_yaw=None, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if start_pos is None and target_pos is None and self.target_pairs:
            idx = self.np_random.integers(len(self.target_pairs))
            pair_start, pair_target, pair_yaw = self.target_pairs[idx]
            start_pos = pair_start.copy()
            target_pos = pair_target.copy()
            if target_yaw is None:
                target_yaw = pair_yaw
        if start_pos is None:
            start_pos = np.array([0, 0, 5], dtype=np.float32)
            if self.spawn_offset_range is not None:
                low, high = self.spawn_offset_range
                start_pos[0] += self.np_random.uniform(low, high)
        if target_pos is None:
            if hasattr(self, "target_pos"):
                # keep a target_pos set directly on the env across resets
                target_pos = self.target_pos
            else:
                target_pos = np.array([0, 0, 5], dtype=np.float32)
                if self.target_offset_range is not None:
                    low, high = self.target_offset_range
                    target_pos[0] += self.np_random.uniform(low, high)

        self.target_pos = target_pos
        # yaw goal defaults to 0 when the caller passes none
        self.target_yaw = float(target_yaw) if target_yaw is not None else 0.0

        self.moving_away_streak = 0
        self.hover_steps_in_zone = 0
        self.last_raw_action = np.zeros(4, dtype=np.float32)
        self.hover_success_achieved = False
        self.hit_time_sec = None

        if getattr(self, "pid_teacher", None) is not None:
            self.pid_teacher.reset()

        # self.wind_vector , self.mass_scale = sample_wind_conditions(np_rand=self.np_random)
        self.wind_vector=[0,0,0]
        self.mass_scale=1

        self.config = create_quad_config(
            mass=1.5 * self.mass_scale,
            inertia=(0.02 * self.mass_scale, 0.02 * self.mass_scale, 0.04 * self.mass_scale),
            arm_length=0.22,
            drag_coeff=0.035,
            max_rpm=12000,
            motor_tau=0.05,
        )

        start_position = Vector3D(start_pos[0],start_pos[1],start_pos[2])

        hover_thrust = self.config.mass * 9.81
        hover_omega = mixer_inversion(self.config, [hover_thrust, 0.0, 0.0, 0.0])

        self.hover_rpm_target = np.array(hover_omega)

        self.drone_state =QuadState(
            position=start_position,
            velocity=Vector3D(0, 0, 0),
            orientation=Quaternion(0, 0, 0, 1),
            angular_velocity=Vector3D(0, 0, 0),
            rotor_rpm=list(hover_omega),
        )

        self.steps_elapsed = 0


        #TODO: NEED DEPRECATION AND FIX IMMODERATELY!!!!
        self.prev_distance = float(np.linalg.norm(self.target_pos - start_pos))
        # start_dist / prev_position feed the potential-based shaping in rewards.reward_func
        self.start_dist = self.prev_distance
        self.prev_position = np.array(start_pos, dtype=np.float32).copy()
        self.milestones_hit = set()


        obs = self._get_obs()
        return obs, {}



    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.last_raw_action = action.copy()

        hover_thrust = self.config.mass * 9.81
        real_action = action.copy()
        real_action[0] = action[0] + hover_thrust

        self.drone_state = timestamp_update(self.drone_state, self.config, list(real_action), self.wind_vector, self.dt)
        self.steps_elapsed += 1

        obs = self._get_obs()
        reward, terminated, reason = self.reward_method(self)

        if self.hit_time_sec is None and (reason == "Hit" or self.hover_success_achieved):
            self.hit_time_sec = self.steps_elapsed * self.dt

        truncated = self.steps_elapsed >= self.max_steps

        return obs, reward, terminated, truncated, {"reason": reason, "hit_time_sec": self.hit_time_sec}


def my_check_env():
    reward_cfg = RewardConfig(oob_radius=7, hit_reward=10, attitude_penalty=-7, oob_penalty=-10,
                              streak_penalty_coef=-0.05)
    reward_fn = make_reward_fn(reward_cfg)

    env1 = BaseDroneEnv(reward_fn)
    obs1, _ = env1.reset(seed=42)
    print("run1:", env1.wind_vector, env1.mass_scale)

    obs2, _ = env1.reset(seed=42)
    print("run2:", env1.wind_vector, env1.mass_scale)

    env=gym.make('base_drone_env_v0',render_mode=None)

    check_env(env.unwrapped)



def my_test():
    reward_cfg = RewardConfig(oob_radius=7, hit_reward=10, attitude_penalty=-7, oob_penalty=-10,
                              streak_penalty_coef=-0.05)
    reward_fn = make_reward_fn(reward_cfg)

    env = BaseDroneEnv(reward_fn)
    obs, _ = env.reset()
    for i in range(5000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if i % 500 == 0:
            print(i, reward, terminated, truncated)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    print(f"Running RL|INTERCEPTOR DRONE|")
    inp = 0

    while inp not in (1, 2, 3):
        print(f"==========================\n"
              f"Enter 1 to start test \n"
              f"Enter 2 to start env check \n"
              f"==========================\n")
        inp = int(input("Please enter your choice: "))

    if inp == 1:
        my_test()
    elif inp == 2:
        my_check_env()
