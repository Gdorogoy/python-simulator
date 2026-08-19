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
from sympy.codegen.ast import none

from app.control.pid_hover import PIDHoverController
from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion, QuadConfig
from app.dynamics.methods import mixer_inversion, timestamp_update
from app.environmental.enviorment import sample_wind_conditions, spawn_drone

import time
import pybullet as p
import pybullet_data

from app.reward_functions.rewards import RewardConfig, make_reward_fn

# norm vals TODO:FIX AFTER DUBUG
POS_SCALE = 15.0 #15.0
VEL_SCALE = 10.0 #10.0
ANG_VEL_SCALE =20.0 #20.0
MAX_RPM = 12000.0 #12000.0
DIST_SCALE = 15.0 #15.0




def build_observation(state: QuadState, target_pos: np.ndarray) -> np.ndarray:
    rel = target_pos - np.array([state.position.x, state.position.y, state.position.z])
    dist = np.linalg.norm(rel)

    return np.concatenate([
        np.array([state.position.x, state.position.y, state.position.z]) / POS_SCALE,
        np.array([state.velocity.x, state.velocity.y, state.velocity.z]) / VEL_SCALE,
        [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w],
        np.array([state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]) / ANG_VEL_SCALE,
        np.array(state.rotor_rpm) / MAX_RPM,
        rel / DIST_SCALE,
        [dist / DIST_SCALE],
    ]).astype(np.float32)


register(
    id='interceptor_drone_v0',
    entry_point='app.environmental.interceptor_drone:InterceptorDroneEnv',
)

"""
step is lowest unit - 1/240
episode full run of the steps  all steps done and the env.reset is called 
rollout spawn the number of steps (can be 0-n) until all steps are collected and then resets
epoch runs full one rollout

batch_size chop the steps into n batches and do gradient on them 
check point- total time stamps / check point every 
"""
class InterceptorDroneEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def _get_obs(self):
        return build_observation(self.drone_state, self.target_pos)


    def __init__(self, custom_reward, render_mode=None, pid_gains_path="app/control/best_pid_gains.json"):


        self.dt=1/240
        self.max_steps=5000


        if custom_reward is None:
            raise ValueError("No custom_reward provided")

        self.reward_method= custom_reward


        self.render_mode=render_mode

        self._init_render()

        self.reset()

        hover_thrust=self.config.mass*9.81
        #What to puty there? maybe an box for training 250m x 250m x 150m
        self.action_space= spaces.Box(
            low=np.array([-hover_thrust, -0.5, -0.5, -0.5], dtype=np.float32),
            high=np.array([hover_thrust,0.5, 0.5, 0.5], dtype=np.float32),
        )

        self.observation_space= spaces.Box(
            low=-inf,
            high=inf,
            shape=(21,),
            dtype=np.float32,

        )

        self.pid_teacher=None

        if pid_gains_path  is not None:
            try:
                with open(pid_gains_path) as f:
                    gains = json.load(f)
                self.pid_teacher = PIDHoverController(**gains)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"[warn] could not load PID gains from {pid_gains_path} — pid_teacher unavailable")

    def reset(self, start_pos=None, target_pos=None, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if start_pos is None:
            start_pos = np.array([0, 0, 5], dtype=np.float32)
        if target_pos is None:
            target_pos = np.array([0, 0,5], dtype=np.float32)

        self.target_pos = target_pos

        self.moving_away_streak = 0
        self.hover_steps_in_zone = 0
        self.last_raw_action = np.zeros(4, dtype=np.float32)
        self.hover_success_achieved = False

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
        # drone_id = spawn_drone(self.config, start_position)

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

        self._spawn_visuals()

        self.steps_elapsed = 0

        self.prev_distance = float(np.linalg.norm(self.target_pos - start_pos))


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
        truncated = self.steps_elapsed >= self.max_steps

        self.render()

        return obs, reward, terminated, truncated, {"reason": reason}

    def render(self):
        if self.render_mode != "human":
            return
        p.resetBasePositionAndOrientation(
            self.drone_id,
            [self.drone_state.position.x, self.drone_state.position.y, self.drone_state.position.z],
            [self.drone_state.orientation.x, self.drone_state.orientation.y,
             self.drone_state.orientation.z, self.drone_state.orientation.w],
        )
        p.stepSimulation()
        time.sleep(self.dt)




    def _init_render(self):
        if self.render_mode == "human":
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, 0)
            p.loadURDF("plane.urdf")
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 5])
            self.drone_id = None
            self.target_marker_id = None

    def _spawn_visuals(self):
        if self.render_mode == "human":
            if self.drone_id is not None:
                p.removeBody(self.drone_id)
            self.drone_id = spawn_drone(self.config, self.drone_state.position)

            if self.target_marker_id is not None:
                p.removeBody(self.target_marker_id)
            vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 0.6])
            self.target_marker_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis_shape,
                basePosition=self.target_pos.tolist(),
            )

def my_check_env():
    reward_cfg = RewardConfig(oob_radius=7, hit_reward=10, attitude_penalty=-7, oob_penalty=-10,
                              streak_penalty_coef=-0.05)
    reward_fn = make_reward_fn(reward_cfg)

    env1 = InterceptorDroneEnv(reward_fn)
    obs1, _ = env1.reset(seed=42)
    print("run1:", env1.wind_vector, env1.mass_scale)

    obs2, _ = env1.reset(seed=42)
    print("run2:", env1.wind_vector, env1.mass_scale)

    env=gym.make('interceptor_drone_v0',render_mode=None)

    check_env(env.unwrapped)



def my_test():
    reward_cfg = RewardConfig(oob_radius=7, hit_reward=10, attitude_penalty=-7, oob_penalty=-10,
                              streak_penalty_coef=-0.05)
    reward_fn = make_reward_fn(reward_cfg)

    env = InterceptorDroneEnv(reward_fn)
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
