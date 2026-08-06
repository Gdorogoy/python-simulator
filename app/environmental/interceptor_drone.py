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

from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion, QuadConfig
from app.dynamics.methods import mixer_inversion, timestamp_update
from app.environmental.enviorment import sample_wind_conditions, spawn_drone

import time
import pybullet as p
import pybullet_data





def build_observation(state: QuadState, target_pos: np.ndarray) -> np.ndarray:
    rel = target_pos - np.array([state.position.x, state.position.y, state.position.z])
    dist = np.linalg.norm(rel)

    return np.concatenate([
        [state.position.x, state.position.y, state.position.z],
        [state.velocity.x, state.velocity.y, state.velocity.z],
        [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w],
        [state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z],
        state.rotor_rpm,
        rel,
        [dist],
    ]).astype(np.float32)


register(
    id='interceptor_drone_v0',
    entry_point='app.environmental.interceptor_drone:InterceptorDroneEnv',
)


class InterceptorDroneEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def _get_obs(self):
        return build_observation(self.drone_state, self.target_pos)

    def _compute_reward(self, reward=5, penalty=-5):
        pos = np.array([self.drone_state.position.x, self.drone_state.position.y, self.drone_state.position.z])
        dist = np.linalg.norm(self.target_pos - pos)

        if np.any(np.isnan(pos)) or pos[2] < 0.0 or np.linalg.norm(pos) > 30:
            return penalty*2, True

        rot = Rotation.from_quat([self.drone_state.orientation.x, self.drone_state.orientation.y,
                                self.drone_state.orientation.z, self.drone_state.orientation.w])
        roll, pitch, yaw = rot.as_euler("xyz")
        if abs(roll) > np.radians(65) or abs(pitch) > np.radians(80):
            return penalty*2, True

        if dist < 0.3:
            return reward, True

        diff = self.prev_distance - dist

        if diff < 0:
            self.moving_away_streak += 1
            progress = diff * 1.25 - 0.01
        else:
            self.moving_away_streak = 0
            progress = diff - 0.01

        streak_penalty = -0.02 * self.moving_away_streak
        closer_bonus = 0.01 if diff > 0 else 0.0

        self.prev_distance = dist

        if self.moving_away_streak >= 60:
            return penalty, True

        return progress + streak_penalty + closer_bonus, False


    def __init__(self,render_mode=None):

        self.dt=1/240
        self.max_steps=5000




        self.render_mode=render_mode

        self._init_render()

        self.reset()


        #What to puty there? maybe an box for training 250m x 250m x 150m
        self.action_space= spaces.Box(
            low=np.array([0.0, -0.5, -0.5, -0.5], dtype=np.float32),
            high=np.array([40.0, 0.5, 0.5, 0.5], dtype=np.float32),
        )

        self.observation_space= spaces.Box(
            low=-inf,
            high=inf,
            shape=(21,),
            dtype=np.float32,

        )



    def reset(self,seed=None,options=None):
        super().reset(seed=seed,options=options)

        self.moving_away_streak = 0 # added

        self.wind_vector , self.mass_scale = sample_wind_conditions(np_rand=self.np_random)


        self.config = create_quad_config(
            mass=1.5 * self.mass_scale,
            inertia=(0.02 * self.mass_scale, 0.02 * self.mass_scale, 0.04 * self.mass_scale),
            arm_length=0.22,
            drag_coeff=0.035,
            max_rpm=12000,
            motor_tau=0.05,
        )

        start_z = 0.5 # change back later to 5
        start_position = Vector3D(0, 0, start_z)
        # drone_id = spawn_drone(self.config, start_position)

        hover_thrust = self.config.mass * 9.81
        hover_omega = mixer_inversion(self.config, [hover_thrust, 0.0, 0.0, 0.0])

        self.drone_state =QuadState(
            position=start_position,
            velocity=Vector3D(0, 0, 0),
            orientation=Quaternion(0, 0, 0, 1),
            angular_velocity=Vector3D(0, 0, 0),
            rotor_rpm=list(hover_omega),
        )

        self.target_pos = np.array([5.0,0.0,5.0], dtype=np.float32)

        self._spawn_visuals()

        self.steps_elapsed = 0
        self.prev_distance = float(np.linalg.norm(self.target_pos - np.array([0, 0, start_z])))


        obs = self._get_obs()
        return obs, {}



    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        action= np.clip( action, self.action_space.low, self.action_space.high )
        self.drone_state= timestamp_update(self.drone_state,self.config,list(action),self.wind_vector,self.dt)
        self.steps_elapsed += 1

        obs=self._get_obs()
        reward,terminated= self._compute_reward()
        truncated= self.steps_elapsed >= self.max_steps

        self.render()

        return obs, reward, terminated,truncated, {}

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

    env1 = InterceptorDroneEnv()
    obs1, _ = env1.reset(seed=42)
    print("run1:", env1.wind_vector, env1.mass_scale)

    obs2, _ = env1.reset(seed=42)
    print("run2:", env1.wind_vector, env1.mass_scale)

    env=gym.make('interceptor_drone_v0',render_mode=None)

    check_env(env.unwrapped)



def my_test():
    env = InterceptorDroneEnv()
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
