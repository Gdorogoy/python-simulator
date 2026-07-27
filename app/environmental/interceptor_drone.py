from math import inf
from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
import pybullet
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from scipy.spatial.transform import Rotation

from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion, QuadConfig
from app.dynamics.methods import mixer_inversion, timestamp_update
from app.environmental.enviorment import sample_wind_conditions, spawn_drone





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

    def _compute_reward(self):
        pos= np.array([self.drone_state.position.x, self.drone_state.position.y, self.drone_state.position.z])
        target= self.target_pos

        dist= np.linalg.norm(target - pos)

        # checking if out of bounds isnan checking if object is even there , pos[2] is z so checking if its not on the ground, and that the drone not too far
        if np.any(np.isnan(pos)) or pos[2] < 0.0 or np.linalg.norm(pos) >30:
            return -10.0, True


        # calculation the new rotation in 3d (!! NEED TO IMPLEMENT THE METHOD AND NOT USE READY)
        # calculation by the same concept as object rotation in 2d but now 4d where z stays the same always and x,y work the same as in 2d and w work like...
        rot= Rotation.from_quat([self.drone_state.orientation.x, self.drone_state.orientation.y, self.drone_state.orientation.z, self.drone_state.orientation.w])

        roll,pitch,yaw=rot.as_euler("xyz")


        # checking if roll or pitch exceeds 80 degrees,
        if abs(roll) > np.radians(80) or abs(pitch) > np.radians(80):
            return -5.0 , True

        # success!! the drone hit the target
        if dist <0.3:
            return 15.0 , True

        progress=(self.prev_distance - dist) - 0.01

        self.prev_distance = dist
        return progress, False


    def __init__(self,render_mode=None):

        self.dt=1/240
        self.max_steps=5000




        self.render_mode=render_mode



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


        self.wind_vector , self.mass_scale = sample_wind_conditions(np_rand=self.np_random)


        self.config = create_quad_config(
            mass=1.5 * self.mass_scale,
            inertia=(0.02 * self.mass_scale, 0.02 * self.mass_scale, 0.04 * self.mass_scale),
            arm_length=0.22,
            drag_coeff=0.035,
            max_rpm=12000,
            motor_tau=0.05,
        )

        start_z = 5.0
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

        return obs, reward, terminated,truncated, {}



    def render(self):
        pass






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
