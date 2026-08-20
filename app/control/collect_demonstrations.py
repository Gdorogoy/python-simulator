import json
import numpy as np

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.control.pid_hover import PIDHoverController
from app.reward_functions.rewards import RewardConfig, make_reward_fn


def collect_demonstrations(pid, n_episodes=50, n_steps=600,
                            save_path="app/control/demonstrations.npz"):
    reward_cfg = RewardConfig(oob_radius=7, hover_success_steps=None)
    reward_fn = make_reward_fn(reward_cfg)
    env = InterceptorDroneEnv(reward_fn)

    all_obs = []
    all_actions = []

    for ep in range(n_episodes):
        offset = np.random.uniform(-0.2, 0.2, size=3)  # random small displacement from target
        start = np.array([0, 0, 5], dtype=np.float32) + offset
        target = np.array([0, 0, 5], dtype=np.float32)
        obs, _ = env.reset(start_pos=start, target_pos=target)
        pid.reset()

        for s in range(n_steps):
            action = pid.compute_action(env.drone_state, env.target_pos)

            # record the STATE the policy would see and the ACTION the PID
            # took in response to it -- this pairing is what BC learns from
            all_obs.append(obs.copy())
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        if ep % 10 == 0:
            print(f"episode {ep}/{n_episodes} -- {len(all_obs)} pairs collected so far")

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)

    np.savez(save_path, obs=all_obs, actions=all_actions)
    print(f"saved {len(all_obs)} (state, action) pairs to {save_path}")
    print(f"obs shape: {all_obs.shape}, actions shape: {all_actions.shape}")






def collect_demonstrations_x(pid, n_epidsodes=50, n_steps=600,
                             save_path="app/control/demonstrations_x.npz"):
    
    reward_cfg = RewardConfig(oob_radius=7, hover_success_steps=None)
    reward_fn = make_reward_fn(reward_cfg)
    env = InterceptorDroneEnv(reward_fn)




if __name__ == "__main__":
    best_params={
                'kp_pos': 2.3348549042980946,
                 'kd_pos': 1.0978513647624821,
                 'kp_att': 7.882461182173537,
                 'kd_att': 1.0936341247316574,
                 'kp_yaw': 0.048930591736932,
                 'kd_yaw': 0.20306927726482113

                 }


    pid = PIDHoverController(**best_params)
    collect_demonstrations(pid, n_episodes=50, n_steps=600)