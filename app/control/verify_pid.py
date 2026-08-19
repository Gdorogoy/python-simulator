import json
import numpy as np

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.control.pid_hover import PIDHoverController
from app.reward_functions.rewards import RewardConfig, make_reward_fn

best_params = {'kp_pos': 10.971886223833621,
               'kd_pos': 6.495006787354487,
               'kp_att': 4.447496977009914,
               'kd_att': 0.612783390615077,
               'kp_yaw': 0.12108129474700983,
               'kd_yaw': 0.1988043435629733}

pid = PIDHoverController(**best_params)

reward_cfg = RewardConfig(oob_radius=70, hover_success_steps=None)
env = InterceptorDroneEnv(make_reward_fn(reward_cfg))

target = np.array([0, 0, 5], dtype=np.float32)
obs, _ = env.reset(start_pos=target.copy(), target_pos=target.copy())

for step in range(750):
    action = pid.compute_action(env.drone_state, env.target_pos)
    obs, reward, terminated, truncated, info = env.step(action)

    pos = np.array([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
    dist = np.linalg.norm(env.target_pos - pos)

    if step % 30 == 0 or terminated:
        print(f"step={step} dist={dist:.4f} reason={info['reason']}")

    if terminated:
        print(f"FAILED at step {step}: {info['reason']}")
        break
else:
    print("held hover for the full 750 steps — good, proceed to collect_demonstrations.py")