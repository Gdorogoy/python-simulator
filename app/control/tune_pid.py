import json

import optuna
import numpy as np
from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.control.pid_hover import PIDHoverController
from app.reward_functions.rewards import RewardConfig, make_reward_fn
from app.dynamics.drone import Vector3D

def evaluate_pid(pid, n_episodes=5, n_steps=600):
    reward_cfg = RewardConfig(
        oob_radius=7,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=-0.05,
        hover_success_steps=None,
        streak_cap=30,
    )
    reward_fn=make_reward_fn(reward_cfg)

    env=InterceptorDroneEnv(reward_fn)

    episode_scores=[]

    for i in range(n_episodes):
        target = np.array([0, 0, 5])
        offset = np.random.uniform(-0.2, 0.2, size=3)
        env.reset(start_pos=(target + offset).astype(np.float32), target_pos=target.copy())
        pid.reset()

        tot_reward = 0
        for s in range(n_steps):
            action = pid.compute_action(env.drone_state, env.target_pos)
            obs, reward, terminated, truncated, info = env.step(action)
            tot_reward += reward

            if s % 30 == 0 or terminated:
                print(s, env.drone_state.position.x, env.drone_state.position.y,
                      env.drone_state.position.z, info["reason"])

            if terminated or truncated:
                break

        episode_scores.append(tot_reward)
        pos = np.array([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        dist = np.linalg.norm(env.target_pos - pos)
        print(f"  ep {i}: reward={tot_reward:.2f}, final_dist={dist}, reason={info['reason']}, steps_survived={s}")



    print(pid.compute_action(env.drone_state,env.target_pos))

    return np.mean(episode_scores)

def objective(trial):
    kp_pos = trial.suggest_float("kp_pos", 1.5, 15)
    kd_pos = trial.suggest_float("kd_pos", 0.8, 8)
    kp_att = trial.suggest_float("kp_att", 0.8, 8.0)
    kd_att = trial.suggest_float("kd_att", 0.15, 1.5)
    kp_yaw = trial.suggest_float("kp_yaw", 0.02, 0.2)
    kd_yaw = trial.suggest_float("kd_yaw", 0.02, 0.22)

    ki_pos = trial.suggest_float("ki_pos", 0.0, 3.0)
    ki_att = trial.suggest_float("ki_att", 0.0, 1.5)
    ki_yaw = trial.suggest_float("ki_yaw", 0.0, 0.05)


    pid=PIDHoverController(
        kp_pos=kp_pos,
        kd_pos=kd_pos,
        kp_att=kp_att,
        kd_att=kd_att,
        kp_yaw=kp_yaw,
        kd_yaw=kd_yaw,
        ki_pos=ki_pos,
        ki_att=ki_att,
        ki_yaw=ki_yaw,

    )

    return evaluate_pid(pid, n_episodes=10, n_steps=750)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)
    print(study.best_params)
    with open("app/control/best_pid_gains.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
