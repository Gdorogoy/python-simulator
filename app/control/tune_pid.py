import json

import optuna
import numpy as np
from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.control.pid_hover import PIDHoverController
from app.reward_functions.rewards import RewardConfig, make_reward_fn

# One gains file at 3m doesn't hold at 250m -- max_tilt_rad caps commanded
# tilt regardless of how far the target is, so a controller tuned tight for a
# short error either can't produce enough sustained thrust angle to close a
# long one in reasonable time, or (if tuned loose enough for long range)
# overshoots/oscillates on short ones. Tune a separate gains set per distance
# instead of one set for everything.


"""FOR CURRENT FIX"""
DISTANCES = [3, 10, 50, 150]
# DISTANCES=[3,250]

MIN_EPISODE_SECONDS = 7.5   # dt=1/240 (interceptor_drone.py) -- floor is settling time, not travel time
MIN_STEPS = int(MIN_EPISODE_SECONDS * 240)


def steps_for_dist(target_dist):
    """
    Longer distance needs more steps to even reach the target once, let alone
    hold it -- but hit_threshold precision (getting within 5cm) takes roughly
    the same SETTLING time regardless of distance, so short distances still
    need a real time floor, not just a proportionally tiny travel-time budget.
    """
    return max(MIN_STEPS, int(750 * target_dist / 3))


def evaluate_pid(pid, target_dist, n_episodes=5, n_steps=600, verbose=True, oob_radius=None):
    # default oob_radius scales with the distance being tuned for, so a
    # 250m trial isn't instantly clipped by a radius sized for 3m trials
    oob_radius = oob_radius if oob_radius is not None else max(20.0, target_dist * 3.0)

    reward_cfg = RewardConfig(
        oob_radius=oob_radius,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=-0.05,
        hover_success_steps=None,
        streak_cap=60,
        outer_dist=1.0,
        inner_dist=0.3,
    )
    reward_fn = make_reward_fn(reward_cfg)

    env = InterceptorDroneEnv(reward_fn)

    episode_scores = []

    start = np.array([0, 0, 5], dtype=np.float32)
    target = np.array([target_dist, 0, 5], dtype=np.float32)

    for i in range(n_episodes):
        env.reset(start_pos=start.copy(), target_pos=target.copy())
        pid.reset()

        tot_reward = 0
        for s in range(n_steps):
            action = pid.compute_action(env.drone_state, env.target_pos)
            obs, reward, terminated, truncated, info = env.step(action)
            tot_reward += reward

            if verbose and (s % 30 == 0 or terminated):
                print(s, env.drone_state.position.x, env.drone_state.position.y,
                      env.drone_state.position.z, info["reason"])

            if terminated or truncated:
                break

        episode_scores.append(tot_reward)
        pos = np.array([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        dist = np.linalg.norm(env.target_pos - pos)
        if verbose:
            print(f"  ep {i}: reward={tot_reward:.2f}, final_dist={dist}, reason={info['reason']}, steps_survived={s}")

    return np.mean(episode_scores)


def make_objective(target_dist, n_steps):
    def objective(trial):
        kp_pos = trial.suggest_float("kp_pos", 1.5, 15)
        kd_pos = trial.suggest_float("kd_pos", 0.8, 8)
        kp_att = trial.suggest_float("kp_att", 0.8, 8.0)
        kd_att = trial.suggest_float("kd_att", 0.15, 1.5)
        kp_yaw = trial.suggest_float("kp_yaw", 0.02, 0.2)
        kd_yaw = trial.suggest_float("kd_yaw", 0.02, 0.22)

        ki_pos = trial.suggest_float("ki_pos", 0.0, 1.75)
        ki_att = trial.suggest_float("ki_att", 0.0, 2.5)
        ki_yaw = trial.suggest_float("ki_yaw", 0.0, 0.75)

        pid = PIDHoverController(
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

        return evaluate_pid(pid, target_dist=target_dist, n_episodes=2, n_steps=n_steps, verbose=False)

    return objective


if __name__ == "__main__":
    per_distance_gains = {}

    for dist in DISTANCES:
        print(f"\n=== tuning PID for target_dist={dist}m ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(dist, steps_for_dist(dist)), n_trials=500)
        print(f"dist={dist}m best_value={study.best_value:.2f} best_params={study.best_params}")

        per_distance_gains[str(dist)] = study.best_params

    with open("app/control/best_pid_gains_per_dist.json", "w") as f:
        json.dump(per_distance_gains, f, indent=2)

    print("\nSaved all per-distance gains to app/control/best_pid_gains_per_dist.json")
