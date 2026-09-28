<<<<<<< HEAD
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
=======
import json

from app.environmental.base_drone_env import BaseDroneEnv
from app.control.pid import PIDController
from app.control.tune_pid import DISTANCES, steps_for_dist
from app.reward_functions.rewards import reward_func
from app.training.eval_matrix import build_eval_pairs, run_eval_matrix, make_pid_action_fn
from app.guidance.plotting import plot_eval_matrix_distance, plot_eval_matrix_pairs


def _make_env(max_steps):
    # reward_func (not the deprecated RewardFnPhase1 roadmap) -- the current
    # reward version. Its oob_radius now scales with env.start_dist (see
    # rewards.reward_func), so it's safe across the full DISTANCES ladder
    # including 250m, not just the original Uniform(3,10) task.
    #
    # max_steps must be passed explicitly: BaseDroneEnv defaults to 15_000,
    # but steps_for_dist(100/150/250) needs 25k/37.5k/62.5k -- leaving the
    # default would silently truncate long-distance episodes before the
    # controller has a chance to converge.
    return BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=max_steps)


with open("app/control/best_pid_gains_per_dist.json") as f:
    gains_by_dist = json.load(f)

all_results = []
all_passed = True

for dist in DISTANCES:
    pid = PIDController(**gains_by_dist[str(dist)])
    oob_radius = max(20.0, dist * 3.0)
    env = _make_env(max_steps=steps_for_dist(dist))

    # Same fixed (start, target) pairs used to score RL checkpoints, so the PID
    # baseline is compared against RL on identical, non-random configs.
    results = run_eval_matrix(
        env, make_pid_action_fn(pid), pairs=build_eval_pairs(oob_radius=oob_radius, distances=(dist,)),
        n_repeats=3, max_steps=steps_for_dist(dist), on_episode_reset=pid.reset,
    )

    for r in results:
        status = "HIT" if r["hit_rate"] > 0 else "FAILED"
        if r["hit_rate"] == 0:
            all_passed = False
        print(f"dist={dist}m start={r['start']} -> target={r['target']}  "
              f"mean_final_dist={r['mean_final_dist']:.4f} (+/-{r['std_final_dist']:.4f})  "
              f"hit_rate={r['hit_rate']:.2f}  mean_steps={r['mean_steps']:.0f}  [{status}]")

    all_results.extend(results)

plot_eval_matrix_distance({"PID": all_results})
plot_eval_matrix_pairs({"PID": all_results})

if all_passed:
    print("all pairs hit at least once â€” proceed to collect_demonstrations.py")
else:
    print("some pairs never hit â€” tune the PID gains before collecting demonstrations")
>>>>>>> later_to_remove
