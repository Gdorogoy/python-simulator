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
