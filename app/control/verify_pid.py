import json

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.control.pid_hover import PIDHoverController
from app.control.tune_pid import DISTANCES, steps_for_dist
from app.reward_functions.reward_fn_phase1 import RewardFnPhase1
from app.training.eval_matrix import build_eval_pairs, run_eval_matrix, make_pid_action_fn
from app.guidance.plotting import plot_eval_matrix_distance, plot_eval_matrix_pairs


def _make_env(oob_radius):
    # warmup_duration_steps=0, phase1_duration_steps=None -> phase_1_fn (which
    # DOES check for "Hit") is the only stage, active from step 1, forever.
    # Leaving warmup_duration_steps at its default (10_000) is a real bug here:
    # chain_reward_fns' stage counter is shared/cumulative across the WHOLE
    # eval run, not per-episode, so it silently falls through to base_fn (no
    # hit-check at all) partway through the run and every pair after that
    # can never register a "Hit" regardless of how well the PID flies.
    reward_fn = RewardFnPhase1(
        hit_steps_streak=1500,
        phase1_pos_coef=0.25,
        hit_reward=5,
        oob_radius=oob_radius,
        hover_success_steps=None,
        streak_cap=60,
        outer_dist=1.0,
        inner_dist=0.3,
        hit_threshold=0.05,
        warmup_duration_steps=0,
        imitation_duration_steps=0,
        phase1_duration_steps=None,
    ).as_roadmap()
    return InterceptorDroneEnv(reward_fn)


with open("app/control/best_pid_gains_per_dist.json") as f:
    gains_by_dist = json.load(f)

all_results = []
all_passed = True

for dist in DISTANCES:
    pid = PIDHoverController(**gains_by_dist[str(dist)])
    oob_radius = max(20.0, dist * 3.0)
    env = _make_env(oob_radius)

    # Same fixed (start,target) pairs used to score RL checkpoints
    # (app/training/eval_matrix.py) -- lets the PID baseline be compared
    # against RL on identical, non-random configs.
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
    print("all pairs hit at least once — proceed to collect_demonstrations.py")
else:
    print("some pairs never hit — tune the PID gains before collecting demonstrations")
