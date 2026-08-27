import json

import numpy as np

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.control.pid_hover import PIDHoverController
from app.control.tune_pid import DISTANCES, steps_for_dist
from app.training.eval_matrix import build_eval_pairs
from app.reward_functions.reward_fn_phase1 import RewardFnPhase1


def collect_demonstrations(gains_by_dist, n_episodes_per_pair=5,
                            save_path="app/control/demonstrations.npz",
                            distances=DISTANCES):
    """
    Collects (obs, action) demonstration pairs across the full distance
    curriculum instead of one fixed 3m target: for every distance in
    `distances` (defaults to tune_pid.DISTANCES), uses that distance's own
    PID gains (gains_by_dist, loaded from best_pid_gains_per_dist.json) and
    every x/y/z direction (+x/-x/+y/-y/+z/-z, minus any -z pair that would
    land underground) at that distance (via build_eval_pairs), so the BC
    dataset covers altitude changes too, not just lateral ones.

    distances lets a caller drop distances they don't actually need demos
    for -- e.g. 250m costs steps_for_dist(250)=62,500 steps/episode vs 1,800
    for 3m, roughly half this function's total runtime, for a distance the
    phase1 curriculum (PHASE1_DISTANCES) doesn't even train on.
    """
    all_obs = []
    all_actions = []

    for dist in distances:
        gains = gains_by_dist[str(dist)]
        pid = PIDHoverController(**gains)
        oob_radius = max(20.0, dist * 3.0)
        n_steps = steps_for_dist(dist)

        # no warmup/imitation curriculum needed here -- this is repeated
        # demonstration collection, not RL training, so stay in phase_1_fn's
        # hit-check the whole time, every episode
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
        env = InterceptorDroneEnv(reward_fn)

        for start, target in build_eval_pairs(oob_radius=oob_radius, distances=(dist,), axes=(0, 1, 2)):
            for ep in range(n_episodes_per_pair):
                # small offset around the nominal start gives BC a neighborhood of
                # corrective examples instead of one exact noiseless curve through
                # state space -- otherwise the model has zero training signal for
                # "slightly off the nominal path" and drifts/compounds in closed loop
                jittered_start = start + np.random.uniform(-0.15, 0.15, size=3).astype(np.float32)
                obs, _ = env.reset(start_pos=jittered_start.astype(np.float32), target_pos=target.copy())
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

            print(f"dist={dist}m target={target.tolist()} -- {len(all_obs)} pairs collected so far")

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)

    np.savez(save_path, obs=all_obs, actions=all_actions)
    print(f"saved {len(all_obs)} (state, action) pairs to {save_path}")
    print(f"obs shape: {all_obs.shape}, actions shape: {all_actions.shape}")


if __name__ == "__main__":
    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    # 250m dropped -- not used by PHASE1_DISTANCES, and by far the most
    # expensive distance (steps_for_dist(250)=62,500 vs 1,800 for 3m)
    collect_demonstrations(gains_by_dist, n_episodes_per_pair=10, distances=(3, 10, 50, 150))
