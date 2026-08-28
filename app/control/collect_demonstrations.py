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
    Collects (obs, action) demonstration pairs across the full distance curriculum in
    `distances`, using each distance's own PID gains and every x/y/z direction
    (via build_eval_pairs) so the BC dataset covers altitude, not just lateral moves.
    `distances` lets a caller drop the expensive-but-unused 250m case (not part of
    PHASE1_DISTANCES) to save runtime.
    """
    all_obs = []
    all_actions = []

    for dist in distances:
        gains = gains_by_dist[str(dist)]
        pid = PIDHoverController(**gains)
        oob_radius = max(20.0, dist * 3.0)
        n_steps = steps_for_dist(dist)

        # No warmup/imitation curriculum -- this is demonstration collection,
        # not RL training, so stay in phase_1_fn's hit-check the whole time.
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
                # Jitter around the nominal start gives BC a neighborhood of corrective
                # examples, not one noiseless curve, so it has signal for off-path states.
                jittered_start = start + np.random.uniform(-0.15, 0.15, size=3).astype(np.float32)
                obs, _ = env.reset(start_pos=jittered_start.astype(np.float32), target_pos=target.copy())
                pid.reset()

                for s in range(n_steps):
                    action = pid.compute_action(env.drone_state, env.target_pos)

                    # Record the state the policy would see paired with the PID's action -- what BC learns from.
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

    # 250m dropped -- unused by PHASE1_DISTANCES and by far the most expensive distance.
    collect_demonstrations(gains_by_dist, n_episodes_per_pair=10, distances=(3, 10, 50, 150))
