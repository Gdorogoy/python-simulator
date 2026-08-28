import json

import numpy as np
import torch

from app.control.pid_hover import PIDHoverController
from app.control.pretrain_bc import pretrain_behavior_cloning
from app.control.tune_pid import DISTANCES, steps_for_dist
from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.plotting import plot_dagger_history
from app.guidance.train import ActorCritic, device
from app.reward_functions.reward_fn_phase1 import RewardFnPhase1
from app.training.eval_matrix import build_eval_pairs


def _make_env(oob_radius):
    # Skip warmup/imitation and stay in phase_1_fn's hit-check the whole time: this is
    # rollout collection, not curriculum RL, and phase-state is shared across every
    # env.reset(), so a nonzero warmup_duration_steps would only fire once total, not per-episode.
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


def _policy_action(model, obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, _, _ = model.forward(obs_t)
        action = model.scale_action(mean)
    return action.squeeze(0).cpu().numpy()


def dagger(gains_by_dist, n_rounds=5, num_episodes_per_pair=3,
           retrain_epochs=20, checkpoint_path="app/control/pretrained_bc.pt",
           demo_path="app/control/demonstrations.npz",
           out_path="app/control/pretrained_bc_dagger.pt",
           plots_dir="plots_final", distances=DISTANCES):
    """
    Runs the same distance curriculum as collect_demonstrations.py (per-distance PID
    gains, every x/y/z direction via build_eval_pairs) to collect fresh rollouts each
    round. `distances` only limits new collection -- any 250m rows already baked into
    demo_path's dataset stay in the aggregate; they just stop growing.

    Long-distance episodes run far more steps than short ones, so raw pair counts per
    distance are wildly unequal; each round, every distance's pairs are subsampled
    down to the smallest distance's count before aggregating, so no distance dominates
    the BC loss.
    """
    shape_env = _make_env(oob_radius=300)
    model = ActorCritic(shape_env.observation_space.shape[0], shape_env.action_space.shape[0],
                         shape_env.action_space.low, shape_env.action_space.high).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    data = np.load(demo_path)
    agg_obs = data["obs"]
    agg_actions = data["actions"]

    history = []

    for round_idx in range(n_rounds):
        obs_by_dist = {d: [] for d in distances}
        actions_by_dist = {d: [] for d in distances}
        hits_by_dist = {d: 0 for d in distances}
        episodes_by_dist = {d: 0 for d in distances}
        info = {"reason": None}

        for dist in distances:
            gains = gains_by_dist[str(dist)]
            pid = PIDHoverController(**gains)
            oob_radius = max(20.0, dist * 3.0)
            max_steps = steps_for_dist(dist)
            env = _make_env(oob_radius)

            # Jitter magnitude scales with task distance -- a fixed absolute range would be
            # negligible at 250m but dominate (and distort) the shortest curriculum bucket.
            jitter_mag = min(0.75, 0.25 * dist)

            for start, target in build_eval_pairs(oob_radius=oob_radius, distances=(dist,), axes=(0, 1, 2)):
                for ep in range(num_episodes_per_pair):
                    # Jitter must cover all 3 axes, including z: z-axis pairs need the same
                    # neighborhood jitter as x/y, since z is the actual task dimension there.
                    drone_offset = np.random.uniform(-jitter_mag, jitter_mag, size=3).astype(np.float32)
                    target_offset = np.random.uniform(-jitter_mag, jitter_mag, size=3).astype(np.float32)

                    obs, _ = env.reset(start_pos=(start + drone_offset).astype(np.float32),
                                        target_pos=(target + target_offset).astype(np.float32))
                    pid.reset()

                    for step in range(max_steps):
                        policy_action = _policy_action(model, obs)  # policy drives the drone
                        pid_action = pid.compute_action(env.drone_state, env.target_pos)  # PID only supplies the label

                        obs_by_dist[dist].append(obs.copy())
                        actions_by_dist[dist].append(pid_action.copy())

                        obs, reward, terminated, truncated, info = env.step(policy_action)
                        if terminated or truncated:
                            break

                    episodes_by_dist[dist] += 1
                    if info["reason"] == "Hit":
                        hits_by_dist[dist] += 1

            print(f"round {round_idx + 1}/{n_rounds} dist={dist}m -- {len(obs_by_dist[dist])} raw pairs, "
                  f"hit_rate={hits_by_dist[dist] / episodes_by_dist[dist]:.2f}, last reason={info['reason']}")

        raw_counts = {d: len(obs_by_dist[d]) for d in distances}
        min_count = max(min(raw_counts.values()), 1)

        rng = np.random.default_rng(round_idx)
        balanced_obs_parts, balanced_actions_parts = [], []
        balanced_counts = {}
        for d in distances:
            n = raw_counts[d]
            idx = rng.choice(n, size=min_count, replace=False) if n > min_count else np.arange(n)
            balanced_counts[d] = len(idx)
            balanced_obs_parts.append(np.array(obs_by_dist[d], dtype=np.float32)[idx])
            balanced_actions_parts.append(np.array(actions_by_dist[d], dtype=np.float32)[idx])

        new_obs = np.concatenate(balanced_obs_parts)
        new_actions = np.concatenate(balanced_actions_parts)

        agg_obs = np.concatenate([agg_obs, new_obs])
        agg_actions = np.concatenate([agg_actions, new_actions])

        hit_rate = {d: hits_by_dist[d] / max(episodes_by_dist[d], 1) for d in distances}
        history.append({"round": round_idx + 1, "raw_counts": raw_counts,
                         "balanced_counts": balanced_counts, "hit_rate": hit_rate})

        print(f"round {round_idx + 1}/{n_rounds}: raw_counts={raw_counts} -> balanced to {min_count}/dist, "
              f"aggregated dataset now {len(agg_obs)} pairs, retraining...")
        model = pretrain_behavior_cloning(model, obs=agg_obs, actions=agg_actions, epochs=retrain_epochs)

        # Keep a growing snapshot on disk in case a later round crashes.
        snapshot_path = demo_path if demo_path.endswith("_dagger.npz") else demo_path.replace(".npz", "_dagger.npz")
        np.savez(snapshot_path, obs=agg_obs, actions=agg_actions)
        torch.save(model.state_dict(), out_path)

        plot_dagger_history(history, output_dir=plots_dir)

    torch.save(model.state_dict(), out_path)
    print(f"saved DAgger-refined weights to {out_path}")
    return model, history


if __name__ == "__main__":
    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    dagger(
        gains_by_dist,
        n_rounds=10, num_episodes_per_pair=5,
        checkpoint_path="app/control/pretrained_bc.pt",
        out_path="app/control/pretrained_bc_dagger.pt",
        demo_path="app/control/demonstrations.npz",
        # 150m/250m dropped -- both far more expensive per episode than 3/10/50m.
        distances=(3, 10, 50),
    )
