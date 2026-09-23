import json

import numpy as np
import torch

from app.control.collect_demonstrations import DEFAULT_DISTANCE_LOW, DEFAULT_DISTANCE_HIGH
from app.control.pid import PIDController
from app.control.pretrain_bc import pretrain_behavior_cloning
from app.control.tune_pid import DISTANCES, steps_for_dist
from app.environmental.base_drone_env import BaseDroneEnv
from app.guidance.plotting import plot_dagger_history
from app.guidance.train import ActorCritic, device
from app.reward_functions.rewards import reward_func
from app.training.eval_matrix import build_omni_eval_pairs

# Aggregate buffer cap, in raw (obs, pid_action) pairs. Without this, agg_obs/
# agg_actions below grows every round for all n_rounds with nothing ever removed,
# so retrain cost (and memory) balloons round over round and old, less-refined
# rounds never get diluted out relative to newer, better ones.
AGG_BUFFER_CAP_PAIRS = 500_000

# Per-round decay applied to both eviction-keep-probability and training loss
# weight: a pair from `k` rounds ago gets weight RECENCY_DECAY**k. This is what
# makes the aggregate "learn off dagger" instead of just uniformly pooling
# everything ever collected -- newer rounds (closer to the current policy's
# actual mistakes) dominate both what survives the buffer cap and what the
# retrain loss emphasizes, while old round-0 baseline data never fully
# disappears (it just fades).
RECENCY_DECAY = 0.85


def _make_env(max_steps=15_000):
    # reward_func (app.reward_functions.rewards), not the deprecated RewardFnPhase1
    # roadmap -- the current reward version, matching what base_training.py actually
    # trains against. Its oob_radius scales with env.start_dist (see
    # rewards.reward_func), so it's safe across the full distance curriculum,
    # not just the 3/10m this script currently runs at.
    #
    # max_steps must be passed per distance: BaseDroneEnv defaults to 15_000,
    # but steps_for_dist(100/150/250) needs 25k/37.5k/62.5k -- leaving the
    # default would silently truncate long-distance rollouts before the
    # policy/PID has a chance to converge, regardless of this function's own
    # max_steps loop bound below.
    return BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=max_steps)


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
           plots_dir="plots_final", distances=DISTANCES,
           hidden=64, num_hidden_layers=4, model=None, snapshot_path=None):
    """
    Runs the same distance curriculum as collect_demonstrations.py (per-distance PID
    gains) to collect fresh rollouts each round, using build_omni_eval_pairs so every
    single/pair/triple-axis diagonal direction gets corrected each round, not just the
    six cardinal ones. `distances` only limits new collection -- any 250m rows already
    baked into demo_path's dataset stay in the aggregate; they just stop growing.

    Long-distance episodes run far more steps than short ones, so raw pair counts per
    distance are wildly unequal; each round, every distance's pairs are subsampled
    down to the smallest distance's count before aggregating, so no distance dominates
    the BC loss.

    `model`/`hidden`/`num_hidden_layers` let a caller hand in an already-built,
    already-BC-pretrained model of a specific shape instead of loading
    `checkpoint_path` off disk -- the on-policy rollouts below depend on the
    exact policy driving the drone, so a non-default architecture needs the matching net.
    `snapshot_path` overrides where the growing aggregate is saved each round; default
    derives it from `demo_path` as before, which is only safe when demo_path is used by
    a single dagger() run at a time.
    """
    shape_env = _make_env()
    if model is None:
        model = ActorCritic(shape_env.observation_space.shape[0], shape_env.action_space.shape[0],
                             shape_env.action_space.low, shape_env.action_space.high,
                             hidden=hidden, num_hidden_layers=num_hidden_layers).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    data = np.load(demo_path)
    agg_obs = data["obs"]
    agg_actions = data["actions"]
    # Round 0 = pre-DAgger baseline demonstrations (oldest, lowest recency weight
    # once rounds start accumulating).
    agg_round = np.zeros(len(agg_obs), dtype=np.int32)

    history = []

    for round_idx in range(n_rounds):
        obs_by_dist = {d: [] for d in distances}
        actions_by_dist = {d: [] for d in distances}
        hits_by_dist = {d: 0 for d in distances}
        episodes_by_dist = {d: 0 for d in distances}
        info = {"reason": None}

        for dist in distances:
            gains = gains_by_dist[str(dist)]
            pid = PIDController(**gains)
            oob_radius = max(20.0, dist * 3.0)
            max_steps = steps_for_dist(dist)
            env = _make_env(max_steps=max_steps)

            # Jitter magnitude scales with task distance -- a fixed absolute range would be
            # negligible at 250m but dominate (and distort) the shortest curriculum bucket.
            jitter_mag = min(0.75, 0.25 * dist)

            # build_omni_eval_pairs (not build_eval_pairs) -- build_eval_pairs only
            # ever offsets ONE axis at a time (x+/x-/y+/y-/z+/z-), so DAgger's own
            # on-policy corrections never covered a diagonal 2/3-axis target even
            # though the round-0 baseline (collect_demonstrations_omni) did. That
            # left diagonal movement uncorrected round over round. build_omni_eval_pairs
            # covers every single/pair/triple-axis combo (all signs) at this distance.
            for start, target, target_yaw in build_omni_eval_pairs(oob_radius=oob_radius, distances=(dist,)):
                for ep in range(num_episodes_per_pair):
                    # Jitter must cover all 3 axes, including z: z-axis pairs need the same
                    # neighborhood jitter as x/y, since z is the actual task dimension there.
                    drone_offset = np.random.uniform(-jitter_mag, jitter_mag, size=3).astype(np.float32)
                    target_offset = np.random.uniform(-jitter_mag, jitter_mag, size=3).astype(np.float32)

                    obs, _ = env.reset(start_pos=(start + drone_offset).astype(np.float32),
                                        target_pos=(target + target_offset).astype(np.float32),
                                        target_yaw=target_yaw)
                    pid.reset()

                    for step in range(max_steps):
                        policy_action = _policy_action(model, obs)  # policy drives the drone
                        pid_action = pid.compute_action(env.drone_state, env.target_pos, env.target_yaw)  # PID only supplies the label

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
        agg_round = np.concatenate([agg_round, np.full(len(new_obs), round_idx + 1, dtype=np.int32)])

        if len(agg_obs) > AGG_BUFFER_CAP_PAIRS:
            # Recency-weighted eviction: a pair's keep-probability decays with how
            # many rounds old it is, so the buffer skews toward recent, more-refined
            # corrections instead of shrinking everything uniformly at random.
            keep_w = RECENCY_DECAY ** (round_idx + 1 - agg_round)
            idx = rng.choice(len(agg_obs), size=AGG_BUFFER_CAP_PAIRS, replace=False,
                              p=keep_w / keep_w.sum())
            agg_obs, agg_actions, agg_round = agg_obs[idx], agg_actions[idx], agg_round[idx]

        hit_rate = {d: hits_by_dist[d] / max(episodes_by_dist[d], 1) for d in distances}
        history.append({"round": round_idx + 1, "raw_counts": raw_counts,
                         "balanced_counts": balanced_counts, "hit_rate": hit_rate})

        print(f"round {round_idx + 1}/{n_rounds}: raw_counts={raw_counts} -> balanced to {min_count}/dist, "
              f"aggregated dataset now {len(agg_obs)} pairs, retraining...")
        # Same recency weighting drives the retrain loss: rounds closer to "now"
        # matter more than the original round-0 PID-only baseline.
        train_w = RECENCY_DECAY ** (round_idx + 1 - agg_round)
        model = pretrain_behavior_cloning(model, obs=agg_obs, actions=agg_actions,
                                           weights=train_w, epochs=retrain_epochs)

        # Keep a growing snapshot on disk in case a later round crashes.
        round_snapshot_path = snapshot_path or (
            demo_path if demo_path.endswith("_dagger.npz") else demo_path.replace(".npz", "_dagger.npz")
        )
        np.savez(round_snapshot_path, obs=agg_obs, actions=agg_actions)
        torch.save(model.state_dict(), out_path)

        plot_dagger_history(history, output_dir=plots_dir)

    torch.save(model.state_dict(), out_path)
    print(f"saved DAgger-refined weights to {out_path}")
    return model, history


# Batched on-policy DAgger on the live Isaac Lab env -- same algorithm as
# dagger() above but num_envs episodes in parallel per round.
AGG_BUFFER_CAP_PAIRS_ISAAC = AGG_BUFFER_CAP_PAIRS


def dagger_base_drone_isaac(env, gains_by_dist: dict, n_rounds: int = 5,
                             distance_low: float = DEFAULT_DISTANCE_LOW, distance_high: float = DEFAULT_DISTANCE_HIGH,
                             rows_per_round: int = 20_000, retrain_epochs: int = 20,
                             checkpoint_path: str = "app/control/pretrained_bc.pt",
                             demo_path: str = "app/control/demonstrations_omni.npz",
                             out_path: str = "app/control/pretrained_bc_dagger_isaac.pt",
                             hidden: int = 64, num_hidden_layers: int = 4, model=None,
                             pool_size: int = 1024, pool_refresh_every_rows: int = 50_000):
    """Returns (model, history) -- history is a list of per-round dicts
    (round, rows_collected, aggregate_size, hit_rate, episodes). Distance is
    drawn continuously from Uniform(distance_low, distance_high) per pool
    entry (see collect_demonstrations_base_drone_isaac's docstring) -- PID
    gains are picked per-env by nearest-distance match, so an arbitrary
    continuous distance is fine.

    Known simplifications vs. dagger(): no per-distance row balancing (the
    numpy version subsamples every distance down to its smallest distance's
    raw count each round before aggregating, since long-distance episodes run
    far more steps and would otherwise dominate the BC loss) -- here envs
    draw targets from the shared omni pool and contribute rows for however
    long their episode runs, unrebalanced. Also no per-round
    plot_dagger_history call."""
    import torch
    import isaaclab.utils.math as math_utils
    from app.control.collect_demonstrations import sample_full_sphere_target
    from app.control.torch_pid import TorchPIDController, assign_gains_by_distance

    unwrapped = env.unwrapped
    device = unwrapped.device
    obs_dim = unwrapped.single_observation_space["policy"].shape[0]
    action_dim = unwrapped.single_action_space.shape[0]

    if model is None:
        model = ActorCritic(obs_dim, action_dim, unwrapped.single_action_space.low,
                             unwrapped.single_action_space.high,
                             hidden=hidden, num_hidden_layers=num_hidden_layers).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    data = np.load(demo_path)
    agg_obs = data["obs"]
    agg_actions = data["actions"]
    agg_round = np.zeros(len(agg_obs), dtype=np.int32)

    rng = np.random.default_rng()

    def _refresh_pool():
        pairs = []
        for _ in range(pool_size):
            dist = float(rng.uniform(distance_low, distance_high))
            target = None
            while target is None:
                target = sample_full_sphere_target(rng, dist)
            start = np.array([0, 0, 5], dtype=np.float32) + rng.uniform(-0.15, 0.15, size=3).astype(np.float32)
            target_yaw = float(rng.uniform(-np.pi, np.pi))
            pairs.append((start, target, target_yaw))
        unwrapped.set_target_pairs(pairs)

    _refresh_pool()
    pid = TorchPIDController(unwrapped.num_envs, device, **next(iter(gains_by_dist.values())))

    # Reset ONCE, not every round -- a per-round reset was silently making hit_rate
    # read near-zero regardless of policy quality (see PROJECT_DEFENSE_GUIDE.md Part 8.5.1).
    obs = env.reset()[0]["policy"]
    all_env_ids = torch.arange(unwrapped.num_envs, device=device)
    assign_gains_by_distance(pid, unwrapped._start_dist, gains_by_dist, all_env_ids)
    pid.reset()

    history = []
    for round_idx in range(n_rounds):
        round_obs, round_actions = [], []
        n_episodes = 0
        n_hits = 0
        n_oob = 0
        n_attitude = 0
        n_timeout = 0
        rows_since_refresh = 0

        while len(round_obs) * unwrapped.num_envs < rows_per_round:
            with torch.no_grad():
                mean, _, _ = model.forward(obs)
                policy_action = model.scale_action(mean)  # policy drives the drone

            pos = unwrapped._robot.data.root_pos_w - unwrapped._terrain.env_origins
            vel = unwrapped._robot.data.root_lin_vel_w
            ang_vel = unwrapped._robot.data.root_ang_vel_b
            quat_wxyz = unwrapped._robot.data.root_quat_w
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_wxyz)
            target_local = unwrapped._desired_pos_w - unwrapped._terrain.env_origins
            with torch.no_grad():
                pid_action = pid.compute_action(pos, vel, ang_vel, roll, pitch, yaw,
                                                 target_local, unwrapped._desired_yaw_w, dt=1 / 240)

            round_obs.append(obs.cpu().numpy())
            round_actions.append(pid_action.cpu().numpy())  # PID only supplies the label

            next_obs_dict, reward, terminated, truncated, extras = env.step(policy_action)
            obs = next_obs_dict["policy"]

            done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                n_episodes += len(done_ids)
                n_hits += int(extras["term_reasons"]["hit"][done_ids].sum().item())
                n_oob += int(extras["term_reasons"]["oob"][done_ids].sum().item())
                n_attitude += int((extras["term_reasons"]["attitude_roll"][done_ids]
                                    | extras["term_reasons"]["attitude_pitch"][done_ids]).sum().item())
                n_timeout += int((truncated[done_ids] & ~terminated[done_ids]).sum().item())  # ran out of time, no hit/crash
                assign_gains_by_distance(pid, unwrapped._start_dist, gains_by_dist, done_ids)
                pid.reset(done_ids)

            rows_since_refresh += unwrapped.num_envs
            if rows_since_refresh >= pool_refresh_every_rows:
                _refresh_pool()
                rows_since_refresh = 0

        new_obs = np.concatenate(round_obs, axis=0)
        new_actions = np.concatenate(round_actions, axis=0)
        hit_rate = n_hits / max(n_episodes, 1)
        oob_rate = n_oob / max(n_episodes, 1)
        attitude_rate = n_attitude / max(n_episodes, 1)
        timeout_rate = n_timeout / max(n_episodes, 1)

        agg_obs = np.concatenate([agg_obs, new_obs])
        agg_actions = np.concatenate([agg_actions, new_actions])
        agg_round = np.concatenate([agg_round, np.full(len(new_obs), round_idx + 1, dtype=np.int32)])

        eviction_rng = np.random.default_rng(round_idx)
        if len(agg_obs) > AGG_BUFFER_CAP_PAIRS_ISAAC:
            # Same recency-weighted eviction as dagger(): a pair's
            # keep-probability decays with how many rounds old it is.
            keep_w = RECENCY_DECAY ** (round_idx + 1 - agg_round)
            idx = eviction_rng.choice(len(agg_obs), size=AGG_BUFFER_CAP_PAIRS_ISAAC, replace=False,
                                       p=keep_w / keep_w.sum())
            agg_obs, agg_actions, agg_round = agg_obs[idx], agg_actions[idx], agg_round[idx]

        history.append({"round": round_idx + 1, "rows_collected": len(new_obs),
                         "aggregate_size": len(agg_obs), "hit_rate": hit_rate,
                         "oob_rate": oob_rate, "attitude_rate": attitude_rate, "timeout_rate": timeout_rate,
                         "episodes": n_episodes})
        print(f"round {round_idx + 1}/{n_rounds}: collected {len(new_obs)} rows "
              f"({n_episodes} episodes, hit_rate={hit_rate:.2f}, oob_rate={oob_rate:.2f}, "
              f"attitude_rate={attitude_rate:.2f}, timeout_rate={timeout_rate:.2f}), "
              f"aggregate now {len(agg_obs)} pairs, retraining...")

        train_w = RECENCY_DECAY ** (round_idx + 1 - agg_round)
        model = pretrain_behavior_cloning(model, obs=agg_obs, actions=agg_actions,
                                           weights=train_w, epochs=retrain_epochs)

        torch.save(model.state_dict(), out_path)

    torch.save(model.state_dict(), out_path)
    print(f"saved DAgger-refined weights to {out_path}")
    return model, history


if __name__ == "__main__":
    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    dagger(
        gains_by_dist,
        n_rounds=5, num_episodes_per_pair=3,
        checkpoint_path="app/control/pretrained_bc.pt",
        out_path="app/control/pretrained_bc_dagger.pt",
        demo_path="app/control/demonstrations_omni.npz",
        distances=(3, 10),
    )
