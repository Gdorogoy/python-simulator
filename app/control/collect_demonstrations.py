<<<<<<< HEAD
import json
import numpy as np

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.control.pid_hover import PIDHoverController
from app.reward_functions.rewards import RewardConfig, make_reward_fn


def collect_demonstrations(pid, n_episodes=50, n_steps=600,
                            save_path="app/control/demonstrations.npz"):
    reward_cfg = RewardConfig(oob_radius=7, hover_success_steps=None)
    reward_fn = make_reward_fn(reward_cfg)
    env = InterceptorDroneEnv(reward_fn)

    all_obs = []
    all_actions = []

    for ep in range(n_episodes):
        offset = np.random.uniform(-0.2, 0.2, size=3)  # random small displacement from target
        start = np.array([0, 0, 5], dtype=np.float32) + offset
        target = np.array([0, 0, 5], dtype=np.float32)
        obs, _ = env.reset(start_pos=start, target_pos=target)

        for s in range(n_steps):
            action = pid.compute_action(env.drone_state, env.target_pos)

            # record the STATE the policy would see and the ACTION the PID
            # took in response to it -- this pairing is what BC learns from
            all_obs.append(obs.copy())
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        if ep % 10 == 0:
            print(f"episode {ep}/{n_episodes} -- {len(all_obs)} pairs collected so far")

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)

    np.savez(save_path, obs=all_obs, actions=all_actions)
    print(f"saved {len(all_obs)} (state, action) pairs to {save_path}")
    print(f"obs shape: {all_obs.shape}, actions shape: {all_actions.shape}")


if __name__ == "__main__":
    best_params={
                'kp_pos': 2.3348549042980946,
                 'kd_pos': 1.0978513647624821,
                 'kp_att': 7.882461182173537,
                 'kd_att': 1.0936341247316574,
                 'kp_yaw': 0.048930591736932,
                 'kd_yaw': 0.20306927726482113

                 }


    pid = PIDHoverController(**best_params)
    collect_demonstrations(pid, n_episodes=50, n_steps=600)
=======
import json

import numpy as np

from app.environmental.base_drone_env import BaseDroneEnv
from app.control.pid import PIDController
from app.control.tune_pid import DISTANCES, steps_for_dist
from app.training.eval_matrix import build_eval_pairs
from app.reward_functions.rewards import reward_func

# Hard ceiling on a single demonstration/imitation collection run, in (obs, action)
# pairs. Without this, n_target_rows has no upper bound and a mistyped value can
# silently commit to a multi-hour collection run.
IMITATION_ROWS_CAP = 1_000_000

# Matches app.training.base_training.DEFAULT_DISTANCE_LOW/HIGH (not imported to avoid that module's mlflow overhead).
DEFAULT_DISTANCE_LOW = 3.0
DEFAULT_DISTANCE_HIGH = 10.0


def _nearest_gain_key(gains_by_dist: dict, dist: float) -> str:
    """Picks the gains_by_dist key closest to a continuously-sampled dist -- gains are
    only tuned at a handful of discrete distances. Mirrors torch_pid.assign_gains_by_distance."""
    return min(gains_by_dist.keys(), key=lambda k: abs(float(k) - dist))


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
        pid = PIDController(**gains)
        oob_radius = max(20.0, dist * 3.0)
        n_steps = steps_for_dist(dist)

        # reward_func (not the deprecated RewardFnPhase1 roadmap) -- the current
        # reward version, matching base_training.py. Its own oob_radius scales
        # with env.start_dist (see rewards.reward_func), so it's safe across the
        # full distance curriculum; the oob_radius computed above is only used
        # for build_eval_pairs' margin filter below, not passed into the env.
        env = BaseDroneEnv(reward_func, max_steps=n_steps)

        for start, target, target_yaw in build_eval_pairs(oob_radius=oob_radius, distances=(dist,), axes=(0, 1, 2)):
            for ep in range(n_episodes_per_pair):
                # Jitter around the nominal start gives BC a neighborhood of corrective
                # examples, not one noiseless curve, so it has signal for off-path states.
                jittered_start = start + np.random.uniform(-0.15, 0.15, size=3).astype(np.float32)
                obs, _ = env.reset(start_pos=jittered_start.astype(np.float32), target_pos=target.copy(),
                                    target_yaw=target_yaw)
                pid.reset()

                for s in range(n_steps):
                    action = pid.compute_action(env.drone_state, env.target_pos, env.target_yaw)

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


def sample_full_sphere_target(rng, dist, base=None):
    """Draws a target at exactly `dist` from `base` (default (0,0,5)) in a uniformly-random
    direction over the full sphere (rng.normal(size=3) normalized, the standard unbiased method
    for a uniform point on a sphere). Returns None if the direction points underground."""
    if base is None:
        base = np.array([0, 0, 5], dtype=np.float32)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    target = (base + (v * dist).astype(np.float32)).astype(np.float32)
    if target[2] < 0:
        return None
    return target


def collect_demonstrations_omni(gains_by_dist, n_target_rows=750_000,
                                 save_path="app/control/demonstrations_omni.npz",
                                 distance_low=DEFAULT_DISTANCE_LOW, distance_high=DEFAULT_DISTANCE_HIGH,
                                 seed=None, log_every=200):
    """Like collect_demonstrations, but samples a fresh, fully-random target direction
    (sample_full_sphere_target) and a continuous Uniform(distance_low, distance_high) distance
    every episode, until n_target_rows pairs are collected. PID gains picked by nearest-distance match."""
    if n_target_rows > IMITATION_ROWS_CAP:
        raise ValueError(f"n_target_rows={n_target_rows} exceeds IMITATION_ROWS_CAP={IMITATION_ROWS_CAP}")

    rng = np.random.default_rng(seed)
    all_obs = []
    all_actions = []
    episodes = 0

    # reward_func (not the deprecated RewardFnPhase1 roadmap) -- see the same
    # swap in collect_demonstrations() above; oob_radius scales with
    # env.start_dist internally, safe across the full distance range.
    # max_steps sized for the longest episode this range can produce; shorter
    # per-episode distances just break out of the range(n_steps) loop early.
    env = BaseDroneEnv(reward_func, max_steps=steps_for_dist(distance_high))

    while len(all_obs) < n_target_rows:
        dist = float(rng.uniform(distance_low, distance_high))
        target = None
        while target is None:
            target = sample_full_sphere_target(rng, dist)  # rejects underground, resamples

        pid = PIDController(**gains_by_dist[_nearest_gain_key(gains_by_dist, dist)])
        n_steps = steps_for_dist(dist)

        start = np.array([0, 0, 5], dtype=np.float32) + rng.uniform(-0.15, 0.15, size=3).astype(np.float32)
        target_yaw = float(rng.uniform(-np.pi, np.pi))
        obs, _ = env.reset(start_pos=start, target_pos=target.copy(), target_yaw=target_yaw)
        pid.reset()

        for s in range(n_steps):
            action = pid.compute_action(env.drone_state, env.target_pos, env.target_yaw)
            all_obs.append(obs.copy())
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        episodes += 1
        if episodes % log_every == 0:
            print(f"episodes={episodes} rows={len(all_obs)}/{n_target_rows} dist={dist:.1f}m")

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)

    np.savez(save_path, obs=all_obs, actions=all_actions)
    print(f"saved {len(all_obs)} (state, action) pairs to {save_path} over {episodes} episodes")
    print(f"obs shape: {all_obs.shape}, actions shape: {all_actions.shape}")


def collect_demonstrations_base_drone_isaac(env, gains_by_dist: dict, n_target_rows: int,
                                             distance_low: float = DEFAULT_DISTANCE_LOW,
                                             distance_high: float = DEFAULT_DISTANCE_HIGH,
                                             save_path: str = "app/control/demonstrations_isaac.npz",
                                             seed=None, log_every: int = 2000,
                                             pool_size: int = 1024, pool_refresh_every_rows: int = 50_000):
    """Batched counterpart to collect_demonstrations_omni: num_envs parallel episodes at once,
    saving the same (obs, actions) .npz shape pretrain_behavior_cloning expects. Targets are
    drawn from a shared pool (env.unwrapped.set_target_pairs), refreshed every
    pool_refresh_every_rows rows; PID gains picked per-env by nearest-distance match.

    Known simplification: env.unwrapped has one fixed max_episode_length for every env, not a
    per-distance cap -- fine for (3, 10)m, but distances beyond ~50-60m would get truncated."""
    import torch
    import isaaclab.utils.math as math_utils
    from app.control.torch_pid import TorchPIDController, assign_gains_by_distance

    unwrapped = env.unwrapped
    device = unwrapped.device
    rng = np.random.default_rng(seed)

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
    obs = env.reset()[0]["policy"]
    all_env_ids = torch.arange(unwrapped.num_envs, device=device)
    assign_gains_by_distance(pid, unwrapped._start_dist, gains_by_dist, all_env_ids)
    pid.reset()

    all_obs, all_actions = [], []
    total_steps = 0
    rows_since_refresh = 0
    while len(all_obs) * unwrapped.num_envs < n_target_rows:
        pos = unwrapped._robot.data.root_pos_w - unwrapped._terrain.env_origins
        vel = unwrapped._robot.data.root_lin_vel_w
        ang_vel = unwrapped._robot.data.root_ang_vel_b
        quat_wxyz = unwrapped._robot.data.root_quat_w
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_wxyz)
        target_local = unwrapped._desired_pos_w - unwrapped._terrain.env_origins

        with torch.no_grad():
            action = pid.compute_action(pos, vel, ang_vel, roll, pitch, yaw,
                                         target_local, unwrapped._desired_yaw_w, dt=1 / 240)

        all_obs.append(obs.cpu().numpy())
        all_actions.append(action.cpu().numpy())

        next_obs_dict, reward, terminated, truncated, extras = env.step(action)
        obs = next_obs_dict["policy"]

        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            assign_gains_by_distance(pid, unwrapped._start_dist, gains_by_dist, done_ids)
            pid.reset(done_ids)

        total_steps += 1
        rows_since_refresh += unwrapped.num_envs
        if rows_since_refresh >= pool_refresh_every_rows:
            _refresh_pool()
            rows_since_refresh = 0
        if total_steps % log_every == 0:
            print(f"steps={total_steps} rows={len(all_obs) * unwrapped.num_envs}/{n_target_rows}")

    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    np.savez(save_path, obs=all_obs, actions=all_actions)
    print(f"saved {len(all_obs)} (state, action) pairs to {save_path}")
    print(f"obs shape: {all_obs.shape}, actions shape: {all_actions.shape}")
    return all_obs, all_actions


if __name__ == "__main__":
    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    collect_demonstrations_omni(gains_by_dist, n_target_rows=150_000, distance_low=3.0, distance_high=10.0)
>>>>>>> later_to_remove
