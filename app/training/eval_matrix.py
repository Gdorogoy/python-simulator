"""Deterministic (start_pos, target_pos) pairs for scoring a policy the same way
every run, instead of depending on whatever random target env.reset() samples."""
import itertools

import numpy as np
import torch

DEFAULT_DISTANCES = (3, 5, 10, 25, 50, 100, 150, 250)


def build_eval_pairs(oob_radius, distances=DEFAULT_DISTANCES, base=None, margin=0.9, axes=(0, 1), target_yaw=0.0):
    """Deterministic (start, target, yaw) triples along each axis (both signs) per distance; oob/underground filtered."""
    if base is None:
        base = np.array([0, 0, 5], dtype=np.float32)
    max_allowed = oob_radius * margin

    pairs = []
    for d in distances:
        if d > max_allowed:
            continue
        for axis in axes:
            for sign in (1, -1):
                offset = np.zeros(3, dtype=np.float32)
                offset[axis] = sign * d
                target = (base + offset).astype(np.float32)
                if target[2] < 0:
                    continue
                pairs.append((base.copy(), target, float(target_yaw)))
    return pairs


# default set built with a generous oob_radius; build your own for smaller radii
EVAL_PAIRS = build_eval_pairs(oob_radius=300)


def build_omni_eval_pairs(oob_radius, distances, base=None, margin=0.9, target_yaw=0.0):
    """Like build_eval_pairs over every single/pair/triple-axis direction, scaled so |offset| == d."""
    if base is None:
        base = np.array([0, 0, 5], dtype=np.float32)
    max_allowed = oob_radius * margin

    axis_groups = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    pairs = []
    for d in distances:
        if d > max_allowed:
            continue
        for axes in axis_groups:
            component = d / np.sqrt(len(axes))
            for signs in itertools.product((1, -1), repeat=len(axes)):
                offset = np.zeros(3, dtype=np.float32)
                for axis, sign in zip(axes, signs):
                    offset[axis] = sign * component
                target = (base + offset).astype(np.float32)
                if target[2] < 0:
                    continue
                pairs.append((base.copy(), target, float(target_yaw)))
    return pairs


def build_random_omni_eval_pairs(oob_radius, distances, base=None, n_per_distance=200, margin=0.9, rng=None,
                                  sample_yaw=True):
    """n_per_distance uniformly random directions per distance (optionally random yaw); used by training."""
    if base is None:
        base = np.array([0, 0, 5], dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()
    max_allowed = oob_radius * margin

    pairs = []
    for d in distances:
        if d > max_allowed:
            continue
        kept = 0
        while kept < n_per_distance:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            target = (base + (v * d).astype(np.float32)).astype(np.float32)
            if target[2] < 0:
                continue
            target_yaw = float(rng.uniform(-np.pi, np.pi)) if sample_yaw else 0.0
            pairs.append((base.copy(), target, target_yaw))
            kept += 1
    return pairs


def build_uniform_omni_eval_pairs(oob_radius, low, high, n_pairs, base=None, margin=0.9, rng=None,
                                   sample_yaw=True):
    """n_pairs targets: uniform direction, distance ~ U(low, high); only the direction is resampled when underground."""
    if base is None:
        base = np.array([0, 0, 5], dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()
    max_allowed = oob_radius * margin

    pairs = []
    for _ in range(n_pairs):
        d = rng.uniform(low, min(high, max_allowed))
        target = None
        while target is None:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            candidate = (base + (v * d).astype(np.float32)).astype(np.float32)
            if candidate[2] >= 0:
                target = candidate
        target_yaw = float(rng.uniform(-np.pi, np.pi)) if sample_yaw else 0.0
        pairs.append((base.copy(), target, target_yaw))
    return pairs


def make_model_action_fn(model, device="cpu"):
    """get_action(obs, env) for an ActorCritic checkpoint -- deterministic (mean) action."""
    def get_action(obs, env):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = model.forward(obs_t)
            action = model.scale_action(mean)
        return action.squeeze(0).cpu().numpy()
    return get_action


def make_pid_action_fn(pid):
    """get_action(obs, env) for a PIDController."""
    def get_action(obs, env):
        return pid.compute_action(env.drone_state, env.target_pos, env.target_yaw)
    return get_action


def run_eval_matrix(env, get_action, pairs=EVAL_PAIRS, n_repeats=5, max_steps=2000, on_episode_reset=None):
    """Run get_action over every triple n_repeats times -> per-pair hit rate, final distance, steps, hit time."""
    results = []
    for start_pos, target_pos, target_yaw in pairs:
        task_dist = float(np.linalg.norm(target_pos - start_pos))
        final_dists, hit_flags, steps_taken, hit_times_sec = [], [], [], []

        for _ in range(n_repeats):
            obs, _ = env.reset(start_pos=start_pos.copy(), target_pos=target_pos.copy(), target_yaw=target_yaw)
            if on_episode_reset is not None:
                on_episode_reset()

            done = False
            step = 0
            reason = None
            info = {}
            while not done and step < max_steps:
                action = get_action(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                done = terminated or truncated
                reason = info["reason"]

            final_dists.append(env.prev_distance)
            hit_flags.append(reason == "Hit" or getattr(env, "hover_success_achieved", False))
            steps_taken.append(step)
            if info.get("hit_time_sec") is not None:
                hit_times_sec.append(info["hit_time_sec"])

        results.append({
            "start": start_pos.tolist(),
            "target": target_pos.tolist(),
            "task_dist": task_dist,
            "mean_final_dist": float(np.mean(final_dists)),
            "std_final_dist": float(np.std(final_dists)),
            "hit_rate": float(np.mean(hit_flags)),
            "mean_steps": float(np.mean(steps_taken)),
            "mean_hit_time_sec": float(np.mean(hit_times_sec)) if hit_times_sec else None,
        })

    return results
