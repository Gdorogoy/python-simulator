"""Deterministic (start_pos, target_pos) pairs for scoring a policy the same way
every run, instead of depending on whatever random target env.reset() samples."""
import numpy as np
import torch

DEFAULT_DISTANCES = (3, 5, 10, 25, 50, 100, 150, 250)


def build_eval_pairs(oob_radius, distances=DEFAULT_DISTANCES, base=None, margin=0.9, axes=(0, 1)):
    """Generates (start, target) pairs at each distance, offset from `base`
    (default (0,0,5)) along each axis, both signs. oob_radius is required: pairs
    beyond oob_radius*margin are skipped (would terminate "oob" immediately), as
    are -z pairs that would put the target underground."""
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
                pairs.append((base.copy(), target))
    return pairs


# Default set built against a generous oob_radius so all of DEFAULT_DISTANCES
# survives the margin filter. Callers under a smaller oob_radius should call
# build_eval_pairs(...) themselves with their own value.
EVAL_PAIRS = build_eval_pairs(oob_radius=300)


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
    """get_action(obs, env) for a PIDHoverController."""
    def get_action(obs, env):
        return pid.compute_action(env.drone_state, env.target_pos)
    return get_action


def run_eval_matrix(env, get_action, pairs=EVAL_PAIRS, n_repeats=5, max_steps=2000, on_episode_reset=None):
    """Runs get_action(obs, env) through env for every (start, target) pair,
    n_repeats times each, and returns one summary dict per pair: task_dist,
    mean/std_final_dist, hit_rate, mean_steps, mean_hit_time_sec (seconds of sim
    time to hit/hover-success, over only the runs that succeeded). on_episode_reset,
    if given, runs after each env.reset() (e.g. pid.reset())."""
    results = []
    for start_pos, target_pos in pairs:
        task_dist = float(np.linalg.norm(target_pos - start_pos))
        final_dists, hit_flags, steps_taken, hit_times_sec = [], [], [], []

        for _ in range(n_repeats):
            obs, _ = env.reset(start_pos=start_pos.copy(), target_pos=target_pos.copy())
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
