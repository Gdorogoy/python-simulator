"""Deterministic (start_pos, target_pos) pairs for scoring a policy the same way
every run, instead of depending on whatever random target env.reset() samples."""
import itertools

import numpy as np
import torch

DEFAULT_DISTANCES = (3, 5, 10, 25, 50, 100, 150, 250)


def build_eval_pairs(oob_radius, distances=DEFAULT_DISTANCES, base=None, margin=0.9, axes=(0, 1), target_yaw=0.0):
    """Generates (start, target, target_yaw) triples at each distance, offset from
    `base` (default (0,0,5)) along each axis, both signs. oob_radius is required:
    pairs beyond oob_radius*margin are skipped (would terminate "oob" immediately),
    as are -z pairs that would put the target underground. target_yaw is fixed
    (not sampled) here -- this builder is meant for deterministic, reproducible
    eval, so yaw stays a plain scalar unless the caller passes a different one."""
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


# Default set built against a generous oob_radius so all of DEFAULT_DISTANCES
# survives the margin filter. Callers under a smaller oob_radius should call
# build_eval_pairs(...) themselves with their own value.
EVAL_PAIRS = build_eval_pairs(oob_radius=300)


def build_omni_eval_pairs(oob_radius, distances, base=None, margin=0.9, target_yaw=0.0):
    """Like build_eval_pairs, but moves every combination of axes at once
    instead of one axis at a time: per distance, every single-axis (x/y/z),
    two-axis diagonal (xy/xz/yz), and three-axis diagonal (xyz) direction,
    every sign combination, each scaled (offset per axis = d/sqrt(len(axes)))
    so the total Euclidean distance still equals exactly `d` -- distances
    stays a true distance ladder, not a per-axis component size. Same
    underground/oob filtering as build_eval_pairs. target_yaw is fixed here too
    -- this is still a deterministic lattice, not the randomized builder."""
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
    """Like build_omni_eval_pairs, but samples n_per_distance directions per
    distance uniformly at random over the full sphere, instead of a fixed
    axis-aligned/equal-component-diagonal lattice -- build_omni_eval_pairs only
    covers directions where every nonzero component has the SAME magnitude
    (e.g. an xyz corner is always (d/sqrt(3), d/sqrt(3), d/sqrt(3))-shaped), so
    it never generalizes to an arbitrary skewed direction like (40, 15, 10).
    distances stay exactly the values passed (a true ladder, not randomized);
    only direction is dense/random. Same underground filtering as
    build_eval_pairs, oversampling via rejection to hit n_per_distance kept
    per distance. sample_yaw=True also draws a uniform target_yaw in
    [-pi, pi) per pair -- this is the builder actually used by training, so
    it's the one that needs to expose the policy to the full yaw-goal task,
    not just yaw=0."""
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
    """Like build_random_omni_eval_pairs, but instead of a fixed distance
    ladder, draws n_pairs targets with direction uniform over the full sphere
    AND magnitude uniform in [low, high] -- a continuous distance range (e.g.
    Uniform(3, 10)) rather than a handful of discrete rungs, so training sees
    every distance in between instead of just the ladder points.

    Distance is drawn ONCE per accepted pair and only the DIRECTION is
    re-sampled on an underground rejection -- re-drawing both together (as
    build_random_omni_eval_pairs does, harmlessly, since its distances are a
    fixed ladder rather than a continuous range) would skew the distance
    distribution here: an underground-pointing direction is far more likely
    at d=10 than at d=3 (for base z=5, any direction is safe up to d=5, then
    the rejection probability grows with d), so redrawing d along with a
    rejected v systematically under-samples the far end of [low, high]."""
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
    """Runs get_action(obs, env) through env for every (start, target, target_yaw)
    triple, n_repeats times each, and returns one summary dict per pair: task_dist,
    mean/std_final_dist, hit_rate, mean_steps, mean_hit_time_sec (seconds of sim
    time to hit/hover-success, over only the runs that succeeded). on_episode_reset,
    if given, runs after each env.reset() (e.g. pid.reset())."""
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
