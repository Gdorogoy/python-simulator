"""
Fixed evaluation matrix -- a deterministic set of (start_pos, target_pos) pairs
used to score a policy (RL checkpoint or the PID controller) the same way every
time, so results are comparable across checkpoints/runs instead of depending on
whatever random target that run's env.reset() happened to sample.
"""
import numpy as np
import torch

DEFAULT_DISTANCES = (3, 5, 10, 25, 50, 100, 150, 250)


def build_eval_pairs(oob_radius, distances=DEFAULT_DISTANCES, base=None, margin=0.9, axes=(0, 1)):
    """
    Generates (start, target) pairs at each distance in `distances`, offset
    from `base` (default (0,0,5)) along each axis in `axes`, both signs.
    Default axes=(0,1) is x,y only (4 cardinal directions) -- matches every
    existing BC/DAgger demonstration (collect_demonstrations.py, dagger.py),
    so leave it alone for anything that has to match that data. Pass
    axes=(0,1,2) to also include z (6 directions total) once you have
    z-inclusive demonstration data to match.

    oob_radius is REQUIRED (no default) -- it must match whatever RewardConfig
    the caller is actually evaluating under. Any distance whose pair would put
    the target beyond oob_radius * margin is skipped, since that episode would
    just terminate "oob" immediately and tell you nothing about the policy --
    e.g. build_eval_pairs(oob_radius=70) silently drops the 100/150/250m
    entries from DEFAULT_DISTANCES; pass a bigger oob_radius to cover them.

    A -z pair that would put the target underground (target z < 0, e.g.
    base=(0,0,5) with a 150m -z offset lands at z=-145) is skipped too --
    unlike +x/-x or +y/-y, +z/-z aren't symmetric once the ground is in the
    way, so this drops the unreachable half rather than generating a pair
    the drone could only reach by crashing through the floor first.
    """
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


# Backward-compatible default set, built against a generous oob_radius so all
# of DEFAULT_DISTANCES (up to 250m) survives the margin filter. Callers
# evaluating under a smaller oob_radius should call build_eval_pairs(...)
# themselves with their own value instead of using this constant.
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
    """
    Runs get_action(obs, env) -> action through env for every (start, target)
    pair in `pairs`, n_repeats times each (policies are stochastic and PID
    integral state resets, so one run per pair is noisy), and returns one
    summary dict per pair:
        task_dist        -- ||target - start||, the "correct"/expected distance closed
        mean_final_dist  -- mean achieved distance-to-target at episode end
        std_final_dist   -- across the n_repeats runs
        hit_rate         -- fraction of runs that terminated with reason == "Hit"
                             or reached hover_success
        mean_steps       -- mean steps taken (hit or timeout)
    on_episode_reset, if given, is called with no args after each env.reset()
    (e.g. pid.reset()) before the episode starts.
    """
    results = []
    for start_pos, target_pos in pairs:
        task_dist = float(np.linalg.norm(target_pos - start_pos))
        final_dists, hit_flags, steps_taken = [], [], []

        for _ in range(n_repeats):
            obs, _ = env.reset(start_pos=start_pos.copy(), target_pos=target_pos.copy())
            if on_episode_reset is not None:
                on_episode_reset()

            done = False
            step = 0
            reason = None
            while not done and step < max_steps:
                action = get_action(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                done = terminated or truncated
                reason = info["reason"]

            final_dists.append(env.prev_distance)
            hit_flags.append(reason == "Hit" or getattr(env, "hover_success_achieved", False))
            steps_taken.append(step)

        results.append({
            "start": start_pos.tolist(),
            "target": target_pos.tolist(),
            "task_dist": task_dist,
            "mean_final_dist": float(np.mean(final_dists)),
            "std_final_dist": float(np.std(final_dists)),
            "hit_rate": float(np.mean(hit_flags)),
            "mean_steps": float(np.mean(steps_taken)),
        })

    return results
