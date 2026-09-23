"""Loads any ActorCritic checkpoint (.pt) -- regardless of which training
script produced it or what hidden/num_hidden_layers/dropout it used -- runs
one deterministic episode against it in the numpy BaseDroneEnv, and renders
the flight (3D trajectory + drone marker + target) as an mp4. No PyBullet/
Isaac Sim needed; BaseDroneEnv no longer renders (see its docstring), so this
does its own matplotlib rendering instead.

Usage:
    python -m app.guidance.record_run [checkpoint_path] [--episodes N] \\
        [--out runs/demo.mp4] [--distance-low 3] [--distance-high 10] [--seed 0]

With no checkpoint_path, picks the most recently modified .pt under runs/ or
app/control/.
"""
import argparse
import glob
import os

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.environmental.base_drone_env import BaseDroneEnv
from app.guidance.train import ActorCritic, load_bc_checkpoint, device
from app.guidance.utils import compute_grade
from app.reward_functions.rewards import reward_func, HIT_THRESHOLD

PHYSICS_DT = 1.0 / 240.0

CHECKPOINT_SEARCH_DIRS = ("runs", "app/control")
MAX_RENDER_FRAMES = 360  # cap on frames actually drawn -- keeps render time/output size bounded
FPS = 24
N_EVAL_SCENARIOS = 75


def find_latest_checkpoint(search_dirs=CHECKPOINT_SEARCH_DIRS):
    candidates = []
    for d in search_dirs:
        candidates += glob.glob(os.path.join(d, "**", "*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"no .pt checkpoints found under {search_dirs}")
    return max(candidates, key=os.path.getmtime)


def load_model_for_inference(checkpoint_path, map_location=None):
    """Infers obs_dim/action_dim/hidden/num_hidden_layers directly from the
    checkpoint's own tensor shapes (no need to know PARAMS/BEST_PARAMS of
    whichever training script produced it), builds a dropout=0 model, and
    remaps the checkpoint onto it the same way load_bc_checkpoint already
    does for a dropout>0 source -- see that function's docstring."""
    map_location = map_location or device
    raw = torch.load(checkpoint_path, map_location=map_location)
    linear_idxs = sorted({
        int(k.split(".")[1]) for k in raw
        if k.startswith("shared.") and k.split(".")[1].isdigit() and k.split(".")[2] in ("weight", "bias")
    })
    if not linear_idxs:
        raise ValueError(f"{checkpoint_path} has no shared.*.weight keys -- not an ActorCritic checkpoint")
    hidden, obs_dim = raw[f"shared.{linear_idxs[0]}.weight"].shape
    action_dim = raw["actor_mean.weight"].shape[0]

    model = ActorCritic(obs_dim, action_dim, torch.zeros(action_dim), torch.ones(action_dim),
                         hidden=hidden, num_hidden_layers=len(linear_idxs), dropout=0.0).to(device)
    load_bc_checkpoint(model, checkpoint_path, map_location=map_location)
    model.eval()
    return model


def sample_target(distance_low, distance_high, rng):
    dist = rng.uniform(distance_low, distance_high)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    start_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    target_pos = (start_pos + direction * dist).astype(np.float32)
    target_pos[2] = max(target_pos[2], 0.5)  # keep target above ground
    return start_pos, target_pos


def run_episode(model, env, start_pos, target_pos, target_yaw=0.0):
    obs, _ = env.reset(start_pos=start_pos, target_pos=target_pos, target_yaw=target_yaw)
    positions, dists, rewards = [], [], []
    done = False
    info = {"reason": None}
    total_reward = 0.0
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = model.forward(obs_t)
            action = model.scale_action(mean).squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        positions.append([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        dists.append(env.prev_distance)
        done = terminated or truncated
    return np.array(positions, dtype=np.float32), np.array(dists, dtype=np.float32), info, total_reward


def render_mp4(positions, dists, target_pos, out_path, reason, total_reward, fps=FPS):
    n = len(positions)
    stride = max(1, n // MAX_RENDER_FRAMES)
    frame_idxs = list(range(0, n, stride))
    if frame_idxs[-1] != n - 1:
        frame_idxs.append(n - 1)

    all_pts = np.vstack([positions, target_pos[None, :]])
    center = all_pts.mean(axis=0)
    span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max(), 2.0) * 0.6 + 0.5

    fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    frames = []
    for frame_i, i in enumerate(frame_idxs):
        ax.clear()
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(max(0.0, center[2] - span), center[2] + span)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        trail = positions[: i + 1]
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="tab:blue", linewidth=1.5, alpha=0.8)
        ax.scatter(*trail[-1], color="tab:blue", s=60, depthshade=False, label="drone")
        ax.scatter(*target_pos, color="tab:red", marker="x", s=80, label="target")

        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 8)
        r = HIT_THRESHOLD
        sx = target_pos[0] + r * np.outer(np.cos(u), np.sin(v))
        sy = target_pos[1] + r * np.outer(np.sin(u), np.sin(v))
        sz = target_pos[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color="tab:red", alpha=0.25, linewidth=0.5)

        ax.view_init(elev=22, azim=(frame_i * 0.6) % 360)
        ax.set_title(f"step {i}/{n - 1}  dist={dists[i]:.2f}m")
        ax.legend(loc="upper left", fontsize=8)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)

    plt.close(fig)

    outcome_text = f"outcome={reason}  reward={total_reward:.1f}"
    print(f"[record_run] rendering {len(frames)} frames -> {out_path}  ({outcome_text})")
    iio.imwrite(out_path, frames, fps=fps, codec="libx264", plugin="FFMPEG")
    return out_path


def record_checkpoint(checkpoint_path, out_path, distance_low=3.0, distance_high=10.0, seed=None):
    """Runs one deterministic episode against checkpoint_path and writes an
    mp4 to out_path. Returns a small dict of episode stats (used by
    serve_run.py's API response)."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)

    env = BaseDroneEnv(reward_func, max_steps=6000)
    start_pos, target_pos = sample_target(distance_low, distance_high, rng)

    positions, dists, info, total_reward = run_episode(model, env, start_pos, target_pos)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    render_mp4(positions, dists, target_pos, out_path, info["reason"], total_reward)

    return {
        "reason": info["reason"],
        "steps": int(len(positions)),
        "final_dist": float(dists[-1]) if len(dists) else None,
        "total_reward": float(total_reward),
        "start_pos": start_pos.tolist(),
        "target_pos": target_pos.tolist(),
    }


def _fmt_vec(v):
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


def _classify_outcome(reason):
    """reason may be None (timeout) or a "+"-joined compound string (e.g. "oob+attitude-ROLL")."""
    parts = reason.split("+") if reason else []
    return {
        "hit": reason == "Hit",
        "oob": "oob" in parts,
        "attitude": any(p.startswith("attitude") for p in parts),
        "timeout": reason is None,
    }


def evaluate_checkpoint(checkpoint_path, n_scenarios=N_EVAL_SCENARIOS, distance_low=3.0, distance_high=10.0, seed=None):
    """Runs checkpoint_path deterministically against n_scenarios random (start, target)
    scenarios, no rendering -- returns per-scenario outcomes, totals, and a text summary."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = BaseDroneEnv(reward_func, max_steps=6000)

    scenarios = []
    totals = {"hit": 0, "oob": 0, "attitude": 0, "timeout": 0}
    for i in range(n_scenarios):
        start_pos, target_pos = sample_target(distance_low, distance_high, rng)
        dist = float(np.linalg.norm(target_pos - start_pos))
        positions, dists, info, total_reward = run_episode(model, env, start_pos, target_pos)
        outcome = _classify_outcome(info["reason"])
        for k, v in outcome.items():
            totals[k] += int(v)
        final_dist = float(dists[-1]) if len(dists) else None
        scenarios.append({
            "idx": i, "spawn": start_pos.tolist(), "target": target_pos.tolist(), "dist": dist,
            "reason": info["reason"] or "timeout", "final_dist": final_dist, "steps": int(len(positions)),
            **outcome,
        })
        print(f"[evaluate_checkpoint] {i + 1}/{n_scenarios} dist={dist:.1f}m "
              f"reason={info['reason'] or 'timeout'} final_dist={final_dist}")

    pct = {k: 100.0 * v / n_scenarios for k, v in totals.items()}
    lines = [f"Checkpoint: {checkpoint_path}",
             f"Scenarios: {n_scenarios}  distance ~ Uniform({distance_low}, {distance_high})", ""]
    for s in scenarios:
        lines.append(f"[{s['idx']:3d}] dist={s['dist']:5.2f}m spawn={_fmt_vec(s['spawn'])} "
                      f"target={_fmt_vec(s['target'])} -> {s['reason']:<16} "
                      f"final_dist={s['final_dist']:.2f}m steps={s['steps']}")
    lines.append("")
    lines.append("Totals:")
    for k in ("hit", "oob", "attitude", "timeout"):
        lines.append(f"  {k:<10} {totals[k]:3d}/{n_scenarios}  ({pct[k]:.1f}%)")
    summary_text = "\n".join(lines)

    return {"checkpoint": checkpoint_path, "scenarios": scenarios, "totals": totals,
            "pct": pct, "summary_text": summary_text}


def rank_checkpoints(checkpoint_paths, n_scenarios=N_EVAL_SCENARIOS, distance_low=3.0, distance_high=10.0,
                      seed=None, top_n=5):
    """Batch-evaluates each of checkpoint_paths against n_scenarios scenarios (no per-scenario
    dump, no video) and returns only the top_n ranked by grade -- a short leaderboard of
    potential suitable weights."""
    oob_radius = max(20.0, distance_high * 3.0)
    max_steps = 6000
    results = []
    for path in checkpoint_paths:
        rng = np.random.default_rng(seed)
        model = load_model_for_inference(path)
        env = BaseDroneEnv(reward_func, max_steps=max_steps)

        totals = {"hit": 0, "oob": 0, "attitude": 0, "timeout": 0}
        hit_times, final_dists = [], []
        for _ in range(n_scenarios):
            start_pos, target_pos = sample_target(distance_low, distance_high, rng)
            positions, dists, info, _ = run_episode(model, env, start_pos, target_pos)
            outcome = _classify_outcome(info["reason"])
            for k, v in outcome.items():
                totals[k] += int(v)
            if outcome["hit"]:
                hit_times.append(len(positions) * PHYSICS_DT)
            if len(dists):
                final_dists.append(float(dists[-1]))

        success_rate = totals["hit"] / n_scenarios
        avg_final_dist = float(np.mean(final_dists)) if final_dists else oob_radius
        avg_hit_time_sec = float(np.mean(hit_times)) if hit_times else None
        grade, _ = compute_grade(success_rate=success_rate, avg_final_dist=avg_final_dist,
                                  avg_hit_time_sec=avg_hit_time_sec, avg_grad_norm=0.0,
                                  oob_radius=oob_radius, episode_time_budget_sec=max_steps * PHYSICS_DT)
        pct = {k: 100.0 * v / n_scenarios for k, v in totals.items()}
        results.append({"checkpoint": path, "grade": grade, **pct})
        print(f"[rank_checkpoints] {os.path.basename(path)}: grade={grade:.3f} "
              f"hit={pct['hit']:.0f}% oob={pct['oob']:.0f}% attitude={pct['attitude']:.0f}% timeout={pct['timeout']:.0f}%")

    results.sort(key=lambda r: r["grade"], reverse=True)
    top = results[:top_n]
    lines = [f"Potential suitable weights (top {len(top)} of {len(results)} by grade):"]
    for i, r in enumerate(top, 1):
        lines.append(f"{i}. {os.path.basename(r['checkpoint'])}  grade={r['grade']:.3f}  "
                      f"hit={r['hit']:.0f}%  oob={r['oob']:.0f}%  attitude={r['attitude']:.0f}%  timeout={r['timeout']:.0f}%")
    summary_text = "\n".join(lines)

    return {"results": results, "top": top, "summary_text": summary_text}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=None, help="path to a .pt checkpoint (default: latest)")
    parser.add_argument("--out", default="runs/demo.mp4")
    parser.add_argument("--distance-low", type=float, default=3.0)
    parser.add_argument("--distance-high", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n-scenarios", type=int, default=N_EVAL_SCENARIOS)
    parser.add_argument("--rank-dir", default=None, help="evaluate every .pt in this dir, print top --top-n by grade")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.rank_dir:
        checkpoints = sorted(glob.glob(os.path.join(args.rank_dir, "*.pt")))
        result = rank_checkpoints(checkpoints, n_scenarios=args.n_scenarios, distance_low=args.distance_low,
                                   distance_high=args.distance_high, seed=args.seed, top_n=args.top_n)
        print(result["summary_text"])
    else:
        checkpoint = args.checkpoint or find_latest_checkpoint()
        print(f"[record_run] checkpoint: {checkpoint}")
        if args.evaluate:
            result = evaluate_checkpoint(checkpoint, n_scenarios=args.n_scenarios, distance_low=args.distance_low,
                                          distance_high=args.distance_high, seed=args.seed)
            print(result["summary_text"])
        else:
            stats = record_checkpoint(checkpoint, args.out, distance_low=args.distance_low,
                                       distance_high=args.distance_high, seed=args.seed)
            print(f"[record_run] done: {stats}")
