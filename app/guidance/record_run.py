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
import json
import os
import time

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.control.step_budget import steps_for_dist
from app.control.two_phase import DEFAULT_SWITCH_DIST, TwoPhaseAgent
from app.environmental.base_drone_env import BaseDroneEnv
from app.environmental.subproc_vec_base_drone_env import _select_pid_teacher
from app.guidance.train import ActorCritic, load_bc_checkpoint, device
from app.guidance.utils import compute_grade
from app.reward_functions.rewards import reward_func, HIT_THRESHOLD

PHYSICS_DT = 1.0 / 240.0

CHECKPOINT_SEARCH_DIRS = ("runs", "app/control")
MAX_RENDER_FRAMES = 360  # cap on frames actually drawn -- keeps render time/output size bounded
FPS = 24
N_EVAL_SCENARIOS = 75
# Must match base_training_isaac.PidResidualEnv's composition exactly, or a residual checkpoint
# (--residual-scale > 0 at training time -- e.g. every _res_v1/_v1 Isaac run as of 2026-09-27)
# evaluates as "hover, don't steer": its actor head only ever learned a SMALL CORRECTION on top of
# the PID (zero-initialized, scaled down at training time too), not a full action -- feeding it alone
# to the env makes every episode time out, regardless of how good the correction actually is.
RESIDUAL_GAINS_PATH = "app/control/best_pid_gains_per_dist.json"


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


def sample_target_at_distance(dist, rng):
    """Like sample_target, but places the target at EXACTLY dist (random
    direction, not a random distance in a range) -- what the two-phase UI's
    single "target distance" field means."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    start_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    target_pos = (start_pos + direction * dist).astype(np.float32)
    if target_pos[2] < 0.5:  # keep target above ground, preserving the exact distance
        direction[2] = abs(direction[2])
        target_pos = (start_pos + direction * dist).astype(np.float32)
        target_pos[2] = max(target_pos[2], 0.5)
    return start_pos, target_pos


def run_episode(model, env, start_pos, target_pos, target_yaw=0.0, residual_scale=0.0, gains_by_dist=None):
    """residual_scale > 0: executed action = pid_action + residual_scale * model_action (clipped to the
    env's action space) -- the same composition base_training_isaac.PidResidualEnv applies during
    training, required to get a meaningful episode out of a residual checkpoint (see RESIDUAL_GAINS_PATH's
    comment). gains_by_dist re-selects env.pid_teacher's gains by this episode's target distance right
    after reset, matching _select_pid_teacher's per-episode swap during training/diagnostics."""
    obs, _ = env.reset(start_pos=start_pos, target_pos=target_pos, target_yaw=target_yaw)
    if residual_scale > 0:
        _select_pid_teacher(env, gains_by_dist)
        env.pid_teacher.reset()
    positions, dists, rewards = [], [], []
    done = False
    info = {"reason": None}
    total_reward = 0.0
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = model.forward(obs_t)
            action = model.scale_action(mean).squeeze(0).cpu().numpy()
        if residual_scale > 0:
            pid_action = env.pid_teacher.compute_action(env.drone_state, env.target_pos, env.target_yaw)
            action = np.clip(pid_action + residual_scale * action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        positions.append([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        dists.append(env.prev_distance)
        done = terminated or truncated
    return np.array(positions, dtype=np.float32), np.array(dists, dtype=np.float32), info, total_reward


# Hard ceiling on the per-episode step budget steps_for_dist(dist) can ask for (see
# step_budget.py: 750*dist/3 steps) -- without SOME cap, a typo'd/malicious distance
# (e.g. a stray extra zero) would block the synchronous /api/two_phase/* request
# indefinitely with zero progress feedback. Set to exactly steps_for_dist(MAX_TEST_DISTANCE)
# so distances up to and including MAX_TEST_DISTANCE always get their full, correctly-
# sized settle-time budget (the math compute_gains_for_distance/steps_for_dist use doesn't
# have a distance ceiling of its own -- it's closed-form pole placement, not a fit to the
# old 8-entry ladder -- so there's no correctness reason to cap lower than this; it's purely
# a wall-clock safety net). At MAX_TEST_DISTANCE=5000m this is 1_250_000 steps (~5208
# simulated seconds) -- expect a single episode at that range to take real wall-clock
# minutes, and a multi-run batch (n_scenarios) proportionally longer; run_episode_two_phase
# logs when a distance beyond this actually gets clamped.
MAX_TEST_DISTANCE = 5000.0
MAX_STEPS_CAP = steps_for_dist(MAX_TEST_DISTANCE)


def _fmt_elapsed(t0):
    """Wall-clock elapsed since t0 (time.monotonic()), as [HH:MM:SS.mmm] -- this
    is real time spent running/rendering the episode, not simulated flight
    time (dt=1/240 sim-seconds != wall-clock seconds). Millisecond, not
    nanosecond, resolution -- events within the same episode routinely land
    under a second apart (see e.g. a whole PID phase logging at [00:00:00]
    with whole-second formatting), and print()/the log buffer's own overhead
    already swamps anything finer than ~1ms, so sub-ms digits would be false
    precision, not real signal."""
    elapsed = time.monotonic() - t0
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def run_episode_two_phase(model, env, start_pos, target_pos, target_yaw=0.0,
                           switch_dist=DEFAULT_SWITCH_DIST, residual_scale=0.0,
                           gains_by_dist=None, log_events=None):
    """Like run_episode, but PID flies midcourse alone -- gains derived
    analytically for this episode's exact distance (app.control.two_phase.
    TwoPhaseAgent, tune_pid.compute_gains_for_distance) by default, OR
    nearest-neighbor-snapped from gains_by_dist if given -- until within
    switch_dist of the target, then the RL phase engages. residual_scale > 0:
    the RL phase composes pid_action + residual_scale*model_action instead of
    pure model output (PID keeps running, same gains, through the RL phase
    too) -- REQUIRED if model was trained with base_training_isaac.py's
    --residual-scale (see TwoPhaseAgent's docstring); must match the value
    training used. Pass gains_by_dist=_load_residual_gains(residual_scale) in
    that case too -- matches _select_pid_teacher's nearest-ladder-snap
    training actually used, not just the action composition; analytic gains
    are a DIFFERENT PID baseline than what a residual correction was trained
    against. switch_dist should sit at or inside the checkpoint's own trained
    distance range for residual_scale > 0 -- see TwoPhaseAgent's docstring
    for the tradeoff. Caller is responsible for sizing env.max_steps (see
    _set_episode_budget) before calling this. log_events, if given, collects
    timestamped [HH:MM:SS.mmm] state-change lines (init / pid guide / rl takes over / done) -- also
    printed to stdout as they happen."""
    t0 = time.monotonic()

    def _log(msg):
        line = f"[{_fmt_elapsed(t0)}] {msg}"
        print(f"[record_run] {line}")
        if log_events is not None:
            log_events.append(line)

    _log("init")
    obs, _ = env.reset(start_pos=start_pos, target_pos=target_pos, target_yaw=target_yaw)
    agent = TwoPhaseAgent(model, gains_by_dist=gains_by_dist, switch_dist=switch_dist,
                           residual_scale=residual_scale, device=device)
    agent.reset(env)
    _log(f"pid guide  dist={env.prev_distance:.2f}m  gains={agent.gain_key}  target={_fmt_vec(target_pos)}")

    positions, dists = [], []
    done = False
    info = {"reason": None}
    total_reward = 0.0
    while not done:
        was_pid = agent.phase == "PID"
        action = agent.get_action(obs, env)
        if was_pid and agent.phase == "RL":
            if residual_scale > 0:
                _log(f"rl (pid+residual) takes over  dist={env.prev_distance:.2f}m  step={agent.switch_step}  "
                     f"re-selected gains={agent.gain_key}")
            else:
                _log(f"rl takes over  dist={env.prev_distance:.2f}m  step={agent.switch_step}")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        positions.append([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        dists.append(env.prev_distance)
        done = terminated or truncated

    outcome = info["reason"] or "timeout"
    _log(f"done  reason={outcome}  final_dist={env.prev_distance:.2f}m  steps={len(positions)}")

    return (np.array(positions, dtype=np.float32), np.array(dists, dtype=np.float32),
            info, total_reward, log_events or [])


def _set_episode_budget(env, dist, log_events=None):
    """Sizes env.max_steps to steps_for_dist(dist) -- the same settling-time
    budget tune_pid.py solves gains against -- clamped to MAX_STEPS_CAP.
    Mutating max_steps per-episode (BaseDroneEnv checks it fresh every step())
    is cheaper than rebuilding the env, and lets short-distance test runs
    stay fast instead of every run paying a worst-case budget."""
    wanted = steps_for_dist(dist)
    budget = min(wanted, MAX_STEPS_CAP)
    if budget < wanted:
        msg = (f"[budget]  dist={dist:.1f}m wants steps_for_dist={wanted} steps -- "
               f"clamped to MAX_STEPS_CAP={MAX_STEPS_CAP} (~{MAX_STEPS_CAP * PHYSICS_DT:.0f}s sim time); "
               f"episode may time out short of hitting the target.")
        print(f"[record_run] {msg}")
        if log_events is not None:
            log_events.append(msg)
    env.max_steps = budget
    return budget


def record_checkpoint_two_phase(checkpoint_path, out_path, distance=30.0,
                                 seed=None, switch_dist=DEFAULT_SWITCH_DIST, residual_scale=0.0):
    """Two-phase counterpart to record_checkpoint: PID flies midcourse (gains
    solved analytically for the exact `distance`), the checkpoint only has to
    fly the last switch_dist meters. residual_scale > 0: composes PID +
    residual_scale*model instead of pure model in the RL phase -- see
    run_episode_two_phase / TwoPhaseAgent docstrings. Renders the same kind
    of mp4, plus returns the timestamped phase-transition log."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    # residual mode: nearest-ladder-snap gains, matching _select_pid_teacher's training-time
    # selection exactly -- analytic gains are a different PID baseline than what a residual
    # correction was actually trained against (see run_episode_two_phase's docstring).
    gains_by_dist = _load_residual_gains(residual_scale)

    env = BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=MAX_STEPS_CAP)
    start_pos, target_pos = sample_target_at_distance(distance, rng)

    log_events = []
    _set_episode_budget(env, distance, log_events=log_events)
    positions, dists, info, total_reward, log_events = run_episode_two_phase(
        model, env, start_pos, target_pos, switch_dist=switch_dist,
        residual_scale=residual_scale, gains_by_dist=gains_by_dist, log_events=log_events,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    render_mp4(positions, dists, target_pos, out_path, info["reason"], total_reward)

    return {
        "reason": info["reason"] or "timeout",
        "steps": int(len(positions)),
        "final_dist": float(dists[-1]) if len(dists) else None,
        "total_reward": float(total_reward),
        "start_pos": start_pos.tolist(),
        "target_pos": target_pos.tolist(),
        "distance": distance,
        "switch_dist": switch_dist,
        "residual_scale": residual_scale,
        "log": log_events,
    }


def evaluate_two_phase(checkpoint_path, n_scenarios=N_EVAL_SCENARIOS, distance=30.0,
                        seed=None, switch_dist=DEFAULT_SWITCH_DIST, residual_scale=0.0):
    """Two-phase counterpart to evaluate_checkpoint: n_scenarios episodes, all
    at the exact same `distance` (random direction each time), no rendering --
    PID flies each one down to switch_dist before the checkpoint takes over.
    residual_scale > 0: composes PID + residual_scale*model in the RL phase
    -- see run_episode_two_phase / TwoPhaseAgent docstrings. Returns
    per-scenario outcomes, totals, and the full timestamped log across every
    scenario."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    gains_by_dist = _load_residual_gains(residual_scale)  # see record_checkpoint_two_phase's comment
    env = BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=MAX_STEPS_CAP)
    budget = _set_episode_budget(env, distance)

    scenarios = []
    totals = {"hit": 0, "oob": 0, "attitude": 0, "timeout": 0}
    all_log_lines = []
    for i in range(n_scenarios):
        start_pos, target_pos = sample_target_at_distance(distance, rng)
        env.max_steps = budget  # _set_episode_budget already logged/clamped once above
        positions, dists, info, total_reward, log_events = run_episode_two_phase(
            model, env, start_pos, target_pos, switch_dist=switch_dist,
            residual_scale=residual_scale, gains_by_dist=gains_by_dist, log_events=[],
        )
        outcome = _classify_outcome(info["reason"])
        for k, v in outcome.items():
            totals[k] += int(v)
        final_dist = float(dists[-1]) if len(dists) else None
        scenarios.append({
            "idx": i, "spawn": start_pos.tolist(), "target": target_pos.tolist(), "dist": distance,
            "reason": info["reason"] or "timeout", "final_dist": final_dist, "steps": int(len(positions)),
            **outcome,
        })
        all_log_lines.append(f"--- scenario {i + 1}/{n_scenarios} ---")
        all_log_lines.extend(log_events)
        print(f"[evaluate_two_phase] {i + 1}/{n_scenarios} dist={distance:.1f}m "
              f"reason={info['reason'] or 'timeout'} final_dist={final_dist}")

    pct = {k: 100.0 * v / n_scenarios for k, v in totals.items()}
    lines = [f"Checkpoint: {checkpoint_path}",
             f"Mode: two-phase (PID midcourse, analytic gains -> "
             f"{'PID+residual' if residual_scale > 0 else 'RL'} terminal at {switch_dist}m"
             f"{f', residual_scale={residual_scale}' if residual_scale > 0 else ''})",
             f"Scenarios: {n_scenarios}  distance={distance}m (random direction each run)", ""]
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
            "pct": pct, "summary_text": summary_text, "log": all_log_lines}


def _load_residual_gains(residual_scale, gains_path=RESIDUAL_GAINS_PATH):
    if residual_scale <= 0:
        return None
    with open(gains_path) as f:
        return json.load(f)


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


def record_checkpoint(checkpoint_path, out_path, distance_low=3.0, distance_high=10.0, seed=None,
                      residual_scale=0.0):
    """Runs one deterministic episode against checkpoint_path and writes an
    mp4 to out_path. Returns a small dict of episode stats (used by
    serve_run.py's API response)."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    gains_by_dist = _load_residual_gains(residual_scale)

    env = BaseDroneEnv(reward_func, max_steps=6000)
    start_pos, target_pos = sample_target(distance_low, distance_high, rng)

    positions, dists, info, total_reward = run_episode(model, env, start_pos, target_pos,
                                                        residual_scale=residual_scale, gains_by_dist=gains_by_dist)
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


def evaluate_checkpoint(checkpoint_path, n_scenarios=N_EVAL_SCENARIOS, distance_low=3.0, distance_high=10.0,
                        seed=None, residual_scale=0.0):
    """Runs checkpoint_path deterministically against n_scenarios random (start, target)
    scenarios, no rendering -- returns per-scenario outcomes, totals, and a text summary."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    gains_by_dist = _load_residual_gains(residual_scale)
    env = BaseDroneEnv(reward_func, max_steps=6000)

    scenarios = []
    totals = {"hit": 0, "oob": 0, "attitude": 0, "timeout": 0}
    for i in range(n_scenarios):
        start_pos, target_pos = sample_target(distance_low, distance_high, rng)
        dist = float(np.linalg.norm(target_pos - start_pos))
        positions, dists, info, total_reward = run_episode(model, env, start_pos, target_pos,
                                                            residual_scale=residual_scale, gains_by_dist=gains_by_dist)
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
                      seed=None, top_n=5, residual_scale=0.0):
    """Batch-evaluates each of checkpoint_paths against n_scenarios scenarios (no per-scenario
    dump, no video) and returns only the top_n ranked by grade -- a short leaderboard of
    potential suitable weights."""
    oob_radius = max(20.0, distance_high * 3.0)
    max_steps = 6000
    gains_by_dist = _load_residual_gains(residual_scale)
    results = []
    for path in checkpoint_paths:
        rng = np.random.default_rng(seed)
        model = load_model_for_inference(path)
        env = BaseDroneEnv(reward_func, max_steps=max_steps)

        totals = {"hit": 0, "oob": 0, "attitude": 0, "timeout": 0}
        hit_times, final_dists = [], []
        for _ in range(n_scenarios):
            start_pos, target_pos = sample_target(distance_low, distance_high, rng)
            positions, dists, info, _ = run_episode(model, env, start_pos, target_pos,
                                                     residual_scale=residual_scale, gains_by_dist=gains_by_dist)
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
    parser.add_argument("--residual-scale", type=float, default=0.0,
                         help="> 0 if checkpoint(s) were trained with --residual-scale (PID-residual mode) -- "
                              "must match the value training used, or the composed action is wrong. Combines "
                              "with --two-phase: the RL phase then composes PID + residual_scale*model instead "
                              "of pure model output (REQUIRED for a residual-trained checkpoint -- its raw "
                              "output alone means 'hover, don't steer').")
    parser.add_argument("--two-phase", action="store_true",
                         help="PID flies midcourse (gains solved analytically for the exact --distance), "
                              "checkpoint flies the last --switch-dist meters (see app.control.two_phase) -- "
                              "alone or, with --residual-scale > 0, as a PID-residual correction. "
                              "Uses --distance instead of --distance-low/--distance-high.")
    parser.add_argument("--switch-dist", type=float, default=DEFAULT_SWITCH_DIST,
                         help="--two-phase only: distance (m) at which control hands off from PID to the checkpoint.")
    parser.add_argument("--distance", type=float, default=30.0,
                         help="--two-phase only: exact target distance (m) -- direction is random, distance is not.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.rank_dir:
        checkpoints = sorted(glob.glob(os.path.join(args.rank_dir, "*.pt")))
        result = rank_checkpoints(checkpoints, n_scenarios=args.n_scenarios, distance_low=args.distance_low,
                                   distance_high=args.distance_high, seed=args.seed, top_n=args.top_n,
                                   residual_scale=args.residual_scale)
        print(result["summary_text"])
    else:
        checkpoint = args.checkpoint or find_latest_checkpoint()
        print(f"[record_run] checkpoint: {checkpoint}")
        if args.two_phase:
            if args.evaluate:
                result = evaluate_two_phase(checkpoint, n_scenarios=args.n_scenarios, distance=args.distance,
                                             seed=args.seed, switch_dist=args.switch_dist,
                                             residual_scale=args.residual_scale)
                print(result["summary_text"])
            else:
                stats = record_checkpoint_two_phase(checkpoint, args.out, distance=args.distance,
                                                     seed=args.seed, switch_dist=args.switch_dist,
                                                     residual_scale=args.residual_scale)
                print(f"[record_run] done: {stats}")
        elif args.evaluate:
            result = evaluate_checkpoint(checkpoint, n_scenarios=args.n_scenarios, distance_low=args.distance_low,
                                          distance_high=args.distance_high, seed=args.seed,
                                          residual_scale=args.residual_scale)
            print(result["summary_text"])
        else:
            stats = record_checkpoint(checkpoint, args.out, distance_low=args.distance_low,
                                       distance_high=args.distance_high, seed=args.seed,
                                       residual_scale=args.residual_scale)
            print(f"[record_run] done: {stats}")
