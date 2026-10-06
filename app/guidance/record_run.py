"""Replay any ActorCritic checkpoint in the numpy env and render the flight to mp4 (single or two-phase).

    python -m app.guidance.record_run [checkpoint.pt] [--episodes N] [--out runs/demo.mp4] [--two-phase ...]

See docs.md "record_run" for all modes and flags.
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
from scipy.spatial.transform import Rotation

from app.control.step_budget import steps_for_dist
from app.control.two_phase import DEFAULT_SWITCH_DIST, TwoPhaseAgent
from app.environmental.base_drone_env import BaseDroneEnv
from app.guidance.train import ActorCritic, load_bc_checkpoint, device
from app.guidance.utils import compute_grade
from app.reward_functions.rewards import reward_func, HIT_THRESHOLD, OOB_RADIUS

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
    """Build a dropout=0 ActorCritic with shapes inferred from the checkpoint, and load it."""
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
    """Target at exactly `dist` in a random direction (the two-phase UI's distance field)."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    start_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    target_pos = (start_pos + direction * dist).astype(np.float32)
    if target_pos[2] < 0.5:  # keep target above ground, preserving the exact distance
        direction[2] = abs(direction[2])
        target_pos = (start_pos + direction * dist).astype(np.float32)
        target_pos[2] = max(target_pos[2], 0.5)
    return start_pos, target_pos


def run_episode(model, env, start_pos, target_pos, target_yaw=0.0):
    """One deterministic rollout of the model alone (no PID)."""
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


# wall-clock safety cap on the step budget (= steps_for_dist(MAX_TEST_DISTANCE), ~1.25M steps)
MAX_TEST_DISTANCE = 5000.0
MAX_STEPS_CAP = steps_for_dist(MAX_TEST_DISTANCE)


def _fmt_elapsed(t0):
    """Wall-clock time since t0 as [HH:MM:SS.mmm] (real time, not sim time)."""
    elapsed = time.monotonic() - t0
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _episode_kin(env):
    """(speed, closing_speed, lateral_speed) in m/s: closing = velocity component toward the target,
    lateral = the rest (cross-track). Cheap numpy only -- called every step."""
    s = env.drone_state
    vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z], dtype=np.float64)
    to_t = np.asarray(env.target_pos, dtype=np.float64) - np.array(
        [s.position.x, s.position.y, s.position.z], dtype=np.float64)
    d = max(float(np.linalg.norm(to_t)), 1e-9)
    speed = float(np.linalg.norm(vel))
    closing = float(vel @ to_t) / d
    return speed, closing, float(np.sqrt(max(speed * speed - closing * closing, 0.0)))


def _tilt_deg(env):
    q = env.drone_state.orientation
    roll, pitch, _ = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
    return float(np.degrees(roll)), float(np.degrees(pitch))


def run_episode_two_phase(model, env, start_pos, target_pos, target_yaw=0.0,
                           switch_dist=DEFAULT_SWITCH_DIST,
                           log_events=None, trace_every=0, cruise_speed=18.0, handoff_speed=None, track=False, recentre_z=None):
    """One two-phase episode: PID/carrot to switch_dist, then the model. trace_every > 0 logs per-step lines; info["diag"] has the miss analysis."""
    t0 = time.monotonic()

    def _log(msg):
        line = f"[{_fmt_elapsed(t0)}] {msg}"
        print(f"[record_run] {line}")
        if log_events is not None:
            log_events.append(line)

    _log("init")
    obs, _ = env.reset(start_pos=start_pos, target_pos=target_pos, target_yaw=target_yaw)
    agent = TwoPhaseAgent(model, switch_dist=switch_dist, device=device, cruise_speed=cruise_speed,
                           handoff_speed=handoff_speed, recentre_z=recentre_z)
    agent.reset(env)
    _log(f"pid guide  dist={env.prev_distance:.2f}m  gains={agent.gain_key}  target={_fmt_vec(target_pos)}  "
         f"hit_radius={HIT_THRESHOLD}m  switch_dist={switch_dist}m  max_steps={env.max_steps} "
         f"(~{env.max_steps * env.dt:.0f}s sim)")

    positions, dists, speeds = [], [], []
    quats, phases = [], []          # only filled when track=True (3D viewer playback)
    done = False
    info = {"reason": None}
    total_reward = 0.0
    d0 = float(env.prev_distance)
    diag = {"min_dist": d0, "min_step": 0, "min_speed": 0.0, "handoff_step": None, "handoff_dist": None,
            "handoff_speed": None, "handoff_closing": None, "handoff_lateral": None, "peak_speed": 0.0,
            "steps_within_1m": 0, "stall_steps": 0, "approaches": 0, "rl_steps": 0, "hit_radius": HIT_THRESHOLD}
    inside2 = False
    step_i = 0
    while not done:
        was_pid = agent.phase == "PID"
        action = agent.get_action(obs, env)
        if was_pid and agent.phase == "RL":
            speed, closing, lateral = _episode_kin(env)
            diag.update(handoff_step=agent.switch_step, handoff_dist=float(env.prev_distance),
                        handoff_speed=speed, handoff_closing=closing, handoff_lateral=lateral)
            extra = (f"speed={speed:.2f}m/s (closing={closing:+.2f}, lateral={lateral:.2f})  "
                     f"step={agent.switch_step}")
            if agent._pos_offset is not None:
                extra += f"  recentred: policy position = pos - {np.round(agent._pos_offset, 1).tolist()}"
            _log(f"rl takes over  dist={env.prev_distance:.2f}m  {extra}  "
                 f"model_action={np.round(action, 3).tolist()}")
        obs, reward, terminated, truncated, info = env.step(action)
        step_i += 1
        total_reward += reward
        positions.append([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        d = float(env.prev_distance)
        speed, closing, lateral = _episode_kin(env)
        dists.append(d)
        speeds.append(speed)
        if track:
            q = env.drone_state.orientation
            quats.append((q.x, q.y, q.z, q.w))
            phases.append(1 if agent.phase == "RL" else 0)

        if d < diag["min_dist"]:
            diag.update(min_dist=d, min_step=step_i, min_speed=speed)
        diag["peak_speed"] = max(diag["peak_speed"], speed)
        if d < 1.0:
            diag["steps_within_1m"] += 1
            if d > HIT_THRESHOLD and speed < 0.3:
                diag["stall_steps"] += 1
        if d < 2.0 and not inside2:
            inside2 = True
            diag["approaches"] += 1
        elif d > 3.0:
            inside2 = False
        if agent.phase == "RL":
            diag["rl_steps"] += 1

        if trace_every and step_i % trace_every == 0:
            roll, pitch = _tilt_deg(env)
            _log(f"t={step_i * env.dt:6.1f}s step={step_i:6d} [{agent.phase:3s}] dist={d:7.2f}m  "
                 f"speed={speed:5.2f} (closing={closing:+6.2f} lateral={lateral:5.2f})  "
                 f"tilt r/p={roll:+5.1f}/{pitch:+5.1f}deg  action={np.round(action, 2).tolist()}")
        done = terminated or truncated

    outcome = info["reason"] or "timeout"
    n = len(positions)
    final_speed, final_closing, _ = _episode_kin(env)
    diag.update(final_speed=final_speed, final_closing=final_closing, steps=n, final_dist=float(env.prev_distance),
                outcome=outcome)
    stride = max(1, n // 300)
    diag["trace"] = [[i + 1, float(dists[i]), float(speeds[i])] for i in range(0, n, stride)]
    if (n - 1) % stride:
        diag["trace"].append([n, float(dists[-1]), float(speeds[-1])])
    info = dict(info)
    info["diag"] = diag
    if track:
        info["track"] = _build_track(positions, quats, phases, dists, speeds, env.dt)

    _log(f"done  reason={outcome}  final_dist={env.prev_distance:.2f}m  steps={n}  final_speed={final_speed:.2f}m/s")
    if "oob" in outcome:
        # same test as rewards.reward_func: world-origin distance > max(OOB_RADIUS, 3 * start_dist), or z < 0
        p = env.drone_state.position
        radius = max(OOB_RADIUS, env.start_dist * 3.0)
        _log(f"OOB  pos=({p.x:.1f}, {p.y:.1f}, {p.z:.1f})  |pos|={np.linalg.norm([p.x, p.y, p.z]):.1f}m  "
             f"limit={radius:.1f}m  z<0={p.z < 0.0}  start_dist={env.start_dist:.1f}m")
    _log(f"closest approach {diag['min_dist']:.2f}m at step {diag['min_step']} (speed {diag['min_speed']:.2f}m/s)  "
         f"| approaches<2m={diag['approaches']}  steps<1m={diag['steps_within_1m']}  stalled(0.25-1m,<0.3m/s)="
         f"{diag['stall_steps']}  peak_speed={diag['peak_speed']:.1f}m/s  rl_steps={diag['rl_steps']}")
    if outcome != "Hit":
        gap = diag["min_dist"] - HIT_THRESHOLD
        if diag["min_dist"] < 2.0:
            _log(f"NEAR MISS: closest {diag['min_dist']:.2f}m vs hit radius {HIT_THRESHOLD}m (missed by {gap:.2f}m)")
        if n and diag["stall_steps"] > 0.2 * n:
            _log(f"STALL: {100.0 * diag['stall_steps'] / n:.0f}% of the episode hovering 0.25-1m from the target "
                 f"at <0.3m/s -- the policy is parking next to the target instead of closing the last bit")
        elif diag["approaches"] >= 3:
            _log(f"LOOPING: {diag['approaches']} separate approaches within 2m -- swings past the target and "
                 f"has to turn around")
    return (np.array(positions, dtype=np.float32), np.array(dists, dtype=np.float32),
            info, total_reward, log_events or [])


MAX_TRACK_FRAMES = 3000


def _build_track(positions, quats, phases, dists, speeds, dt):
    """Downsampled per-frame pose for the browser's three.js player (z-up, same frame as the sim).
    The first and last steps are always kept."""
    n = len(positions)
    stride = max(1, n // MAX_TRACK_FRAMES)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return {
        "dt": float(dt) * stride,
        "stride": stride,
        "steps": idx,
        "pos": [[round(float(v), 3) for v in positions[i]] for i in idx],
        "quat": [[round(float(v), 4) for v in quats[i]] for i in idx],
        "phase": [phases[i] for i in idx],
        "dist": [round(float(dists[i]), 3) for i in idx],
        "speed": [round(float(speeds[i]), 3) for i in idx],
    }


def render_dist_plot(diag, out_path, switch_dist, reason):
    """distance-to-target and speed vs time, with the hit radius, the PID->RL hand-off and the closest
    approach marked -- the same information the video title shows, but over the whole episode."""
    tr = np.array(diag["trace"], dtype=np.float64)
    if len(tr) == 0:
        return None
    t = tr[:, 0] / 240.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, tr[:, 1], color="tab:blue", label="distance to target")
    ax1.axhline(HIT_THRESHOLD, color="tab:red", ls="--", label=f"hit radius {HIT_THRESHOLD} m")
    ax1.axhline(switch_dist, color="gray", ls=":", label=f"switch_dist {switch_dist} m")
    ax1.scatter([diag["min_step"] / 240.0], [diag["min_dist"]], color="k", zorder=5,
                label=f"closest {diag['min_dist']:.2f} m")
    if diag.get("handoff_step") is not None:
        for ax in (ax1, ax2):
            ax.axvline(diag["handoff_step"] / 240.0, color="green", ls="-.", alpha=0.7)
        ax1.annotate("RL takes over", (diag["handoff_step"] / 240.0, ax1.get_ylim()[1] * 0.9), color="green",
                     fontsize=8)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_ylim(bottom=0.0)
    ax1.set_ylabel("distance (m)")
    ax1.set_title(f"{reason or 'timeout'}  final={diag['final_dist']:.2f} m  closest={diag['min_dist']:.2f} m")
    ax1.legend(fontsize=8)
    ax2.plot(t, tr[:, 2], color="tab:orange")
    ax2.set_ylabel("speed (m/s)")
    ax2.set_xlabel("sim time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


def _set_episode_budget(env, dist, log_events=None):
    """Set env.max_steps to steps_for_dist(dist), clamped to MAX_STEPS_CAP."""
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
                                 seed=None, switch_dist=DEFAULT_SWITCH_DIST, cruise_speed=18.0, handoff_speed=None, recentre_z=None):
    """Two-phase record_checkpoint: renders the mp4 and returns the phase-transition log."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=MAX_STEPS_CAP,
                       position_mode=_position_mode_for(checkpoint_path))
    start_pos, target_pos = sample_target_at_distance(distance, rng)

    log_events = []
    _set_episode_budget(env, distance, log_events=log_events)
    note = _trained_range_note(checkpoint_path, switch_dist)
    if note:
        print(f"[record_run] {note}")
        log_events.append(note)
    positions, dists, info, total_reward, log_events = run_episode_two_phase(
        model, env, start_pos, target_pos, switch_dist=switch_dist,
        log_events=log_events, trace_every=240, cruise_speed=cruise_speed, handoff_speed=handoff_speed, recentre_z=recentre_z,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    render_mp4(positions, dists, target_pos, out_path, info["reason"], total_reward)
    plot_path = os.path.splitext(out_path)[0] + "_dist.png"
    try:
        render_dist_plot(info["diag"], plot_path, switch_dist, info["reason"])
    except Exception as e:  # the plot is a diagnostic extra, never fail the run over it
        print(f"[record_run] dist plot failed: {e}")
        plot_path = None

    return {
        "reason": info["reason"] or "timeout",
        "steps": int(len(positions)),
        "final_dist": float(dists[-1]) if len(dists) else None,
        "total_reward": float(total_reward),
        "start_pos": start_pos.tolist(),
        "target_pos": target_pos.tolist(),
        "distance": distance,
        "switch_dist": switch_dist,
        "recentre_z": recentre_z,
        "position_mode": env.position_mode,
        "log": log_events,
        "diag": {k: v for k, v in info["diag"].items() if k != "trace"},
        "plot_path": plot_path,
    }


def trajectory_two_phase(checkpoint_path, distance=30.0, seed=None, switch_dist=DEFAULT_SWITCH_DIST,
                          cruise_speed=18.0, handoff_speed=None, recentre_z=None):
    """Like record_checkpoint_two_phase but skips the mp4/png: returns the episode stats plus a per-frame
    "track" (positions, orientation, phase, distance, speed) for the browser's three.js player."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=MAX_STEPS_CAP,
                       position_mode=_position_mode_for(checkpoint_path))
    start_pos, target_pos = sample_target_at_distance(distance, rng)
    log_events = []
    _set_episode_budget(env, distance, log_events=log_events)
    note = _trained_range_note(checkpoint_path, switch_dist)
    if note:
        log_events.append(note)
    positions, dists, info, total_reward, log_events = run_episode_two_phase(
        model, env, start_pos, target_pos, switch_dist=switch_dist, log_events=log_events, trace_every=240,
        cruise_speed=cruise_speed, handoff_speed=handoff_speed, track=True, recentre_z=recentre_z)
    return {
        "reason": info["reason"] or "timeout",
        "steps": int(len(positions)),
        "final_dist": float(dists[-1]) if len(dists) else None,
        "total_reward": float(total_reward),
        "start_pos": start_pos.tolist(),
        "target_pos": target_pos.tolist(),
        "distance": distance,
        "switch_dist": switch_dist,
        "recentre_z": recentre_z,
        "position_mode": env.position_mode,
        "hit_radius": float(HIT_THRESHOLD),
        "log": log_events,
        "diag": {k: v for k, v in info["diag"].items() if k != "trace"},
        "track": info["track"],
    }


def evaluate_two_phase(checkpoint_path, n_scenarios=N_EVAL_SCENARIOS, distance=30.0,
                        seed=None, switch_dist=DEFAULT_SWITCH_DIST, cruise_speed=18.0, handoff_speed=None, recentre_z=None):
    """Two-phase evaluate_checkpoint: n_scenarios at the same distance, random directions, no rendering."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=MAX_STEPS_CAP,
                       position_mode=_position_mode_for(checkpoint_path))
    budget = _set_episode_budget(env, distance)

    scenarios = []
    totals = {"hit": 0, "oob": 0, "attitude": 0, "timeout": 0}
    all_log_lines = []
    for i in range(n_scenarios):
        start_pos, target_pos = sample_target_at_distance(distance, rng)
        env.max_steps = budget  # _set_episode_budget already logged/clamped once above
        positions, dists, info, total_reward, log_events = run_episode_two_phase(
            model, env, start_pos, target_pos, switch_dist=switch_dist,
            log_events=[], cruise_speed=cruise_speed, handoff_speed=handoff_speed, recentre_z=recentre_z,
        )
        outcome = _classify_outcome(info["reason"])
        for k, v in outcome.items():
            totals[k] += int(v)
        final_dist = float(dists[-1]) if len(dists) else None
        dg = info["diag"]
        scenarios.append({
            "idx": i, "spawn": start_pos.tolist(), "target": target_pos.tolist(), "dist": distance,
            "reason": info["reason"] or "timeout", "final_dist": final_dist, "steps": int(len(positions)),
            "min_dist": dg["min_dist"], "min_step": dg["min_step"], "handoff_speed": dg["handoff_speed"],
            "handoff_dist": dg["handoff_dist"], "approaches": dg["approaches"], "stall_steps": dg["stall_steps"],
            "final_speed": dg["final_speed"],
            **outcome,
        })
        all_log_lines.append(f"--- scenario {i + 1}/{n_scenarios} ---")
        all_log_lines.extend(log_events)
        print(f"[evaluate_two_phase] {i + 1}/{n_scenarios} dist={distance:.1f}m "
              f"reason={info['reason'] or 'timeout'} final_dist={final_dist} min_dist={dg['min_dist']:.2f} "
              f"handoff_speed={dg['handoff_speed']}")

    pct = {k: 100.0 * v / n_scenarios for k, v in totals.items()}
    lines = [f"Checkpoint: {checkpoint_path}",
             f"Mode: two-phase (PID midcourse, analytic gains -> "
             f"RL terminal at {switch_dist}m)",
             f"Scenarios: {n_scenarios}  distance={distance}m (random direction each run)", ""]
    for s in scenarios:
        hs = f"{s['handoff_speed']:.1f}" if s["handoff_speed"] is not None else "n/a"
        lines.append(f"[{s['idx']:3d}] dist={s['dist']:5.2f}m spawn={_fmt_vec(s['spawn'])} "
                      f"target={_fmt_vec(s['target'])} -> {s['reason']:<16} "
                      f"final_dist={s['final_dist']:.2f}m closest={s['min_dist']:.2f}m@{s['min_step']} "
                      f"handoff_speed={hs}m/s approaches={s['approaches']} stalled={s['stall_steps']} "
                      f"steps={s['steps']}")
    lines.append("")
    lines.append("Totals:")
    for k in ("hit", "oob", "attitude", "timeout"):
        lines.append(f"  {k:<10} {totals[k]:3d}/{n_scenarios}  ({pct[k]:.1f}%)")
    misses = [s for s in scenarios if not s["hit"]]
    near = [s for s in misses if s["min_dist"] < 1.0]
    lines.append("")
    lines.append(f"Near-misses (not a hit, but closest approach < 1 m): {len(near)}/{len(misses)} of the misses")
    if misses:
        lines.append(f"  mean closest approach of misses: {np.mean([s['min_dist'] for s in misses]):.2f} m "
                     f"(hit radius {HIT_THRESHOLD} m); mean final distance: "
                     f"{np.mean([s['final_dist'] for s in misses]):.2f} m")
    hs_all = [s["handoff_speed"] for s in scenarios if s["handoff_speed"] is not None]
    if hs_all:
        lines.append(f"  hand-off speed: mean {np.mean(hs_all):.1f} m/s, min {np.min(hs_all):.1f}, "
                     f"max {np.max(hs_all):.1f}")
    note = _trained_range_note(checkpoint_path, switch_dist)
    if note:
        lines.append(f"  {note}")
    summary_text = "\n".join(lines)

    return {"checkpoint": checkpoint_path, "scenarios": scenarios, "totals": totals,
            "pct": pct, "summary_text": summary_text, "log": all_log_lines}


def _position_mode_for(checkpoint_path):
    """obs[0:3] mode this checkpoint was trained with ('full' | 'height'), from the run_config.json next to it
    (rtl_train_isaac.py --obs-position-mode); checkpoints without one are 'full' (the original layout)."""
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")
        with open(cfg_path) as f:
            return json.load(f).get("obs_position_mode", "full")
    except (OSError, ValueError):
        return "full"


def _trained_range_note(checkpoint_path, switch_dist):
    """If the checkpoint sits next to a run_config.json (rtl_train_isaac.py writes one) that records the
    distance range it was trained on, warn when the RL phase is handed control OUTSIDE that range."""
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path) as f:
            cfg = json.load(f)
        hi, lo = cfg.get("distance_high"), cfg.get("distance_low")
        if hi is None:
            return None
        rng = f"{lo:g}-{hi:g} m"
        if switch_dist > hi:
            return (f"WARNING: switch_dist={switch_dist:g}m is OUTSIDE this checkpoint's trained distance range "
                    f"({rng}); the policy takes over from states it never saw -- set switch_dist <= {hi:g}.")
        return f"checkpoint trained on {rng}; switch_dist={switch_dist:g}m is inside it."
    except Exception:
        return None



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
    """One deterministic episode -> mp4 at out_path; returns episode stats (used by serve_run)."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)

    env = BaseDroneEnv(reward_func, max_steps=6000, position_mode=_position_mode_for(checkpoint_path))
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


def evaluate_checkpoint(checkpoint_path, n_scenarios=N_EVAL_SCENARIOS, distance_low=3.0, distance_high=10.0,
                        seed=None):
    """Runs checkpoint_path deterministically against n_scenarios random (start, target)
    scenarios, no rendering -- returns per-scenario outcomes, totals, and a text summary."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = BaseDroneEnv(reward_func, max_steps=6000, position_mode=_position_mode_for(checkpoint_path))

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
        hs = f"{s['handoff_speed']:.1f}" if s["handoff_speed"] is not None else "n/a"
        lines.append(f"[{s['idx']:3d}] dist={s['dist']:5.2f}m spawn={_fmt_vec(s['spawn'])} "
                      f"target={_fmt_vec(s['target'])} -> {s['reason']:<16} "
                      f"final_dist={s['final_dist']:.2f}m closest={s['min_dist']:.2f}m@{s['min_step']} "
                      f"handoff_speed={hs}m/s approaches={s['approaches']} stalled={s['stall_steps']} "
                      f"steps={s['steps']}")
    lines.append("")
    lines.append("Totals:")
    for k in ("hit", "oob", "attitude", "timeout"):
        lines.append(f"  {k:<10} {totals[k]:3d}/{n_scenarios}  ({pct[k]:.1f}%)")
    misses = [s for s in scenarios if not s["hit"]]
    near = [s for s in misses if s["min_dist"] < 1.0]
    lines.append("")
    lines.append(f"Near-misses (not a hit, but closest approach < 1 m): {len(near)}/{len(misses)} of the misses")
    if misses:
        lines.append(f"  mean closest approach of misses: {np.mean([s['min_dist'] for s in misses]):.2f} m "
                     f"(hit radius {HIT_THRESHOLD} m); mean final distance: "
                     f"{np.mean([s['final_dist'] for s in misses]):.2f} m")
    hs_all = [s["handoff_speed"] for s in scenarios if s["handoff_speed"] is not None]
    if hs_all:
        lines.append(f"  hand-off speed: mean {np.mean(hs_all):.1f} m/s, min {np.min(hs_all):.1f}, "
                     f"max {np.max(hs_all):.1f}")
    note = _trained_range_note(checkpoint_path, switch_dist)
    if note:
        lines.append(f"  {note}")
    summary_text = "\n".join(lines)

    return {"checkpoint": checkpoint_path, "scenarios": scenarios, "totals": totals,
            "pct": pct, "summary_text": summary_text}


def rank_checkpoints(checkpoint_paths, n_scenarios=N_EVAL_SCENARIOS, distance_low=3.0, distance_high=10.0,
                      seed=None, top_n=5):
    """Evaluate checkpoints on n_scenarios each and return the top_n by grade."""
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
    parser.add_argument("--two-phase", action="store_true",
                         help="PID flies midcourse (gains solved analytically for the exact --distance), "
                              "checkpoint flies the last --switch-dist meters (see app.control.two_phase) -- "
                              "Uses --distance instead of --distance-low/--distance-high.")
    parser.add_argument("--switch-dist", type=float, default=DEFAULT_SWITCH_DIST,
                         help="--two-phase only: distance (m) at which control hands off from PID to the checkpoint.")
    parser.add_argument("--cruise-speed", type=float, default=18.0,
                         help="--two-phase only: the PID leg flies the training spawn-leg profile (0 -> this -> handoff speed). "
                              "Use the run's --max-speed. 0 = old position-hold PID (arrives at ~0.5 m/s, nothing like training).")
    parser.add_argument("--handoff-speed", type=float, default=None,
                         help="--two-phase only: speed (m/s) at which the checkpoint takes over; default cruise-speed / 2.")
    parser.add_argument("--recentre-z", type=float, default=None,
                         help="--two-phase only: feed a 'full'-mode policy its position relative to the handoff point "
                              "(what rtl_train_isaac's re-centring trained it on); z is capped at this value. Off by default.")
    parser.add_argument("--distance", type=float, default=30.0,
                         help="--two-phase only: exact target distance (m) -- direction is random, distance is not.")
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
        if args.two_phase:
            if args.evaluate:
                result = evaluate_two_phase(checkpoint, n_scenarios=args.n_scenarios, distance=args.distance,
                                             seed=args.seed, switch_dist=args.switch_dist,
                                             cruise_speed=args.cruise_speed or None, handoff_speed=args.handoff_speed,
                                             recentre_z=args.recentre_z)
                print(result["summary_text"])
            else:
                stats = record_checkpoint_two_phase(checkpoint, args.out, distance=args.distance,
                                                     seed=args.seed, switch_dist=args.switch_dist,
                                             cruise_speed=args.cruise_speed or None, handoff_speed=args.handoff_speed,
                                             recentre_z=args.recentre_z)
                print(f"[record_run] done: {stats}")
        elif args.evaluate:
            result = evaluate_checkpoint(checkpoint, n_scenarios=args.n_scenarios, distance_low=args.distance_low,
                                          distance_high=args.distance_high, seed=args.seed)
            print(result["summary_text"])
        else:
            stats = record_checkpoint(checkpoint, args.out, distance_low=args.distance_low,
                                       distance_high=args.distance_high, seed=args.seed)
            print(f"[record_run] done: {stats}")
