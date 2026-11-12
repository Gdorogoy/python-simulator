"""Replay any ActorCritic checkpoint in the numpy env (two-phase mp4 / evaluation) and evaluate acro checkpoints (--acro).

    python -m app.guidance.record_run [checkpoint.pt] [--episodes N] [--out runs/demo.mp4] [--two-phase ...]

See docs.md "record_run" for all modes and flags.
"""
import argparse
import glob
import json
import math
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
from app.control.two_phase import DEFAULT_SWITCH_DIST, CarrotController, TwoPhaseAgent, load_gains_by_dist, snap_handoff_state
from app.dynamics.drone import Vector3D
from app.environmental.base_drone_env import BaseDroneEnv
from app.guidance.dyn_track import DynRecorder
from app.environmental.base_drone_env import VEL_SCALE
from app.environmental.threats import DEFAULT_INTERCEPT_RADIUS, RUNNER_FEATURES, Runner, Threats, ThreatDroneEnv, layout_from_obs_dim
from app.guidance.train import ActorCritic, load_bc_checkpoint, device
from app.reward_functions.rewards import reward_func, HIT_THRESHOLD, OOB_RADIUS

PHYSICS_DT = 1.0 / 240.0
ACRO_MIN_RATE_BOUND = 3.0   # an acro checkpoint's roll bound is a body rate (6 rad/s); an angle one's a torque (<= 1.5 N m)

CHECKPOINT_SEARCH_DIRS = ("runs", "app/control", "stable_models")
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
    model.obs_dim = obs_dim      # 23 + 7 * (threat slots); the viewer reads the slot count from it
    return model


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


def _intercept_radius_for(checkpoint_path):
    """The catch radius the run was trained with (run_config.json -> reward_qv2.intercept_radius), else the default."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            return float(json.load(f)["reward_qv2"]["intercept_radius"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_INTERCEPT_RADIUS


def _attitude_limits_for(checkpoint_path):
    """(roll_deg, pitch_deg) the run was trained with (run_config.json -> reward_qv2.roll_limit_deg / pitch_limit_deg), else None for
    the numpy reward's own 65 / 80 deg. --reward rc runs often switch them off (360 deg), and ending a flight at 65 deg that the
    policy was never penalised for would show a crash the training never had."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            cfg = json.load(f)["reward_qv2"]
        return float(cfg["roll_limit_deg"]), float(cfg["pitch_limit_deg"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _zone_for(checkpoint_path):
    """(hit_radius, zone_time_s) the run was trained with (run_config.json -> reward_qv2, which holds the rc reward's settings
    for --reward rc runs), else the rc defaults. Only used for checkpoints that take the zone-timer inputs."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            cfg = json.load(f)["reward_qv2"]
        return float(cfg["hit_radius"]), float(cfg["zone_time"])
    except (OSError, ValueError, KeyError, TypeError):
        return 1.0, 2.0


def _v2_layout_for(checkpoint_path, obs_dim):
    """(threat slots, has zone timer, ally slots) when the run was trained with --obs-layout v2 (run_config.json: obs_layout, num_threats,
    num_allies), else None. v2 sizes can equal v1 sizes, so the layout is read from the run, not guessed from the input count."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    if cfg.get("obs_layout") != "v2":
        return None
    slots, allies = int(cfg.get("num_threats") or 0), int(cfg.get("num_allies") or 0)
    extra = obs_dim - (23 + 4 * slots + 5 * allies + 3)
    if extra not in (0, 2):
        raise ValueError(f"{checkpoint_path}: observation size {obs_dim} does not fit obs_layout v2 with {slots} threat slot(s), "
                         f"{allies} ally slot(s) (23 + [2 timer] + 4*{slots} + 5*{allies} + 3)")
    return slots, extra == 2, allies


def _make_two_phase_env(checkpoint_path, model, threats=None, rng=None, hit_radius=None):
    """The numpy env for an angle checkpoint. `threats` = {"active", "speed", "speed_min", "spread", "spawn_min", "spawn", "buffer",
    "radius", "release_frac_range", "max_tilt_deg"} switches on that many chasing drones (None = none; missing keys take the
    trainer's defaults). The chasers are drones with the learner's own physics, flown by the limit-LOS controller.
    A checkpoint trained with threat slots always gets all its slots filled -- unused ones are zeros, exactly as in training
    with fewer active threats than slots -- so it loads whether or not any chaser is on. A checkpoint with the two extra
    zone-timer inputs (23 + 7*K + 2) gets them too.
    `hit_radius` (m) is the zone that ends the flight as a Hit; None = rewards.HIT_THRESHOLD. The zone-timer inputs keep the radius
    the checkpoint was trained with, so the policy sees what it was trained on."""
    threats = threats or {}
    v2 = _v2_layout_for(checkpoint_path, model.obs_dim)
    slots, has_timer = (v2[0], v2[1]) if v2 else layout_from_obs_dim(model.obs_dim)
    env = ThreatDroneEnv(reward_func, pid_gains_path=None, max_steps=MAX_STEPS_CAP, position_mode=_position_mode_for(checkpoint_path))
    if v2:
        env.set_layout_v2(v2[2])
    if has_timer:
        env.set_timer(*_zone_for(checkpoint_path))
    env.set_threats(Threats(
        slots, int(threats.get("active", 0)), max_speed=threats.get("speed", 50.0), min_speed=threats.get("speed_min"),
        spread=threats.get("spread", 30.0), spawn_min=threats.get("spawn_min", 15.0), spawn=threats.get("spawn", "uniform"),
        buffer=threats.get("buffer", 0.5), radius=threats.get("radius"),
        release_frac_range=threats.get("release_frac_range", (0.01, 0.67)),      # the trainer's --threat-release-after 0.33 0.99
        max_tilt=math.radians(threats.get("max_tilt_deg", 90.0)),
        geometry_weights=threats.get("geometry_weights", (1.0, 1.0)), reaction_s=threats.get("reaction_s", 1.5),
        tail_gap=threats.get("tail_gap", (15.0, 40.0)),
        jitter=threats.get("jitter", 2.0),
        intercept_radius=_intercept_radius_for(checkpoint_path), rng=rng))
    env.hit_radius = float(hit_radius) if hit_radius else float(HIT_THRESHOLD)      # read by rewards.reward_func
    env.los_cfg = _los_cfg_for(checkpoint_path)
    limits = _attitude_limits_for(checkpoint_path)
    if limits is not None:
        env.attitude_roll_deg, env.attitude_pitch_deg = limits       # read by rewards.reward_func
    return env


def _los_cfg_for(checkpoint_path):
    """The line-of-sight settings the run was trained with (run_config.json -> reward_qv2), else the rc reward's current
    defaults, so the viewer shows what the term pays even for a checkpoint trained before it existed."""
    from app.reward_functions.angle_reward_rc import RewardRCConfig
    cfg = RewardRCConfig()
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            saved = json.load(f)["reward_qv2"]
    except (OSError, ValueError, KeyError, TypeError):
        saved = {}
    keys = ("los_k", "los_min_speed", "los_mode", "los_a", "los_b", "los_c", "los_far")
    return {k: type(getattr(cfg, k))(saved.get(k, getattr(cfg, k))) for k in keys}


def _threats_meta(env):
    """What the page needs to know about the chasers in a run (None when there were none)."""
    t = env.threats
    if t is None or not t.active:
        return None
    return {"active": t.active, "slots": t.slots, "speed": t.max_speed, "speed_min": t.min_speed,
            "spread": t.spread, "spawn_min": t.spawn_min, "spawn": t.spawn, "buffer": t.buffer, "radius": t.radius,
            "body_radius": t.body_radius, "release_frac_range": t.release_frac_range, "release_frac": t.release_frac,
            "max_tilt_deg": math.degrees(t.max_tilt), "catch_radius": t.intercept_radius,
            "geometry_weights": list(t.geometry_weights) if t.spawn == "geometry" else None}


def run_episode_two_phase(model, env, start_pos, target_pos, target_yaw=0.0,
                           switch_dist=DEFAULT_SWITCH_DIST,
                           log_events=None, trace_every=0, cruise_speed=18.0, handoff_speed=None, track=False, recentre_z=None,
                           lateral_frac=0.0, tilt_deg=0.0, rng=None):
    """One two-phase episode: PID/carrot to switch_dist, then the model. trace_every > 0 logs per-step lines; info["diag"] has the miss analysis."""
    t0 = time.monotonic()
    hit_r = float(getattr(env, "hit_radius", HIT_THRESHOLD))

    def _log(msg):
        line = f"[{_fmt_elapsed(t0)}] {msg}"
        print(f"[record_run] {line}")
        if log_events is not None:
            log_events.append(line)

    _log("init")
    obs, _ = env.reset(start_pos=start_pos, target_pos=target_pos, target_yaw=target_yaw)
    agent = TwoPhaseAgent(model, switch_dist=switch_dist, device=device, cruise_speed=cruise_speed,
                           handoff_speed=handoff_speed, recentre_z=recentre_z, lateral_frac=lateral_frac, tilt_deg=tilt_deg, rng=rng)
    agent.reset(env)
    _log(f"pid guide  dist={env.prev_distance:.2f}m  gains={agent.gain_key}  target={_fmt_vec(target_pos)}  "
         f"hit_radius={hit_r:g}m  switch_dist={switch_dist}m  max_steps={env.max_steps} "
         f"(~{env.max_steps * env.dt:.0f}s sim)")

    positions, dists, speeds = [], [], []
    quats, phases = [], []          # only filled when track=True (3D viewer playback)
    los_cos = []                    # cos(angle between velocity and the line of sight to the target), per step
    los_cfg = getattr(env, "los_cfg", None)
    done = False
    info = {"reason": None}
    total_reward = 0.0
    d0 = float(env.prev_distance)
    diag = {"min_dist": d0, "min_step": 0, "min_speed": 0.0, "handoff_step": None, "handoff_dist": None,
            "handoff_speed": None, "handoff_closing": None, "handoff_lateral": None, "peak_speed": 0.0,
            "steps_within_1m": 0, "stall_steps": 0, "approaches": 0, "rl_steps": 0, "hit_radius": hit_r}
    inside2 = False
    step_i = 0
    threats = getattr(env, "threats", None)
    n_threats = threats.active if threats is not None else 0
    threat_frames, threat_dists, threat_speeds, threat_los, threat_vels = [], [], [], [], []         # only filled when track=True and chasers are on
    dyn = DynRecorder(env)
    diag["min_threat_dist"] = None if not n_threats else float("inf")
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
            for line in dyn.log_lines("OPEN  (policy takes over)"):
                _log(line)
        if threats is not None and agent.phase == "RL":      # chasers hover until the policy flies AND the drone has closed to their release fraction
            p, v = env.drone_state.position, env.drone_state.velocity
            threats.spawn_at_handoff([p.x, p.y, p.z], [v.x, v.y, v.z], env.target_pos)      # spawn="geometry": chasers appear now
            threats.update_release(float(env.prev_distance), d0 if diag["handoff_dist"] is None else diag["handoff_dist"])
        obs, reward, terminated, truncated, info = env.step(action)
        dyn.add()
        step_i += 1
        total_reward += reward
        positions.append([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
        d = float(env.prev_distance)
        speed, closing, lateral = _episode_kin(env)
        dists.append(d)
        speeds.append(speed)
        los_cos.append(closing / speed if speed > 1e-6 else 0.0)
        if n_threats:
            dt_near = threats.nearest_distance([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
            if not np.isfinite(dt_near):         # geometry spawn: nobody there yet (JSON has no infinity)
                dt_near = 1.0e4
            diag["min_threat_dist"] = min(diag["min_threat_dist"], dt_near)
            if track:
                threat_frames.append(np.round(threats.pos[:n_threats], 3).tolist())
                threat_dists.append(dt_near)
                c_speed, c_los, c_vel = threats.chaser_states([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
                threat_speeds.append(c_speed)
                threat_los.append(c_los)
                threat_vels.append(np.round(c_vel, 3).tolist())
        if track:
            q = env.drone_state.orientation
            quats.append((q.x, q.y, q.z, q.w))
            phases.append(1 if agent.phase == "RL" else 0)

        if d < diag["min_dist"]:
            diag.update(min_dist=d, min_step=step_i, min_speed=speed)
        diag["peak_speed"] = max(diag["peak_speed"], speed)
        if d < 1.0:
            diag["steps_within_1m"] += 1
            if d > hit_r and speed < 0.3:
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
    for line in dyn.log_lines(f"CLOSE (episode ended: {outcome})"):
        _log(line)
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
        info["track"] = _build_track(positions, quats, phases, dists, speeds, env.dt, threat_frames, threat_dists, dyn,
                                     los=None if los_cfg is None else (los_cos, speeds, los_cfg),
                                     threat_radius=threats.body_radius if n_threats else None,
                                     threat_motion=(threat_speeds, threat_los, threat_vels))

    if n_threats:
        _log(f"chasers: {n_threats} active, closest any came to the drone {diag['min_threat_dist']:.2f}m "
             f"(catch radius {threats.intercept_radius:g}m)" + ("  -> CAUGHT" if outcome == "Intercepted" else ""))
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
        gap = diag["min_dist"] - hit_r
        if diag["min_dist"] < 2.0:
            _log(f"NEAR MISS: closest {diag['min_dist']:.2f}m vs hit radius {hit_r:g}m (missed by {gap:.2f}m)")
        if n and diag["stall_steps"] > 0.2 * n:
            _log(f"STALL: {100.0 * diag['stall_steps'] / n:.0f}% of the episode hovering 0.25-1m from the target "
                 f"at <0.3m/s -- the policy is parking next to the target instead of closing the last bit")
        elif diag["approaches"] >= 3:
            _log(f"LOOPING: {diag['approaches']} separate approaches within 2m -- swings past the target and "
                 f"has to turn around")
    return (np.array(positions, dtype=np.float32), np.array(dists, dtype=np.float32),
            info, total_reward, log_events or [])


MAX_TRACK_FRAMES = 3000


POLICY_HZ = 60.0     # the rc reward is paid once per policy step; the numpy env steps at 240 Hz


def _smoothstep_np(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _build_track(positions, quats, phases, dists, speeds, dt, threat_frames=None, threat_dists=None, dyn=None, los=None,
                 threat_radius=None, threat_motion=None, targets=None):
    """Downsampled per-frame pose for the browser's three.js player (z-up, same frame as the sim).
    The first and last steps are always kept."""
    n = len(positions)
    stride = max(1, n // MAX_TRACK_FRAMES)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    track = {
        "dt": float(dt) * stride,
        "stride": stride,
        "steps": idx,
        "pos": [[round(float(v), 3) for v in positions[i]] for i in idx],
        "quat": [[round(float(v), 4) for v in quats[i]] for i in idx],
        "phase": [phases[i] for i in idx],
        "dist": [round(float(dists[i]), 3) for i in idx],
        "speed": [round(float(speeds[i]), 3) for i in idx],
    }
    if los is not None:        # line of sight: angle velocity<->target and the rc reward's los term (per policy step, and summed)
        cos_all, speed_all, cfg = los
        cos_np = np.clip(np.asarray(cos_all, dtype=np.float64), -1.0, 1.0)
        if cfg["los_mode"] == "penalty":          # -k * m(angle), the same bands as angle_reward_rc.compute_reward_rc
            deg = np.degrees(np.arccos(cos_np))
            m = cfg.get("los_aligned", 1.0) + _smoothstep_np(cfg["los_a"], cfg["los_b"], deg) + cfg["los_far"] * _smoothstep_np(cfg["los_b"], cfg["los_c"], deg)
            value = -cfg["los_k"] * m
        else:
            value = cfg["los_k"] * cos_np
        term = np.where(np.asarray(speed_all) > cfg["los_min_speed"], value, 0.0)
        total = np.cumsum(term) * (float(dt) * POLICY_HZ)          # each 240 Hz step is 1/4 of a policy step
        track["los_deg"] = [round(float(np.degrees(np.arccos(np.clip(cos_all[i], -1.0, 1.0)))), 2) for i in idx]
        track["los_reward"] = [round(float(term[i]), 4) for i in idx]
        track["los_reward_sum"] = [round(float(total[i]), 3) for i in idx]
        track["los_meta"] = {"k": float(cfg["los_k"]), "min_speed": float(cfg["los_min_speed"]), "mode": cfg["los_mode"]}
    if threat_frames:          # chasing drones: their positions per frame and the nearest one's distance to the drone
        track["threats"] = [threat_frames[i] for i in idx]
        track["threat_dist"] = [round(float(threat_dists[i]), 3) for i in idx]
        if threat_motion is not None:       # per chaser: speed, velocity, and how far its velocity points off its line to the drone
            speeds, los_deg, vels = threat_motion
            track["threat_speed"] = [[round(float(v), 2) for v in speeds[i]] for i in idx]
            track["threat_los_deg"] = [[None if np.isnan(v) else round(float(v), 1) for v in los_deg[i]] for i in idx]
            track["threat_vel"] = [vels[i] for i in idx]
        track["threat_radius"] = threat_radius      # m, the chaser body drawn by the player
    if targets is not None:    # moving target (acro --runner): where the target is at every frame
        track["target"] = [[round(float(v), 3) for v in targets[i]] for i in idx]
    if dyn is not None:        # forces, rotor thrusts and target angles per frame (see dyn_track.py)
        track["dyn"] = dyn.build(idx, positions, quats)
        track["dyn_meta"] = dyn.meta()
    return track


def render_dist_plot(diag, out_path, switch_dist, reason):
    """distance-to-target and speed vs time, with the hit radius, the PID->RL hand-off and the closest
    approach marked -- the same information the video title shows, but over the whole episode."""
    tr = np.array(diag["trace"], dtype=np.float64)
    if len(tr) == 0:
        return None
    t = tr[:, 0] / 240.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, tr[:, 1], color="tab:blue", label="distance to target")
    hit_r = diag.get("hit_radius", HIT_THRESHOLD)
    ax1.axhline(hit_r, color="tab:red", ls="--", label=f"hit radius {hit_r:g} m")
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
                                 seed=None, switch_dist=DEFAULT_SWITCH_DIST, cruise_speed=18.0, handoff_speed=None, recentre_z=None,
                                 lateral_frac=0.0, tilt_deg=0.0, threats=None, hit_radius=None):
    """Two-phase record_checkpoint: renders the mp4 and returns the phase-transition log."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = _make_two_phase_env(checkpoint_path, model, threats, rng, hit_radius)
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
        lateral_frac=lateral_frac, tilt_deg=tilt_deg, rng=rng,
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
                          cruise_speed=18.0, handoff_speed=None, recentre_z=None, lateral_frac=0.0, tilt_deg=0.0, threats=None,
                          hit_radius=None):
    """Like record_checkpoint_two_phase but skips the mp4/png: returns the episode stats plus a per-frame
    "track" (positions, orientation, phase, distance, speed, chaser positions) for the browser's three.js player."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = _make_two_phase_env(checkpoint_path, model, threats, rng, hit_radius)
    start_pos, target_pos = sample_target_at_distance(distance, rng)
    log_events = []
    _set_episode_budget(env, distance, log_events=log_events)
    note = _trained_range_note(checkpoint_path, switch_dist)
    if note:
        log_events.append(note)
    positions, dists, info, total_reward, log_events = run_episode_two_phase(
        model, env, start_pos, target_pos, switch_dist=switch_dist, log_events=log_events, trace_every=240,
        cruise_speed=cruise_speed, handoff_speed=handoff_speed, track=True, recentre_z=recentre_z,
        lateral_frac=lateral_frac, tilt_deg=tilt_deg, rng=rng)
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
        "hit_radius": float(env.hit_radius),
        "threats": _threats_meta(env),
        "log": log_events,
        "diag": {k: v for k, v in info["diag"].items() if k != "trace"},
        "track": info["track"],
    }


def evaluate_two_phase(checkpoint_path, n_scenarios=N_EVAL_SCENARIOS, distance=30.0,
                        seed=None, switch_dist=DEFAULT_SWITCH_DIST, cruise_speed=18.0, handoff_speed=None, recentre_z=None,
                        lateral_frac=0.0, tilt_deg=0.0, threats=None, hit_radius=None):
    """Two-phase evaluate_checkpoint: n_scenarios at the same distance, random directions, no rendering."""
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = _make_two_phase_env(checkpoint_path, model, threats, rng, hit_radius)
    budget = _set_episode_budget(env, distance)

    scenarios = []
    totals = {"hit": 0, "oob": 0, "attitude": 0, "timeout": 0, "caught": 0}
    all_log_lines = []
    for i in range(n_scenarios):
        start_pos, target_pos = sample_target_at_distance(distance, rng)
        env.max_steps = budget  # _set_episode_budget already logged/clamped once above
        positions, dists, info, total_reward, log_events = run_episode_two_phase(
            model, env, start_pos, target_pos, switch_dist=switch_dist,
            log_events=[], cruise_speed=cruise_speed, handoff_speed=handoff_speed, recentre_z=recentre_z,
            lateral_frac=lateral_frac, tilt_deg=tilt_deg, rng=rng,
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
            "final_speed": dg["final_speed"], "min_threat_dist": dg["min_threat_dist"],
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
             f"RL terminal at {switch_dist}m)" + (
                 f"  | {env.threats.active} chasing drone(s) of {env.threats.slots} slot(s): speed cap {env.threats.max_speed:g} m/s, "
                 f"{env.threats.spawn} spawn, spread {env.threats.spread:g} m, catch radius {env.threats.intercept_radius:g} m"
                 if env.threats.active else ""),
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
    for k in ("hit", "oob", "attitude", "timeout") + (("caught",) if env.threats.active else ()):
        lines.append(f"  {k:<10} {totals[k]:3d}/{n_scenarios}  ({pct[k]:.1f}%)")
    misses = [s for s in scenarios if not s["hit"]]
    near = [s for s in misses if s["min_dist"] < 1.0]
    lines.append("")
    lines.append(f"Near-misses (not a hit, but closest approach < 1 m): {len(near)}/{len(misses)} of the misses")
    if misses:
        lines.append(f"  mean closest approach of misses: {np.mean([s['min_dist'] for s in misses]):.2f} m "
                     f"(hit radius {env.hit_radius:g} m); mean final distance: "
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
    (angle_rtl_train_isaac.py --obs-position-mode); checkpoints without one are 'full' (the original layout)."""
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")
        with open(cfg_path) as f:
            return json.load(f).get("obs_position_mode", "full")
    except (OSError, ValueError):
        return "full"


def _drone_style_for(checkpoint_path):
    """'acro' or 'angle': which plant the checkpoint was trained on. Acro policies output body rates (bounds 6, 6, 3 rad/s), angle
    policies torques (0.5 N*m). Acro checkpoints saved before the aliasing fix carry the torque bounds, so the run_config.json
    next to the checkpoint decides first (only acro_rtl_train_isaac.py writes a `tilt_limit` key); the saved bounds are the fallback."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            if "tilt_limit" in json.load(f):
                return "acro"
    except (OSError, ValueError):
        pass
    try:
        raw = torch.load(checkpoint_path, map_location="cpu")
        return "acro" if float(raw["action_high"][1]) > ACRO_MIN_RATE_BOUND else "angle"
    except (OSError, KeyError, IndexError, ValueError, RuntimeError):
        return "angle"


def _make_eval_env(checkpoint_path, max_steps):
    style = _drone_style_for(checkpoint_path)
    if style == "acro":
        print(f"[record_run] {os.path.basename(checkpoint_path)}: acro checkpoint -> acro plant "
              f"(note: the numpy reward still crashes the episode at roll > 65 / pitch > 80 deg; use evaluate_acro)")
    return BaseDroneEnv(reward_func, max_steps=max_steps, position_mode=_position_mode_for(checkpoint_path),
                        drone_style=style)


def _trained_range_note(checkpoint_path, switch_dist):
    """If the checkpoint sits next to a run_config.json (angle_rtl_train_isaac.py writes one) that records the
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
        "caught": reason == "Intercepted",      # a chasing drone reached it
    }


def _spawn_speed_range_for(checkpoint_path, start="auto"):
    """(low, high) m/s the episode starts with, toward the target; None = from rest. start="auto" reads the run_config.json next to
    the checkpoint (the trainer's --spawn-speed-low/high apply to its eval too, so a run trained with them is scored launched);
    start="rest" forces a start from rest."""
    if start == "rest":
        return None
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            cfg = json.load(f)
        lo, hi = cfg.get("spawn_speed_low"), cfg.get("spawn_speed_high")
        if lo is not None and hi is not None:
            return (min(lo, hi), max(lo, hi))
    except (OSError, ValueError):
        pass
    return None


def _acro_overrides(checkpoint_path, tilt_limit_deg, tilt_margin_deg, hit_radius=None):
    """AcroRewardConfig overrides for the numpy acro eval. The hit zone is `hit_radius` if given, else the one the run was trained
    with (run_config.json -> acro_reward.hit_threshold), else the reward's default."""
    over = {"tilt_limit_deg": tilt_limit_deg, "tilt_hard_margin_deg": tilt_margin_deg}
    if hit_radius is None:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
                trained = json.load(f).get("acro_reward", {})
        except (OSError, ValueError):
            trained = {}
        hit_radius = trained.get("hit_threshold")
        if "zone_time" in trained:
            over["zone_time"] = float(trained["zone_time"])      # the run's zone time goes with its radius
    if hit_radius is not None:
        over["hit_threshold"] = float(hit_radius)
    return over


def _acro_los_cfg_for(checkpoint_path):
    """The line-of-sight settings of the acro run (run_config.json -> acro_reward), else acro reward v2's defaults, so the viewer
    shows what the term pays even for a v1 checkpoint."""
    from app.reward_functions.acro_reward_v2 import AcroRewardV2Config
    cfg = AcroRewardV2Config()
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            saved = json.load(f)["acro_reward"]
    except (OSError, ValueError, KeyError, TypeError):
        saved = {}
    keys = ("los_k", "los_min_speed", "los_mode", "los_a", "los_b", "los_c", "los_far", "los_aligned")
    return {k: type(getattr(cfg, k))(saved.get(k, getattr(cfg, k))) for k in keys}


def _acro_near_for(checkpoint_path):
    """(near_radius m, near_limit_s s) of the acro run's near-miss timer (run_config.json -> acro_reward, acro_reward_v2 B term).
    A v1 run has no timer: (0, 0) = off, so its episodes still end only on hit / oob / tilt / hover / timeout."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            saved = json.load(f)["acro_reward"]
        return float(saved.get("near_radius", 0.0)), float(saved.get("near_limit_s", 0.0))
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0, 0.0


ISAAC_EPISODE_S = 62.5    # BaseDroneEnvIsaacCfg.episode_length_s (MAX_POLICY_STEPS * PHYSICS_DT)


def _model_obs_dim(model):
    """Number of inputs the loaded ActorCritic takes (its first Linear layer)."""
    return next(m for m in model.modules() if isinstance(m, torch.nn.Linear)).in_features


def _acro_runner_for(checkpoint_path, model, overrides=None, rng=None):
    """The moving-target settings of an acro run (run_config.json: --runner and its options), or None for a fixed target.
    `overrides` (dict, None values ignored) replaces speed / speed_min / escape_dist from the page. A checkpoint with the 3 runner inputs
    but no run_config still gets a runner (8 m/s, 100 m), because its inputs need one."""
    cfg = {}
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        pass
    if not cfg.get("runner") and _model_obs_dim(model) != 23 + RUNNER_FEATURES:
        return None
    o = {k: v for k, v in (overrides or {}).items() if v is not None}
    speed = float(o.get("speed", cfg.get("runner_speed") or 8.0))
    speed_min = o.get("speed_min", cfg.get("runner_speed_min"))
    speed_min = speed if speed_min is None else min(float(speed_min), speed)
    return {"speed": speed, "speed_min": speed_min,
            "escape_dist": float(o.get("escape_dist", cfg.get("runner_escape_dist") or 100.0)),
            "min_alt": float(cfg.get("runner_min_alt") or 2.0),
            "max_tilt": math.radians(float(cfg.get("los_max_tilt_deg") or 35.0)),
            "episode_s": float(cfg.get("episode_s") or ISAAC_EPISODE_S), "rng": rng}


def _make_runner(runner_cfg):
    r = Runner(runner_cfg["speed"], min_speed=runner_cfg["speed_min"], escape_dist=runner_cfg["escape_dist"], min_alt=runner_cfg["min_alt"],
               max_tilt=runner_cfg["max_tilt"], rng=runner_cfg["rng"])
    r.episode_s = runner_cfg["episode_s"]
    return r


def _carrot_rate_action(env, carrot):
    """The carrot controller's action as an acro action: its torque becomes the rate command that gives the same first-substep torque,
    cmd = w + tau / (I * kp) (the trainer's angle_to_acro)."""
    tau = carrot.compute_action(env.drone_state, env.target_pos, env.target_yaw)
    rc, av = env.rate_controller, env.drone_state.angular_velocity
    cmd = np.array([av.x, av.y, av.z]) + tau[1:4] / (rc.inertia * rc.kp)
    return np.clip(np.concatenate([[tau[0]], cmd]), env.action_space.low, env.action_space.high)


def _roll_pitch_deg(q):
    """Display-only roll / pitch (deg) of a quaternion with .x .y .z .w (ZYX Euler; ambiguous once the drone is flipped, so the log also prints the tilt)."""
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x ** 2 + q.y ** 2))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return math.degrees(roll), math.degrees(pitch)


ACRO_LOG_NEAR_M = 40.0     # inside this distance every policy step is logged; farther out one in ACRO_LOG_FAR_EVERY
ACRO_LOG_FAR_EVERY = 6


def _acro_episode(model, env, cfg, dist0, start_pos, target_pos, yaw, action_repeat, record=False, spawn_speed=0.0, handoff=None, log=None,
                  near=(0.0, 0.0), runner=None):
    """One acro episode in the numpy plant, ended exactly like isaac_eval (hit / oob / tilt / hover / near_miss / timeout).
    near = (radius m, limit s) of the run's near-miss timer (acro_reward_v2): the episode ends 'near_miss' once the policy has spent
    limit s in total within radius m of the target (0 = off).
    record=True also returns the per-policy-step pose for the 3D player.
    handoff (dict h, max_speed, speed, lateral_frac, tilt_deg, rng) = two-stage episode like the trainer's --handoff: the carrot controller
    flies from the spawn point (dist0 from the target, from rest) down to h m from the target, where the velocity (speed toward the
    target + random sideways part) and the attitude are snapped and the policy takes over. None = the policy flies from the start.
    runner (Runner) = moving target, as the trainer's --runner: it hovers on the target until the policy takes over, then flees and IS
    the target; the policy gets its velocity as 3 extra inputs, and the episode ends 'escape' past its escape distance."""
    step_dt = action_repeat * PHYSICS_DT
    grace_steps = max(1, int(round(cfg.hover_grace_s / step_dt)))
    limit_steps = max(grace_steps + 1, int(round(cfg.hover_limit_s / step_dt)))
    limit_hit = max(1, int(round(cfg.zone_time / step_dt)))      # consecutive steps inside the hit zone (the trainer's zone_time)
    zone = 0
    tilt_crash = math.radians(cfg.tilt_limit_deg + cfg.tilt_hard_margin_deg)
    obs, _ = env.reset(start_pos=start_pos, target_pos=target_pos, target_yaw=yaw)
    target_pos = np.asarray(target_pos, dtype=np.float64).copy()      # moves with the runner; the caller's array is left alone
    if runner is not None:
        runner.reset(target_pos)

    def full_obs(base):
        """The policy's inputs: the env's 23, plus the runner's 3 (target velocity - drone velocity) when there is one."""
        if runner is None:
            return base
        dv = env.drone_state.velocity
        return np.concatenate([base, (runner.obs_velocity() - np.array([dv.x, dv.y, dv.z])) / VEL_SCALE]).astype(np.float32)

    obs = full_obs(obs)
    carrot = None
    if handoff is not None:
        carrot = CarrotController(load_gains_by_dist(), handoff["max_speed"], handoff["speed"])
        carrot.reset(env.drone_state, target_pos, handoff["h"])
    elif spawn_speed > 0.0:   # launched toward the target, like the Isaac env's set_spawn_speed_range
        u = (np.asarray(target_pos, dtype=np.float64) - np.asarray(start_pos, dtype=np.float64))
        u = u / max(np.linalg.norm(u), 1e-9)
        env.drone_state.velocity = Vector3D(*(u * spawn_speed))
        obs = full_obs(env._get_obs())
    budget = max(1, int(round(steps_for_dist(dist0) / action_repeat)))
    if runner is not None:     # a fleeing target has no per-distance budget: hit, crash, escape or the Isaac episode cap
        budget = max(1, int(round(runner.episode_s / (action_repeat * PHYSICS_DT))))
    oob_radius = max(dist0 * cfg.oob_start_dist_mult, cfg.oob_min_radius)
    reason, stall, n, dist = "timeout", 0, 0, dist0
    near_radius, near_limit_s = near
    near_limit = int(round(near_limit_s / step_dt)) if near_limit_s > 0 else 0
    near_count = 0
    d_min, speed_sum, tilt_max = float("inf"), 0.0, 0.0
    d_min_step, d_min_speed, pol_dists = 0, 0.0, []     # closest approach step / speed, and the distance at every policy step (for the log's diagnosis)
    on_model, ho = carrot is None, None
    rec = {"pos": [], "quat": [], "dist": [], "speed": [], "phase": [], "los_cos": [], "target": [], "dyn": DynRecorder(env) if record else None}
    for n in range(1, budget + 1):
        if not on_model and dist <= handoff["h"]:
            on_model = True
            vel = snap_handoff_state(env.drone_state, target_pos, handoff["speed"], handoff["lateral_frac"], handoff["tilt_deg"], handoff["rng"])
            obs = full_obs(env._get_obs())
            to_t = np.asarray(target_pos, dtype=np.float64) - np.array([env.drone_state.position.x, env.drone_state.position.y, env.drone_state.position.z])
            closing = float(vel @ to_t) / max(float(np.linalg.norm(to_t)), 1e-9)
            ho = {"step": n, "dist": dist, "speed": float(np.linalg.norm(vel)), "lateral": float(np.sqrt(max(float(vel @ vel) - closing ** 2, 0.0)))}
            if log is not None:
                log.append(f"controller flew {dist0:.1f} m -> {handoff['h']:.1f} m from the target (max {handoff['max_speed']:g} m/s); "
                           f"policy took over at step {n}: {dist:.2f} m out, {ho['speed']:.1f} m/s ({ho['lateral']:.1f} sideways)")
        if on_model:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model.scale_action(model.forward(obs_t)[0]).squeeze(0).cpu().numpy()
        else:
            action = _carrot_rate_action(env, carrot)
        if runner is not None and on_model and not runner.released:
            p0 = env.drone_state.position
            runner.release([p0.x, p0.y, p0.z])
            if log is not None:
                log.append(f"runner released at step {n}: flees at {runner.cap:.1f} m/s, escape once it is {runner.escape_dist:g} m "
                           f"from its spawn point ({', '.join(f'{c:.1f}' for c in runner.spawn)})")
        for _ in range(action_repeat):
            if runner is not None:
                p0 = env.drone_state.position
                runner.step([p0.x, p0.y, p0.z], env.dt)
            obs, _r, _te, _tr, _info = env.step(action)
            if runner is not None and runner.released:
                target_pos = runner.pos.copy()
                env.target_pos = target_pos
        obs = full_obs(env._get_obs()) if runner is not None else obs
        p, v, q = env.drone_state.position, env.drone_state.velocity, env.drone_state.orientation
        pos = np.array([p.x, p.y, p.z])
        dist = float(np.linalg.norm(target_pos - pos))
        speed = float(np.linalg.norm([v.x, v.y, v.z]))
        tilt = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q.x ** 2 + q.y ** 2))))
        speed_sum += speed
        tilt_max = max(tilt_max, tilt)
        if on_model:
            if dist < d_min:
                d_min, d_min_step, d_min_speed = dist, n, speed    # closest approach counts from the takeover on, like the trainer's eval
            pol_dists.append(dist)
            if log is not None and (dist < ACRO_LOG_NEAR_M or n % ACRO_LOG_FAR_EVERY == 0):
                vv = np.array([v.x, v.y, v.z])
                closing = float(vv @ (target_pos - pos)) / max(dist, 1e-9)
                lateral = math.sqrt(max(speed ** 2 - closing ** 2, 0.0))
                los_deg = math.degrees(math.acos(max(-1.0, min(1.0, closing / speed)))) if speed > 1e-6 else float("nan")
                roll_d, pitch_d = _roll_pitch_deg(q)
                w = env.drone_state.angular_velocity
                log.append(f"step {n:5d} t={n * step_dt:7.2f}s pos=({pos[0]:8.2f},{pos[1]:8.2f},{pos[2]:8.2f}) dist={dist:7.2f} speed={speed:5.2f} "
                           f"closing={closing:+6.2f} lateral={lateral:5.2f} vel-to-LOS={los_deg:5.1f}deg roll/pitch={roll_d:+6.1f}/{pitch_d:+6.1f} "
                           f"tilt={math.degrees(tilt):5.1f} w=({w.x:+5.2f},{w.y:+5.2f},{w.z:+5.2f}) action=[{', '.join(f'{float(a_):+.2f}' for a_ in action)}]")
        if record:
            rec["pos"].append(pos.tolist())
            rec["quat"].append((q.x, q.y, q.z, q.w))
            rec["dist"].append(dist)
            rec["speed"].append(speed)
            rec["phase"].append(1 if on_model else 0)
            vel = np.array([v.x, v.y, v.z])
            rec["los_cos"].append(float(vel @ (target_pos - pos)) / (speed * dist) if speed > 1e-6 and dist > 1e-9 else 0.0)
            rec["target"].append(target_pos.tolist())
            rec["dyn"].add()
        stall = stall + 1 if (on_model and dist > cfg.hit_threshold and speed < cfg.stall_speed) else 0
        oob_r = oob_radius if runner is None else max(oob_radius, float(np.linalg.norm(runner.spawn)) + runner.escape_dist
                                                      + (handoff["h"] if handoff is not None else dist0))   # trainer's oob_min
        crash_oob = bool(np.isnan(pos).any()) or pos[2] < 0.0 or np.linalg.norm(pos) > oob_r
        if crash_oob or tilt > tilt_crash:
            reason = "oob" if crash_oob else "tilt"
            break
        zone = zone + 1 if dist < cfg.hit_threshold else 0
        if zone >= limit_hit:
            reason = "hit"
            break
        if runner is not None and runner.escaped():
            reason = "escape"
            break
        if stall >= limit_steps:
            reason = "hover"
            break
        if on_model and near_limit and dist < near_radius:
            near_count += 1
            if near_count >= near_limit:
                reason = "near_miss"
                break
    if d_min == float("inf"):
        d_min = dist
    if log is not None and pol_dists:
        log.append(f"closest approach {d_min:.2f} m at step {d_min_step} (speed {d_min_speed:.2f} m/s); episode ended '{reason}' at step {n}, {dist:.2f} m out")
        if reason != "hit":
            log.append(f"NEAR MISS: closest {d_min:.2f} m vs hit radius {cfg.hit_threshold:g} m (missed by {d_min - cfg.hit_threshold:.2f} m)")
            after = pol_dists[max(0, len(pol_dists) - 1 - (n - d_min_step)):]      # distances from the closest approach on
            rise = (max(after) - d_min) if after else 0.0
            if rise > 3.0:
                log.append(f"PASSED AND DRIFTED AWAY: after the closest approach it got {rise:.1f} m further out (final {dist:.1f} m)")
            inside, approaches = False, 0
            for dd in pol_dists:
                if dd < 2.0 and not inside:
                    inside, approaches = True, approaches + 1
                elif dd > 3.0:
                    inside = False
            if approaches >= 3:
                log.append(f"LOOPING: {approaches} separate approaches within 2 m")
    out = {"reason": reason, "steps": n, "final_dist": dist, "min_dist": d_min, "speed_sum": speed_sum, "tilt_max": tilt_max,
           "handoff": ho, "runner_speed": runner.cap if runner is not None else None}
    if record:
        out["rec"] = rec
    return out


def _handoff_cfg(h, max_speed, speed, lateral_frac, tilt_deg, rng):
    """Handoff settings of one acro episode. speed=None = max_speed / 2 (the angle two-phase default)."""
    speed = float(speed) if speed is not None else float(max_speed) / 2
    return {"h": float(h), "max_speed": float(max_speed), "speed": min(speed, float(max_speed)), "lateral_frac": float(lateral_frac),
            "tilt_deg": float(tilt_deg), "rng": rng}


def _make_acro_env(checkpoint_path):
    return BaseDroneEnv(lambda e: (0.0, False, None), pid_gains_path=None, max_steps=MAX_STEPS_CAP,
                        position_mode=_position_mode_for(checkpoint_path), drone_style="acro")


def trajectory_acro(checkpoint_path, distance=4.0, seed=None, tilt_limit_deg=120.0, tilt_margin_deg=30.0, action_repeat=4,
                    start="auto", spawn_dist=None, max_speed=18.0, handoff_speed=None, lateral_frac=0.3, tilt_deg=15.0,
                    hit_radius=None, runner=None):
    """One acro episode with a per-frame track for the browser's 3D player (same format as trajectory_two_phase).
    spawn_dist=None: the policy flies from the start (`distance` from the target, launched or from rest per `start`).
    spawn_dist=D: two-stage episode, the carrot controller flies from D m out (from rest, up to max_speed) to the handoff point
    `distance` m from the target, where the policy takes over at handoff_speed (None = max_speed / 2) with a
    random sideways speed <= lateral_frac and tilt <= tilt_deg.
    runner: page overrides (speed, speed_min, escape_dist) of a --runner run's moving target; the run_config decides whether there is one."""
    from app.reward_functions.acro_reward_v1 import AcroRewardConfig
    if _drone_style_for(checkpoint_path) != "acro":
        raise ValueError(f"{checkpoint_path} is not an acro checkpoint")
    cfg = AcroRewardConfig.from_overrides(_acro_overrides(checkpoint_path, tilt_limit_deg, tilt_margin_deg, hit_radius))
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    env = _make_acro_env(checkpoint_path)
    dist0 = spawn_dist if spawn_dist is not None else distance
    start_pos, target_pos = sample_target_at_distance(dist0, rng)
    yaw = float(rng.uniform(-math.pi, math.pi))
    sp = None if spawn_dist is not None else _spawn_speed_range_for(checkpoint_path, start)
    speed0 = float(rng.uniform(*sp)) if sp else 0.0
    ho_cfg = _handoff_cfg(distance, max_speed, handoff_speed, lateral_frac, tilt_deg, rng) if spawn_dist is not None else None
    flight_log = []
    runner_cfg = _acro_runner_for(checkpoint_path, model, runner, rng)
    ep = _acro_episode(model, env, cfg, dist0, start_pos, target_pos, yaw, action_repeat, record=True, spawn_speed=speed0, handoff=ho_cfg,
                       log=flight_log, near=_acro_near_for(checkpoint_path), runner=_make_runner(runner_cfg) if runner_cfg else None)
    rec, step_dt = ep["rec"], action_repeat * PHYSICS_DT
    track = _build_track(rec["pos"], rec["quat"], rec["phase"], rec["dist"], rec["speed"], step_dt, dyn=rec["dyn"],
                         los=(rec["los_cos"], rec["speed"], _acro_los_cfg_for(checkpoint_path)),
                         targets=rec["target"] if runner_cfg else None)
    ho = ep["handoff"]
    return {
        "reason": ep["reason"], "steps": ep["steps"], "final_dist": ep["final_dist"], "total_reward": 0.0,
        "start_pos": start_pos.tolist(), "target_pos": target_pos.tolist(), "distance": dist0,
        "switch_dist": float(distance) if spawn_dist is not None else 0.0,
        "position_mode": env.position_mode, "hit_radius": float(cfg.hit_threshold), "action_repeat": action_repeat,
        "spawn_speed": speed0,
        "runner": ({"speed": ep["runner_speed"], "escape_dist": runner_cfg["escape_dist"], "spawn": target_pos.tolist()} if runner_cfg else None),
        "log": ([f"controller flew {dist0:.1f} m toward the handoff point {distance:.1f} m out, but never got there"]
                if spawn_dist is not None and not ho else [])
               + flight_log
               + [f"acro episode: {ep['reason']} after {ep['steps']} policy steps ({ep['steps'] * step_dt:.1f} s), "
                  f"closest {ep['min_dist']:.2f} m, max tilt {math.degrees(ep['tilt_max']):.0f} deg"],
        "diag": {"min_dist": ep["min_dist"], "max_tilt_deg": math.degrees(ep["tilt_max"]),
                 "mean_speed": ep["speed_sum"] / max(ep["steps"], 1)},
        "track": track,
    }


ACRO_ENDINGS = ("hit", "oob", "tilt", "hover", "near_miss", "escape", "timeout")   # escape: --runner target got away


def evaluate_acro(checkpoint_path, n_scenarios=N_EVAL_SCENARIOS, distance_low=3.0, distance_high=5.0, seed=None,
                  tilt_limit_deg=120.0, tilt_margin_deg=30.0, action_repeat=4, start="auto",
                  spawn_dist_low=None, spawn_dist_high=None, max_speed=18.0, handoff_speed=None, lateral_frac=0.3, tilt_deg=15.0,
                  hit_radius=None, runner=None):
    """Acro-checkpoint eval in the numpy plant, scored like the trainer's isaac_eval (acro_rtl_train_isaac.py):
    clean start from rest at (0, 0, 5), target at a random distance/direction/yaw, deterministic tanh(mean), first
    episode only, per-distance time budget (steps_for_dist). Endings: hit / oob / tilt / hover / timeout, with the
    thresholds of AcroRewardConfig (NOT the numpy reward's roll 65 / pitch 80 deg crash).

    action_repeat: physics steps (240 Hz) each policy action is held. Isaac runs the policy at 60 Hz = 4 substeps; the numpy
    env steps one physics step per call, so 1 would run the policy at 240 Hz, a different control rate than it trained on.

    spawn_dist_low/high set: handoff mode (see trajectory_acro). distance_low/high is then the HANDOFF distance range (the policy's
    range, as in the trainer), the spawn distance is U(spawn_dist_low, spawn_dist_high), and the time budget is that of the spawn distance."""
    from app.reward_functions.acro_reward_v1 import AcroRewardConfig
    cfg = AcroRewardConfig.from_overrides(_acro_overrides(checkpoint_path, tilt_limit_deg, tilt_margin_deg, hit_radius))
    rng = np.random.default_rng(seed)
    model = load_model_for_inference(checkpoint_path)
    style = _drone_style_for(checkpoint_path)
    if style != "acro":
        raise ValueError(f"{checkpoint_path} is not an acro checkpoint (its action bounds are angle-style torques)")
    env = _make_acro_env(checkpoint_path)
    step_dt = action_repeat * PHYSICS_DT

    two_stage = spawn_dist_low is not None and spawn_dist_high is not None
    sp = None if two_stage else _spawn_speed_range_for(checkpoint_path, start)
    near = _acro_near_for(checkpoint_path)
    runner_cfg = _acro_runner_for(checkpoint_path, model, runner, rng)
    runner_obj = _make_runner(runner_cfg) if runner_cfg else None
    scenarios, totals = [], {k: 0 for k in ACRO_ENDINGS}
    for i in range(n_scenarios):
        h = float(rng.uniform(distance_low, distance_high))
        dist0 = float(rng.uniform(min(spawn_dist_low, spawn_dist_high), max(spawn_dist_low, spawn_dist_high))) if two_stage else h
        start_pos, target_pos = sample_target_at_distance(dist0, rng)
        yaw = float(rng.uniform(-math.pi, math.pi))
        speed0 = float(rng.uniform(*sp)) if sp else 0.0
        ho_cfg = _handoff_cfg(h, max_speed, handoff_speed, lateral_frac, tilt_deg, rng) if two_stage else None
        ep = _acro_episode(model, env, cfg, dist0, start_pos, target_pos, yaw, action_repeat, spawn_speed=speed0, handoff=ho_cfg,
                           near=near, runner=runner_obj)
        reason, n, dist = ep["reason"], ep["steps"], ep["final_dist"]
        totals[reason] += 1
        scenarios.append({"idx": i, "spawn": start_pos.tolist(), "target": target_pos.tolist(), "dist": dist0,
                          "reason": reason, "final_dist": dist, "min_dist": ep["min_dist"], "steps": n,
                          "spawn_speed": speed0 if not two_stage else (ep["handoff"] or {}).get("speed", 0.0), "handoff_dist": h if two_stage else None,
                          "time_s": n * step_dt, "mean_speed": ep["speed_sum"] / max(n, 1),
                          "tilt_max_deg": math.degrees(ep["tilt_max"]), "hit": reason == "hit", "runner_speed": ep["runner_speed"]})
        print(f"[evaluate_acro] {i + 1}/{n_scenarios} dist={dist0:.1f}m -> {reason} final={dist:.2f}m closest={ep['min_dist']:.2f}m")

    pct = {k: 100.0 * v / n_scenarios for k, v in totals.items()}
    lines = [f"Checkpoint: {checkpoint_path}",
             f"Mode: acro eval in the numpy plant, scored like isaac_eval (clean start from rest, deterministic, first episode)",
             f"Scenarios: {n_scenarios}  " + (f"handoff point ~ U({distance_low}, {distance_high}) m from the target, spawn ~ U({spawn_dist_low}, {spawn_dist_high}) m"
                                              if two_stage else f"distance ~ U({distance_low}, {distance_high}) m") + ", random direction + yaw",
             ("Start: carrot controller from rest (max " + f"{max_speed:g} m/s) to the handoff point, then the policy at "
              + (f"{(handoff_speed if handoff_speed is not None else max_speed / 2):g} m/s")
              + f" with sideways <= {lateral_frac:.0%}, tilt <= {tilt_deg:g} deg") if two_stage else
             "Start: " + (f"launched toward the target at U({sp[0]:g}, {sp[1]:g}) m/s (from the run's --spawn-speed)" if sp else "from rest"),
             f"Policy rate: {action_repeat} physics step(s) per action = {1.0 / step_dt:.0f} Hz (Isaac trains at 60 Hz = 4)",
             f"Hit radius {cfg.hit_threshold:g} m",
             (f"Moving target (runner): flees at U({runner_cfg['speed_min']:g}, {runner_cfg['speed']:g}) m/s once the policy takes over; "
              f"escape (loss) once it is {runner_cfg['escape_dist']:g} m from its spawn point; episode cap {runner_cfg['episode_s']:g} s "
              f"instead of the distance budget")
             if runner_cfg else "Fixed target",
             f"Tilt limit {cfg.tilt_limit_deg:g} deg, crash at {cfg.tilt_limit_deg + cfg.tilt_hard_margin_deg:g} deg", ""]
    for s in scenarios:
        lines.append(f"[{s['idx']:3d}] dist={s['dist']:5.2f}m" + (f" h={s['handoff_dist']:5.2f}m" if s["handoff_dist"] is not None else "") + f" -> {s['reason']:<8} final={s['final_dist']:6.2f}m "
                     f"closest={s['min_dist']:5.2f}m v0={s['spawn_speed']:4.1f} time={s['time_s']:5.1f}s speed={s['mean_speed']:4.1f}m/s "
                     f"max_tilt={s['tilt_max_deg']:4.0f}deg")
    lines += ["", "Totals:"] + [f"  {k:<8} {totals[k]:3d}/{n_scenarios}  ({pct[k]:.1f}%)" for k in ACRO_ENDINGS]
    bands = [(lo, hi) for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 1e9))
             if any(lo <= s["dist"] < hi for s in scenarios)]
    lines += ["", "Hit rate by start distance:"]
    for lo, hi in bands:
        grp = [s for s in scenarios if lo <= s["dist"] < hi]
        lines.append(f"  {lo:g}-{(str(hi) if hi < 1e9 else 'inf')} m: {sum(s['hit'] for s in grp)}/{len(grp)}")
    return {"checkpoint": checkpoint_path, "scenarios": scenarios, "totals": totals, "pct": pct,
            "action_repeat": action_repeat, "summary_text": "\n".join(lines)}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=None, help="path to a .pt checkpoint (default: latest)")
    parser.add_argument("--out", default="runs/demo.mp4")
    parser.add_argument("--distance-low", type=float, default=3.0)
    parser.add_argument("--distance-high", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n-scenarios", type=int, default=N_EVAL_SCENARIOS)
    parser.add_argument("--acro", action="store_true",
                         help="evaluate an ACRO checkpoint like the trainer's isaac_eval (see evaluate_acro); uses --distance-low/high, "
                              "--n-scenarios, --seed, --tilt-limit, --action-repeat")
    parser.add_argument("--hit-radius", type=float, default=None,
                         help="--acro only: hit zone in metres (default: the radius the run was trained with, else 0.25)")
    parser.add_argument("--start", choices=("auto", "rest"), default="auto",
                         help="--acro only: auto = launch like the run trained (run_config spawn speed), rest = from rest")
    parser.add_argument("--tilt-limit", type=float, default=120.0, help="--acro only: tilt limit (deg); crash at limit + 30")
    parser.add_argument("--action-repeat", type=int, default=4,
                         help="--acro only: physics steps each action is held (Isaac: 4 = 60 Hz policy; 1 = 240 Hz)")
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
                              "(what angle_rtl_train_isaac's re-centring trained it on); z is capped at this value. Off by default.")
    parser.add_argument("--distance", type=float, default=30.0,
                         help="--two-phase only: exact target distance (m) -- direction is random, distance is not.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    checkpoint = args.checkpoint or find_latest_checkpoint()
    print(f"[record_run] checkpoint: {checkpoint}")
    if args.acro:
        result = evaluate_acro(checkpoint, n_scenarios=args.n_scenarios, distance_low=args.distance_low,
                               distance_high=args.distance_high, seed=args.seed, tilt_limit_deg=args.tilt_limit,
                               action_repeat=args.action_repeat, start=args.start, hit_radius=args.hit_radius)
        print(result["summary_text"])
    elif args.two_phase:
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
    else:
        raise SystemExit("pick a mode: --two-phase [--evaluate] or --acro")
