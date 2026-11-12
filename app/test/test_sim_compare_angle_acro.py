"""Angle drone vs acro drone flown by the same LOS teacher, on the numpy plants (no Isaac, no torch).

Three drones fly the same scenario:
  angle        angle plant, LOS controller torque straight into the mixer (what the angle model / LOS teacher does natively)
  acro_teacher acro plant, the SAME LOS torque converted by the trainer's angle_to_acro (cmd = w + tau/(I*kp), clamped to the
               acro action bounds) and sent through the rate loop (P loop + torque limit). This is the label BC copies.
  acro_ideal   same as acro_teacher but with the rate cap and torque limit lifted (--ideal-*): shows how much of any gap
               is the acro limits and how much is the conversion itself.

Every parameter is a CLI flag and a SimConfig field; the defaults are the values the trainer uses today.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe app/test/test_sim_compare_angle_acro.py --out runs/sim_compare
    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -m pytest app/test/test_sim_compare_angle_acro.py -q

Output (in --out): sim_compare.png (graphs), <drone>.csv (every logged quantity, every policy step), params.json, summary.txt.
The numpy plant has no PhysX gyroscopic coupling and no per-step PhysX solver; everything else is the trainer's maths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from app.control.los_np import velocity_to_action
from app.dynamics import drone_params as P
from app.dynamics.acro_style_methods import (MAX_BODY_RATE, RATE_ZETA, TORQUE_MARGIN, RateController,
                                             acro_timestamp_update, rate_kp_from_plant)
from app.dynamics.angle_style_methods import net_combining_torque, timestamp_update
from app.dynamics.drone import (Quaternion, QuadState, Vector3D, create_quad_config)

G = 9.81
GAINS_JSON = Path(__file__).resolve().parents[1] / "control" / "best_pid_gains_per_dist.json"


# --------------------------------------------------------------------------------------------- parameters

@dataclass
class SimConfig:
    # ---- airframe / plant (drone_params.py) ----
    mass: float = P.MASS
    inertia: tuple = tuple(P.INERTIA)
    arm_length: float = P.ARM_LENGTH
    drag_coeff: float = P.DRAG_COEFF
    max_rpm: float = P.MAX_RPM
    motor_tau: float = P.MOTOR_TAU
    physics_dt: float = 1.0 / 240.0          # base_drone_env_isaac.PHYSICS_DT
    decimation: int = 4                      # physics ticks per policy step (60 Hz)
    # ---- angle-style action bounds (the LOS label is clamped to these) ----
    thrust_up_factor: float = P.THRUST_UP_FACTOR
    max_torque_rp: float = P.MAX_TORQUE_ROLL_PITCH
    max_torque_yaw: float = P.MAX_TORQUE_YAW
    # ---- acro plant: rate loop and action space ----
    rate_zeta: float = RATE_ZETA
    rate_kp: float | None = None             # None -> rate_kp_from_plant(motor_tau, rate_zeta)
    rate_ki: float = 0.0
    torque_margin: float = TORQUE_MARGIN     # rate-loop torque limit = margin * hover torque authority
    max_rate_rp: float = MAX_BODY_RATE[0]    # |p|,|q| cap of the acro action space, rad/s
    max_rate_yaw: float = MAX_BODY_RATE[2]
    # ---- acro_ideal (diagnostic): limits lifted ----
    ideal_torque_margin: float = 1.0
    ideal_max_rate: float = 50.0
    # ---- LOS teacher (trainer flags --max-speed, --los-stop-speed, --teacher-cruise-gain, --los-max-tilt-deg, --teacher-accel) ----
    max_speed: float = 25.0
    stop_speed: float = 25.0                 # >= max_speed: never brakes (the trainer commands used 35/35 and 25/25)
    cruise_gain: float = 4.0
    max_tilt_deg: float = 45.0
    accel: float | None = None               # None -> 0.5 * g * tan(max_tilt)
    gains_key_dist: float | None = None      # None -> smallest key of best_pid_gains_per_dist.json (what make_los_label uses)
    kp_att: float | None = None              # override the JSON values
    kd_att: float | None = None
    kp_yaw: float | None = None
    kd_yaw: float | None = None
    yaw_des: float = 0.0
    # ---- scenario ----
    start_dist: float = 40.0                 # m, drone-to-target distance at the handoff (trainer: --distance-low/high)
    start_speed: float = 25.0                # m/s at the handoff
    vel_offset_deg: float = 15.0             # angle between the initial velocity and the line of sight
    vel_offset_az_deg: float = 90.0          # direction of that offset around the LOS (90 = sideways, 0 = up)
    approach_az_deg: float = 0.0             # azimuth of the start point around the target
    approach_el_deg: float = 0.0             # elevation of the start point above the target
    target_z: float = 128.0
    hit_radius: float = 1.0
    max_time_s: float = 20.0
    tilt_limit_deg: float = 180.0            # end the run when the thrust axis is further than this from up (180 = off)
    wind: tuple = (0.0, 0.0, 0.0)
    # ---- reporting ----
    out: str = "runs/sim_compare"
    log_every: int = 20                      # print a calculation line every N policy steps (0 = none)
    no_plot: bool = False
    drones: tuple = ("angle", "acro_teacher", "acro_ideal")
    # ---- sweep (--sweep): grid over handoff distance x handoff speed x velocity offset x offset direction ----
    sweep: bool = False
    sweep_dist: tuple = (15.0, 30.0, 50.0)   # handoff distances, m
    sweep_speed_frac: tuple = (0.25, 0.6, 1.0)   # handoff speed as a fraction of --max-speed (trainer buckets: 0.1-0.25-0.75-1.0)
    sweep_offset: tuple = (0.0, 20.0, 40.0, 60.0)    # deg between the velocity and the line of sight
    sweep_az: tuple = (0.0, 90.0, 180.0, 270.0)      # direction of that offset around the LOS
    straight_s: float = 6.0                  # "straight" = hit within this many seconds ...
    path_ratio_max: float = 1.6              # ... and a flown path no longer than this x the start distance


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(SimConfig):
        flag = "--" + f.name.replace("_", "-")
        d = f.default
        if f.name == "drones":
            p.add_argument(flag, nargs="+", default=None, choices=["angle", "acro_teacher", "acro_ideal"], help=f"default {d}")
        elif isinstance(d, tuple):
            p.add_argument(flag, type=float, nargs="+", default=None, help=f"default {d}")
        elif f.name in ("no_plot", "sweep"):
            p.add_argument(flag, action="store_true", default=None)
        elif isinstance(d, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(d, int):
            p.add_argument(flag, type=int, default=None, help=f"default {d}")
        elif isinstance(d, str):
            p.add_argument(flag, type=str, default=None, help=f"default {d}")
        else:
            p.add_argument(flag, type=float, default=None, help=f"default {d}")
    return p


def config_from_args(ns) -> SimConfig:
    cfg = SimConfig()
    for f in fields(SimConfig):
        v = getattr(ns, f.name, None)
        if v is not None:
            setattr(cfg, f.name, tuple(v) if isinstance(v, list) else v)
    return cfg


# --------------------------------------------------------------------------------------------- helpers

def quad_config(cfg: SimConfig):
    return create_quad_config(mass=cfg.mass, inertia=tuple(cfg.inertia), arm_length=cfg.arm_length, drag_coeff=cfg.drag_coeff,
                              max_rpm=cfg.max_rpm, motor_tau=cfg.motor_tau)


def angle_bounds(cfg: SimConfig):
    h = cfg.mass * G
    t, y = cfg.max_torque_rp, cfg.max_torque_yaw
    return np.array([-h, -t, -t, -y]), np.array([cfg.thrust_up_factor * h, t, t, y])


def euler(state: QuadState):
    q = state.orientation
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")      # roll, pitch, yaw (same as base_drone_env.py)


def attitude_gains(cfg: SimConfig) -> dict:
    with open(GAINS_JSON) as f:
        table = json.load(f)
    key = str(int(cfg.gains_key_dist)) if cfg.gains_key_dist is not None else min(table, key=lambda k: float(k))
    g = table[key]
    out = {"kp_att": g["kp_att"], "kd_att": g["kd_att"], "kp_yaw": g["kp_yaw"], "kd_yaw": g["kd_yaw"], "key": key}
    for n in ("kp_att", "kd_att", "kp_yaw", "kd_yaw"):
        if getattr(cfg, n) is not None:
            out[n] = getattr(cfg, n)
    return out


def initial_state(cfg: SimConfig, hover_rpm):
    az, el = math.radians(cfg.approach_az_deg), math.radians(cfg.approach_el_deg)
    target = np.array([0.0, 0.0, cfg.target_z])
    u_start = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = target + cfg.start_dist * u_start
    los = -u_start                                                   # unit vector drone -> target
    # build two vectors perpendicular to the LOS, tilt the initial velocity away from it by vel_offset_deg
    ref = np.array([0.0, 0.0, 1.0]) if abs(los[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(los, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, los)                                           # = up for a level LOS, so az 0 = up, 90 = +e1 side, 180 = down
    phi, off = math.radians(cfg.vel_offset_az_deg), math.radians(cfg.vel_offset_deg)
    perp = math.cos(phi) * e2 + math.sin(phi) * e1
    vel = cfg.start_speed * (math.cos(off) * los + math.sin(off) * perp)
    state = QuadState(position=Vector3D(*pos), velocity=Vector3D(*vel), orientation=Quaternion(0, 0, 0, 1),
                      angular_velocity=Vector3D(0, 0, 0), rotor_rpm=list(hover_rpm))
    return state, target


# --------------------------------------------------------------------------------------------- one flight

def fly(kind: str, cfg: SimConfig) -> dict:
    """One flight. Returns {"rows": [dict per policy step], "outcome": str, ...}."""
    qc = quad_config(cfg)
    dt, dec = cfg.physics_dt, cfg.decimation
    step_dt = dt * dec
    hover = cfg.mass * G
    a_low, a_high = angle_bounds(cfg)
    gains = attitude_gains(cfg)
    max_tilt = math.radians(cfg.max_tilt_deg)
    accel = cfg.accel if cfg.accel is not None else 0.5 * G * math.tan(max_tilt)
    inertia = np.array(cfg.inertia, dtype=np.float64)

    acro = kind != "angle"
    ideal = kind == "acro_ideal"
    kp = cfg.rate_kp if cfg.rate_kp is not None else rate_kp_from_plant(cfg.motor_tau, cfg.rate_zeta)
    margin = cfg.ideal_torque_margin if ideal else cfg.torque_margin
    rate_cap = np.array([cfg.max_rate_rp, cfg.max_rate_rp, cfg.max_rate_yaw]) if not ideal else np.full(3, cfg.ideal_max_rate)
    ctrl = RateController(qc, kp=kp, ki=cfg.rate_ki, torque_margin=margin) if acro else None

    from app.dynamics.angle_style_methods import mixer_inversion
    hover_rpm = mixer_inversion(qc, [hover, 0.0, 0.0, 0.0])
    state, target = initial_state(cfg, hover_rpm)
    wind = list(cfg.wind)

    rows, outcome = [], "timeout"
    n_steps = int(round(cfg.max_time_s / step_dt))
    for k in range(n_steps):
        pos = np.array([state.position.x, state.position.y, state.position.z])
        vel = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        w = np.array([state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z])
        roll, pitch, yaw = euler(state)
        rel = target - pos
        dist = float(np.linalg.norm(rel))
        los_u = rel / max(dist, 1e-6)
        closing = float(vel @ los_u)
        v_perp = vel - closing * los_u
        tilt = math.degrees(math.acos(float(np.clip(Rotation.from_quat(
            [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w]).apply([0, 0, 1])[2], -1, 1))))

        if dist <= cfg.hit_radius:
            outcome = "hit"
            break
        if tilt > cfg.tilt_limit_deg:
            outcome = "tilt"
            break

        # ---------------- the LOS teacher (make_los_label), line for line ----------------
        # trainer: sqrt(2*accel*dist).clamp(max=max_speed).clamp_min(min(stop_speed, max_speed))
        v_cmd = max(min(math.sqrt(2 * accel * dist), cfg.max_speed), min(cfg.stop_speed, cfg.max_speed))
        v_des = los_u * v_cmd
        raw, des_roll, des_pitch = velocity_to_action(
            v_des[None], vel[None], w[None], np.array([roll]), np.array([pitch]), np.array([yaw]), np.array([cfg.yaw_des]),
            hover_thrust=hover, kp_att=gains["kp_att"], kd_att=gains["kd_att"], kp_yaw=gains["kp_yaw"], kd_yaw=gains["kd_yaw"],
            max_tilt=max_tilt, cruise_gain=cfg.cruise_gain)
        raw = raw[0]
        a = np.clip(raw, a_low, a_high)                          # angle-style action [thrust_delta, tau_x, tau_y, tau_z]
        accel_cmd = cfg.cruise_gain * (v_des - vel)

        row = {"t": k * step_dt, "x": pos[0], "y": pos[1], "z": pos[2], "vx": vel[0], "vy": vel[1], "vz": vel[2],
               "speed": float(np.linalg.norm(vel)), "dist": dist, "closing": closing, "lateral_speed": float(np.linalg.norm(v_perp)),
               "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw), "tilt_deg": tilt,
               "des_roll_deg": math.degrees(float(des_roll[0])), "des_pitch_deg": math.degrees(float(des_pitch[0])),
               "p": w[0], "q": w[1], "r": w[2],
               "v_cmd": v_cmd, "v_des_x": v_des[0], "v_des_y": v_des[1], "v_des_z": v_des[2],
               "accel_cmd_x": accel_cmd[0], "accel_cmd_y": accel_cmd[1], "accel_cmd_z": accel_cmd[2],
               "thrust_delta_raw": raw[0], "tau_x_raw": raw[1], "tau_y_raw": raw[2], "tau_z_raw": raw[3],
               "thrust_delta": a[0], "tau_x_req": a[1], "tau_y_req": a[2], "tau_z_req": a[3],
               "tau_clipped_by_angle_bounds": bool(np.any(np.abs(raw[1:] - a[1:]) > 1e-9))}

        if acro:
            # angle_to_acro: cmd = w + tau / (I * kp), then clamp to the acro action bounds
            cmd_raw = w + a[1:4] / (inertia * kp)
            cmd = np.clip(cmd_raw, -rate_cap, rate_cap)
            tau_first = inertia * kp * (cmd - w)                                    # what the rate loop asks on the first substep
            tau_lim = ctrl.torque_limit
            row.update({"rate_cmd_raw_p": cmd_raw[0], "rate_cmd_raw_q": cmd_raw[1], "rate_cmd_raw_r": cmd_raw[2],
                        "rate_cmd_p": cmd[0], "rate_cmd_q": cmd[1], "rate_cmd_r": cmd[2],
                        "rate_cap_hit": bool(np.any(np.abs(cmd_raw) > rate_cap + 1e-9)),
                        "tau_x_rateloop": tau_first[0], "tau_y_rateloop": tau_first[1], "tau_z_rateloop": tau_first[2],
                        "torque_limit_x": tau_lim[0], "torque_limit_y": tau_lim[1], "torque_limit_z": tau_lim[2],
                        "torque_limit_hit": bool(np.any(np.abs(tau_first) > tau_lim + 1e-9))})
            action = [hover + a[0], *cmd]
        else:
            action = [hover + a[0], a[1], a[2], a[3]]

        for _ in range(dec):                                         # action held for the whole policy step
            state = acro_timestamp_update(state, qc, ctrl, action, wind, dt) if acro else timestamp_update(state, qc, action, wind, dt)

        torque_made = net_combining_torque(qc, list(state.rotor_rpm))     # N m made by the rotors after the last substep
        row.update({"tau_x_made": torque_made[0], "tau_y_made": torque_made[1], "tau_z_made": torque_made[2],
                    "rpm1": state.rotor_rpm[0], "rpm2": state.rotor_rpm[1], "rpm3": state.rotor_rpm[2], "rpm4": state.rotor_rpm[3]})
        rows.append(row)

        if not np.all(np.isfinite([state.position.x, state.position.y, state.position.z])):
            outcome = "nan"
            break

        if cfg.log_every and k % cfg.log_every == 0:
            msg = (f"[{kind:12s}] t={row['t']:5.2f} dist={dist:7.2f} speed={row['speed']:6.2f} closing={closing:+6.2f} "
                   f"lat={row['lateral_speed']:5.2f} | v_cmd={v_cmd:5.1f} acc_cmd=({accel_cmd[0]:+6.1f},{accel_cmd[1]:+6.1f},{accel_cmd[2]:+6.1f}) "
                   f"| des r/p={row['des_roll_deg']:+6.1f}/{row['des_pitch_deg']:+6.1f} act r/p={row['roll_deg']:+6.1f}/{row['pitch_deg']:+6.1f} "
                   f"rates={w[0]:+5.2f},{w[1]:+5.2f},{w[2]:+5.2f} | tau_req={a[1]:+5.2f},{a[2]:+5.2f},{a[3]:+5.2f}")
            if acro:
                msg += (f" | rate_cmd raw={cmd_raw[0]:+5.2f},{cmd_raw[1]:+5.2f} used={cmd[0]:+5.2f},{cmd[1]:+5.2f} "
                        f"tau_rateloop={tau_first[0]:+5.2f},{tau_first[1]:+5.2f} lim={tau_lim[0]:.2f} "
                        f"{'RATE-CAP ' if row['rate_cap_hit'] else ''}{'TORQUE-LIMIT' if row['torque_limit_hit'] else ''}")
            print(msg)

    return {"kind": kind, "rows": rows, "outcome": outcome, "step_dt": step_dt}


# --------------------------------------------------------------------------------------------- reporting

def col(res, name):
    return np.array([r[name] for r in res["rows"]], dtype=np.float64)


def summarize(res) -> dict:
    rows = res["rows"]
    if not rows:
        return {"kind": res["kind"], "outcome": res["outcome"]}
    dist = col(res, "dist")
    s = {"kind": res["kind"], "outcome": res["outcome"], "time_s": rows[-1]["t"], "min_dist": float(dist.min()),
         "final_dist": float(dist[-1]), "max_tilt_deg": float(col(res, "tilt_deg").max()),
         "max_abs_roll_pitch_rate": float(max(np.abs(col(res, "p")).max(), np.abs(col(res, "q")).max())),
         "max_lateral_speed": float(col(res, "lateral_speed").max())}
    xyz = np.stack([col(res, "x"), col(res, "y"), col(res, "z")], axis=1)
    s["path_ratio"] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() / max(dist[0], 1e-6))   # 1.0 = dead straight
    for ax in ("x", "y"):
        req, made = col(res, f"tau_{ax}_req"), col(res, f"tau_{ax}_made")
        big = np.abs(req) > 0.05
        s[f"tau_{ax}_max_req"] = float(np.abs(req).max())
        s[f"tau_{ax}_max_made"] = float(np.abs(made).max())
        s[f"tau_{ax}_made_over_req"] = float(np.sum(np.abs(made[big])) / max(np.sum(np.abs(req[big])), 1e-9)) if big.any() else float("nan")
    if "rate_cap_hit" in rows[0]:
        s["frac_rate_cap_hit"] = float(np.mean([r["rate_cap_hit"] for r in rows]))
        s["frac_torque_limit_hit"] = float(np.mean([r["torque_limit_hit"] for r in rows]))
    return s


def write_outputs(results, cfg: SimConfig):
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "params.json", "w") as f:
        json.dump({**asdict(cfg), "attitude_gains": attitude_gains(cfg),
                   "rate_kp_used": cfg.rate_kp if cfg.rate_kp is not None else rate_kp_from_plant(cfg.motor_tau, cfg.rate_zeta)},
                  f, indent=2, default=str)
    lines = []
    for res in results:
        if res["rows"]:
            with open(out / f"{res['kind']}.csv", "w", newline="") as f:
                keys = list(res["rows"][0].keys())
                wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                wr.writeheader()
                wr.writerows(res["rows"])
        s = summarize(res)
        lines.append("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print("\n=== summary ===\n" + text)
    if not cfg.no_plot:
        plot(results, cfg, out / "sim_compare.png")
        print(f"graphs: {out / 'sim_compare.png'}\ncsv/params/summary in {out}")


def plot(results, cfg: SimConfig, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"angle": "tab:blue", "acro_teacher": "tab:red", "acro_ideal": "tab:green"}
    fig, ax = plt.subplots(4, 3, figsize=(20, 16))
    ax = ax.ravel()

    def line(i, name, title, ylabel, hlines=()):
        for res in results:
            if res["rows"]:
                ax[i].plot(col(res, "t"), col(res, name), color=colors[res["kind"]], label=res["kind"])
        for h in hlines:
            ax[i].axhline(h, color="k", ls=":", lw=0.8)
        ax[i].set_title(title)
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel("t [s]")
        ax[i].grid(alpha=0.3)

    line(0, "dist", "distance to target", "m", (cfg.hit_radius,))
    line(1, "speed", "speed", "m/s")
    line(2, "closing", "closing speed (+ = toward target)", "m/s", (0,))
    line(3, "lateral_speed", "velocity component off the line of sight", "m/s", (0,))
    line(4, "tilt_deg", "tilt (thrust axis vs up)", "deg", (cfg.max_tilt_deg,))
    # roll/pitch vs the desired angle
    for res in results:
        if res["rows"]:
            ax[5].plot(col(res, "t"), col(res, "roll_deg"), color=colors[res["kind"]], label=f"{res['kind']} roll")
            ax[5].plot(col(res, "t"), col(res, "pitch_deg"), color=colors[res["kind"]], ls="--", label=f"{res['kind']} pitch")
    r0 = next((r for r in results if r["rows"]), None)
    if r0:
        ax[5].plot(col(r0, "t"), col(r0, "des_roll_deg"), color="k", lw=0.8, alpha=0.6, label="desired roll (angle drone)")
        ax[5].plot(col(r0, "t"), col(r0, "des_pitch_deg"), color="gray", lw=0.8, alpha=0.6, ls="--", label="desired pitch (angle drone)")
    ax[5].set_title("roll / pitch vs desired")
    ax[5].set_ylabel("deg")
    ax[5].set_xlabel("t [s]")
    ax[5].legend(fontsize=6)
    ax[5].grid(alpha=0.3)
    # body rates with the acro cap
    for res in results:
        if res["rows"]:
            ax[6].plot(col(res, "t"), col(res, "p"), color=colors[res["kind"]], label=f"{res['kind']} p")
            ax[6].plot(col(res, "t"), col(res, "q"), color=colors[res["kind"]], ls="--", label=f"{res['kind']} q")
    for h in (cfg.max_rate_rp, -cfg.max_rate_rp):
        ax[6].axhline(h, color="k", ls=":", lw=0.8)
    ax[6].set_title(f"roll/pitch rate (dotted = acro cap +-{cfg.max_rate_rp} rad/s)")
    ax[6].set_ylabel("rad/s")
    ax[6].set_xlabel("t [s]")
    ax[6].legend(fontsize=6)
    ax[6].grid(alpha=0.3)
    # torque requested by the teacher vs made by the rotors (pitch axis and roll axis)
    for i, axn in ((7, "x"), (8, "y")):
        for res in results:
            if res["rows"]:
                ax[i].plot(col(res, "t"), col(res, f"tau_{axn}_req"), color=colors[res["kind"]], ls=":", label=f"{res['kind']} requested")
                ax[i].plot(col(res, "t"), col(res, f"tau_{axn}_made"), color=colors[res["kind"]], label=f"{res['kind']} made")
        for h in (cfg.max_torque_rp, -cfg.max_torque_rp):
            ax[i].axhline(h, color="k", ls=":", lw=0.8)
        ax[i].set_title(f"{'roll' if axn == 'x' else 'pitch'} torque: teacher request (dotted) vs rotors made (solid)")
        ax[i].set_ylabel("N m")
        ax[i].set_xlabel("t [s]")
        ax[i].legend(fontsize=6)
        ax[i].grid(alpha=0.3)
    # trajectories
    for res in results:
        if res["rows"]:
            ax[9].plot(col(res, "x"), col(res, "y"), color=colors[res["kind"]], label=res["kind"])
            ax[10].plot(col(res, "x"), col(res, "z"), color=colors[res["kind"]], label=res["kind"])
    for i, (xl, yl, t) in ((9, ("x [m]", "y [m]", "top view")), (10, ("x [m]", "z [m]", "side view"))):
        ax[i].plot([0], [0 if i == 9 else cfg.target_z], "k*", ms=12)
        ax[i].set_title(t)
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel(yl)
        ax[i].set_aspect("equal", adjustable="datalim")
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=7)
    # acro limits: how far the rate loop's torque is from the teacher's
    for res in results:
        if res["rows"] and "tau_x_rateloop" in res["rows"][0]:
            ax[11].plot(col(res, "t"), col(res, "tau_x_rateloop") - col(res, "tau_x_req"), color=colors[res["kind"]], label=f"{res['kind']} roll")
            ax[11].plot(col(res, "t"), col(res, "tau_y_rateloop") - col(res, "tau_y_req"), color=colors[res["kind"]], ls="--", label=f"{res['kind']} pitch")
    ax[11].set_title("conversion loss: rate-loop torque (first substep) - teacher torque")
    ax[11].set_ylabel("N m")
    ax[11].set_xlabel("t [s]")
    ax[11].legend(fontsize=6)
    ax[11].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"start {cfg.start_dist:.0f} m, {cfg.start_speed:.0f} m/s, velocity {cfg.vel_offset_deg:.0f} deg off the LOS; "
                 f"teacher max {cfg.max_speed:.0f} / stop {cfg.stop_speed:.1f} m/s, tilt cap {cfg.max_tilt_deg:.0f} deg, gain {cfg.cruise_gain}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- sweep

def sweep(cfg: SimConfig):
    """Fly every drone over the grid; print an aggregate per drone and write sweep.csv (one row per case)."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    quiet = replace(cfg, log_every=0, no_plot=True)
    records = []
    for d in cfg.sweep_dist:
        for fr in cfg.sweep_speed_frac:
            for off in cfg.sweep_offset:
                for az in (cfg.sweep_az if off > 0 else (0.0,)):
                    c = replace(quiet, start_dist=d, start_speed=fr * cfg.max_speed, vel_offset_deg=off, vel_offset_az_deg=az)
                    for kind in cfg.drones:
                        res = fly(kind, c)
                        s = summarize(res)
                        hit = res["outcome"] == "hit"
                        t = s.get("time_s", float("nan"))
                        straight = hit and t <= cfg.straight_s and s["path_ratio"] <= cfg.path_ratio_max
                        records.append({"drone": kind, "dist": d, "speed": fr * cfg.max_speed, "offset_deg": off, "az_deg": az,
                                        "outcome": res["outcome"], "time_s": t, "path_ratio": s.get("path_ratio", float("nan")),
                                        "min_dist": s.get("min_dist", float("nan")), "max_tilt_deg": s.get("max_tilt_deg", float("nan")),
                                        "hit": hit, "straight": bool(straight)})
    with open(out / "sweep.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)

    def rate(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    lines = [f"sweep: teacher max {cfg.max_speed:g} / stop {cfg.stop_speed:g} m/s, tilt cap {cfg.max_tilt_deg:g} deg, gain {cfg.cruise_gain:g}; "
             f"'straight' = hit within {cfg.straight_s:g} s and path <= {cfg.path_ratio_max:g} x start distance"]
    for kind in cfg.drones:
        rs = [r for r in records if r["drone"] == kind]
        times = np.array([r["time_s"] for r in rs if r["hit"]] or [float("nan")])
        lines.append(f"\n[{kind}] cases={len(rs)} hit={rate(rs, 'hit'):.3f} straight={rate(rs, 'straight'):.3f} "
                     f"hit_time median={np.nanmedian(times):.2f}s p90={np.nanpercentile(times, 90):.2f}s")
        for label, key, vals in (("handoff offset (deg)", "offset_deg", cfg.sweep_offset), ("handoff distance (m)", "dist", cfg.sweep_dist),
                                 ("handoff speed (m/s)", "speed", sorted({r["speed"] for r in rs}))):
            lines.append(f"   by {label}: " + "  ".join(
                f"{v:g}: hit {rate([r for r in rs if abs(r[key] - v) < 1e-6], 'hit'):.2f} straight {rate([r for r in rs if abs(r[key] - v) < 1e-6], 'straight'):.2f}"
                for v in vals))
        bad = sorted([r for r in rs if not r["straight"]], key=lambda r: (r["hit"], -r["time_s"]))[:5]
        for r in bad:
            lines.append(f"   not straight: dist={r['dist']:g} speed={r['speed']:g} offset={r['offset_deg']:g}@{r['az_deg']:g} -> {r['outcome']} "
                         f"t={r['time_s']:.1f}s path_ratio={r['path_ratio']:.2f} min_dist={r['min_dist']:.2f}")
    text = "\n".join(lines)
    (out / "sweep_summary.txt").write_text(text + "\n")
    print(text + f"\n\nper-case rows: {out / 'sweep.csv'}")
    return records


def run(cfg: SimConfig):
    results = [fly(kind, cfg) for kind in cfg.drones]
    write_outputs(results, cfg)
    return results


# --------------------------------------------------------------------------------------------- pytest

def _quiet(**kw) -> SimConfig:
    cfg = SimConfig(log_every=0, no_plot=True, max_time_s=12.0, **kw)
    return cfg


def test_angle_drone_hits_with_los_teacher():
    res = fly("angle", _quiet())
    assert res["outcome"] == "hit", summarize(res)


def test_acro_with_lifted_limits_matches_angle_closely():
    """If the conversion were exact, lifting the rate cap and torque limit should leave acro close to the angle drone."""
    a, b = fly("angle", _quiet()), fly("acro_ideal", _quiet())
    assert b["outcome"] == "hit", summarize(b)
    assert abs(a["rows"][-1]["t"] - b["rows"][-1]["t"]) < 1.0


def test_all_logged_values_finite():
    for kind in ("angle", "acro_teacher", "acro_ideal"):
        res = fly(kind, _quiet())
        for key in res["rows"][0]:
            v = col(res, key)
            assert np.all(np.isfinite(v)), (kind, key)


def test_sweep_runs_and_angle_hits_when_aligned(tmp_path):
    cfg = SimConfig(drones=("angle",), sweep_dist=(40.0,), sweep_speed_frac=(1.0,), sweep_offset=(0.0,), out=str(tmp_path))
    rec = sweep(cfg)
    assert len(rec) == 1 and rec[0]["hit"]


if __name__ == "__main__":
    _cfg = config_from_args(build_parser().parse_args())
    if _cfg.sweep:
        sweep(_cfg)
    else:
        run(_cfg)
