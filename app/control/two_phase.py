"""Two-phase controller: PID / carrot cruise flies the midcourse, the RL policy takes over at switch_dist. See docs.md."""
from __future__ import annotations

import json

import numpy as np
import torch

from app.control.pid import PIDController
from app.environmental.base_drone_env import POS_SCALE, _symlog_scale
from app.control.tune_pid import compute_gains_for_distance

DEFAULT_GAINS_PATH = "app/control/best_pid_gains_per_dist.json"
DEFAULT_SWITCH_DIST = 20.0
MIN_GAIN_DIST = 1.0  # floor for the gain-solve distance: log(dist/HIT_THRESHOLD) breaks below it


def load_gains_by_dist(path: str = DEFAULT_GAINS_PATH) -> dict:
    """Load best_pid_gains_per_dist.json (only for nearest-neighbour gains instead of analytic ones)."""
    with open(path) as f:
        return json.load(f)


def _nearest_gains(gains_by_dist: dict, dist: float):
    key = min(gains_by_dist.keys(), key=lambda k: abs(float(k) - dist))
    return key, gains_by_dist[key]


def gains_for_distance(dist: float):
    """(label, gains) solved analytically for the exact distance; works beyond the 250 m ladder."""
    solved_dist = max(dist, MIN_GAIN_DIST)
    gains, diag = compute_gains_for_distance(solved_dist)
    label = f"analytic@{solved_dist:.1f}m (wn_pos={diag['wn_pos']:.3f}, sat_ratio={diag['saturation_ratio']:.2f})"
    return label, gains


class CarrotController:
    """Numpy port of rtl_train_isaac's carrot controller: trapezoid speed profile along the spawn->target line."""

    def __init__(self, gains_by_dist: dict, cruise_speed: float, handoff_speed: float, cruise_gain: float = 1.0):
        self.table = sorted(((float(k), v) for k, v in gains_by_dist.items()), key=lambda kv: kv[0])
        self.cruise_speed = cruise_speed
        self.handoff_speed = handoff_speed
        self.cruise_gain = cruise_gain
        base = self.table[0][1]
        self.max_tilt = base.get("max_tilt_rad", 0.3)
        self.accel = 0.5 * 9.81 * np.tan(self.max_tilt)   # same default as training: half the tilt-limited accel
        self.pid = PIDController(**base)                   # only for its action bounds / yaw wrapping helpers

    def reset(self, drone_state, target_pos, switch_dist):
        self.spawn = np.array([drone_state.position.x, drone_state.position.y, drone_state.position.z])
        to_t = np.asarray(target_pos, dtype=np.float64) - self.spawn
        total = float(np.linalg.norm(to_t))
        self.unit = to_t / max(total, 1e-6)
        self.prefix = max(total - switch_dist, 0.5)
        vh = min(self.handoff_speed, self.cruise_speed)
        self.vh = vh
        a = self.accel
        self.vpeak = min(self.cruise_speed, float(np.sqrt(max(a * self.prefix + 0.5 * vh * vh, 0.0))))
        self.d_up = self.vpeak ** 2 / (2 * a)
        d_down = max(self.vpeak ** 2 - vh ** 2, 0.0) / (2 * a)
        self.d_cruise = max(self.prefix - self.d_up - d_down, 0.0)

    def _gains(self, remaining):
        return min(self.table, key=lambda kv: abs(kv[0] - remaining))[1]

    def compute_action(self, drone_state, target_pos, target_yaw):
        from scipy.spatial.transform import Rotation
        from app.control.pid import wrap_angle
        pos = np.array([drone_state.position.x, drone_state.position.y, drone_state.position.z])
        vel = np.array([drone_state.velocity.x, drone_state.velocity.y, drone_state.velocity.z])
        ang = np.array([drone_state.angular_velocity.x, drone_state.angular_velocity.y, drone_state.angular_velocity.z])
        o = drone_state.orientation
        roll, pitch, yaw = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler("xyz")
        g = self._gains(float(np.linalg.norm(np.asarray(target_pos) - pos)))

        s = float(np.clip(np.dot(pos - self.spawn, self.unit), 0.0, self.prefix))
        if s < self.d_up:
            v_cmd = np.sqrt(max(2 * self.accel * s, 0.0))
        elif s < self.d_up + self.d_cruise:
            v_cmd = self.vpeak
        else:
            v_cmd = np.sqrt(max(self.vpeak ** 2 - 2 * self.accel * (s - self.d_up - self.d_cruise), self.vh ** 2))
        v_cmd = max(v_cmd, 0.1)

        accel_xy = self.cruise_gain * (self.unit[:2] * v_cmd - vel[:2])
        thrust_delta = g["kp_pos"] * (target_pos[2] - pos[2]) - g["kd_pos"] * vel[2]
        c, sn = np.cos(yaw), np.sin(yaw)
        ax, ay = accel_xy[0] * c + accel_xy[1] * sn, -accel_xy[0] * sn + accel_xy[1] * c
        des_roll = np.clip(-ay / 9.81, -self.max_tilt, self.max_tilt)
        des_pitch = np.clip(ax / 9.81, -self.max_tilt, self.max_tilt)
        roll_t = g["kp_att"] * (des_roll - roll) - g["kd_att"] * ang[0]
        pitch_t = g["kp_att"] * (des_pitch - pitch) - g["kd_att"] * ang[1]
        yaw_t = g["kp_yaw"] * wrap_angle(target_yaw - yaw) - g["kd_yaw"] * ang[2]
        return np.clip(np.array([thrust_delta, roll_t, pitch_t, yaw_t]), self.pid.action_low, self.pid.action_high)


class TwoPhaseAgent:
    """Stateful get_action(obs, env) — call reset(env) right after env.reset()."""

    def __init__(self, model, gains_by_dist: dict | None = None, switch_dist: float = DEFAULT_SWITCH_DIST,
                 device: str = "cpu", log=None, cruise_speed: float | None = 18.0,
                 handoff_speed: float | None = None, recentre_z: float | None = None):
        """See docs.md "Two-phase agent" for gains_by_dist, cruise_speed, handoff_speed and recentre_z."""
        self.model = model
        self.gains_by_dist = gains_by_dist
        self.cruise_speed = cruise_speed
        self.handoff_speed = handoff_speed if handoff_speed is not None else (cruise_speed or 0.0) / 2
        self.carrot: CarrotController | None = None
        self.switch_dist = switch_dist
        self.device = device
        self.log = log or (lambda msg: None)

        self.pid: PIDController | None = None
        self.phase: str | None = None
        self.gain_key = None
        self.switch_step: int | None = None
        self._step = 0
        self.recentre_z = recentre_z
        self._pos_offset: np.ndarray | None = None

    def reset(self, env):
        dist = env.prev_distance
        if self.gains_by_dist is not None:
            self.gain_key, gains = _nearest_gains(self.gains_by_dist, dist)
        else:
            self.gain_key, gains = gains_for_distance(dist)
        self.pid = PIDController(**gains)
        self.pid.reset()
        self._step = 0
        if self.cruise_speed:
            self.carrot = CarrotController(load_gains_by_dist(), self.cruise_speed, self.handoff_speed)
            self.carrot.reset(env.drone_state, env.target_pos, self.switch_dist)
            self.gain_key = f"carrot cruise {self.cruise_speed:g} -> {self.handoff_speed:g} m/s"
        # PID alone flies until switch_dist, then only the model acts (keep switch_dist inside its trained range)
        self.phase = "PID"
        self.switch_step = None
        self._pos_offset = None
        self.log(f"[PID]    start   dist={dist:8.2f}m  gains={self.gain_key}  target={env.target_pos.tolist()}")

    def get_action(self, obs, env):
        dist = env.prev_distance

        if self.phase == "PID" and dist <= self.switch_dist:
            self.phase = "RL"
            self.switch_step = self._step
            # real handoff speed, for comparing against training's spawn speed range
            v = env.drone_state.velocity
            speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
            self.log(f"[SWITCH] PID -> pure RL  dist={dist:6.2f}m  threshold={self.switch_dist}m  "
                     f"speed={speed:5.2f}m/s  step={self._step}")
            if self.carrot is not None:
                # snap velocity to the trained handoff speed along the line to target, then rebuild obs
                p = env.drone_state.position
                to_t = np.asarray(env.target_pos, dtype=np.float64) - np.array([p.x, p.y, p.z])
                unit = to_t / max(float(np.linalg.norm(to_t)), 1e-6)
                vx, vy, vz = (float(c) * self.handoff_speed for c in unit)
                env.drone_state.velocity = type(v)(vx, vy, vz)
                self.log(f"[SNAP]   velocity set to handoff speed {self.handoff_speed:.2f}m/s along the line to target")
                obs = env._get_obs()
            if self.recentre_z is not None and env.position_mode == "full":
                p = env.drone_state.position
                self._pos_offset = np.array([p.x, p.y, max(0.0, p.z - self.recentre_z)])
                self.log(f"[RECENTRE] policy position input = pos - {np.round(self._pos_offset, 2).tolist()}")

        self._step += 1

        if self.phase == "PID" and self.carrot is not None:
            return self.carrot.compute_action(env.drone_state, env.target_pos, env.target_yaw)
        if self.phase == "PID":
            return self.pid.compute_action(env.drone_state, env.target_pos, env.target_yaw, dt=env.dt)

        if self._pos_offset is not None:
            p = env.drone_state.position
            obs = np.array(obs, dtype=np.float32, copy=True)
            obs[0:3] = _symlog_scale(np.array([p.x, p.y, p.z]) - self._pos_offset, POS_SCALE)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = self.model.forward(obs_t)
            return self.model.scale_action(mean).squeeze(0).cpu().numpy()
