"""Per-frame forces, rotor thrusts and angles for the 3D player's side panel and vectors.

A DynRecorder rides along with a numpy episode: add() after every env step stores the few raw numbers that are cheap to copy
(rotor speeds, the applied action, velocity, body rates); build() then works out the derived physics for the frames the track
keeps, all vectorised. "wanted" values are what the commanded action asks the mixer for given the state at that frame:
a rotor wanting a NEGATIVE thrust is saturated (a rotor cannot push down), which is the clearest sign the drone is
asking for more than it can do.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from app.dynamics.drone_params import AIR_DENS, CROSS_SEC_AREA


def _angle(a, b):
    """Angle in degrees between row vectors a and b (NaN where either is ~zero)."""
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)
    out = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    out[(na < 1e-9) | (nb < 1e-9)] = np.nan
    return out


def _clean(arr, digits):
    """numpy -> nested lists rounded for JSON; NaN becomes None."""
    a = np.round(np.asarray(arr, dtype=np.float64), digits)
    if a.ndim == 1:
        return [None if np.isnan(v) else float(v) for v in a]
    return a.tolist()


class DynRecorder:
    def __init__(self, env):
        self.env = env
        self.rpm, self.action, self.vel, self.rate = [], [], [], []

    def add(self):
        s = self.env.drone_state
        self.rpm.append(list(s.rotor_rpm))
        self.action.append(np.array(self.env.last_raw_action, dtype=np.float64))
        self.vel.append([s.velocity.x, s.velocity.y, s.velocity.z])
        self.rate.append([s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z])

    def build(self, idx, positions, quats):
        """dict for the frames `idx` (the positions / quats lists are the full per-step ones the track was built from)."""
        env, cfg = self.env, self.env.config
        idx = np.asarray(idx)
        rpm, act = np.array(self.rpm)[idx], np.array(self.action)[idx]
        vel, rate = np.array(self.vel)[idx], np.array(self.rate)[idx]
        pos, quat = np.asarray(positions, dtype=np.float64)[idx], np.asarray(quats, dtype=np.float64)[idx]
        n = len(idx)

        kf = np.array([r.k_f for r in cfg.rotors])
        km = np.array([r.k_m for r in cfg.rotors])
        spin = np.array([r.spin_dir for r in cfg.rotors], dtype=np.float64)
        rx = np.array([r.position.x for r in cfg.rotors])
        ry = np.array([r.position.y for r in cfg.rotors])
        mass, weight = cfg.mass, cfg.mass * 9.81
        inertia = np.array(cfg.inertia)

        # mixer matrix exactly as angle_style_methods.mixer_inversion builds it
        mix = np.stack([kf, ry * kf, -rx * kf, km * spin])
        mix_inv = np.linalg.inv(mix)

        # what the action asks of the airframe: [thrust N, roll / pitch / yaw torque Nm]
        thrust_cmd = act[:, 0] + weight
        if env.drone_style == "acro":    # action[1:4] is a body-rate setpoint; the rate loop turns it into a torque
            rc = env.rate_controller
            torque_cmd = np.clip(inertia * rc.kp * (act[:, 1:4] - rate), -rc.torque_limit, rc.torque_limit)
        else:
            torque_cmd = act[:, 1:4]
        want_sq = np.column_stack([thrust_cmd, torque_cmd]) @ mix_inv.T      # per-rotor w^2 the mixer asks for (can be < 0)
        rotor_f_want = kf * want_sq                                           # N, negative = unreachable
        rotor_f = kf * rpm ** 2

        torque = np.column_stack([(ry * rotor_f).sum(1), (-rx * rotor_f).sum(1), (km * rpm ** 2 * spin).sum(1)])
        thrust = rotor_f.sum(1)

        rot = Rotation.from_quat(quat)
        z_body = rot.apply([0.0, 0.0, 1.0])
        thrust_w = z_body * thrust[:, None]
        speed = np.linalg.norm(vel, axis=1)
        drag = -vel / np.maximum(speed, 1e-9)[:, None] * (0.5 * AIR_DENS * speed ** 2 * cfg.drag_coeff * CROSS_SEC_AREA)[:, None]
        drag[speed < 0.01] = 0.0
        grav = np.tile([0.0, 0.0, -weight], (n, 1))
        f_net = thrust_w + drag + grav

        # angles between the drone and the target
        los = np.asarray(env.target_pos, dtype=np.float64)[None, :] - pos
        los_body = rot.inv().apply(los)
        _, _, yaw = rot.as_euler("xyz").T
        yaw_err = (env.target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        vel_los = _angle(vel, los)
        vel_los[speed < 0.3] = np.nan
        angles = {
            "thrust_los": _clean(_angle(z_body, los), 2),                       # thrust axis vs line to the target
            "vel_los": _clean(vel_los, 2),                                      # velocity vs line to the target
            "tilt": _clean(_angle(z_body, np.tile([0.0, 0.0, 1.0], (n, 1))), 2),   # thrust axis vs straight up
            "yaw_err": _clean(np.degrees(yaw_err), 2),
            "az": _clean(np.degrees(np.arctan2(los_body[:, 1], los_body[:, 0])), 2),    # target bearing in the body frame
            "el": _clean(np.degrees(np.arctan2(los_body[:, 2], np.hypot(los_body[:, 0], los_body[:, 1]))), 2),
        }

        return {
            "rotor_w": _clean(rpm, 1), "rotor_f": _clean(rotor_f, 3), "rotor_f_want": _clean(rotor_f_want, 3),
            "thrust": _clean(thrust, 3), "thrust_want": _clean(thrust_cmd, 3),
            "torque": _clean(torque, 4), "torque_want": _clean(torque_cmd, 4),
            "drag": _clean(drag, 3), "f_net": _clean(f_net, 3), "accel": _clean(f_net / mass, 3),
            "action": _clean(act, 4), "angles": angles,
        }

    def log_lines(self, label):
        """Per-rotor log block for the CURRENT state (the last buffered step): one line per rotor R1..R4 with the thrust it makes,
        the thrust the commands want from it, its speed, its share of hover, and whether the commands ask it to push down.
        Under it: the whole-drone numbers (thrust, torques made vs wanted, drag, net force, angles to the target)."""
        n_before = len(self.rpm)
        self.add()                                  # the live state, not the last buffered step
        n = n_before + 1
        pos, quats = np.zeros((n, 3)), np.zeros((n, 4))
        p, q = self.env.drone_state.position, self.env.drone_state.orientation
        pos[-1] = [p.x, p.y, p.z]
        quats[-1] = [q.x, q.y, q.z, q.w]
        d = self.build([n - 1], pos, quats)
        for buf in (self.rpm, self.action, self.vel, self.rate):
            buf.pop()
        hover_f = self.env.config.mass * 9.81 / 4
        names = ["R1", "R2", "R3", "R4"]
        lines = [f"{label}: rotor forces and commands (per rotor R1..R4, positions as the sim's mixer has them)"]
        for k in range(4):
            made, want = d["rotor_f"][0][k], d["rotor_f_want"][0][k]
            r = self.env.config.rotors[k]
            flag = "PUSH-DOWN ASKED, CANNOT (rotor stops at 0 N)" if want < 0 else "ok"
            lines.append(
                f"  {names[k]} at ({r.position.x:+.2f}, {r.position.y:+.2f}) m spin {r.spin_dir:+d}: "
                f"speed {d['rotor_w'][0][k]:.0f} (sim units) | force made {made:+.3f} N, wanted {want:+.3f} N "
                f"({100 * made / hover_f:.0f}% of hover) | {flag}")
        t, tw = d["thrust"][0], d["thrust_want"][0]
        lines.append(f"  total thrust made {t:.2f} N, wanted {tw:.2f} N (weight {self.env.config.mass * 9.81:.2f} N)")
        lines.append(f"  torque made (roll, pitch, yaw) {d['torque'][0]} N m, wanted {d['torque_want'][0]} N m")
        lines.append(f"  drag {d['drag'][0]} N, net force {d['f_net'][0]} N, accel {d['accel'][0]} m/s^2")
        a = d["angles"]
        lines.append(f"  angles: thrust axis to target {a['thrust_los'][0]:.1f} deg, velocity to target {a['vel_los'][0]} deg, "
                     f"tilt {a['tilt'][0]:.1f} deg, yaw error {a['yaw_err'][0]:+.1f} deg")
        return lines

    def meta(self):
        """Constants the page needs to draw the vectors and judge saturation."""
        env, cfg = self.env, self.env.config
        return {
            "style": env.drone_style, "mass": cfg.mass, "weight": cfg.mass * 9.81,
            "rotor_xy": [[r.position.x, r.position.y] for r in cfg.rotors], "rotor_spin": [r.spin_dir for r in cfg.rotors],
            "rotor_hover_f": cfg.mass * 9.81 / 4,
            "action_low": [float(v) for v in env.action_space.low], "action_high": [float(v) for v in env.action_space.high],
        }
