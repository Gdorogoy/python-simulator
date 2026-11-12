"""Chasing drones ("threats") for the numpy viewer sim -- the same model the Isaac env uses (base_drone_env_isaac.py).

Each threat is a drone: a QuadState stepped with the viewer drone's own physics (`timestamp_update`), flown by the limit-LOS
controller (straight at the drone at its speed cap, the cap being the only chase limit; app/control/los_np.py). It catches the drone
when it passes within the intercept radius. The policy sees `THREAT_FEATURES` numbers per slot after the 23 base inputs: rel pos
(symlog), rel vel, alive; nearest first, unused slots all zero. A checkpoint trained with K slots takes 23 + 7*K inputs (+ 2 more
when it was trained with the rc reward's zone timer), so the viewer always fills K slots and switches on only `active` of them.

Spawn, speed and release mirror the Isaac trainer (base_drone_env_isaac.set_threats): "uniform" puts every slot at a random point
`spawn_min`..`spread` m from the base, "base" puts them all at the base with `buffer` clearance; each draws its speed cap from
U(min_speed, max_speed) and waits, hovering, until the caller latches its release (Threats.update_release: the drone has closed to
the threat's release fraction of the distance it had left at the handoff).
"""
import json
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from app.control.los_np import CHASER_CRUISE_GAIN, chase_velocity, hold_velocity, velocity_to_action
from app.dynamics import drone_params as P
from app.dynamics.angle_style_methods import mixer_inversion, timestamp_update
from app.dynamics.drone import Quaternion, QuadState, Vector3D, create_quad_config
from app.environmental.base_drone_env import DIST_SCALE, VEL_SCALE, BaseDroneEnv, _symlog_scale

THREAT_FEATURES = 7          # keep in sync with base_drone_env_isaac.THREAT_FEATURES
BASE_OBS_DIM = 23
TIMER_FEATURES = 2           # in_zone, zone seconds / zone_time -- keep in sync with angle_reward_rc.TIMER_FEATURES
# obs_layout "v2" (23 base, timer, opponents, allies, flags) -- keep in sync with environmental/perception.py (torch, so not imported here)
OPPONENT_FEATURES = 4        # estimated rel position (symlog) x, y, z + status
ALLY_FEATURES = 5            # x, y, z, status, id
FLAG_FEATURES = 3            # link up, jammed, autonomous (placeholder: always [1, 0, 0])
DEFAULT_INTERCEPT_RADIUS = 0.25   # RewardRCConfig.intercept_radius
DRONE_BODY_RADIUS = 0.5 * math.sqrt(0.4 ** 2 + 0.4 ** 2 + 0.02 ** 2)   # half the diagonal of the drone's 0.4 x 0.4 x 0.02 m body (Isaac cfg.robot)


def layout_from_obs_dim(obs_dim):
    """(threat slots, has zone timer) for a checkpoint with `obs_dim` inputs: 23 + 7*K, or 23 + 7*K + 2 with the timer.
    The two never collide (2 is not a multiple of 7). ValueError for any other size."""
    extra = obs_dim - BASE_OBS_DIM
    if extra >= 0 and extra % THREAT_FEATURES == 0:
        return extra // THREAT_FEATURES, False
    if extra >= TIMER_FEATURES and (extra - TIMER_FEATURES) % THREAT_FEATURES == 0:
        return (extra - TIMER_FEATURES) // THREAT_FEATURES, True
    raise ValueError(f"observation size {obs_dim} is not {BASE_OBS_DIM} + {THREAT_FEATURES}*K threat slots "
                     f"(+ {TIMER_FEATURES} for the zone timer)")


def slots_from_obs_dim(obs_dim):
    """How many threat slots a checkpoint with `obs_dim` inputs was built for."""
    return layout_from_obs_dim(obs_dim)[0]


def default_gains():
    """Attitude gains of the threat drones: the SHORT-RANGE entry of the PID table (smallest distance key), as in the Isaac env's
    set_threat_gains."""
    path = Path(__file__).resolve().parents[1] / "control" / "best_pid_gains_per_dist.json"
    with open(path) as f:
        table = json.load(f)
    g = table[min(table, key=float)]
    return {k: float(g[k]) for k in ("kp_att", "kd_att", "kp_yaw", "kd_yaw")}


class Threats:
    """K chasing drones. Each is a QuadState stepped with the viewer drone's own `timestamp_update` (angle style), flown by the
    limit-LOS controller (control/los_np.py): straight at the drone at its speed cap, the cap being the only chase limit. The
    controller runs at the policy rate (every DECIMATION ticks), the physics every tick, like the Isaac env.
    Not modelled here: contact. The Isaac threats are rigid bodies and bounce off the drone; here a catch is the swept
    closest-approach test against `intercept_radius` (the reward's rule), and threats pass through each other."""

    DECIMATION = 4

    def __init__(self, slots, active, max_speed=50.0, spread=30.0, spawn_z=3.0,
                 intercept_radius=DEFAULT_INTERCEPT_RADIUS, rng=None, spawn="uniform", spawn_min=15.0, min_speed=None,
                 release_frac=0.7, release_frac_range=None, buffer=0.5, radius=None, gains=None, max_tilt=math.radians(90.0),
                 geometry_weights=(1.0, 1.0), reaction_s=1.5, tail_gap=(15.0, 40.0), jitter=2.0):
        spawn = "uniform" if spawn == "sphere" else spawn
        if spawn not in ("uniform", "base", "geometry"):
            raise ValueError(f"spawn must be 'uniform', 'base' or 'geometry', got {spawn!r}")
        if len(geometry_weights) != 2 or min(geometry_weights) < 0 or sum(geometry_weights) <= 0:
            raise ValueError(f"geometry_weights = (head-on, tail) must be >= 0 and not all zero, got {geometry_weights}")
        if reaction_s <= 0 or jitter < 0 or not 0 <= tail_gap[0] <= tail_gap[1]:
            raise ValueError("geometry spawn: need reaction_s > 0, jitter >= 0, 0 <= tail_gap lo <= hi")
        min_speed = max_speed if min_speed is None else min_speed
        if not 0.0 <= spawn_min <= spread:
            raise ValueError(f"need 0 <= spawn_min <= spread, got {spawn_min} and {spread}")
        if not 0.0 < min_speed <= max_speed:
            raise ValueError(f"need 0 < min_speed <= max_speed, got {min_speed} and {max_speed}")
        if not 0.0 < release_frac <= 1.0:
            raise ValueError(f"release_frac must be in (0, 1], got {release_frac}")
        if release_frac_range is not None and not 0.0 < release_frac_range[0] <= release_frac_range[1] <= 1.0:
            raise ValueError(f"release_frac_range needs 0 < lo <= hi <= 1, got {release_frac_range}")
        if buffer < 0.0:
            raise ValueError(f"buffer must be >= 0, got {buffer}")
        if active > slots:
            raise ValueError(f"{active} chasers asked for, but this checkpoint only has {slots} threat slot(s)"
                             if slots else f"{active} chasers asked for, but this checkpoint has no threat inputs "
                                           f"(it takes the plain {BASE_OBS_DIM}-number observation)")
        self.slots, self.active = int(slots), int(active)
        self.max_speed, self.spread, self.spawn_z = float(max_speed), float(spread), float(spawn_z)
        self.spawn, self.spawn_min, self.min_speed = spawn, float(spawn_min), float(min_speed)
        self.release_frac = float(release_frac)
        self.geometry_weights = tuple(float(x) for x in geometry_weights)       # spawn="geometry": head-on, tail
        self.reaction_s, self.jitter = float(reaction_s), float(jitter)
        self.tail_gap = (float(tail_gap[0]), float(tail_gap[1]))
        self.release_frac_range = None if release_frac_range is None else (float(release_frac_range[0]), float(release_frac_range[1]))
        self.buffer = float(buffer)
        self.radius = None if radius is None else float(radius)          # collision-sphere radius in the Isaac run (drawn, not simulated)
        self.body_radius = self.radius if self.radius is not None else DRONE_BODY_RADIUS
        self.max_tilt = float(max_tilt)
        self.gains = gains if gains is not None else default_gains()
        self.speed_cap = np.full(self.slots, self.max_speed)       # per slot, redrawn every reset
        self.release_frac_slot = np.full(self.slots, self.release_frac)
        self.released = np.zeros(self.slots, dtype=bool)
        self.intercept_radius = float(intercept_radius)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.config = create_quad_config(mass=P.MASS, inertia=P.INERTIA, arm_length=P.ARM_LENGTH, drag_coeff=P.DRAG_COEFF,
                                         max_rpm=P.MAX_RPM, motor_tau=P.MOTOR_TAU)
        self.hover_thrust = self.config.mass * 9.81
        self._hover_rpm = list(mixer_inversion(self.config, [self.hover_thrust, 0.0, 0.0, 0.0]))
        low, high = P.angle_action_bounds(self.config.mass)
        self._low, self._high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        self.pos = np.zeros((self.slots, 3))
        self.vel = np.zeros((self.slots, 3))
        self.alive = np.arange(self.slots) < self.active
        self._prev_rel = np.zeros((self.slots, 3))
        self._hold = np.zeros((self.slots, 3))
        self._states = [self._new_state(np.zeros(3)) for _ in range(self.slots)]
        self._action = np.zeros((self.slots, 4))
        self._tick = 0
        self._spawned = False
        self.geom = np.full(self.slots, -1)

    def _new_state(self, pos, vel=(0.0, 0.0, 0.0)):
        return QuadState(position=Vector3D(*pos), velocity=Vector3D(*vel), orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                         angular_velocity=Vector3D(0.0, 0.0, 0.0), rotor_rpm=list(self._hover_rpm))

    def _base_ring(self):
        """(slots, 3) offsets of the "base" layout: slot 0 at the centre, the rest on a horizontal ring (random rotation) with every pair at
        least gap = 2*body_radius + buffer apart: rho = max(gap, gap / (2 sin(pi/m))), m = slots - 1 (same as the Isaac env)."""
        off = np.zeros((self.slots, 3))
        m = self.slots - 1
        if m <= 0:
            return off
        gap = 2.0 * self.body_radius + self.buffer
        rho = gap if m == 1 else max(gap, gap / (2.0 * math.sin(math.pi / m)))
        ang = self.rng.random() * 2.0 * math.pi + np.arange(m) * (2.0 * math.pi / m)
        off[1:, 0], off[1:, 1] = rho * np.cos(ang), rho * np.sin(ang)
        return off

    def reset(self, base_pos, drone_pos):
        """Re-spawn. "uniform" = every slot at a random point spawn_min..spread m from the base (uniform in volume, mirrored up if
        underground); "base" = all from the base at spawn_z, slot 0 on it, the others on a ring with `buffer` clearance. Every threat
        starts level and at rest, hovering, and draws its speed cap and release fraction."""
        self.speed_cap = self.min_speed + (self.max_speed - self.min_speed) * self.rng.random(self.slots)
        if self.release_frac_range is None:
            self.release_frac_slot = np.full(self.slots, self.release_frac)
        else:
            lo, hi = self.release_frac_range
            self.release_frac_slot = lo + (hi - lo) * self.rng.random(self.slots)
        self.released = np.zeros(self.slots, dtype=bool)
        if self.spawn == "uniform":
            base = np.array([base_pos[0], base_pos[1], base_pos[2]], dtype=np.float64)
            direction = self.rng.normal(size=(self.slots, 3))
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
            lo3, hi3 = self.spawn_min ** 3, self.spread ** 3
            radius = (lo3 + (hi3 - lo3) * self.rng.random((self.slots, 1))) ** (1.0 / 3.0)
            self.pos = base + direction * radius
            self.pos[:, 2] = np.abs(self.pos[:, 2])
        elif self.spawn == "base":
            base = np.array([base_pos[0], base_pos[1], self.spawn_z], dtype=np.float64)
            self.pos = np.repeat(base[None, :], self.slots, axis=0) + self._base_ring()
        else:                                    # "geometry": parked high up until spawn_at_handoff
            self.pos = np.array([[10.0 * k, 0.0, 300.0] for k in range(self.slots)], dtype=np.float64).reshape(self.slots, 3)
        self.vel = np.zeros((self.slots, 3))
        self._hold = self.pos.copy()
        self._states = [self._new_state(p) for p in self.pos]
        self._action = np.zeros((self.slots, 4))
        self._tick = 0
        self._prev_rel = self.pos - np.asarray(drone_pos, dtype=np.float64)
        self._spawned = False
        self.geom = np.full(self.slots, -1)
        self.alive = np.zeros(self.slots, dtype=bool) if self.spawn == "geometry" else np.arange(self.slots) < self.active

    def spawn_at_handoff(self, drone_pos, drone_vel, target_pos):
        """spawn="geometry": the numpy twin of BaseDroneEnvIsaac._spawn_geometry. Call every step while the policy flies; it acts once. Puts each
        active chaser head-on to / behind the drone (weights geometry_weights), already moving, released at once:
          head-on : ahead along the drone's velocity at max(20 m, reaction_s * (drone speed + cap)), +-jitter off the line, flying back at its cap;
          tail    : U(tail_gap) m behind, flying forward at its cap.
        Positions are kept >= 1 m above the ground. Slots beyond `active` stay parked."""
        if self.spawn != "geometry" or self._spawned or self.active == 0:
            return
        p, v = np.asarray(drone_pos, dtype=np.float64), np.asarray(drone_vel, dtype=np.float64)
        speed = float(np.linalg.norm(v))
        to_target = np.asarray(target_pos, dtype=np.float64) - p
        to_target = to_target / max(float(np.linalg.norm(to_target)), 1e-9)
        fwd = v / speed if speed > 1.0 else to_target
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(fwd, up)
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-3 else np.array([1.0, 0.0, 0.0])
        w = np.asarray(self.geometry_weights, dtype=np.float64)
        geom = self.rng.choice(2, size=self.slots, p=w / w.sum())                       # 0 head-on, 1 tail
        cap = self.speed_cap[:, None]

        def rnd(lo, hi):
            return lo + (hi - lo) * self.rng.random((self.slots, 1))

        j = self.jitter
        head_on = fwd * np.maximum((speed + cap) * self.reaction_s, 20.0) + side * rnd(-j, j) + up * rnd(-j, j)
        tail = -fwd * rnd(*self.tail_gap)
        g = geom[:, None]
        pos = p + np.where(g == 0, head_on, tail)
        pos[:, 2] = np.maximum(pos[:, 2], 1.0)
        to_drone = p[None, :] - pos                                                       # start moving AT the drone at the cap
        vel = to_drone / np.maximum(np.linalg.norm(to_drone, axis=1, keepdims=True), 1e-9) * cap
        act = (np.arange(self.slots) < self.active)[:, None]
        self.pos = np.where(act, pos, self.pos)
        self.vel = np.where(act, vel, 0.0)
        self._states = [self._new_state(self.pos[k], self.vel[k]) for k in range(self.slots)]
        self.alive = act[:, 0].copy()
        self.released = np.ones(self.slots, dtype=bool)
        self._hold = self.pos.copy()
        self._prev_rel = self.pos - p
        self._action = np.zeros((self.slots, 4))
        self._tick = 0
        self.geom = np.where(act[:, 0], geom, -1)
        self._spawned = True

    def update_release(self, left, ref):
        """Latch the release of every threat whose fraction is reached: the drone, flown by the policy, has at most
        release_frac_slot * ref of the distance left (ref = distance left at the handoff). Call only while the policy flies."""
        self.released |= float(left) <= self.release_frac_slot * float(ref)

    def _attitude(self):
        quats = np.array([[s.orientation.x, s.orientation.y, s.orientation.z, s.orientation.w] for s in self._states])
        roll, pitch, yaw = Rotation.from_quat(quats).as_euler("xyz").T
        ang = np.array([[s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z] for s in self._states])
        return roll, pitch, yaw, ang

    def _control(self, drone_pos):
        """Angle-style action (slots, 4) for every threat: chase at the speed cap once released, else hold the spawn point."""
        cap = self.speed_cap[:, None]
        v_chase = chase_velocity(np.asarray(drone_pos, dtype=np.float64) - self.pos, cap)
        v_hold = hold_velocity(self._hold - self.pos)
        moving = (self.alive & self.released)[:, None]
        v_des = np.where(moving, v_chase, v_hold)
        roll, pitch, yaw, ang = self._attitude()
        g = self.gains
        action, _, _ = velocity_to_action(v_des, self.vel, ang, roll, pitch, yaw, np.zeros(self.slots), hover_thrust=self.hover_thrust,
                                          kp_att=g["kp_att"], kd_att=g["kd_att"], kp_yaw=g["kp_yaw"], kd_yaw=g["kd_yaw"],
                                          max_tilt=self.max_tilt, cruise_gain=CHASER_CRUISE_GAIN)
        return np.clip(action, self._low, self._high)

    def step(self, drone_pos, drone_vel, dt):
        """Advance one tick. Threats wait (hovering) until update_release() latched them; dead slots are not simulated."""
        if self.active == 0:
            return
        if self._tick % self.DECIMATION == 0:
            self._action = self._control(drone_pos)
        self._tick += 1
        for k in np.flatnonzero(self.alive):
            real = self._action[k].copy()
            real[0] += self.hover_thrust
            s = timestamp_update(self._states[k], self.config, list(real), [0.0, 0.0, 0.0], dt)
            if s.position.z < 0.0:                                   # the ground (Isaac has a plane)
                s.position.z = 0.0
                s.velocity.z = max(s.velocity.z, 0.0)
            self._states[k] = s
            self.pos[k] = (s.position.x, s.position.y, s.position.z)
            self.vel[k] = (s.velocity.x, s.velocity.y, s.velocity.z)

    def closest_approach(self, drone_pos):
        """Distance of the nearest alive threat at its CLOSEST point since the last call (swept, so a fast pass cannot tunnel
        through the catch radius between two ticks). inf when nothing is active."""
        if self.active == 0:
            return np.inf
        r1 = self.pos - np.asarray(drone_pos, dtype=np.float64)
        r0 = self._prev_rel
        dr = r1 - r0
        t = np.clip(-(r0 * dr).sum(axis=1) / np.maximum((dr * dr).sum(axis=1), 1e-9), 0.0, 1.0)
        closest = r0 + t[:, None] * dr
        self._prev_rel = r1
        d = np.linalg.norm(closest, axis=1)
        d[~self.alive] = np.inf
        return float(d.min())

    def nearest_distance(self, drone_pos):
        """Plain current distance to the nearest alive threat (for the HUD), inf when nothing is active."""
        if self.active == 0:
            return np.inf
        d = np.linalg.norm(self.pos - np.asarray(drone_pos, dtype=np.float64), axis=1)
        d[~self.alive] = np.inf
        return float(d.min())

    def chaser_states(self, drone_pos):
        """Per active chaser, for the viewer: (speed m/s (n,), angle deg between its velocity and its line of sight to the drone (n,),
        velocity (n, 3)). The angle is NaN while a chaser is nearly at rest or not alive (geometry spawn before the handoff)."""
        n = self.active
        rel = np.asarray(drone_pos, dtype=np.float64) - self.pos[:n]
        vel = self.vel[:n].copy()
        speed = np.linalg.norm(vel, axis=1)
        dist = np.linalg.norm(rel, axis=1)
        cos = np.einsum("ij,ij->i", vel, rel) / np.maximum(speed * dist, 1e-12)
        los = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        los[(speed < 0.3) | (dist < 1e-9) | ~self.alive[:n]] = np.nan
        return speed, los, vel

    def observation(self, drone_pos, drone_vel):
        """(7*slots,) policy inputs: slots sorted nearest first, dead slots all zero (same layout as the Isaac env)."""
        if self.slots == 0:
            return np.zeros(0, dtype=np.float32)
        rel = self.pos - np.asarray(drone_pos, dtype=np.float64)
        rvel = self.vel - np.asarray(drone_vel, dtype=np.float64)
        key = np.where(self.alive, np.linalg.norm(rel, axis=1), 1e9)
        order = np.argsort(key, kind="stable")
        alive = self.alive[order][:, None].astype(np.float64)
        feat = np.concatenate([_symlog_scale(rel[order], DIST_SCALE), rvel[order] / VEL_SCALE, alive], axis=1) * alive
        return feat.reshape(-1).astype(np.float32)


    def observation_v2(self, drone_pos):
        """(4*slots,) policy inputs of obs_layout "v2": relative position (symlog) + status per slot, nearest first, unknown slots zero.
        The perception layer is a placeholder (the truth), so status = alive; same as BaseDroneEnvIsaac._opponent_obs."""
        if self.slots == 0:
            return np.zeros(0, dtype=np.float32)
        rel = self.pos - np.asarray(drone_pos, dtype=np.float64)
        key = np.where(self.alive, np.linalg.norm(rel, axis=1), 1e9)
        order = np.argsort(key, kind="stable")
        status = self.alive[order][:, None].astype(np.float64)
        feat = np.concatenate([_symlog_scale(rel[order], DIST_SCALE), status], axis=1) * status
        return feat.reshape(-1).astype(np.float32)


class ThreatDroneEnv(BaseDroneEnv):
    """BaseDroneEnv that also flies the chasing drones: appends their inputs to the observation and ends the episode
    with reason "Intercepted" (reward -15, like angle_reward_rc) when one reaches the drone.

    Call set_threats() once after construction. Until the caller latches their release (Threats.update_release, once the policy flies
    and the release point is reached) the threats hover where they spawned, as in the Isaac handoff stage.

    set_timer() adds the rc reward's two zone-timer inputs (in_zone, seconds in the zone / zone_time) after the threat slots,
    for checkpoints trained with them (observation 23 + 7*K + 2). The viewer still ends a flight with the old numpy hit rule."""

    R_INTERCEPTED = -15.0

    def __init__(self, *args, **kwargs):
        self.threats = None
        self.layout_v2 = False       # set_layout_v2()
        self.allies = 0
        self.timer = None            # (hit_radius, zone_time_s) once set_timer() was called
        self._zone_s = 0.0
        super().__init__(*args, **kwargs)

    def set_threats(self, threats):
        self.threats = threats
        self._update_observation_space()

    def set_timer(self, hit_radius, zone_time):
        """Append the zone-timer inputs; `hit_radius` / `zone_time` (s) are the values the checkpoint was trained with."""
        self.timer = (float(hit_radius), float(zone_time))
        self._zone_s = 0.0
        self._update_observation_space()

    def set_layout_v2(self, allies=0):
        """Switch to obs_layout "v2": 23 base, [timer], 4 per opponent slot, 5 per ally slot, 3 flags (what the trainer's --obs-layout v2 builds)."""
        self.layout_v2, self.allies = True, int(allies)
        self._update_observation_space()

    def _update_observation_space(self):
        slots = 0 if self.threats is None else self.threats.slots
        timer = TIMER_FEATURES if self.timer else 0
        if self.layout_v2:
            dim = BASE_OBS_DIM + timer + OPPONENT_FEATURES * slots + ALLY_FEATURES * self.allies + FLAG_FEATURES
        else:
            dim = BASE_OBS_DIM + THREAT_FEATURES * slots + timer
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def _timer_obs(self):
        hit_radius, zone_time = self.timer
        dist = float(np.linalg.norm(np.asarray(self.target_pos, dtype=np.float64) - self._drone_xyz()))
        return np.array([float(dist < hit_radius), min(self._zone_s / zone_time, 1.0)], dtype=np.float32)

    def _drone_xyz(self):
        p = self.drone_state.position
        return np.array([p.x, p.y, p.z])

    def _drone_vel(self):
        v = self.drone_state.velocity
        return np.array([v.x, v.y, v.z])

    def _get_obs(self):
        obs = super()._get_obs()
        if self.layout_v2:
            parts = [obs]
            if self.timer:
                parts.append(self._timer_obs())
            if self.threats is not None and self.threats.slots:
                parts.append(self.threats.observation_v2(self._drone_xyz()))
            parts.append(np.zeros(ALLY_FEATURES * self.allies, dtype=np.float32))      # no ally behaviour yet
            parts.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))                  # link up, not jammed, not autonomous (placeholder)
            return np.concatenate(parts).astype(np.float32)
        if self.threats is not None and self.threats.slots:
            obs = np.concatenate([obs, self.threats.observation(self._drone_xyz(), self._drone_vel())])
        if self.timer:
            obs = np.concatenate([obs, self._timer_obs()])
        return obs

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._zone_s = 0.0
        if self.threats is None:
            return self._get_obs() if (self.timer or self.layout_v2) else obs, info
        self.threats.reset(self.target_pos, self._drone_xyz())
        return self._get_obs(), info

    def step(self, action):
        if self.threats is not None:
            self.threats.step(self._drone_xyz(), self._drone_vel(), self.dt)
        obs, reward, terminated, truncated, info = super().step(action)
        if self.threats is not None and self.threats.closest_approach(self._drone_xyz()) < self.threats.intercept_radius:
            reward, terminated = self.R_INTERCEPTED, True
            info = dict(info, reason="Intercepted")
        if self.timer:           # consecutive time inside the zone, counting this step, as the trainer's _zone_count does
            hit_radius = self.timer[0]
            in_zone = np.linalg.norm(np.asarray(self.target_pos, dtype=np.float64) - self._drone_xyz()) < hit_radius
            self._zone_s = self._zone_s + self.dt if in_zone else 0.0
            obs = self._get_obs()
        return obs, reward, terminated, truncated, info


RUNNER_FEATURES = 3          # keep in sync with base_drone_env_isaac.RUNNER_FEATURES: target velocity - drone velocity, / VEL_SCALE, last inputs


class Runner:
    """Numpy twin of the Isaac acro runner (base_drone_env_isaac "runner"): the TARGET is a drone with the viewer drone's physics that
    hovers on the target point until release() (the handoff, or the first step without one), then flees straight away from the drone at
    its speed cap ~ U(min_speed, max_speed), never below min_alt. Released already moving at the cap, like the Isaac one. The caller makes
    `pos` the target once released and ends the episode as 'escape' once escaped(): the runner is more than escape_dist from its spawn
    point (the target point it hovered on), wherever the drone is."""

    DECIMATION = 4

    def __init__(self, max_speed, min_speed=None, escape_dist=100.0, min_alt=2.0, max_tilt=math.radians(35.0), gains=None, rng=None):
        self.max_speed = float(max_speed)
        self.min_speed = self.max_speed if min_speed is None else float(min_speed)
        if not 0.0 < self.min_speed <= self.max_speed:
            raise ValueError(f"need 0 < runner min_speed <= max_speed, got {self.min_speed} and {self.max_speed}")
        self.escape_dist, self.min_alt, self.max_tilt = float(escape_dist), float(min_alt), float(max_tilt)
        self.gains = gains if gains is not None else default_gains()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.config = create_quad_config(mass=P.MASS, inertia=P.INERTIA, arm_length=P.ARM_LENGTH, drag_coeff=P.DRAG_COEFF,
                                         max_rpm=P.MAX_RPM, motor_tau=P.MOTOR_TAU)
        self.hover_thrust = self.config.mass * 9.81
        self._hover_rpm = list(mixer_inversion(self.config, [self.hover_thrust, 0.0, 0.0, 0.0]))
        low, high = P.angle_action_bounds(self.config.mass)
        self._low, self._high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        self.reset(np.zeros(3))

    def reset(self, target_pos):
        self.pos = np.asarray(target_pos, dtype=np.float64).copy()
        self.vel = np.zeros(3)
        self._hold = self.pos.copy()
        self._state = QuadState(position=Vector3D(*self.pos), velocity=Vector3D(0.0, 0.0, 0.0), orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                                angular_velocity=Vector3D(0.0, 0.0, 0.0), rotor_rpm=list(self._hover_rpm))
        self.cap = float(self.rng.uniform(self.min_speed, self.max_speed))
        self.released = False
        self._action = np.zeros(4)
        self._tick = 0

    def _flee(self, drone_pos):
        v = chase_velocity((self.pos - np.asarray(drone_pos, dtype=np.float64))[None], self.cap)[0]
        v[2] = max(v[2], 2.0 * (self.min_alt - self.pos[2]))         # 0 at min_alt, climbs back above it (ground at z = 0)
        return v

    def release(self, drone_pos):
        """Start fleeing, already at the cap along the flee direction."""
        v0 = self._flee(drone_pos)
        v0 = v0 / max(np.linalg.norm(v0), 1e-6) * self.cap
        self._state.velocity = Vector3D(*v0)
        self.vel = v0.copy()
        self.released = True

    @property
    def spawn(self):
        """Centre of the escape sphere: where the runner started (the episode's target point)."""
        return self._hold

    def escaped(self):
        return self.released and float(np.linalg.norm(self.pos - self._hold)) > self.escape_dist

    def obs_velocity(self):
        """What the policy sees as the target velocity: 0 until released (the Isaac env reports the same)."""
        return self.vel if self.released else np.zeros(3)

    def _control(self, drone_pos):
        v_des = self._flee(drone_pos) if self.released else hold_velocity((self._hold - self.pos)[None])[0]
        s = self._state
        roll, pitch, yaw = Rotation.from_quat([s.orientation.x, s.orientation.y, s.orientation.z, s.orientation.w]).as_euler("xyz")
        ang = np.array([[s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z]])
        g = self.gains
        action, _, _ = velocity_to_action(v_des[None], self.vel[None], ang, np.array([roll]), np.array([pitch]), np.array([yaw]), np.zeros(1),
                                          hover_thrust=self.hover_thrust, kp_att=g["kp_att"], kd_att=g["kd_att"], kp_yaw=g["kp_yaw"],
                                          kd_yaw=g["kd_yaw"], max_tilt=self.max_tilt, cruise_gain=CHASER_CRUISE_GAIN)
        return np.clip(action[0], self._low, self._high)

    def step(self, drone_pos, dt):
        """One physics tick; the controller runs every DECIMATION ticks (the policy rate), like the Isaac env."""
        if self._tick % self.DECIMATION == 0:
            self._action = self._control(drone_pos)
        self._tick += 1
        real = self._action.copy()
        real[0] += self.hover_thrust
        s = timestamp_update(self._state, self.config, list(real), [0.0, 0.0, 0.0], dt)
        if s.position.z < 0.0:
            s.position.z = 0.0
            s.velocity.z = max(s.velocity.z, 0.0)
        self._state = s
        self.pos = np.array([s.position.x, s.position.y, s.position.z])
        self.vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
