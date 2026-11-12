# environmental

## Purpose
Turns the physics in `dynamics` into RL environments. There are two implementations of the same task: a numpy
Gymnasium env (single drone, CPU, used for replay, evaluation and the web viewer) and an Isaac Lab env (thousands
of drones on the GPU, used for training).

## Files

| File | What it holds |
|---|---|
| `base_drone_env.py` | `BaseDroneEnv` (Gymnasium id `base_drone_env_v0`), `build_observation`, observation scales. The numpy reference. |
| `base_drone_env_isaac.py` | `BaseDroneEnvIsaac` (DirectRLEnv). Two gym ids: `Isaac-Base-Drone-Direct-v0` (angle, `BaseDroneEnvIsaacCfg`) and `Isaac-Acro-Drone-Direct-v0` (acro, `AcroDroneEnvIsaacCfg`). Also the threat drones. Importable only inside the Isaac venv. |
| `subproc_vec_base_drone_env.py` | `SubprocVecBaseDroneEnv`: runs N numpy envs across worker processes. |
| `enviorment.py` | `sample_wind_conditions` (wind vector, mass scale). |
| `threats.py` | Numpy copy of the Isaac chasing drones for the web viewer: `Threats` (each chaser a `QuadState` stepped with `timestamp_update`, flown by the numpy limit-LOS `control/los_np.py`, per-threat release latch, uniform / base spawn; and the 7-per-slot observation) and `ThreatDroneEnv` (`BaseDroneEnv` that appends them to the observation and ends the episode `Intercepted`). |

## Core

**Observation (23 dims, same layout in both envs):**

| Slice | Content | Scaling |
|---|---|---|
| 0:3 | position, depends on the position mode | symlog, `POS_SCALE = 15` |
| 3:6 | linear velocity (world) | `/ VEL_SCALE` (10 m/s): plain normalisation, not a speed cap |
| 6:10 | orientation quaternion (x, y, z, w) | raw |
| 10:13 | angular velocity (body) | `/ ANG_VEL_SCALE` (20) |
| 13:17 | rotor rpm | `/ MAX_RPM` |
| 17:20 | vector to target | symlog, `DIST_SCALE = 15` |
| 20 | distance to target | symlog |
| 21:23 | sin/cos of yaw error | sin/cos, so the +-pi wrap is continuous |

**Extra inputs (Isaac, angle trainer):** `7 * K` threat numbers when `cfg.max_threats = K`, then 2 zone-timer numbers when
`cfg.timer_obs` (the rc reward). They are always appended after the 23, never inserted, so older checkpoints still load.

**Action:** angle style `[thrust delta around hover, roll, pitch, yaw torque]`; acro style `[thrust delta, p, q, r]` (body
rates). In both, the env adds hover thrust back on. Bounds come from `dynamics/drone_params.py`.

**Timing:** physics at 240 Hz. The Isaac env uses `decimation = 4`, so the policy runs at 60 Hz. An episode lasts
at most `MAX_POLICY_STEPS * PHYSICS_DT` = 62.5 s.

**Reward:** injected. The numpy env takes `custom_reward(env)`. The Isaac env's `_compute_dones_and_reward` is
replaced per instance by `install_reward_qv1/qv2` (see `reward_functions/docs.md`).

## Position modes
- `"full"` gives obs[0:3] = symlog(x, y, z). The policy can see where in the world it is. A policy trained with the
  handoff near the origin breaks when the handoff happens 200+ m away (measured: 3/24 hits vs 24/24).
- `"height"` gives obs[0:3] = (0, 0, symlog(clip(z, 0, 30 m))). There is no absolute horizontal position, and
  "30 m up" looks the same as "200 m up". The rest of the observation is already relative or body-frame. The
  layout is unchanged, so a `full` checkpoint still loads.
- The mode must match between training (Isaac) and replay (numpy). It is stored in `run_config.json` as
  `obs_position_mode`.

## Isaac env specifics
- **Spawn and targets:** use `set_target_pairs` (a fixed pool) or `set_target_distance_range` (a fresh target each
  reset: uniform direction and random yaw). Underground directions are mirrored upward, so the 0.5 m floor clamp
  never changes the distance.
- **`set_spawn_speed_range`** spawns the drone already moving toward its target, simulating a PID handoff. A policy
  trained only from rest never sees the cruise-speed state a real handoff leaves it in.
- **Action space** must be an explicit bounded `Box`. A bare int creates an unbounded Box, and `ActorCritic`'s tanh
  rescale then produces NaN on the first action.
- **Robot asset** is a procedural `RigidObject` (cuboid), not an Articulation. Its size is chosen so its
  uniform-density inertia ratio matches `QuadConfig`. See `info/PROJECT_DEFENSE_GUIDE.md` Part 8.2 for the
  mass-override bug chain that led to this.
- `_get_rewards` stores the observation in `extras["terminal_observation"]` before `_reset_idx` overwrites it, so
  truncated episodes bootstrap from the true final state.

## Edge cases / gotchas
- **Action bounds are a design limit, not physics** (`drone_params.py`). Roll/pitch torque is +-1.5 N*m (was 0.5): with
  0.02 kg*m^2 inertia that is 75 rad/s^2, and the rotors could deliver about 4-5 N*m. Yaw stays +-0.5 N*m (about 0.3 N*m is
  physically reachable at hover). Thrust delta runs from -1x to +2x hover, so up to 3x weight in total (altitude can be held up
  to about 70 deg of tilt).
- **Wind and mass randomization exist but are disabled:** `reset()` sets `wind = 0` and `mass_scale = 1`.
- `render_mode` is accepted but ignored. The PyBullet GUI was removed, and the visual paths are Isaac Sim and the
  web viewer (`guidance/serve_run.py`).
- `reset()` keeps a `target_pos` that was set directly on the env if none is passed.

## SubprocVecBaseDroneEnv
- Uses `spawn`, not `fork`, because the main process has already initialised CUDA, and forking after that is unsafe.
- Reward closures can't be pickled, so each worker rebuilds its envs from plain data: a module-level
  `reward_fn_factory(p, **kwargs)`, a param dict, and target pairs.
- Each episode's PID teacher is swapped to the per-distance gain set (`best_pid_gains_per_dist.json`). One generic
  gain set is a much worse teacher across 3-250 m (for example kp_pos 1.65 at 3 m vs 0.23 at 30 m in the current file).
- `step_with_pid_actions` returns the PID action for the pre-step state, used for imitation pairs.
  `get_target_positions` returns a live snapshot of each env's target, start distance and current distance.
- Always call `close()`. Garbage collection does not stop the worker processes.

## Depends on
`gymnasium`, `numpy`, `scipy`, `torch`, Isaac Lab (Isaac env only). Internally: `dynamics`, `control.pid`,
`reward_functions.rewards`.

## Threats and allies (Isaac env)

`cfg.max_threats = K` allocates K threat DRONES up front: one `RigidObject` per slot (`Threat0..`, a different colour each, the drone
is red), cloned into every env. `threat_pos_w`, `threat_vel_w` (shape `(N,K,3)`) mirror the simulated bodies each step, and
`threat_alive` masks the unused slots; empty `allied_*` arrays are reserved for later.

A threat flies the drone's own physics: the same rotor model (`motor_lag`, `mixer_inversion`, drag, wind) stepped every substep
in `_apply_threat_forces`, with an angle-style action from `_step_threats` once per policy step. The controller is the limit-LOS in
`control/los.py`: `v_des = speed_cap * unit(drone - threat)`, then the same velocity -> tilt/thrust maths the LOS teacher uses.
The speed cap (`set_threats(max_speed, min_speed)`, drawn per threat) is the only limit; the tilt clamp is the airframe's
(`set_threat_gains(gains_by_dist, max_tilt)` must be called before stepping; it takes the short-range PID attitude gains).
A waiting or parked threat hovers on a hold point with the same controller.

`cfg.threat_radius`: `None` = the threat body is the drone's own cuboid; a number = a collision sphere of that radius (m), with the
drone's mass and inertia copied onto it (`_sync_threat_bodies`). Threats collide with the drone and with each other.

`set_threats(active, max_speed, min_speed, spawn_z, spread, spawn, spawn_min, release_frac, release_frac_range, buffer, policy)`:
- `spawn="uniform"` (alias `"sphere"`): each threat at a random point of the ball `spawn_min..spread` around the target, uniform
  in volume (`r = (lo^3 + (hi^3 - lo^3) u)^(1/3)`), mirrored above the ground.
- `spawn="geometry"` (`geometry_weights`, `reaction_s`, `tail_gap`, `jitter`): nothing exists until the handoff. The step after the
  model takes over (the first step when there is no handoff stage) `_spawn_geometry` puts every active chaser relative to the DRONE, already
  moving, and releases it at once (the release fraction is ignored). Per chaser, drawn with weights (head-on, tail):
  head-on = ahead along the drone's velocity at `max(20 m, reaction_s * (drone speed + chaser cap))`, +-`jitter` m off the line, flying back at its cap; tail =
  `U(tail_gap)` m behind, flying forward at its cap. Positions are kept 1 m above the ground. `_threat_geom` (N, K) holds each
  chaser's geometry (0 head-on / 1 tail, -1 = none) and `GEOM_NAMES` the names; the trainer's eval reports hit / caught per geometry (the first chaser's).
  Before the spawn `threat_alive` is False, so the observation shows no chaser. The first policy step after the handoff is flown without one.
- `spawn="base"`: all from one point, the target at `spawn_z` height. Slot 0 on it, slots 1..K-1 on a horizontal ring (random
  rotation per env) of radius `rho = max(g, g / (2 sin(pi/m)))`, `g = 2 r_body + buffer`, `m = K-1`, so every pair is at least
  `g` apart.
- Release: a threat waits until the policy-flown drone has `release_frac` of the handoff distance LEFT (latched, per threat).
  `release_frac_range=(lo, hi)` draws it per threat and episode; `(0.01, 0.67)` = released between 33% and 99% of the way after the
  handoff (the trainer's `--threat-release-after 0.33 0.99`).
- `policy(env)` -> `(N,K,3)` desired velocity replaces the chase (still speed-limited); a trained interceptor can plug in here.

Spawn is lazy, after the handoff wrapper has placed the drone. Slots beyond `active` are parked 300 m above their env, hovering.

Observation: `23 + 7*K`. The 7 per slot (relative position symlog, relative velocity, alive flag) are appended AFTER the 23
base inputs and sorted nearest-first, dead slots all zero. Appending is what lets 23-input checkpoints load
(`load_bc_checkpoint` zero-pads the first layer). Allocate the maximum K you will use and raise `active` per stage.

## Observation layouts

`n` = opponent slots, `k` = ally slots (the angle drone's and the acro drone's `n` and `k` are unrelated). The planned layouts are written
for many v many: `n` is how many opposing drones one drone can track, `k` how many friendly ones. **1v1 is the same layout with `n = 1`
and `k = 0`** (the ally section then has zero width), so growing to 3v3 or 1 v many means raising `n` / `k` only. Slots are fixed capacity:
an unused or lost slot is all zeros. The plan behind the planned layouts is in `future/plan.md`; **only the two "today" tables are
implemented.**

### The 23 base inputs (`_get_observations`, the same for the angle and the acro drone)

| Idx | Count | Input | Scaling | Note |
|---|---|---|---|---|
| 0-2 | 3 | own position x, y, z | symlog (`POS_SCALE`) | `obs_position_mode=height`: x, y zero, z clipped to `HEIGHT_CLIP` |
| 3-5 | 3 | own linear velocity (world) | `/ VEL_SCALE` | |
| 6-9 | 4 | attitude quaternion (x, y, z, w) | none | Isaac's (w, x, y, z) reordered to scipy's order |
| 10-12 | 3 | body angular velocity | `/ ANG_VEL_SCALE` | |
| 13-16 | 4 | rotor speeds | `/ OBS_MAX_RPM` | |
| 17-19 | 3 | target position relative to the drone | symlog (`DIST_SCALE`) | target-dependent |
| 20 | 1 | distance to the target | symlog (`DIST_SCALE`) | target-dependent |
| 21-22 | 2 | yaw error to the target's desired yaw (sin, cos) | none | target-dependent |

Inputs 17-22 are the only ones that depend on the target; a jam or an autonomous mode would hold exactly these six.

### Today, angle drone: `23 + 7K + 2`

| Section | Size | Content |
|---|---|---|
| Base | 23 | table above |
| Threat slot x K | 7 each | relative position (symlog) 3, relative velocity 3, alive 1; sorted nearest-first, dead slots zero |
| Timer (`--reward rc`, off with `--no-timer-obs`) | 2 | inside the hit radius now (0/1), consecutive in-zone steps / steps needed for the hit (0..1) |

### Today, acro drone: `23`

| Section | Size | Content |
|---|---|---|
| Base | 23 | table above; one static target, no threat or ally inputs |

### Planned, angle drone (evader): `25 + 4n + 5k + 3`

n = threats it can track, k = other evaders. 1v1: n=1, k=0 -> 32. Example 2 evaders v 3 threats: n=3, k=1 -> 25 + 12 + 5 + 3 = 45.

| Section | Size | Content |
|---|---|---|
| Base | 23 | table above |
| Timer | 2 | as today, kept so existing checkpoints stay valid |
| Opponent slot x n (threats) | 4 each | estimated position x, y, z + status (known / alive) |
| Ally slot x k (other evaders) | 5 each | x, y, z, status (alive / lost), id; the allies report their own true position |
| Flags | 3 | link up, jammed, autonomous (see below) |

### Planned, acro drone (threat): `23 + 4n + 5k + 3`

n = evaders it can track (the assigned target is one of them, and also fills base inputs 17-22), k = other threats. 1v1: n=1, k=0 -> 30.
Example 3 threats v 2 evaders: n=2, k=2 -> 23 + 8 + 10 + 3 = 44.

| Section | Size | Content |
|---|---|---|
| Base | 23 | table above, with 17-22 = the assigned evader's estimate instead of a fixed target (the yaw error 21-22 is a placeholder) |
| Opponent slot x n (evaders) | 4 each | estimated position x, y, z + status |
| Ally slot x k (other threats) | 5 each | x, y, z, status, id |
| Flags | 3 | link up, jammed, autonomous (see below) |

Flags (last section, so more slots never shift them): `link up` = the base link works now; `jammed` = the jump detector fired (the target block changed by more than a threshold, ~20 m, in one step); `autonomous` = the drone is in autonomous mode (jammed, or the link down longer than T seconds, decided by an env rule; the six target inputs 17-22 are then held). Moving the timer behind the threat slots and widening the layout needs
the loader column map: an existing 25-input checkpoint (23 + timer) keeps columns 0-22, its timer goes to 23-24, and the new columns
start at zero (today `load_bc_checkpoint` only zero-pads at the end, which puts the old timer weights on threat slot 0).

### The numpy viewer env and v2

The web viewer flies the numpy twin (`threats.py` `ThreatDroneEnv`), not Isaac. It builds the v2 layout when the checkpoint's `run_config.json`
says `obs_layout: v2` (`record_run._v2_layout_for` reads `num_threats` / `num_allies`; the timer is whatever fills the remaining 2 inputs).
`Threats.observation_v2`, `ThreatDroneEnv.set_layout_v2` and the constants at the top of `threats.py` copy `base_drone_env_isaac._opponent_obs`,
`_ally_obs` and `perception.py` (perception is a placeholder, so status = alive and the flags are always `[1, 0, 0]`). **Any change to the
Isaac layout has to be copied there by hand.**
