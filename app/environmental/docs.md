# environmental

## Purpose
Turns the physics in `dynamics` into RL environments. There are two implementations of the same task: a numpy
Gymnasium env (single drone, CPU, used for replay, evaluation and the web viewer) and an Isaac Lab env (thousands
of drones on the GPU, used for training).

## Files

| File | What it holds |
|---|---|
| `base_drone_env.py` | `BaseDroneEnv` (Gymnasium id `base_drone_env_v0`), `build_observation`, observation scales. The numpy reference. |
| `base_drone_env_isaac.py` | `BaseDroneEnvIsaac` (DirectRLEnv, gym id `Isaac-Base-Drone-Direct-v0`). Importable only inside the Isaac venv. |
| `subproc_vec_base_drone_env.py` | `SubprocVecBaseDroneEnv`: runs N numpy envs across worker processes. |
| `enviorment.py` | `sample_wind_conditions` (wind vector, mass scale). |

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

**Action:** `[thrust delta around hover, roll, pitch, yaw torque]`. The env adds hover thrust back on.

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
- **Action torque bounds of +-0.5 N*m** are a deliberate comfort limit, not physics. The mixer could deliver about
  4.6 N*m, but inertia is tiny (0.02 kg*m^2), so 0.5 N*m already gives 25 rad/s^2.
- **Wind and mass randomization exist but are disabled:** `reset()` sets `wind = 0` and `mass_scale = 1`.
- `render_mode` is accepted but ignored. The PyBullet GUI was removed, and the visual paths are Isaac Sim and the
  web viewer (`guidance/serve_run.py`).
- `reset()` keeps a `target_pos` that was set directly on the env if none is passed.

## SubprocVecBaseDroneEnv
- Uses `spawn`, not `fork`, because the main process has already initialised CUDA, and forking after that is unsafe.
- Reward closures can't be pickled, so each worker rebuilds its envs from plain data: a module-level
  `reward_fn_factory(p, **kwargs)`, a param dict, and target pairs.
- Each episode's PID teacher is swapped to the per-distance gain set (`best_pid_gains_per_dist.json`). One generic
  gain set is a much worse teacher across 3-250 m (for example kp_pos 10.98 at 3 m vs 3.09 at 50 m).
- `step_with_pid_actions` returns the PID action for the pre-step state, used for imitation pairs.
  `get_target_positions` returns a live snapshot of each env's target, start distance and current distance.
- Always call `close()`. Garbage collection does not stop the worker processes.

## Depends on
`gymnasium`, `numpy`, `scipy`, `torch`, Isaac Lab (Isaac env only). Internally: `dynamics`, `control.pid`,
`reward_functions.rewards`.
