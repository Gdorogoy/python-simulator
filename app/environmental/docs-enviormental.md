# environmental

## Purpose
The glue between the pure physics engine (`dynamics`) and the outside world: spawns/visualizes the drone in PyBullet, samples wind/mass-scale disturbances, and wraps everything as a Gymnasium environment (`InterceptorDroneEnv`) that RL/PID code drives.

## Methods/Exposes

### `enviorment.py`

- `spawn_drone` - creates a PyBullet multi-body matching the drone's mass/inertia (base body + 4 cosmetic rotor spheres as fixed child links) and returns its `body_id`.
- `sample_wind_conditions(np_rand)` - samples a random `wind_vector` (`±2.0` per axis) and `mass_scale` (`0.92-1.08`) from the given RNG.
- `apply_rotor_thrust` / `apply_rotor_torque` - apply the mixer's combined thrust/torque directly to a PyBullet body via `p.applyExternalForce`/`applyExternalTorque`. Still unused (see Notes) - `InterceptorDroneEnv` advances state through `timestamp_update` (pure physics, `dynamics/methods.py`) and only calls PyBullet for rendering, not for force application.

### `interceptor_drone.py`

- `build_observation(state, target_pos)` - maps a `QuadState` + target into the 21-dim observation vector: normalized position, velocity, raw orientation quaternion, normalized angular velocity, normalized rotor RPMs, normalized relative-to-target vector, and normalized scalar distance (see `POS_SCALE`/`VEL_SCALE`/`ANG_VEL_SCALE`/`MAX_RPM`/`DIST_SCALE` constants at the top of the file).
- `InterceptorDroneEnv(gym.Env)`
  - `__init__(custom_reward, render_mode=None, pid_gains_path="app/control/best_pid_gains.json")` - stores `dt=1/240`, `max_steps=5000`, the reward function; calls `reset()` once to establish `self.config` before building `action_space`/`observation_space` off it; loads `PIDHoverController` from `pid_gains_path` into `self.pid_teacher` (used by `reward_functions.phase_imitation_fn`) - warns and leaves `pid_teacher=None` if the file is missing/unparseable.
  - `reset(start_pos=None, target_pos=None, seed=None, options=None)` - defaults both to `(0,0,5)`; resets episode-tracking state (`moving_away_streak`, `hover_steps_in_zone`, `last_raw_action`, `hover_success_achieved`, `prev_distance`); builds a fresh `QuadConfig` each reset (`mass_scale` currently hardcoded to `1`, wind currently hardcoded to `[0,0,0]` - `sample_wind_conditions` is defined but not called here, see Notes); spawns the drone pre-set at exact hover RPM (`mixer_inversion`) at `start_pos`.
  - `step(action)` - clips `action` to the action space, adds `hover_thrust` back onto `action[0]` (the policy outputs a thrust *delta* around hover, not absolute thrust), advances state via `dynamics.methods.timestamp_update`, computes `(reward, terminated, reason)` via `self.reward_method(self)`, `truncated` on `steps_elapsed >= max_steps`, calls `render()`.
  - `render()` - no-op unless `render_mode == "human"`; otherwise syncs the PyBullet body's position/orientation to `self.drone_state` and steps the PyBullet sim for visualization.
  - `_get_obs()` - `build_observation(self.drone_state, self.target_pos)`.
  - `_init_render()` - (human mode only) connects PyBullet GUI, sets up the ground plane/camera, initializes `drone_id`/`target_marker_id` to `None`.
  - `_spawn_visuals()` - (human mode only) (re)spawns the drone body and a red sphere marking `target_pos`, removing any previous ones first.
- `my_check_env()` - reset-determinism smoke check (same seed twice -> same `wind_vector`/`mass_scale`) plus `gymnasium.utils.env_checker.check_env` against the registered `interceptor_drone_v0` env.
- `my_test()` - samples random actions for 5000 steps, resetting on `terminated`/`truncated`, printing reward periodically - checks the env doesn't crash/hang, not that a policy can learn.
- `__main__` - interactive menu (`1` = `my_test()`, `2` = `my_check_env()`).

## Depends on
`pybullet`, `gymnasium`, `numpy`, `scipy`. Internally: `app.dynamics.drone`, `app.dynamics.methods` (`mixer_inversion`, `timestamp_update`), `app.environmental.enviorment` (used by `interceptor_drone.py`), `app.control.pid_hover.PIDHoverController`, `app.reward_functions.rewards`.

## Notes
- `apply_rotor_thrust`/`apply_rotor_torque` remain unused - force/torque application happens through the pure-physics `timestamp_update` path, not through PyBullet's force API; these exist for a possible future PyBullet-as-physics-engine mode.
- `reset()` hardcodes `mass_scale=1` and `wind_vector=[0,0,0]`, with the `sample_wind_conditions(np_rand=self.np_random)` call commented out directly above - domain randomization is wired up but currently disabled.
- `render()` is fully implemented (not a stub) - only active when `render_mode="human"`.
