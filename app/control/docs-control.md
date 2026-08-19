# control

## Purpose
The classical PID controller used two ways: as a scriptable hover baseline to compare RL against, and as a teacher whose (state, action) pairs seed the RL policy via behavior cloning before PPO ever takes over.

## Methods/Exposes

### `pid_hover.py` - the controller itself

- **`PIDHoverController`** - cascaded P/D hover controller (no integral term).
  - `__init__` - stores `kp_pos`/`kd_pos` (outer position loop), 
  - `kp_att`/`kd_att` (inner attitude loop), `kp_yaw`/`kd_yaw`, \
  - and `max_tilt_rad` (tilt clamp, default `0.3` rad).


 - `compute_action` - given `drone_state` and `target_pos`, returns `[thrust_delta, roll_torque, pitch_torque, yaw_torque]`:
    1. outer loop: `accel_cmd = kp_pos*pos_err - kd_pos*vel` (desired acceleration).
    2. converts the x/y components of `accel_cmd` into desired tilt angles and clips them to `max_tilt_rad`. Roll tilts the drone in `y`, pitch tilts it in `x` (confirmed empirically against `timestamp_update` - a positive roll produces negative `y`-velocity, a positive pitch produces positive `x`-velocity), so `des_roll` is driven by `-accel_cmd[1]` and `des_pitch` by `accel_cmd[0]`.
    3. inner loop: `roll_torque`/`pitch_torque` = `kp_att*(desired - current) - kd_att*ang_vel`; `yaw_torque` drives yaw to `0`.
    4. `thrust_delta = kp_pos*pos_err[2] - kd_pos*vel[2]` (z is P/D-controlled directly, no separate "z gain" - reuses the position-loop gains).

### `tune_pid.py` - finds the best gains (Optuna search)

- **`evaluate_pid`** - runs a given `PIDHoverController` for `n_episodes` (each starting from a random `±0.2` offset from the `(0,0,5)` target, not exactly at target - so the search actually scores recovery behavior, not just standing still) and returns the mean total episode reward.
- **`objective`** - Optuna trial function: samples `kp_pos, kd_pos, kp_att, kd_att, kp_yaw, kd_yaw` from fixed ranges, builds a `PIDHoverController`, scores it via `evaluate_pid(n_episodes=10, n_steps=750)`.
- `__main__` - runs a 300-trial `optuna.create_study(direction="maximize")`, prints `study.best_params`, and writes them to `app/control/best_pid_gains.json`.

### `verify_pid.py` - smoke test

Loads the hardcoded `best_params`, builds a `PIDHoverController` + env, resets **exactly at target** (no offset), and runs up to 750 steps printing `dist`/`reason` every 30 steps. Only proves the controller holds a zero-error hover - it does not exercise recovery from an offset (that's what `tune_pid.py`'s `evaluate_pid` and `collect_demonstrations.py` are for).

### `collect_demonstrations.py` - builds the BC dataset

- **`collect_demonstrations`** - runs the given `pid` for `n_episodes` (each starting from a random `±0.05` offset from `(0,0,5)`), recording every `(obs, action)` pair the PID produces, and saves them to `demonstrations.npz` (`obs`, `actions` arrays).
- `__main__` - builds a `PIDHoverController` from a hardcoded `best_params` dict (should match `best_pid_gains.json`) and calls `collect_demonstrations(n_episodes=50, n_steps=600)`.

### `pretrain_bc.py` - behavior cloning

- **`pretrain_behavior_cloning`** - loads `demonstrations.npz`, then for `epochs` runs mini-batch MSE regression of the `ActorCritic`'s predicted action mean against the PID's recorded actions (standard behavior cloning, not RL - no reward/advantage involved).
- `__main__` - builds a fresh `ActorCritic`, pretrains it on `app/control/demonstrations.npz`, saves weights to `app/control/pretrained_bc.pt` - this is what `phase_0_training.py` loads as the PPO policy's starting point.

## Depends on
`numpy`, `scipy.spatial.transform.Rotation`, `optuna` (`tune_pid.py` only), `torch` (`pretrain_bc.py` only, via `app.guidance.train.ActorCritic`). Internally: `app.environmental.interceptor_drone.InterceptorDroneEnv`, `app.reward_functions.rewards`.

## Notes
- `best_pid_gains.json` (used by `InterceptorDroneEnv` as `env.pid_teacher`, and by `phase_imitation_fn` in `rewards.py`) and the hardcoded `best_params` dicts in `verify_pid.py`/`collect_demonstrations.py`/`tune_pid.py`'s trailing comment must be kept in sync manually - nothing enforces it. If you re-tune, update all of them, then regenerate `demonstrations.npz` and `pretrained_bc.pt` in order (`tune_pid.py` -> `verify_pid.py` -> `collect_demonstrations.py` -> `pretrain_bc.py`).
- No integral term anywhere in the loop - by design the controller relies on `kp`/`kd` alone, so persistent tracking bias is expected under nonlinear/coupled dynamics; that's a controller limitation, not a bug.
