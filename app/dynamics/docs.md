# dynamics

## Purpose
The physics engine. Holds the drone's physical description and the only code allowed to change its true state.
Everything else (envs, controllers, policies) reads that state or sends commands into it.

## Files

| File | What it holds |
|---|---|
| `drone_params.py` | The single source of the airframe numbers (mass, inertia, arm, max rpm, motor lag, drag) and the angle-style action bounds. Every plant, PID and tuner reads it, so numpy and Isaac cannot drift apart. Changing a value means retraining. |
| `drone.py` | Data types: `Vector3D`, `Quaternion`, `RotorConfig`, `QuadConfig`, `QuadState`, and the builders `create_quad_rotors`, `create_quad_config`, `create_initial_state`. |
| `angle_style_methods.py` | Numpy physics, one drone at a time. This is the reference ("oracle") implementation. |
| `angle_style_torch_methods.py` | Batched torch mirror of `angle_style_methods.py` used by the Isaac env. It has no Isaac imports, so it can be tested without Isaac Sim. |
| `acro_style_methods.py` | Acro-style plant: `RateController` (body-rate loop) and `acro_timestamp_update`. Numpy reference. |
| `acro_style_torch_methods.py` | Batched `TorchRateController`, the torch mirror of `acro_style_methods.py`. |

## Core model
- Per rotor: thrust `F_i = k_f * w_i^2` along body z, and reaction torque `M_i = k_m * w_i^2 * spin_dir`.
- **Mixer.** The 4x4 matrix `M` maps the four rotor thrusts to `[thrust, roll, pitch, yaw]`. It is invertible, so a
  command from the policy or the PID turns into four rotor speeds through `mixer_inversion`. A combination the
  rotors cannot produce gives `w^2 < 0`, which is clipped to 0.
- **Motor lag.** Each rotor moves toward its target speed with time constant `motor_tau`, so speed can't jump.
- **Rotation.** The numpy oracle integrates `w_dot = tau / I` per axis and does **not** model the gyroscopic term
  `w x (I w)`. In Isaac, the forces and torques are handed to PhysX, which integrates the full rigid body and does include
  it. That is one reason `diff_test_trajectory.py` shows some divergence, and why the rate loop's `gyro_ff` is off for the
  numpy plant.
- **Airframe numbers** (`drone_params.py`): 1.5 kg, inertia (0.02, 0.02, 0.04) kg m^2, 0.22 m arms, motor lag 0.05 s,
  drag coefficient 0.5 on 0.05 m^2 (about 5 N at 18 m/s, a top speed of roughly 50 m/s at full thrust).
- **Angle action bounds** (`drone_params.angle_action_bounds`): thrust delta from -1x hover (zero thrust) to +2x hover
  (3x weight in total), roll/pitch torque +-1.5 N m, yaw torque +-0.5 N m. These are design limits, not the rotor maximum.
- **Hover balance.** `k_f` is solved so that 4 rotors at `hover_rpm_fraction * max_rpm` exactly cancel gravity.
- `timestamp_update` is the per-tick integrator: action -> mixer inversion -> motor lag -> forces and torques
  (thrust, drag, wind, gravity) -> new `QuadState`.

## Drone styles
Both styles share the same airframe, mixer, motor lag and integrator. They differ only in what the action means.

| Style | Action | Loop |
|---|---|---|
| **angle** (original) | `[thrust_delta, roll_torque, pitch_torque, yaw_torque]` | state -> angle error (PID, outside the plant) -> torque -> `mixer_inversion` -> motor lag -> `timestamp_update` |
| **acro** | `[thrust_delta, p_des, q_des, r_des]` body rates | state -> rate setpoint -> `RateController` (inside the plant, every physics tick) -> torque -> `mixer_inversion` -> motor lag -> `timestamp_update` |

- The acro loop never reads an attitude angle, so there is no tilt clamp and no Euler singularity in the loop: a
  `2*pi` rad/s roll for 1 s is one clean flip (`scripts/isaac_lab/diff_test_acro.py`).
- **Rate gains** come from pole placement on the motor lag, with no search: the closed loop is
  `kp / (motor_tau s^2 + s + kp)`, so `kp = 1 / (4 zeta^2 motor_tau)` (10.2 1/s at `zeta=0.7`, `motor_tau=0.05`).
- **Torque clip** is `TORQUE_MARGIN` (0.5) times the hover torque authority, which leaves headroom for collective
  thrust. Yaw authority is only about 0.29 N*m (`k_m` is 2% of `k_f`), so a large `r_des` saturates.
- `MAX_BODY_RATE` = (6, 6, 3) rad/s sets the acro action bounds.
- In Isaac, the rate loop runs in `_apply_action` (each of the `decimation` substeps, at 240 Hz) on
  `root_ang_vel_b`, so it is faster than the 60 Hz policy. Use `Isaac-Acro-Drone-Direct-v0` / `AcroDroneEnvIsaacCfg`,
  or `BaseDroneEnv(..., drone_style="acro")`.
- A policy is tied to its style: the same four action numbers mean torques for angle and rates for acro, so an angle checkpoint cannot be run
  or scored on the acro plant, or the other way round (why, and how the evals differ: `training/docs.md` "Angle vs acro").
- `gyro_ff` adds `w x (I w)` feedforward. The numpy integrator does not model that term, so it is off by default.

## Edge cases / gotchas
- **Orientation update.** `angular_velocity` is in the body frame (p, q, r), so the per-tick rotation composes on
  the right: `current_rot * delta_rot`. Composing on the left is only correct for a world-frame rate, and it
  diverges as soon as the drone isn't level.
- **Drag** is forced to zero below 0.01 m/s to avoid dividing by a near-zero speed. Both implementations do this.
- **Quaternions over Euler angles.** At pitch = 90 deg, roll and yaw become the same axis (gimbal lock).
  Quaternions have no such singularity.

## Torch mirror
- `angle_style_torch_methods.py` must match `angle_style_methods.py` exactly (the files were renamed from `torch_methods.py` /
  `methods.py` when the acro style was added). `scripts/isaac_lab/diff_test_physics.py` checks this in
  float64. The env runs float32.
- Frames: torque and angular velocity are in the body frame, and linear force and velocity are in the world frame.
  The caller (`base_drone_env_isaac.py`) rotates into whatever frame Isaac's wrench API expects.
- Rotor geometry and coefficients are plain `(4,)` tensors shared by all envs. Every env uses the same
  `QuadConfig`: there is no per-env domain randomization (mass and wind sampling exist but are disabled).

## Depends on
`numpy`, `scipy` (`Rotation`), `torch`. No other `app` package.
