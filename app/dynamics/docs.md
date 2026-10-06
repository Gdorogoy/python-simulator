# dynamics

## Purpose
The physics engine. Holds the drone's physical description and the only code allowed to change its true state.
Everything else (envs, controllers, policies) reads that state or sends commands into it.

## Files

| File | What it holds |
|---|---|
| `drone.py` | Data types: `Vector3D`, `Quaternion`, `RotorConfig`, `QuadConfig`, `QuadState`, and the builders `create_quad_rotors`, `create_quad_config`, `create_initial_state`. |
| `methods.py` | Numpy physics, one drone at a time. This is the reference ("oracle") implementation. |
| `torch_methods.py` | Batched torch mirror of `methods.py` used by the Isaac env. It has no Isaac imports, so it can be tested without Isaac Sim. |

## Core model
- Per rotor: thrust `F_i = k_f * w_i^2` along body z, and reaction torque `M_i = k_m * w_i^2 * spin_dir`.
- **Mixer.** The 4x4 matrix `M` maps the four rotor thrusts to `[thrust, roll, pitch, yaw]`. It is invertible, so a
  command from the policy or the PID turns into four rotor speeds through `mixer_inversion`. A combination the
  rotors cannot produce gives `w^2 < 0`, which is clipped to 0.
- **Motor lag.** Each rotor moves toward its target speed with time constant `motor_tau`, so speed can't jump.
- **Rotation** includes the gyroscopic term `w x (I w)`. Roll and pitch rates are coupled even with zero torque, so
  they can't be treated as independent linear channels.
- **Hover balance.** `k_f` is solved so that 4 rotors at `hover_rpm_fraction * max_rpm` exactly cancel gravity.
- `timestamp_update` is the per-tick integrator: action -> mixer inversion -> motor lag -> forces and torques
  (thrust, drag, wind, gravity) -> new `QuadState`.

## Edge cases / gotchas
- **Orientation update.** `angular_velocity` is in the body frame (p, q, r), so the per-tick rotation composes on
  the right: `current_rot * delta_rot`. Composing on the left is only correct for a world-frame rate, and it
  diverges as soon as the drone isn't level.
- **Drag** is forced to zero below 0.01 m/s to avoid dividing by a near-zero speed. Both implementations do this.
- **Quaternions over Euler angles.** At pitch = 90 deg, roll and yaw become the same axis (gimbal lock).
  Quaternions have no such singularity.

## Torch mirror
- `torch_methods.py` must match `methods.py` exactly. `scripts/isaac_lab/diff_test_physics.py` checks this in
  float64. The env runs float32.
- Frames: torque and angular velocity are in the body frame, and linear force and velocity are in the world frame.
  The caller (`base_drone_env_isaac.py`) rotates into whatever frame Isaac's wrench API expects.
- Rotor geometry and coefficients are plain `(4,)` tensors shared by all envs. Every env uses the same
  `QuadConfig`: there is no per-env domain randomization (mass and wind sampling exist but are disabled).

## Depends on
`numpy`, `scipy` (`Rotation`), `torch`. No other `app` package.
