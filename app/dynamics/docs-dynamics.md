# dynamics

## Purpose
Core physics of the project - holds the drone's physical structure and force calculations.

## Methods/Exposes

### `drone.py` - drone structure

- **`Vector3D`** - represents a 3D vector (x, y, z) as a class.
- **`Quaternion`** - represents orientation as a quaternion (used for angle-change calculations).
- **`RotorConfig`** - represents a single rotor:
  - position relative to the drone's center of mass
  - spin direction (`+1` = CW / `-1` = CCW)
  - `k_f` - thrust coefficient (`F = k_f * omega^2`)
  - `k_m` - torque coefficient (`M = k_m * omega^2`)
  - `max_rpm`
  - `motor_tau` - spin-up/down time constant (seconds)
- **`QuadConfig`** - static per-drone-type config:
  - `mass`
  - `inertia` (Ixx, Iyy, Izz)
  - `arm_length`
  - `drag_coeff`
  - list of 4 `RotorConfig`, in X configuration
- **`QuadState`** - dynamic state that changes every simulation tick:
  - `position`, `velocity` (world frame)
  - `orientation` (quaternion)
  - `angular_velocity` (p, q, r in body frame)
  - `rotor_rpm` - current actual rpm per rotor
- **`create_quad_rotors`** - builds the 4 `RotorConfig`s: places them 90° apart in an X layout, assigns alternating spin direction so reaction torques cancel at hover, and derives `k_f` (and `k_m` from it) so 4 rotors balance the drone's weight at the given hover RPM fraction.
- **`create_quad_config`** - assembles a full `QuadConfig` from raw parameters (mass, inertia, arm length, drag coefficient, motor specs), calling `create_quad_rotors` internally.
- **`create_initial_state`** - creates the drone's `QuadState` at spawn: given position, at rest, level orientation, zero rotor rpm (unless overridden).

### `methods.py` - force calculations

- **`mixer`** - given rotor speeds, returns the resulting `[thrust, roll, pitch, yaw]` produced on the body (forward direction: speeds -> body forces/moments).
- **`mixer_inversion`** - the inverse: given desired `[thrust, roll, pitch, yaw]`, solves for the rotor speeds needed to produce them.
- **`motor_lag`** - applies a first-order lag (via `motor_tau`) so a rotor's actual speed approaches its target speed gradually instead of instantly, like a real motor spinning up/down.
- **`thrust`** - force produced by a single rotor along its body z-axis: `F = k_f * w^2`.
- **`torque`** - reaction torque produced by a single rotor about the vertical axis (opposes its spin direction): `M = k_m * w^2 * spin_dir`.
- **`net_combining_thrust`** - sums the thrust of all 4 rotors into one net thrust value.
- **`net_combining_torque`** - sums, per rotor, the moment-arm contribution (`position x thrust`, giving roll/pitch) plus the reaction torque (`torque()`, giving yaw), into one net torque vector.
- **`angular_acceleration`** - net torque divided by inertia, per axis.
- **`update_angular_velocity`** - integrates angular acceleration over `dt` to get the new angular velocity.
- **`drag_force`** - force opposing the drone's current velocity direction, scaled by speed², drag coefficient, cross-sectional area and air density.
- **`wind`** - force from an external wind vector, scaled by drone mass and a wind coefficient.
- **`timestamp_update`** - the main per-tick integrator: takes the RL action and wind, runs it through `mixer_inversion` -> `motor_lag` -> thrust/torque/drag/wind/gravity -> linear and angular acceleration, and returns the next `QuadState` (position, velocity, orientation, angular velocity, rotor rpm).

## Depends on
`numpy` and `scipy.spatial.transform.Rotation` (external). Internally, `methods.py` imports the dataclasses (`QuadConfig`, `QuadState`, `Vector3D`, `Quaternion`) from `drone.py`.

## Notes
The use of quaternions is justified because the regular angle change may cause data loss on some certain angles in which some forces do one anothers job,
for example when the angle on the y axis=90 then the yaw=roll because the pitch becomes 0.
