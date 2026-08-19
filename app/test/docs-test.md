# test

## Purpose
Validates that a given `QuadConfig` is physically sane before spending time training/tuning against it - catches unit mismatches, mixer degeneracy, uncalibrated hover RPM, and instability up front instead of discovering them as NaN/crashes deep into a training run.

## Methods/Exposes

### `test_config.py`

- `_pass(name)` / `_fail(name, detail)` / `_warn(name, detail)` - uniform `[PASS]`/`[FAIL]`/`[WARN]` print helpers used by every check below.
- `check_static_fields(config)` - sanity on raw numbers: mass/inertia/arm_length/drag_coeff positivity (mass also warned if outside `0.3-5.0kg`), and per-rotor `k_f`/`k_m`/`max_rpm`/`motor_tau` positivity + `spin_dir in (1,-1)`.
- `check_mixer_invertibility(config)` - builds the same 4x4 mixer matrix as `dynamics.methods.mixer`/`mixer_inversion`, checks its **condition number** (not raw determinant - `k_f`/`k_m` are naturally tiny, ~1e-7, so a small determinant is a scale artifact, not degeneracy) is finite and `<= 1e12`.
- `check_hover_equilibrium(config)` - confirms the RPM needed to hover (`sqrt((mass*9.81/4)/k_f)`) lands in a sane `10-90%` fraction of `max_rpm`, i.e. `k_f` is actually calibrated against `max_rpm` and not left at a disconnected default.
- `check_mixer_roundtrip(config)` - `mixer_inversion(desired) -> speeds -> mixer(speeds**2)` should approximately recover the original `[thrust, roll, pitch, yaw]`, within a loose tolerance.
- `check_hover_stability(config, n_steps=480, dt=1/240)` - runs pure hover (zero commanded torque, pre-spun rotors) for 2 sim-seconds via `timestamp_update`; fails on NaN/Inf, falling through the ground, or `>0.5m` xy drift / `>0.3m` altitude drift.
- `check_step_response(config, n_steps=120, dt=1/240)` - applies a small constant roll torque and confirms the drone rolls in a bounded way (no NaN, `|roll| <= 90°`) rather than exploding under an active command.
- `check_saturation(config)` - warns if hover RPM leaves less than 15% headroom to `max_rpm` on any rotor - not enough room left for an agent to actually maneuver.
- `test_configuration(config, verbose=True)` - runs all 7 checks above in order, prints a summary table, returns `True` only if every check passed.
- `test()` - builds the project's standard `QuadConfig` (mass=1.5, arm_length=0.22, etc. - matches `interceptor_drone.py`'s `reset()`) and runs `test_configuration` against it.

## Depends on
`numpy`, `scipy.spatial.transform.Rotation`. Internally: `app.dynamics.drone.create_quad_config`/`QuadState`/`Vector3D`/`Quaternion`/`QuadConfig`, `app.dynamics.methods` (`mixer`, `mixer_inversion`, `timestamp_update`, `thrust`, `torque`).

## Notes
- Every check here targets a real bug this project actually hit historically (per the module docstring): units mismatches, mixer inversion issues, yaw/body-frame bugs, NaN from bad hover equilibrium, and the "falls before motors spin up" bug (fixed in `check_hover_stability` by pre-spinning rotors to `hover_omega` in the initial `QuadState` instead of starting from zero RPM).
- Intended to run before every training session and any time `QuadConfig` parameters change - see the module docstring's usage snippet (`test_configuration(config)` -> `raise SystemExit` if it fails).
