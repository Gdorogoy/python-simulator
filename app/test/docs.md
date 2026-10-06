# test

## Purpose
Physics and configuration validation for a `QuadConfig`. Run it before training and after any change to the airframe
numbers. Each check targets a bug this project has actually hit: unit mismatches, mixer inversion, yaw/body-frame
mix-ups, and NaN from a bad hover equilibrium.

## Run
```bash
uv run python -c "from app.test.test_config import test; test()"
```
Prints PASS/WARN/FAIL per check and ends with `RESULT: 7/7 checks passed`.

## Checks (`test_config.py`)

| Check | What it catches |
|---|---|
| `check_static_fields` | Non-physical mass, inertia, arm length or rotor fields (warns outside typical small-quad ranges). |
| `check_mixer_invertibility` | A near-singular mixer. It uses the **condition number**, not the determinant: `k_f`/`k_m` are about 1e-7, so the determinant is tiny even for a healthy matrix. |
| `check_hover_equilibrium` | Hover rpm outside 10-90% of max. |
| `check_mixer_roundtrip` | `mixer(mixer_inversion(cmd)) != cmd`. |
| `check_hover_stability` | 2 s of pure hover must stay put. This is the test that would have caught the "falls before motors spin up" bug. |
| `check_step_response` | A small constant roll torque must give a bounded roll, with no NaN or blow-up. |
| `check_saturation` | Enough rotor headroom above hover. |

`test_configuration(config)` runs them all and returns `True` only if every check passes. `test()` runs them on the
project's standard config.
