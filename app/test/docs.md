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

## Unit tests (pytest)
Pure torch / numpy, no Isaac needed: `uv run pytest app/test --ignore=app/test/test_config.py` (117 tests). `test_config.py` is
not a pytest file: pytest collects its `test_configuration(config)` helper and errors on the missing `config` fixture.

| File | What it pins down |
|---|---|
| `test_tilt_cone.py` | `app/control/tilt.py`: numpy and torch agree, a diagonal saturates the cone (not 1.41x the cap at the box's corner), an off-diagonal keeps its heading, inside the cone nothing changes, and the numpy PID / torch teacher both use it. |
| `test_workflow_profiles.py` | `app/workflow`: every profile's script exists and every flag is a real trainer flag, `build_cmd` handles dict/list/bool/None, `--set` typing, distinct run dirs, and `collect_results` copying out of the run folder (and failing loudly when it is missing). |
| `test_rc_reward.py` | `compute_reward_rc`: closing and retreat, the potential telescopes, hover and timeout, zone entry then hold, the post-hit window, caught beats hit, attitude limits, zone-timer features, per-axis progress, the line-of-sight term and its bands. |
| `test_acro_reward.py` | `compute_acro_reward`: quaternion tilt, tilt zones, no limit at 180 deg, faster hits pay more, the away penalty's grace, the braking penalty's zone and direction (and that it ignores attitude). |
| `test_acro_reward_v2.py` | `compute_los_term` (acro v2): aligned / sideways / away cost, slow and gated = 0, `los_k 0` off, `cos` mode, unknown override refused. |
| `test_reward_v3.py` | acro v3: the brake-limited target-speed profile and its cap, the crossing speed (`speed_end_speed`) and that it charges the final creep, the speed term paying exactly `speed_leg` per leg / tracking / sideways / away / near-target braking / gate / 0 = off, the cross-track term (saturates at `line_ref`, overshoot along the line is free), the soft attitude ramp to the hard cap and the "180 deg + no margin" no-op, `time_bonus_budget_s` accepted, unknown override refused, and `install_acro_reward_v3` end-to-end on a fake env (charges the slow env only, reset hook, `rew_*` and the `lat`/`short` diagnostics). |
| `test_reward_v4.py` | acro v4: `segment_min_dist` on the step segment, the swept hit (crossing between two samples counts, point-only does not, a reset step cannot, `zone_time > 0` still needs sampled time inside), cross-track per METRE (same total at any speed, linear to `line_ref`), the `line_lead_s` look-ahead (drifting off the line is charged, a stationary offset is not), the terminal path bonus (full straight, half at half slack, 0 at slack / beyond / without a hit / with `path_leg` 0), v4 defaults and overrides, and `install_acro_reward_v4` on a fake env (forces `line_leg` 0, per-metre charge, reset odometer, `r_path` on a swept hit, `rew_path` / `train_hit_speed`). |
| `test_reward_proportion.py` | `proportion.py`: `N(D)` at the 55 m/s limit and its clamps, per-step terms total one leg at 250/500/1000/1500 m, terminals are the leg values, miss at the radius edge, attempt beats giving up, `0 = off`, `auto_scale=False` unchanged, overrides. |
| `test_distance_buckets.py` | `distance_buckets.py`: the default `--buckets` spec parses to the three weighted ranges, optional weights and whitespace, `none`/`off`/`0`/empty disable it, malformed specs are refused, the summary normalises weights, and the time-bonus clock is `distance / speed` clamped to the leg window (0 speed = keep the fixed clock). |
| `test_threats_layout.py` | `layout_from_obs_dim` (23 + 7K [+2]), the sphere spawn shell and speed range, the numpy threat env's observation size and release wait. |
| `test_carrot_pursuit.py` | `CarrotController`: flies straight on the spawn line, steers back toward the target when off it. |
| `test_widen_checkpoint.py` | `load_bc_checkpoint`: a narrower checkpoint loads into a wider model (zero-padded inputs); a wider one is refused. |
