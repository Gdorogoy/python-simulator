# navigation

## Purpose
State estimation. It reads the true state from `dynamics` (read-only) and produces an estimate with sensor noise,
delay and drift added. This is a one-way transform; it never updates the physics.

**Status:** standalone. It is **not** wired into either env's observation yet: the policy currently sees the exact
state and the exact target vector. It is the starting point for the planned server-feed and camera target estimator
(see `info/alt_model.md` section 4.2).

## Files

| File | What it holds |
|---|---|
| `kalmans.py` | Linear Kalman filter and a self-test (`test()`, `calc_diff`). |

## Model
- State `[x, y, z, vx, vy, vz]` with a constant-velocity motion model `F`. Only position is measured (`H` is 3x6).
- `Q` and `R` are scalar process and measurement noise, expanded to identity matrices.
- There is no control input (`B`, `u`): RL actions aren't known accelerations here.

## Run
```bash
uv run python -m app.navigation.kalmans
```
Prints estimated vs true state and the per-component error.
