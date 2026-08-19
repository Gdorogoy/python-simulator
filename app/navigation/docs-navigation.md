# navigation

## Purpose
A standalone linear Kalman filter that estimates position+velocity from noisy position-only measurements. Read-only with respect to `dynamics`/`environmental` - not currently wired into `InterceptorDroneEnv`'s observation pipeline, it's a self-contained estimator exercised by its own `test()`.

## Methods/Exposes

### `kalmans.py`

- **`Kalman`** - constant-velocity, 6-state (`[px, py, pz, vx, vy, vz]`) linear Kalman filter; position-only measurement (`H` selects the first 3 states).
  - `__init__(position, velocity, q, r, dt)` - sets `x_hat` (initial state), `Q = I*q` (process noise), `R = I*r` (measurement noise), `P = I` (initial covariance), `H` (measurement matrix), and `F` (constant-velocity state-transition matrix built from `dt`). Prints every matrix on construction via `display`.
  - `estimate_state(F, x_hat_prev)` - `x_hat_pred = F @ x_hat_prev`.
  - `estimated_state_covariance(F, P_prev, Q)` - `P_pred = F @ P_prev @ F.T + Q`.
  - `kalman_gain(P_est, H, R)` - `K = P_est @ H.T @ inv(H @ P_est @ H.T + R)`.
  - `update_estimated_state(est_x_hat, K, z, H)` - `x_hat = est_x_hat + K @ (z - H @ est_x_hat)`.
  - `update_estimated_state_covariance(K, H, P_est)` - `P = (I - K@H) @ P_est`.
  - `predict()` - runs the predict step, stores `est_x_hat`/`P_est` on `self`.
  - `update(z)` - runs the correction step using the `est_x_hat`/`P_est` stashed by the last `predict()` call, overwrites `self.x_hat`/`self.P`.
  - `loop(measurements)` - `predict()` + `update(z)` for each `z` in a measurement sequence, prints the final estimate.
  - `display(name, val)` - debug print helper, called throughout `__init__`/`predict`/`update`.
- **`test()`** - simulates a drone moving at constant velocity, adds Gaussian noise (`var=0.34`) to the position to fake sensor readings, runs it through `Kalman.loop`-equivalent predict/update calls, and prints the final estimate vs. ground truth.
- **`calc_diff(expected, actual)`** - per-state `(relative_error, accuracy_ratio, absolute_error)`, used by `test()` to report filter accuracy. Returns `nan` for relative/accuracy on any state whose expected value is ~0 (division-by-zero guard).

## Depends on
`numpy` only.

## Notes
- `predict()` must be called before `update()` - `update()` reads `self.est_x_hat`/`self.P_est`, which only exist after a `predict()` call has run (not set in `__init__`). `loop()` always calls them in the right order; calling `update()` standalone first will raise `AttributeError`.
- Assumes a constant-velocity model with no control input (`B`/`u` from the docstring's math notes are omitted entirely from the actual `F`/`estimate_state` - there's no acceleration term, matching the docstring's "u=0 in the RL case because the acceleration is unknown").
- Not currently imported by `environmental` or `guidance` - `build_observation` in `interceptor_drone.py` uses ground-truth `drone_state` directly, not a `Kalman`-filtered estimate.
