# control

## Purpose
The classical controllers and the imitation-learning pipeline built on them. The PID has three jobs:
- **teacher** for behaviour cloning and DAgger;
- **baseline** that RL is scored against;
- **midcourse pilot** in the two-phase system, flying the drone until the RL policy takes over.

## Files

| File | What it holds |
|---|---|
| `pid.py` | `PIDController`: cascaded position -> attitude P(I)D (numpy, one drone). |
| `torch_pid.py` | `TorchPIDController`: batched mirror with per-env gains, plus `assign_gains_by_distance`. |
| `tune_pid.py` | Closed-form gains per distance -> `best_pid_gains_per_dist.json` and `best_pid_gains.json`. |
| `step_budget.py` | `steps_for_dist`: episode step budget per distance. |
| `collect_demonstrations.py` | PID (obs, action) datasets (numpy and batched Isaac). |
| `collect_demo_isaac.py` | CLI: pipeline step 1, collect demonstrations in Isaac. |
| `pretrain_bc.py` | Behaviour cloning on a demo dataset (pipeline step 2). |
| `dagger.py` / `dagger_isaac.py` | DAgger refinement (library and Isaac CLI, pipeline step 3). |
| `two_phase.py` | `CarrotController` + `TwoPhaseAgent`: PID cruise, then RL handoff (used by the viewer and `record_run`). |
| `verify_pid.py` | Scores the PID on the fixed evaluation matrix used for RL checkpoints. |
| `best_pid_gains_per_dist.json` | Gains for 3/10/20/30/50/100/150/250 m (generated). |
| `best_pid_gains.json` | The 50 m entry, used as `BaseDroneEnv`'s default teacher (generated). |
| `pretrained_bc.pt`, `pretrained_bc_dagger.pt` | BC and DAgger checkpoints (64 hidden x 4 layers). |
| `pid_and_reward_math.md` | Longer derivations. |

## PID
- **Outer loop:** `accel_cmd = kp_pos * err - kd_pos * vel`. The x/y acceleration becomes a desired tilt (`accel / g`,
  clipped to `max_tilt_rad`). z drives thrust directly.
- The world-frame x/y acceleration is rotated by **-yaw** into the drone's heading frame before mapping to roll and
  pitch. Without this, the mapping is only correct at yaw = 0.
- **Inner loops:** roll, pitch and yaw torque from the attitude error and rate.
- The output is **clipped to the env's action bounds**. `pretrain_bc` maps labels through `atanh` into raw pre-tanh
  space, so an unclipped label would turn into a huge target, not the command the env would actually apply.
- The `ki_*` knobs exist but default to 0.

## PID tuning (`tune_pid.py`)
There is no search: the gains come from pole placement on the real plant (`create_quad_config` numbers).
- **Position loop** (a mass-normalised double integrator): `kp = wn^2`, `kd = 2 * zeta * wn`.
  `wn` comes from the settle-time budget `steps_for_dist(d) * dt * SETTLE_TIME_FRACTION`, using
  `ln(d / HIT_THRESHOLD)` time constants. The usual "4 / (zeta * wn)" rule only gets the error down to 2% of `d`,
  which is 5 m of slack at 250 m. With the fixed 4, the 250 m case ended at 0.073 m, just short of target.
- **Attitude and yaw loops** use the same identity with inertia (`kp = I * wn^2`). They are set to a fixed multiple
  of the position bandwidth (cascade separation). The yaw multiple is lower because yaw authority comes from
  `k_m`, which is only 2% of `k_f`.
- **One gain set per distance.** `max_tilt_rad` caps the commanded tilt, so gains tight enough for 3 m saturate
  and overshoot at 250 m, and loose ones are too slow up close. The 20 m and 30 m entries were added because
  without them nearest-neighbour lookup snapped all of 10-30 m to the 10 m gains.
- **Generic gains** (`best_pid_gains.json`) are the middle of the ladder (50 m), the least-bad single compromise.
- `calibrate_approach_milestone_budget()` measures the tuned PID's best `reward_func` total, with the step penalty
  forced to 0 so the current budget constant doesn't feed into its own measurement. Copy the result into
  `rewards.APPROACH_MILESTONE_BUDGET` by hand.

## Imitation pipeline
1. **Demonstrations:** the PID flies; each (obs, action) pair is recorded. The start position is jittered so the
   dataset also contains corrective, off-path states. Directions cover single, pair and triple axes ("omni"), not
   only the 6 cardinal directions.
2. **BC:** actions map into raw pre-tanh space and are scored with Gaussian NLL, so large-magnitude dimensions don't
   dominate. Optional per-sample weights are supported.
3. **DAgger:** the policy flies, the PID labels the states it visits, the data is aggregated and retrained.

## DAgger
- **Recency weighting.** A pair from `k` rounds ago is kept with probability, and weighted in the loss by,
  `RECENCY_DECAY ** k` (0.85). Recent corrections dominate, while round-0 PID data fades but never disappears.
  The buffer is capped (`AGG_BUFFER_CAP_PAIRS`).
- **Numpy `dagger()`** subsamples every distance to the smallest distance's row count, because long episodes would
  otherwise dominate the loss. Jitter scales with distance (`min(0.75, 0.25 * d)`) and covers all 3 axes.
- **Isaac `dagger_base_drone_isaac()`** has no per-distance balancing. It resets the env **once**, not every
  round: a per-round reset made `hit_rate` read near zero regardless of policy quality
  (`info/PROJECT_DEFENSE_GUIDE.md` Part 8.5.1).
- `max_steps` must be passed per distance. The env default (15k) would silently truncate 100-250 m episodes, which
  need 25k-62.5k steps.

## Two-phase agent (`two_phase.py`)
- **Midcourse.** `CarrotController` (the default) is a numpy port of `rtl_train_isaac`'s carrot controller. It
  tracks velocity along the straight spawn->target line on a trapezoid profile: ramp up to `cruise_speed`, cruise,
  and ramp down to `handoff_speed` by `switch_dist`. It holds altitude on the target z. With `cruise_speed=None`
  it uses the old analytic position-hold PID, with gains solved for the exact distance. That mode arrives at
  about 0.5-1 m/s, a state training never produced.
- **Handoff.** Inside `switch_dist`, only the model acts. Velocity is **snapped** to the trained handoff speed
  along the line to the target, and the observation is rebuilt. That is why `record_run` logs exactly
  `speed = handoff_speed, lateral = 0` at takeover.
- `recentre_z` applies to `full`-mode policies only: after the switch, the policy sees its position relative to
  the handoff point, with z mapped down to `recentre_z` when the handoff is higher. It is ignored in `height` mode.
- Keep `switch_dist` inside the distance range the checkpoint was trained on.
- `MIN_GAIN_DIST` floors the distance used to solve the gains, because `log(d / HIT_THRESHOLD)` breaks below it.
  The real target is never moved.

## Edge cases / gotchas
- `tune_pid.py` solves for 0.05 m precision, which is tighter than the 0.25 m hit radius.
- `collect_demonstrations_base_drone_isaac` uses one episode length for every env. That is fine up to about
  50-60 m; longer distances get truncated.
- Regenerate in order after re-tuning: `tune_pid` -> demonstrations -> BC -> DAgger.

## Depends on
`numpy`, `scipy`, `torch`. Internally: `dynamics`, `environmental`, `reward_functions.rewards`, `guidance.train`
(`ActorCritic`), `training.eval_matrix`.
