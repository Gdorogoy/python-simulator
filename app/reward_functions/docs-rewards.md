# reward_functions

## Purpose
Builds the reward function handed to `InterceptorDroneEnv` - shared termination checks plus a curriculum of three reward shapes (imitation -> phase 0 -> base) that can be chained to activate in order as training progresses.

## Methods/Exposes

### `rewards.py`

- **`RewardConfig`** - all reward tuning knobs in one object: OOB/drift/attitude termination thresholds, hit/attitude/oob reward-penalty values, `streak_cap` (episode-ending moving-away-streak limit) and `streak_penalty_coef`, `inner_dist`/`outer_dist` (the hover "zone" band used to blend `pos_term` and `approach_term`), `tilt_penalty_coef`/`ang_vel_penalty_coef` (stability shaping in `base_fn`), `phase0_pos_coef`/`rpm_penalty_coef` (used by `phase_0_fn`), `imitation_coef` (used by `phase_imitation_fn`), and `phase0_duration_steps`/`imitation_duration_steps` (how many chained-function calls each phase stays active for; `None` means "don't include this phase").
- **`_kinematics(env)`** - pulls `pos`, `vel`, `ang_vel`, `roll`/`pitch`/`yaw` (from the orientation quaternion), and `dist` to target out of `env.drone_state`/`env.target_pos`. Shared by all three reward functions.
- **`_terminal_checks(cfg, env, pos, roll, pitch)`** - hard-failure conditions, checked first by every phase: NaN position / below ground / outside `oob_radius` -> `"oob"`; inside `drift_radius` (if set) -> `"drift"`; roll beyond `attitude_roll_deg` -> `"attitude-ROLL"`; pitch beyond `attitude_pitch_deg` -> `"attitude-PITCH"`. Returns `None` if nothing tripped.
- **`chain_reward_fns(phases)`** - takes a list of `(reward_fn, duration_steps)` tuples (in order, last one's duration should be `None`) and returns one function that counts total calls made to it (across episode resets, not per-episode) and permanently advances to the next phase once its `duration_steps` is exceeded.
- **`make_reward_fn(cfg)`** - builds and returns the actual reward function passed to the env. Internally defines three variants and chains whichever ones have a duration set:
  - **`base_fn`** - the "graduated" reward, active once the curriculum is done (or from the start if no phases are configured). Inside `outer_dist`: tracks `hover_steps_in_zone` (one-time `hit_reward` bonus at `hover_success_steps`, sets `env.hover_success_achieved`), and computes a `stability_term` (rewards low speed/tilt/angular-velocity while close) blended with a distance-shrinking `pos_term` and a distance-delta `approach_term` via `blend = clip((outer_dist-dist)/(outer_dist-inner_dist), 0, 1)`. Outside `outer_dist`: pure `approach_term` off `diff = prev_distance - dist`, with a 1.25x multiplier when moving away vs. plain delta when approaching. Always adds `streak_penalty` (grows with `moving_away_streak`) and a small `closer_bonus`. Terminates with `moving_away_cap` if the streak hits `cfg.streak_cap`.
  - **`phase_0_fn`** - simpler curriculum-start shaping: same in/out-of-zone `approach_term` split as `base_fn`, but additionally penalizes `rpm_deviation` (how far the current rotor RPMs are from the exact hover RPMs for this episode's mass, via `mixer_inversion`) - teaches "hold hover thrust", not just "get close".
  - **`phase_imitation_fn`** - same `approach_term` shaping as `phase_0_fn` (minus the RPM term) plus an `imitation_term = -imitation_coef * ||(last_raw_action - teacher_action) / action_range||`, where `teacher_action` comes from `env.pid_teacher.compute_action(...)` - directly pulls the policy's raw action toward what the PID would have done.
  - Phase order when chained: `phase_imitation_fn` (if `imitation_duration_steps` set) -> `phase_0_fn` (if `phase0_duration_steps` set) -> `base_fn` (always last, runs indefinitely). If only one phase ends up configured, `make_reward_fn` returns it directly (unchained).

## Depends on
`numpy`, `scipy.spatial.transform.Rotation`. Internally: `app.dynamics.methods.mixer_inversion` (for `phase_0_fn`'s RPM term), `app.control.pid_hover.PIDHoverController` (imported but only actually used via `env.pid_teacher`, which the env constructs itself from `best_pid_gains.json`).

## Notes
- All three reward variants call `_terminal_checks` first and return its `(reward, terminated, reason)` unchanged on a hard failure - termination behavior is identical across phases, only the in-episode shaping differs.
- `env.moving_away_streak`, `env.hover_steps_in_zone`, `env.prev_distance`, `env.hover_success_achieved` are mutated directly on the passed-in `env` by these functions - the reward function is stateful with respect to the env object, not pure.
- `phase_imitation_fn` requires `env.pid_teacher` to be non-`None` (i.e. `best_pid_gains.json` must load successfully) - see `InterceptorDroneEnv.__init__`.
