# Current numpy simulation flow (baseline, pre-Isaac Lab)

Documents the existing per-step pipeline as of the migration start, so the Isaac
Lab port has a ground truth to diff-test against. Source of truth: `app/dynamics/methods.py`,
`app/dynamics/drone.py`, `app/environmental/base_drone_env.py`, `app/reward_functions/rewards.py`,
`app/guidance/train.py`.

## 1. Entities (`app/dynamics/drone.py`)

- `RotorConfig`: static per-motor — position (rel. to COM), spin_dir (+1 CW/-1 CCW),
  k_f (thrust coeff), k_m (torque coeff), max_rpm, motor_tau (spin-up/down time const).
- `QuadConfig`: static per-drone — mass, inertia (Ixx,Iyy,Izz), arm_length, drag_coeff, 4 rotors in X config.
- `QuadState`: dynamic per-tick — position, velocity, orientation (quat xyzw), angular_velocity (body-frame p,q,r), rotor_rpm[4].
- `create_quad_rotors`: places 4 rotors at 45/135/225/315 deg, alternating spin_dir so hover yaw torque cancels; solves k_f so hover_rpm_fraction*max_rpm balances gravity.

## 2. Per-step pipeline (`BaseDroneEnv.step` -> `timestamp_update`)

1. **Action clip**: policy action clipped to `action_space` = `[-hover_thrust, -0.5, -0.5, -0.5]..[hover_thrust, 0.5, 0.5, 0.5]` (thrust delta, roll/pitch/yaw torque cmd).
2. **Hover offset**: `real_action[0] = action[0] + hover_thrust` (policy learns thrust *delta* around hover, not absolute thrust).
3. **Mixer inversion** (`mixer_inversion`): desired `[thrust, roll, pitch, yaw]` -> per-rotor target speeds via `inv(M) @ desired`, clipped >=0, sqrt'd (M built from each rotor's k_f/k_m/position).
4. **Motor lag** (`motor_lag`, per rotor): first-order lag toward target rpm — `w_new = w_current + ((w_target - w_current)/motor_tau) * dt`. This is what actually integrates rotor speed each step (mixer output is a *target*, not the applied speed).
5. **Forces/torques**:
   - `net_combining_thrust`: sum `k_f * w_i^2` over 4 rotors -> scalar body-z thrust.
   - `net_combining_torque`: sum of moment-arm cross products (`r_i x [0,0,F_i]`) + reaction torque (`k_m * w_i^2 * spin_dir_i`) per rotor -> 3-vector.
   - `drag_force`: opposes velocity, `0.5 * air_dens * speed^2 * drag_coeff * cross_sec_area` (air_dens=1.225, cross_sec_area=0.05 hardcoded, not in QuadConfig).
   - `wind`: `wind_vector[i] * mass * k_wind_coeff` (k_wind_coeff=0.1 hardcoded); currently disabled in `BaseDroneEnv.reset` (`self.wind_vector=[0,0,0]`, `sample_wind_conditions` call commented out).
   - `gravity_force = [0,0,-mass*9.81]`.
   - thrust rotated body->world via current orientation quaternion (`scipy Rotation.apply`).
   - `total_force = thrust_world + drag + wind + gravity`.
6. **Integrate linear**: `linear_accel = total_force/mass`; semi-implicit Euler — `new_velocity = velocity + accel*dt`; `new_position = position + new_velocity*dt`.
7. **Integrate angular**: `alpha = torque / inertia` (per-axis); `new_angular_velocity = angular_velocity + alpha*dt`; incremental rotation `delta_rot = Rotation.from_rotvec(new_angular_velocity*dt)` composed as `new_rot = current_rot * delta_rot` (right-composition — angular_velocity is body-frame, so this is required for correctness away from near-identity orientation).
8. **New QuadState** assembled: position, velocity, orientation (quat), angular_velocity, rotor_rpm = w_actual (post-lag).
9. `dt = 1/240` fixed, `max_steps` (default 15000) is the episode step budget — **this is the value the Isaac Lab port must keep fixed and control simulation speed via decimation, not by enlarging dt.**

## 3. Observation (`build_observation`, 23-dim)

`symlog(position, POS_SCALE=15)` (3) + `velocity/VEL_SCALE=10` (3) + `orientation quat xyzw` (4) +
`angular_velocity/ANG_VEL_SCALE=20` (3) + `rotor_rpm/MAX_RPM=12000` (4) + `symlog(target_pos - position, DIST_SCALE=15)` (3) +
`symlog(dist)` (1) + `[sin(yaw_err), cos(yaw_err)]` (2) = 23. `symlog`: linear near zero (`x/linthresh`), log beyond it, keeps large/small distances on comparable scale.

## 4. Reward (`app/reward_functions/rewards.py`)

Two independent reward systems currently live side by side:

- **`RewardConfig`/`make_reward_fn`/`base_reward_fn`**: class-based, config-driven curriculum
  (optional `phase_imitation_fn` chained into `base_reward_fn` via `chain_reward_fns` for N steps,
  then permanently switches). Terminal checks (`_terminal_checks`): NaN/z<0/oob_radius, roll/pitch
  attitude limits, optional drift_radius. Hit check (`_check_hit`): dist < hit_threshold. In-zone
  (`dist < outer_dist`): blended `pos_term`/`approach_term` + stability penalties (vel/dist/tilt/ang_vel,
  each capped) + one-time hover_success bonus. Out-of-zone: potential-style `diff = prev_distance - dist`
  progress term, moving-away streak penalty/cap (terminates on cap).
- **`reward_func`** (flat, module-constant-driven, used by current training runs): terminal checks
  (`terminal_checks`, oob_radius scaled by `max(30, start_dist*3)`), hit check (`hit_target`, threshold 0.25),
  potential-based shaping `phi_now - phi_prev` where `phi = -L1(target-pos)/start_dist` (plain difference,
  NOT `gamma*phi_now - phi_prev` — see in-file comment on why GAMMA there left a stand-still reward residual),
  flat step_penalty scaled by `steps_for_dist(start_dist)`, one-time `milestone_bonus` at 25/50/75% progress
  (`env.milestones_hit` set, reset in `BaseDroneEnv.reset`).

Both read/write env-mutated state (`env.prev_distance`, `env.moving_away_streak`, `env.hover_steps_in_zone`,
`env.milestones_hit`, `env.prev_position`) — any batched port must vectorize this per-env state, not just the physics.

## 5. Episode boundary (`BaseDroneEnv.step`/`reset`)

`terminated` from reward fn (oob/attitude/hit/drift/streak_cap); `truncated = steps_elapsed >= max_steps`.
`reset()`: optional `target_pairs` sampling (start/target/yaw triples) else random offset ranges; rebuilds
`QuadConfig` fresh every episode (mass=1.5, inertia=(0.02,0.02,0.04), arm_length=0.22, drag_coeff=0.035,
max_rpm=12000, motor_tau=0.05 — currently fixed, `mass_scale`/wind randomization present but disabled);
spawns at hover rpm (`mixer_inversion([hover_thrust,0,0,0])`), zeroes velocity/angular_velocity.

## 6. Vectorization (`VecBaseDroneEnv`, `SubprocVecBaseDroneEnv`)

Pure Python: `VecBaseDroneEnv` loops N `BaseDroneEnv` instances per step (no batching — GPU never
sees a batch bigger than what the policy forward pass gets). `SubprocVecBaseDroneEnv` shards envs
across `spawn`-context worker processes (physics-only parallelism, CPU-bound: ~92 env-steps/sec/core
is the measured bottleneck, not NN size). Both auto-reset a done env before returning its obs, and
stash the true terminal obs in `info["terminal_observation"]` for truncation bootstrapping.
**This is the layer Isaac Lab's native GPU-batched (`NUM_ENVS`) stepping replaces.**

## 7. Buffer / GAE / PPO (`app/guidance/train.py`)

- `RolloutBuffer`: flat `(num_steps, ...)` tensors — obs, actions (pre-squash raw), log_probs, rewards, values, dones.
- `ActorCritic`: shared MLP trunk (Linear+Tanh(+Dropout) x num_hidden_layers) -> actor_mean + learned
  actor_log_std (clamped via tanh into `[log_std_min, log_std_max]`) + critic_head. Actions squashed via
  tanh and rescaled into `[action_low, action_high]`; log_prob correction applied for the tanh+affine
  change-of-variables.
- `compute_gae`: standard backward-recursion GAE; also handles the `(num_steps, num_envs)` batched case via broadcasting.
- `ppo_update`: clipped surrogate objective + value loss + entropy bonus, minibatched over `num_epochs`,
  early-stops epochs once `approx_kl > target_kl`.
- `ppo_train` (single env) vs `vec_ppo_train` (batched over `vec_env.num_envs`, truncation-bootstraps via
  `info["terminal_observation"]`): the Isaac Lab port's PPO must be batched natively across `NUM_ENVS`
  the way `vec_ppo_train` already is, generalized to GPU-resident tensors instead of a numpy vec-env loop.
- `warmup_critic`: freezes actor, trains only `critic_head` for N rounds before real PPO starts (post BC/DAgger load).

## 8. Constants that must survive the port unchanged

- `dt = 1/240` (physics fixed step) — speed via decimation, never by growing dt.
- `GAMMA = 0.97` (`app/reward_functions/rewards.py`) — shared by PPO's own gamma and reward-shaping context (though `reward_func`'s shaping itself intentionally does NOT multiply by GAMMA).
- Observation scales: `POS_SCALE=15, VEL_SCALE=10, ANG_VEL_SCALE=20, MAX_RPM=12000, DIST_SCALE=15`.
- Reward constants: `OOB_RADIUS=30, HIT_THRESHOLD=0.25, HIT_REWARD=50, ATTITUDE_ROLL_DEG=65, ATTITUDE_PITCH_DEG=80`, milestone fracs/bonuses `(0.25,0.5,0.75)`/`(10,15,20)`.
- Drone physical params (`BaseDroneEnv.reset`): mass=1.5, inertia=(0.02,0.02,0.04), arm_length=0.22, drag_coeff=0.035, max_rpm=12000, motor_tau=0.05, kf_km_ratio=0.02 (default), hover_rpm_fraction=0.5 (default).
