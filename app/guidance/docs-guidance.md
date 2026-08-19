# guidance

## Purpose
The backbone-brain of the whole project that makes the interceptor learn

## Methods/Exposes

### `train.py` - actor-critic + PPO core

- `device` - `"cpu"` (CUDA line present but commented out - see Notes).
- `ActorCritic` - shared actor-critic network (two hidden Tanh layers, ReLU on their output).
  - `__init__` - builds the shared trunk, the actor-mean head, a learned per-action `actor_log_std` parameter, and the critic (value) head. Registers `action_low`/`action_high` as buffers (saved/loaded with the model) so actions can be rescaled into the env's action box without relying on `env.step`'s clipping.
  - `forward` - runs `obs` through the shared trunk; `log_std` is `actor_log_std` squashed through `tanh` and rescaled into `[LOG_STD_MIN, LOG_STD_MAX] = [-3.0, 0.0]` (not a hard clamp - clamping stopped learning); returns `(action_mean, action_std, state_value)`.
  - `scale_action(raw_action)` - squashes an unbounded raw action through `tanh` and rescales into `[action_low, action_high]`; used for deterministic (eval) actions.
  - `get_action_and_value(obs, raw_action=None)` - builds `Normal(mean, std)`; samples a raw (pre-squash) action if none given, otherwise re-scores the given one. Returns `(scaled_action, raw_action, log_prob, entropy, value)`, where `log_prob` includes the tanh+affine change-of-variables correction so it's the density of the action actually sent to the env, not of the pre-squash Gaussian.
- `RolloutBuffer` - fixed-size storage for one rollout's worth of transitions.
  - `__init__` - preallocates zero tensors for `obs`, `actions` (raw, pre-squash), `log_probs`, `rewards`, `values`, `dones`, plus a write pointer `ptr`.
  - `add` - writes one timestep's transition at `ptr`, advances it.
- `compute_gae(rewards, values, dones, last_value, gamma, lam)` - generalized advantage estimation, computed backwards through the rollout; returns `(advantages, returns)`.
- `ppo_update(model, optimizer, buffer, advantages, returns, ...)` - the PPO clipped-objective update: normalizes advantages, then for `num_epochs` over shuffled minibatches computes clipped policy loss, value loss, and entropy bonus, steps the optimizer, and early-stops the remaining epochs on this batch if `approx_kl` exceeds `target_kl`. Returns the last executed epoch's averaged stats dict (`policy_loss`, `value_loss`, `entropy_loss`, `approx_kl`, `grad_norm`, `early_stopped`).
- `ppo_train(env, total_timesteps, num_steps, gamma, lam, lr, target_kl, model=None, optimizer=None, ent_coef=0.015, global_timesteps_offset=0, global_total_timesteps=None)` - main training loop: repeatedly collects a `num_steps` rollout with the current policy, computes GAE, updates via `ppo_update`, until `total_timesteps` is reached. Accepts an existing `model`/`optimizer` to resume from (what makes `../training/phase_0_training.py`'s chunked/checkpointed training work instead of restarting each chunk); `model=None` builds a fresh one. Clamps `actor_log_std` to `[-2.0, -0.5]` after each update. Returns `(model, optimizer, episode_rewards, last_losses)`.
- `evaluate(model, env, n_episodes)` - runs `n_episodes` deterministically (distribution mean via `scale_action`, not a sampled action) and prints the fraction of episodes that reached a reward `>= 15.0`.

### `../training/phase_0_training.py` - the actual training entry point

- `best_params` - hardcoded reward/PPO hyperparameters found by `optuna_search.py`'s best trial.
- `TrainConfig` - run-level constants: total timesteps (1,000,000), rollout length (4096), `gamma`/`lam`, base LR (from `best_params`), checkpoint cadence/dir/CSV path, number of diagnostic episodes per checkpoint.
- `compute_target_pos(start_position, distance, angle_deg, y_offset)` - polar-offset target placement helper (currently unused in `train()` - target is hardcoded to `(0,0,5)`; kept for other placement experiments).
- `diagnose_with_model(model, env, n_episodes)` - runs `n_episodes` deterministically, tallies outcome reasons (`oob`, `attitude-ROLL/PITCH`, `hit`, `hover_success`, `moving_away_cap`, `drift`, `timeout`) and avg steps survived; used by `optuna_search.py`.
- `save_checkpoint(model, timesteps_done, cfg)` - saves `model.state_dict()` to `<CHECKPOINT_DIR>/ppo_stage2_<timesteps>.pt`.
- `log_metrics(env, model, episode_rewards, timesteps_done, ...)` - runs `n_diagnostic_episodes` deterministically, computes the same outcome tally as `diagnose_with_model` plus `avg_final_dist`/`min_final_dist`/`std_final_dist`/`max_hover_streak`/`effective_std_mean`/`total_param_norm`/reward stats, and appends one flat row to `METRICS_CSV` (creating it with a header on first call) - this CSV is what `plotting.py` reads.
- `train(cfg)` - builds the chained `RewardConfig` (imitation phase for 150k steps -> phase0 for 200k steps -> base indefinitely, per `chain_reward_fns`), builds the env with target fixed at `(0,0,5)`, builds an `ActorCritic` and **loads `app/control/pretrained_bc.pt` as its starting weights** (the actual "imitate the PID" hookup - PPO fine-tunes from the BC-cloned policy, not from scratch), then loops in `CHECKPOINT_EVERY_TIMESTEPS`-sized chunks calling `ppo_train` (resuming the same model/optimizer each chunk), decaying entropy coefficient/LR/`target_kl` linearly with progress, checkpointing and logging metrics after each chunk. Plots the full run via `plot_training_run` at the end. Returns the trained model.
- `__main__` - runs `train(TrainConfig())`, then `evaluate(model, InterceptorDroneEnv(), n_episodes=10)`.

### `optuna_search.py` - hyperparameter search over `phase_0_training`'s knobs

- `build_env(trial)` - builds a `RewardConfig` with several fields sampled from the trial (`streak_penalty_coef`, `streak_cap`, `phase0_pos_coef`, `tilt_penalty_coef`, `ang_vel_penalty_coef`, `imitation_coef`), fixed `imitation_duration_steps=phase0_duration_steps=20_000` (short - this is a search budget, not a full run), returns the env.
- `objective(trial)` - samples PPO hyperparameters (`lr`, `gamma`, `lam`, `ent_coef`, `target_kl`), builds an `ActorCritic` (loading `pretrained_bc.pt` if present), trains it for `SEARCH_TIMESTEPS=60_000` via `ppo_train`, scores via `diagnose_with_model` over `N_EVAL_EPISODES=20`. Returns `success_rate + 0.001*mean_reward` (success rate is the real objective; reward only breaks ties).
- `run_search(n_trials=N_TRIALS)` - runs `optuna.create_study(direction="maximize", sampler=TPESampler())`, prints the best trial, saves `study.trials_dataframe()` to `runs/optuna/search_<timestamp>.csv`.
- `__main__` - `python -m app.guidance.optuna_search [n_trials]`, defaults `N_TRIALS=7`.
- Trials run sequentially by design - a network this small doesn't get real throughput gains from parallel GPU trials contending for the same device.

### `plotting.py` - post-run diagnostics/plots

- `compute_pid_baseline(pid_gains_path, n_episodes, ...)` - runs the tuned PID (see `app.control`) through the same env/reward machinery used to score the RL policy, so a checkpoint can be compared against it apples-to-apples. Returns `None` (printing why) if the gains file is missing, rather than raising.
- `_streak_windows(values, threshold, min_len=STREAK_LEN)` / `_hover_ratios` / `_shade_streak_windows` - identify and shade spans where the diagnostic hover-success ratio held at or above a tier (`STREAK_TIERS = (0.25, 0.50, 0.75, 1.00)`) for `STREAK_LEN=5`+ consecutive checkpoints in a row - a single lucky checkpoint doesn't count as convergence.
- `plot_training_run(csv_path, output_dir="plots_final", ...)` - reads `METRICS_CSV` and writes 5 figures: `training_error.png` (policy/entropy/value loss + grad norm), `policy_std.png` (exploration decay), `distance_distribution.png` (final-distance mean/std/min band vs. `hit_threshold`), `success_and_outcomes.png` (hover-success rate with streak shading + stacked outcome-reason plot), and `vs_pid_baseline.png` (RL final checkpoint vs. `compute_pid_baseline`, 4-metric bar comparison). Also prints a convergence summary via `_print_convergence_summary`.
- `_print_convergence_summary(rows, hover_success_steps, n_diag_episodes, hit_threshold)` - two-part textual check: (1) a snapshot pass/fail table for the last checkpoint only, (2) the real signal - which hover-success-rate tier (if any) has been held for `STREAK_LEN`+ consecutive checkpoints.
- `__main__` - `python -m app.guidance.plotting [csv_path] [output_dir]`, defaults to `runs/1m_10epochs_v2/metrics.csv` / `plots_final`.

### `utils.py`

- `calc_drone_state(drone_state_arr, n=10)` - averages `position`/`velocity`/`orientation`/`rotor_rpm` across the last `n` entries of a `QuadState` list (or fewer, if not enough collected yet), returns the averaged dict plus `n_averaged`. Used by `phase_0_training.train()` to log a rolling snapshot of recent drone states.

### `watch_hover.py` - visually replay a checkpoint

- `main()` - loads a checkpoint (`sys.argv[1]`, default `runs/1m_10epochs_v4/ppo_stage1_1005000.pt`), builds the env with `render_mode="human"` (opens the PyBullet GUI) and the **same** `RewardConfig` used in `phase_0_training.train()` so termination reasons match what the checkpoint was trained under, runs `n_episodes` (`sys.argv[2]`, default 5) deterministically, prints steps survived + termination reason per episode.
- `python -m app.guidance.watch_hover [checkpoint_path] [n_episodes]`

### `test_free_hover.py` - diagnostic: does it hold past the success threshold?

- `main()` - loads a checkpoint (`sys.argv[1]`, default `runs/1m_10epochs_v2/ppo_stage2_660000.pt`), spawns it exactly at the target `(0,0,5)`, and disables the `hover_success` early-termination (`hover_success_steps` set absurdly high, `oob_radius`/`streak_cap` widened) so the episode keeps running under only the attitude/oob/moving-away failure conditions. Prints `dist`/`reason` every 20 steps for up to `max_steps` (default effectively unlimited). Answers "does the policy actually hold hover, or did it only learn to survive exactly ~200 steps and then fall apart?"
- `python -m app.guidance.test_free_hover [checkpoint_path] [max_steps]` (note: `max_steps` is accepted on the command line description but not actually read from `sys.argv[2]` in the current code - it's hardcoded to `10000000`)

## Depends on
`torch`, `numpy`, `optuna` (`optuna_search.py`), `matplotlib` (`plotting.py`, backend forced to `Agg`). Internally: `app.environmental.interceptor_drone.InterceptorDroneEnv`, `app.reward_functions.rewards`, `app.control.pid_hover`/`pretrained_bc.pt`/`best_pid_gains.json`.

## Notes
- `device` is hardcoded to `"cpu"` - the CUDA-selecting line is present but commented out.
- The BC-pretraining -> PPO handoff happens in exactly one place: `phase_0_training.train()`'s `model.load_state_dict(torch.load("app/control/pretrained_bc.pt"))` call, before the PPO loop starts. If `app/control/pretrain_bc.py` hasn't been re-run since a `pid_hover.py` change, this loads a stale imitation target.
- `training-goals.md` (same directory) has the higher-level curriculum/design rationale this module implements.
