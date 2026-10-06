# training

## Purpose
Training entry points and the evaluation plumbing around them. **The current trainer is `rtl_train_isaac.py`.**
`rtl-handoff.md` is its full guide: model, PPO phases, PID teacher, reward, handoff stage, how to read the logs, and run
history.

## Files

| File | What it holds |
|---|---|
| `rtl_train_isaac.py` | **Main trainer.** PPO with a separate critic, PID-teacher distillation with an adaptive guard, phased unfreezing, the controller->policy handoff stage, in-Isaac eval, MLflow and plots. |
| `rtl-handoff.md` | Guide for the trainer above. |
| `base_training_isaac.py` | Earlier Isaac stage trainer (`train()`, PID-residual mode). The RTL trainer still imports its imitation stage, kinematics helper and JSON metric log. `train()` itself is only used by `deprecated/training/train_isaac.py`. |
| `config.py` | Shared PPO / imitation hyperparameters (`PARAMS`, imitation and entropy constants). |
| `diagnostics.py` | `diagnose_with_model` / `diagnose_with_pid`: outcome counts on fixed pairs in the numpy env. |
| `eval_matrix.py` | Target-pair builders (axis, omni lattice, random omni, uniform range) and `run_eval_matrix`. |

## Concepts
- **Step:** one policy decision. The policy runs at 60 Hz, with 4 physics substeps at 240 Hz.
- **Chunk:** `--num-steps-per-chunk` (256) steps of **every** env, followed by one PPO update. One chunk is
  `256 * num_envs` samples. Episodes don't line up with chunk boundaries.
- **`--total_timesteps`** counts samples across all envs. With 8192 envs, the default 128M gives about 61 chunks. Every
  chunk-count setting (warmup, eval cadence, decay) does **not** scale with `num_envs`.
- **Phases** (`PHASES`): `settle` (rollout only) -> `critic_warmup` (actor frozen) -> `head_only` (actor head + log_std)
  -> `full`.

## Running
See the README "Training" section for the exact commands. Every run writes `runs/<checkpoint-dir>/` (`best.pt`,
`latest_full.pt` for resume, `model_<steps>.pt` / `.onnx`, `metrics.csv`, `run_config.json`, plots) and logs to the
MLflow store `mlflow_isaac.db`.

Useful flags (all have `--help` text):

| Flag | What it does |
|---|---|
| `--handoff`, `--max-speed`, `--spawn-dist-low/high` | Handoff stage: speed buckets, and the carrot leg flown from spawn. |
| `--distance-low/high` | The policy's own range: handoff distance `h`. |
| `--obs-position-mode {full,height}` | `height` removes absolute x/y. Use it for new runs. |
| `--bc-checkpoint`, `--anchor-checkpoint` | Start weights, and the frozen policy used as the distillation anchor. |
| `--distill-start/end/decay-chunks`, `--guard-*` | Teacher-loss schedule and the eval-driven guard that boosts it. |
| `--settle/warmup/head-chunks`, `--lr-actor/critic` | Phase lengths and learning rates. |
| `--eval-every`, `--eval-burnin`, `--pid-eval-every` | In-Isaac evaluation cadence. |
| `--reward-json` | Override any `RewardQV2Config` field, from a file or inline JSON. |

## base_training_isaac
These notes describe `train()`, the earlier stage trainer.
- **Imitation -> critic warmup -> PPO**, in one call on a pre-built env. `IMITATION_FRACTION` is 0.08; it was 0.15,
  which is 19.2M steps of pure imitation at 128M.
- **`actor_log_std` is reset after imitation.** BC's Gaussian NLL against a deterministic teacher drives std to the
  floor. Resetting only after loading the checkpoint didn't survive the on-policy imitation stage, which collapsed it
  again. `tanh(0)` puts std at the midpoint of its range, where PPO still has gradient.
- **PID-residual mode** (`residual_scale > 0`, `PidResidualEnv`): `env_action = pid + residual_scale * policy`. A policy
  that outputs about 0 *is* the PID, so PPO starts at PID performance, and drift and exploration are bounded by the
  scale. `actor_mean` is zeroed unless `residual_resume` is set, because a direct-policy head used as a correction would
  double the PID. Imitation is skipped.
- **Floors, not fractions.** `MIN_WARMUP_CHUNKS = 16` was raised from 8 after a run still collapsed post-unfreeze.
  `MIN_IMITATION_RETRAIN_ROUNDS = 8` keeps a large `num_envs` from shrinking the retrain rounds.
- **Schedules.** `training_progress_at` is 0 through warmup, then 0 -> 1 over PPO only. Previously, entropy hit its
  floor about 10 chunks into warmup and the LR had decayed 14% before unfreeze. The LR is flat through warmup and then
  cosine over the *full* warmup+PPO span: a training-only cosine ran about 45% higher LR mid-run and drifted more (mean
  grade 0.648 vs 0.731 over chunks 12-40).
- **Soft reset.** When the trailing grade drops `RESET_DEGRADE_MARGIN` below the best, the weights blend back toward the
  best checkpoint and the optimizer is reinitialised. That is followed by `RESET_WARMUP_CHUNKS` of critic-only training.
  A margin of 0.25 fired 4 resets in `base_training_isaac_3_10_v1`, each recovering only to 0.75-0.83 while pulling the
  critic back. 0.4 makes it a collapse-only safety net.
- **Promotion gate.** `SOLID_WINDOW` consecutive chunks, all at index `>= SOLID_MIN_CHUNK` (40, which counts warmup),
  must average at least `SOLID_GRADE` with none below the window minimum. The gate then writes `SOLID.json`. The index
  floor stops a checkpoint that is just the imitation policy (about 0.95 grade) from being promoted as if PPO had
  produced it.
- `N_DIAGNOSTIC_EPISODES = 30` here, because 10 was noisy enough to false-trigger the soft reset.

## config.py
- `PARAMS['dropout']` **must stay 0.** PPO's ratio and KL need old and new log-probs from the *same* deterministic
  function. With dropout on, every forward pass draws a new mask, and one run showed approx_kl of about 29 during
  critic warmup, while the actor was frozen and couldn't move at all.
- `gamma` comes from `rewards.GAMMA`, because potential-based shaping is only policy-invariant if the shaping gamma
  equals the PPO gamma.

## diagnostics / eval_matrix
- Outcome keys are the reward functions' exact reason strings (`"Hit"`). Compound reasons (`"attitude-ROLL+attitude-PITCH"`)
  are split, and each part is credited separately. `+` is also not allowed in MLflow metric names.
- `diagnose_with_pid` swaps in per-distance gains when `gains_by_dist` is given. One fixed gain set would understate
  the PID ceiling.
- `build_uniform_omni_eval_pairs` draws the distance once and only re-draws the **direction** when the target would be
  underground. Re-drawing both under-samples the far end of the range, because underground directions are more likely
  at long distances.
- The omni lattice scales each axis offset by `d / sqrt(len(axes))`, so the total distance is exactly `d`. The random
  omni builder covers skewed directions the lattice never produces.

## Depends on
`torch`, `numpy`, `mlflow`, Isaac Lab (trainers only). Internally: every other `app` package.
