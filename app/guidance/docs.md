# guidance

## Purpose
The learning side and the tools for looking at its results. It holds:
- the policy network and the PPO core;
- checkpoint replay and recording, with the local web viewer;
- training plots, ONNX export, and MLflow helpers.

## Files

| File | What it holds |
|---|---|
| `train.py` | `ActorCritic`, `RolloutBuffer`, `compute_gae`, `ppo_update`, single/vec/Isaac PPO loops, critic warm-up, `load_bc_checkpoint`. |
| `record_run.py` | Replays a checkpoint in the numpy env and renders mp4 (single policy or two-phase). Also evaluates and ranks checkpoints. |
| `serve_run.py` | FastAPI app behind the web viewer. |
| `viewer.html` | Viewer page: checkpoint picker, two-phase controls, 3D trajectory, logs. |
| `plotting.py` | Every training/eval plot (`plot_training_run`, grad-norm views, eval matrix, DAgger, per-worker). |
| `export_onnx.py` | `.pt` -> `.onnx`, with the architecture inferred from tensor shapes. |
| `mlflow_utils.py` | `start_run`, `log_params_safe`, `log_metrics_safe`. |
| `utils.py` | `compute_grade`: one scalar score per checkpoint. |

## ActorCritic
- **Trunk:** `num_hidden_layers` Linear layers with Tanh, width `hidden`. Production uses 64 x 4. It has an
  `actor_mean` head, a state-independent `actor_log_std`, and a `critic_head`.
- **std** is bounded: `log_std = min + (max - min) * (tanh(p) + 1) / 2`.
- **Actions** are sampled in raw space, squashed with tanh, and rescaled into `[action_low, action_high]`. These are
  buffers, so they're saved in the checkpoint. `log_prob` includes the tanh + affine change-of-variables correction,
  so it's the density of the action the env actually receives.
- `detach_critic` stops value gradients from reaching the trunk. It's a runtime flag and isn't saved.
- `load_bc_checkpoint` matches `shared.*` Linear layers **by position**, because an inserted Dropout layer shifts
  the raw Sequential indices. It raises an error on a real shape mismatch.

## PPO
- `ppo_update` returns the last epoch's averaged stats (policy/value/entropy loss, approx_kl, grad_norm,
  early_stopped). Two extras, both off by default: PPO2-style value clipping (`value_clip_eps`) and teacher
  distillation (`distill_target_actions`, `distill_coef`), which adds `coef * MSE(student, teacher)`.
- **Truncation bootstrap.** A time-limit cutoff isn't a real terminal state. The missing future value is added to
  the reward using `terminal_observation`, because envs auto-reset before returning, so `next_obs` already belongs
  to the next episode.
- `isaac_ppo_train` reuses the same network, GAE and update with GPU rollouts. Pass the previous call's `final_obs`
  as `initial_obs`: resetting on every call corrupts reward-shaping state and randomises `episode_length_buf`
  (`info/PROJECT_DEFENSE_GUIDE.md` Part 8.5.1). `skip_update` collects rollouts with no update at all, for a
  "settle" period. Freezing with `requires_grad` can't do this, because backward fails with every parameter frozen.
- `adaptive_kl_lr_step` (rl_games style) shrinks the lr when `approx_kl > 2 * target` and grows it when
  `approx_kl < target / 2`.
- `warmup_critic` / `isaac_warmup_critic` train only the critic while the actor is frozen. A BC-loaded actor
  starts with a random critic, and untrained-critic advantage noise can wreck a good warm start.

## record_run
- Replays a checkpoint **deterministically** (mean action) in the numpy `BaseDroneEnv`, with no Isaac needed, and
  renders a matplotlib 3D mp4 (at most 360 frames, 24 fps).
- The architecture is inferred from the checkpoint's tensor shapes, so any training script's `.pt` loads.
  `latest_full.pt` (from `rtl_train_isaac`) contains the critic and optimizer and is not an inference checkpoint.
- Modes:

```bash
python -m app.guidance.record_run ckpt.pt --out runs/demo.mp4 --distance-low 3 --distance-high 10
python -m app.guidance.record_run ckpt.pt --evaluate --n-scenarios 75            # stats only
python -m app.guidance.record_run --rank-dir runs/<run> --top-n 5                # leaderboard by grade
python -m app.guidance.record_run ckpt.pt --two-phase --distance 200 --switch-dist 20 --cruise-speed 18
```

- **Two-phase** flags: `--handoff-speed` (default cruise/2), `--recentre-z` (full-mode policies only).
- With `trace_every > 0`, two-phase runs log one line per simulated second: phase, distance, speed, closing and
  lateral speed, tilt, and action. The final line carries a closest-approach summary, and `info["diag"]` holds the
  same numbers. This separates a near-miss from a parked policy or one that never arrives.
- `MAX_STEPS_CAP` is a wall-clock safety cap: `steps_for_dist(5000 m)`, about 1.25M steps. A 5 km episode takes
  minutes of real time.

## Web viewer (`serve_run.py` + `viewer.html`)
- Run with `uv run uvicorn app.guidance.serve_run:app --port 8000` and open http://localhost:8000.
- It is a local dev tool: no authentication, and synchronous (a request blocks while the episode runs).
- Checkpoints are listed from `runs/` and `app/control/` (`best.pt` first, then newest). Anything else, including
  `stable_models/`, can be **uploaded** from the page.
- API:

| Endpoint | What it does |
|---|---|
| `GET /api/checkpoints` | List of checkpoints. |
| `GET /api/checkpoint_info` | Stored run config for a checkpoint. |
| `POST /api/record` | Single-policy mp4. |
| `POST /api/evaluate` | Single-policy stats. |
| `POST /api/rank` | Leaderboard for a directory. |
| `POST /api/two_phase/record` | Two-phase mp4. |
| `POST /api/two_phase/trajectory` | Two-phase 3D trajectory data. |
| `POST /api/two_phase/evaluate` | Two-phase stats. |

- Videos and JSON summaries go to `recordings/` (gitignored). For security, a picked checkpoint path must be one of
  the listed ones.

## Plots
- `plot_training_run` draws `training_error`, `policy_std`, `distance_distribution`, `success_and_outcomes` and
  `vs_pid_baseline`. The PID line is the PID on the **same target pairs**: a high RL line next to a low PID line
  means the task is solvable and the gap is the policy's.
- The 3D grad-norm "landscape" is an interpolated surface. Only the drawn path is real data.
- Sustained-checkpoint ranking accepts a checkpoint only if its trailing window passes both the average and the
  minimum grade. This filters out lucky spikes.

## MLflow
- The store is `MLFLOW_TRACKING_URI`, or `sqlite:///mlflow.db` if that's unset. Isaac runs write
  `mlflow_isaac.db`; see the README.
- When several processes first touch a fresh sqlite store, they race to create the schema. Create the experiment
  once before starting workers.

## Depends on
`torch`, `numpy`, `matplotlib`, `imageio[ffmpeg]`, `mlflow`, `onnx`, `fastapi` and `uvicorn` (viewer only).
Internally: `environmental`, `control` (`two_phase`, `step_budget`), `reward_functions.rewards`, `training.eval_matrix`.
