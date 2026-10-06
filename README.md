# python-simulator: quadcopter interception with RL

A quadcopter has to **hit a target**: get within 0.25 m of it. The drone flies on custom rigid-body physics, a
classical PID controller acts as both teacher and baseline, and PPO policies are trained on thousands of drones in
parallel in **NVIDIA Isaac Lab**.

The deployed system is **two-phase**:
1. a controller flies the long midcourse leg at cruise speed;
2. at the **handoff point**, a learned policy takes over for the terminal phase: braking, aiming, hitting, and turning
   around if it missed.

```
spawn ──── PID / carrot cruise (≤ 18 m/s) ────► handoff (3-50 m out) ── RL policy ──► hit (< 0.25 m)
```

## Highlights
- **Own physics engine** (`app/dynamics`): rotor thrust and torque, mixer inversion, motor lag, drag, wind,
  gyroscopic coupling. A numpy reference implementation plus a batched torch mirror, diff-tested to 1e-9.
- **Closed-form PID tuning**: pole placement per target distance, with no search.
- **Imitation pipeline**: PID demonstrations -> behaviour cloning -> DAgger.
- **RTL trainer**: PPO with a decaying PID-teacher distillation loss, an eval-driven teacher guard, phased unfreezing
  and a realistic controller->policy handoff stage. Runs on 8192 envs on one GPU.
- **Web viewer** to fly any checkpoint (single policy or two-phase) and inspect the 3D trajectory, plus **MLflow** for
  every run.

## Results (released models in `stable_models/`)

| Model | Task | Eval hit rate |
|---|---|---|
| `handoff_v3_150_250_30_50_18ms` | Two-phase: 150-250 m spawn, handoff at 3-50 m, up to 18 m/s, `height` obs | **0.95** |
| `handoff_v5_3_50m_18` | Handoff stage at 3-50 m, up to 18 m/s (no long leg) | 0.91 |
| `3_10m_v2_res01` | Earlier PID-residual policy, 3-10 m from rest | grade 0.92 |
| `3_30m_v1_res01` | Earlier PID-residual policy, 3-30 m from rest | grade 0.85 |

Hit rates are the trainer's deterministic in-Isaac evaluation on fresh random targets (`BEST.json`). Grades are the
older composite score (`SOLID.json`).

## Repository layout

```
app/
  dynamics/          physics engine (numpy oracle + torch mirror)
  environmental/     numpy Gymnasium env + Isaac Lab env
  control/           PID, gain tuning, demos / BC / DAgger, two-phase agent
  reward_functions/  reward_qv2 (current) + termination geometry
  guidance/          ActorCritic + PPO, replay/recording, web viewer, plots
  training/          rtl_train_isaac.py (current trainer) + eval tools
  navigation/        Kalman estimator (standalone)
  test/              physics/config validation
stable_models/       released checkpoints (best.pt, .onnx, metrics, plots)
scripts/             numpy <-> torch/Isaac parity tests
deprecated/          retired trainers, rewards and migration scripts
info/                design notes, project defense guide, migration log
```

Every `app/<folder>/docs.md` explains that folder's design, gotchas and history. Start with [`app/docs.md`](app/docs.md).
The trainer's full guide is [`app/training/rtl-handoff.md`](app/training/rtl-handoff.md).

## Requirements
The project uses **two Python environments**:

| Environment | Python | Used for |
|---|---|---|
| **uv env** (`.venv`) | 3.12 | Web viewer, replay/recording, BC, tests, plots. No Isaac needed. |
| **Isaac Lab venv** | 3.11 | Anything that simulates in Isaac: demo collection, DAgger, training, the MLflow UI for Isaac runs. |

- **uv env:** install [uv](https://docs.astral.sh/uv/), then run `uv sync` in the repo root. PyTorch comes from the
  CUDA 13.0 index configured in `pyproject.toml`.
- **Isaac Lab:** Isaac Sim + Isaac Lab (developed on Isaac Lab 2.3.2, Windows, RTX GPU). Install `mlflow` into that
  venv too. Call the venv's `python.exe` directly; `isaaclab.bat -p` can pick the wrong interpreter.

The commands below write that interpreter as `<isaac-python>`. On the development machine it's
`E:\Isaac\env_isaaclab\Scripts\python.exe`, or `/mnt/e/Isaac/env_isaaclab/Scripts/python.exe` from WSL.

**Always run commands from the repository root.** Modules import as `app.*`.

## Quick start (no Isaac needed)

```bash
uv sync
uv run python -c "from app.test.test_config import test; test()"     # physics sanity: 7/7 checks
uv run uvicorn app.guidance.serve_run:app --port 8000                  # web viewer
```

Open **http://localhost:8000**.

## Web viewer
The viewer flies a checkpoint **deterministically in the numpy simulator**, with no Isaac needed. Start it with
`uv run uvicorn app.guidance.serve_run:app --port 8000`. The server lists every `.pt` under `runs/` and
`app/control/`, with `best.pt` first. To make a released model selectable, copy its **whole folder** so its
`run_config.json` comes along:

```bash
mkdir -p runs && cp -r stable_models/handoff_v3_150_250_30_50_18ms runs/
```

| Tab | What it does |
|---|---|
| **Two-phase** | The PID / carrot cruise flies to the switch distance, then the checkpoint takes over. Pick a listed checkpoint, or **Upload .pt** (an upload has no `run_config.json`, so it is treated as a `full`-mode model). Set the target distance, switch distance, cruise speed (use the run's `--max-speed`) and handoff speed. **Run & play in 3D** gives an interactive replay (orbit, zoom, scrub, follow); **Run test batch** gives the hit rate over N random scenarios. |
| **Single policy** | The policy flies the whole episode from rest and returns an mp4. It uses server-listed checkpoints only. |
| **Rank checkpoints** | Evaluates every `.pt` in a directory and shows the top ones by grade. |

Keep the switch distance inside the range the model was trained on (3-50 m for the released handoff models). The observation
mode (`full`/`height`) is read from the `run_config.json` next to the checkpoint. Without one, a `height` model would
be replayed with the wrong inputs. Videos and summaries
are written to `recordings/`. The viewer is a local dev tool: it has no authentication, and each request blocks until
the episode is done.

The same functionality is available from the command line:

```bash
uv run python -m app.guidance.record_run stable_models/handoff_v5_3_50m_18/best.pt --distance-low 3 --distance-high 50
uv run python -m app.guidance.record_run stable_models/handoff_v3_150_250_30_50_18ms/best.pt \
    --two-phase --distance 200 --switch-dist 30 --cruise-speed 18 --evaluate --n-scenarios 50
uv run python -m app.guidance.record_run --rank-dir runs/<run> --top-n 5     # leaderboard of a run's checkpoints
```

## MLflow
Isaac training runs log to `mlflow_isaac.db` in the repo root. Open the UI with the **Isaac venv's** mlflow: the db
schema matches that mlflow version.

```bash
<isaac-python> -m mlflow ui --backend-store-uri sqlite:///mlflow_isaac.db
```

Then open **http://localhost:5000**. Older numpy-side runs (deprecated trainers) are in `mlflow.db`; open it with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Training pipeline
Steps 1-3 build the warm-start policy. Step 4 is the real training. Pretrained outputs of steps 1-3 are already
committed (`app/control/pretrained_bc.pt`, `pretrained_bc_dagger.pt`), so you can start at step 4.

```bash
# 1. PID demonstrations (Isaac, batched)
<isaac-python> -u app/control/collect_demo_isaac.py --headless --num_envs 4096 \
    --n_target_rows 2000000 --distance-low 3 --distance-high 10 \
    --save-path app/control/demonstrations_isaac.npz

# 2. Behaviour cloning (uv env)
uv run python -m app.control.pretrain_bc --demo-path app/control/demonstrations_isaac.npz \
    --out-path app/control/pretrained_bc.pt

# 3. DAgger refinement (Isaac)
<isaac-python> -u app/control/dagger_isaac.py --headless --num_envs 4096 --n_rounds 5 \
    --rows_per_round 500000 --distance-low 3 --distance-high 10 \
    --checkpoint-path app/control/pretrained_bc.pt \
    --demo-path app/control/demonstrations_isaac.npz \
    --out-path app/control/pretrained_bc_dagger.pt

# 4. RTL training with the handoff stage (Isaac)
<isaac-python> -u app/training/rtl_train_isaac.py --headless \
    --num_envs 8192 --handoff --max-speed 18 \
    --distance-low 3 --distance-high 50 \
    --spawn-dist-low 150 --spawn-dist-high 250 \
    --obs-position-mode height \
    --bc-checkpoint stable_models/handoff_v3_150_250_30_50_18ms/best.pt \
    --distill-start 0.25 --distill-end 0.05 --init-std 0.1 --warmup-chunks 8 \
    --checkpoint-dir runs/<new_run_name>
```

Outputs go to `runs/<new_run_name>/`: `best.pt`, `latest_full.pt` (resume), `model_*.pt/.onnx`, `metrics.csv`,
`run_config.json`, plots. Runs also log to MLflow. `runs/` is never committed. Copy a good run into `stable_models/` to
release it.

**Sizing a run.** One PPO update ("chunk") uses `256 x num_envs` samples, and `--total_timesteps` (default 128M) counts
samples across all envs. At 8192 envs that is about 61 updates. Chunk-based settings (warmup, eval cadence, teacher
decay) don't scale with `num_envs`, so double `--total_timesteps` when you double `num_envs` to keep the same number of
updates.

Run `--help` on any script for every flag. The trainer's flags and logs are explained in
[`app/training/rtl-handoff.md`](app/training/rtl-handoff.md).

## Tests

```bash
uv run python -c "from app.test.test_config import test; test()"   # physics/config validation
uv run python scripts/isaac_lab/diff_test_physics.py                 # torch physics == numpy oracle
uv run python scripts/isaac_lab/diff_test_pid.py                     # torch PID == numpy PID
<isaac-python> -u scripts/isaac_lab/diff_test_trajectory.py --headless --num_steps 100   # PhysX vs oracle
```

## Known limitations
- The policy observes the **exact** target position. A noisy server-feed / camera estimator is planned
  (`info/alt_model.md` section 4.2; `app/navigation` is the starting point).
- The progress reward is path-independent, so handoff policies may climb before approaching
  (`app/reward_functions/docs.md`).
- Wind and mass randomisation exist but are disabled.
