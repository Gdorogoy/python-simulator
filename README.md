# python-simulator: quadcopter interception with RL

## Update

- The acro policy now flies the full 1000-1500 m leg in one pass: 0.966 hit rate, 35.1 s (PID 85s , LoS 45s) 37.1 m/s avg on hit, and `--buckets` now trains 100-200 / 500-800 / 1000-1500 m slices with a distance-scaled reward clock.
- Reference notes live in `app/<folder>/docs.md` and `logs_guide.md`.

- starting to teach the drones on not relaying on 2-phase but rather fly from 0 → target.
- angle has 2 version: 1 for interceptor based on LoS chasing him , 1 w/o (evader & non_evader)

- updates that were applied on acro : new reward functions (v3 & v4) adding penalty for non-optimal path and not using enough speed.

A quadcopter has to **hit a target**: get within 1 m of it. The drone flies on custom rigid-body physics, a
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
- **Acro trainer**: a single-pass policy for 250-1500 m legs, thrust + body-rate actions, rewards v1-v4 on a shared
  per-step scaling contract.
- **Web viewer** to fly any checkpoint (two-phase, plus an acro eval tab) and inspect the 3D trajectory, plus **MLflow** for
  every run.

## Results (released models in `stable_models/`)

| Model                           | Task                                                                                                        | Eval hit rate |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------- |
| `handoff_v3_150_250_30_50_18ms` | Two-phase: 150-250 m spawn (can be any dist 150-250 is base), handoff at 3-50 m, up to 18 m/s, `height` obs | **0.95**      |
| `handoff_v5_3_50m_18`           | Handoff stage at 3-50 m, up to 18 m/s (no long leg)                                                         | 0.91          |
| `acro_1000_1500_40ms`           | Acro: one pass over 1000-1500 m, no handoff, 1 m hit radius, no tilt cap, `height` obs                      | **0.966**     |
| `acro_1000_1500_40ms_unstable`  | Same stage after the 25-chunk constant-LR fine-tune; kept to document the failure (peaks 0.960, ends 0.758) | 0.960         |
| `angle_1000_1500_non_evader`    | Angle: one pass over 1000-1500m , no handoff , 1m hit radius , 85 deg tilt cap , `heigth` obs               | 0.95          |
| `angle_600_40ms_non_evader`     | Handoff 60-160 m from a 400-600 m spawn, up to 40 m/s, no threats                                           | 0.964         |
| `anlge_evader_600_40ms`         | Handoff 100-250 m from a 600-800 m spawn, up to 45 m/s, 1 active threat of 5                                | 0.846         |
| `3_10m_v2_res01`                | Earlier PID-residual policy, 3-10 m from rest                                                               | grade 0.92    |
| `3_30m_v1_res01`                | Earlier PID-residual policy, 3-30 m from rest                                                               | grade 0.85    |

Hit rates are the trainer's deterministic in-Isaac evaluation on fresh random targets (`BEST.json`); grades are the
older composite score (`SOLID.json`). The two acro rows are the v4 speed run: 35.1 s and 37.1 m/s mean over the leg,
against a 52.1 m/s airframe limit; the `_unstable` copy is the failed fine-tune, not a candidate to deploy. The angle
rows use `height` obs and the rc reward in the two-phase setup. The `3_10m` / `3_30m` folders predate
`run_config.json` and ship without it.

## Repository layout

```
app/
  dynamics/          physics engine (numpy oracle + torch mirror), angle + acro plants
  environmental/     numpy Gymnasium env + Isaac Lab env, threats, perception
  control/           PID, gain tuning, demos / BC / DAgger, two-phase agent, LOS guidance
  reward_functions/  angle qv2 / rc, acro v1-v4, shared per-leg contract
  guidance/          ActorCritic + PPO, replay/recording, web viewer, plots
  training/          angle_rtl_train_isaac.py, acro_rtl_train_isaac.py + eval tools
  navigation/        Kalman estimator (standalone)
  test/              physics, reward, LOS, threat and checkpoint tests
guide/               the intuition guide, 7 chapters read as a book
stable_models/       released checkpoints (best.pt, .onnx, metrics, plots)
scripts/             numpy <-> torch/Isaac parity tests
future/              roadmap notes
logs_guide.md        how to read the trainer's console line
deprecated/          retired trainers, rewards and migration scripts
```

Every `app/<folder>/docs.md` explains that folder's design, gotchas and history. Start with [`app/docs.md`](app/docs.md).
The trainer's full guide is [`app/training/rtl-handoff.md`](app/training/rtl-handoff.md).

## Requirements

The project uses **two Python environments**:

| Environment          | Python | Used for                                                                                           |
| -------------------- | ------ | -------------------------------------------------------------------------------------------------- |
| **uv env** (`.venv`) | 3.12   | Web viewer, replay/recording, BC, tests, plots. No Isaac needed.                                   |
| **Isaac Lab venv**   | 3.11   | Anything that simulates in Isaac: demo collection, DAgger, training, the MLflow UI for Isaac runs. |

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

| Tab           | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Two-phase** | The PID / carrot cruise flies to the switch distance, then the checkpoint takes over. Pick a listed checkpoint, or **Upload .pt** (an upload has no `run_config.json`, so it is treated as a `full`-mode model). Set the target distance, switch distance, cruise speed (use the run's `--max-speed`) and handoff speed. **Run & play in 3D** gives an interactive replay (orbit, zoom, scrub, follow); **Run test batch** gives the hit rate over N random scenarios.                                                        |
| **Acro eval** | Scores an acro checkpoint the way the trainer's eval does (clean start from rest, deterministic, first episode, per-distance budget) in the numpy plant: hit / out of bounds / tilt / hover / timeout, per scenario and per distance. **Fly one episode & play in 3D** gives the same interactive replay as the two-phase tab. Start "as trained" launches the drone with the run's spawn speed (what the trainer's eval does), "from rest" starts at rest. Use it to check that the viewer plant agrees with the Isaac eval. |

Keep the switch distance inside the range the model was trained on (3-50 m for the released handoff models). The observation
mode (`full`/`height`) is read from the `run_config.json` next to the checkpoint. Without one, a `height` model would
be replayed with the wrong inputs. Videos and summaries
are written to `recordings/`. The viewer is a local dev tool: it has no authentication, and each request blocks until
the episode is done.

The same functionality is available from the command line:

```bash
uv run python -m app.guidance.record_run stable_models/handoff_v3_150_250_30_50_18ms/best.pt \
    --two-phase --distance 200 --switch-dist 30 --cruise-speed 18 --evaluate --n-scenarios 50
uv run python -m app.guidance.record_run runs/<acro run>/best.pt --acro --n-scenarios 75 --distance-low 3 --distance-high 5
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
<isaac-python> -u app/training/angle_rtl_train_isaac.py --headless \
    --num_envs 8192 --handoff --max-speed 18 \
    --distance-low 3 --distance-high 50 \
    --spawn-dist-low 150 --spawn-dist-high 250 \
    --obs-position-mode height \
    --bc-checkpoint stable_models/handoff_v3_150_250_30_50_18ms/best.pt \
    --distill-start 0.25 --distill-end 0.05 --init-std 0.1 --warmup-chunks 8 \
    --checkpoint-dir runs/<new_run_name>
```

Outputs go to `runs/<new_run_name>/`: `best.pt`, `latest_full.pt` (the full training state; there is no `--resume`
flag), `model_*.pt/.onnx`, `metrics.csv`, `run_config.json`, plots. Runs also log to MLflow. `runs/` is never
committed. Copy a good run into `stable_models/` to release it.

**Sizing a run.** One PPO update ("chunk") uses `256 x num_envs` samples, and `--total_timesteps` (default 128M) counts
samples across all envs. At 8192 envs that is about 61 updates. Chunk-based settings (warmup, eval cadence, teacher
decay) don't scale with `num_envs`, so double `--total_timesteps` when you double `num_envs` to keep the same number of
updates.

Run `--help` on any script for every flag. The trainer's flags and logs are explained in
[`app/training/rtl-handoff.md`](app/training/rtl-handoff.md).

### Acro runs (single pass, no handoff)

The acro policy flies the whole leg itself: no cruise controller, no switch distance. The stages lengthen the leg:
250-500 m (s1, s2), 500-1000 m (s3), 1000-1500 m (s4). The exact commands are in `commands.txt`; the reward versions
(`--reward v1` to `v4`) and the flags are in [`app/training/docs.md`](app/training/docs.md), and every console field is
explained in [`logs_guide.md`](logs_guide.md). The v4 run hit 0.966 at 35.1 s with a 37.1 m/s mean, 1.0 m/s faster
than the tuned reward-v3 run. It converges at a ~38 m/s cruise with thrust saturation at only 5-7%, so the ceiling is
the reward (a hit pays the same at any pass speed), not the plant. A 25-chunk constant-LR fine-tune made it slower,
which confirms it: the next lever is a terminal `v_hit` term, not more training. Add `--buckets` to train on the
100-200 / 500-800 / 1000-1500 m distribution with a distance-scaled time clock; the eval stays on
`--distance-low/high` and logs a per-bucket split (`eval_hit_d0..d2`).

## Tests

```bash
uv run python -c "from app.test.test_config import test; test()"   # physics/config validation
uv run pytest app/test --ignore=app/test/test_config.py              # reward / LOS / threat / checkpoint unit tests
uv run python scripts/isaac_lab/diff_test_physics.py                 # torch physics == numpy oracle
uv run python scripts/isaac_lab/diff_test_pid.py                     # torch PID == numpy PID
<isaac-python> -u scripts/isaac_lab/diff_test_trajectory.py --headless --num_steps 100   # PhysX vs oracle
```

## Known limitations

- The policy observes the **exact** target position. A noisy server-feed / camera estimator is planned
  (`app/environmental/perception.py` is the starting point, `app/navigation` the estimator).
- The handoff rewards are path-independent, so handoff policies may climb before approaching
  (`app/reward_functions/docs.md`). The acro v4 reward adds an explicit path term.
- The acro policy cruises at ~38 m/s, below the 52.1 m/s airframe limit: nothing in v4 pays for the pass speed
  (`app/reward_functions/docs.md`).
- Wind and mass randomisation exist but are disabled.
