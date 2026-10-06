# deprecated

Code that is no longer part of the current pipeline, kept for reference and for reproducing old results. It is
**not maintained**. Imports were re-pointed at the moved paths so most modules still import, but nothing here is
tested.

| Path | What it was | Replaced by |
|---|---|---|
| `training/train_isaac.py` | Non-RTL Isaac PPO trainer (produced the `stable_models/*_res01` models, PID-residual mode). | `app/training/rtl_train_isaac.py` |
| `training/best_hparams_isaac.json` | Optuna-tuned hyperparameters for `train_isaac.py`. | RTL CLI defaults |
| `training/base_training.py` | Numpy/PyBullet-era trainer (multiprocess CPU envs). | Isaac trainers. Shared constants moved to `app/training/config.py`. |
| `training/phase_0_training.py`, `phase_1_training.py` | Original hover / single-axis curriculum. `phase_1_training` is already broken: it imports `reward_fn_phase1`, deleted in `c5c8775`. | `reward_qv2` + RTL |
| `guidance/optuna_search*.py`, `analyze_optuna_isaac.py`, `optuna-search-space-audit.md` | Hyperparameter search and analysis for the old trainers. | n/a |
| `guidance/training-goals.md` | Original phase-0/1 curriculum plan. | `app/training/rtl-handoff.md` |
| `reward_functions/phase1_rewards.py` | Curriculum reward for phase 1. | `reward_qv2` |
| `reward_functions/reward_qv1.py` | First Isaac "just hit" reward. | `reward_qv2` (a standalone copy plus stall terms) |
| `environmental/vec_base_drone_env.py` | Single-process numpy vec-env. | `SubprocVecBaseDroneEnv` / Isaac |
| `demo/`, `app_main.py`, `main.py` | PyBullet demo menu and the uv template entry point. | Web viewer, Isaac Sim |
| `scripts/` | Isaac-migration smoke and debug scripts. | `scripts/isaac_lab/diff_test_*` |

To run something here, use the repo root as the working directory (`python -m deprecated.training.train_isaac ...`
or the Isaac python with the file path).
