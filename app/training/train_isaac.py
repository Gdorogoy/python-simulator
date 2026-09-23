"""Real training launcher for app.training.base_training_isaac.train (imitation -> critic-warmup -> PPO).
Every budget/path knob is a CLI flag -- this is the one to use for a real run, not a smoke test.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u app\\training\\train_isaac.py --headless ^
        --num_envs 4096 --total_timesteps 128000000 ^
        --distance-low 3 --distance-high 10 ^
        --checkpoint-dir runs/base_training_isaac_3_10 ^
        --bc-checkpoint app/control/pretrained_bc_dagger.pt

Long-running -- use run_in_background or its own terminal, and watch <checkpoint-dir>/metrics.csv
and plots/*.png as it goes. mlflow run lives in mlflow_isaac.db, experiment "base-training-isaac".

Curriculum staging: call again with a new --checkpoint-dir and
--bc-checkpoint <previous stage's dir>/model_<last timesteps>.pt.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--total_timesteps", type=int, default=128_000_000)
parser.add_argument("--distance-low", type=float, default=3.0)
parser.add_argument("--distance-high", type=float, default=10.0)
parser.add_argument("--checkpoint-dir", default="runs/base_training_isaac")
parser.add_argument("--bc-checkpoint", dest="bc_checkpoint_path", default="app/control/pretrained_bc_dagger.pt")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--hparams-json", dest="hparams_json", default="app/training/best_hparams_isaac.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.training.base_training_isaac import train

# app/training/best_hparams_isaac.json -- optuna_search_isaac's best trial (grade=0.8655), rounded.
# lr 0.00013: peak LR, 2.6x baseline's 5e-5. lr_min_ratio 0.275: cosine floor, vs baseline 0.01.
# lam 0.855: GAE lambda, trusts critic more than baseline's 0.97. clip_eps 0.245: vs baseline 0.285.
# vf_coef 0.605: vs baseline 0.525. target_kl 0.035: looser than baseline's 0.0245.
# max_grad_norm 1.65: far looser than baseline's 0.25. log_std_max -1.35: tighter than baseline's -0.9.
# ent_coef_start 0.01 / ent_coef_end 0.001: Optuna's 0.035/0.025 barely decayed and pinned
# effective_std near-constant all training (observed in runs/base_training_isaac_3_10's
# metrics.csv) -- reset to baseline's values so exploration actually tightens over time.
# weight_decay 0.00016: close to baseline's 1e-4.


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    hparams_override = None
    if args_cli.hparams_json and os.path.exists(args_cli.hparams_json):
        with open(args_cli.hparams_json) as f:
            hparams_override = json.load(f)
        print(f"[train_isaac] hyperparameter overrides loaded from {args_cli.hparams_json}: {hparams_override}")
    elif args_cli.hparams_json:
        print(f"[train_isaac] no hparams file at {args_cli.hparams_json}, using base_training_isaac.py's own defaults")

    train(
        env, distance_low=args_cli.distance_low, distance_high=args_cli.distance_high,
        checkpoint_dir=args_cli.checkpoint_dir,
        bc_checkpoint_path=args_cli.bc_checkpoint_path,
        total_timesteps=args_cli.total_timesteps,
        seed=args_cli.seed,
        hparams_override=hparams_override,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
