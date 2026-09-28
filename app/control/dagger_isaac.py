"""Real DAgger launcher for app.control.dagger.dagger_base_drone_isaac (batched, on-policy).
Every knob is a CLI flag.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u app\\control\\dagger_isaac.py --headless ^
        --num_envs 4096 --n_rounds 5 --rows_per_round 500000 --distance-low 3 --distance-high 10 ^
        --checkpoint-path app/control/pretrained_bc.pt ^
        --demo-path app/control/demonstrations_isaac.npz ^
        --out-path app/control/pretrained_bc_dagger.pt

checkpoint_path is the BC starting point; demo_path is the aggregate dataset to build on;
out_path is where refined weights + the growing dataset snapshot get written each round.
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--n_rounds", type=int, default=5)
parser.add_argument("--rows_per_round", type=int, default=500_000)
parser.add_argument("--retrain_epochs", type=int, default=20)
parser.add_argument("--distance-low", dest="distance_low", type=float, default=3.0)
parser.add_argument("--distance-high", dest="distance_high", type=float, default=10.0)
parser.add_argument("--checkpoint-path", dest="checkpoint_path", default="app/control/pretrained_bc.pt")
parser.add_argument("--demo-path", dest="demo_path", default="app/control/demonstrations_omni.npz")
parser.add_argument("--out-path", dest="out_path", default="app/control/pretrained_bc_dagger.pt")
parser.add_argument("--gains-path", dest="gains_path", default="app/control/best_pid_gains_per_dist.json")
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--num_hidden_layers", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.control.dagger import dagger_base_drone_isaac


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    with open(args_cli.gains_path) as f:
        gains_by_dist = json.load(f)

    dagger_base_drone_isaac(
        env, gains_by_dist, n_rounds=args_cli.n_rounds,
        distance_low=args_cli.distance_low, distance_high=args_cli.distance_high,
        rows_per_round=args_cli.rows_per_round, retrain_epochs=args_cli.retrain_epochs,
        checkpoint_path=args_cli.checkpoint_path, demo_path=args_cli.demo_path, out_path=args_cli.out_path,
        hidden=args_cli.hidden, num_hidden_layers=args_cli.num_hidden_layers,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
