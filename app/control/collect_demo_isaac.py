"""CLI launcher for collect_demonstrations_base_drone_isaac (batched PID demos). See README "Pipeline"."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--n_target_rows", type=int, default=2_000_000)
parser.add_argument("--distance-low", dest="distance_low", type=float, default=3.0)
parser.add_argument("--distance-high", dest="distance_high", type=float, default=10.0)
parser.add_argument("--save-path", dest="save_path", default="app/control/demonstrations_isaac.npz")
parser.add_argument("--gains-path", dest="gains_path", default="app/control/best_pid_gains_per_dist.json")
parser.add_argument("--seed", type=int, default=None)
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

from app.control.collect_demonstrations import collect_demonstrations_base_drone_isaac


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    with open(args_cli.gains_path) as f:
        gains_by_dist = json.load(f)

    collect_demonstrations_base_drone_isaac(
        env, gains_by_dist, n_target_rows=args_cli.n_target_rows,
        distance_low=args_cli.distance_low, distance_high=args_cli.distance_high,
        save_path=args_cli.save_path, seed=args_cli.seed,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
