"""
Migration step 6 smoke test: runs dagger_base_drone_isaac for 1-2 small
rounds using the repo's existing pretrained_bc.pt / demonstrations_omni.npz,
checks it doesn't crash/NaN and that hit_rate/aggregate size get reported
sanely.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\dagger_smoke.py --headless --num_envs 16
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--n_rounds", type=int, default=1)
parser.add_argument("--rows_per_round", type=int, default=3000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import gymnasium as gym

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.control.dagger import dagger_base_drone_isaac


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    model, history = dagger_base_drone_isaac(
        env, gains_by_dist, n_rounds=args_cli.n_rounds, distance_low=3.0, distance_high=10.0,
        rows_per_round=args_cli.rows_per_round, retrain_epochs=2,
        checkpoint_path="app/control/pretrained_bc.pt",
        demo_path="app/control/demonstrations_omni.npz",
        out_path="app/control/scratch_dagger_smoke.pt",
    )

    print(f"[RESULT] history={history}")
    nan_found = any(torch.isnan(v).any() for v in model.state_dict().values())
    print(f"[{'FAIL' if nan_found else 'PASS'}] dagger_base_drone_isaac smoke test, nan_found={nan_found}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
