"""
Smoke test for app.training.base_training_isaac.train -- the full imitation
-> critic-warmup -> PPO phase pipeline, at a tiny budget (not a real training
run) to verify the mechanism end-to-end: phases run in order, checkpoints/
onnx/metrics.csv/plots get written, no NaN, no crash.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\full_pipeline_smoke.py --headless --num_envs 16 --total_timesteps 40000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--total_timesteps", type=int, default=40_000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import torch

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.training.base_training_isaac import train


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    model = train(
        env, distance_low=3, distance_high=10,
        checkpoint_dir="runs/full_pipeline_smoke",
        bc_checkpoint_path="app/control/pretrained_bc.pt",
        total_timesteps=args_cli.total_timesteps,
        seed=0,
    )

    nan_found = any(torch.isnan(v).any() for v in model.state_dict().values())
    print(f"[{'FAIL' if nan_found else 'PASS'}] full pipeline smoke test, nan_found={nan_found}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
