"""
Migration step 5 smoke test: runs isaac_ppo_train for a small number of
timesteps and checks it actually trains (losses computed, no NaN, no crash,
episode_rewards populated once episodes start ending).

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\train_smoke.py --headless --num_envs 16 --total_timesteps 4096 --num_steps 64
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--total_timesteps", type=int, default=4096)
parser.add_argument("--num_steps", type=int, default=64)
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

from app.guidance.train import isaac_ppo_train


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    model, optimizer, episode_rewards, last_losses, _final_obs = isaac_ppo_train(
        env, total_timesteps=args_cli.total_timesteps, num_steps=args_cli.num_steps,
        gamma=0.97, lam=0.95, lr=3e-4, target_kl=0.02,
        num_epochs=4, batch_size=64,
    )

    print(f"[RESULT] episodes_completed={len(episode_rewards)}")
    if episode_rewards:
        print(f"[RESULT] episode_rewards (first 10)={episode_rewards[:10]}")
    print(f"[RESULT] last_losses={last_losses}")

    for name, val in model.state_dict().items():
        if torch.isnan(val).any():
            print(f"[FAIL] NaN in model parameter: {name}")
            env.close()
            return
    print("[PASS] training ran without NaN parameters.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
