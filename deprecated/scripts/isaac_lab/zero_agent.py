"""
Standalone smoke test for BaseDroneEnvIsaac -- launches Isaac Sim, steps the
env with zero actions. Run with the py3.11 Isaac venv, e.g. from repo root:

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe scripts\\isaac_lab\\zero_agent.py --num_envs 4 --headless

Adapted from IsaacLab's own scripts/environments/zero_agent.py.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero agent smoke test for the Isaac base-drone task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

# repo root (this file is scripts/isaac_lab/zero_agent.py) -- Kit's python.exe
# resolves sys.path[0] to this script's own directory, not the repo root, so
# `import app.*` needs the root added explicitly regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import torch

import app.environmental.base_drone_env_isaac  # noqa: F401 -- registers Isaac-Base-Drone-Direct-v0
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
