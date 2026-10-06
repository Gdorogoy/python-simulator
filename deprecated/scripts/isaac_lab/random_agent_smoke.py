"""
Migration step 3 smoke test: reset + N steps of random actions, checking for
exceptions/NaNs. Unlike zero_agent.py this actually exercises the ported
physics (_apply_action/_get_observations/_get_rewards/_get_dones/_reset_idx),
not just the registration/scaffold path.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe scripts\\isaac_lab\\random_agent_smoke.py --num_envs 4 --headless --num_steps 50
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random-action smoke test for the Isaac base-drone task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=50, help="Number of steps to run.")
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

import app.environmental.base_drone_env_isaac  # noqa: F401 -- registers Isaac-Base-Drone-Direct-v0
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    obs, _ = env.reset()
    print(f"[INFO]: reset OK, obs shape={obs['policy'].shape if isinstance(obs, dict) else obs.shape}, "
          f"nan={torch.isnan(obs['policy'] if isinstance(obs, dict) else obs).any().item()}")

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = (torch.rand(env.action_space.shape, device=env.unwrapped.device) * 2 - 1) * 0.3
            obs, reward, terminated, truncated, info = env.step(actions)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        nan_obs = torch.isnan(obs_t).any().item()
        nan_rew = torch.isnan(reward).any().item()
        if step % 10 == 0 or nan_obs or nan_rew:
            print(f"step {step}: reward.mean={reward.mean().item():.4f}  "
                  f"terminated={terminated.sum().item()}  truncated={truncated.sum().item()}  "
                  f"nan_obs={nan_obs}  nan_reward={nan_rew}")
        if nan_obs or nan_rew:
            print("[FAIL] NaN detected, stopping early.")
            break
    else:
        print(f"[PASS] ran {args_cli.num_steps} steps with {args_cli.num_envs} envs, no NaNs, no exceptions.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
