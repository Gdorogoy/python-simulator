"""
Smoke test for app.training.base_training_isaac.PidResidualEnv: with a ZERO correction the wrapped env must
behave like the PID (episodes end in hits with PID-level reward, per-distance gains get assigned), while the
same env fed zero actions directly (no PID) must be far worse. Also checks a small random correction degrades
gracefully instead of breaking the PID.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\residual_smoke.py --headless --num_envs 64
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=700)
parser.add_argument("--residual-scale", dest="residual_scale", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import numpy as np
import torch

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.training.base_training_isaac import PidResidualEnv
from app.training.eval_matrix import build_uniform_omni_eval_pairs


def run_episodes(env, steps, action_fn):
    """Returns (episode_rewards, episode_lengths) for every episode that finished within `steps` policy steps."""
    unwrapped = env.unwrapped
    env.reset()
    n = unwrapped.num_envs
    ep_reward = torch.zeros(n, device=unwrapped.device)
    ep_len = torch.zeros(n, device=unwrapped.device)
    rewards, lengths = [], []
    for _ in range(steps):
        action = action_fn(n, unwrapped.device)
        _, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        ep_len += 1
        for i in torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1).tolist():
            rewards.append(ep_reward[i].item())
            lengths.append(ep_len[i].item())
            ep_reward[i] = 0.0
            ep_len[i] = 0.0
    return np.array(rewards), np.array(lengths)


def summarize(label, rewards, lengths):
    if len(rewards) == 0:
        print(f"[RESULT] {label}: no episode finished")
        return
    print(f"[RESULT] {label}: episodes={len(rewards)}  mean_reward={rewards.mean():.1f}  "
          f"median_reward={np.median(rewards):.1f}  frac_reward>30={np.mean(rewards > 30):.2f}  "
          f"mean_len={lengths.mean():.0f} policy steps")


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    rng = np.random.default_rng(0)
    unwrapped.set_target_pairs(build_uniform_omni_eval_pairs(oob_radius=30.0, low=3.0, high=10.0, n_pairs=500, rng=rng))
    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    zero = lambda n, dev: torch.zeros(n, 4, device=dev)
    raw_r, raw_l = run_episodes(env, args_cli.steps, zero)
    summarize("raw env, zero action (no PID)", raw_r, raw_l)

    wrapped = PidResidualEnv(env, gains_by_dist, args_cli.residual_scale)
    zr, zl = run_episodes(wrapped, args_cli.steps, zero)
    summarize(f"wrapped, zero correction (scale={args_cli.residual_scale})", zr, zl)
    print(f"[RESULT] distinct kp_pos gains across envs: {torch.unique(wrapped._pid.kp_pos).numel()} "
          f"(>1 means per-distance gains were assigned)")

    # policy_action in [-half_range, half_range]; +-0.3 of it, times the scale, is a small perturbation.
    half = 0.5 * (torch.as_tensor(unwrapped.single_action_space.high) - torch.as_tensor(unwrapped.single_action_space.low))
    noisy = lambda n, dev: (torch.rand(n, 4, device=dev) * 2 - 1) * 0.3 * half.to(dev)
    nr, nl = run_episodes(wrapped, args_cli.steps, noisy)
    summarize("wrapped, random correction (+-0.3 range)", nr, nl)

    ok = len(zr) > 0 and np.mean(zr > 30) > 0.8 and (len(raw_r) == 0 or zr.mean() > raw_r.mean() + 20)
    print("[PASS] zero-correction wrapper behaves like the PID." if ok else
          "[FAIL] zero-correction wrapper does NOT look like a working PID -- inspect the numbers above.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
