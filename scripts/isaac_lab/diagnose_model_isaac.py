"""Dev-only diagnostic: runs a trained checkpoint's deterministic action directly in the real
Isaac env (decimation=4), same instrumentation as diagnose_pid_isaac.py for direct comparison.
Checks --checkpoint-a (pre-DAgger) against --checkpoint-b (post-DAgger) in one run.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\diagnose_model_isaac.py --headless ^
        --num_envs 4096 --n_episodes 2000 --distance-low 3 --distance-high 10
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--n_episodes", type=int, default=2000)
parser.add_argument("--distance-low", dest="distance_low", type=float, default=3.0)
parser.add_argument("--distance-high", dest="distance_high", type=float, default=10.0)
parser.add_argument("--checkpoint-a", dest="checkpoint_a", default="app/control/pretrained_bc.pt")
parser.add_argument("--checkpoint-b", dest="checkpoint_b", default="app/control/pretrained_bc_dagger.pt")
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--num_hidden_layers", type=int, default=4)
parser.add_argument("--log-every", dest="log_every", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import numpy as np
import torch

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.control.collect_demonstrations import sample_full_sphere_target
from app.guidance.train import ActorCritic, device


def run_checkpoint(env, checkpoint_path, label):
    unwrapped = env.unwrapped

    model = ActorCritic(unwrapped.single_observation_space["policy"].shape[0], unwrapped.single_action_space.shape[0],
                         unwrapped.single_action_space.low, unwrapped.single_action_space.high,
                         hidden=args_cli.hidden, num_hidden_layers=args_cli.num_hidden_layers).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    obs = env.reset()[0]["policy"]
    n_episodes = n_hits = n_oob = n_attitude = n_timeout = 0
    steps = 0

    while n_episodes < args_cli.n_episodes:
        with torch.no_grad():
            mean, _, _ = model.forward(obs)
            action = model.scale_action(mean)

        next_obs_dict, reward, terminated, truncated, extras = env.step(action)
        obs = next_obs_dict["policy"]

        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            n_episodes += len(done_ids)
            n_hits += int(extras["term_reasons"]["hit"][done_ids].sum().item())
            n_oob += int(extras["term_reasons"]["oob"][done_ids].sum().item())
            n_attitude += int((extras["term_reasons"]["attitude_roll"][done_ids]
                                | extras["term_reasons"]["attitude_pitch"][done_ids]).sum().item())
            n_timeout += int((truncated[done_ids] & ~terminated[done_ids]).sum().item())

        steps += 1
        if steps % args_cli.log_every == 0:
            n = max(n_episodes, 1)
            print(f"[{label}] steps={steps} episodes={n_episodes}/{args_cli.n_episodes} "
                  f"hit_rate={n_hits / n:.2f} timeout_rate={n_timeout / n:.2f}")

    n = max(n_episodes, 1)
    print(f"\n=== FINAL [{label}] ({checkpoint_path}), {n_episodes} episodes ===")
    print(f"hit_rate={n_hits / n:.3f}  oob_rate={n_oob / n:.3f}  "
          f"attitude_rate={n_attitude / n:.3f}  timeout_rate={n_timeout / n:.3f}\n")


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped

    rng = np.random.default_rng()
    pairs = []
    for _ in range(1024):
        dist = float(rng.uniform(args_cli.distance_low, args_cli.distance_high))
        target = None
        while target is None:
            target = sample_full_sphere_target(rng, dist)
        start = np.array([0, 0, 5], dtype=np.float32) + rng.uniform(-0.15, 0.15, size=3).astype(np.float32)
        target_yaw = float(rng.uniform(-np.pi, np.pi))
        pairs.append((start, target, target_yaw))
    unwrapped.set_target_pairs(pairs)

    if os.path.exists(args_cli.checkpoint_a):
        run_checkpoint(env, args_cli.checkpoint_a, "pre-DAgger (BC only)")
    else:
        print(f"skipping checkpoint-a, not found: {args_cli.checkpoint_a}")

    if os.path.exists(args_cli.checkpoint_b):
        run_checkpoint(env, args_cli.checkpoint_b, "post-DAgger (5 rounds)")
    else:
        print(f"skipping checkpoint-b, not found: {args_cli.checkpoint_b}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
