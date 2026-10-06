"""Dev-only diagnostic: does the PID teacher itself time out under real decimation=4
full-sphere conditions, or only the DAgger-trained student? Tracks
hit_rate/oob_rate/attitude_rate/timeout_rate over n_episodes, PID-only, no data saved.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\diagnose_pid_isaac.py --headless ^
        --num_envs 4096 --n_episodes 2000 --distance-low 3 --distance-high 10
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--n_episodes", type=int, default=2000)
parser.add_argument("--distance-low", dest="distance_low", type=float, default=3.0)
parser.add_argument("--distance-high", dest="distance_high", type=float, default=10.0)
parser.add_argument("--gains-path", dest="gains_path", default="app/control/best_pid_gains_per_dist.json")
parser.add_argument("--log-every", dest="log_every", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import numpy as np
import torch

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.control.collect_demonstrations import sample_full_sphere_target
from app.control.torch_pid import TorchPIDController, assign_gains_by_distance


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    with open(args_cli.gains_path) as f:
        gains_by_dist = json.load(f)

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

    pid = TorchPIDController(unwrapped.num_envs, device, **next(iter(gains_by_dist.values())))
    obs = env.reset()[0]["policy"]
    all_env_ids = torch.arange(unwrapped.num_envs, device=device)
    assign_gains_by_distance(pid, unwrapped._start_dist, gains_by_dist, all_env_ids)
    pid.reset()

    n_episodes = n_hits = n_oob = n_attitude = n_timeout = 0
    steps = 0
    import isaaclab.utils.math as math_utils

    while n_episodes < args_cli.n_episodes:
        pos = unwrapped._robot.data.root_pos_w - unwrapped._terrain.env_origins
        vel = unwrapped._robot.data.root_lin_vel_w
        ang_vel = unwrapped._robot.data.root_ang_vel_b
        quat_wxyz = unwrapped._robot.data.root_quat_w
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_wxyz)
        target_local = unwrapped._desired_pos_w - unwrapped._terrain.env_origins

        with torch.no_grad():
            action = pid.compute_action(pos, vel, ang_vel, roll, pitch, yaw,
                                         target_local, unwrapped._desired_yaw_w, dt=1 / 240)

        _, reward, terminated, truncated, extras = env.step(action)

        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            n_episodes += len(done_ids)
            n_hits += int(extras["term_reasons"]["hit"][done_ids].sum().item())
            n_oob += int(extras["term_reasons"]["oob"][done_ids].sum().item())
            n_attitude += int((extras["term_reasons"]["attitude_roll"][done_ids]
                                | extras["term_reasons"]["attitude_pitch"][done_ids]).sum().item())
            n_timeout += int((truncated[done_ids] & ~terminated[done_ids]).sum().item())
            assign_gains_by_distance(pid, unwrapped._start_dist, gains_by_dist, done_ids)
            pid.reset(done_ids)

        steps += 1
        if steps % args_cli.log_every == 0:
            n = max(n_episodes, 1)
            print(f"steps={steps} episodes={n_episodes}/{args_cli.n_episodes} "
                  f"hit_rate={n_hits / n:.2f} oob_rate={n_oob / n:.2f} "
                  f"attitude_rate={n_attitude / n:.2f} timeout_rate={n_timeout / n:.2f}")

    n = max(n_episodes, 1)
    print(f"\n=== FINAL: PID teacher, real Isaac decimation=4, full-sphere Uniform"
          f"({args_cli.distance_low},{args_cli.distance_high}), {n_episodes} episodes ===")
    print(f"hit_rate={n_hits / n:.3f}  oob_rate={n_oob / n:.3f}  "
          f"attitude_rate={n_attitude / n:.3f}  timeout_rate={n_timeout / n:.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
