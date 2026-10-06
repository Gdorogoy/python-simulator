"""
Migration step 6 smoke test: collect a small batch of demonstrations on the
live Isaac env using the batched PID teacher, check the PID actually tracks
toward targets (mean distance-to-goal should trend down, not diverge).

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\collect_demo_smoke.py --headless --num_envs 16
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--n_target_rows", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import gymnasium as gym

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.control.collect_demonstrations import collect_demonstrations_base_drone_isaac


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    # distance range scoped to (3, 10) for now -- up to 250/3000 later, once
    # the per-distance episode-length cap is addressed (see the function's
    # docstring for why that matters beyond ~50-60m).
    obs, actions = collect_demonstrations_base_drone_isaac(
        env, gains_by_dist, n_target_rows=args_cli.n_target_rows, distance_low=3.0, distance_high=10.0,
        save_path="app/control/scratch_demo_smoke.npz",
    )

    nan_obs = np.isnan(obs).any()
    nan_act = np.isnan(actions).any()
    print(f"[RESULT] obs.shape={obs.shape} actions.shape={actions.shape} nan_obs={nan_obs} nan_act={nan_act}")

    # rough distance-to-goal sanity: recover it from obs' symlog(dist) field
    # (index 20, see build_observation's layout -- pos(3)+vel(3)+quat(4)+ang_vel(3)+rpm(4)+rel(3)+dist(1)+yaw(2)=23)
    symlog_dist = obs[:, 20]
    first_100 = symlog_dist[:100].mean()
    last_100 = symlog_dist[-100:].mean()
    print(f"[RESULT] mean symlog(dist) first 100 rows={first_100:.4f}  last 100 rows={last_100:.4f} "
          f"(should trend down if the PID is tracking toward targets)")

    print("[FAIL]" if (nan_obs or nan_act) else "[PASS]", "demo collection smoke test")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
