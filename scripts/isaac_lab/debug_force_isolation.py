"""
Debug probe for the force-under-delivery bug documented in
MIGRATION_PROGRESS.md (now fixed -- see app/environmental/base_drone_env_isaac.py's
module docstring for the resolution). Isolates force vs torque application by
monkeypatching _apply_action to send ONLY a known force (hover thrust, zero
torque) or ONLY a known torque, one step at a time, and reading back
root_lin_vel_w / root_ang_vel_b directly -- bypassing our own physics
entirely so the numbers reflect PhysX/wrench-composer behavior alone, not
anything in app.dynamics.torch_methods (already diff-tested bit-exact
against the numpy oracle separately, see diff_test_physics.py).

Kept for future debugging if this ever regresses (e.g. after an IsaacLab
version bump) -- use this script to try the next hypothesis rather than
re-deriving the isolation setup from scratch.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\debug_force_isolation.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

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


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=1)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    action = torch.zeros(1, 4, device=u.device)

    def zero_force():
        u._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=torch.zeros(1, 1, 3, device=u.device),
            torques=torch.zeros(1, 1, 3, device=u.device),
        )

    def hover_thrust_only():
        f = torch.zeros(1, 1, 3, device=u.device)
        f[:, 0, 2] = u._hover_thrust
        u._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=f, torques=torch.zeros(1, 1, 3, device=u.device),
        )

    def torque_only():
        u._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=torch.zeros(1, 1, 3, device=u.device),
            torques=torch.tensor([[[0.05, 0.0, 0.0]]], device=u.device),
        )

    for name, fn in [("zero_force (gravity only, expect -9.81*dt)", zero_force),
                      ("hover_thrust_only (expect ~0 net accel)", hover_thrust_only),
                      ("torque_only 0.05 N*m x (expect torque/inertia*dt = 0.05/0.02*dt = 0.010417)", torque_only)]:
        env.reset()
        u._apply_action = fn
        env.step(action)  # not wrapped in inference_mode: repeated reset()+step() across iterations
        # of this loop otherwise hits "Inplace update to inference tensor" on the second reset().
        print(f"{name}:")
        print(f"  root_lin_vel_w = {u._robot.data.root_lin_vel_w}")
        print(f"  root_ang_vel_b = {u._robot.data.root_ang_vel_b}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
