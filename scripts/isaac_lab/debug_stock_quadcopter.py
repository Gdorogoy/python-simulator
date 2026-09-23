"""
Debug probe: does IsaacLab's OWN official, unmodified quadcopter example
(Isaac-Quadcopter-Direct-v0 -- no overrides, no custom code of ours at all)
also under-deliver force via permanent_wrench_composer.set_forces_and_torques?

Historical: used to confirm the force-under-delivery bug (see
MIGRATION_PROGRESS.md) wasn't in the wrench-composer API itself, only in our
then-placeholder Articulation asset (since fixed -- see
app/environmental/base_drone_env_isaac.py's module docstring). Kept for
future reference if a similar bug is ever suspected again.

QuadcopterEnvCfg's _pre_physics_step: thrust = thrust_to_weight * robot_weight
* (action[0]+1)/2. Solving for thrust == robot_weight (net ~0 accel against
gravity, the same style of isolation test used in debug_force_isolation.py):
action[0] = 2/thrust_to_weight - 1.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u scripts\\isaac_lab\\debug_stock_quadcopter.py --headless
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

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 -- registers Isaac-Quadcopter-Direct-v0
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg("Isaac-Quadcopter-Direct-v0", num_envs=1)
    env = gym.make("Isaac-Quadcopter-Direct-v0", cfg=env_cfg)
    u = env.unwrapped
    env.reset()

    thrust_to_weight = u.cfg.thrust_to_weight
    robot_weight = u._robot_weight
    hover_action0 = 2.0 / thrust_to_weight - 1.0
    print(f"thrust_to_weight={thrust_to_weight}  robot_weight={robot_weight}  hover_action0={hover_action0}")

    action = torch.zeros(1, 4, device=u.device)
    action[:, 0] = hover_action0  # roll/pitch/yaw moment stay 0
    env.step(action)
    print(f"root_lin_vel_w AFTER 1 hover-equivalent step (expect ~0): {u._robot.data.root_lin_vel_w}")

    # zero-action baseline (thrust = 0.5*weight, should fall)
    env.reset()
    action[:, 0] = 0.0
    env.step(action)
    dt = u.cfg.sim.dt
    expected_az = (0.5 * thrust_to_weight * robot_weight - robot_weight) / u._robot_mass.item()
    print(f"zero-action: expected accel_z={expected_az:.4f} m/s^2, "
          f"expected dv={expected_az * dt:.6f} m/s over dt={dt}")
    print(f"root_lin_vel_w AFTER 1 zero-action step: {u._robot.data.root_lin_vel_w}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
