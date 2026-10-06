"""Same action sequence through the numpy oracle and the Isaac env: per-step state divergence. See scripts/README.md."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trajectory diff-test: Isaac Lab env vs numpy oracle.")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to compare.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1  # single-env, 1:1 comparison against the (single-drone) numpy oracle

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

import gymnasium as gym

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion
from app.dynamics.methods import timestamp_update
from app.dynamics import torch_methods as tp

MASS, INERTIA, ARM, DRAG_COEFF, MAX_RPM, MOTOR_TAU = 1.5, (0.02, 0.02, 0.04), 0.22, 0.035, 12000, 0.05
DT = 1 / 240
START_POS = np.array([0.0, 0.0, 5.0])

RNG = np.random.default_rng(42)


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=1)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    decimation = unwrapped.cfg.decimation
    print(f"[INFO] decimation={decimation} -- oracle side calls timestamp_update {decimation}x per Isaac step")

    env.reset()
    print(f"[DEBUG post-reset mass] readback={unwrapped._robot.root_physx_view.get_masses()}")
    # pin env 0 to the oracle's initial condition (spawn (0,0,5))
    zero_ids = torch.tensor([0], device=unwrapped.device)
    root_state = unwrapped._robot.data.default_root_state[zero_ids].clone()
    root_state[:, :3] = torch.tensor(START_POS, device=unwrapped.device, dtype=root_state.dtype) + unwrapped._terrain.env_origins[zero_ids]
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=unwrapped.device)
    root_state[:, 7:] = 0.0
    unwrapped._robot.write_root_pose_to_sim(root_state[:, :7], zero_ids)
    unwrapped._robot.write_root_velocity_to_sim(root_state[:, 7:], zero_ids)

    hover_thrust = unwrapped._hover_thrust
    hover_desired = torch.zeros(1, 4, device=unwrapped.device)
    hover_desired[:, 0] = hover_thrust
    hover_w = tp.mixer_inversion(unwrapped._t_cfg, hover_desired)
    unwrapped._rotor_rpm[zero_ids] = hover_w[zero_ids] if hover_w.shape[0] > 1 else hover_w

    # --- numpy oracle, matching initial condition exactly ---
    np_cfg = create_quad_config(mass=MASS, inertia=INERTIA, arm_length=ARM, drag_coeff=DRAG_COEFF,
                                 max_rpm=MAX_RPM, motor_tau=MOTOR_TAU)
    np_state = QuadState(
        position=Vector3D(*START_POS), velocity=Vector3D(0, 0, 0),
        orientation=Quaternion(0, 0, 0, 1), angular_velocity=Vector3D(0, 0, 0),
        rotor_rpm=list(hover_w[0].cpu().numpy().astype(np.float64)),
    )

    # --- fixed action sequence, mild random perturbations around hover ---
    actions = RNG.uniform(-0.15, 0.15, size=(args_cli.num_steps, 4)).astype(np.float32)
    actions[:, 0] *= 20.0  # thrust delta in Newtons, roll/pitch/yaw stay in [-0.15,0.15] N*m

    pos_err, vel_err, ang_vel_err, rpm_err = [], [], [], []

    for step in range(args_cli.num_steps):
        action_t = torch.tensor(actions[step], device=unwrapped.device).unsqueeze(0)
        with torch.inference_mode():
            env.step(action_t)

        real_action = list(actions[step].astype(np.float64) + [hover_thrust, 0, 0, 0])
        for _ in range(decimation):
            np_state = timestamp_update(np_state, np_cfg, real_action, [0, 0, 0], DT)

        iso_pos = (unwrapped._robot.data.root_pos_w[0] - unwrapped._terrain.env_origins[0]).cpu().numpy()
        iso_vel = unwrapped._robot.data.root_lin_vel_w[0].cpu().numpy()
        iso_ang_vel = unwrapped._robot.data.root_ang_vel_b[0].cpu().numpy()
        iso_rpm = unwrapped._rotor_rpm[0].cpu().numpy()

        np_pos = np.array([np_state.position.x, np_state.position.y, np_state.position.z])
        np_vel = np.array([np_state.velocity.x, np_state.velocity.y, np_state.velocity.z])
        np_ang_vel = np.array([np_state.angular_velocity.x, np_state.angular_velocity.y, np_state.angular_velocity.z])
        np_rpm = np.array(np_state.rotor_rpm)

        pos_err.append(np.abs(iso_pos - np_pos).max())
        vel_err.append(np.abs(iso_vel - np_vel).max())
        ang_vel_err.append(np.abs(iso_ang_vel - np_ang_vel).max())
        rpm_err.append(np.abs(iso_rpm - np_rpm).max())

        if step < 5 or step % 20 == 0:
            print(f"step {step:3d}: pos_err={pos_err[-1]:.6e}  vel_err={vel_err[-1]:.6e}  "
                  f"ang_vel_err={ang_vel_err[-1]:.6e}  rpm_err={rpm_err[-1]:.4e}  "
                  f"iso_pos={iso_pos}  np_pos={np_pos}")
            print(f"          iso_vel={iso_vel}  np_vel={np_vel}  iso_ang_vel={iso_ang_vel}  np_ang_vel={np_ang_vel}  action={actions[step]}")

    print("\n=== summary ===")
    print(f"max pos_err over {args_cli.num_steps} steps:     {max(pos_err):.6e} m")
    print(f"max vel_err over {args_cli.num_steps} steps:     {max(vel_err):.6e} m/s")
    print(f"max ang_vel_err over {args_cli.num_steps} steps: {max(ang_vel_err):.6e} rad/s")
    print(f"max rpm_err over {args_cli.num_steps} steps:     {max(rpm_err):.6e} rad/s")
    print(f"final pos_err: {pos_err[-1]:.6e} m  (step {args_cli.num_steps-1})")
    print("\nrpm_err should be ~0 (motor_lag/mixer are our own torch ops, diff-tested bit-exact in "
          "diff_test_physics.py -- any nonzero here means a wiring bug, not integrator drift).")
    print("pos/vel/ang_vel_err reflect PhysX's rigid-body integrator vs the oracle's manual "
          "semi-implicit-Euler + exact rotvec exponential map -- small and slowly-growing is expected "
          "integrator difference, not a bug; large or immediately-diverging values indicate a real bug "
          "(wrong frame/sign in _apply_action, mass/inertia override not taking effect, etc).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
