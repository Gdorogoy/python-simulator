"""
Migration step 6 diff-test: TorchPIDController (app.control.torch_pid) vs the
numpy PIDController (app.control.pid), on random states/targets, batched vs
looped. Pure torch/numpy/scipy -- no Kit boot needed:

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe scripts\\isaac_lab\\diff_test_pid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from app.control.pid import PIDController
from app.dynamics.drone import QuadState, Vector3D, Quaternion
from app.control.torch_pid import TorchPIDController

RNG = np.random.default_rng(0)
ATOL, RTOL = 1e-5, 1e-5
N_ENVS = 32
N_STEPS = 20

GAINS = dict(kp_pos=8.5, kd_pos=4.2, kp_att=6.0, kd_att=1.5, kp_yaw=3.0, kd_yaw=0.8,
             ki_pos=0.3, ki_att=0.1, ki_yaw=0.05)

np_pids = [PIDController(**GAINS) for _ in range(N_ENVS)]
t_pid = TorchPIDController(N_ENVS, "cpu", **GAINS)

failures = []


def check(name, np_val, t_val):
    np_val = np.asarray(np_val, dtype=np.float64)
    t_val = t_val.detach().cpu().numpy().astype(np.float64)
    ok = np.allclose(np_val, t_val, atol=ATOL, rtol=RTOL)
    if not ok:
        failures.append(name)
        print(f"[FAIL] {name}: max_abs_diff={np.abs(np_val - t_val).max():.3e}")
    return ok


for step in range(N_STEPS):
    positions = RNG.uniform(-10, 10, size=(N_ENVS, 3))
    velocities = RNG.uniform(-3, 3, size=(N_ENVS, 3))
    ang_vels = RNG.uniform(-2, 2, size=(N_ENVS, 3))
    eulers = RNG.uniform(-0.5, 0.5, size=(N_ENVS, 3))  # roll, pitch, yaw -- small angles
    targets = RNG.uniform(-10, 10, size=(N_ENVS, 3))
    target_yaws = RNG.uniform(-np.pi, np.pi, size=N_ENVS)

    quats_xyzw = Rotation.from_euler("xyz", eulers).as_quat()  # (N,4) x,y,z,w

    np_actions = np.zeros((N_ENVS, 4))
    for i in range(N_ENVS):
        state = QuadState(
            position=Vector3D(*positions[i]), velocity=Vector3D(*velocities[i]),
            orientation=Quaternion(*quats_xyzw[i]), angular_velocity=Vector3D(*ang_vels[i]),
            rotor_rpm=[0, 0, 0, 0],
        )
        np_actions[i] = np_pids[i].compute_action(state, targets[i], target_yaws[i], dt=1 / 240)

    roll_t = torch.tensor(eulers[:, 0], dtype=torch.float32)
    pitch_t = torch.tensor(eulers[:, 1], dtype=torch.float32)
    yaw_t = torch.tensor(eulers[:, 2], dtype=torch.float32)
    t_actions = t_pid.compute_action(
        pos=torch.tensor(positions, dtype=torch.float32),
        vel=torch.tensor(velocities, dtype=torch.float32),
        ang_vel=torch.tensor(ang_vels, dtype=torch.float32),
        roll=roll_t, pitch=pitch_t, yaw=yaw_t,
        target_pos=torch.tensor(targets, dtype=torch.float32),
        target_yaw=torch.tensor(target_yaws, dtype=torch.float32),
        dt=1 / 240,
    )

    check(f"step{step}.action", np_actions, t_actions)
    check(f"step{step}.integral_pos", np.stack([p.integral_pos for p in np_pids]), t_pid.integral_pos)
    check(f"step{step}.integral_att", np.stack([p.integral_att for p in np_pids]), t_pid.integral_att)
    check(f"step{step}.integral_yaw", np.array([p.integral_yaw for p in np_pids]), t_pid.integral_yaw)

print(f"\n=== result ===")
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
else:
    print(f"ALL {N_STEPS} steps x {N_ENVS} envs PASSED (atol={ATOL}, rtol={RTOL})")
