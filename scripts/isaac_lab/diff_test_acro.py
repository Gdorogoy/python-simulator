"""Diff-test the acro plant: TorchRateController vs numpy RateController, plus a closed-loop flip (no Isaac needed).

See scripts/README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from app.dynamics import angle_style_torch_methods as tp
from app.dynamics.acro_style_methods import RateController, acro_timestamp_update, hover_torque_authority
from app.dynamics.drone import QuadState, Quaternion, Vector3D, create_quad_config
from app.dynamics.angle_style_methods import mixer_inversion
from app.dynamics.acro_style_torch_methods import TorchRateController
from app.dynamics.acro_style_torch_methods import hover_torque_authority as t_hover_torque_authority

RNG = np.random.default_rng(0)
ATOL, RTOL = 1e-9, 1e-9
N_ENVS = 32
N_STEPS = 20
DT = 1 / 240

PARAMS = dict(mass=1.5, inertia=(0.02, 0.02, 0.04), arm_length=0.22, drag_coeff=0.035, max_rpm=12000.0,
              motor_tau=0.05)
np_cfg = create_quad_config(**PARAMS)
t_cfg = tp.build_quad_config(**PARAMS, dtype=torch.float64)

failures = []


def check(name, np_val, t_val):
    np_val = np.asarray(np_val, dtype=np.float64)
    t_val = t_val.detach().cpu().numpy().astype(np.float64)
    if not np.allclose(np_val, t_val, atol=ATOL, rtol=RTOL):
        failures.append(name)
        print(f"[FAIL] {name}: max_abs_diff={np.abs(np_val - t_val).max():.3e}")


check("hover_torque_authority", hover_torque_authority(np_cfg), t_hover_torque_authority(t_cfg))

for gyro_ff in (False, True):
    np_ctrls = [RateController(np_cfg, ki=0.5, gyro_ff=gyro_ff) for _ in range(N_ENVS)]
    t_ctrl = TorchRateController(t_cfg, N_ENVS, ki=0.5, gyro_ff=gyro_ff)
    for step in range(N_STEPS):
        ang_vel = RNG.uniform(-8, 8, size=(N_ENVS, 3))
        rate_cmd = RNG.uniform(-6, 6, size=(N_ENVS, 3))
        np_tau = np.stack([np_ctrls[i].compute_torque(ang_vel[i], rate_cmd[i], DT) for i in range(N_ENVS)])
        t_tau = t_ctrl.compute_torque(torch.tensor(ang_vel), torch.tensor(rate_cmd), DT)
        check(f"compute_torque gyro_ff={gyro_ff} step {step}", np_tau, t_tau)
    check(f"integral gyro_ff={gyro_ff}", np.stack([c.integral for c in np_ctrls]), t_ctrl.integral)

# Closed loop on the numpy plant: 2*pi rad/s roll for 1 s, then 0. Must end one full turn later, level again.
hover_w = mixer_inversion(np_cfg, [np_cfg.mass * 9.81, 0.0, 0.0, 0.0])
state = QuadState(Vector3D(0, 0, 10), Vector3D(0, 0, 0), Quaternion(0, 0, 0, 1), Vector3D(0, 0, 0), list(hover_w))
ctrl = RateController(np_cfg)
rolled = 0.0
for k in range(int(1.6 / DT)):
    p_cmd = 2 * np.pi if k * DT < 1.0 else 0.0
    state = acro_timestamp_update(state, np_cfg, ctrl, [np_cfg.mass * 9.81 * 1.3, p_cmd, 0.0, 0.0], [0, 0, 0], DT)
    rolled += state.angular_velocity.x * DT
body_up = Rotation.from_quat([state.orientation.x, state.orientation.y, state.orientation.z,
                              state.orientation.w]).apply([0, 0, 1])
if abs(np.degrees(rolled) - 360.0) > 10.0 or body_up[2] < 0.99:
    failures.append("flip")
    print(f"[FAIL] flip: rolled {np.degrees(rolled):.1f} deg, body_up={body_up.round(3)}")

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print(f"All acro checks passed (flip rolled {np.degrees(rolled):.1f} deg, body_up_z={body_up[2]:.4f}).")
