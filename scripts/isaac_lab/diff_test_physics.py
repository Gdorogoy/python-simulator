"""Diff-test dynamics.torch_methods against the numpy oracle on random inputs (no Isaac needed). See scripts/README.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from app.dynamics.drone import create_quad_config
from app.dynamics.methods import (
    mixer_inversion as np_mixer_inversion,
    motor_lag as np_motor_lag,
    net_combining_thrust as np_net_combining_thrust,
    net_combining_torque as np_net_combining_torque,
    drag_force as np_drag_force,
    wind as np_wind,
)
from app.dynamics import torch_methods as tp

RNG = np.random.default_rng(0)
ATOL = 1e-9
RTOL = 1e-9
N_CASES = 64

MASS, INERTIA, ARM, DRAG_COEFF, MAX_RPM, MOTOR_TAU = 1.5, (0.02, 0.02, 0.04), 0.22, 0.035, 12000, 0.05
DT = 1 / 240

np_cfg = create_quad_config(mass=MASS, inertia=INERTIA, arm_length=ARM, drag_coeff=DRAG_COEFF,
                             max_rpm=MAX_RPM, motor_tau=MOTOR_TAU)
t_cfg = tp.build_quad_config(mass=MASS, inertia=INERTIA, arm_length=ARM, drag_coeff=DRAG_COEFF,
                              max_rpm=MAX_RPM, motor_tau=MOTOR_TAU, dtype=torch.float64)

failures = []


def check(name, np_val, t_val):
    np_val = np.asarray(np_val, dtype=np.float64)
    t_val = t_val.detach().cpu().numpy().astype(np.float64)
    ok = np.allclose(np_val, t_val, atol=ATOL, rtol=RTOL)
    status = "OK" if ok else "FAIL"
    if not ok:
        failures.append(name)
        print(f"[{status}] {name}: numpy={np_val}  torch={t_val}  max_abs_diff={np.abs(np_val - t_val).max():.3e}")
    return ok


print(f"=== rotor geometry / mixer_inv sanity ===")
np_rotor_xy = np.array([[r.position.x, r.position.y] for r in np_cfg.rotors])
check("rotor_xy", np_rotor_xy, t_cfg.rotor_xy)
np_k_f = np.array([r.k_f for r in np_cfg.rotors])
check("k_f", np_k_f, t_cfg.k_f)

print(f"\n=== {N_CASES} random-case function diff tests ===")
for case in range(N_CASES):
    # --- mixer_inversion: random desired [thrust, roll, pitch, yaw] around hover ---
    hover_thrust = MASS * 9.81
    desired = np.array([
        hover_thrust + RNG.uniform(-5, 5),
        RNG.uniform(-0.5, 0.5),
        RNG.uniform(-0.5, 0.5),
        RNG.uniform(-0.5, 0.5),
    ])
    np_w_target = np.array(np_mixer_inversion(np_cfg, list(desired)))
    t_w_target = tp.mixer_inversion(t_cfg, torch.tensor(desired, dtype=torch.float64).unsqueeze(0))[0]
    check(f"case{case}.mixer_inversion", np_w_target, t_w_target)

    # --- motor_lag: random current/target rpm ---
    w_current = RNG.uniform(0, MAX_RPM, size=4)
    w_target = RNG.uniform(0, MAX_RPM, size=4)
    np_w_new = np.array([np_motor_lag(w_current[i], w_target[i], np_cfg.rotors[i].motor_tau, DT) for i in range(4)])
    t_w_new = tp.motor_lag(
        torch.tensor(w_current, dtype=torch.float64).unsqueeze(0),
        torch.tensor(w_target, dtype=torch.float64).unsqueeze(0),
        t_cfg.motor_tau, DT,
    )[0]
    check(f"case{case}.motor_lag", np_w_new, t_w_new)

    # --- thrust/torque from a random rotor-speed vector ---
    w = RNG.uniform(0, MAX_RPM, size=4)
    w_t = torch.tensor(w, dtype=torch.float64).unsqueeze(0)
    check(f"case{case}.net_combining_thrust", np_net_combining_thrust(np_cfg, w), tp.net_combining_thrust(t_cfg, w_t)[0])
    check(f"case{case}.net_combining_torque", np_net_combining_torque(np_cfg, w), tp.net_combining_torque(t_cfg, w_t)[0])

    # --- drag: random velocity, including near-zero to exercise the <0.01 branch ---
    vel = RNG.uniform(-15, 15, size=3) if case % 8 != 0 else RNG.uniform(-0.005, 0.005, size=3)
    np_drag = np_drag_force(velocity=list(vel), drag_coeff=DRAG_COEFF, cross_sec_area=0.05, air_dens=1.225)
    t_drag = tp.drag_force(torch.tensor(vel, dtype=torch.float64).unsqueeze(0), DRAG_COEFF, 0.05, 1.225)[0]
    check(f"case{case}.drag_force", np_drag, t_drag)

    # --- wind ---
    wind_vec = RNG.uniform(-2, 2, size=3)
    np_w_force = np_wind(wind=list(wind_vec), mass=MASS, k_wind_coeff=0.1)
    t_w_force = tp.wind_force(torch.tensor(wind_vec, dtype=torch.float64).unsqueeze(0), MASS, 0.1)[0]
    check(f"case{case}.wind_force", np_w_force, t_w_force)

print(f"\n=== result ===")
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
else:
    print(f"ALL {N_CASES} cases x 5 functions PASSED (atol={ATOL}, rtol={RTOL})")
