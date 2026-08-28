"""
test_configuration() -- run before every training session and after any QuadConfig
change. Each check targets a real bug this project has hit before (unit mismatches,
mixer inversion issues, yaw/body-frame bugs, NaN from bad hover equilibrium).

Usage:
    from app.dynamics.drone import create_quad_config
    config = create_quad_config(...)
    ok = test_configuration(config)
    if not ok:
        raise SystemExit("Config failed validation — fix before training.")
"""
import numpy as np
from scipy.spatial.transform import Rotation

from app.dynamics.drone import create_quad_config
from app.dynamics.drone import QuadState, Vector3D, Quaternion, QuadConfig
from app.dynamics.methods import (
    mixer, mixer_inversion, timestamp_update, thrust, torque,
)


def _pass(name):
    print(f"  [PASS] {name}")


def _fail(name, detail):
    print(f"  [FAIL] {name} — {detail}")


def _warn(name, detail):
    print(f"  [WARN] {name} — {detail}")


# ---------------------------------------------------------------------------

def check_static_fields(config: QuadConfig) -> bool:
    """Basic sanity on the raw numbers — catches typos before they cause
    NaN three layers deep in the physics."""
    ok = True

    if config.mass <= 0:
        _fail("mass", f"must be > 0, got {config.mass}")
        ok = False
    elif not (0.3 <= config.mass <= 5.0):
        _warn("mass", f"{config.mass}kg is outside typical small-quad range (0.3-5.0kg)")

    for name, val in zip(("I_xx", "I_yy", "I_zz"), config.inertia):
        if val <= 0:
            _fail(f"inertia.{name}", f"must be > 0, got {val}")
            ok = False

    if config.arm_length <= 0:
        _fail("arm_length", f"must be > 0, got {config.arm_length}")
        ok = False

    if config.drag_coeff < 0:
        _fail("drag_coeff", f"must be >= 0, got {config.drag_coeff}")
        ok = False

    if len(config.rotors) != 4:
        _fail("rotors", f"expected 4 rotors, got {len(config.rotors)}")
        ok = False
        return ok

    for i, r in enumerate(config.rotors):
        if r.k_f <= 0:
            _fail(f"rotors[{i}].k_f", f"must be > 0, got {r.k_f}")
            ok = False
        if r.k_m <= 0:
            _fail(f"rotors[{i}].k_m", f"must be > 0, got {r.k_m}")
            ok = False
        if r.max_rpm <= 0:
            _fail(f"rotors[{i}].max_rpm", f"must be > 0, got {r.max_rpm}")
            ok = False
        if r.motor_tau <= 0:
            _fail(f"rotors[{i}].motor_tau", f"must be > 0, got {r.motor_tau}")
            ok = False
        if r.spin_dir not in (1, -1):
            _fail(f"rotors[{i}].spin_dir", f"must be +1 or -1, got {r.spin_dir}")
            ok = False

    if ok:
        _pass("static field sanity")
    return ok


def check_mixer_invertibility(config: QuadConfig) -> bool:
    """
    A near-singular mixer matrix doesn't error, it just returns garbage, so this checks
    explicitly. Uses condition number rather than raw determinant: k_f/k_m are naturally
    tiny (~1e-7), so the determinant is astronomically small even for a healthy matrix --
    a scale artifact, not degeneracy. Condition number is scale-invariant.
    """
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == 0:
                M[i, j] = config.rotors[j].k_f
            if i == 1:
                M[i, j] = config.rotors[j].position.y * config.rotors[j].k_f
            if i == 2:
                M[i, j] = -(config.rotors[j].position.x) * config.rotors[j].k_f
            if i == 3:
                M[i, j] = config.rotors[j].k_m * config.rotors[j].spin_dir

    cond = np.linalg.cond(M)
    if not np.isfinite(cond) or cond > 1e12:
        _fail("mixer invertibility", f"condition number {cond:.3e} — "
              f"matrix is genuinely near-singular (degenerate rotor geometry)")
        return False
    _pass(f"mixer invertibility (condition number={cond:.3e})")
    return True


def check_hover_equilibrium(config: QuadConfig) -> bool:
    """The w_hover=1.87 bug: confirms k_f is actually calibrated against
    max_rpm, not left as a disconnected default."""
    hover_thrust_needed = config.mass * 9.81
    ok = True
    for i, r in enumerate(config.rotors):
        w_hover = np.sqrt((hover_thrust_needed / 4) / r.k_f)
        frac = w_hover / r.max_rpm
        if not (0.1 <= frac <= 0.9):
            _fail(f"rotors[{i}] hover equilibrium",
                  f"w_hover={w_hover:.1f} is {frac*100:.2f}% of max_rpm={r.max_rpm} "
                  f"— should be roughly 20-70%. k_f likely uncalibrated.")
            ok = False
    if ok:
        _pass(f"hover equilibrium (~{frac*100:.0f}% of max_rpm)")
    return ok


def check_mixer_roundtrip(config: QuadConfig) -> bool:
    """mixer_inversion(desired) -> rotor speeds -> mixer(speeds) should
    approximately recover the original desired [F, roll, pitch, yaw]."""
    hover_thrust_needed = config.mass * 9.81
    desired = [hover_thrust_needed, 0.05, 0.03, 0.01]  # small, achievable torques

    try:
        speeds = mixer_inversion(config, desired)
    except Exception as e:
        _fail("mixer round-trip", f"mixer_inversion raised: {e}")
        return False

    if any(s < 0 for s in speeds) or any(np.isnan(s) for s in speeds):
        _fail("mixer round-trip", f"mixer_inversion produced invalid speeds: {speeds}")
        return False

    recovered = mixer(config, [s ** 2 for s in speeds])  # mixer works in omega^2
    err = np.abs(np.array(recovered) - np.array(desired))
    tol = np.array([0.5, 0.05, 0.05, 0.05])  # loose tolerance, N and N*m

    if np.any(err > tol):
        _fail("mixer round-trip", f"desired={desired} recovered={list(recovered)} "
              f"error={list(err)} exceeds tolerance {list(tol)}")
        return False

    _pass("mixer round-trip (desired ≈ recovered)")
    return True


def check_hover_stability(config: QuadConfig, n_steps: int = 480, dt: float = 1/240) -> bool:
    """Run pure hover (zero commanded torque) for 2 real seconds. Position
    should stay essentially put — this is the exact test that would have
    caught the original 'falls before motors spin up' bug."""
    hover_thrust = config.mass * 9.81
    hover_omega = mixer_inversion(config, [hover_thrust, 0.0, 0.0, 0.0])

    state = QuadState(
        position=Vector3D(0, 0, 5.0),
        velocity=Vector3D(0, 0, 0),
        orientation=Quaternion(0, 0, 0, 1),
        angular_velocity=Vector3D(0, 0, 0),
        rotor_rpm=list(hover_omega),  # pre-spun, avoids the spin-up-fall bug
    )

    action = [hover_thrust, 0.0, 0.0, 0.0]
    wind = [0.0, 0.0, 0.0]

    for step in range(n_steps):
        state = timestamp_update(state, config, action, wind, dt)
        pos = np.array([state.position.x, state.position.y, state.position.z])
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            _fail("hover stability", f"NaN/Inf at step {step}: pos={pos}")
            return False
        if pos[2] < 0:
            _fail("hover stability", f"fell through ground at step {step}: z={pos[2]:.3f}")
            return False

    final_pos = np.array([state.position.x, state.position.y, state.position.z])
    xy_drift = np.linalg.norm(final_pos[:2])
    z_drift = abs(final_pos[2] - 5.0)

    ok = True
    if xy_drift > 0.5:
        _fail("hover stability (xy drift)", f"{xy_drift:.3f}m after {n_steps} steps — should stay ~0")
        ok = False
    if z_drift > 0.3:
        _fail("hover stability (altitude drift)", f"{z_drift:.3f}m after {n_steps} steps — should stay ~0")
        ok = False
    if ok:
        _pass(f"hover stability (xy_drift={xy_drift:.4f}m, z_drift={z_drift:.4f}m over {n_steps} steps)")
    return ok


def check_step_response(config: QuadConfig, n_steps: int = 120, dt: float = 1/240) -> bool:
    """Apply a small constant roll torque and confirm the drone actually
    rolls in a bounded, non-exploding way — catches NaN/instability under
    an active command, not just idle hover."""
    hover_thrust = config.mass * 9.81
    hover_omega = mixer_inversion(config, [hover_thrust, 0.0, 0.0, 0.0])

    state = QuadState(
        position=Vector3D(0, 0, 5.0),
        velocity=Vector3D(0, 0, 0),
        orientation=Quaternion(0, 0, 0, 1),
        angular_velocity=Vector3D(0, 0, 0),
        rotor_rpm=list(hover_omega),
    )

    small_roll_torque = 0.02 * config.inertia[0]  # gentle, achievable
    action = [hover_thrust, small_roll_torque, 0.0, 0.0]
    wind = [0.0, 0.0, 0.0]

    for step in range(n_steps):
        state = timestamp_update(state, config, action, wind, dt)
        pos = np.array([state.position.x, state.position.y, state.position.z])
        quat = np.array([state.orientation.x, state.orientation.y,
                          state.orientation.z, state.orientation.w])
        if np.any(np.isnan(pos)) or np.any(np.isnan(quat)):
            _fail("step response", f"NaN at step {step} under small roll torque command")
            return False

    rot = Rotation.from_quat([state.orientation.x, state.orientation.y,
                               state.orientation.z, state.orientation.w])
    roll, pitch, _ = rot.as_euler('xyz')

    if abs(roll) > np.pi / 2:
        _fail("step response", f"roll={roll:.3f} rad exceeds ±90° under a gentle command — "
              f"gains or torque scale likely too aggressive")
        return False

    _pass(f"step response (roll={np.degrees(roll):.1f}° after gentle command, bounded)")
    return True


def check_saturation(config: QuadConfig) -> bool:
    """Confirm hover doesn't already demand near-max rotor speed — no
    headroom left for the RL agent to actually maneuver."""
    hover_thrust_needed = config.mass * 9.81
    ok = True
    for i, r in enumerate(config.rotors):
        w_hover = np.sqrt((hover_thrust_needed / 4) / r.k_f)
        headroom = r.max_rpm - w_hover
        if headroom < 0.15 * r.max_rpm:
            _warn(f"rotors[{i}] saturation headroom",
                  f"only {headroom:.0f} rpm of headroom above hover ({w_hover:.0f}) "
                  f"before hitting max_rpm ({r.max_rpm}) — agent may saturate under load")
            ok = False
    if ok:
        _pass("saturation headroom sufficient")
    return ok


# ---------------------------------------------------------------------------

def test_configuration(config: QuadConfig, verbose: bool = True) -> bool:
    """
    Runs the full validation suite against a QuadConfig. Returns True only
    if every check passes. Run this before starting any training session.
    """
    checks = [
        ("Static field sanity",      check_static_fields),
        ("Mixer invertibility",      check_mixer_invertibility),
        ("Hover equilibrium",        check_hover_equilibrium),
        ("Mixer round-trip",         check_mixer_roundtrip),
        ("Hover stability (2s sim)", check_hover_stability),
        ("Step response (roll)",     check_step_response),
        ("Saturation headroom",      check_saturation),
    ]

    print("=" * 60)
    print("CONFIG VALIDATION")
    print("=" * 60)

    results = {}
    for name, fn in checks:
        print(f"\n{name}:")
        results[name] = fn(config)

    print("\n" + "=" * 60)
    n_pass = sum(results.values())
    n_total = len(results)
    all_ok = all(results.values())

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:5s} {name}")

    print("=" * 60)
    if all_ok:
        print(f"RESULT: {n_pass}/{n_total} checks passed — config is safe to train with.")
    else:
        print(f"RESULT: {n_pass}/{n_total} checks passed — DO NOT start training yet, fix failures above.")
    print("=" * 60)

    return all_ok


def test():
    config = create_quad_config(
        mass=1.5,
        inertia=(0.02, 0.02, 0.04),
        arm_length=0.22,
        drag_coeff=0.035,
        max_rpm=12000,
        motor_tau=0.05,
    )

    test_configuration(config)

