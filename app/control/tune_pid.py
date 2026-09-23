"""Derives PID gains per target distance by closed-form pole-placement math --
no Optuna, no trial-and-error search. Two coupled loops, each solved from the
actual drone physics (app/dynamics/drone.py's create_quad_config numbers) and
the actuator limit that makes a single gain set unable to cover every distance
(see the note below):

Position loop (per axis): compute_action treats accel_cmd = kp_pos*err -
kd_pos*vel directly as the commanded acceleration (the desired-tilt mapping
is accel/g, so as long as the attitude loop tracks its setpoint fast relative
to the position loop, pos'' ~= accel_cmd). That's a plain mass-normalized
double integrator, so standard 2nd-order pole placement applies directly:
    kp_pos = wn_pos**2
    kd_pos = 2*zeta_pos*wn_pos
wn_pos is chosen from a settling-time BUDGET (steps_for_dist(dist)*dt), not a
fixed constant -- shorter distances get a smaller time budget per the same
formula, so they naturally get a larger wn (tighter/faster loop) and longer
distances get a smaller wn (looser/slower loop). This is exactly the fix for
the problem tune_pid.py always had: one gain set tuned tight for a short
error either can't close a long one in time, or (reused at short range) is
so aggressive it saturates max_tilt_rad and overshoots.

Attitude loop (roll/pitch) and yaw loop: same pole-placement identity but
using I*angular_accel = torque directly (angular_acceleration() in
dynamics/methods.py divides net_torque by inertia per axis), so
    kp_att = I_xx * wn_att**2,  kd_att = 2*zeta_att*wn_att*I_xx
    kp_yaw = I_zz * wn_yaw**2,  kd_yaw = 2*zeta_yaw*wn_yaw*I_zz
Both inner loops are set to a fixed multiple of the position loop's own wn
(cascade-control bandwidth separation -- an inner loop needs to track its
setpoint much faster than the outer loop moves it, or the two fight each
other). Yaw's multiple is deliberately smaller than roll/pitch's: yaw torque
authority comes from k_m = kf_km_ratio * k_f (create_quad_rotors), only 2% of
the thrust coefficient, so the same-size wn_yaw as roll/pitch would demand
torque the rotors can't actually deliver without saturating max_rpm.
"""
import json

import numpy as np

from app.dynamics.drone import create_quad_config
from app.control.pid import PIDController
from app.control.step_budget import DT, MIN_STEPS, steps_for_dist
from app.environmental.base_drone_env import BaseDroneEnv
from app.reward_functions.rewards import reward_func
from app.training.eval_matrix import build_eval_pairs, run_eval_matrix, make_pid_action_fn

# One gains set doesn't hold across distances: max_tilt_rad caps commanded tilt
# regardless of target distance, so gains tuned tight for a short error can't
# close a long one, while gains loose enough for long range overshoot on short
# ones. Solve a separate gains set per distance instead.
DISTANCES = (3, 10, 50, 100, 150, 250)

G = 9.81
MAX_TILT_RAD = 0.3  # must match PIDController's default

# Must match the QuadConfig base_drone_env.reset() actually builds -- gains
# derived from any other mass/inertia would be tuned for the wrong plant.
_CONFIG = create_quad_config(
    mass=1.5, inertia=(0.02, 0.02, 0.04), arm_length=0.22,
    drag_coeff=0.035, max_rpm=12000, motor_tau=0.05,
)
I_ROLL_PITCH = _CONFIG.inertia[0]  # Ixx == Iyy
I_YAW = _CONFIG.inertia[2]

# Must match the hit_threshold every _make_verify_env / collect_demonstrations.py
# / dagger.py / verify_pid.py RewardFnPhase1 construction uses -- the settling
# time below is solved for actually reaching this precision, not a generic
# settling-time definition (see compute_gains_for_distance's docstring).
HIT_THRESHOLD = 0.05

# Damping ratios: >1 an overdamped attitude/yaw loop would fight the position
# loop's own settling time; slightly under 1 gives a fast, ~no-overshoot response.
ZETA_POS = 0.85
ZETA_ATT = 0.85
ZETA_YAW = 0.9

# Inner-loop-must-be-faster-than-outer-loop bandwidth separation. Yaw's is
# lower than roll/pitch's -- see module docstring on k_m authority.
BANDWIDTH_SEP_ATT = 5.0
BANDWIDTH_SEP_YAW = 2.0

# Fraction of the per-distance step budget the position loop is asked to
# settle within, leaving the rest as margin (jitter, off-axis approach,
# imitation noise) rather than tuning to the exact deadline.
SETTLE_TIME_FRACTION = 0.5


def compute_gains_for_distance(dist):
    """Pure math, no simulation: returns the PIDController kwargs for `dist`,
    plus a couple of diagnostic numbers (wn_pos, saturation_ratio) useful for
    sanity-checking the result.

    wn_pos is NOT solved from the generic "settling time = 4/(zeta*wn)" (2%-
    of-initial-error) rule of thumb -- that rule targets error decaying to 2%
    of `dist`, e.g. 5m of slack at dist=250m, wildly looser than this
    controller's actual HIT_THRESHOLD=0.05m target. The required number of
    time-constants to decay from `dist` down to HIT_THRESHOLD is
    ln(dist/HIT_THRESHOLD), which itself grows with distance (~4.1 at 3m,
    ~8.5 at 250m) -- using the fixed "4" here under-budgets wn at long range
    and the controller simply runs out of allotted steps just short of
    HIT_THRESHOLD (verified: at 250m it reached 0.073m, just above the 0.05m
    target, right at the step budget)."""
    settle_time = steps_for_dist(dist) * DT * SETTLE_TIME_FRACTION

    # error(t) ~= dist * exp(-zeta*wn*t) for a near-critically-damped 2nd order
    # system; solving error(settle_time) = HIT_THRESHOLD for wn:
    wn_pos = np.log(dist / HIT_THRESHOLD) / (ZETA_POS * settle_time)
    kp_pos = wn_pos ** 2
    kd_pos = 2 * ZETA_POS * wn_pos

    wn_att = BANDWIDTH_SEP_ATT * wn_pos
    kp_att = I_ROLL_PITCH * wn_att ** 2
    kd_att = 2 * ZETA_ATT * wn_att * I_ROLL_PITCH

    wn_yaw = BANDWIDTH_SEP_YAW * wn_pos
    kp_yaw = I_YAW * wn_yaw ** 2
    kd_yaw = 2 * ZETA_YAW * wn_yaw * I_YAW

    # Purely informational: >1 means the position loop commands more than
    # max_tilt_rad at the initial (full-distance) error, i.e. it starts
    # saturated and coasts at max tilt before desaturating on approach --
    # expected and fine for long distances, worth a glance for short ones.
    saturation_ratio = (kp_pos * dist) / (G * MAX_TILT_RAD)

    gains = {
        "kp_pos": kp_pos, "kd_pos": kd_pos,
        "kp_att": kp_att, "kd_att": kd_att,
        "kp_yaw": kp_yaw, "kd_yaw": kd_yaw,
        "max_tilt_rad": MAX_TILT_RAD,
    }
    return gains, {"wn_pos": wn_pos, "saturation_ratio": saturation_ratio}


def _make_verify_env(max_steps):
    # reward_func (not the deprecated RewardFnPhase1 roadmap) -- the current
    # reward version. Its oob_radius scales with env.start_dist (see
    # rewards.reward_func), so it's safe across the full DISTANCES ladder
    # including 250m. pid_gains_path=None sidesteps the chicken-and-egg
    # problem of BaseDroneEnv's default pid_teacher load -- best_pid_gains.json
    # is exactly what this script is computing, and reward_func needs no
    # pid_teacher anyway (unlike the old phase_1_imitation stage).
    #
    # max_steps must be passed explicitly: BaseDroneEnv defaults to 15_000,
    # but steps_for_dist(100/150/250) needs 25k/37.5k/62.5k -- leaving the
    # default would silently truncate every long-distance episode long before
    # the controller has a chance to converge, well before max_steps even
    # becomes the loop bound a caller thinks it's using.
    return BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=max_steps)


def verify_gains(dist, gains, n_repeats=2, target_yaws=(0.0, np.pi / 2)):
    """One-shot sanity check (not a search -- gains are fixed already): runs
    the analytically-derived gains through the real sim, every x/y/z
    direction, at a couple of target yaws, and reports hit rate. Purely
    diagnostic logging to catch a derivation bug, same spirit as
    verify_pid.py's smoke test."""
    pid = PIDController(**gains)
    oob_radius = max(20.0, dist * 3.0)
    max_steps = steps_for_dist(dist)
    env = _make_verify_env(max_steps)

    worst_hit_rate = 1.0
    for target_yaw in target_yaws:
        pairs = [(s, t, target_yaw) for s, t, _ in
                 build_eval_pairs(oob_radius=oob_radius, distances=(dist,), axes=(0, 1, 2))]
        results = run_eval_matrix(env, make_pid_action_fn(pid), pairs=pairs,
                                   n_repeats=n_repeats, max_steps=max_steps, on_episode_reset=pid.reset)
        hit_rate = float(np.mean([r["hit_rate"] for r in results]))
        worst_hit_rate = min(worst_hit_rate, hit_rate)
        print(f"    target_yaw={target_yaw:+.2f}rad  hit_rate={hit_rate:.2f}")

    return worst_hit_rate


def measure_max_episode_reward(env, pid, start_pos, target_pos, target_yaw, max_steps):
    """Runs `pid` for one episode (fixed start/target/yaw) against the real
    app.reward_functions.rewards.reward_func, returning the cumulative
    reward -- phi-shaping + milestones + the hit bonus if the PID actually
    reaches it. Building block for calibrate_approach_milestone_budget."""
    obs, _ = env.reset(start_pos=start_pos.copy(), target_pos=target_pos.copy(), target_yaw=target_yaw)
    pid.reset()
    total = 0.0
    for _ in range(max_steps):
        action = pid.compute_action(env.drone_state, env.target_pos, env.target_yaw)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


def calibrate_approach_milestone_budget(distances=(3, 10), n_episodes=5, seed=0):
    """Measures the tuned PID's max achievable reward_func total (hit bonus
    included) across `distances`, using each distance's own gains and a
    random direction/yaw per episode, and returns the max observed -- the
    value to hand-copy into rewards.APPROACH_MILESTONE_BUDGET. Rerun this
    (and update that constant) whenever HIT_REWARD, the milestone bonuses,
    or best_pid_gains_per_dist.json change -- it's not read automatically.

    reward_func reads APPROACH_MILESTONE_BUDGET as a live module constant to
    compute its own step_penalty, so measuring "the max reward, budget
    included" while the module's CURRENT (possibly stale, possibly what
    we're about to overwrite) budget value is still driving step_penalty
    would contaminate the very number we're trying to derive -- self-
    referential. Budget is defined as the max reward EXCLUDING time
    pressure, so step_penalty is forced to 0 for the duration of this
    measurement (module constant patched, then restored) regardless of
    whatever value happens to be sitting in rewards.py right now."""
    import app.reward_functions.rewards as rewards_module

    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    rng = np.random.default_rng(seed)
    base = np.array([0, 0, 5], dtype=np.float32)
    best = 0.0
    original_budget = rewards_module.APPROACH_MILESTONE_BUDGET
    rewards_module.APPROACH_MILESTONE_BUDGET = 0.0
    try:
        for dist in distances:
            max_steps = steps_for_dist(dist)
            env = BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=max_steps)
            pid = PIDController(**gains_by_dist[str(dist)])
            for _ in range(n_episodes):
                target = None
                while target is None:
                    v = rng.normal(size=3)
                    v /= np.linalg.norm(v)
                    candidate = (base + (v * dist).astype(np.float32))
                    if candidate[2] >= 0:
                        target = candidate
                target_yaw = float(rng.uniform(-np.pi, np.pi))
                total = measure_max_episode_reward(env, pid, base, target, target_yaw, max_steps)
                best = max(best, total)
    finally:
        rewards_module.APPROACH_MILESTONE_BUDGET = original_budget
    return best


if __name__ == "__main__":
    per_distance_gains = {}

    for dist in DISTANCES:
        gains, diag = compute_gains_for_distance(dist)
        print(f"\n=== dist={dist}m ===")
        print(f"  wn_pos={diag['wn_pos']:.3f} rad/s  saturation_ratio={diag['saturation_ratio']:.2f} "
              f"({'starts saturated, coasts at max tilt' if diag['saturation_ratio'] > 1 else 'never saturates'})")
        print(f"  gains={ {k: round(v, 4) for k, v in gains.items()} }")

        hit_rate = verify_gains(dist, gains)
        status = "OK" if hit_rate > 0 else "NO HIT -- inspect before using these gains"
        print(f"  worst-case hit_rate={hit_rate:.2f}  [{status}]")

        per_distance_gains[str(dist)] = gains

    with open("app/control/best_pid_gains_per_dist.json", "w") as f:
        json.dump(per_distance_gains, f, indent=2)
    print("\nSaved all per-distance gains to app/control/best_pid_gains_per_dist.json")

    # BaseDroneEnv's own default pid_teacher (used before SubprocVecBaseDroneEnv's
    # per-episode _select_pid_teacher swap kicks in) needs one generic gain
    # set -- middle of the distance ladder is the least-bad single compromise.
    generic_dist = DISTANCES[len(DISTANCES) // 2]
    with open("app/control/best_pid_gains.json", "w") as f:
        json.dump(per_distance_gains[str(generic_dist)], f, indent=2)
    print(f"Saved generic (dist={generic_dist}m) gains to app/control/best_pid_gains.json")
