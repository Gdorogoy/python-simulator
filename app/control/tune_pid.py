"""Closed-form (pole-placement) PID gains per target distance; no search. See docs.md "PID tuning"."""
import json

import numpy as np

from app.dynamics.drone import create_quad_config
from app.control.pid import PIDController
from app.control.step_budget import DT, MIN_STEPS, steps_for_dist
from app.environmental.base_drone_env import BaseDroneEnv
from app.reward_functions.rewards import reward_func
from app.training.eval_matrix import build_eval_pairs, run_eval_matrix, make_pid_action_fn

# one gain set can't cover every distance (max_tilt_rad saturation), so solve one per distance
DISTANCES = (3, 10, 20, 30, 50, 100, 150, 250)

G = 9.81
MAX_TILT_RAD = 0.3  # must match PIDController's default

# must match the QuadConfig base_drone_env.reset() builds
_CONFIG = create_quad_config(
    mass=1.5, inertia=(0.02, 0.02, 0.04), arm_length=0.22,
    drag_coeff=0.035, max_rpm=12000, motor_tau=0.05,
)
I_ROLL_PITCH = _CONFIG.inertia[0]  # Ixx == Iyy
I_YAW = _CONFIG.inertia[2]

# precision the settling time is solved for (tighter than the 0.25 m hit radius)
HIT_THRESHOLD = 0.05

# damping ratios: slightly under 1 = fast with ~no overshoot
ZETA_POS = 0.85
ZETA_ATT = 0.85
ZETA_YAW = 0.9

# inner loops must be faster than the outer loop; yaw lower (weak k_m authority)
BANDWIDTH_SEP_ATT = 5.0
BANDWIDTH_SEP_YAW = 2.0

# settle within this fraction of the step budget; the rest is margin
SETTLE_TIME_FRACTION = 0.5


def compute_gains_for_distance(dist):
    """PIDController kwargs for `dist` (+ wn_pos, saturation_ratio). wn from ln(dist/HIT_THRESHOLD), see docs.md."""
    settle_time = steps_for_dist(dist) * DT * SETTLE_TIME_FRACTION

    # error(t) ~= dist*exp(-zeta*wn*t); solve error(settle_time) = HIT_THRESHOLD for wn
    wn_pos = np.log(dist / HIT_THRESHOLD) / (ZETA_POS * settle_time)
    kp_pos = wn_pos ** 2
    kd_pos = 2 * ZETA_POS * wn_pos

    wn_att = BANDWIDTH_SEP_ATT * wn_pos
    kp_att = I_ROLL_PITCH * wn_att ** 2
    kd_att = 2 * ZETA_ATT * wn_att * I_ROLL_PITCH

    wn_yaw = BANDWIDTH_SEP_YAW * wn_pos
    kp_yaw = I_YAW * wn_yaw ** 2
    kd_yaw = 2 * ZETA_YAW * wn_yaw * I_YAW

    # informational: >1 means the loop starts saturated at max tilt (fine at long range)
    saturation_ratio = (kp_pos * dist) / (G * MAX_TILT_RAD)

    gains = {
        "kp_pos": kp_pos, "kd_pos": kd_pos,
        "kp_att": kp_att, "kd_att": kd_att,
        "kp_yaw": kp_yaw, "kd_yaw": kd_yaw,
        "max_tilt_rad": MAX_TILT_RAD,
    }
    return gains, {"wn_pos": wn_pos, "saturation_ratio": saturation_ratio}


def _make_verify_env(max_steps):
    # pid_gains_path=None: best_pid_gains.json is what this script produces; max_steps per distance
    return BaseDroneEnv(reward_func, pid_gains_path=None, max_steps=max_steps)


def verify_gains(dist, gains, n_repeats=2, target_yaws=(0.0, np.pi / 2)):
    """Sanity check of the derived gains in the real sim (all axes, a few yaws); logs hit rate."""
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
    """Cumulative reward_func total of one PID episode (used by calibrate_approach_milestone_budget)."""
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
    """Max reward_func total of the tuned PID (step penalty forced to 0) -> copy into rewards.APPROACH_MILESTONE_BUDGET."""
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

    # generic gains for BaseDroneEnv's default teacher: middle of the distance ladder
    generic_dist = DISTANCES[len(DISTANCES) // 2]
    with open("app/control/best_pid_gains.json", "w") as f:
        json.dump(per_distance_gains[str(generic_dist)], f, indent=2)
    print(f"Saved generic (dist={generic_dist}m) gains to app/control/best_pid_gains.json")
