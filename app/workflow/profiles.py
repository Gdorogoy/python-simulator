"""Training profiles for the GitHub Actions runner (app/workflow/workflow.py, collect_results.py).

One entry per drone style, so switching acro <-> angle is `--profile angle` instead of editing code:
`script` is the trainer, `params` are its CLI flags (dict -> JSON, list -> one token per item, True -> bare flag),
`checkpoint_dir` is the run folder the trainer writes and collect_results copies back.

Add a mode by adding an entry to PROFILES. Any single value can be overridden at run time without editing this file:
    python3 app/workflow/workflow.py --profile angle --set num_envs=4096 --set total_timesteps=10000000

The `angle_ground_*` entries are the ground-start curriculum stages (one per distance band, no handoff, no threats).
They chain: each stage warm-starts from the previous stage's `best.pt` and distils from it, so run them in order.
Run it dry first against the runner's Isaac python (it prints the full command):
    python3 app/workflow/workflow.py --profile acro --dry-run

Runner paths are env-overridable (ISAAC_PYTHON, WIN_PROJECT); the defaults are that Windows/WSL machine.
"""

from __future__ import annotations

import os

ISAAC_PYTHON = os.environ.get(
    "ISAAC_PYTHON", "/mnt/e/Isaac/env_isaaclab/Scripts/python.exe"
)
WIN_PROJECT = os.environ.get(
    "WIN_PROJECT", "/mnt/c/Users/Pc-Egor/PycharmProjects/WelcomeScreen/python-simulator"
)

# ---- acro: the 1000-1500 m v4 speed run (previous stage's policy as BC init and teacher) ----
ACRO_V4_REWARD = {
    "speed_target": 50,
    "speed_accel": 16,
    "speed_end_speed": 22,
    "speed_leg": 50,
    "line_path_leg": 30,
    "line_lead_s": 0.5,
    "line_ref": 150,
    "attitude_soft_deg": 85,
    "attitude_leg": 8,
    "los_k": 0.1,
    "los_leg": 25,
    "los_gate_frac": 0.0,
    "los_aligned": 0.0,
    "los_a": 5,
    "los_b": 30,
    "closing_speed_k": 0.02,
    "closing_speed_ref": 20,
    "close_speed_leg": 10,
    "c_step": 0.03,
    "step_leg": 8,
    "closing_lambda": 2.5,
    "c_decel": 0.0,
    "tilt_limit_deg": 80,
    "k_miss": 2.0,
    "near_limit_s": 0,
    "r_time_bonus": 20,
    "time_bonus_budget_s": 62.5,
    "path_leg": 30,
    "path_slack": 0.10,
}

# ---- angle: the rc reward with head-on 600-800 m handoff and the LOS teacher (anlge_evader_600_40ms) ----
ANGLE_RC_REWARD = {
    "intercept_radius": 1.0,
    "closing_lambda": 1.5,
    "c_step": 0.03,
    "los_k": 0.10,
    "los_gate_frac": 0.0,
    "los_aligned": 0.0,
    "los_a": 15,
    "los_b": 60,
    "zone_time": 0.05,
    "post_hit_steps": 6,
    "r_zone_entry": 0,
    "tilt_limit_deg": 85,
    "roll_limit_deg": 360,
    "pitch_limit_deg": 360,
}

# ---- angle, ground start (no handoff, no threats): the rc2 reward on top of rc ----
# Per-step numbers are only on/off markers under auto_scale: the per-leg totals below are what the run actually pays.
ANGLE_RC2_REWARD = {
    "intercept_radius": 1.0,
    "hit_radius": 1.0,
    "zone_time": 0.0,  # swept hit: a crossing is one step long by construction
    "post_hit_steps": 10,
    "hit_swept": True,
    "tilt_limit_deg": 180,
    "roll_limit_deg": 360,
    "pitch_limit_deg": 360,
    "closing_lambda": 1.5,
    "closing_eps": 0.5,
    "axis_eps": 0.5,
    "los_gate_frac": 0.0,
    "los_aligned": 0.0,
    "los_a": 15,
    "los_b": 60,
    "los_c": 135,
    "los_far": 3.0,
    "los_min_speed": 0.5,
    "min_leg_s": 7.5,
    "max_leg_s": 62.5,
    # ---- per-leg totals (one leg = one unit of reward, whatever the distance) ----
    "hit_leg": 50,
    "zone_entry_leg": 7,
    "step_leg": 5,
    "los_leg": 3,
    "closing_leg": 15,
    "axis_leg": 15,
    "hover_accrual_leg": 5,
    "hover_leg": 45,
    "crash_leg": 45,
    "timeout_leg": 45,
    "intercepted_leg": 45,
    "near_leg": 45,
    # ---- speed: shortfall against the accel/brake-limited profile (per step) ----
    "speed_leg": 25,
    "speed_accel": 12,
    "speed_end_speed": 25,
    "speed_min_dist": 20,
    # ---- cross-track: linear + MSE, per METRE flown, scored line_lead_s ahead ----
    "line_path_leg": 15,
    "line_mse_leg": 10,
    "line_lead_s": 0.5,
    # ---- terminal straightness + pace ----
    "path_leg": 20,
    "path_slack": 0.15,
    "r_time_bonus": 15,
    # ---- closing speed (linear, per step) ----
    "closing_speed_k": 0.02,  # marker only; close_speed_leg is the total
    "close_speed_leg": 10,
    "closing_speed_ref": 20,
    "closing_speed_min_dist": 50,
    # ---- fly-by: a pass that misses, and loitering near the target ----
    "k_miss": 1.0,  # marker only; k_miss = miss_leg / miss_radius
    "miss_leg": 30,
    "miss_radius": 15,
    "r_near": 1.0,  # marker only; near_leg is the total
    "near_radius": 15,
    "near_limit_s": 6,
}

# name, distance low/high (the whole leg: the policy flies it from rest), max_speed,
# speed_target, line_ref (m of cross-track that costs a whole leg), time_bonus_speed, total_timesteps
ANGLE_GROUND_STAGES = (
    ("angle_ground_150_350", 150, 350, 45, 38, 35, 30, 128_000_000),
    ("angle_ground_300_750", 300, 750, 50, 42, 75, 34, 128_000_000),
    ("angle_ground_700_1250", 700, 1250, 55, 48, 125, 38, 160_000_000),
    ("angle_ground_1000_1500", 1000, 1500, 60, 52, 150, 42, 200_000_000),
)

ANGLE_GROUND_FIRST_BC = (
    "runs/angle_los_60_160/best.pt"  # stage 1 warm-start (the last LOS run)
)


def _angle_ground_profile(
    *,
    name,
    low,
    high,
    max_speed,
    speed_target,
    line_ref,
    time_bonus_speed,
    total_timesteps,
    bc_checkpoint,
    teacher,
):
    """One ground-start stage: no handoff (the policy flies the whole leg from rest at 5 m), no threats,
    rc2 reward; distance / speed / path / clock knobs come from the stage."""
    reward = dict(ANGLE_RC2_REWARD)
    reward.update(
        speed_target=speed_target,
        reference_speed=speed_target,  # a leg is measured at the cruise speed
        line_ref=line_ref,
        time_bonus_speed=time_bonus_speed,
    )
    return {
        "script": "app/training/angle_rtl_train_isaac.py",
        "checkpoint_dir": f"runs/{name}",
        "params": {
            # Environment & rollout: ground start, no handoff, no threats
            "--num_envs": 8192,
            "--num-steps-per-chunk": 256,
            "--total_timesteps": total_timesteps,
            "--distance-low": low,
            "--distance-high": high,
            "--from-rest": True,
            "--max-speed": max_speed,
            "--obs-position-mode": "height",
            "--obs-layout": "v2",
            "--num-threats": 0,
            "--num-allies": 0,
            # Reward
            "--reward": "rc2",
            "--reward-json": reward,
            # PPO
            "--gamma": 0.99,
            "--lam": 0.95,
            "--clip-eps": 0.2,
            "--vf-coef": 0.5,
            "--ent-coef": 0.0,
            "--epochs": 5,
            "--warmup-epochs": 8,
            "--num-minibatches": 32,
            "--target-kl": 0.01,
            "--max-grad-norm": 0.5,
            "--lr-actor": 1e-4,
            "--weight-decay": 1e-4,
            "--lr-critic": 3e-4,
            "--lr-min-ratio": 0.3,
            "--critic-hidden": 128,
            "--critic-layers": 2,
            "--init-std": 0.15,
            "--log-std-min": -3.5,
            "--log-std-max": -1.2,
            # Distillation: stage 1 from the LOS controller, later stages from the previous policy
            "--teacher": teacher,
            "--bc-checkpoint": bc_checkpoint,
            "--distill-start": 1.0,
            "--distill-end": 0.05,
            "--distill-decay-chunks": 5,
            "--distill-max": 5.0,
            "--distill-fade-speed": 6,
            "--teacher-cruise-gain": 4,
            "--los-stop-speed": max_speed,
            "--los-max-tilt-deg": 90,
            "--guard-margin": 0.05,
            "--guard-gain": 2,
            "--guard-boost-max": 20,
            "--settle-chunks": 2,
            "--warmup-chunks": 2,
            "--head-chunks": 2,
            "--lr-ramp-chunks": 3,
            # Eval
            "--eval-every": 5,
            "--eval-burnin": 600,
            "--pid-eval-every": 10,
            # Output
            "--headless": True,
        },
    }


PROFILES = {
    "acro": {
        "script": "app/training/acro_rtl_train_isaac.py",
        "checkpoint_dir": "runs/interceptor_v1_bckts_runner30",
        "params": {
            # Environment & rollout
            "--num_envs": 8192,
            "--num-steps-per-chunk": 256,
            "--total_timesteps": 52428800,
            "--distance-low": 1000,
            "--distance-high": 1500,
            "--obs-position-mode": "height",
            "--tilt-limit": 80,
            "--tilt-margin": 45,
            "--hit-radius": 1.0,
            "--max-speed": 60,
            # Reward
            "--reward": "v4",
            "--reward-json": ACRO_V4_REWARD,
            # PPO & distillation
            "--gamma": 0.995,
            "--lam": 1.0,
            "--bc-checkpoint": "stable_models/acro_1000_1500_40ms/best.pt",
            "--teacher": "acro",
            "--teacher-checkpoint": "stable_models/acro_1000_1500_40ms/best.pt",
            "--distill-start": 0.33,
            "--distill-end": 0.0,
            "--lr-actor": 3e-4,
            "--lr-min-ratio": 0.05,
            "--settle-chunks": 0,
            "--warmup-chunks": 2,
            "--head-chunks": 0,
            "--lr-ramp-chunks": 0,
            "--init-std": 0.12,
            "--log-std-max": -1.5,
            "--ent-coef": 0.001,
            "--guard-gain": 1,
            "--desync-steps": 256,
            # Eval & curriculum / runner
            "--eval-every": 5,
            "--stop-hit": 0,
            "--buckets": True,
            "--runner": True,
            "--runner-speed": 30,
            "--runner-escape-dist": 750,
            # Output
            "--headless": True,
        },
    },
    "angle": {
        "script": "app/training/angle_rtl_train_isaac.py",
        "checkpoint_dir": "runs/angle_evader_workflow",
        "params": {
            # Environment & rollout
            "--num_envs": 8192,
            "--num-steps-per-chunk": 256,
            "--total_timesteps": 128000000,
            "--distance-low": 100,
            "--distance-high": 250,
            "--spawn-speed-low": 8,
            "--spawn-speed-high": 12,
            "--handoff": True,
            "--max-speed": 45,
            "--prefix-low": 2,
            "--prefix-high": 5,
            "--spawn-dist-low": 600,
            "--spawn-dist-high": 800,
            "--teacher-cruise-gain": 4,
            "--lateral-frac": 0.3,
            "--handoff-tilt-deg": 15,
            "--distill-fade-speed": 6,
            "--obs-position-mode": "height",
            # Reward
            "--reward": "rc",
            "--reward-json": ANGLE_RC_REWARD,
            # PPO & distillation
            "--gamma": 0.99,
            "--lam": 0.95,
            "--settle-chunks": 2,
            "--warmup-chunks": 2,
            "--head-chunks": 2,
            "--lr-ramp-chunks": 3,
            "--epochs": 5,
            "--warmup-epochs": 8,
            "--num-minibatches": 32,
            "--target-kl": 0.01,
            "--max-grad-norm": 0.5,
            "--lr-actor": 1e-4,
            "--weight-decay": 1e-4,
            "--lr-critic": 3e-4,
            "--lr-min-ratio": 0.3,
            "--critic-hidden": 128,
            "--critic-layers": 2,
            "--init-std": 0.15,
            "--log-std-min": -3.5,
            "--log-std-max": -1.2,
            "--ent-coef": 0.0,
            "--teacher": "los",
            "--bc-checkpoint": "runs/angle_los_40_160_600/best.pt",
            "--distill-start": 1.0,
            "--distill-end": 0.05,
            "--distill-decay-chunks": 5,
            "--distill-max": 5.0,
            "--guard-margin": 0.05,
            "--guard-gain": 2,
            "--guard-boost-max": 20,
            # Threats
            "--num-threats": 5,
            "--active-threats": 1,
            "--threat-speed": 50,
            "--threat-speed-min": 45,
            "--threat-radius": 1.0,
            "--threat-spread": 30,
            "--threat-spawn-min": 15,
            "--threat-spawn": "geometry",
            "--threat-geometry-weights": (1.0, 0.0),
            "--threat-reaction-s": 0.65,
            "--threat-tail-gap": (3.0, 3.0),
            "--threat-jitter": 2,
            "--threat-prob": 0.6,
            "--threat-buffer": 0.5,
            "--threat-release-after": (0.0, 0.0),
            "--obs-layout": "v2",
            "--num-allies": 0,
            # Eval
            "--eval-every": 5,
            "--eval-burnin": 600,
            "--pid-eval-every": 10,
            # Output
            "--headless": True,
        },
    },
}


# ---- register the ground-start curriculum stages (stage 1 from the LOS run, then stage -> stage) ----
for _i, (
    _name,
    _low,
    _high,
    _max_speed,
    _speed_target,
    _line_ref,
    _tb_speed,
    _steps,
) in enumerate(ANGLE_GROUND_STAGES):
    PROFILES[_name] = _angle_ground_profile(
        name=_name,
        low=_low,
        high=_high,
        max_speed=_max_speed,
        speed_target=_speed_target,
        line_ref=_line_ref,
        time_bonus_speed=_tb_speed,
        total_timesteps=_steps,
        bc_checkpoint=ANGLE_GROUND_FIRST_BC
        if _i == 0
        else f"runs/{ANGLE_GROUND_STAGES[_i - 1][0]}/best.pt",
        teacher="los" if _i == 0 else "policy",
    )
del _i, _name, _low, _high, _max_speed, _speed_target, _line_ref, _tb_speed, _steps


def profile_names() -> list[str]:
    return sorted(PROFILES)


def get_profile(name: str) -> dict:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {profile_names()}")
    return PROFILES[name]
