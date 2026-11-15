import json
import subprocess
import sys

ISAAC_PYTHON = "/mnt/e/Isaac/env_isaaclab/Scripts/python.exe"
TRAIN_SCRIPT = "app/training/acro_rtl_train_isaac.py"

REWARD_CONFIG = {
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

PARAMS = {
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
    "--reward-json": json.dumps(REWARD_CONFIG),

    # PPO & Distillation
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

    # Eval & Curriculum / Runner
    "--eval-every": 5,
    "--stop-hit": 0,
    "--buckets": True,
    "--runner": True,
    "--runner-speed": 30,
    "--runner-escape-dist": 750,

    # Output & Execution
    "--checkpoint-dir": "runs/interceptor_v1_bckts_runner30",
    "--headless": True,
}


def main():
    cmd = [ISAAC_PYTHON, TRAIN_SCRIPT]
    for flag, value in PARAMS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    print("Executing command:\n", " ".join(cmd), "\n", flush=True)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()