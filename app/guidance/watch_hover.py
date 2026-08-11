"""
Load a trained checkpoint and watch it fly in the pybullet GUI.

Usage:
    python -m app.guidance.watch_hover [checkpoint_path] [n_episodes]

Defaults to the converged Phase 0 checkpoint from the 1m_10epochs_v4 run
(streak-confirmed 100% hover_success across timesteps 930000..1005000).
"""
import sys
from pathlib import Path

import numpy as np
import torch

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic
from app.reward_functions.rewards import RewardConfig, make_reward_fn

RUNS_DIR = Path(__file__).parent / "runs"
DEFAULT_CHECKPOINT = RUNS_DIR / "1m_10epochs_v4" / "ppo_stage1_1005000.pt"


def main():
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # same RewardConfig as train() in phase_0_training.py, so termination
    # reasons/behavior match what the checkpoint was actually trained under
    reward_cfg = RewardConfig(
        oob_radius=7,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=-0.05,
        hover_success_steps=200,
        streak_cap=15,
    )
    reward_fn = make_reward_fn(reward_cfg)

    env = InterceptorDroneEnv(reward_fn, render_mode="human")

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    print(f"Loaded {checkpoint}")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        info = {"reason": None}

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = model.forward(obs_t)
                action = model.scale_action(mean)
            action = action.squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated

        print(f"episode {ep + 1}/{n_episodes}: {step_count} steps, reason={info['reason']}")


if __name__ == "__main__":
    main()
