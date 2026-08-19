"""
Diagnostic: load a trained checkpoint, spawn it at (0,0,5) (== target),
and let it run WITHOUT the hover_success early-termination -- normally the
episode ends the instant the drone holds the zone for 200 steps
(hover_success_steps), so we've never actually watched what it does past
that point. This disables that early exit (hover_success_steps set huge)
so the episode keeps running under the SAME other failure conditions
(attitude/oob/moving_away_cap) and prints distance-to-target over time --
if the policy only learned to survive exactly ~200 steps and then falls
apart, this is what will show it.

Usage:
    python -m app.guidance.test_free_hover [checkpoint_path] [max_steps]
"""
import sys
from pathlib import Path

import numpy as np
import torch

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic
from app.reward_functions.rewards import RewardConfig, make_reward_fn

# RUNS_DIR = Path(__file__).parent / "runs"
# DEFAULT_CHECKPOINT = RUNS_DIR / "ppo_stage2_510000.pt"


DEFAULT_CHECKPOINT = "runs/1m_10epochs_v2/ppo_stage2_660000.pt"
def main():
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    max_steps =  10000000

    reward_cfg = RewardConfig(
        oob_radius=7000,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=-0.05,
        attitude_roll_deg=90,
        hover_success_steps=10000 ** 9,  # effectively disables the early "success" exit
        streak_cap=10000,

    )
    reward_fn = make_reward_fn(reward_cfg)
    
    env = InterceptorDroneEnv(reward_fn, render_mode="human")

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded {checkpoint}")

    obs, _ = env.reset(start_pos=np.array([0, 0, 5], dtype=np.float32),
                        target_pos=np.array([0, 0, 5], dtype=np.float32))
    done = False
    step = 0
    info = {"reason": None}

    while not done and step < max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = model.forward(obs_t)
            action = model.scale_action(mean)
        action = action.squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

        if step % 20 == 0 or done:
            print(f"step {step:4d}  dist={env.prev_distance:.4f}  reason={info['reason']}")

    if not done:
        print(f"reached max_steps={max_steps} without terminating -- "
              f"still going, dist={env.prev_distance:.4f}")


if __name__ == "__main__":
    main()
