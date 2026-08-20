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
    python -m app.guidance.test_free_hover [checkpoint_path] [--save-video [path.mp4]]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import torch

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic
from app.reward_functions.rewards import RewardConfig, make_reward_fn

# RUNS_DIR = Path(__file__).parent / "runs"
# DEFAULT_CHECKPOINT = RUNS_DIR / "ppo_stage2_510000.pt"


DEFAULT_CHECKPOINT = "z_final_version_1m_10epoch/ppo_stage2_660000.pt"
MAX_VIDEO_SECONDS = 10


def _capture_frame():
    width, height, view_matrix, proj_matrix = p.getDebugVisualizerCamera()[:4]
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = np.reshape(np.array(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
    return width, height, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--save-video", nargs="?", const="hover_flight.mp4", default=None, metavar="PATH",
        help=f"save a video of the run (max {MAX_VIDEO_SECONDS}s) to PATH",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
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

    video_writer = None
    frame_interval = None
    max_frames = None
    frames_written = 0
    fps = env.metadata.get("render_fps", 30)
    if args.save_video:
        frame_interval = max(1, round((1 / fps) / env.dt))
        max_frames = fps * MAX_VIDEO_SECONDS
        width, height, first_frame = _capture_frame()
        video_writer = cv2.VideoWriter(
            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )
        video_writer.write(first_frame)
        frames_written += 1

    while not done and step < max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = model.forward(obs_t)
            action = model.scale_action(mean)
        action = action.squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

        if video_writer is not None and frames_written < max_frames and step % frame_interval == 0:
            _, _, frame = _capture_frame()
            video_writer.write(frame)
            frames_written += 1

        if step % 20 == 0 or done:
            print(f"step {step:4d}  dist={env.prev_distance:.4f}  reason={info['reason']}")

        if video_writer is not None and frames_written >= max_frames:
            video_writer.release()
            print(f"Saved {frames_written / fps:.1f}s video to {args.save_video}")
            video_writer = None

    if video_writer is not None:
        video_writer.release()
        print(f"Saved {frames_written / fps:.1f}s video to {args.save_video}")

    if not done:
        print(f"reached max_steps={max_steps} without terminating -- "
              f"still going, dist={env.prev_distance:.4f}")


if __name__ == "__main__":
    main()
