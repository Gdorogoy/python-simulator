"""
Diagnostic: loads a trained checkpoint and lets it fly with the hover_success
early-termination effectively disabled, so behavior past the normal 200-step success
cutoff (which would otherwise end the episode) is visible -- reveals a policy that only
learned to survive to that point and then falls apart.

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

DEFAULT_CHECKPOINT = "/mnt/c/Users/Pc-Egor/PycharmProjects/python-simulator/app/z_final_version_5m_30epoch_50xyz/ppo_stage2_225000.pt"
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
        oob_radius=70000000,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=-0.05,
        attitude_roll_deg=90,
        hover_success_steps=10000 ** 9,  # effectively disables the early "success" exit
        streak_cap=10000,

    )
    reward_fn = make_reward_fn(reward_cfg)
    
    env = InterceptorDroneEnv(reward_fn, render_mode="None")

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded {checkpoint}")

    obs, _ = env.reset(start_pos=np.array([0, 0, 5], dtype=np.float32),
                        target_pos=np.array([-100,0,5], dtype=np.float32))
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

    hit_threshold = 0.05
    was_in_zone = False

    while not done and step < max_steps:
        try:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        except (RuntimeError, TypeError) as e:
            print(f"stopped at step {step}: lost a valid observation from the sim ({e}) "
                  f"-- likely the pybullet GUI window/X connection died, not a model issue")
            break

        with torch.no_grad():
            mean, _, _ = model.forward(obs_t)
            action = model.scale_action(mean)
        action = action.squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

        # make_reward_fn has no hit-detection (unlike RewardFnPhase1), so log zone
        # entry manually here without terminating, to preserve the "watch what
        # happens after success" purpose of this script.
        in_zone = env.prev_distance < hit_threshold
        if in_zone and not was_in_zone:
            print(f"step {step:4d}  HIT -- dist={env.prev_distance:.4f} (within {hit_threshold}m of target)")
        was_in_zone = in_zone

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
