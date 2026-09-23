"""
Diagnostic: loads a trained checkpoint and flies it at N_TESTS random targets,
each sampled Uniform(distance_low, distance_high) away in a random 2-axis or
3-axis direction (see app.control.collect_demonstrations.sample_omni_target) --
single-axis targets are skipped on purpose, since this is meant to probe
diagonal/off-axis generalization, not the easy axis-aligned case.

distance_low/distance_high are CLI flags, not hardcoded, so the same script
re-tests a wider curriculum stage later (e.g. --distance-high 50) without
editing this file -- see app.training.base_training's curriculum staging.

Usage:
    python -m app.guidance.test_free_hover [checkpoint_path]
        [--distance-low 3] [--distance-high 10] [--n-tests 5] [--seed N]
        [--save-video [path.mp4]]   # records the FIRST test only
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import torch

from app.control.collect_demonstrations import sample_omni_target
from app.control.step_budget import steps_for_dist
from app.environmental.base_drone_env import BaseDroneEnv
from app.guidance.train import ActorCritic
from app.reward_functions.rewards import RewardConfig, make_reward_fn

DEFAULT_CHECKPOINT = "app/control/pretrained_bc_dagger.pt"
MAX_VIDEO_SECONDS = 10
PRINT_EVERY = 20
TEST_CATEGORIES = ("pair", "triple")  # 2-axis / 3-axis targets only -- no single-axis


def _capture_frame():
    width, height, view_matrix, proj_matrix = p.getDebugVisualizerCamera()[:4]
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = np.reshape(np.array(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
    return width, height, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _run_episode(env, model, start, target, video_writer, fps, frame_interval, max_frames, frames_written):
    """Drives one episode to termination/truncation, printing drone/target
    position and distance every PRINT_EVERY steps. Returns
    (steps, reason, final_dist, frames_written)."""
    obs, _ = env.reset(start_pos=start, target_pos=target)
    done = False
    step = 0
    info = {"reason": None}

    while not done and step < env.max_steps:
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

        if video_writer is not None and frames_written < max_frames and step % frame_interval == 0:
            _, _, frame = _capture_frame()
            video_writer.write(frame)
            frames_written += 1

        if step % PRINT_EVERY == 0 or done:
            pos = env.drone_state.position
            print(f"step {step:4d}  drone_pos=[{pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}]  "
                  f"target_pos=[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]  "
                  f"dist={env.prev_distance:.3f}")

    return step, info.get("reason"), env.prev_distance, frames_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--distance-low", type=float, default=3.0)
    parser.add_argument("--distance-high", type=float, default=10.0)
    parser.add_argument("--n-tests", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save-video", nargs="?", const="hover_flight.mp4", default=None, metavar="PATH",
        help=f"save a video of the FIRST test only (max {MAX_VIDEO_SECONDS}s) to PATH",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    rng = np.random.default_rng(args.seed)

    # Same OOB_RADIUS/max_steps scaling convention as base_training.py/
    # collect_demonstrations.py -- one env reused across all tests, sized for
    # the widest distance so every sampled distance fits comfortably inside it.
    oob_radius = max(20.0, args.distance_high * 3.0)
    max_steps = steps_for_dist(args.distance_high)

    reward_cfg = RewardConfig(oob_radius=oob_radius)
    reward_fn = make_reward_fn(reward_cfg)

    env = BaseDroneEnv(reward_fn, render_mode="None", max_steps=max_steps)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded {checkpoint}")
    print(f"Running {args.n_tests} random tests, distance ~ Uniform({args.distance_low}, {args.distance_high}), "
          f"categories={TEST_CATEGORIES}")

    fps = env.metadata.get("render_fps", 30)
    frame_interval = max(1, round((1 / fps) / env.dt))
    max_frames = fps * MAX_VIDEO_SECONDS
    video_writer = None
    frames_written = 0

    start = np.array([0, 0, 5], dtype=np.float32)
    results = []

    for test_idx in range(args.n_tests):
        category = str(rng.choice(TEST_CATEGORIES))
        dist = float(rng.uniform(args.distance_low, args.distance_high))

        target = None
        while target is None:  # sample_omni_target returns None on an underground direction -- resample
            target = sample_omni_target(rng, dist, category, base=start)

        print(f"\n=== test {test_idx + 1}/{args.n_tests}: category={category}  dist={dist:.2f}m  "
              f"target=[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] ===")

        if args.save_video and test_idx == 0:
            width, height, first_frame = _capture_frame()
            video_writer = cv2.VideoWriter(
                args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
            )
            video_writer.write(first_frame)
            frames_written = 1

        steps, reason, final_dist, frames_written = _run_episode(
            env, model, start, target,
            video_writer=video_writer, fps=fps, frame_interval=frame_interval,
            max_frames=max_frames, frames_written=frames_written,
        )

        if video_writer is not None:
            video_writer.release()
            print(f"Saved {frames_written / fps:.1f}s video to {args.save_video}")
            video_writer = None

        print(f"test {test_idx + 1}/{args.n_tests} done: steps={steps}  reason={reason}  final_dist={final_dist:.3f}")
        results.append({"category": category, "dist": dist, "steps": steps, "reason": reason,
                         "final_dist": final_dist})

    n_hit = sum(1 for r in results if r["reason"] == "Hit")
    print(f"\n{n_hit}/{args.n_tests} tests hit the target "
          f"(distance ~ Uniform({args.distance_low}, {args.distance_high}))")


if __name__ == "__main__":
    main()
