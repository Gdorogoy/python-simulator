"""
Phase 1
training-goals.md for more info
"""

import torch

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.plotting import plot_training_run
from app.guidance.train import ActorCritic, ppo_train, device
from app.reward_functions.reward_fn_phase1 import RewardFnPhase1
from app.training.phase_0_training import (
    TrainConfig, log_metrics, save_checkpoint, best_params as phase0_best_params,
)
import numpy as np


best_params = {
    **phase0_best_params,
    "phase1_pos_coef": phase0_best_params["phase0_pos_coef"],
    "hit_reward": 10,
}


def train(cfg: TrainConfig):
    reward_cfg = RewardFnPhase1(
        hit_steps_streak=1500,
        phase1_pos_coef=best_params["phase1_pos_coef"],
        hit_reward=best_params["hit_reward"],

        oob_radius=7,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=best_params["streak_penalty_coef"],
        hover_success_steps=200,
        streak_cap=best_params["streak_cap"],
        phase0_pos_coef=best_params["phase0_pos_coef"],
        tilt_penalty_coef=best_params["tilt_penalty_coef"],
        ang_vel_penalty_coef=best_params["ang_vel_penalty_coef"],
        imitation_coef=best_params["imitation_coef"],
        imitation_duration_steps=50_000,
        phase0_duration_steps=100_000,
    )
    reward_fn = reward_cfg.as_roadmap()

    env = InterceptorDroneEnv(reward_fn)

    env.target_pos = np.array([3, 0, 5], dtype=np.float32)
    print("target placed at:", env.target_pos)
    print("dorne placet at:", env.drone_state.position)


    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                             env.action_space.low, env.action_space.high).to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.LR,
    )

    timesteps_done = 0
    all_episode_rewards = []
    drone_state_arr = []
    dist_history = []


    model.load_state_dict(torch.load("app/z_final_version_1m_10epoch/ppo_stage2_660000.pt", map_location=device))


    while timesteps_done < cfg.TOTAL_TIMESTEPS:
        progress = timesteps_done / cfg.TOTAL_TIMESTEPS
        current_ent_coef = max(0.001, 0.01 * (1 - progress))

        #todo try use cos
        current_lr = cfg.LR * max(0.1, 0.01 * (1 - progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        current_target_kl = 0.02 - 0.005 * progress


        model, optimizer, episode_rewards, last_losses = ppo_train(
            env,
            total_timesteps=cfg.CHECKPOINT_EVERY_TIMESTEPS,   # one chunk per call
            num_steps=cfg.NUM_STEPS,
            gamma=cfg.GAMMA, lam=cfg.LAM, lr=current_lr,
            model=model,                  #  resumes, not restarts
            ent_coef=current_ent_coef,
            optimizer=optimizer,
            global_timesteps_offset=timesteps_done,
            global_total_timesteps=cfg.TOTAL_TIMESTEPS,
            target_kl=current_target_kl,
        )


        all_episode_rewards.extend(episode_rewards)
        drone_state_arr.append(env.drone_state)
        timesteps_done += cfg.CHECKPOINT_EVERY_TIMESTEPS



        print()  # move off the in-place epoch/step progress line
        save_checkpoint(model, timesteps_done, cfg)
        log_metrics(
            env, model, all_episode_rewards, timesteps_done, current_ent_coef,
            last_losses.get("policy_loss", 0.0), last_losses.get("value_loss", 0.0),
            last_losses.get("entropy_loss", 0.0), last_losses.get("approx_kl", 0.0),
            last_losses.get("early_stopped", False), last_losses.get("grad_norm", 0.0),
            csv_path=cfg.METRICS_CSV, n_diag_episodes=cfg.N_DIAGNOSTIC_EPISODES,
        )


        plot_training_run(
                cfg.METRICS_CSV,
                hover_success_steps=reward_cfg.hover_success_steps,
                n_diag_episodes=cfg.N_DIAGNOSTIC_EPISODES,
                hit_threshold=reward_cfg.hit_threshold,
                streak_cap=reward_cfg.streak_cap,
                max_steps=env.max_steps,
                oob_radius=reward_cfg.oob_radius,
                hit_reward=reward_cfg.hit_reward,
                attitude_penalty=reward_cfg.attitude_penalty,
                oob_penalty=reward_cfg.oob_penalty,
                streak_penalty_coef=reward_cfg.streak_penalty_coef,
            )

    return model


if __name__ == "__main__":
    cfg = TrainConfig(
        CHECKPOINT_DIR="runs/phase1",
        METRICS_CSV="runs/phase1/metrics.csv",
    )

    train(cfg)
