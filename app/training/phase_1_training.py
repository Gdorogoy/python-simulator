"""
Phase 1.0-1.3
training-goals.md for more info
"""
from datetime import datetime

import numpy as np
import torch
import os

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic, ppo_train, evaluate, device
from app.guidance.plotting import plot_training_run

import logging

from app.reward_functions.phase1_rewards import Phase1Config, make_phase1_reward_fn
from app.training.phase_0_training import diagnose_with_model, log_metrics, save_checkpoint

os.makedirs("prod_logs", exist_ok=True)
log_filename = f"prod_logs/phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

log.info(f"Logging this run to: {log_filename}")


#----------------------------------------------------------------------------
# Best params found from optuna (app/guidance/optuna_search.py) -- last search
# ran with a 3m target and never reached it (0/20 target_hit across trials,
# every episode ended in moving_away_cap), so these are unproven. Re-run the
# search against a closer target (e.g. 1m, matching training-goals.md's first
# curriculum band) before trusting this dict for a full run.
#----------------------------------------------------------------------------
best_params = {'streak_penalty_coef': -0.08644380865355883, 'streak_cap': 50,
               'axis_pos_coef': 0.9242194560002361, 'axis_penalty_coef': 0.4099346761000414,
               'hit_streak_bonus': 2.4453202890078996, 'hit_threshold': 0.022478862330064513,
               'lr': 0.00022591308247204179, 'gamma': 0.9845645092370297, 'lam': 0.9355856700498213,
               'ent_coef': 0.01107379064705624, 'target_kl': 0.04506868267653374}

PHASE0_CHECKPOINT = "app/z_final_version_1m_10epoch/ppo_stage2_660000.pt"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TrainConfig:
    SPAWN_POS = np.array([0, 0, 5], dtype=np.float32)   # matches phase 0's hover altitude
    TARGET_DISTANCE = 3   # meters off spawn along the active axis

    AXIS_DURATION_STEPS = 550_000   # steps each of left/right/forward stays active before advancing
    TOTAL_TIMESTEPS = 2_500_000     # 3 * AXIS_DURATION_STEPS + buffer left on "back" (last, indefinite)

    NUM_STEPS = 4096
    GAMMA = best_params["gamma"]
    LAM = best_params["lam"]
    LR = best_params["lr"]

    CHECKPOINT_EVERY_TIMESTEPS = 15_000
    CHECKPOINT_DIR = "runs/p1_1m_10epochs"
    METRICS_CSV = "runs/p1_1m_10epochs/metrics.csv"
    N_DIAGNOSTIC_EPISODES = 20


# ---------------------------------------------------------------------------
# Direction curriculum -- order must match make_phase1_reward_fn's chaining
# (left -> right -> forward -> back) so target_pos tracks whichever axis_fn
# is currently active in the chained reward.
# ---------------------------------------------------------------------------
def axis_targets(cfg: TrainConfig):
    d = cfg.TARGET_DISTANCE
    x, y, z = cfg.SPAWN_POS
    return {
        "left":    np.array([x, y - d, z], dtype=np.float32),
        "right":   np.array([x, y + d, z], dtype=np.float32),
        "forward": np.array([x + d, y, z], dtype=np.float32),
        "back":    np.array([x - d, y, z], dtype=np.float32),
    }


def active_axis_name(timesteps_done, axis_duration_steps):
    order = ["left", "right", "forward", "back"]
    idx = min(timesteps_done // axis_duration_steps, len(order) - 1)
    return order[int(idx)]


# ---------------------------------------------------------------------------
# Main training loop -- chunked, resuming the SAME model each chunk, same
# pattern as phase_0_training.train()
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig):
    reward_cfg = Phase1Config(
        oob_radius=7,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=best_params["streak_penalty_coef"],
        streak_cap=best_params["streak_cap"],
        axis_pos_coef=best_params["axis_pos_coef"],
        axis_penalty_coef=best_params["axis_penalty_coef"],
        hit_streak_target=200,
        hit_streak_bonus=best_params["hit_streak_bonus"],
        hit_threshold=best_params["hit_threshold"],
    )

    reward_fn = make_phase1_reward_fn(reward_cfg, axis_duration_steps=cfg.AXIS_DURATION_STEPS)
    targets = axis_targets(cfg)

    env = InterceptorDroneEnv(reward_fn)
    env.target_pos = targets["left"]
    print("target placed at:", env.target_pos)
    print("drone placed at:", env.drone_state.position)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.LR)

    # warm-start from the phase 0 hover checkpoint (training-goals.md)
    if os.path.exists(PHASE0_CHECKPOINT):
        model.load_state_dict(torch.load(PHASE0_CHECKPOINT, map_location=device))
        print(f"warm-started from {PHASE0_CHECKPOINT}")
    else:
        print(f"[warn] {PHASE0_CHECKPOINT} not found, starting from scratch")

    timesteps_done = 0
    all_episode_rewards = []
    current_axis = "left"

    while timesteps_done < cfg.TOTAL_TIMESTEPS:
        axis_name = active_axis_name(timesteps_done, cfg.AXIS_DURATION_STEPS)
        if axis_name != current_axis:
            current_axis = axis_name
            print(f"[curriculum] advancing to axis: {current_axis}")
        env.target_pos = targets[current_axis]

        progress = timesteps_done / cfg.TOTAL_TIMESTEPS
        current_ent_coef = max(0.001, best_params["ent_coef"] * (1 - progress))
        current_lr = cfg.LR * max(0.1, 0.01 * (1 - progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        current_target_kl = best_params["target_kl"] * (1 - 0.5 * progress)

        model, optimizer, episode_rewards, last_losses = ppo_train(
            env,
            total_timesteps=cfg.CHECKPOINT_EVERY_TIMESTEPS,
            num_steps=cfg.NUM_STEPS,
            gamma=cfg.GAMMA, lam=cfg.LAM, lr=current_lr,
            model=model,
            ent_coef=current_ent_coef,
            optimizer=optimizer,
            global_timesteps_offset=timesteps_done,
            global_total_timesteps=cfg.TOTAL_TIMESTEPS,
            target_kl=current_target_kl,
        )
        all_episode_rewards.extend(episode_rewards)
        timesteps_done += cfg.CHECKPOINT_EVERY_TIMESTEPS

        print()
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
    cfg = TrainConfig()
    model = train(cfg)
    evaluate(model, InterceptorDroneEnv(), n_episodes=10)
