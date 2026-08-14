"""
Phase 0
training-goals.md for more info
"""
from datetime import datetime
import csv

import numpy as np
import torch
import os

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from scipy.spatial.transform import Rotation

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic, ppo_train, evaluate
from app.guidance.plotting import plot_training_run

import logging

from app.guidance.utils import calc_drone_state
from app.reward_functions.rewards import make_reward_fn, RewardConfig

os.makedirs("prod_logs", exist_ok=True)
log_filename = f"prod_logs/phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename), # TO WRITE
        logging.StreamHandler(),  # TO LOG INTO CONSOLE
    ],
)
log = logging.getLogger(__name__)

log.info(f"Logging this run to: {log_filename}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TrainConfig:
    TARGET_DISTANCE = 1
    TARGET_ANGLE_DEG = 45.0
    TARGET_Y_OFFSET = 1.0

    TOTAL_TIMESTEPS = 1_000_000 # was 2_000_000
    NUM_STEPS = 4096                  # rollout length per PPO update
    GAMMA = 0.98 # was 0.97
    LAM = 0.95
    LR = 3e-4 # was originaly 3e-5 , but drone is not doing anything so now th gradient will be 100 times bigger

    CHECKPOINT_EVERY_TIMESTEPS = 15_000   # coarse enough that diagnostics
    CHECKPOINT_DIR = "runs/1m_10epochs_v2"
    METRICS_CSV = "runs/1m_10epochs_v2/metrics.csv"
    N_DIAGNOSTIC_EPISODES = 20


# ---------------------------------------------------------------------------
# Target placement
# ---------------------------------------------------------------------------
def compute_target_pos(start_position, distance, angle_deg, y_offset):
    theta = np.radians(angle_deg)
    dx = distance * np.cos(theta)
    dz = distance * np.sin(theta)
    return np.array([
        start_position[0] + dx,
        start_position[1] + y_offset,
        start_position[2] + dz,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Outcome-distribution diagnostic (trained-model version)
# ---------------------------------------------------------------------------
def diagnose_with_model(model, env, n_episodes):
    outcomes = {"oob": 0, "attitude-ROLL": 0, "attitude-PITCH": 0, "hit": 0,
                "hover_success": 0, "moving_away_cap": 0, "drift": 0, "timeout": 0}
    steps_survived = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        last_reason = None

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
                action = model.scale_action(mean)
            action = action.squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
            last_reason = info["reason"]

        steps_survived.append(step_count)

        if env.hover_success_achieved:
            outcomes["hover_success"] += 1
        elif truncated and not terminated:
            outcomes["timeout"] += 1
        else:
            outcomes[last_reason] += 1

    print(f"[diagnostic] outcomes over {n_episodes} eps:", outcomes)
    print(f"[diagnostic] avg steps survived: {np.mean(steps_survived):.1f}")
    print(f"step {step_count}: raw_mean={mean.squeeze(0).numpy()}")
    return outcomes


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model, timesteps_done, cfg: TrainConfig):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(cfg.CHECKPOINT_DIR, f"ppo_stage1_{timesteps_done}.pt")
    torch.save(model.state_dict(), path)
    print(f"[checkpoint] saved {path}")


# ---------------------------------------------------------------------------
# Structured per-checkpoint metrics -> CSV (one flat row per checkpoint)
# ---------------------------------------------------------------------------
def log_metrics(env, model, episode_rewards, timesteps_done, ent_coef,
                 policy_loss, value_loss, entropy_loss, approx_kl, early_stopped,
                 grad_norm, csv_path, n_diag_episodes=20):
    outcomes = {"oob": 0, "attitude": 0, "hit": 0, "hover_success": 0,
                "moving_away_cap": 0, "drift": 0, "timeout": 0}
    steps_survived = []
    final_dists = []
    max_hover_streak = 0

    for _ in range(n_diag_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        last_reason = None
        episode_max_hover_streak = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
                action = model.scale_action(mean)
            action = action.squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
            last_reason = info["reason"]
            episode_max_hover_streak = max(episode_max_hover_streak, getattr(env, "hover_steps_in_zone", 0))

        steps_survived.append(step_count)
        final_dists.append(env.prev_distance)
        max_hover_streak = max(max_hover_streak, episode_max_hover_streak)

        if getattr(env, "hover_success_achieved", False):
            outcomes["hover_success"] += 1
        elif truncated and not terminated:
            outcomes["timeout"] += 1
        else:
            outcomes[last_reason] = outcomes.get(last_reason, 0) + 1

    avg_steps_survived = float(np.mean(steps_survived)) if steps_survived else 0.0
    avg_final_dist = float(np.mean(final_dists)) if final_dists else 0.0
    min_final_dist = float(np.min(final_dists)) if final_dists else 0.0

    LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.0
    effective_std = np.exp(LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) *
                            (np.tanh(model.actor_log_std.detach().numpy()) + 1))
    effective_std_mean = float(np.mean(effective_std))
    total_param_norm = sum(p.data.norm().item() for p in model.parameters())

    recent = episode_rewards[-10:] if episode_rewards else [0.0]
    reward_mean = float(np.mean(recent))
    reward_std = float(np.std(recent))

    row = {
        "timesteps": timesteps_done,
        "ent_coef": ent_coef,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
        "early_stopped": early_stopped,
        "grad_norm": grad_norm,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "avg_steps_survived": avg_steps_survived,
        "max_hover_streak": max_hover_streak,
        "avg_final_dist": avg_final_dist,
        "min_final_dist": min_final_dist,
        "effective_std_mean": effective_std_mean,
        "total_param_norm": total_param_norm,
    }
    for key in ("oob", "attitude-ROLL","attitude-PITCH", "hit", "hover_success", "moving_away_cap", "drift", "timeout"):
        row[f"outcome_{key}"] = outcomes.get(key, 0)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



#----------------------------------------------------------------------------
# Graph for the distance
#----------------------------------------------------------------------------
def show_graph(dist_history):
    plt.figure(figsize=(8, 4))
    plt.plot(dist_history)
    plt.xlabel("Step")
    plt.ylabel("Distance to target (m)")
    plt.title("Drone distance from target over one episode")
    plt.axhline(y=0.3, color='r', linestyle='--', label='hit threshold (0.3m)')
    plt.legend()
    plt.savefig("distance_trace.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main training loop  chunked, resuming the SAME model each chunk
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig):
    reward_cfg = RewardConfig(
        oob_radius=7,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=-0.05,
        hover_success_steps=200,
        streak_cap=30,
        phase0_pos_coef=0.5,           
        imitation_coef=0.25,           
        imitation_duration_steps=20_000,   
        phase0_duration_steps=20_000,      
    )

    reward_fn= make_reward_fn(reward_cfg)

    env = InterceptorDroneEnv(reward_fn)
    # env.target_pos = compute_target_pos(
    #     (0, 0,0), cfg.TARGET_DISTANCE, cfg.TARGET_ANGLE_DEG, cfg.TARGET_Y_OFFSET
    # )

    #to go upo
    env.target_pos=np.array([0,0,5],dtype=np.float32)
    print("target placed at:", env.target_pos)
    print("dorne placet at:", env.drone_state.position)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high)
    optimizer =torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.LR,

    )
    timesteps_done = 0
    all_episode_rewards = []
    drone_state_arr = []
    dist_history=[]

    model.load_state_dict(torch.load("app/control/pretrained_bc.pt"))


    while timesteps_done < cfg.TOTAL_TIMESTEPS:

        # Decay entropy coefficient over the course of the full run: early on,
        # exploration is fine (policy needs to try things); late in training
        # it should be near-zero so entropy stops dragging the policy back
        # toward max noise once the advantage signal is weak/near-zero.
        progress = timesteps_done / cfg.TOTAL_TIMESTEPS
        current_ent_coef = max(0.001, 0.01 * (1 - progress))

        model, optimizer, episode_rewards, last_losses = ppo_train(
            env,
            total_timesteps=cfg.CHECKPOINT_EVERY_TIMESTEPS,   # one chunk per call
            num_steps=cfg.NUM_STEPS,
            gamma=cfg.GAMMA, lam=cfg.LAM, lr=cfg.LR,
            model=model,                  #  resumes, not restarts
            ent_coef=current_ent_coef,
            optimizer=optimizer,
            global_timesteps_offset=timesteps_done,
            global_total_timesteps=cfg.TOTAL_TIMESTEPS,
        )
        all_episode_rewards.extend(episode_rewards)
        drone_state_arr.append(env.drone_state)
        timesteps_done += cfg.CHECKPOINT_EVERY_TIMESTEPS

        recent = all_episode_rewards[-10:] if all_episode_rewards else [0]

        print()  # move off the in-place epoch/step progress line
        save_checkpoint(model, timesteps_done, cfg)
        log_metrics(
            env, model, all_episode_rewards, timesteps_done, current_ent_coef,
            last_losses.get("policy_loss", 0.0), last_losses.get("value_loss", 0.0),
            last_losses.get("entropy_loss", 0.0), last_losses.get("approx_kl", 0.0),
            last_losses.get("early_stopped", False), last_losses.get("grad_norm", 0.0),
            csv_path=cfg.METRICS_CSV, n_diag_episodes=cfg.N_DIAGNOSTIC_EPISODES,
        )

    show_graph(dist_history)
    plot_training_run(
        cfg.METRICS_CSV,
        hover_success_steps=reward_cfg.hover_success_steps,
        n_diag_episodes=cfg.N_DIAGNOSTIC_EPISODES,
        hit_threshold=reward_cfg.hit_threshold,
        streak_cap=reward_cfg.streak_cap,
        max_steps=env.max_steps,
    )

    return model


if __name__ == "__main__":
    cfg = TrainConfig()
    model = train(cfg)
    evaluate(model, InterceptorDroneEnv(), n_episodes=10)