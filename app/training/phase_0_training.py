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
from app.guidance.train import ActorCritic, ppo_train, evaluate, device
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




#----------------------------------------------------------------------------
# Best params found from optuna
#----------------------------------------------------------------------------
best_params={'streak_penalty_coef': -0.04983211032477406,
             'streak_cap': 21, 'phase0_pos_coef': 0.9233025352276092, 'tilt_penalty_coef': 0.13620117520669245,
             'ang_vel_penalty_coef': 0.09137289750622843, 'imitation_coef': 0.2545331593681599,
             'lr': 0.00021775015806847712, 'gamma': 0.9663964477093601, 'lam': 0.9076864484897095,
             'ent_coef': 0.013508073722014471, 'target_kl': 0.03251152911493923}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TrainConfig:
    def __init__(self,
                 TARGET_DISTANCE=1,
                 TARGET_ANGLE_DEG=45.0,
                 TARGET_Y_OFFSET=1.0,
                 TOTAL_TIMESTEPS=5_000_000,  # was 2_000_000
                 NUM_STEPS=4096,                  # rollout length per PPO update
                 NUM_EPOCHS=10,                   # PPO epochs per rollout batch
                 GAMMA=0.98,  # was 0.97
                 LAM=0.95,
                 LR=best_params["lr"],  # was originaly 3e-5 , but drone is not doing anything so now th gradient will be 100 times bigger
                 CHECKPOINT_EVERY_TIMESTEPS=15_000,   # coarse enough that diagnostics
                 CHECKPOINT_DIR="runs/1m_10epochs_v2",
                 METRICS_CSV="runs/1m_10epochs_v2/metrics.csv",
                 N_DIAGNOSTIC_EPISODES=20):
        self.TARGET_DISTANCE = TARGET_DISTANCE
        self.TARGET_ANGLE_DEG = TARGET_ANGLE_DEG
        self.TARGET_Y_OFFSET = TARGET_Y_OFFSET
        self.TOTAL_TIMESTEPS = TOTAL_TIMESTEPS
        self.NUM_STEPS = NUM_STEPS
        self.NUM_EPOCHS = NUM_EPOCHS
        self.GAMMA = GAMMA
        self.LAM = LAM
        self.LR = LR
        self.CHECKPOINT_EVERY_TIMESTEPS = CHECKPOINT_EVERY_TIMESTEPS
        self.CHECKPOINT_DIR = CHECKPOINT_DIR
        self.METRICS_CSV = METRICS_CSV
        self.N_DIAGNOSTIC_EPISODES = N_DIAGNOSTIC_EPISODES


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
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
                action = model.scale_action(mean)
            action = action.squeeze(0).cpu().numpy()

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
    print(f"step {step_count}: raw_mean={mean.squeeze(0).cpu().numpy()}")
    return outcomes


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model, timesteps_done, cfg: TrainConfig):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(cfg.CHECKPOINT_DIR, f"ppo_stage2_{timesteps_done}.pt")
    torch.save(model.state_dict(), path)
    print(f"[checkpoint] saved {path}")


# ---------------------------------------------------------------------------
# Structured per-checkpoint metrics -> CSV (one flat row per checkpoint)
# ---------------------------------------------------------------------------
def log_metrics(env, model, timesteps_done, ent_coef,
                 policy_loss, value_loss, entropy_loss, approx_kl, early_stopped,
                 grad_norm, csv_path, n_diag_episodes=20):
    # keys must match the EXACT reason strings the reward fns return (_check_hit
    # returns "Hit", capital H) -- outcomes[last_reason] below creates/increments
    # by that literal string, so a mismatched case here silently eats the count
    # into an unread key (outcome_columns below maps this back to the lowercase
    # outcome_hit CSV column).
    outcomes = {"oob": 0, "Hit": 0, "hover_success": 0,
                "moving_away_cap": 0, "drift": 0, "timeout": 0}
    steps_survived = []
    final_dists = []
    diag_episode_rewards = []
    max_hover_streak = 0

    for _ in range(n_diag_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        last_reason = None
        episode_max_hover_streak = 0
        episode_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
                action = model.scale_action(mean)
            action = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            episode_reward += reward
            done = terminated or truncated
            last_reason = info["reason"]
            episode_max_hover_streak = max(episode_max_hover_streak, getattr(env, "hover_steps_in_zone", 0))

        steps_survived.append(step_count)
        final_dists.append(env.prev_distance)
        diag_episode_rewards.append(episode_reward)
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
    std_final_dist = float(np.std(final_dists)) if final_dists else 0.0

    LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.0
    effective_std = np.exp(LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) *
                            (np.tanh(model.actor_log_std.detach().cpu().numpy()) + 1))
    effective_std_mean = float(np.mean(effective_std))
    total_param_norm = sum(p.data.norm().item() for p in model.parameters())

    # from the diagnostic episodes themselves (deterministic policy, same 20
    # episodes as avg_final_dist/outcome_*/avg_steps_survived below), NOT the
    # last 10 TRAINING rollout episodes (stochastic, a different sample
    # entirely) -- that mismatch was silently making vs_pid_baseline's
    # avg_reward describe different episodes than its other 3 stats.
    reward_mean = float(np.mean(diag_episode_rewards)) if diag_episode_rewards else 0.0
    reward_std = float(np.std(diag_episode_rewards)) if diag_episode_rewards else 0.0

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
        "std_final_dist": std_final_dist,
        "effective_std_mean": effective_std_mean,
        "total_param_norm": total_param_norm,
    }
    # (lookup key in `outcomes`, CSV column suffix) -- only "Hit" needs the
    # case fixed up (column stays outcome_hit); everything else already
    # matches its own reason string exactly, including attitude-ROLL/PITCH's
    # capitalization, so don't blanket-lowercase these.
    outcome_columns = [
        ("oob", "oob"), ("attitude-ROLL", "attitude-ROLL"), ("attitude-PITCH", "attitude-PITCH"),
        ("Hit", "hit"), ("hover_success", "hover_success"), ("moving_away_cap", "moving_away_cap"),
        ("drift", "drift"), ("timeout", "timeout"),
    ]
    for lookup_key, col_suffix in outcome_columns:
        row[f"outcome_{col_suffix}"] = outcomes.get(lookup_key, 0)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row

    return row






# ---------------------------------------------------------------------------
# Main training loop  chunked, resuming the SAME model each chunk
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig):
    reward_cfg = RewardConfig(
        oob_radius=7,
        hit_reward=10,
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
                         env.action_space.low, env.action_space.high).to(device)
    optimizer =torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.LR,

    )
    timesteps_done = 0
    all_episode_rewards = []
    drone_state_arr = []
    dist_history=[]

    model.load_state_dict(torch.load("app/z_final_version_1m_10epoch/ppo_stage2_660000.pt", map_location=device))


    while timesteps_done < cfg.TOTAL_TIMESTEPS:

        # Decay entropy coefficient over the course of the full run: early on,
        # exploration is fine (policy needs to try things); late in training
        # it should be near-zero so entropy stops dragging the policy back
        # toward max noise once the advantage signal is weak/near-zero.
        progress = timesteps_done / cfg.TOTAL_TIMESTEPS
        current_ent_coef = max(0.001, 0.01 * (1 - progress))


        current_lr = cfg.LR* max(0.1, 0.01 * (1 - progress))

        #adam dosent take new rl
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
            num_epochs=cfg.NUM_EPOCHS,
        )
        all_episode_rewards.extend(episode_rewards)
        drone_state_arr.append(env.drone_state)
        timesteps_done += cfg.CHECKPOINT_EVERY_TIMESTEPS


        print()  # move off the in-place epoch/step progress line
        save_checkpoint(model, timesteps_done, cfg)
        log_metrics(
            env, model, timesteps_done, current_ent_coef,
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