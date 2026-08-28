"""Phase 0 training loop -- see training-goals.md for background."""
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

from app.guidance.utils import calc_drone_state, compute_grade
from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe
import mlflow
from app.reward_functions.rewards import make_reward_fn, RewardConfig

_log_dir = os.environ.get("OPTUNA_WORKER_LOG_DIR", "prod_logs")
os.makedirs(_log_dir, exist_ok=True)
log_filename = f"{_log_dir}/phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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




best_params={'streak_penalty_coef': -0.04983211032477406,
             'streak_cap': 21, 'phase0_pos_coef': 0.9233025352276092, 'tilt_penalty_coef': 0.13620117520669245,
             'ang_vel_penalty_coef': 0.09137289750622843, 'imitation_coef': 0.2545331593681599,
             'lr': 0.00021775015806847712, 'gamma': 0.9663964477093601, 'lam': 0.9076864484897095,
             'ent_coef': 0.013508073722014471, 'target_kl': 0.03251152911493923}


# ---------------------------------------------------------------------------
# Config -- every training hyperparameter lives here so a run is fully
# reproducible from one TrainConfig instance, and so optuna_search.py can
# sweep any of them without touching this file.
# ---------------------------------------------------------------------------
class TrainConfig:
    def __init__(self,
                 TARGET_DISTANCE=1,
                 TARGET_ANGLE_DEG=45.0,
                 TARGET_Y_OFFSET=1.0,
                 TOTAL_TIMESTEPS=5_000_000,
                 NUM_STEPS=4096,                  # rollout length per PPO update
                 NUM_EPOCHS=10,                   # PPO epochs per rollout batch
                 BATCH_SIZE=32,
                 GAMMA=0.98,
                 LAM=0.95,
                 LR=best_params["lr"],
                 CLIP_EPS=0.2,
                 VF_COEF=0.5,
                 MAX_GRAD_NORM=0.5,
                 HIDDEN=64,
                 NUM_HIDDEN_LAYERS=3,
                 DROPOUT=0.0,
                 LOG_STD_MIN=-3.0,
                 LOG_STD_MAX=0.0,
                 LOG_STD_CLAMP_MIN=-3.0,
                 LOG_STD_CLAMP_MAX=-0.5,
                 ENT_COEF_START=0.01,
                 ENT_COEF_END=0.001,
                 TARGET_KL_START=0.02,
                 TARGET_KL_END=0.015,
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
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.LAM = LAM
        self.LR = LR
        self.CLIP_EPS = CLIP_EPS
        self.VF_COEF = VF_COEF
        self.MAX_GRAD_NORM = MAX_GRAD_NORM
        self.HIDDEN = HIDDEN
        self.NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
        self.DROPOUT = DROPOUT
        self.LOG_STD_MIN = LOG_STD_MIN
        self.LOG_STD_MAX = LOG_STD_MAX
        self.LOG_STD_CLAMP_MIN = LOG_STD_CLAMP_MIN
        self.LOG_STD_CLAMP_MAX = LOG_STD_CLAMP_MAX
        self.ENT_COEF_START = ENT_COEF_START
        self.ENT_COEF_END = ENT_COEF_END
        self.TARGET_KL_START = TARGET_KL_START
        self.TARGET_KL_END = TARGET_KL_END
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
    hit_times_sec = []
    final_dists = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        last_reason = None
        info = {}

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
        final_dists.append(env.prev_distance)
        if info.get("hit_time_sec") is not None:
            hit_times_sec.append(info["hit_time_sec"])

        if env.hover_success_achieved:
            outcomes["hover_success"] += 1
        elif truncated and not terminated:
            outcomes["timeout"] += 1
        else:
            outcomes[last_reason] = outcomes.get(last_reason, 0) + 1

    avg_hit_time_sec = float(np.mean(hit_times_sec)) if hit_times_sec else None
    print(f"[diagnostic] outcomes over {n_episodes} eps:", outcomes)
    print(f"[diagnostic] avg steps survived: {np.mean(steps_survived):.1f}")
    print(f"[diagnostic] avg final dist: {np.mean(final_dists):.3f}  avg time-to-hit: {avg_hit_time_sec}")
    outcomes["avg_hit_time_sec"] = avg_hit_time_sec
    outcomes["avg_final_dist"] = float(np.mean(final_dists))
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
                 grad_norm, csv_path, n_diag_episodes=20, oob_radius=7.0):
    # Keys must match the exact reason strings the reward fns return (_check_hit
    # returns "Hit"); outcome_columns below maps this back to lowercase "outcome_hit".
    outcomes = {"oob": 0, "Hit": 0, "hover_success": 0,
                "moving_away_cap": 0, "drift": 0, "timeout": 0}
    steps_survived = []
    final_dists = []
    diag_episode_rewards = []
    hit_times_sec = []
    max_hover_streak = 0

    for _ in range(n_diag_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        last_reason = None
        episode_max_hover_streak = 0
        episode_reward = 0.0
        info = {}

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
        if info.get("hit_time_sec") is not None:
            hit_times_sec.append(info["hit_time_sec"])

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
    avg_hit_time_sec = float(np.mean(hit_times_sec)) if hit_times_sec else None

    effective_std = np.exp(model.log_std_min + 0.5 * (model.log_std_max - model.log_std_min) *
                            (np.tanh(model.actor_log_std.detach().cpu().numpy()) + 1))
    effective_std_mean = float(np.mean(effective_std))
    total_param_norm = sum(p.data.norm().item() for p in model.parameters())

    # From the diagnostic episodes themselves (deterministic policy), not the
    # stochastic training-rollout episodes, so all stats describe the same sample.
    reward_mean = float(np.mean(diag_episode_rewards)) if diag_episode_rewards else 0.0
    reward_std = float(np.std(diag_episode_rewards)) if diag_episode_rewards else 0.0

    success_rate = outcomes["hover_success"] / n_diag_episodes if n_diag_episodes else 0.0
    grade, _ = compute_grade(
        success_rate=success_rate, avg_final_dist=avg_final_dist,
        avg_hit_time_sec=avg_hit_time_sec, avg_grad_norm=grad_norm,
        oob_radius=oob_radius, episode_time_budget_sec=env.max_steps * env.dt,
    )

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
        "avg_hit_time_sec": avg_hit_time_sec,
        "grade": grade,
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
    env.target_pos=np.array([0,0,5],dtype=np.float32)
    print("target placed at:", env.target_pos)
    print("drone placed at:", env.drone_state.position)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high,
                         hidden=cfg.HIDDEN, num_hidden_layers=cfg.NUM_HIDDEN_LAYERS, dropout=cfg.DROPOUT,
                         log_std_min=cfg.LOG_STD_MIN, log_std_max=cfg.LOG_STD_MAX).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.LR)
    timesteps_done = 0
    all_episode_rewards = []
    drone_state_arr = []
    last_checkpoint_path = None

    model.load_state_dict(torch.load("app/z_final_version_1m_10epoch/ppo_stage2_660000.pt", map_location=device))

    start_run(experiment_name="phase0-training", run_name=cfg.CHECKPOINT_DIR)
    log_params_safe({**vars(cfg), **vars(reward_cfg)})

    while timesteps_done < cfg.TOTAL_TIMESTEPS:
        # Decay entropy coefficient and target_kl over the run; exploration matters
        # early, but late in training entropy should stop dragging the policy toward
        # max noise once the advantage signal is weak.
        progress = timesteps_done / cfg.TOTAL_TIMESTEPS
        current_ent_coef = max(cfg.ENT_COEF_END, cfg.ENT_COEF_START * (1 - progress))
        current_lr = cfg.LR * max(0.1, 0.01 * (1 - progress))
        current_target_kl = cfg.TARGET_KL_START - (cfg.TARGET_KL_START - cfg.TARGET_KL_END) * progress

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model, optimizer, episode_rewards, last_losses = ppo_train(
            env,
            total_timesteps=cfg.CHECKPOINT_EVERY_TIMESTEPS,   # one chunk per call
            num_steps=cfg.NUM_STEPS,
            gamma=cfg.GAMMA, lam=cfg.LAM, lr=current_lr,
            model=model,                  # resumes, not restarts
            ent_coef=current_ent_coef,
            optimizer=optimizer,
            global_timesteps_offset=timesteps_done,
            global_total_timesteps=cfg.TOTAL_TIMESTEPS,
            target_kl=current_target_kl,
            num_epochs=cfg.NUM_EPOCHS, batch_size=cfg.BATCH_SIZE,
            clip_eps=cfg.CLIP_EPS, vf_coef=cfg.VF_COEF, max_grad_norm=cfg.MAX_GRAD_NORM,
            log_std_clamp_min=cfg.LOG_STD_CLAMP_MIN, log_std_clamp_max=cfg.LOG_STD_CLAMP_MAX,
        )
        all_episode_rewards.extend(episode_rewards)
        drone_state_arr.append(env.drone_state)
        timesteps_done += cfg.CHECKPOINT_EVERY_TIMESTEPS


        print()  # move off the in-place epoch/step progress line
        save_checkpoint(model, timesteps_done, cfg)
        last_checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"ppo_stage2_{timesteps_done}.pt")
        row = log_metrics(
            env, model, timesteps_done, current_ent_coef,
            last_losses.get("policy_loss", 0.0), last_losses.get("value_loss", 0.0),
            last_losses.get("entropy_loss", 0.0), last_losses.get("approx_kl", 0.0),
            last_losses.get("early_stopped", False), last_losses.get("grad_norm", 0.0),
            csv_path=cfg.METRICS_CSV, n_diag_episodes=cfg.N_DIAGNOSTIC_EPISODES,
            oob_radius=reward_cfg.oob_radius,
        )
        log_metrics_safe(row, step=timesteps_done)

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

    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
        mlflow.log_artifact(last_checkpoint_path)
    if os.path.exists(cfg.METRICS_CSV):
        mlflow.log_artifact(cfg.METRICS_CSV)
    mlflow.end_run()

    return model


if __name__ == "__main__":
    cfg = TrainConfig()
    model = train(cfg)
    evaluate(model, InterceptorDroneEnv(), n_episodes=10)