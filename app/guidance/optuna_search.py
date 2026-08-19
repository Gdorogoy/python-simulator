"""
Optuna search over PPO hyperparameters and reward-shaping coefficients.

Each trial builds a fresh env/model with Optuna-suggested settings, trains it
for a short budget (SEARCH_TIMESTEPS, much less than a full run), then scores
it by hover-success rate over a diagnostic eval. Trials run sequentially --
GPU trials contending for the same device in parallel don't get real
throughput gains with a network this small (see train.py's `device`), so
running one at a time is both simpler and not slower in practice.

Usage:
    python -m app.guidance.optuna_search [n_trials]
"""
import os
import sys
from datetime import datetime

import numpy as np
import optuna
import torch

from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic, ppo_train, device
from app.training import diagnose_with_model
from app.reward_functions.rewards import RewardConfig, make_reward_fn

N_TRIALS = 7
SEARCH_TIMESTEPS = 60_000   # short budget per trial -- this is a search, not a full run
NUM_STEPS = 4096
N_EVAL_EPISODES = 20
PRETRAINED_BC_PATH = "app/control/pretrained_bc.pt"
RESULTS_DIR = "runs/optuna"


def build_env(trial: optuna.Trial) -> InterceptorDroneEnv:
    reward_cfg = RewardConfig(
        oob_radius=7,
        hit_reward=10,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=trial.suggest_float("streak_penalty_coef", -0.1, -0.01),
        hover_success_steps=200,
        streak_cap=trial.suggest_int("streak_cap", 15, 40),
        phase0_pos_coef=trial.suggest_float("phase0_pos_coef", 0.2, 1.0),
        tilt_penalty_coef=trial.suggest_float("tilt_penalty_coef", 0.02, 0.3),
        ang_vel_penalty_coef=trial.suggest_float("ang_vel_penalty_coef", 0.02, 0.3),
        imitation_coef=trial.suggest_float("imitation_coef", 0.0, 0.5),
        imitation_duration_steps=20_000,
        phase0_duration_steps=20_000,
    )

    env = InterceptorDroneEnv(make_reward_fn(reward_cfg))
    env.target_pos = np.array([0, 0, 5], dtype=np.float32)
    return env


def objective(trial: optuna.Trial) -> float:
    env = build_env(trial)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.995)
    lam = trial.suggest_float("lam", 0.9, 0.98)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.02, log=True)
    hidden = 64
    target_kl=trial.suggest_float("target_kl", 0.02, 0.05, log=True)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high, hidden=hidden).to(device)
    if os.path.exists(PRETRAINED_BC_PATH):
        model.load_state_dict(torch.load(PRETRAINED_BC_PATH, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optimizer, episode_rewards, _ = ppo_train(
        env, total_timesteps=SEARCH_TIMESTEPS, num_steps=NUM_STEPS,
        gamma=gamma, lam=lam, lr=lr, model=model, optimizer=optimizer,
        ent_coef=ent_coef,target_kl=target_kl
    )

    outcomes = diagnose_with_model(model, env, N_EVAL_EPISODES)
    success_rate = outcomes["hover_success"] / N_EVAL_EPISODES
    mean_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0

    trial.set_user_attr("mean_reward", mean_reward)
    trial.set_user_attr("outcomes", outcomes)

    # success rate is the real objective; mean_reward only breaks ties between
    # trials that land on the same success rate.
    return success_rate + 0.001 * mean_reward


def run_search(n_trials: int = N_TRIALS) -> optuna.Study:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Optuna search done ===")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Best trial outcomes: {study.best_trial.user_attrs.get('outcomes')}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    study.trials_dataframe().to_csv(out_path, index=False)
    print(f"Saved trial results to {out_path}")

    return study


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_TRIALS
    run_search(n)




# est is trial 218 with value: 304.3300000000019.
# {'kp_pos': 2.3348549042980946, 'kd_pos': 1.0978513647624821, 'kp_att': 7.882461182173537, 'kd_att': 1.0936341247316574, 'kp_yaw': 0.048930591736932, 'kd_yaw': 0.20306927726482113}


best_params={

"Best value": 1.0253,
"Best params":
    {'streak_penalty_coef': -0.04251396820069286, 'streak_cap': 16, 'phase0_pos_coef': 0.6245260969779807,
     'tilt_penalty_coef': 0.059215545465061845, 'ang_vel_penalty_coef': 0.034466026555733276,
     'imitation_coef': 0.09441791089816459, 'lr': 2.5730200011306355e-05, 'gamma': 0.9887623315052021,
     'lam': 0.9451178282992497, 'ent_coef': 0.013260412980739588, 'target_kl': 0.041078703954645385},
"Best trial outcomes":{'oob': 0, 'attitude-ROLL': 0, 'attitude-PITCH': 0, 'hit': 0, 'hover_success': 15, 'moving_away_cap': 0, 'drift': 0, 'timeout': 0}

}