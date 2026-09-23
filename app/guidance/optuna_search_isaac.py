"""Optuna hyperparameter search over the Isaac Lab training pipeline (app.training.base_training_isaac),
same search space as app.guidance.optuna_search except hidden/num_hidden_layers/num_epochs/num_minibatches
are pinned (64/4/10/64) so every trial keeps the BC checkpoint's warm start valid. Builds one Isaac env
and reuses it across all trials; --num_envs defaults to 256, not 4096, so warmup doesn't eat the budget.

Usage:
    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u app\\guidance\\optuna_search_isaac.py --headless ^
        --n-trials 100 --num_envs 256 --total-timesteps 2000000
Resumes automatically under the same --study-name/--storage.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n-trials", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=2_000_000)
parser.add_argument("--distance-low", dest="distance_low", type=float, default=3.0)
parser.add_argument("--distance-high", dest="distance_high", type=float, default=10.0)
parser.add_argument("--bc-checkpoint", dest="bc_checkpoint_path", default="app/control/pretrained_bc_dagger.pt")
parser.add_argument("--gains-path", dest="gains_path", default="app/control/best_pid_gains_per_dist.json")
parser.add_argument("--study-name", default="base_training_isaac_optuna_v2")
parser.add_argument("--storage", default="sqlite:///runs/optuna_isaac/study.db")
parser.add_argument("--n-startup-trials", type=int, default=10)
parser.add_argument("--n-warmup-steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Isaac venv's mlflow is a different version than the numpy venv's and can't
# share mlflow.db's schema -- use the same separate store base_training_isaac.py
# points at. Must be set before `import mlflow` touches any tracking state.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow_isaac.db")

import gymnasium as gym
import mlflow
import numpy as np
import optuna
import torch

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.control.step_budget import steps_for_dist
from app.environmental.base_drone_env import BaseDroneEnv
from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe
from app.guidance.train import ActorCritic, isaac_ppo_train, cosine_lr, device, load_bc_checkpoint
from app.guidance.utils import compute_grade
from app.reward_functions.rewards import reward_func, GAMMA as REWARD_GAMMA
from app.training.base_training_isaac import _run_imitation_stage_isaac, MIN_WARMUP_CHUNKS, NUM_STEPS_PER_CHUNK
from app.training.diagnostics import diagnose_with_model
from app.training.eval_matrix import build_uniform_omni_eval_pairs

MLFLOW_EXPERIMENT = "base-training-isaac-optuna"
BEST_PARAMS_PATH = "runs/optuna_isaac/best_params.json"
# Fewer than base_training_isaac.py's N_DIAGNOSTIC_EPISODES -- speed over
# precision, this only needs to rank trials against each other.
N_DIAGNOSTIC_EPISODES = 5


def suggest_params(trial: optuna.Trial) -> dict:
    """Same search space as app.guidance.optuna_search.suggest_params, except
    hidden/num_hidden_layers are pinned (see module docstring for why)."""
    ent_coef_start = trial.suggest_float("ent_coef_start", 1e-4, 0.1, log=True)
    return dict(
        hidden=64,
        num_hidden_layers=4,
        dropout=0.0,
        lr=trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        gamma=REWARD_GAMMA,
        lam=trial.suggest_float("lam", 0.85, 0.999),
        clip_eps=trial.suggest_float("clip_eps", 0.05, 0.4),
        vf_coef=trial.suggest_float("vf_coef", 0.1, 1.5),
        target_kl=trial.suggest_float("target_kl", 0.005, 0.15, log=True),
        num_epochs=10,  # pinned to base_training.PARAMS's value, not searched
        num_minibatches=64,  # pinned to base_training.PARAMS's value, not searched
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.1, 2.0),
        log_std_max=trial.suggest_float("log_std_max", -2.5, -0.05),
        ent_coef_start=ent_coef_start,
        ent_coef_end=ent_coef_start * trial.suggest_float("ent_coef_end_frac", 0.01, 0.9),  # fraction of start, not independent
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        lr_min_ratio=trial.suggest_float("lr_min_ratio", 0.0, 0.5),
    )


def make_objective(env, gains_by_dist: dict, target_pairs, eval_env, oob_radius: float,
                    total_timesteps: int, bc_checkpoint_path: str):
    """Builds the per-trial objective as a closure over the shared, expensive-
    to-construct Isaac env and the numpy-oracle eval_env used for diagnostics
    (policy-quality checks, not physics throughput)."""
    unwrapped = env.unwrapped
    obs_dim = unwrapped.single_observation_space["policy"].shape[0]
    action_dim = unwrapped.single_action_space.shape[0]
    chunk_grand_steps = NUM_STEPS_PER_CHUNK * unwrapped.num_envs

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial)
        unwrapped.set_target_pairs(target_pairs)  # same pool every trial -- fair comparison

        model = ActorCritic(obs_dim, action_dim, unwrapped.single_action_space.low,
                             unwrapped.single_action_space.high,
                             hidden=p["hidden"], num_hidden_layers=p["num_hidden_layers"],
                             dropout=p["dropout"], log_std_max=p["log_std_max"]).to(device)

        grade = -1.0
        try:
            start_run(experiment_name=MLFLOW_EXPERIMENT, run_name=f"trial-{trial.number}")
            log_params_safe({**p, "num_envs": unwrapped.num_envs, "total_timesteps": total_timesteps,
                              "distance_low": args_cli.distance_low, "distance_high": args_cli.distance_high})

            if os.path.exists(bc_checkpoint_path):
                try:
                    load_bc_checkpoint(model, bc_checkpoint_path, map_location=device)
                except RuntimeError:
                    mlflow.set_tag("bc_load_failed", "true")
            else:
                mlflow.set_tag("bc_checkpoint_missing", "true")

            # Same 15% imitation / floor-not-fraction warmup / remainder PPO split as base_training_isaac.train().
            imitation_timesteps_grand = int(0.15 * total_timesteps)
            warmup_timesteps_grand = max(int(0.01 * total_timesteps), MIN_WARMUP_CHUNKS * chunk_grand_steps)
            ppo_total_timesteps_grand = total_timesteps - imitation_timesteps_grand - warmup_timesteps_grand
            post_imitation_timesteps_grand = warmup_timesteps_grand + ppo_total_timesteps_grand

            if imitation_timesteps_grand > 0:
                imitation_steps_per_env = max(1, imitation_timesteps_grand // unwrapped.num_envs)
                model = _run_imitation_stage_isaac(env, model, imitation_steps_per_env, gains_by_dist)

            actor_frozen = warmup_timesteps_grand > 0
            if actor_frozen:
                for param in model.shared.parameters():
                    param.requires_grad_(False)
                model.actor_mean.weight.requires_grad_(False)
                model.actor_mean.bias.requires_grad_(False)
                model.actor_log_std.requires_grad_(False)

            optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
            n_chunks = max(1, post_imitation_timesteps_grand // chunk_grand_steps)

            timesteps_done = imitation_timesteps_grand
            ppo_timesteps_done = 0
            ppo_obs = None  # None -> isaac_ppo_train does one full env.reset() at trial start
            for chunk in range(n_chunks):
                if actor_frozen and ppo_timesteps_done >= warmup_timesteps_grand:
                    for param in model.parameters():
                        param.requires_grad_(True)
                    actor_frozen = False

                progress = ppo_timesteps_done / post_imitation_timesteps_grand
                current_ent_coef = max(p["ent_coef_end"], p["ent_coef_start"] * (1 - progress))
                current_lr = cosine_lr(p["lr"], progress, min_ratio=p["lr_min_ratio"])
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                model, optimizer, episode_rewards, last_losses, ppo_obs = isaac_ppo_train(
                    env, total_timesteps=chunk_grand_steps, num_steps=NUM_STEPS_PER_CHUNK,
                    gamma=p["gamma"], lam=p["lam"], lr=current_lr, model=model, optimizer=optimizer,
                    ent_coef=current_ent_coef, target_kl=p["target_kl"], num_epochs=p["num_epochs"],
                    batch_size=max(1, chunk_grand_steps // p["num_minibatches"]),
                    clip_eps=p["clip_eps"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
                    initial_obs=ppo_obs,
                )
                timesteps_done += chunk_grand_steps
                ppo_timesteps_done += chunk_grand_steps
                grad_norm = last_losses.get("grad_norm", 0.0)

                outcomes = diagnose_with_model(model, eval_env, N_DIAGNOSTIC_EPISODES)
                success_rate = (outcomes["hover_success"] + outcomes.get("Hit", 0)) / N_DIAGNOSTIC_EPISODES
                grade, breakdown = compute_grade(
                    success_rate=success_rate, avg_final_dist=outcomes["avg_final_dist"],
                    avg_hit_time_sec=outcomes["avg_hit_time_sec"], avg_grad_norm=grad_norm,
                    oob_radius=oob_radius, episode_time_budget_sec=eval_env.max_steps * eval_env.dt,
                )

                recent_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
                log_metrics_safe({"recent_reward": recent_reward, "ent_coef": current_ent_coef,
                                   "lr": current_lr, "grad_norm": grad_norm, **breakdown}, step=timesteps_done)
                print(f"[optuna_search_isaac][trial {trial.number}] {timesteps_done}/{total_timesteps}  "
                      f"grade={grade:.4f}  success_rate={success_rate:.2f}")

                trial.report(grade, step=chunk)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    raise optuna.TrialPruned()

            mlflow.log_metric("final_grade", grade)
            return grade
        finally:
            if mlflow.active_run() is not None:
                mlflow.end_run()

    return objective


def run_study(objective, n_trials, study_name, storage, n_startup_trials, n_warmup_steps):
    if storage.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(storage[len("sqlite:///"):]) or ".", exist_ok=True)

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                                 sampler=sampler, pruner=pruner, load_if_exists=True)

    run_start = datetime.now()
    print(f"[optuna_search_isaac] starting {n_trials} trials at {run_start.isoformat(timespec='seconds')} "
          f"(study already has {len(study.trials)} trial(s) recorded)")
    # catch=(Exception,) -- a trial hitting a transient error is recorded
    # FAILED instead of aborting the whole unattended study.
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    run_end = datetime.now()

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"[optuna_search_isaac] done in {run_end - run_start} -- "
          f"{len(completed)} completed, {len(pruned)} pruned, {len(failed)} failed")

    if completed:
        print(f"[optuna_search_isaac] best grade={study.best_value:.4f}")
        print(f"[optuna_search_isaac] best params: {study.best_params}")
        os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump({"best_value": study.best_value, "best_params": study.best_params,
                       "best_trial_number": study.best_trial.number}, f, indent=2)

        start_run(experiment_name=MLFLOW_EXPERIMENT, run_name="study-summary")
        mlflow.log_metric("best_grade", study.best_value)
        mlflow.log_metric("n_completed", len(completed))
        mlflow.log_metric("n_pruned", len(pruned))
        mlflow.log_metric("n_failed", len(failed))
        log_params_safe({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.end_run()
    return study


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    oob_radius = max(20.0, args_cli.distance_high * 3.0)  # must match rewards.OOB_RADIUS's own derivation
    rng = np.random.default_rng()
    target_pairs = build_uniform_omni_eval_pairs(oob_radius=oob_radius, low=args_cli.distance_low,
                                                  high=args_cli.distance_high, n_pairs=500, rng=rng)

    with open(args_cli.gains_path) as f:
        gains_by_dist = json.load(f)

    eval_env = BaseDroneEnv(reward_func, target_pairs=target_pairs,
                             max_steps=steps_for_dist(args_cli.distance_high))

    objective = make_objective(env, gains_by_dist, target_pairs, eval_env, oob_radius,
                                args_cli.total_timesteps, args_cli.bc_checkpoint_path)

    run_study(objective, args_cli.n_trials, args_cli.study_name, args_cli.storage,
              args_cli.n_startup_trials, args_cli.n_warmup_steps)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
