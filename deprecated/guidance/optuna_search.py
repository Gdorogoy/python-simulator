"""Optuna hyperparameter search over deprecated.training.base_training's network/PPO
knobs and entropy/weight-decay/lr-floor schedule -- same pipeline as that
script (imitation -> critic warmup -> PPO against reward_func, same NUM_ENVS,
same distance_low/distance_high range, same BC warm start), just leaner per
trial: no per-chunk onnx export or plotting, fewer diagnostic episodes, and
optuna pruning so a trial that's clearly not working stops before burning its
full 1,000,000-step budget. Every trial's params and per-chunk grade are
logged to MLflow (same sqlite:///mlflow.db store base_training.py uses).

gamma is NOT searched -- base_training.py requires it to equal rewards.GAMMA
for potential-based shaping's policy-invariance guarantee (see that module's
PARAMS comment). dropout is NOT searched either -- fixed at 0.0, because
PPO's ratio/KL check relies on the actor's forward pass being deterministic
(old_log_prob vs new_log_prob), and this pipeline never switches eval()/
train() mode; base_training.py's PARAMS comment documents a confirmed
empirical failure (approx_kl~29 during a frozen-actor warmup, when the actor
literally cannot move) traced to exactly this.

Only one trial's SubprocVecBaseDroneEnv runs at a time (num_workers already
uses cpu_count-1 worker processes internally) -- don't raise study.optimize's
n_jobs above 1, that would oversubscribe every core several times over.

Usage:
    python -m deprecated.guidance.optuna_search --n-trials 100
Resumes automatically if run again with the same --study-name/--storage
(same trial count target: re-running with --n-trials 100 after 40 already
ran adds 100 MORE, not tops up to 100).
"""
import argparse
import json
import os
from datetime import datetime

import mlflow
import numpy as np
import optuna
import torch

from app.control.step_budget import steps_for_dist
from app.environmental.base_drone_env import BaseDroneEnv
from app.environmental.subproc_vec_base_drone_env import SubprocVecBaseDroneEnv
from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe
from app.guidance.train import ActorCritic, vec_ppo_train, cosine_lr, device, load_bc_checkpoint
from app.guidance.utils import compute_grade
from app.reward_functions.rewards import reward_func, GAMMA as REWARD_GAMMA
from deprecated.training.base_training import (
    _build_reward_fn, build_target_pairs,
    _run_imitation_stage, DEFAULT_DISTANCE_LOW, DEFAULT_DISTANCE_HIGH,
    DEFAULT_BC_CHECKPOINT_PATH, NUM_ENVS, TOTAL_TIMESTEPS, CHUNK_TIMESTEPS,
    IMITATION_DURATION_STEPS, WARMUP_DURATION_STEPS, PPO_TOTAL_TIMESTEPS,
    UNFREEZE_LR_RAMP_CHUNKS,
)
from app.training.diagnostics import diagnose_with_model

DEFAULT_STUDY_NAME = "base_training_optuna"
DEFAULT_STORAGE = "sqlite:///runs/optuna/study.db"
MLFLOW_EXPERIMENT = "base-training-optuna"
BEST_PARAMS_PATH = "runs/optuna/best_params.json"
# Fewer than base_training.py's 10 -- speed over precision, this only needs to
# rank trials against each other, not report a publication-grade number.
N_DIAGNOSTIC_EPISODES = 5


def suggest_params(trial: optuna.Trial) -> dict:
    """Every tunable knob in base_training.PARAMS, plus the entropy-decay/
    weight-decay/lr-floor constants that script hardcodes as module-level
    constants, all over deliberately wide ranges -- see module docstring for
    what's excluded (gamma, dropout) and why."""
    ent_coef_start = trial.suggest_float("ent_coef_start", 1e-4, 0.1, log=True)
    return dict(
        hidden=trial.suggest_categorical("hidden", [32, 64, 96, 128, 192, 256]),
        num_hidden_layers=trial.suggest_int("num_hidden_layers", 2, 8),
        dropout=0.0,
        lr=trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        gamma=REWARD_GAMMA,
        lam=trial.suggest_float("lam", 0.85, 0.999),
        clip_eps=trial.suggest_float("clip_eps", 0.05, 0.4),
        vf_coef=trial.suggest_float("vf_coef", 0.1, 1.5),
        target_kl=trial.suggest_float("target_kl", 0.005, 0.15, log=True),
        num_epochs=trial.suggest_int("num_epochs", 3, 20),
        num_minibatches=trial.suggest_categorical("num_minibatches", [4, 8, 16, 32, 64, 128]),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.1, 2.0),
        log_std_max=trial.suggest_float("log_std_max", -2.5, -0.05),
        ent_coef_start=ent_coef_start,
        # End-of-run entropy as a FRACTION of the start value, not an independent
        # draw -- an independent draw could put ent_coef_end above ent_coef_start,
        # which would turn base_training.py's decay schedule
        # (max(ent_coef_end, ent_coef_start*(1-progress))) into an increase.
        ent_coef_end=ent_coef_start * trial.suggest_float("ent_coef_end_frac", 0.01, 0.9),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        lr_min_ratio=trial.suggest_float("lr_min_ratio", 0.0, 0.5),
    )


def objective(trial: optuna.Trial) -> float:
    p = suggest_params(trial)
    rng = np.random.default_rng(trial.number)
    oob_radius = max(20.0, DEFAULT_DISTANCE_HIGH * 3.0)  # must match rewards.OOB_RADIUS's own derivation
    episode_max_steps = steps_for_dist(DEFAULT_DISTANCE_HIGH)
    target_pairs = build_target_pairs(DEFAULT_DISTANCE_LOW, DEFAULT_DISTANCE_HIGH, oob_radius, rng=rng)

    vec_env = SubprocVecBaseDroneEnv(_build_reward_fn, p, target_pairs, NUM_ENVS, max_steps=episode_max_steps)
    grade = -1.0
    try:
        start_run(experiment_name=MLFLOW_EXPERIMENT, run_name=f"trial-{trial.number}")
        log_params_safe({**p, "num_envs": NUM_ENVS, "total_timesteps": TOTAL_TIMESTEPS,
                          "distance_low": DEFAULT_DISTANCE_LOW, "distance_high": DEFAULT_DISTANCE_HIGH})

        model = ActorCritic(vec_env.observation_space.shape[0], vec_env.action_space.shape[0],
                             vec_env.action_space.low, vec_env.action_space.high,
                             hidden=p["hidden"], num_hidden_layers=p["num_hidden_layers"],
                             dropout=p["dropout"], log_std_max=p["log_std_max"]).to(device)

        if os.path.exists(DEFAULT_BC_CHECKPOINT_PATH):
            try:
                load_bc_checkpoint(model, DEFAULT_BC_CHECKPOINT_PATH, map_location=device)
            except RuntimeError:
                mlflow.set_tag("bc_load_failed", "true")
        else:
            mlflow.set_tag("bc_checkpoint_missing", "true")

        if IMITATION_DURATION_STEPS > 0:
            model = _run_imitation_stage(vec_env, model, IMITATION_DURATION_STEPS)

        actor_frozen = WARMUP_DURATION_STEPS > 0
        if actor_frozen:
            for param in model.shared.parameters():
                param.requires_grad_(False)
            model.actor_mean.weight.requires_grad_(False)
            model.actor_mean.bias.requires_grad_(False)
            model.actor_log_std.requires_grad_(False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

        buffer_size = CHUNK_TIMESTEPS * NUM_ENVS
        batch_size = max(1, buffer_size // p["num_minibatches"])
        eval_env = BaseDroneEnv(reward_func, target_pairs=target_pairs, max_steps=episode_max_steps)

        timesteps_done = IMITATION_DURATION_STEPS
        ppo_timesteps_done = 0
        n_chunks = PPO_TOTAL_TIMESTEPS // CHUNK_TIMESTEPS
        chunks_since_unfreeze = None

        for chunk in range(n_chunks):
            if actor_frozen and ppo_timesteps_done >= WARMUP_DURATION_STEPS:
                for param in model.parameters():
                    param.requires_grad_(True)
                actor_frozen = False
                chunks_since_unfreeze = 0
            elif chunks_since_unfreeze is not None:
                chunks_since_unfreeze += 1

            progress = ppo_timesteps_done / PPO_TOTAL_TIMESTEPS
            current_ent_coef = max(p["ent_coef_end"], p["ent_coef_start"] * (1 - progress))
            current_lr = cosine_lr(p["lr"], progress, min_ratio=p["lr_min_ratio"])
            if chunks_since_unfreeze is not None and chunks_since_unfreeze < UNFREEZE_LR_RAMP_CHUNKS:
                current_lr *= (chunks_since_unfreeze + 1) / UNFREEZE_LR_RAMP_CHUNKS
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            model, optimizer, episode_rewards, last_losses = vec_ppo_train(
                vec_env, total_timesteps=buffer_size, num_steps=CHUNK_TIMESTEPS,
                gamma=p["gamma"], lam=p["lam"], lr=current_lr, model=model, optimizer=optimizer,
                ent_coef=current_ent_coef, target_kl=p["target_kl"], num_epochs=p["num_epochs"],
                batch_size=batch_size, clip_eps=p["clip_eps"], vf_coef=p["vf_coef"],
                max_grad_norm=p["max_grad_norm"],
            )
            timesteps_done += CHUNK_TIMESTEPS
            ppo_timesteps_done += CHUNK_TIMESTEPS
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
            print(f"[optuna_search][trial {trial.number}] {timesteps_done}/{TOTAL_TIMESTEPS}  "
                  f"grade={grade:.4f}  success_rate={success_rate:.2f}")

            trial.report(grade, step=chunk)
            if trial.should_prune():
                mlflow.set_tag("pruned", "true")
                raise optuna.TrialPruned()

        mlflow.log_metric("final_grade", grade)
        return grade
    finally:
        vec_env.close()
        if mlflow.active_run() is not None:
            mlflow.end_run()


def run_study(n_trials, study_name, storage, n_startup_trials, n_warmup_steps):
    if storage.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(storage[len("sqlite:///"):]) or ".", exist_ok=True)

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                                 sampler=sampler, pruner=pruner, load_if_exists=True)

    run_start = datetime.now()
    print(f"[optuna_search] starting {n_trials} trials at {run_start.isoformat(timespec='seconds')} "
          f"(study already has {len(study.trials)} trial(s) recorded)")
    # catch=(Exception,) -- a single trial hitting a transient error (CUDA OOM,
    # a worker pipe hiccup) shouldn't abort a run meant to go unattended for
    # many hours; that trial is just recorded FAILED and the study moves on.
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    run_end = datetime.now()

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"[optuna_search] done in {run_end - run_start} -- "
          f"{len(completed)} completed, {len(pruned)} pruned, {len(failed)} failed")

    if completed:
        print(f"[optuna_search] best grade={study.best_value:.4f}")
        print(f"[optuna_search] best params: {study.best_params}")
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


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--n-startup-trials", type=int, default=10,
                         help="Trials that always run to completion (no pruning) before the "
                              "median pruner has enough history to judge new trials against.")
    parser.add_argument("--n-warmup-steps", type=int, default=8,
                         help="PPO chunks each trial always gets before it becomes prunable -- "
                              "covers critic warmup (4 chunks) plus a few chunks of real PPO, so "
                              "a trial isn't killed while its actor is still frozen.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_study(args.n_trials, args.study_name, args.storage, args.n_startup_trials, args.n_warmup_steps)
