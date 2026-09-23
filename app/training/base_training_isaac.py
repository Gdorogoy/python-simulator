"""Isaac Lab-native counterpart to app.training.base_training -- same algorithm
and phase split (imitation -> critic warmup -> PPO), driven by a live
GPU-batched DirectRLEnv instead of the numpy SubprocVecBaseDroneEnv. See
PROJECT_DEFENSE_GUIDE.md Part 6/8 for the full phase-split rationale and
MIGRATION_PROGRESS.md for the bug history behind MIN_WARMUP_CHUNKS etc.

Usage:
    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -m app.training.base_training_isaac --headless \\
        --num_envs 4096 --distance-low 3 --distance-high 10 --checkpoint-dir runs/base_training_isaac_3_10
"""

import argparse
import csv
import json
import os
from datetime import datetime

# Isaac venv's mlflow is a different version than what created the numpy venv's
# mlflow.db; point this file at a separate store instead of migrating that db.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow_isaac.db")

import mlflow
import numpy as np
import torch

import isaaclab.utils.math as math_utils

from app.control.pretrain_bc import pretrain_behavior_cloning
from app.control.step_budget import steps_for_dist
from app.control.torch_pid import TorchPIDController, assign_gains_by_distance
from app.environmental.base_drone_env import BaseDroneEnv
from app.guidance.export_onnx import export_onnx_model
from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe
from app.guidance.plotting import plot_training_run, plot_grad_norm, plot_grad_norm_3d
from app.guidance.train import ActorCritic, isaac_ppo_train, cosine_lr, load_bc_checkpoint
from app.guidance.utils import compute_grade
from app.reward_functions.rewards import reward_func, HIT_REWARD, GAMMA as REWARD_GAMMA
from app.training.diagnostics import diagnose_with_model, diagnose_with_pid
from app.training.eval_matrix import build_uniform_omni_eval_pairs
# Reused directly, not redefined, so the two configs can't silently drift apart.
from app.training.base_training import (
    PARAMS, ENT_COEF_START, ENT_COEF_END, WEIGHT_DECAY, LR_MIN_RATIO,
    N_DIAGNOSTIC_EPISODES, RECENCY_DECAY,
    IMITATION_RETRAIN_EPOCHS, IMITATION_BC_LR, IMITATION_BC_BATCH_SIZE, IMITATION_BUFFER_CAP_PAIRS,
    DEFAULT_DISTANCE_LOW, DEFAULT_DISTANCE_HIGH, DEFAULT_BC_CHECKPOINT_PATH,
)

# GRAND TOTAL env-steps across ALL parallel envs, not per-env like base_training.py's TOTAL_TIMESTEPS.
TOTAL_TIMESTEPS_ISAAC = 128_000_000
IMITATION_FRACTION = 0.15
WARMUP_FRACTION = 0.01
NUM_ENVS_ISAAC_DEFAULT = 4096
NUM_STEPS_PER_CHUNK = 256
# Floor, not a fixed per-env cadence -- a fixed cadence silently shrinks retrain
# rounds as NUM_ENVS_ISAAC grows. See MIGRATION_PROGRESS.md for the derivation.
MIN_IMITATION_RETRAIN_ROUNDS = 8
# Floor on critic-warmup chunks, same floor-not-fraction reasoning. Raised
# 8->16 after a real run still collapsed post-unfreeze at 8 -- see
# MIGRATION_PROGRESS.md's "Insufficient critic warmup" entry for the numbers.
MIN_WARMUP_CHUNKS = 16
# Ramps LR linearly over the first few post-unfreeze chunks instead of jumping
# straight to the cosine-schedule value, to blunt the grad_norm spike at unfreeze.
UNFREEZE_LR_RAMP_CHUNKS = 3
DEFAULT_CHECKPOINT_DIR_ISAAC = "runs/base_training_isaac"
# Soft periodic reset: once actor-unfrozen, if the trailing grade average drops
# RESET_DEGRADE_MARGIN below the best grade seen so far, blend weights back toward
# the best checkpoint instead of letting PPO drift indefinitely away from it.
# alpha=1 would be a hard reset; alpha<1 keeps some of the drifted weights.
RESET_SUSTAIN_CHUNKS = 5
RESET_DEGRADE_MARGIN = 0.15
RESET_BLEND_ALPHA = 0.5


def _select_pid_teacher_isaac(env, pid: TorchPIDController, gains_by_dist: dict, env_ids):
    assign_gains_by_distance(pid, env.unwrapped._start_dist, gains_by_dist, env_ids)


def _read_kinematics(env):
    unwrapped = env.unwrapped
    pos = unwrapped._robot.data.root_pos_w - unwrapped._terrain.env_origins
    vel = unwrapped._robot.data.root_lin_vel_w
    ang_vel = unwrapped._robot.data.root_ang_vel_b
    quat_wxyz = unwrapped._robot.data.root_quat_w
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_wxyz)
    target_local = unwrapped._desired_pos_w - unwrapped._terrain.env_origins
    return pos, vel, ang_vel, roll, pitch, yaw, target_local


def _run_imitation_stage_isaac(env, model, steps_budget_per_env: int, gains_by_dist: dict,
                                min_retrain_rounds: int = MIN_IMITATION_RETRAIN_ROUNDS):
    """Isaac-native counterpart to base_training._run_imitation_stage. Retrain cadence
    is derived from min_retrain_rounds, not a fixed per-env constant, to avoid
    under-training at large NUM_ENVS_ISAAC."""
    unwrapped = env.unwrapped
    device = unwrapped.device
    num_envs = unwrapped.num_envs
    retrain_every_per_env = max(1, steps_budget_per_env // max(1, min_retrain_rounds))

    pid = TorchPIDController(num_envs, device, **next(iter(gains_by_dist.values())))
    obs = env.reset()[0]["policy"]
    all_env_ids = torch.arange(num_envs, device=device)
    _select_pid_teacher_isaac(env, pid, gains_by_dist, all_env_ids)
    pid.reset()

    obs_buf, act_buf = [], []
    agg_obs, agg_actions, agg_block = None, None, None
    rng = np.random.default_rng()
    steps_done = 0
    block_idx = 0

    while steps_done < steps_budget_per_env:
        obs_before = obs
        with torch.no_grad():
            mean, _, _ = model.forward(obs_before)
            policy_action = model.scale_action(mean)

            pos, vel, ang_vel, roll, pitch, yaw, target_local = _read_kinematics(env)
            pid_action = pid.compute_action(pos, vel, ang_vel, roll, pitch, yaw,
                                             target_local, unwrapped._desired_yaw_w, dt=1 / 240)

        obs_buf.append(obs_before.cpu().numpy())
        act_buf.append(pid_action.cpu().numpy())

        next_obs_dict, reward, terminated, truncated, extras = env.step(policy_action)
        obs = next_obs_dict["policy"]
        steps_done += 1

        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            _select_pid_teacher_isaac(env, pid, gains_by_dist, done_ids)
            pid.reset(done_ids)

        if steps_done % retrain_every_per_env == 0 or steps_done >= steps_budget_per_env:
            block_idx += 1
            new_obs = np.concatenate(obs_buf, axis=0)
            new_actions = np.concatenate(act_buf, axis=0)
            new_block = np.full(len(new_obs), block_idx, dtype=np.int32)
            if agg_obs is None:
                agg_obs, agg_actions, agg_block = new_obs, new_actions, new_block
            else:
                agg_obs = np.concatenate([agg_obs, new_obs], axis=0)
                agg_actions = np.concatenate([agg_actions, new_actions], axis=0)
                agg_block = np.concatenate([agg_block, new_block], axis=0)

            if len(agg_obs) > IMITATION_BUFFER_CAP_PAIRS:
                keep_w = RECENCY_DECAY ** (block_idx - agg_block)
                idx = rng.choice(len(agg_obs), size=IMITATION_BUFFER_CAP_PAIRS, replace=False,
                                  p=keep_w / keep_w.sum())
                agg_obs, agg_actions, agg_block = agg_obs[idx], agg_actions[idx], agg_block[idx]

            train_w = RECENCY_DECAY ** (block_idx - agg_block)
            model = pretrain_behavior_cloning(
                model, obs=agg_obs, actions=agg_actions, weights=train_w,
                epochs=IMITATION_RETRAIN_EPOCHS, batch_size=IMITATION_BC_BATCH_SIZE, lr=IMITATION_BC_LR)
            print(f"[train_isaac][imitation] {steps_done}/{steps_budget_per_env} per-env steps -- "
                  f"retrained on {len(agg_obs)} aggregated (obs, pid_action) pairs "
                  f"({len(new_obs)} new this block)")
            obs_buf, act_buf = [], []

    return model


def _soft_reset_toward_checkpoint(model, optimizer, ckpt_path: str, blend_alpha: float,
                                   lr: float, weight_decay: float, device):
    """Blends model weights toward a saved checkpoint (alpha=1 fully replaces them) and
    reinitializes the optimizer, since Adam's momentum from the drifted region is no
    longer meaningful once the weights themselves jump back."""
    best_state = torch.load(ckpt_path, map_location=device)
    current_state = model.state_dict()
    blended = {k: blend_alpha * best_state[k] + (1 - blend_alpha) * current_state[k] for k in current_state}
    model.load_state_dict(blended)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


def _append_csv_row(row: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(env, distance_low=DEFAULT_DISTANCE_LOW, distance_high=DEFAULT_DISTANCE_HIGH,
          checkpoint_dir=DEFAULT_CHECKPOINT_DIR_ISAAC, bc_checkpoint_path=DEFAULT_BC_CHECKPOINT_PATH,
          total_timesteps=TOTAL_TIMESTEPS_ISAAC, seed=None, hparams_override: dict | None = None):
    """Runs one fixed-range training stage against a live Isaac env (build it yourself via
    gym.make(...) and pass it in). hparams_override optionally overrides any PARAMS key
    or the module-level entropy/weight-decay/lr-floor constants; unrecognized keys are ignored."""
    hparams_override = hparams_override or {}
    run_start = datetime.now()
    print(f"[train_isaac] run started at {run_start.isoformat(timespec='seconds')}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    unwrapped = env.unwrapped
    device = unwrapped.device
    num_envs = unwrapped.num_envs
    obs_dim = unwrapped.single_observation_space["policy"].shape[0]
    action_dim = unwrapped.single_action_space.shape[0]

    oob_radius = max(20.0, distance_high * 3.0)  # must match rewards.OOB_RADIUS

    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_csv = os.path.join(checkpoint_dir, "metrics.csv")
    p = dict(PARAMS)
    p.update({k: v for k, v in hparams_override.items() if k in p})
    ent_coef_start = hparams_override.get("ent_coef_start", ENT_COEF_START)
    ent_coef_end = hparams_override.get("ent_coef_end", ENT_COEF_END)
    weight_decay = hparams_override.get("weight_decay", WEIGHT_DECAY)
    lr_min_ratio = hparams_override.get("lr_min_ratio", LR_MIN_RATIO)

    target_pairs = build_uniform_omni_eval_pairs(oob_radius=oob_radius, low=distance_low, high=distance_high,
                                                  n_pairs=2000, rng=rng)
    unwrapped.set_target_pairs(target_pairs)
    print(f"[train_isaac] {len(target_pairs)} omni-directional target pairs, "
          f"distance ~ Uniform({distance_low}, {distance_high}), num_envs={num_envs}, seed={seed}")

    with open("app/control/best_pid_gains_per_dist.json") as f:
        gains_by_dist = json.load(f)

    model = ActorCritic(obs_dim, action_dim, unwrapped.single_action_space.low, unwrapped.single_action_space.high,
                         hidden=p["hidden"], num_hidden_layers=p["num_hidden_layers"],
                         dropout=p["dropout"], log_std_max=p["log_std_max"]).to(device)

    if os.path.exists(bc_checkpoint_path):
        load_bc_checkpoint(model, bc_checkpoint_path, map_location=device)
        print(f"[train_isaac] loaded BC/DAgger checkpoint: {bc_checkpoint_path}")
    else:
        print(f"[train_isaac] no BC checkpoint at {bc_checkpoint_path}, starting imitation from random init")

    chunk_grand_steps = NUM_STEPS_PER_CHUNK * num_envs

    imitation_timesteps_grand = int(IMITATION_FRACTION * total_timesteps)
    warmup_timesteps_grand = int(WARMUP_FRACTION * total_timesteps)
    warmup_timesteps_grand = max(warmup_timesteps_grand, MIN_WARMUP_CHUNKS * chunk_grand_steps)
    ppo_total_timesteps_grand = total_timesteps - imitation_timesteps_grand - warmup_timesteps_grand
    # warmup + PPO share one chunk loop below, matching base_training.py's real mechanism.
    post_imitation_timesteps_grand = warmup_timesteps_grand + ppo_total_timesteps_grand

    if imitation_timesteps_grand > 0:
        imitation_steps_per_env = max(1, imitation_timesteps_grand // num_envs)
        print(f"[train_isaac] imitation stage: {imitation_timesteps_grand} grand-total steps "
              f"({imitation_steps_per_env} per env)")
        model = _run_imitation_stage_isaac(env, model, imitation_steps_per_env, gains_by_dist)

    # Actor frozen for warmup_timesteps_grand worth of chunks, then unfrozen -- the
    # same isaac_ppo_train call used for the rest of training also IS the warmup.
    actor_frozen = warmup_timesteps_grand > 0
    if actor_frozen:
        for param in model.shared.parameters():
            param.requires_grad_(False)
        model.actor_mean.weight.requires_grad_(False)
        model.actor_mean.bias.requires_grad_(False)
        model.actor_log_std.requires_grad_(False)
        print(f"[train_isaac] actor frozen for the first {warmup_timesteps_grand} grand-total steps (critic warmup)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=weight_decay)
    n_chunks = max(1, post_imitation_timesteps_grand // chunk_grand_steps)

    eval_env = BaseDroneEnv(reward_func, target_pairs=target_pairs,
                             max_steps=steps_for_dist(distance_high))

    start_run(experiment_name="base-training-isaac",
              run_name=f"base_training_isaac_{distance_low:g}_{distance_high:g}")
    mlflow.set_tag("run_start_time", run_start.isoformat(timespec="seconds"))
    log_params_safe({**p, "num_envs": num_envs, "total_timesteps": total_timesteps,
                      "distance_low": distance_low, "distance_high": distance_high,
                      "bc_checkpoint_path": bc_checkpoint_path, "checkpoint_dir": checkpoint_dir,
                      "seed": seed, "n_target_pairs": len(target_pairs),
                      "ent_coef_start": ent_coef_start, "ent_coef_end": ent_coef_end,
                      "n_diagnostic_episodes": N_DIAGNOSTIC_EPISODES,
                      "weight_decay": weight_decay, "lr_min_ratio": lr_min_ratio,
                      "run_start_time": run_start.isoformat(timespec="seconds")})

    # Starts at imitation_timesteps_grand so checkpoint/mlflow steps read as progress
    # against the grand total, even though the loop below only spans post-imitation steps.
    timesteps_done = imitation_timesteps_grand
    ppo_timesteps_done = 0
    ckpt_path = None
    ppo_obs = None  # carried across chunks so isaac_ppo_train doesn't reset the env every chunk
    chunks_since_unfreeze = None
    best_grade = float("-inf")
    best_ckpt_path = None
    grade_history = []
    for chunk in range(n_chunks):
        if actor_frozen and ppo_timesteps_done >= warmup_timesteps_grand:
            for param in model.parameters():
                param.requires_grad_(True)
            actor_frozen = False
            chunks_since_unfreeze = 0
            print(f"[train_isaac] critic warmup done at {timesteps_done} steps -- actor unfrozen")
        elif chunks_since_unfreeze is not None:
            chunks_since_unfreeze += 1

        stage = "critic_warmup" if ppo_timesteps_done < warmup_timesteps_grand else "training"

        progress = ppo_timesteps_done / post_imitation_timesteps_grand
        current_ent_coef = max(ent_coef_end, ent_coef_start * (1 - progress))
        current_lr = cosine_lr(p["lr"], progress, min_ratio=lr_min_ratio)
        if chunks_since_unfreeze is not None and chunks_since_unfreeze < UNFREEZE_LR_RAMP_CHUNKS:
            current_lr *= (chunks_since_unfreeze + 1) / UNFREEZE_LR_RAMP_CHUNKS
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

        recent_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
        grad_norm = last_losses.get("grad_norm", 0.0)
        mlflow.log_metric("recent_reward", recent_reward, step=timesteps_done)
        mlflow.log_metric("ent_coef", current_ent_coef, step=timesteps_done)
        mlflow.log_metric("lr", current_lr, step=timesteps_done)
        mlflow.log_metric("grad_norm", grad_norm, step=timesteps_done)

        ckpt_path = os.path.join(checkpoint_dir, f"model_{timesteps_done}.pt")
        torch.save(model.state_dict(), ckpt_path)
        export_onnx_model(model, obs_dim, ckpt_path[:-3] + ".onnx")

        outcomes = diagnose_with_model(model, eval_env, N_DIAGNOSTIC_EPISODES)
        # reward_func never sets hover_success, so counting "Hit" too keeps this comparable to pid_success_rate.
        success_rate = (outcomes["hover_success"] + outcomes.get("Hit", 0)) / N_DIAGNOSTIC_EPISODES
        grade, breakdown = compute_grade(
            success_rate=success_rate, avg_final_dist=outcomes["avg_final_dist"],
            avg_hit_time_sec=outcomes["avg_hit_time_sec"], avg_grad_norm=grad_norm,
            oob_radius=oob_radius, episode_time_budget_sec=eval_env.max_steps * eval_env.dt,
        )
        reset_triggered = False
        if stage == "training":
            if grade > best_grade:
                best_grade = grade
                best_ckpt_path = ckpt_path
            grade_history.append(grade)
            if len(grade_history) >= RESET_SUSTAIN_CHUNKS and best_ckpt_path is not None:
                trailing_avg = float(np.mean(grade_history[-RESET_SUSTAIN_CHUNKS:]))
                if best_grade - trailing_avg >= RESET_DEGRADE_MARGIN:
                    print(f"[train_isaac] trailing grade {trailing_avg:.3f} is {best_grade - trailing_avg:.3f} "
                          f"below best {best_grade:.3f} -- soft-resetting {RESET_BLEND_ALPHA:.2f} toward "
                          f"{best_ckpt_path}")
                    model, optimizer = _soft_reset_toward_checkpoint(
                        model, optimizer, best_ckpt_path, RESET_BLEND_ALPHA, current_lr, weight_decay, device)
                    reset_triggered = True
                    grade_history = []  # cooldown: need a fresh sustain window before resetting again

        pid_outcomes = diagnose_with_pid(eval_env.pid_teacher, eval_env, N_DIAGNOSTIC_EPISODES,
                                          gains_by_dist=gains_by_dist)
        pid_outcomes.pop("final_dists")
        pid_avg_final_dist = pid_outcomes.pop("avg_final_dist")
        pid_outcomes.pop("avg_hit_time_sec")
        pid_success_rate = pid_outcomes.pop("Hit", 0) / N_DIAGNOSTIC_EPISODES

        effective_std = np.exp(model.log_std_min + 0.5 * (model.log_std_max - model.log_std_min) *
                                (np.tanh(model.actor_log_std.detach().cpu().numpy()) + 1))
        effective_std_mean = float(np.mean(effective_std))
        total_param_norm = sum(param.data.norm().item() for param in model.parameters())

        final_dists = outcomes.pop("final_dists")
        avg_final_dist = outcomes.pop("avg_final_dist")
        avg_hit_time_sec = outcomes.pop("avg_hit_time_sec")
        hit_count = outcomes.pop("Hit", 0)
        outcome_counts = {**outcomes, "hit": hit_count}
        std_final_dist = float(np.std(final_dists)) if final_dists else 0.0
        min_final_dist = float(np.min(final_dists)) if final_dists else 0.0

        row = {
            "timesteps": timesteps_done, "stage": stage, "ent_coef": current_ent_coef, "lr": current_lr,
            "policy_loss": last_losses.get("policy_loss", 0.0),
            "value_loss": last_losses.get("value_loss", 0.0),
            "entropy_loss": last_losses.get("entropy_loss", 0.0),
            "approx_kl": last_losses.get("approx_kl", 0.0),
            "early_stopped": last_losses.get("early_stopped", False),
            "grad_norm": grad_norm, "recent_reward": recent_reward,
            "avg_final_dist": avg_final_dist, "std_final_dist": std_final_dist,
            "min_final_dist": min_final_dist,
            "avg_hit_time_sec": avg_hit_time_sec if avg_hit_time_sec is not None else "",
            "effective_std_mean": effective_std_mean, "total_param_norm": total_param_norm,
            "pid_avg_final_dist": pid_avg_final_dist, "pid_success_rate": pid_success_rate,
            "reset_triggered": reset_triggered, "best_grade": best_grade,
            **breakdown,
            **{f"outcome_{k}": v for k, v in outcome_counts.items()},
            **{f"pid_outcome_{k}": v for k, v in pid_outcomes.items()},
        }
        log_metrics_safe({k: v for k, v in row.items() if k not in ("stage", "early_stopped")},
                          step=timesteps_done)
        mlflow.set_tag("stage", stage)
        _append_csv_row(row, metrics_csv)
        print(f"[train_isaac] {timesteps_done}/{total_timesteps}  stage={stage}  ent_coef={current_ent_coef:.4f}  "
              f"lr={current_lr:.2e}  grad_norm={grad_norm:.3f}  grade={grade:.4f}  best_grade={best_grade:.4f}  "
              f"reset={reset_triggered}  success_rate={success_rate:.2f}  outcomes={outcome_counts}  "
              f"pid_success_rate={pid_success_rate:.2f}  pid_avg_final_dist={pid_avg_final_dist:.2f}")

    if ckpt_path:
        mlflow.log_artifact(ckpt_path)
        onnx_path = ckpt_path[:-3] + ".onnx"
        if os.path.exists(onnx_path):
            mlflow.log_artifact(onnx_path)
    if os.path.exists(metrics_csv):
        mlflow.log_artifact(metrics_csv)

    plots_dir = os.path.join(checkpoint_dir, "plots")
    plot_training_run(
        metrics_csv, output_dir=plots_dir,
        hover_success_steps=200, n_diag_episodes=N_DIAGNOSTIC_EPISODES, hit_threshold=0.3,
        max_steps=eval_env.max_steps, oob_radius=oob_radius,
        hit_reward=HIT_REWARD, attitude_penalty=-1.0, oob_penalty=-1.5,
    )
    plot_grad_norm(metrics_csv, output_dir=plots_dir)
    plot_grad_norm_3d(metrics_csv, output_dir=plots_dir)
    if os.path.isdir(plots_dir):
        mlflow.log_artifacts(plots_dir, artifact_path="plots")

    run_end = datetime.now()
    mlflow.set_tag("run_end_time", run_end.isoformat(timespec="seconds"))
    mlflow.end_run()
    print(f"[train_isaac] done. Checkpoints in {checkpoint_dir}, metrics in {metrics_csv}, plots in {plots_dir}")
    print(f"[train_isaac] run started at {run_start.isoformat(timespec='seconds')}, "
          f"ended at {run_end.isoformat(timespec='seconds')} (duration {run_end - run_start})")
    return model
