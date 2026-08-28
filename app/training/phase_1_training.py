"""Phase 1 training loop (precision interception curriculum) -- see training-goals.md."""

import json
import os

import mlflow
import torch

from app.control.pid_hover import PIDHoverController
from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe
from app.guidance.plotting import plot_training_run
from app.guidance.train import ActorCritic, ppo_train, warmup_critic, device, cosine_lr
from app.reward_functions.reward_fn_phase1 import RewardFnPhase1
from app.reward_functions.rewards import chain_reward_fns
from app.training.eval_matrix import build_eval_pairs
from app.training.phase_0_training import (
    TrainConfig, log_metrics, save_checkpoint, best_params as phase0_best_params,
)
import numpy as np

# Gains are per-distance (not per-axis, since PID position error is a plain 3D vector).
with open("app/control/best_pid_gains_per_dist.json") as _f:
    PID_GAINS_BY_DIST = json.load(_f)

# 250m dropped: DAgger's hit-rate never moved off 0.00 there while 3/10/50/150m
# converged to 75-100%; deferred to a later curriculum phase.
PHASE1_DISTANCES = (3, 10, 50, 150)
PHASE1_OOB_RADIUS = 200  # comfortably above the 150m max distance above

# Two-ladder curriculum: xy first (matches existing BC/DAgger demo coverage), then
# xyz once xy is solid -- z is new territory for the policy and physically
# asymmetric (+z costs more thrust than -z), so it gets its own ladder.
PHASE1_STAGES = [(d, (0, 1)) for d in PHASE1_DISTANCES] + [(d, (0, 1, 2)) for d in PHASE1_DISTANCES]

# Curriculum gating: train on PHASE1_STAGES[0] alone until its hit-rate clears
# CURRICULUM_HIT_THRESHOLD for CURRICULUM_STREAK_LEN checkpoints in a row, then
# advance and reset the streak.
CURRICULUM_HIT_THRESHOLD = 0.7
CURRICULUM_STREAK_LEN = 5

# Re-warm (not cold-start) the critic on every stage transition, since return
# scale shifts with distance/episode length; fewer rounds than the initial
# cold-start warmup since this is a refresh, not building from random init.
STAGE_CRITIC_WARMUP_ROUNDS = 7

# Re-trigger PID-guided imitation shaping on every stage transition too, since
# each new stage's target_pairs is genuinely new territory for the policy.
STAGE_IMITATION_STEPS = 150_000

# phase0_best_params["lr"] was tuned for a from-scratch policy with a wide
# actor_log_std; phase 1 warm-starts from a much more precise BC/DAgger
# checkpoint, and reusing that LR blew approx_kl up on the first PPO update.
PHASE1_LR = phase0_best_params["lr"] / 20


best_params = {
    **phase0_best_params,
    "phase1_pos_coef": phase0_best_params["phase0_pos_coef"],
    "hit_reward": 10,
}

# AdamW's decoupled weight decay, plus cosine LR decay down to LR*LR_MIN_RATIO.
WEIGHT_DECAY = 1e-4
LR_MIN_RATIO = 0.01


def train(cfg: TrainConfig):
    reward_cfg = RewardFnPhase1(
        hit_steps_streak=1500,
        phase1_pos_coef=best_params["phase1_pos_coef"],
        hit_reward=best_params["hit_reward"],

        oob_radius=PHASE1_OOB_RADIUS,
        attitude_penalty=-1.0,
        oob_penalty=-1.5,
        streak_penalty_coef=best_params["streak_penalty_coef"],
        hover_success_steps=200,
        outer_dist=1.0,
        inner_dist=0.3,
        streak_cap=best_params["streak_cap"],
        phase0_pos_coef=best_params["phase0_pos_coef"],
        tilt_penalty_coef=best_params["tilt_penalty_coef"],
        ang_vel_penalty_coef=best_params["ang_vel_penalty_coef"],
        imitation_coef=best_params["imitation_coef"],
        imitation_duration_steps=50_000,
        phase0_duration_steps=100_000,
    )
    reward_fn = reward_cfg.as_roadmap()

    stage_idx = 0

    def set_stage(idx):
        nonlocal model
        dist, axes = PHASE1_STAGES[idx]
        env.target_pairs = build_eval_pairs(oob_radius=PHASE1_OOB_RADIUS, distances=(dist,), axes=axes)
        print(f"[phase1] curriculum stage -> {dist}m axes={axes} ({len(env.target_pairs)} pairs)")

        # Per-distance PID gains, so the imitation teacher is tuned for the current stage.
        env.pid_teacher = PIDHoverController(**PID_GAINS_BY_DIST[str(dist)])

        # Replaces env.reward_method outright (rather than mutating reward_cfg's
        # own chain) since that chain's step counter is cumulative across the run.
        env.reward_method = chain_reward_fns([
            (reward_cfg.phase_1_imitation, STAGE_IMITATION_STEPS),
            (reward_cfg.phase_1_fn, None),
        ])
        print(f"[phase1] imitation re-triggered for {STAGE_IMITATION_STEPS} steps")

        print(f"[phase1] re-warming critic for new stage ({STAGE_CRITIC_WARMUP_ROUNDS} rounds)...")
        model = warmup_critic(env, model, num_rounds=STAGE_CRITIC_WARMUP_ROUNDS,
                               num_steps=cfg.NUM_STEPS, gamma=cfg.GAMMA, lam=cfg.LAM)

    _stage0_dist, _stage0_axes = PHASE1_STAGES[0]
    env = InterceptorDroneEnv(
        reward_fn,
        target_pairs=build_eval_pairs(oob_radius=PHASE1_OOB_RADIUS, distances=(_stage0_dist,), axes=_stage0_axes),
    )
    # Same per-distance gains set_stage uses for every later transition.
    env.pid_teacher = PIDHoverController(**PID_GAINS_BY_DIST[str(_stage0_dist)])
    env.reset()
    print("target placed at:", env.target_pos)
    print("drone placed at:", env.drone_state.position)


    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                             env.action_space.low, env.action_space.high).to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.LR,
        weight_decay=WEIGHT_DECAY,
    )

    timesteps_done = 0
    all_episode_rewards = []
    drone_state_arr = []
    dist_history = []
    stage_streak = 0
    last_checkpoint_path = None

    model.load_state_dict(torch.load("app/control/pretrained_bc_dagger.pt", map_location=device))

    start_run(experiment_name="phase1-training", run_name=cfg.CHECKPOINT_DIR)
    log_params_safe({**vars(cfg), **vars(reward_cfg)})

    # critic_head is still at random init (BC/DAgger's imitation loss never trains
    # it); give it real value estimates before PPO updates touch the actor.
    print("[phase1] warming up critic before PPO updates...")
    model = warmup_critic(env, model, num_rounds=10, num_steps=cfg.NUM_STEPS, gamma=cfg.GAMMA, lam=cfg.LAM)

    while timesteps_done < cfg.TOTAL_TIMESTEPS:
        progress = timesteps_done / cfg.TOTAL_TIMESTEPS
        # Lower entropy coefficient than phase 0's schedule: this run resumes an
        # already-precise BC/DAgger checkpoint, not a from-scratch policy.
        current_ent_coef = max(0.00025, 0.005 * (1 - progress))

        current_lr = cosine_lr(cfg.LR, progress, min_ratio=LR_MIN_RATIO)

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
        mlflow.log_metric("stage_idx", stage_idx, step=timesteps_done)

        # Hit-rate (not hit+hover_success) gates curriculum progression: the task
        # is precision interception, not loitering nearby.
        success_rate = row["outcome_hit"] / cfg.N_DIAGNOSTIC_EPISODES
        stage_streak = stage_streak + 1 if success_rate >= CURRICULUM_HIT_THRESHOLD else 0
        _cur_dist, _cur_axes = PHASE1_STAGES[stage_idx]
        print(f"[phase1] stage={_cur_dist}m axes={_cur_axes}  success_rate={success_rate:.2f}  "
              f"streak={stage_streak}/{CURRICULUM_STREAK_LEN}")

        if stage_streak >= CURRICULUM_STREAK_LEN and stage_idx < len(PHASE1_STAGES) - 1:
            stage_idx += 1
            stage_streak = 0
            set_stage(stage_idx)


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
                # Only compare against distances trained so far, deduped (the same
                # distance appears in both the xy and xyz ladders).
                pid_baseline_distances=sorted(set(d for d, _ in PHASE1_STAGES[:stage_idx + 1])),
                pid_gains_by_dist_path="app/control/best_pid_gains_per_dist.json",
            )

    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
        mlflow.log_artifact(last_checkpoint_path)
    if os.path.exists(cfg.METRICS_CSV):
        mlflow.log_artifact(cfg.METRICS_CSV)
    mlflow.end_run()

    return model


if __name__ == "__main__":
    cfg = TrainConfig(
        CHECKPOINT_DIR="runs/phase1",
        METRICS_CSV="runs/phase1/metrics.csv",
        TOTAL_TIMESTEPS=5_000_000,
        NUM_STEPS=8192,
        NUM_EPOCHS=30,
        LR=PHASE1_LR,
    )

    train(cfg)
