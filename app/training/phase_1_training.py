"""
Phase 1
training-goals.md for more info
"""

import json

import torch

from app.control.pid_hover import PIDHoverController
from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.plotting import plot_training_run
from app.guidance.train import ActorCritic, ppo_train, warmup_critic, device, cosine_lr
from app.reward_functions.reward_fn_phase1 import RewardFnPhase1
from app.reward_functions.rewards import chain_reward_fns
from app.training.eval_matrix import build_eval_pairs
from app.training.phase_0_training import (
    TrainConfig, log_metrics, save_checkpoint, best_params as phase0_best_params,
)
import numpy as np

# gains are per-distance, not per-axis (PID position error is a plain 3D
# vector -- pid_hover.py's compute_action already handles x/y/z targets the
# same way) so the same lookup covers both the xy and xyz ladders.
with open("app/control/best_pid_gains_per_dist.json") as _f:
    PID_GAINS_BY_DIST = json.load(_f)

# 250m dropped -- DAgger's hit-rate never moved off 0.00 there across 10 full
# rounds even after fixing the observation scaling and step-time floor, while
# 3/10/50/150m all converged to 75-100%. Deferred to its own later curriculum
# phase rather than blocking this one. Matches the distances DAgger actually
# trained pretrained_bc_dagger.pt on.
PHASE1_DISTANCES = (3, 10, 50, 150)
PHASE1_OOB_RADIUS = 200  # comfortably above the 150m max distance above

# Two-ladder curriculum: xy first (axes=(0,1), 4 directions -- matches all
# existing BC/DAgger demo coverage), then xyz (axes=(0,1,2), up to 6
# directions, minus any -z pair build_eval_pairs drops for landing
# underground) once xy is solid. z is its own ladder rather than folded into
# the first because it's new territory for the warm-started policy (no
# BC/DAgger data existed for it before this session's collect_demonstrations
# /dagger.py update) AND physically asymmetric (+z costs more thrust than
# -z, unlike the x/-x, y/-y symmetry the xy ladder gets for free).
PHASE1_STAGES = [(d, (0, 1)) for d in PHASE1_DISTANCES] + [(d, (0, 1, 2)) for d in PHASE1_DISTANCES]

# Curriculum gating: train on PHASE1_STAGES[0] alone (not the pooled 16-pair
# set) until its hit-rate (outcome_hit / n_diag_episodes, from log_metrics'
# diagnostic episodes -- which run on whatever env.target_pairs currently is,
# so they automatically score the CURRENT stage) clears CURRICULUM_HIT_THRESHOLD
# for CURRICULUM_STREAK_LEN checkpoints in a row, then advance to the next
# stage and reset the streak. Matches plotting.py's own STREAK_LEN=5
# convergence-check convention.
CURRICULUM_HIT_THRESHOLD = 0.7
CURRICULUM_STREAK_LEN = 5

# Re-warm (not cold-start) the critic on every stage transition -- value_loss
# bouncing around in runs/phase1/metrics.csv is consistent with the critic
# being calibrated to the OLD stage's return scale (return magnitude scales
# with distance/episode length) right after a switch. Fewer rounds than the
# initial cold-start warmup since this is a refresh of an already-decent
# critic, not building one from random init.
STAGE_CRITIC_WARMUP_ROUNDS = 7

# Re-trigger imitation (PID-guided reward shaping) on every stage transition
# too, not just once at the very start of the whole run -- the new stage's
# target_pairs is genuinely new territory (different distance and/or the
# xyz ladder's new axes), so leaning back on the PID teacher's guidance for
# a window before falling back to phase_1_fn alone helps stabilize the
# transition the same way it helped the initial BC/DAgger warm start.
STAGE_IMITATION_STEPS = 150_000

# phase0_best_params["lr"] was Optuna-tuned (and then manually inflated 100x,
# see phase_0_training.py's TrainConfig.LR comment) for a FROM-SCRATCH policy
# whose actor_log_std starts near its wide end (std~0.22). Phase 1 warm-starts
# from a BC/DAgger checkpoint and clamps actor_log_std down to std~0.05-0.11
# (train.py's post-update clamp_) to preserve that precision -- reusing the
# from-scratch LR against that much tighter std blew approx_kl into the
# 100s-1000s (target_kl ~0.02) on the very first PPO update in
# runs/phase1/metrics.csv, permanently wrecking the warm start (0 hits across
# the entire ~9.87M-timestep run). Starting an order of magnitude lower.
PHASE1_LR = phase0_best_params["lr"] / 20


best_params = {
    **phase0_best_params,
    "phase1_pos_coef": phase0_best_params["phase0_pos_coef"],
    "hit_reward": 10,
}

# AdamW's own decoupled weight decay (not hand-rolled L2 anywhere), and cosine
# LR decay down to LR * LR_MIN_RATIO (not to 0) over the course of the run.
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

    # curriculum starts on PHASE1_STAGES[0] alone, not the old pooled-16-pair
    # set -- see set_stage below for how/when it advances to the next stage.
    stage_idx = 0

    def set_stage(idx):
        nonlocal model
        dist, axes = PHASE1_STAGES[idx]
        env.target_pairs = build_eval_pairs(oob_radius=PHASE1_OOB_RADIUS, distances=(dist,), axes=axes)
        print(f"[phase1] curriculum stage -> {dist}m axes={axes} ({len(env.target_pairs)} pairs)")

        # per-distance gains, not the single-target best_pid_gains.json
        # InterceptorDroneEnv defaults to -- matches what collect_demonstrations.py/
        # dagger.py/the PID-baseline comparison already use, so phase_1_imitation's
        # teacher is actually tuned for the distance it's currently guiding at.
        env.pid_teacher = PIDHoverController(**PID_GAINS_BY_DIST[str(dist)])

        # fresh imitation window for the new stage -- no warmup stage here (not
        # a cold start), just imitation_coef-guided shaping for STAGE_IMITATION_STEPS
        # then phase_1_fn on its own for the rest of this stage. Replaces
        # env.reward_method outright rather than mutating reward_cfg's own
        # as_roadmap() chain, since that chain's internal step counter is
        # cumulative across the whole run and was never meant to reset.
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
    # same per-distance gains as set_stage uses for every later transition --
    # stage 0 would otherwise be the one stage still using the generic
    # single-target best_pid_gains.json InterceptorDroneEnv defaults to.
    env.pid_teacher = PIDHoverController(**PID_GAINS_BY_DIST[str(_stage0_dist)])
    env.reset()
    print("target placed at:", env.target_pos)
    print("dorne placet at:", env.drone_state.position)


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


    model.load_state_dict(torch.load("app/control/pretrained_bc_dagger.pt", map_location=device))

    # critic_head is still at random init -- BC/DAgger's imitation loss never
    # trained it (see warmup_critic's docstring). Give it real value estimates
    # before real PPO updates start touching the actor.
    print("[phase1] warming up critic before PPO updates...")
    model = warmup_critic(env, model, num_rounds=10, num_steps=cfg.NUM_STEPS, gamma=cfg.GAMMA, lam=cfg.LAM)

    while timesteps_done < cfg.TOTAL_TIMESTEPS:
        progress = timesteps_done / cfg.TOTAL_TIMESTEPS
        # lower than phase_0's schedule (was max(0.001, 0.01*(1-progress))) --
        # this run resumes an already-precise BC/DAgger checkpoint, not a
        # from-scratch policy, so it doesn't need phase-0-scale exploration
        # pressure racing its std back up toward the (now lower) clamp ceiling.
        # Raised from (0.0003, 0.003) -> (0.00025, 0.005): effective_std_mean
        # sat essentially frozen (~0.057, drifting <0.02% over 4.8M steps) the
        # whole prior run -- more entropy pressure within the SAME actor_log_std
        # clamp_(-3.0,-1.2) bounds (unchanged) is a smaller, more contained lever
        # than widening the clamp itself.
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
        row = log_metrics(
            env, model, timesteps_done, current_ent_coef,
            last_losses.get("policy_loss", 0.0), last_losses.get("value_loss", 0.0),
            last_losses.get("entropy_loss", 0.0), last_losses.get("approx_kl", 0.0),
            last_losses.get("early_stopped", False), last_losses.get("grad_norm", 0.0),
            csv_path=cfg.METRICS_CSV, n_diag_episodes=cfg.N_DIAGNOSTIC_EPISODES,
        )

        # curriculum gate: this checkpoint's diagnostics just ran on env.target_pairs
        # as it stood during THIS chunk, i.e. the current stage -- so row["outcome_hit"]
        # is exactly the current stage's hit-rate signal to gate progression on.
        # Deliberately hit-only, not hit+hover_success: the task is precision
        # interception (crossing hit_threshold), not loitering nearby.
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
                # only compare against distances actually trained so far -- pooling
                # the full curriculum while e.g. still gated on the 3m stage would
                # score the policy against 150m tasks it hasn't been trained on yet.
                # dedupe since the same distance appears in both the xy and xyz
                # ladders (this PID comparison itself stays xy-only regardless --
                # compute_pid_baseline's distances-mode doesn't take an axes param).
                pid_baseline_distances=sorted(set(d for d, _ in PHASE1_STAGES[:stage_idx + 1])),
                pid_gains_by_dist_path="app/control/best_pid_gains_per_dist.json",
            )

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
