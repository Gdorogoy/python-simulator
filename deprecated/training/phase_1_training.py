"""Phase 1 training loop (precision interception): trains one fixed model to
convergence on a hand-picked BEST_PARAMS set (originally found by a since-removed
hyperparameter search) instead of running a search.

Builds its own reward config (_build_reward_cfg/build_phase1_reward_fn below), but:
  - builds its own omni-directional target-pair set (every combined-axis
    direction, not just one axis at a time -- see build_omni_eval_pairs),
    capped to DISTANCES -- replaces the old distance/axis-staged curriculum
    ladder with one unified set covering every direction from the start;
  - uses SubprocVecBaseDroneEnv (multiprocess, one worker per core)
    instead of a single non-vectorized env -- VecBaseDroneEnv's
    Python for-loop is single-threaded and was the measured throughput
    bottleneck (~92 per-env steps/sec regardless of network architecture
    size); the multiprocess version measured ~526 per-env steps/sec;
  - uses AdamW (not plain Adam) with weight decay, and cosine LR decay
    across the whole run. max_grad_norm clipping is applied inside
    ppo_update regardless; grad_norm is logged/printed every chunk so
    that's visibly verified, not assumed.

Usage:
    python -m deprecated.training.phase_1_training
"""
import csv
import os

import mlflow
import numpy as np
import torch

from app.environmental.base_drone_env import BaseDroneEnv
from app.environmental.subproc_vec_base_drone_env import SubprocVecBaseDroneEnv, PID_GAINS_BY_DIST_PATH
from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe
from app.guidance.plotting import plot_training_run, plot_time_to_hit
from app.guidance.train import ActorCritic, vec_ppo_train, cosine_lr, device, load_bc_checkpoint
from app.guidance.utils import compute_grade
from app.training.diagnostics import diagnose_with_model
from app.training.eval_matrix import build_random_omni_eval_pairs
from deprecated.reward_fn_phase1 import RewardFnPhase1

OOB_RADIUS = 200.0
BC_CHECKPOINT_PATH = "app/control/pretrained_bc_dagger.pt"  # matches dagger.py's default out_path


def _build_reward_cfg(p: dict, warmup_duration_steps=0, imitation_duration_steps=0, phase1_duration_steps=None):
    return RewardFnPhase1(
        hit_steps_streak=1500,  # required arg, unused within the class -- fixed placeholder,
                                 # matching every other caller in the repo (all hardcode 1500)
        phase1_pos_coef=p["phase1_pos_coef"], hit_reward=p["hit_reward"],
        warmup_duration_steps=warmup_duration_steps,
        imitation_duration_steps=imitation_duration_steps, phase1_duration_steps=phase1_duration_steps,
        velocity_gain=p["velocity_gain"],
        oob_radius=OOB_RADIUS, attitude_penalty=-1.0, oob_penalty=-1.5, hover_success_steps=200,
        outer_dist=1.0, inner_dist=0.3,
        streak_penalty_coef=p["streak_penalty_coef"], streak_cap=p["streak_cap"],
        tilt_penalty_coef=p["tilt_penalty_coef"], ang_vel_penalty_coef=p["ang_vel_penalty_coef"],
        imitation_coef=p["imitation_coef"], zone_bonus=p["zone_bonus"],
        dist_penalty_coef=p["dist_penalty_coef"], approach_gain=p["approach_gain"],
        vel_penalty_coef=p["vel_penalty_coef"], step_penalty=p["step_penalty"],
        closer_bonus_val=p["closer_bonus_val"],
    )


def build_phase1_reward_fn(p: dict, warmup_duration_steps=0, imitation_duration_steps=0, phase1_duration_steps=None):
    """Plain top-level wrapper returning RewardFnPhase1's roadmap -- the
    reward_fn_factory SubprocVecBaseDroneEnv's workers pickle by reference (a
    lambda/closure isn't picklable across the spawn boundary, but a plain
    module-level function is). Mirrors rewards.build_base_reward_fn."""
    return _build_reward_cfg(p, warmup_duration_steps=warmup_duration_steps,
                              imitation_duration_steps=imitation_duration_steps,
                              phase1_duration_steps=phase1_duration_steps).as_roadmap()


# Hand-tuned param set (originally the best trial from a since-removed
# hyperparameter search).
BEST_PARAMS = {
    # Continuous values rounded to 2 sig figs (last digit snapped to 0/5) since
    # there's no search to re-derive them from; integer arch/count knobs left exact.
    # dropout=0 (was 0.365) -- see deprecated.training.base_training.PARAMS['dropout']'s
    # comment: with no eval()/train() mode switching anywhere in this pipeline,
    # dropout>0 in the shared trunk makes PPO's old_log_prob/new_log_prob (and
    # therefore approx_kl and the clipped ratio) noise-corrupted rather than a
    # real measure of policy change, for the whole run.
    'hidden': 64, 'num_hidden_layers': 4, 'dropout': 0.0,
    'lr': 0.000995, 'gamma': 0.97,
    'lam': 0.97, 'ent_coef': 0.00915,
    'clip_eps': 0.285, 'vf_coef': 0.525,
    'target_kl': 0.0245, 'num_epochs': 7, 'num_minibatches': 64,
    'max_grad_norm': 0.585, 'log_std_max': -0.89,
    'streak_penalty_coef': -0.0135, 'streak_cap': 15,
    'phase1_pos_coef': 0.66, 'hit_reward': 13.5,
    'velocity_gain': 0.0655, 'tilt_penalty_coef': 0.205,
    'ang_vel_penalty_coef': 0.15, 'imitation_coef': 0.25,
    'zone_bonus': 0.39, 'dist_penalty_coef': 1.3,
    'approach_gain': 1.6, 'vel_penalty_coef': 0.195,
    'step_penalty': 0.02, 'closer_bonus_val': 0.0105,
}

DISTANCES = (3, 10, 50)
# SubprocVecBaseDroneEnv defaults num_workers to cpu_count-1 (19 here); at
# NUM_ENVS=32 that's <2 envs/worker, underusing the per-worker step-batching
# that made the multiprocess vec-env faster in the first place. Bumped to
# spread ~4 envs/worker instead. NOTE: buffer_size = CHUNK_TIMESTEPS*NUM_ENVS,
# so this also grows the PPO minibatch size (num_minibatches stays fixed at
# 64) -- BEST_PARAMS was tuned against the old NUM_ENVS=32, so re-verify
# training stability/throughput after this change rather than assuming it's free.
NUM_ENVS = 76
# Per-env steps -- ~526 per-env steps/sec measured with the multiprocess
# vec-env, so 2_000_000 here is roughly an hour.
TOTAL_TIMESTEPS = 2_000_000
CHUNK_TIMESTEPS = 50_000  # per-env steps between checkpoints/diagnostics/LR updates
EVAL_EPISODES = 20
CHECKPOINT_DIR = "runs/phase1"
MLFLOW_EXPERIMENT = "phase1-training"
WEIGHT_DECAY = 1e-4
LR_MIN_RATIO = 0.01

# Reward-curriculum roadmap (per-env env.step() counts -- chain_reward_fns
# counts one call per env, see _build_reward_cfg): warmup (pure approach
# shaping, no imitation term) -> imitation (approach + PID-teacher-matching
# term) -> phase1 (approach + closing-velocity bonus) -> base_reward_fn
# (the real objective) for the remaining budget. This run's budget is big
# enough to give warmup its own steps instead of skipping straight to imitation.
WARMUP_DURATION_STEPS = 20_000
IMITATION_DURATION_STEPS = 200_000
PHASE1_DURATION_STEPS = 300_000


def _roadmap_stage(timesteps_done: int) -> str:
    if timesteps_done <= WARMUP_DURATION_STEPS:
        return "warmup"
    if timesteps_done <= WARMUP_DURATION_STEPS + IMITATION_DURATION_STEPS:
        return "imitation"
    if timesteps_done <= WARMUP_DURATION_STEPS + IMITATION_DURATION_STEPS + PHASE1_DURATION_STEPS:
        return "phase1"
    return "base"


def _append_csv_row(row: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_target_pairs():
    # Random directions, not the fixed axis-aligned/equal-component lattice --
    # generalizes to arbitrary skewed targets (e.g. (40, 15, 10)), not just the
    # engineered single/combined-axis corners. Distances stay locked to
    # DISTANCES; only direction is dense/random.
    return build_random_omni_eval_pairs(oob_radius=OOB_RADIUS, distances=DISTANCES, n_per_distance=200)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    metrics_csv = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    p = BEST_PARAMS
    target_pairs = build_target_pairs()
    print(f"[phase1] {len(target_pairs)} omni-directional target pairs across distances {DISTANCES}")
    print(f"[phase1] roadmap (per-env steps): warmup 0-{WARMUP_DURATION_STEPS}, "
          f"imitation {WARMUP_DURATION_STEPS}-{WARMUP_DURATION_STEPS + IMITATION_DURATION_STEPS}, "
          f"phase1 {WARMUP_DURATION_STEPS + IMITATION_DURATION_STEPS}-"
          f"{WARMUP_DURATION_STEPS + IMITATION_DURATION_STEPS + PHASE1_DURATION_STEPS}, "
          f"base {WARMUP_DURATION_STEPS + IMITATION_DURATION_STEPS + PHASE1_DURATION_STEPS}+")

    vec_env = SubprocVecBaseDroneEnv(
        build_phase1_reward_fn, p, target_pairs, NUM_ENVS,
        factory_kwargs=dict(warmup_duration_steps=WARMUP_DURATION_STEPS,
                             imitation_duration_steps=IMITATION_DURATION_STEPS,
                             phase1_duration_steps=PHASE1_DURATION_STEPS))
    try:
        model = ActorCritic(vec_env.observation_space.shape[0], vec_env.action_space.shape[0],
                             vec_env.action_space.low, vec_env.action_space.high,
                             hidden=p["hidden"], num_hidden_layers=p["num_hidden_layers"],
                             dropout=p["dropout"], log_std_max=p["log_std_max"]).to(device)

        if os.path.exists(BC_CHECKPOINT_PATH):
            # Remaps by Linear position, not raw nn.Sequential index -- the
            # checkpoint was saved at dropout=0, this run's dropout=0.3634 shifts
            # every later Linear's raw index (see load_bc_checkpoint).
            load_bc_checkpoint(model, BC_CHECKPOINT_PATH, map_location=device)
            print(f"[phase1] loaded BC/DAgger checkpoint: {BC_CHECKPOINT_PATH}")
        else:
            print(f"[phase1] WARNING: no BC checkpoint at {BC_CHECKPOINT_PATH}, starting from random init")

        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=WEIGHT_DECAY)

        buffer_size = CHUNK_TIMESTEPS * NUM_ENVS
        batch_size = max(1, buffer_size // p["num_minibatches"])

        start_run(experiment_name=MLFLOW_EXPERIMENT, run_name="phase1")
        log_params_safe({**p, "num_envs": NUM_ENVS, "total_timesteps": TOTAL_TIMESTEPS,
                          "distances": str(DISTANCES), "n_target_pairs": len(target_pairs),
                          "weight_decay": WEIGHT_DECAY, "lr_min_ratio": LR_MIN_RATIO})

        timesteps_done = 0
        n_chunks = TOTAL_TIMESTEPS // CHUNK_TIMESTEPS
        ckpt_path = None
        for chunk in range(n_chunks):
            progress = timesteps_done / TOTAL_TIMESTEPS
            current_lr = cosine_lr(p["lr"], progress, min_ratio=LR_MIN_RATIO)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            model, optimizer, episode_rewards, last_losses = vec_ppo_train(
                vec_env, total_timesteps=buffer_size, num_steps=CHUNK_TIMESTEPS,
                gamma=p["gamma"], lam=p["lam"], lr=current_lr, model=model, optimizer=optimizer,
                ent_coef=p["ent_coef"], target_kl=p["target_kl"], num_epochs=p["num_epochs"],
                batch_size=batch_size, clip_eps=p["clip_eps"], vf_coef=p["vf_coef"],
                max_grad_norm=p["max_grad_norm"],
            )
            timesteps_done += CHUNK_TIMESTEPS

            recent_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
            grad_norm = last_losses.get("grad_norm", 0.0)
            mlflow.log_metric("recent_reward", recent_reward, step=timesteps_done)
            mlflow.log_metric("lr", current_lr, step=timesteps_done)
            mlflow.log_metric("grad_norm", grad_norm, step=timesteps_done)

            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_{timesteps_done}.pt")
            torch.save(model.state_dict(), ckpt_path)

            eval_reward_cfg = _build_reward_cfg(p, warmup_duration_steps=WARMUP_DURATION_STEPS,
                                                 imitation_duration_steps=IMITATION_DURATION_STEPS,
                                                 phase1_duration_steps=PHASE1_DURATION_STEPS)
            eval_env = BaseDroneEnv(eval_reward_cfg.as_roadmap(), target_pairs=target_pairs)
            outcomes = diagnose_with_model(model, eval_env, EVAL_EPISODES)
            # _check_hit terminates an episode as "Hit" before hover_steps_in_zone
            # can ever reach hover_success_steps, so the two are mutually
            # exclusive per episode -- counting both credits either kind of
            # success instead of only the stricter sustained-hover one.
            success_rate = (outcomes["hover_success"] + outcomes.get("Hit", 0)) / EVAL_EPISODES
            grade, breakdown = compute_grade(
                success_rate=success_rate, avg_final_dist=outcomes["avg_final_dist"],
                avg_hit_time_sec=outcomes["avg_hit_time_sec"], avg_grad_norm=grad_norm,
                oob_radius=OOB_RADIUS, episode_time_budget_sec=eval_env.max_steps * eval_env.dt,
            )

            # Live snapshot of what targets the 32 training envs are actually
            # spawning right now (not just the static target_pairs list) --
            # answers "is training really touching the full spread of
            # distances/directions" empirically, not just by construction.
            # start_dists (fixed at each episode's reset), not live_dists (the
            # current, drifting distance-to-target) -- this metric is about
            # spawn coverage, not live divergence.
            live_targets, start_dists, _live_dists = vec_env.get_target_positions()
            target_pos_metrics = {
                "target_dist_mean": float(start_dists.mean()), "target_dist_min": float(start_dists.min()),
                "target_dist_max": float(start_dists.max()),
                "target_x_mean": float(live_targets[:, 0].mean()), "target_y_mean": float(live_targets[:, 1].mean()),
                "target_z_mean": float(live_targets[:, 2].mean()),
            }
            stage = _roadmap_stage(timesteps_done)

            # effective_std_mean/total_param_norm -- same computation as
            # phase_0_training.log_metrics, needed by plot_training_run's
            # policy_std plot.
            effective_std = np.exp(model.log_std_min + 0.5 * (model.log_std_max - model.log_std_min) *
                                    (np.tanh(model.actor_log_std.detach().cpu().numpy()) + 1))
            effective_std_mean = float(np.mean(effective_std))
            total_param_norm = sum(param.data.norm().item() for param in model.parameters())

            # outcomes mixes per-episode counts ("oob", "Hit", ...) with two
            # aggregate stats (avg_final_dist, avg_hit_time_sec) and a raw
            # per-episode list (final_dists) -- only the counts belong in the
            # outcome_ stackplot; plot_training_run also expects the count key
            # named "hit" (lowercase), matching _check_hit's "Hit" reason
            # string renamed for that one column.
            final_dists = outcomes.pop("final_dists")
            avg_final_dist = outcomes.pop("avg_final_dist")
            avg_hit_time_sec = outcomes.pop("avg_hit_time_sec")
            hit_count = outcomes.pop("Hit", 0)
            outcome_counts = {**outcomes, "hit": hit_count}
            std_final_dist = float(np.std(final_dists)) if final_dists else 0.0
            min_final_dist = float(np.min(final_dists)) if final_dists else 0.0

            row = {
                "timesteps": timesteps_done, "stage": stage, "lr": current_lr,
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
                **breakdown,
                **{f"outcome_{k}": v for k, v in outcome_counts.items()},
                **target_pos_metrics,
            }
            log_metrics_safe({k: v for k, v in row.items() if k not in ("stage", "early_stopped")},
                              step=timesteps_done)
            mlflow.set_tag("roadmap_stage", stage)
            _append_csv_row(row, metrics_csv)
            print(f"[phase1] {timesteps_done}/{TOTAL_TIMESTEPS}  stage={stage}  lr={current_lr:.2e}  "
                  f"grad_norm={grad_norm:.3f}  grade={grade:.4f}  success_rate={success_rate:.2f}  "
                  f"avg_hit_time_sec={avg_hit_time_sec}  "
                  f"target_dist=[{target_pos_metrics['target_dist_min']:.1f},{target_pos_metrics['target_dist_max']:.1f}] "
                  f"outcomes={outcome_counts}")

        if ckpt_path:
            mlflow.log_artifact(ckpt_path)
        if os.path.exists(metrics_csv):
            mlflow.log_artifact(metrics_csv)

        plots_dir = os.path.join(CHECKPOINT_DIR, "plots")
        plot_training_run(
            metrics_csv, output_dir=plots_dir,
            hover_success_steps=200, n_diag_episodes=EVAL_EPISODES, hit_threshold=0.3,
            streak_cap=p["streak_cap"], max_steps=eval_env.max_steps, oob_radius=OOB_RADIUS,
            hit_reward=p["hit_reward"], attitude_penalty=-1.0, oob_penalty=-1.5,
            streak_penalty_coef=p["streak_penalty_coef"],
            pid_baseline_distances=DISTANCES, pid_gains_by_dist_path=PID_GAINS_BY_DIST_PATH,
        )
        plot_time_to_hit(metrics_csv, output_dir=plots_dir)
        for png in os.listdir(plots_dir):
            if png.endswith(".png"):
                mlflow.log_artifact(os.path.join(plots_dir, png))

        mlflow.end_run()
        print(f"[phase1] done. Checkpoints in {CHECKPOINT_DIR}, metrics in {metrics_csv}, plots in {plots_dir}")
        return model
    finally:
        vec_env.close()


if __name__ == "__main__":
    train()
