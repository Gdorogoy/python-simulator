"""Training loop for app.reward_functions.rewards.reward_func -- the one flat
potential-based reward, used identically on every step of every episode. No
reward-fn phases/roadmap/config class.

Budget split (1,000,000 steps total):
  - 15% imitation: on-policy, run INSIDE this script (_run_imitation_stage) --
    not the same thing as the DAgger/BC checkpoint below. The actor is first
    warm-started from that checkpoint, then this stage keeps refining it
    on-policy against each worker's distance-matched PID teacher before PPO
    ever touches it.
  - >=8% critic-only warmup (WARMUP_DURATION_STEPS, floored to MIN_WARMUP_CHUNKS
    chunks, not a raw fraction -- see that constant's comment): actor frozen so
    the (randomly-initialized) critic calibrates against the warm-started policy
    before PPO starts updating it -- untrained-critic advantage noise can
    otherwise immediately wreck a warm-started policy. Which reward is active
    during this stretch doesn't matter for the policy, only for the value
    targets, so it's the same reward_func throughout, not a different one. The
    first UNFREEZE_LR_RAMP_CHUNKS chunks after unfreeze also ramp LR up
    linearly rather than jumping to the full cosine-schedule value.
  - ~77% (rest of the budget after imitation+warmup): standard PPO, actor+critic,
    against reward_func.

Targets are drawn from a CONTINUOUS Uniform(distance_low, distance_high)
range with a random 3D direction (build_uniform_omni_eval_pairs), not a
discrete distance ladder.

Curriculum staging: this script trains one FIXED distance range per run, not
an auto-widening curriculum. To teach longer range, run it multiple times
with an increasing (distance_low, distance_high) window, each stage
warm-starting from the previous stage's final checkpoint:

    python -m app.training.base_training --distance-low 3 --distance-high 10 \
        --checkpoint-dir runs/base_training_3_10
    python -m app.training.base_training --distance-low 3 --distance-high 50 \
        --checkpoint-dir runs/base_training_3_50 \
        --bc-checkpoint runs/base_training_3_10/model_1000000.pt
    python -m app.training.base_training --distance-low 3 --distance-high 100 \
        --checkpoint-dir runs/base_training_3_100 \
        --bc-checkpoint runs/base_training_3_50/model_1000000.pt
    python -m app.training.base_training --distance-low 3 --distance-high 200 \
        --checkpoint-dir runs/base_training_3_200 \
        --bc-checkpoint runs/base_training_3_100/model_1000000.pt

Goal of each stage:
  - 3-10m: learn clean close-range approach/hover against the PID teacher
    while the search space is small; this is also what the DAgger checkpoint
    (BC_CHECKPOINT_PATH-equivalent, --bc-checkpoint default) was built for.
  - 3-50m / 3-100m / 3-200m: reuse that close-range competence as a warm
    start and stretch it to longer straight-line flight + terminal approach,
    one distance-bucket step at a time, rather than forcing PPO to learn
    both "fly far" and "land precisely" from scratch at long range. Each
    step's own OOB_RADIUS auto-scales with --distance-high
    (max(20, distance_high*3)); best_pid_gains_per_dist.json already has
    tuned PID gains for the 3/10/50/100/150/250 buckets that
    SubprocVecBaseDroneEnv's per-episode teacher selection needs, so no
    extra tuning is required to add a stage within that ladder.
  - --checkpoint-dir MUST differ per stage -- timesteps_done restarts at
    IMITATION_DURATION_STEPS every run, so two stages sharing a directory
    will silently overwrite each other's model_<timesteps>.pt/metrics.csv.

Uses SubprocVecBaseDroneEnv (multiprocess, one worker per core) -- see
subproc_vec_base_drone_env.py; ~526 (vs ~92 single-process) per-env steps/sec.

reward_func's own knobs (TARGET_FRACTION, APPROACH_MILESTONE_BUDGET, HIT_REWARD,
milestone fractions/bonuses, OOB_RADIUS/attitude limits) are plain module-level
constants in rewards.py, not per-run hyperparameters -- see that module for
their values and APPROACH_MILESTONE_BUDGET's provenance (measured via the
tuned PID, hit bonus included; rerun that calibration and update the constant
if best_pid_gains_per_dist.json or the reward changes).

Every checkpoint is saved as both .pt and .onnx; metrics are appended to a
CSV row per checkpoint (one row per CHUNK_TIMESTEPS of training, not per raw
env-step -- 1,000,000/20,000 = 50 rows over a full run); diagnostics run
10 deterministic episodes per checkpoint, plus a PID-teacher reference run
on the same target_pairs (see diagnose_with_pid) so metrics.csv/plots show
what a hand-tuned controller achieves on the identical task, not just the
policy in isolation.

Usage:
    python -m app.training.base_training
    python -m app.training.base_training --distance-low 3 --distance-high 50 \
        --checkpoint-dir runs/base_training_3_50 --bc-checkpoint <prior stage's .pt>
"""
import argparse
import csv
import json
import os
from datetime import datetime

import mlflow
import numpy as np
import torch

from app.control.pretrain_bc import pretrain_behavior_cloning
from app.control.step_budget import steps_for_dist
from app.environmental.base_drone_env import BaseDroneEnv
from app.environmental.subproc_vec_base_drone_env import SubprocVecBaseDroneEnv, PID_GAINS_BY_DIST_PATH
from app.guidance.export_onnx import export_onnx_model
from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe
from app.guidance.plotting import plot_training_run, plot_worker_metrics
from app.guidance.train import ActorCritic, vec_ppo_train, cosine_lr, device, load_bc_checkpoint
from app.guidance.utils import compute_grade
from app.reward_functions.rewards import reward_func, HIT_REWARD, GAMMA as REWARD_GAMMA
from app.training.diagnostics import diagnose_with_model, diagnose_with_pid
from app.training.eval_matrix import build_uniform_omni_eval_pairs

# Defaults for the first (shortest-range) curriculum stage; override any of
# these per stage via train()'s args or the --distance-low/--distance-high/
# --checkpoint-dir/--bc-checkpoint CLI flags -- see "Curriculum staging" above.
DEFAULT_DISTANCE_LOW = 3.0
DEFAULT_DISTANCE_HIGH = 10.0
DEFAULT_CHECKPOINT_DIR = "runs/base_training"
DEFAULT_BC_CHECKPOINT_PATH = "app/control/pretrained_bc_dagger.pt"  # good hit ratios per
                                                                     # DAgger eval; must match
                                                                     # PARAMS['hidden']/'num_hidden_layers'
# SubprocVecBaseDroneEnv defaults num_workers to cpu_count-1 (~19 on a 20-core
# box); bumped so each worker gets several envs instead of <1, which is what
# actually made the multiprocess vec-env faster than single-process in the
# first place.
NUM_ENVS = 128

# No hyperparameter search has been run for reward_func's training loop yet --
# network/PPO knobs borrowed from phase1's tuned BEST_PARAMS where the knob
# means the same thing. reward_func itself takes no per-run params (see
# rewards.py's module-level constants). 'gamma' below MUST equal
# rewards.GAMMA -- potential-based shaping's policy-invariance guarantee
# requires the reward's own shaping gamma and PPO's return/advantage gamma
# to be the same value, so it's pulled from rewards.py rather than repeated.
PARAMS = {
    # dropout=0 -- PPO's ratio/KL relies on old_log_prob and new_log_prob coming
    # from the SAME deterministic function; with dropout>0 and no eval()/train()
    # mode switching anywhere in this pipeline, every forward pass draws a fresh
    # random mask, so old/new log_probs differ from mask noise alone, not policy
    # change. Confirmed empirically: a run showed approx_kl~29 (target_kl=0.0245)
    # during critic warmup, when the actor is frozen and literally cannot move --
    # the only source of that "divergence" is the dropout-mask mismatch. This
    # corrupts the trust-region check for the whole run, not just warmup, and is
    # the leading suspect behind post-unfreeze collapses that never recover.
    'hidden': 64, 'num_hidden_layers': 4, 'dropout': 0.0,
    'lr': 5e-5, 'gamma': REWARD_GAMMA, 'lam': 0.97,
    'clip_eps': 0.285, 'vf_coef': 0.525, 'target_kl': 0.0245,
    'num_epochs': 10, 'num_minibatches': 64, 'max_grad_norm': 0.25,
    'log_std_max': -0.9,
}

# Per-env steps (NOT scaled by NUM_ENVS; total env-steps across all workers =
# TOTAL_TIMESTEPS * NUM_ENVS). Split 25% imitation / 1% critic warmup / 74% PPO.
TOTAL_TIMESTEPS = 1_000_000
IMITATION_DURATION_STEPS = int(0.15 * TOTAL_TIMESTEPS)  # 15%: on-policy imitation (see _run_imitation_stage)
CHUNK_TIMESTEPS = 20_000  # per-env steps between checkpoints/diagnostics/ent_coef+lr updates
# Floor, not the raw 1% fraction -- at this TOTAL_TIMESTEPS/CHUNK_TIMESTEPS, 1%
# (10,000) is LESS than one chunk (20,000), so the actor was unfreezing after
# just ~1 chunk of real critic-only updates. base_training_isaac.py already
# learned this lesson (its own MIN_WARMUP_CHUNKS floor, added after a run showed
# the collapse-and-never-recover failure mode) -- mirrored here. 4 chunks here
# carries roughly the same total (experience-steps * NUM_ENVS) critic-update
# volume as that file's 8-chunk floor, since this script's chunks are bigger.
MIN_WARMUP_CHUNKS = 4
WARMUP_DURATION_STEPS = max(int(0.01 * TOTAL_TIMESTEPS), MIN_WARMUP_CHUNKS * CHUNK_TIMESTEPS)
PPO_TOTAL_TIMESTEPS = TOTAL_TIMESTEPS - IMITATION_DURATION_STEPS  # rest: warmup + training
# Chunks right after the actor unfreezes ramp LR up linearly instead of jumping
# straight to the full cosine-schedule value -- a real run showed grad_norm
# spike to ~121 (vs. ~20-30 elsewhere) on the very first post-unfreeze chunk,
# with avg_final_dist exploding 0.7m -> 13.9m -> 23.3m over the next two chunks
# and never recovering for the rest of the run. Tempering the first few
# post-unfreeze updates gives the (still not perfectly calibrated) critic's
# advantage estimates less room to blow up the policy in one shot.
UNFREEZE_LR_RAMP_CHUNKS = 3
IMITATION_RETRAIN_EVERY = CHUNK_TIMESTEPS  # per-env steps between BC retrain passes during imitation
IMITATION_RETRAIN_EPOCHS = 5
IMITATION_BC_LR = 3e-4
IMITATION_BC_BATCH_SIZE = 4096
# Aggregate buffer cap, in raw (obs, pid_action) pairs -- see _run_imitation_stage.
# 1,000,000 pairs max (was 5,120,000 = 2 blocks of IMITATION_RETRAIN_EVERY*NUM_ENVS,
# then 2,000,000) -- still aggregates across multiple rounds (not back to the
# single-block catastrophic-forgetting case the buffer was added to fix), just a
# shorter retained window, for less per-round BC retrain compute/memory. Does not
# affect IMITATION_RETRAIN_EVERY/NUM_ENVS (how much is collected per round).
IMITATION_BUFFER_CAP_PAIRS = 1_000_000
# Per-block decay for both eviction-keep-probability and retrain loss weight, in
# _run_imitation_stage -- a pair from `k` blocks ago gets weight RECENCY_DECAY**k.
# Matches app.control.dagger.RECENCY_DECAY.
RECENCY_DECAY = 0.85
N_DIAGNOSTIC_EPISODES = 10
MLFLOW_EXPERIMENT = "base-training"
WEIGHT_DECAY = 1e-4
LR_MIN_RATIO = 0.01  # cosine lr floor as a fraction of PARAMS['lr']

# Entropy decays over the run: exploration matters more early -- combined-xyz
# is a much larger joint-action search space than single-axis -- less once
# the policy has something worth refining.
#
# TODO: TEMPORARY -- lowered from ENT_COEF_START=0.02/ENT_COEF_END=0.0095 to
# test whether the lingering entropy bonus (std is already tightly bounded to
# [0.05, 0.41] by log_std_max, so 0.0095 was never doing much useful
# exploration this late) is why a trained policy circles near a missed target
# instead of committing to a corrective final approach -- see 2026-09-17
# discussion. Revert to the values above if this run doesn't fix that behavior.
ENT_COEF_START = 0.01
ENT_COEF_END = 0.001


def _build_reward_fn(p, **kwargs):
    """reward_fn_factory required by SubprocVecBaseDroneEnv (called as
    reward_fn_factory(p, **factory_kwargs) inside each worker) -- reward_func
    itself takes no config (see rewards.py's module-level constants), so
    this just hands it back; it exists only to satisfy that call signature."""
    return reward_func


def build_target_pairs(distance_low, distance_high, oob_radius, rng=None):
    return build_uniform_omni_eval_pairs(oob_radius=oob_radius, low=distance_low, high=distance_high,
                                          n_pairs=2000, rng=rng)


def _run_imitation_stage(vec_env, model, steps_budget):
    """On-policy imitation: the CURRENT policy (deterministic mean action,
    matching app.control.dagger's convention) drives every env, while each
    worker's PID teacher (already swapped to the nearest-distance tuned gains
    on every reset -- see subproc_vec_base_drone_env._select_pid_teacher)
    supplies the corrective action label for that same pre-step state via
    step_with_pid_actions(). Every IMITATION_RETRAIN_EVERY steps, the newly
    collected (obs, pid_action) pairs are folded into a GROWING aggregate
    (capped at IMITATION_BUFFER_CAP_PAIRS pairs, subsampled at
    random rather than truncated chronologically once over cap -- keeps a
    representative mix of every round instead of silently dropping whichever
    happens to be oldest) and one supervised BC retrain pass
    (pretrain_behavior_cloning) runs on that aggregate.

    Training only on each block's own fresh pairs (discarding them
    afterward) was tried first and produces catastrophic forgetting: the
    network fits whatever narrow slice of state-space the latest rollout
    visited, with nothing anchoring it to what earlier blocks already
    taught it, so a later block's low training loss doesn't reflect actual
    competence on the full target distribution. Aggregating (like
    app.control.dagger.dagger() already does across its rounds) is what
    DAgger's own convergence argument actually depends on -- this is that
    same mechanic, budgeted by steps instead of rounds so it fits inside
    this script's continuous run, and driven through the fast multiprocess
    vec_env instead of a single non-vectorized env.

    Both the buffer-cap eviction and the retrain loss are recency-weighted by
    block index (RECENCY_DECAY ** blocks-old, mirroring app.control.dagger's
    RECENCY_DECAY): plain uniform aggregation gives an early, barely-trained
    policy's mistakes the same say as the current policy's, which is backwards
    for DAgger -- the whole point is correcting where the CURRENT policy
    actually goes wrong."""
    obs = vec_env.reset()
    obs_buf, act_buf = [], []
    agg_obs, agg_actions, agg_block = None, None, None
    buffer_cap = IMITATION_BUFFER_CAP_PAIRS
    rng = np.random.default_rng()
    steps_done = 0
    block_idx = 0

    while steps_done < steps_budget:
        obs_before = obs
        obs_t = torch.as_tensor(obs_before, dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, _, _ = model.forward(obs_t)
            policy_actions = model.scale_action(mean).cpu().numpy()

        obs, rewards, terminated, truncated, infos, pid_actions = vec_env.step_with_pid_actions(policy_actions)
        obs_buf.append(obs_before)
        act_buf.append(pid_actions)
        steps_done += 1

        if steps_done % IMITATION_RETRAIN_EVERY == 0 or steps_done >= steps_budget:
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

            if len(agg_obs) > buffer_cap:
                # Recency-weighted eviction, mirroring app.control.dagger -- older
                # blocks are more likely to be dropped than uniform random would give.
                keep_w = RECENCY_DECAY ** (block_idx - agg_block)
                idx = rng.choice(len(agg_obs), size=buffer_cap, replace=False,
                                  p=keep_w / keep_w.sum())
                agg_obs, agg_actions, agg_block = agg_obs[idx], agg_actions[idx], agg_block[idx]

            train_w = RECENCY_DECAY ** (block_idx - agg_block)
            model = pretrain_behavior_cloning(
                model, obs=agg_obs, actions=agg_actions, weights=train_w,
                epochs=IMITATION_RETRAIN_EPOCHS, batch_size=IMITATION_BC_BATCH_SIZE, lr=IMITATION_BC_LR)
            print(f"[train][imitation] {steps_done}/{steps_budget} steps -- "
                  f"retrained on {len(agg_obs)} aggregated (obs, pid_action) pairs "
                  f"({len(new_obs)} new this block)")
            obs_buf, act_buf = [], []

    return model


def _append_csv_row(row: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(distance_low=DEFAULT_DISTANCE_LOW, distance_high=DEFAULT_DISTANCE_HIGH,
          checkpoint_dir=DEFAULT_CHECKPOINT_DIR, bc_checkpoint_path=DEFAULT_BC_CHECKPOINT_PATH, seed=None):
    """Runs one fixed-range training stage. See the module docstring's
    "Curriculum staging" section for how to chain several calls/CLI
    invocations (each with a wider distance_high and checkpoint_dir, warm
    starting bc_checkpoint_path from the previous stage's final .pt) into a
    short-to-long curriculum.

    seed, if given, makes model init, PPO's action sampling/minibatch
    shuffling (torch's global RNG), and target-pair sampling (a seeded
    Generator, not numpy's global RNG) reproducible run-to-run. It does NOT
    make the SubprocVecBaseDroneEnv workers' own PyBullet physics
    deterministic across processes -- each worker seeds its own env
    independently, so two seeded runs still diverge once PPO starts
    collecting rollouts. Useful for isolating "did my code change behavior"
    from "did the RNG just roll differently" during the imitation/warmup
    stages and target-pair generation, not for bit-identical full runs."""
    run_start = datetime.now()
    print(f"[train] run started at {run_start.isoformat(timespec='seconds')}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    oob_radius = max(20.0, distance_high * 3.0)  # must match rewards.OOB_RADIUS

    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_csv = os.path.join(checkpoint_dir, "metrics.csv")
    p = dict(PARAMS)

    target_pairs = build_target_pairs(distance_low, distance_high, oob_radius, rng=rng)
    print(f"[train] {len(target_pairs)} omni-directional target pairs, "
          f"distance ~ Uniform({distance_low}, {distance_high}), seed={seed}")

    # For diagnose_with_pid's per-episode gain-matching below -- eval_env spans the
    # whole distance_low-distance_high range, so a single fixed gain set understates
    # what PID actually achieves (see subproc_vec_base_drone_env._select_pid_teacher,
    # which is why the imitation-stage teacher already gets swapped this way).
    with open(PID_GAINS_BY_DIST_PATH) as f:
        gains_by_dist = json.load(f)

    # BaseDroneEnv defaults to max_steps=15_000 -- 6-8x steps_for_dist(distance_high)
    # for this 3-10m task. Left unset, episodes that should truncate at
    # ~1800-2500 steps instead run to 15_000 before resetting: step_penalty
    # (calibrated assuming the shorter budget) massively over-accumulates on
    # a stuck/non-progressing episode, and both PPO rollout collection and
    # the imitation stage waste most of their step budget on stalled
    # episodes instead of fresh restarts.
    episode_max_steps = steps_for_dist(distance_high)
    vec_env = SubprocVecBaseDroneEnv(_build_reward_fn, p, target_pairs, NUM_ENVS, max_steps=episode_max_steps)

    try:
        model = ActorCritic(vec_env.observation_space.shape[0], vec_env.action_space.shape[0],
                             vec_env.action_space.low, vec_env.action_space.high,
                             hidden=p["hidden"], num_hidden_layers=p["num_hidden_layers"],
                             dropout=p["dropout"], log_std_max=p["log_std_max"]).to(device)

        if os.path.exists(bc_checkpoint_path):
            # Optional warm start -- helps the imitation stage below converge
            # faster, but isn't required; remaps by Linear position, not raw
            # nn.Sequential index (see load_bc_checkpoint). Works equally well
            # for a prior curriculum stage's full model_<timesteps>.pt as it
            # does for a plain BC/DAgger checkpoint.
            load_bc_checkpoint(model, bc_checkpoint_path, map_location=device)
            print(f"[train] loaded BC/DAgger checkpoint: {bc_checkpoint_path}")
        else:
            print(f"[train] no BC checkpoint at {bc_checkpoint_path}, starting imitation from random init")

        # 25% imitation: on-policy rollout + BC retrain against the PID teacher
        # (see _run_imitation_stage) -- runs BEFORE critic warmup/PPO, using
        # this same vec_env and model.
        if IMITATION_DURATION_STEPS > 0:
            print(f"[train] imitation stage: {IMITATION_DURATION_STEPS} steps")
            model = _run_imitation_stage(vec_env, model, IMITATION_DURATION_STEPS)

        # Critic-only warmup: freeze everything the actor depends on (shared trunk
        # included -- it's shared with the critic, so leaving it trainable would
        # still shift the actor's output even with actor_mean/actor_log_std frozen)
        # so the warm-started policy can't move while the critic calibrates against
        # it. optimizer.step() skips any param with grad=None, which frozen params
        # always have, so this is enough -- no change needed in ppo_update itself.
        actor_frozen = WARMUP_DURATION_STEPS > 0
        if actor_frozen:
            for param in model.shared.parameters():
                param.requires_grad_(False)
            model.actor_mean.weight.requires_grad_(False)
            model.actor_mean.bias.requires_grad_(False)
            model.actor_log_std.requires_grad_(False)
            print(f"[train] actor frozen for the first {WARMUP_DURATION_STEPS} steps (critic warmup)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=WEIGHT_DECAY)

        buffer_size = CHUNK_TIMESTEPS * NUM_ENVS
        batch_size = max(1, buffer_size // p["num_minibatches"])

        # reward_func is identical regardless of training stage -- no per-chunk
        # reward_method switching.
        eval_env = BaseDroneEnv(reward_func, target_pairs=target_pairs, max_steps=episode_max_steps)

        start_run(experiment_name=MLFLOW_EXPERIMENT, run_name=f"base_training_{distance_low:g}_{distance_high:g}")
        mlflow.set_tag("run_start_time", run_start.isoformat(timespec="seconds"))
        log_params_safe({**p, "num_envs": NUM_ENVS, "total_timesteps": TOTAL_TIMESTEPS,
                          "distance_low": distance_low, "distance_high": distance_high,
                          "bc_checkpoint_path": bc_checkpoint_path, "checkpoint_dir": checkpoint_dir,
                          "seed": seed, "n_target_pairs": len(target_pairs),
                          "ent_coef_start": ENT_COEF_START, "ent_coef_end": ENT_COEF_END,
                          "n_diagnostic_episodes": N_DIAGNOSTIC_EPISODES,
                          "weight_decay": WEIGHT_DECAY, "lr_min_ratio": LR_MIN_RATIO,
                          "run_start_time": run_start.isoformat(timespec="seconds")})

        # timesteps_done starts at IMITATION_DURATION_STEPS (not 0) so checkpoint
        # filenames/mlflow steps/prints read as progress against the GRAND total
        # (imitation + warmup + training), even though the decay schedules below
        # only run over this PPO portion's own budget (ppo_timesteps_done).
        timesteps_done = IMITATION_DURATION_STEPS
        ppo_timesteps_done = 0
        n_chunks = PPO_TOTAL_TIMESTEPS // CHUNK_TIMESTEPS
        ckpt_path = None
        worker_history = {"timesteps": [], "start_dist": [], "live_dist": []}
        chunks_since_unfreeze = None  # None until the actor actually unfreezes
        for chunk in range(n_chunks):
            if actor_frozen and ppo_timesteps_done >= WARMUP_DURATION_STEPS:
                for param in model.parameters():
                    param.requires_grad_(True)
                actor_frozen = False
                chunks_since_unfreeze = 0
                print(f"[train] critic warmup done at {timesteps_done} steps -- actor unfrozen")
            elif chunks_since_unfreeze is not None:
                chunks_since_unfreeze += 1

            stage = "critic_warmup" if ppo_timesteps_done < WARMUP_DURATION_STEPS else "training"

            progress = ppo_timesteps_done / PPO_TOTAL_TIMESTEPS
            current_ent_coef = max(ENT_COEF_END, ENT_COEF_START * (1 - progress))
            current_lr = cosine_lr(p["lr"], progress, min_ratio=LR_MIN_RATIO)
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

            recent_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
            grad_norm = last_losses.get("grad_norm", 0.0)
            mlflow.log_metric("recent_reward", recent_reward, step=timesteps_done)
            mlflow.log_metric("ent_coef", current_ent_coef, step=timesteps_done)
            mlflow.log_metric("lr", current_lr, step=timesteps_done)
            mlflow.log_metric("grad_norm", grad_norm, step=timesteps_done)

            ckpt_path = os.path.join(checkpoint_dir, f"model_{timesteps_done}.pt")
            torch.save(model.state_dict(), ckpt_path)
            export_onnx_model(model, vec_env.observation_space.shape[0], ckpt_path[:-3] + ".onnx")

            outcomes = diagnose_with_model(model, eval_env, N_DIAGNOSTIC_EPISODES)
            # hover_success is a RewardConfig/make_reward_fn concept (sustained
            # in-zone dwelling) that reward_func -- what this script actually
            # trains against -- never sets, so it's permanently 0 here; counting
            # "Hit" (reward_func's actual terminal-success condition) too is what
            # makes this comparable to pid_success_rate below, which already
            # counts "Hit" and nothing else.
            success_rate = (outcomes["hover_success"] + outcomes.get("Hit", 0)) / N_DIAGNOSTIC_EPISODES
            grade, breakdown = compute_grade(
                success_rate=success_rate, avg_final_dist=outcomes["avg_final_dist"],
                avg_hit_time_sec=outcomes["avg_hit_time_sec"], avg_grad_norm=grad_norm,
                oob_radius=oob_radius, episode_time_budget_sec=eval_env.max_steps * eval_env.dt,
            )

            # Reference: what eval_env's own PID teacher achieves on the same
            # target_pairs -- lets plot_training_run show whether the RL curve
            # is converging toward this ceiling or diverging from it, at every
            # checkpoint rather than only a single end-of-run snapshot.
            pid_outcomes = diagnose_with_pid(eval_env.pid_teacher, eval_env, N_DIAGNOSTIC_EPISODES,
                                              gains_by_dist=gains_by_dist)
            pid_outcomes.pop("final_dists")
            pid_avg_final_dist = pid_outcomes.pop("avg_final_dist")
            pid_outcomes.pop("avg_hit_time_sec")
            pid_success_rate = pid_outcomes.pop("Hit", 0) / N_DIAGNOSTIC_EPISODES

            live_targets, start_dists, live_dists = vec_env.get_target_positions()
            target_pos_metrics = {
                "target_dist_mean": float(start_dists.mean()), "target_dist_min": float(start_dists.min()),
                "target_dist_max": float(start_dists.max()),
                "target_x_mean": float(live_targets[:, 0].mean()), "target_y_mean": float(live_targets[:, 1].mean()),
                "target_z_mean": float(live_targets[:, 2].mean()),
            }
            worker_history["timesteps"].append(timesteps_done)
            worker_history["start_dist"].append(start_dists)
            worker_history["live_dist"].append(live_dists)

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
                **breakdown,
                **{f"outcome_{k}": v for k, v in outcome_counts.items()},
                **{f"pid_outcome_{k}": v for k, v in pid_outcomes.items()},
                **target_pos_metrics,
            }
            log_metrics_safe({k: v for k, v in row.items() if k not in ("stage", "early_stopped")},
                              step=timesteps_done)
            mlflow.set_tag("stage", stage)
            _append_csv_row(row, metrics_csv)
            print(f"[train] {timesteps_done}/{TOTAL_TIMESTEPS}  stage={stage}  ent_coef={current_ent_coef:.4f}  "
                  f"lr={current_lr:.2e}  grad_norm={grad_norm:.3f}  grade={grade:.4f}  "
                  f"success_rate={success_rate:.2f}  outcomes={outcome_counts}  "
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
        plot_worker_metrics(worker_history, vec_env.counts, output_dir=plots_dir)
        if os.path.isdir(plots_dir):
            # log_artifacts (plural) walks the whole tree -- plot_worker_metrics
            # writes into worker_*/ subfolders, which a top-level os.listdir loop
            # would miss.
            mlflow.log_artifacts(plots_dir, artifact_path="plots")

        run_end = datetime.now()
        mlflow.set_tag("run_end_time", run_end.isoformat(timespec="seconds"))
        mlflow.end_run()
        print(f"[train] done. Checkpoints in {checkpoint_dir}, metrics in {metrics_csv}, plots in {plots_dir}")
        print(f"[train] run started at {run_start.isoformat(timespec='seconds')}, "
              f"ended at {run_end.isoformat(timespec='seconds')} "
              f"(duration {run_end - run_start})")
        return model
    finally:
        vec_env.close()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distance-low", type=float, default=DEFAULT_DISTANCE_LOW)
    parser.add_argument("--distance-high", type=float, default=DEFAULT_DISTANCE_HIGH)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--bc-checkpoint", dest="bc_checkpoint_path", default=DEFAULT_BC_CHECKPOINT_PATH,
                         help="Warm-start weights: the original BC/DAgger checkpoint for the first curriculum "
                              "stage, or a previous stage's model_<timesteps>.pt for later stages.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seeds model init, PPO action sampling/minibatch order, and target-pair "
                              "sampling. Does not make the multiprocess env workers deterministic -- see "
                              "train()'s docstring.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(distance_low=args.distance_low, distance_high=args.distance_high,
          checkpoint_dir=args.checkpoint_dir, bc_checkpoint_path=args.bc_checkpoint_path, seed=args.seed)
