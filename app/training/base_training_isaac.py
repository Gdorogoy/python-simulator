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

# Overrides the imported value -- the soft-reset trigger below reads grade off
# N_DIAGNOSTIC_EPISODES eval episodes, and 10 is noisy enough to false-trigger.
N_DIAGNOSTIC_EPISODES = 30

# GRAND TOTAL env-steps across ALL parallel envs, not per-env like base_training.py's TOTAL_TIMESTEPS.
TOTAL_TIMESTEPS_ISAAC = 128_000_000
# Lowered from 0.15 -- at 128M total_timesteps that was 19.2M steps (15%) of pure
# imitation before critic_warmup/PPO even start, bigger than it needs to be now that
# actor_log_std gets reset right after this stage anyway (see the reset above), so
# there's less riding on imitation alone to hand off a well-calibrated actor.
IMITATION_FRACTION = 0.08
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
# 0.25/0.5 fired 4 resets in runs/base_training_isaac_3_10_v1 and each one only recovered the grade to
# ~0.75-0.83 while blending the critic back toward the (worse) early one (value_loss +5..8 every time).
# 0.4 makes it a collapse-only safety net (trigger ~0.57 with best ~0.97; that run's worst trailing-5
# mean was 0.65, so it would have fired 0 times); 0.75 pulls back harder when it does fire.
RESET_DEGRADE_MARGIN = 0.4
RESET_BLEND_ALPHA = 0.75
# Critic-only chunks right after a reset fires, before resuming joint actor+critic
# updates -- the blend also yanks the critic back, so it needs the same kind of
# recalibration window the run-start warmup exists for.
RESET_WARMUP_CHUNKS = 2
# Rollout-only chunks at the very start of the post-imitation loop, before the
# critic-warmup stage or anything else -- no gradient updates at all (not even
# critic). Needs isaac_ppo_train's skip_update, not requires_grad freezing --
# freezing every param leaves nothing for backward() to hook into.
RUN_START_FROZEN_CHUNKS = 3
# Promotion gate: a stage counts as "solid" -- ready to seed the next distance range -- once SOLID_WINDOW
# consecutive chunks, ALL at loop-chunk index >= SOLID_MIN_CHUNK, average >= SOLID_GRADE with none below
# SOLID_WINDOW_MIN_GRADE (same avg/min shape as plotting._print_suitable_weights). The chunk index counts
# warmup (training starts at chunk MIN_WARMUP_CHUNKS = 16), so 40 means >= 24 PPO chunks of real training:
# it keeps a checkpoint that is just the imitation policy (~0.95 grade, first few post-unfreeze chunks)
# from being promoted as if PPO had produced it.
SOLID_GRADE = 0.8
SOLID_MIN_CHUNK = 40
SOLID_WINDOW = 5
SOLID_WINDOW_MIN_GRADE = 0.6


def training_progress_at(ppo_timesteps_done: int, warmup_timesteps_grand: int, ppo_total_timesteps_grand: int) -> float:
    """0 through critic warmup, then 0 -> 1 across the PPO stage alone. Both the entropy and LR schedules
    run on this: the actor is frozen during warmup, so any schedule that started counting at chunk 0
    was just burning itself down where it can't act (entropy hit its floor ~10 chunks into warmup and
    sat flat all training; LR had already decayed ~14% by unfreeze)."""
    return min(1.0, max(0.0, (ppo_timesteps_done - warmup_timesteps_grand) / max(1, ppo_total_timesteps_grand)))


def scheduled_lr(ppo_timesteps_done: int, warmup_timesteps_grand: int, post_imitation_timesteps_grand: int,
                 base_lr: float, min_ratio: float) -> float:
    """Flat at base_lr through critic warmup (nothing to decay for -- only the critic trains, and it was
    still improving at chunk 16), then the cosine over the FULL warmup+PPO span from that point on. That
    second part is deliberately the original profile, not one spanning the PPO stage alone: v2 (cosine over
    training only) ran ~45% higher lr mid-run than v1 and drifted faster late (chunks 12-40 mean grade
    0.648 vs 0.731)."""
    if ppo_timesteps_done < warmup_timesteps_grand:
        return base_lr
    return cosine_lr(base_lr, ppo_timesteps_done / post_imitation_timesteps_grand, min_ratio=min_ratio)


def entropy_coef_at(ppo_timesteps_done: int, warmup_timesteps_grand: int, ppo_total_timesteps_grand: int,
                    start: float, end: float) -> float:
    """Held at `start` through critic warmup, then linear start -> end across the PPO stage. Reaches `end`
    exactly at the end of the run for any end/start ratio (the old max(end, start*(1-progress)) only did
    when end << start)."""
    return start + (end - start) * training_progress_at(ppo_timesteps_done, warmup_timesteps_grand,
                                                         ppo_total_timesteps_grand)


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


class PidResidualEnv:
    """Gym-env wrapper: the policy's action becomes a small CORRECTION on top of the per-distance PID.
        env_action = pid_action + residual_scale * policy_action
    policy_action is the usual tanh-scaled action in [action_low, action_high], centered on 0 (the bounds
    are symmetric), so a policy that outputs ~0 IS the PID -- PPO starts at PID performance by construction
    and can only add corrections of at most residual_scale x the action range, which also bounds how far
    drift can take it and shrinks the exploration noise's effect by the same factor. The env clamps the
    final action itself (_pre_physics_step). PID state is handled exactly as the imitation stage does:
    gains re-picked by start distance and integrators cleared for every env that just reset. Everything
    else (unwrapped, spaces, close, ...) is delegated to the wrapped env."""

    def __init__(self, env, gains_by_dist: dict, residual_scale: float):
        self.env = env
        self._gains_by_dist = gains_by_dist
        self._residual_scale = residual_scale
        unwrapped = env.unwrapped
        self._pid = TorchPIDController(unwrapped.num_envs, unwrapped.device, **next(iter(gains_by_dist.values())))
        self._all_env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
        self._reinit_pid(self._all_env_ids)

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def _reinit_pid(self, env_ids):
        _select_pid_teacher_isaac(self.env, self._pid, self._gains_by_dist, env_ids)
        self._pid.reset(env_ids)

    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        self._reinit_pid(self._all_env_ids)
        return out

    def step(self, action):
        with torch.no_grad():
            pos, vel, ang_vel, roll, pitch, yaw, target_local = _read_kinematics(self.env)
            pid_action = self._pid.compute_action(pos, vel, ang_vel, roll, pitch, yaw, target_local,
                                                   self.env.unwrapped._desired_yaw_w, dt=1 / 240)
        obs, reward, terminated, truncated, extras = self.env.step(pid_action + self._residual_scale * action)
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self._reinit_pid(done_ids)
        return obs, reward, terminated, truncated, extras


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


METRICS_JSON_PATH = "runs/isaac_training_metrics.json"


def _append_json_row(row: dict, json_path: str, run_key: str):
    """Local mlflow-style log, one shared file across every run: {"runs":
    {run_key: {metric_name: [value_per_chunk, ...], ...}}}. Rewrites the whole
    file each call (not a real append) -- simplest way to stay valid JSON
    without a streaming writer; fine at this scale (one number per metric per
    chunk, a few hundred chunks per run)."""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {"runs": {}}
    run_data = data["runs"].setdefault(run_key, {})
    for k, v in row.items():
        run_data.setdefault(k, []).append(v)
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, json_path)  # atomic -- a crash mid-write can't corrupt the real file


def train(env, distance_low=DEFAULT_DISTANCE_LOW, distance_high=DEFAULT_DISTANCE_HIGH,
          checkpoint_dir=DEFAULT_CHECKPOINT_DIR_ISAAC, bc_checkpoint_path=DEFAULT_BC_CHECKPOINT_PATH,
          total_timesteps=TOTAL_TIMESTEPS_ISAAC, seed=None, hparams_override: dict | None = None,
          num_steps_per_chunk: int = NUM_STEPS_PER_CHUNK,
          solid_grade: float = SOLID_GRADE, solid_min_chunk: int = SOLID_MIN_CHUNK,
          solid_window: int = SOLID_WINDOW, stop_when_solid: bool = True,
          residual_scale: float = 0.0, detach_critic: bool = False, residual_resume: bool = False):
    """Runs one fixed-range training stage against a live Isaac env (build it yourself via
    gym.make(...) and pass it in). hparams_override optionally overrides any PARAMS key
    or the module-level entropy/weight-decay/lr-floor constants; unrecognized keys are ignored.

    residual_scale > 0: PID-residual mode. The policy learns a correction on top of the per-distance PID
    (PidResidualEnv: env_action = pid + residual_scale * policy_action; eval applies the same composition).
    The BC checkpoint's trunk is still loaded, but actor_mean is zeroed so the correction starts at exactly 0
    (= pure PID), and the imitation stage is skipped (it would just re-fit actor_mean to the PID's actions,
    which as a correction would double the PID). 0 = the original direct-policy training.
    residual_resume: the loaded checkpoint is itself a residual-mode one (e.g. the previous distance stage's
    SOLID.json checkpoint), so keep its actor head -- the learned correction -- instead of zeroing it.
    Zeroing is only right when the checkpoint is a direct-policy BC/DAgger one (its head outputs full PID
    actions, which as a correction would double the PID).
    detach_critic: value-head gradients don't reach the shared trunk (ActorCritic.detach_critic).

    Promotion gate (solid_*): once solid_window consecutive chunks, all at loop-chunk index >=
    solid_min_chunk, average >= solid_grade (none below SOLID_WINDOW_MIN_GRADE), writes
    <checkpoint_dir>/SOLID.json naming the checkpoint to seed the next range from and, if
    stop_when_solid, ends the stage there. If it never fires, prints the best qualifying window.

    num_steps_per_chunk: per-env rollout length per PPO chunk -- chunk_grand_steps =
    num_steps_per_chunk * env.unwrapped.num_envs, so n_chunks (and every fixed-chunk-count
    floor: MIN_WARMUP_CHUNKS, RESET_SUSTAIN_CHUNKS, etc.) shrinks as num_envs grows unless
    this is scaled down to compensate. Defaults to the module constant for backward compat."""
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
                         dropout=p["dropout"], log_std_max=p["log_std_max"],
                         detach_critic=detach_critic).to(device)

    if os.path.exists(bc_checkpoint_path):
        load_bc_checkpoint(model, bc_checkpoint_path, map_location=device)
        print(f"[train_isaac] loaded BC/DAgger checkpoint: {bc_checkpoint_path}")
    else:
        print(f"[train_isaac] no BC checkpoint at {bc_checkpoint_path}, starting imitation from random init")

    if residual_scale > 0:
        if not residual_resume:
            with torch.no_grad():
                model.actor_mean.weight.zero_()
                model.actor_mean.bias.zero_()
        env = PidResidualEnv(env, gains_by_dist, residual_scale)
        head_note = ("actor head KEPT from the loaded residual checkpoint (--residual-resume)" if residual_resume
                     else "actor_mean zeroed (correction starts at 0)")
        print(f"[train_isaac] PID-residual mode: env_action = pid + {residual_scale} * policy_action, "
              f"{head_note}, imitation stage skipped")
    if detach_critic:
        print("[train_isaac] detach_critic: value loss no longer trains the shared trunk")

    chunk_grand_steps = num_steps_per_chunk * num_envs

    imitation_timesteps_grand = 0 if residual_scale > 0 else int(IMITATION_FRACTION * total_timesteps)
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

    # Both load_bc_checkpoint AND _run_imitation_stage_isaac (via pretrain_behavior_cloning's
    # Gaussian NLL, same call used by the offline BC/DAgger scripts) score actor_log_std
    # against a deterministic PID teacher -- zero true variance in the target, so NLL is
    # minimized by collapsing std toward the log_std_min floor. Resetting only after
    # load_bc_checkpoint (tried first) doesn't survive this on-policy imitation stage
    # re-collapsing it right back down -- verified in runs/base_training_isaac_3_10_fixed
    # (effective_std_mean still pinned ~exp(-3.0) the WHOLE run despite that earlier reset).
    # Reset here instead, as the last thing before the critic-warmup freeze locks it in:
    # tanh(0)=0 sits at the midpoint of [log_std_min, log_std_max], in tanh's live gradient
    # region, so PPO actually gets to choose an exploration level instead of inheriting a
    # frozen imitation-stage artifact. Keeps the actor_mean/critic weights untouched.
    with torch.no_grad():
        model.actor_log_std.zero_()
    print("[train_isaac] actor_log_std reset to 0 post-imitation, pre-critic-warmup-freeze")

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

    run_name = f"base_training_isaac_{distance_low:g}_{distance_high:g}"
    run_key = f"{run_name}_{run_start.strftime('%Y%m%d_%H%M%S')}"  # unique even if the same range is rerun
    start_run(experiment_name="base-training-isaac", run_name=run_name)
    mlflow.set_tag("run_start_time", run_start.isoformat(timespec="seconds"))
    log_params_safe({**p, "num_envs": num_envs, "num_steps_per_chunk": num_steps_per_chunk,
                      "residual_scale": residual_scale, "detach_critic": detach_critic,
                      "total_timesteps": total_timesteps,
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
    post_reset_frozen_chunks_left = 0
    pending_post_reset_unfreeze = False
    chunk_grades = {}  # loop-chunk index -> (grade, checkpoint path), for the promotion gate
    solid_info = None
    if n_chunks < solid_min_chunk + solid_window:
        print(f"[train_isaac] WARNING: n_chunks={n_chunks} < solid_min_chunk({solid_min_chunk}) + "
              f"solid_window({solid_window}) -- the promotion gate can never fire this run")
    for chunk in range(n_chunks):
        run_start_frozen = chunk < RUN_START_FROZEN_CHUNKS
        if run_start_frozen and chunk == 0:
            print(f"[train_isaac] chunks 0-{RUN_START_FROZEN_CHUNKS - 1}: rollout only, "
                  f"no gradient updates (run-start settle)")
        if post_reset_frozen_chunks_left > 0:
            post_reset_frozen_chunks_left -= 1
            if post_reset_frozen_chunks_left == 0:
                pending_post_reset_unfreeze = True
        elif pending_post_reset_unfreeze:
            for param in model.parameters():
                param.requires_grad_(True)
            chunks_since_unfreeze = 0
            pending_post_reset_unfreeze = False
            print(f"[train_isaac] post-reset critic warmup done at {timesteps_done} steps -- actor unfrozen")
        elif actor_frozen and ppo_timesteps_done >= warmup_timesteps_grand:
            for param in model.parameters():
                param.requires_grad_(True)
            actor_frozen = False
            chunks_since_unfreeze = 0
            print(f"[train_isaac] critic warmup done at {timesteps_done} steps -- actor unfrozen")
        elif chunks_since_unfreeze is not None:
            chunks_since_unfreeze += 1

        stage = "critic_warmup" if ppo_timesteps_done < warmup_timesteps_grand else "training"

        current_ent_coef = entropy_coef_at(ppo_timesteps_done, warmup_timesteps_grand,
                                            ppo_total_timesteps_grand, ent_coef_start, ent_coef_end)
        current_lr = scheduled_lr(ppo_timesteps_done, warmup_timesteps_grand, post_imitation_timesteps_grand,
                                   p["lr"], lr_min_ratio)
        if chunks_since_unfreeze is not None and chunks_since_unfreeze < UNFREEZE_LR_RAMP_CHUNKS:
            current_lr *= (chunks_since_unfreeze + 1) / UNFREEZE_LR_RAMP_CHUNKS
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model, optimizer, episode_rewards, last_losses, ppo_obs = isaac_ppo_train(
            env, total_timesteps=chunk_grand_steps, num_steps=num_steps_per_chunk,
            gamma=p["gamma"], lam=p["lam"], lr=current_lr, model=model, optimizer=optimizer,
            ent_coef=current_ent_coef, target_kl=p["target_kl"], num_epochs=p["num_epochs"],
            batch_size=max(1, chunk_grand_steps // p["num_minibatches"]),
            clip_eps=p["clip_eps"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
            initial_obs=ppo_obs, skip_update=run_start_frozen,
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

        outcomes = diagnose_with_model(model, eval_env, N_DIAGNOSTIC_EPISODES,
                                       residual_scale=residual_scale, gains_by_dist=gains_by_dist)
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

                    # Blend also moved the critic -- give it RESET_WARMUP_CHUNKS
                    # critic-only chunks before the actor trains again, same
                    # reasoning as the run-start warmup.
                    for param in model.shared.parameters():
                        param.requires_grad_(False)
                    model.actor_mean.weight.requires_grad_(False)
                    model.actor_mean.bias.requires_grad_(False)
                    model.actor_log_std.requires_grad_(False)
                    post_reset_frozen_chunks_left = RESET_WARMUP_CHUNKS
                    print(f"[train_isaac] freezing actor for {RESET_WARMUP_CHUNKS} chunks (post-reset critic warmup)")

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
        _append_json_row(row, METRICS_JSON_PATH, run_key)
        print(f"[train_isaac] {timesteps_done}/{total_timesteps}  stage={stage}  ent_coef={current_ent_coef:.4f}  "
              f"lr={current_lr:.2e}  grad_norm={grad_norm:.3f}  grade={grade:.4f}  best_grade={best_grade:.4f}  "
              f"reset={reset_triggered}  success_rate={success_rate:.2f}  outcomes={outcome_counts}  "
              f"pid_success_rate={pid_success_rate:.2f}  pid_avg_final_dist={pid_avg_final_dist:.2f}")

        chunk_grades[chunk] = (grade, ckpt_path)
        window_start = chunk - solid_window + 1
        if solid_info is None and stage == "training" and window_start >= solid_min_chunk:
            window = [(c, *chunk_grades[c]) for c in range(window_start, chunk + 1)]
            w_grades = [g for _, g, _ in window]
            if float(np.mean(w_grades)) >= solid_grade and min(w_grades) >= SOLID_WINDOW_MIN_GRADE:
                pick_chunk, pick_grade, pick_ckpt = max(window, key=lambda w: w[1])
                solid_info = {
                    "solid_at_chunk": chunk, "window_chunks": [window_start, chunk],
                    "window_grades": [round(g, 4) for g in w_grades],
                    "window_mean": round(float(np.mean(w_grades)), 4), "window_min": round(min(w_grades), 4),
                    "checkpoint": pick_ckpt, "checkpoint_chunk": pick_chunk, "checkpoint_grade": round(pick_grade, 4),
                    "distance_low": distance_low, "distance_high": distance_high,
                }
                with open(os.path.join(checkpoint_dir, "SOLID.json"), "w") as f:
                    json.dump(solid_info, f, indent=2)
                mlflow.set_tag("solid_at_chunk", str(chunk))
                print(f"[train_isaac] SOLID at chunk {chunk}: chunks {window_start}-{chunk} average "
                      f"{solid_info['window_mean']:.3f} (min {solid_info['window_min']:.3f}) >= {solid_grade} -- "
                      f"promote {pick_ckpt} (chunk {pick_chunk}, grade {pick_grade:.3f}) via "
                      f"--bc-checkpoint; details in {checkpoint_dir}/SOLID.json")
                if stop_when_solid:
                    break

    if solid_info is None:
        eligible = [(float(np.mean([chunk_grades[c][0] for c in range(s, s + solid_window)])), s)
                    for s in range(solid_min_chunk, max(chunk_grades, default=-1) - solid_window + 2)
                    if s in chunk_grades]
        if eligible:
            best_mean, best_start = max(eligible)
            print(f"[train_isaac] NOT solid: best {solid_window}-chunk window at chunk >= {solid_min_chunk} "
                  f"was chunks {best_start}-{best_start + solid_window - 1}, average {best_mean:.3f} "
                  f"(need >= {solid_grade}, none below {SOLID_WINDOW_MIN_GRADE})")
        else:
            print(f"[train_isaac] NOT solid: run never had {solid_window} chunks at index >= {solid_min_chunk}")

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
