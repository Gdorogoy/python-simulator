"""Global Optuna search over network architecture, PPO hyperparameters, and
RewardFnPhase1 reward shaping (the same phase-1 approach/interception reward
app.training.phase_1_training.py trains against -- not the old phase-0
hover-in-place task). Each trial trains a fresh model for a short budget in a
few chunks (reporting an intermediate metric each chunk so bad trials get
pruned early), then scores the final policy with a composite "grade": success
rate, rewarded, minus normalized penalties for final-distance error,
time-to-hit-target, and policy gradient norm (see app.guidance.utils.compute_grade).

A single env's per-step tensors are tiny, so a GPU only does meaningful work once
many envs are stepped in lockstep and batched into one forward/backward pass --
see VecInterceptorDroneEnv and train.vec_ppo_train. --num-envs controls how many
run in parallel within each trial; --device/--n-jobs control how trials themselves
are distributed. On one GPU, running many trials as separate OS processes (the old
CPU strategy) means N processes fighting over one device's compute, which usually
loses to running fewer trials at a time but giving each a bigger --num-envs. Each
worker is still launched with CUDA_VISIBLE_DEVICES fixed BEFORE its interpreter
starts (see _spawn_worker) -- patching a `device` variable after import doesn't
work, since some modules import it by value at their own module-load time.

Every trial also logs its params, per-chunk reward, and final grade to MLflow
(default local sqlite:///mlflow.db store) -- run `mlflow ui` to browse/compare trials.

Usage:
    python -m app.guidance.optuna_search --n-trials 60 --device cuda --num-envs 32
Resumes automatically if run again with the same --study-name/--storage.
"""
import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime

import mlflow
import numpy as np
import optuna

from app.guidance.mlflow_utils import start_run, log_params_safe, log_metrics_safe

DEFAULT_STUDY_NAME = "interceptor_drone_phase1_search"
DEFAULT_STORAGE = "sqlite:///runs/optuna/study.db"
PRETRAINED_BC_DIR = "app/control"
RESULTS_DIR = "runs/optuna"

# Same distance ladder as phase_1_training.py's PHASE1_DISTANCES/PHASE1_OOB_RADIUS,
# pooled into one target_pairs set (env.reset() samples one pair uniformly at random
# each episode) instead of replicating that file's curriculum-stage gating/per-distance
# PID-gain swapping -- overkill for a short hyperparameter-search budget, but every
# distance still gets exercised across a trial's many episodes.
PHASE1_DISTANCES = (3, 10, 50, 150)
OOB_RADIUS = 200.0


def suggest_params(trial: optuna.Trial) -> dict:
    """All tunable hyperparameters in one place: network architecture, PPO, and
    reward shaping. ~25 dimensions -- give the search enough trials to cover it."""
    return dict(
        # Network architecture
        hidden=trial.suggest_categorical("hidden", [32, 64, 96, 128]),
        num_hidden_layers=trial.suggest_int("num_hidden_layers", 2, 6),
        dropout=trial.suggest_float("dropout", 0.0, 0.55),
        # PPO
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
        lam=trial.suggest_float("lam", 0.85, 0.98),
        ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        clip_eps=trial.suggest_float("clip_eps", 0.1, 0.3),
        vf_coef=trial.suggest_float("vf_coef", 0.25, 1.0),
        target_kl=trial.suggest_float("target_kl", 0.01, 0.08, log=True),
        num_epochs=trial.suggest_int("num_epochs", 4, 15),
        # Minibatch COUNT, not size: batch_size = (num_steps*num_envs) // num_minibatches,
        # computed in the objective where those are known. A fixed batch_size doesn't scale
        # with --num-steps/--num-envs and can silently turn into tens of thousands of
        # Python-loop minibatches per epoch (measured: 61s/epoch at batch_size=16 on a
        # 262k-sample buffer, vs 0.3s/epoch at batch_size=4096).
        num_minibatches=trial.suggest_categorical("num_minibatches", [8, 16, 32, 64]),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.1, 1.0),
        log_std_clamp_max=trial.suggest_float("log_std_clamp_max", -1.0, -0.2),
        # Reward shaping (RewardFnPhase1)
        streak_penalty_coef=trial.suggest_float("streak_penalty_coef", -0.1, -0.01),
        streak_cap=trial.suggest_int("streak_cap", 15, 40),
        phase1_pos_coef=trial.suggest_float("phase1_pos_coef", 0.2, 1.0),
        hit_reward=trial.suggest_float("hit_reward", 5.0, 20.0),
        velocity_gain=trial.suggest_float("velocity_gain", 0.0, 0.3),
        tilt_penalty_coef=trial.suggest_float("tilt_penalty_coef", 0.02, 0.3),
        ang_vel_penalty_coef=trial.suggest_float("ang_vel_penalty_coef", 0.02, 0.3),
        imitation_coef=trial.suggest_float("imitation_coef", 0.0, 0.5),
        zone_bonus=trial.suggest_float("zone_bonus", 0.1, 0.4),
        dist_penalty_coef=trial.suggest_float("dist_penalty_coef", 0.8, 1.6),
        approach_gain=trial.suggest_float("approach_gain", 1.0, 1.6),
        vel_penalty_coef=trial.suggest_float("vel_penalty_coef", 0.05, 0.2),
        step_penalty=trial.suggest_float("step_penalty", 0.0, 0.03),
        closer_bonus_val=trial.suggest_float("closer_bonus_val", 0.0, 0.03),
    )


def _build_reward_cfg(p: dict):
    from app.reward_functions.reward_fn_phase1 import RewardFnPhase1

    return RewardFnPhase1(
        hit_steps_streak=1500,  # required arg, but unused within the class -- fixed placeholder,
                                 # matching every other caller in the repo (all hardcode 1500)
        phase1_pos_coef=p["phase1_pos_coef"], hit_reward=p["hit_reward"],
        warmup_duration_steps=0,  # skip straight to imitation
        imitation_duration_steps=20_000, phase1_duration_steps=20_000,
        velocity_gain=p["velocity_gain"],
        oob_radius=OOB_RADIUS, attitude_penalty=-1.0, oob_penalty=-1.5, hover_success_steps=200,
        outer_dist=1.0, inner_dist=0.3,  # matches phase_1_training.py's larger approach zone
        streak_penalty_coef=p["streak_penalty_coef"], streak_cap=p["streak_cap"],
        tilt_penalty_coef=p["tilt_penalty_coef"], ang_vel_penalty_coef=p["ang_vel_penalty_coef"],
        imitation_coef=p["imitation_coef"], zone_bonus=p["zone_bonus"],
        dist_penalty_coef=p["dist_penalty_coef"], approach_gain=p["approach_gain"],
        vel_penalty_coef=p["vel_penalty_coef"], step_penalty=p["step_penalty"],
        closer_bonus_val=p["closer_bonus_val"],
    )


def _build_target_pairs():
    from app.training.eval_matrix import build_eval_pairs

    # axes=(0,1): xy only, matching every existing BC/DAgger demo's coverage (same
    # choice phase_1_training.py's stage 0 uses). One static list, safe to share --
    # each env's own np_random samples an index independently on its own reset().
    return build_eval_pairs(oob_radius=OOB_RADIUS, distances=PHASE1_DISTANCES, axes=(0, 1))


def build_env(p: dict):
    """One InterceptorDroneEnv, used for the final (non-vectorized) diagnostic eval."""
    from app.environmental.interceptor_drone import InterceptorDroneEnv

    return InterceptorDroneEnv(_build_reward_cfg(p).as_roadmap(), target_pairs=_build_target_pairs())


def build_vec_env(p: dict, num_envs: int):
    """num_envs independent InterceptorDroneEnv instances for vectorized rollout
    collection. Each gets its own .as_roadmap() call (a fresh curriculum-phase
    closure), even though they share one RewardFnPhase1 -- reward_cfg itself holds
    no mutable per-step state, only the per-env attributes on `env` do."""
    from app.environmental.interceptor_drone import InterceptorDroneEnv
    from app.environmental.vec_interceptor_drone import VecInterceptorDroneEnv

    reward_cfg = _build_reward_cfg(p)
    target_pairs = _build_target_pairs()
    envs = [
        InterceptorDroneEnv(reward_cfg.as_roadmap(), target_pairs=target_pairs)
        for _ in range(num_envs)
    ]
    return VecInterceptorDroneEnv(envs)


def make_objective(search_timesteps: int, num_steps: int, eval_episodes: int,
                    n_chunks: int, pretrained_bc_dir: str, mlflow_experiment: str, num_envs: int):
    """Builds the objective(trial) fn. Only imported/called inside a worker process,
    after CUDA_VISIBLE_DEVICES has already been fixed for that process (see
    _spawn_worker) -- every module's `device` global is then correct by construction."""
    import torch
    from app.control.pretrain_bc_multi_arch import checkpoint_path
    from app.guidance.train import ActorCritic, vec_ppo_train, device
    from app.guidance.utils import compute_grade
    from app.training.phase_0_training import diagnose_with_model

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial)
        start_run(experiment_name=mlflow_experiment, run_name=f"trial-{trial.number}")
        log_params_safe({**p, "num_envs": num_envs})
        try:
            vec_env = build_vec_env(p, num_envs)

            model = ActorCritic(vec_env.observation_space.shape[0], vec_env.action_space.shape[0],
                                 vec_env.action_space.low, vec_env.action_space.high,
                                 hidden=p["hidden"], num_hidden_layers=p["num_hidden_layers"],
                                 dropout=p["dropout"]).to(device)

            # One BC checkpoint per (hidden, num_hidden_layers) combo -- see
            # pretrain_bc_multi_arch.py. Path is architecture-derived so it's always the
            # exact shape match; no more silent skip-on-mismatch.
            bc_path = checkpoint_path(p["hidden"], p["num_hidden_layers"], pretrained_bc_dir)
            if os.path.exists(bc_path):
                try:
                    model.load_state_dict(torch.load(bc_path, map_location=device))
                except RuntimeError:
                    mlflow.set_tag("bc_load_failed", "true")
            else:
                mlflow.set_tag("bc_checkpoint_missing", "true")

            optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])

            # Derived from the actual buffer size, not a fixed constant -- see
            # suggest_params' num_minibatches comment for why.
            buffer_size = num_steps * num_envs
            batch_size = max(1, buffer_size // p["num_minibatches"])
            mlflow.log_param("batch_size", batch_size)

            chunk_timesteps = max(buffer_size, search_timesteps // n_chunks)
            last_losses = {}
            for chunk in range(n_chunks):
                model, optimizer, episode_rewards, last_losses = vec_ppo_train(
                    vec_env, total_timesteps=chunk_timesteps, num_steps=num_steps,
                    gamma=p["gamma"], lam=p["lam"], lr=p["lr"], model=model, optimizer=optimizer,
                    ent_coef=p["ent_coef"], target_kl=p["target_kl"], num_epochs=p["num_epochs"],
                    batch_size=batch_size, clip_eps=p["clip_eps"], vf_coef=p["vf_coef"],
                    max_grad_norm=p["max_grad_norm"], log_std_clamp_max=p["log_std_clamp_max"],
                )
                # Cheap pruning signal: no extra rollouts, just the training episodes just collected.
                recent_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
                mlflow.log_metric("recent_reward", recent_reward, step=chunk)
                trial.report(recent_reward, chunk)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    raise optuna.TrialPruned()

            eval_env = build_env(p)
            outcomes = diagnose_with_model(model, eval_env, eval_episodes)
            success_rate = outcomes["hover_success"] / eval_episodes
            grade, breakdown = compute_grade(
                success_rate=success_rate, avg_final_dist=outcomes["avg_final_dist"],
                avg_hit_time_sec=outcomes["avg_hit_time_sec"], avg_grad_norm=last_losses.get("grad_norm", 0.0),
                oob_radius=OOB_RADIUS, episode_time_budget_sec=eval_env.max_steps * eval_env.dt,
            )

            trial.set_user_attr("outcomes", outcomes)
            trial.set_user_attr("grade_breakdown", breakdown)
            log_metrics_safe({**breakdown, **{f"outcome_{k}": v for k, v in outcomes.items()}})
            return grade
        except optuna.TrialPruned:
            raise
        except Exception:
            mlflow.set_tag("failed", "true")
            raise
        finally:
            mlflow.end_run()

    return objective


def _worker_main(args):
    """Runs inside a worker process (own interpreter, own CUDA_VISIBLE_DEVICES) --
    optimizes n_trials against the shared study, then exits."""
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    objective = make_objective(args.search_timesteps, args.num_steps, args.eval_episodes,
                                args.n_chunks, args.pretrained_bc_dir, args.mlflow_experiment, args.num_envs)
    study.optimize(objective, n_trials=args.worker_trials)


def _spawn_worker(worker_idx: int, n_trials: int, device: str, args) -> subprocess.Popen:
    """Launches one worker as a fresh OS process with CUDA_VISIBLE_DEVICES fixed
    BEFORE its interpreter starts, so torch.cuda.is_available() -- and therefore
    every module's `device` global -- resolves consistently for its whole lifetime.
    Also gives this worker its own log subfolder (OPTUNA_WORKER_LOG_DIR), since
    phase_0_training.py's import-time log setup uses a second-resolution timestamp
    that collides if two workers import it within the same second."""
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["OPTUNA_WORKER_LOG_DIR"] = os.path.join(RESULTS_DIR, args.study_name, f"worker_{worker_idx}", "logs")
    cmd = [
        sys.executable, "-m", "app.guidance.optuna_search",
        "--_worker-trials", str(n_trials),
        "--search-timesteps", str(args.search_timesteps),
        "--num-steps", str(args.num_steps),
        "--n-chunks", str(args.n_chunks),
        "--eval-episodes", str(args.eval_episodes),
        "--study-name", args.study_name,
        "--storage", args.storage,
        "--pretrained-bc-dir", args.pretrained_bc_dir,
        "--mlflow-experiment", args.mlflow_experiment,
        "--num-envs", str(args.num_envs),
    ]
    return subprocess.Popen(cmd, env=env)


def _export_trials_csv(study: optuna.Study, out_path: str):
    """Writes one row per trial (params + value + grade breakdown), no pandas required."""
    rows = []
    param_keys, breakdown_keys = set(), set()
    for t in study.trials:
        param_keys.update(t.params.keys())
        breakdown_keys.update(t.user_attrs.get("grade_breakdown", {}).keys())

    fieldnames = ["number", "state", "value", *sorted(param_keys), *sorted(breakdown_keys)]
    for t in study.trials:
        row = {"number": t.number, "state": t.state.name, "value": t.value}
        row.update(t.params)
        row.update(t.user_attrs.get("grade_breakdown", {}))
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_search(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create the MLflow experiment once, here, before any worker process touches
    # it -- avoids a first-use race between concurrently-spawned workers.
    mlflow.set_experiment(args.mlflow_experiment)

    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage, direction="maximize",
        sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        load_if_exists=True,
    )
    print(f"Study '{args.study_name}' has {len(study.trials)} existing trial(s); running {args.n_trials} more "
          f"across {args.n_jobs} worker process(es) on device={args.device}, {args.num_envs} envs/trial. "
          f"MLflow experiment: '{args.mlflow_experiment}' (run `mlflow ui` to browse).")

    counts = [args.n_trials // args.n_jobs] * args.n_jobs
    for i in range(args.n_trials % args.n_jobs):
        counts[i] += 1

    procs = [_spawn_worker(idx, count, args.device, args) for idx, count in enumerate(counts) if count > 0]
    for proc in procs:
        proc.wait()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    print("\n=== Optuna search done ===")
    print(f"Best grade: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Best trial outcomes: {study.best_trial.user_attrs.get('outcomes')}")
    print(f"Best trial grade breakdown: {study.best_trial.user_attrs.get('grade_breakdown')}")

    out_path = os.path.join(RESULTS_DIR, f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    _export_trials_csv(study, out_path)
    print(f"Saved trial results to {out_path}")

    return study


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=60,
                         help="Total trials to run this invocation. The study is resumable, "
                              "so re-run with more any time.")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Parallel worker processes/trials at once. With --device cuda, "
                              "each trial already uses --num-envs environments batched through "
                              "the GPU, so keep this low (1-2) -- N processes each running their "
                              "own batch would just contend for the same GPU. With --device cpu, "
                              "this is the old strategy: raise it toward your core count instead.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                         help="Device to train on. cuda (default) batches --num-envs environments' "
                              "observations through one forward/backward pass per step -- that's "
                              "what actually gives a GPU work; a single env's tensors are too small "
                              "on their own. Falls back to CPU automatically if no GPU is visible.")
    parser.add_argument("--num-envs", type=int, default=32,
                         help="Environments stepped in lockstep per trial (default 32). Raise this "
                              "on cuda to feed the GPU bigger batches; on cpu it still helps some "
                              "(fewer, bigger tensor ops) but less dramatically.")
    parser.add_argument("--search-timesteps", type=int, default=300_000)
    parser.add_argument("--num-steps", type=int, default=8192,
                         help="Rollout length PER ENV before each PPO update (total timesteps "
                              "collected per update = num_steps * num_envs).")
    parser.add_argument("--n-chunks", type=int, default=4, help="Pruning checkpoints per trial.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--pretrained-bc-dir", default=PRETRAINED_BC_DIR,
                         help="Directory holding one pretrained_bc_dagger_h{hidden}_l{layers}.pt "
                              "per architecture combo (see pretrain_bc_multi_arch.py).")
    parser.add_argument("--mlflow-experiment", default=None,
                         help="MLflow experiment name (default: 'optuna-<study-name>'). "
                              "Each trial logs its params, per-chunk reward, and final grade "
                              "as one MLflow run; view with `mlflow ui`.")
    parser.add_argument("--_worker-trials", type=int, default=None,
                         help=argparse.SUPPRESS)  # internal: marks this invocation as a worker
    args = parser.parse_args()
    if args.mlflow_experiment is None:
        args.mlflow_experiment = f"optuna-{args.study_name}"
    return args


if __name__ == "__main__":
    args = parse_args()
    if args._worker_trials is not None:
        args.worker_trials = args._worker_trials
        _worker_main(args)
    else:
        run_search(args)
