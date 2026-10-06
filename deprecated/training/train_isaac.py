"""Real training launcher for app.training.base_training_isaac.train (imitation -> critic-warmup -> PPO).
Every budget/path knob is a CLI flag -- this is the one to use for a real run, not a smoke test.

    E:\\Isaac\\env_isaaclab\\Scripts\\python.exe -u app\\training\\train_isaac.py --headless ^
        --num_envs 4096 --total_timesteps 128000000 ^
        --distance-low 3 --distance-high 10 ^
        --checkpoint-dir runs/base_training_isaac_3_10 ^
        --bc-checkpoint app/control/pretrained_bc_dagger.pt

Scaling --num_envs up? Scale --num-steps-per-chunk DOWN by the same ratio to
keep chunk count (and n_chunks-based floors like MIN_WARMUP_CHUNKS) from
collapsing -- e.g. --num_envs 16384 --num-steps-per-chunk 128 keeps
chunk_grand_steps == the --num_envs 8192 --num-steps-per-chunk 256 baseline.

Long-running -- use run_in_background or its own terminal, and watch <checkpoint-dir>/metrics.csv
and plots/*.png as it goes. mlflow run lives in mlflow_isaac.db, experiment "base-training-isaac".

Curriculum staging: call again with a new --checkpoint-dir and
--bc-checkpoint <previous stage's dir>/model_<last timesteps>.pt.
Promotion gate: when the stage holds a >= --solid-grade average over --solid-window consecutive chunks, all at
chunk index >= --solid-min-chunk, it stops and writes <checkpoint-dir>/SOLID.json naming the checkpoint to pass
as the next stage's --bc-checkpoint (see SOLID_* in base_training_isaac.py). No SOLID.json = not promoted.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--total_timesteps", type=int, default=128_000_000)
parser.add_argument("--distance-low", type=float, default=3.0)
parser.add_argument("--distance-high", type=float, default=10.0)
parser.add_argument("--checkpoint-dir", default="runs/base_training_isaac")
parser.add_argument("--bc-checkpoint", dest="bc_checkpoint_path", default="app/control/pretrained_bc_dagger.pt")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--hparams-json", dest="hparams_json", default="deprecated/training/best_hparams_isaac.json")
parser.add_argument("--num-steps-per-chunk", dest="num_steps_per_chunk", type=int, default=256,
                     help="Per-env rollout length per PPO chunk. chunk_grand_steps = this * num_envs, "
                          "so scale this DOWN as --num_envs goes up to keep n_chunks (and the fixed-chunk-"
                          "count floors like MIN_WARMUP_CHUNKS) from shrinking away.")
parser.add_argument("--solid-grade", dest="solid_grade", type=float, default=0.8,
                     help="Promotion gate: grade every chunk in the window must average at least.")
parser.add_argument("--solid-min-chunk", dest="solid_min_chunk", type=int, default=40,
                     help="Promotion gate: loop-chunk index (counts warmup; training starts at chunk 16) "
                          "the whole window must be at or after.")
parser.add_argument("--solid-window", dest="solid_window", type=int, default=5,
                     help="Promotion gate: number of consecutive chunks averaged.")
parser.add_argument("--no-stop-when-solid", dest="stop_when_solid", action="store_false",
                     help="Keep training after the gate fires (SOLID.json is still written).")
parser.add_argument("--residual-scale", dest="residual_scale", type=float, default=0.0,
                     help="> 0 enables PID-residual mode: env_action = pid + scale * policy_action (policy starts "
                          "as a zero correction = pure PID, imitation stage skipped). 0 = direct-policy training.")
parser.add_argument("--residual-resume", dest="residual_resume", action="store_true",
                     help="With --residual-scale: --bc-checkpoint is itself a residual-mode checkpoint (e.g. the "
                          "previous stage's SOLID.json one), so KEEP its learned actor head instead of zeroing it.")
parser.add_argument("--detach-critic", dest="detach_critic", action="store_true",
                     help="Value-head gradients don't reach the shared actor/critic trunk.")
parser.add_argument("--spawn-speed-low", dest="spawn_speed_low", type=float, default=None,
                     help="Both this and --spawn-speed-high set: every episode spawns already moving at "
                          "a linear speed ~ Uniform(low, high) toward its own target, instead of from rest "
                          "-- retrain for a mid-flight hand-off (app.control.two_phase's switch_dist), not "
                          "a from-rest approach. Read realistic values off the two-phase test tool's "
                          "[SWITCH] log line. Omit both to keep the old always-from-rest spawn.")
parser.add_argument("--spawn-speed-high", dest="spawn_speed_high", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym

import app.environmental.base_drone_env_isaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from app.training.base_training_isaac import train

# deprecated/training/best_hparams_isaac.json -- Optuna trial 137 (2026-09-25 study, base_training_isaac_optuna_v2,
# 256 envs / 2M steps), picked by app/guidance/analyze_optuna_isaac.py for holding its grade best after the
# actor unfreezes (delta -0.11, 67% of post-unfreeze chunks >= 0.9), NOT by the study's own "best" trial 162
# (ranked 27th -- its 0.955 was one lucky 5-episode final chunk). All values as trial 137 found them EXCEPT:
#   lr: 2.9e-4 / 2 = 1.45e-4. The search ran at minibatch 1024 over 12 post-unfreeze chunks; production is
#   minibatch ~32,768 (16384 envs x 128 steps / 64) over ~35 chunks -- ~3x more AdamW steps, each a cleaner (more
#   coherent) direction, so a given lr moves the policy further; hence smaller, not the textbook "bigger batch,
#   bigger lr". But not divided by the batch ratio (~1e-5): within the working band, lower lr did WORSE in the
#   search (Spearman lr-vs-delta +0.55; lr < 1e-5 was the worst bucket) -- likely because critic warmup is
#   lr-limited (value_loss still falling at chunk 16). Results were flat from 1e-4 to 1e-3 and every trial at
#   lr >= 1e-3 collapsed, so 1.45e-4 sits at the low edge of the flat region. Raise toward 2.9e-4 if critic
#   value_loss is still high when the actor unfreezes; lower toward 1e-4 if grade drops right after unfreeze.
#   ent_coef_end: Optuna's ent_coef_end_frac (0.831) x ent_coef_start, since train() reads an absolute value.
# v3 changes on top of that (after v1/v2 both drifted: PPO's own training reward peaks ~chunks 5-15 after
# unfreeze, then falls 20-25%; std crept up 0.084 -> 0.090 under a 0.0166-0.02 entropy bonus):
#   ent_coef_start/end 0.01 / 0.002 (was 0.02 / 0.0166) -- the entropy bonus was the only steady push on
#   actor_log_std; smaller lets the policy gradient shrink std if noise hurts this precision task.
#   log_std_max -2.3 (was -1.96): std ceiling ~0.10, start 0.071. Optuna: tighter log_std_max correlated
#   with less post-unfreeze degradation (Spearman -0.27; 9 of the top 10 trials were <= -1.7).
#   (scheduled_lr also changed in code: flat lr through warmup, then the ORIGINAL full-span cosine -- see
#   its docstring.) Both are bets from two runs, not verified causes; if v3 misses the promotion gate, the
#   mechanism-agnostic fallback is anchoring PPO to the PID teacher (distillation loss).
# log_std_max -1.96 in trial 137 = std ceiling ~0.14 (tighter than the -0.9 default).


def main():
    env_cfg = parse_env_cfg("Isaac-Base-Drone-Direct-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Base-Drone-Direct-v0", cfg=env_cfg)

    hparams_override = None
    if args_cli.hparams_json and os.path.exists(args_cli.hparams_json):
        with open(args_cli.hparams_json) as f:
            hparams_override = json.load(f)
        print(f"[train_isaac] hyperparameter overrides loaded from {args_cli.hparams_json}: {hparams_override}")
    elif args_cli.hparams_json:
        print(f"[train_isaac] no hparams file at {args_cli.hparams_json}, using base_training_isaac.py's own defaults")

    train(
        env, distance_low=args_cli.distance_low, distance_high=args_cli.distance_high,
        checkpoint_dir=args_cli.checkpoint_dir,
        bc_checkpoint_path=args_cli.bc_checkpoint_path,
        total_timesteps=args_cli.total_timesteps,
        seed=args_cli.seed,
        hparams_override=hparams_override,
        num_steps_per_chunk=args_cli.num_steps_per_chunk,
        solid_grade=args_cli.solid_grade, solid_min_chunk=args_cli.solid_min_chunk,
        solid_window=args_cli.solid_window, stop_when_solid=args_cli.stop_when_solid,
        residual_scale=args_cli.residual_scale, detach_critic=args_cli.detach_critic,
        residual_resume=args_cli.residual_resume,
        spawn_speed_low=args_cli.spawn_speed_low, spawn_speed_high=args_cli.spawn_speed_high,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
