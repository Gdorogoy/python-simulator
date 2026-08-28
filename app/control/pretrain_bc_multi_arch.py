"""Retrains one BC checkpoint per (hidden, num_hidden_layers) combo that
app.guidance.optuna_search's suggest_params can sample, all from the same
aggregated DAgger dataset (app/control/demonstrations_dagger.npz).

Only the final BC fit depends on architecture -- the on-policy rollout/correction
data collected by dagger.py doesn't, so there's no need to rerun DAgger itself per
architecture. This just repeats the cheap supervised-regression step (same as
pretrain_bc.py's standalone run) once per shape, so optuna_search.py can load a
checkpoint that actually matches whatever architecture a trial samples instead of
silently falling back to a random init on any shape mismatch.

Usage:
    python -m app.control.pretrain_bc_multi_arch
"""
import os

import torch

from app.control.pretrain_bc import pretrain_behavior_cloning
from app.guidance.train import ActorCritic, device

# Must match the choices in app.guidance.optuna_search.suggest_params exactly --
# those are what a trial's p["hidden"]/p["num_hidden_layers"] will be.
HIDDEN_CHOICES = (32, 64, 96, 128)
NUM_HIDDEN_LAYERS_CHOICES = (2, 3, 4, 5, 6)

DEMO_PATH = "app/control/demonstrations_dagger.npz"
OUT_DIR = "app/control"


def checkpoint_path(hidden: int, num_hidden_layers: int, out_dir: str = OUT_DIR) -> str:
    return os.path.join(out_dir, f"pretrained_bc_dagger_h{hidden}_l{num_hidden_layers}.pt")


if __name__ == "__main__":
    from app.environmental.interceptor_drone import InterceptorDroneEnv
    from app.reward_functions.reward_fn_phase1 import RewardFnPhase1

    reward_fn = RewardFnPhase1(
        hit_steps_streak=1500,
        phase1_pos_coef=0.25,
        hit_reward=5,
        oob_radius=300,
        hover_success_steps=None,
        streak_cap=60,
        outer_dist=1.0,
        inner_dist=0.3,
        hit_threshold=0.05,
        imitation_duration_steps=0,
        phase1_duration_steps=100,
    ).as_roadmap()
    shape_env = InterceptorDroneEnv(reward_fn)
    obs_dim = shape_env.observation_space.shape[0]
    action_dim = shape_env.action_space.shape[0]
    action_low = shape_env.action_space.low
    action_high = shape_env.action_space.high

    for hidden in HIDDEN_CHOICES:
        for num_hidden_layers in NUM_HIDDEN_LAYERS_CHOICES:
            out_path = checkpoint_path(hidden, num_hidden_layers)
            print(f"=== training hidden={hidden} num_hidden_layers={num_hidden_layers} -> {out_path} ===")

            model = ActorCritic(obs_dim, action_dim, action_low, action_high,
                                 hidden=hidden, num_hidden_layers=num_hidden_layers).to(device)
            model = pretrain_behavior_cloning(model, demo_path=DEMO_PATH, epochs=55, batch_size=256, lr=1e-3)

            torch.save(model.state_dict(), out_path)
            print(f"saved {out_path}")
