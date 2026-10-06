"""One-off diagnostic: isolates whether a hit-rate collapse is a PID-teacher/task-geometry
problem or a BC/DAgger learning problem, by running both against the same full-sphere target
distribution. Pure numpy oracle -- no Isaac/GPU needed. Note: no decimation, so not a faithful
comparison to the real deployed control rate (see scripts/isaac_lab/diagnose_model_isaac.py).

Usage: python scripts/diagnose_full_sphere.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.control.pid import PIDController
from app.environmental.base_drone_env import BaseDroneEnv
from app.reward_functions.rewards import reward_func
from app.training.diagnostics import diagnose_with_model, diagnose_with_pid
from app.training.eval_matrix import build_uniform_omni_eval_pairs

N_EPISODES = 200
OOB_RADIUS = 30.0  # matches reward_func's max(20, start_dist*3) at start_dist=10

with open("app/control/best_pid_gains_per_dist.json") as f:
    gains_by_dist = json.load(f)

target_pairs = build_uniform_omni_eval_pairs(oob_radius=OOB_RADIUS, low=3.0, high=10.0, n_pairs=N_EPISODES)
env = BaseDroneEnv(reward_func, target_pairs=target_pairs)

print(f"=== PID teacher, full-sphere targets, n={N_EPISODES} ===")
pid = PIDController(**gains_by_dist["3"])
diagnose_with_pid(pid, env, N_EPISODES, gains_by_dist=gains_by_dist)

try:
    import torch
    from app.guidance.train import ActorCritic, device

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         env.action_space.low, env.action_space.high,
                         hidden=64, num_hidden_layers=4).to(device)
    model.load_state_dict(torch.load("app/control/pretrained_bc_dagger.pt", map_location=device))
    model.eval()

    print(f"\n=== trained student (pretrained_bc_dagger.pt), full-sphere targets, n={N_EPISODES} ===")
    diagnose_with_model(model, env, N_EPISODES)
except FileNotFoundError:
    print("\n(no pretrained_bc_dagger.pt found -- skipping student comparison)")
