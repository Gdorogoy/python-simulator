"""What a drone KNOWS about other drones and about its target. PLACEHOLDERS: the interfaces and the observation features are real, the
sensing behind them is not written yet (see future/plan.md).

Priority per observed drone, fixed in code (not learned): own camera (FOV) -> base link -> prediction (filter). Today every drone is
"seen" through the first one with the true position, so the observation equals the simulator's truth.
"""
from __future__ import annotations

import torch

SOURCE_FOV, SOURCE_BASE, SOURCE_PREDICT = 0, 1, 2      # which source produced an estimate (kept for logging; not an observation input)

# observation sections of the v2 layout (environmental/docs.md "Observation layouts")
OPPONENT_FEATURES = 4      # estimated position x, y, z + status (known / alive)
ALLY_FEATURES = 5          # x, y, z, status (alive / lost), id
FLAG_FEATURES = 3          # link up, jammed, autonomous

JUMP_THRESHOLD = 20.0      # m: the target estimate moving further than this in ONE policy step cannot be flight (<= ~1 m per 60 Hz step)


def estimate_opponents(true_pos_w: torch.Tensor, alive: torch.Tensor):
    """PLACEHOLDER. (N,K,3) true world positions + (N,K) alive -> (estimated position (N,K,3), known (N,K) bool, source (N,K) long).
    Later: FOV detection, then base link, then a filter's prediction. Now: the truth, always seen by the own camera."""
    return true_pos_w, alive, torch.full(alive.shape, SOURCE_FOV, dtype=torch.long, device=alive.device)


def jump_detected(previous: torch.Tensor, current: torch.Tensor, threshold: float = JUMP_THRESHOLD) -> torch.Tensor:
    """(N,) bool: the (N,3) target estimate moved by more than `threshold` m since the last step -- a jam, not flight."""
    return torch.linalg.norm(current - previous, dim=-1) > threshold


def mode_flags(num_envs: int, device) -> torch.Tensor:
    """PLACEHOLDER. (N, FLAG_FEATURES) = [link up, jammed, autonomous]. Now: link always up, never jammed, never autonomous."""
    flags = torch.zeros(num_envs, FLAG_FEATURES, device=device)
    flags[:, 0] = 1.0
    return flags
