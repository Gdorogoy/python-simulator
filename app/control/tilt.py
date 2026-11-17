"""Tilt-cone (isotropic) clamp shared by every controller that turns an acceleration command into roll/pitch.

The airframe limit is a cone: tilt = atan(|a_xy| / g) <= max_tilt, i.e. |(roll, pitch)| <= tan(max_tilt).
Clamping roll and pitch separately is a BOX: on a pure diagonal its corners reach sqrt(2) x the cap (~1.41x the
tilt at small angles), and when only one axis saturates the commanded direction rotates, so diagonal targets are
approached off-line and undershoot. Every caller (PID, PID teacher, carrot, LOS, both sims) uses these helpers so
the limit is defined once. See docs.md "Controllers" / control/los.py.
"""
from __future__ import annotations

import math

import numpy as np


def tilt_cone_np(raw_roll, raw_pitch, max_tilt):
    """Scale (raw_roll, raw_pitch) onto the cone |(roll, pitch)| <= tan(max_tilt), direction preserved.
    Inputs are already accel / g; scalars and arrays both work."""
    cap = math.tan(float(max_tilt))
    norm = np.maximum(np.hypot(raw_roll, raw_pitch), 1e-9)
    scale = np.minimum(cap / norm, 1.0)
    return raw_roll * scale, raw_pitch * scale


def tilt_cone_torch(raw_roll, raw_pitch, max_tilt):
    """Torch mirror of tilt_cone_np (same formula, elementwise over (N,))."""
    import torch  # local: numpy-only callers (pid.py) must not pull torch in

    cap = math.tan(float(max_tilt))
    norm = torch.sqrt(raw_roll ** 2 + raw_pitch ** 2).clamp_min(1e-9)
    scale = torch.clamp(cap / norm, max=1.0)
    return raw_roll * scale, raw_pitch * scale
