"""PID-midcourse -> RL-terminal two-phase controller, for testing an RL
checkpoint against the interception task without asking it to fly the whole
distance itself (see alt_model.md's midcourse/terminal split).

Phase PID: drives the drone toward the target. Gains are derived ON THE FLY
for the episode's ACTUAL sampled distance via tune_pid.compute_gains_for_distance
-- the same closed-form pole-placement math tune_pid.py uses to build the
pre-baked best_pid_gains_per_dist.json ladder, just evaluated at the exact
distance instead of snapped to the nearest of 8 pre-tuned entries. This is
what "max speed" means here: the fastest settle time that doesn't saturate
max_tilt_rad at THAT distance, not a fixed/nearest-neighbor approximation --
and it isn't capped at the ladder's 250m ceiling either.

Phase RL: switch_dist gates engagement for BOTH modes -- PID alone flies the
midcourse leg in either case (the gain derivation above extrapolates fine
well past any distance a checkpoint was trained at; it's closed-form pole
placement, not a fit to a trained range). Once distance <= switch_dist:
  - residual_scale == 0 (default): pure model action (deterministic/mean,
    via scale_action) -- the real PID-midcourse / RL-terminal split.
  - residual_scale > 0: composed action = pid_action + residual_scale *
    model_action, clipped to the env's action space -- REQUIRED for a
    checkpoint trained with base_training_isaac.py's --residual-scale /
    PidResidualEnv mode, where the actor head only ever learned a small
    correction on top of a continuously-running PID, never a full action
    (see record_run.run_episode's docstring/RESIDUAL_GAINS_PATH comment:
    feeding a residual checkpoint's raw output ALONE means "hover, don't
    steer"). The PID keeps running through the RL phase in this mode -- it's
    the baseline the correction sits on -- but its gains get RE-SELECTED at
    the exact moment of hand-off, for the hand-off distance, not reused from
    the episode's original (possibly much larger) start distance: a training
    episode only ever spanned a short range (e.g. Uniform(3,30)), so gains
    chosen once at episode start always roughly matched the CURRENT distance
    too. Reusing gains solved for a long start distance well past hand-off
    would leave the PID baseline loose/slow right when the correction
    engages -- nothing like the tight, appropriately-tuned baseline it
    actually trained against.

    Gating a residual model's correction behind switch_dist is an
    approximation training itself never did (training's own correction ran
    continuously from step 0 -- see _select_pid_teacher) -- the hand-off
    state may not perfectly match what continuous correction would have
    produced. But running the correction across distances the checkpoint
    never trained on AT ALL (the "always on" alternative) is a strictly
    bigger mismatch, not a smaller one, for any distance beyond the
    checkpoint's trained range. Set switch_dist at or inside the
    checkpoint's actual trained distance range (e.g. <=30 for a
    Uniform(3,30)-trained checkpoint) so hand-off happens somewhere the
    correction has actually seen before.

Usage:
    agent = TwoPhaseAgent(model, switch_dist=20.0, residual_scale=0.0, log=print)
    agent.reset(env)          # call right after env.reset()
    action = agent.get_action(obs, env)   # call each step
"""
from __future__ import annotations

import json

import numpy as np
import torch

from app.control.pid import PIDController
from app.control.tune_pid import compute_gains_for_distance

DEFAULT_GAINS_PATH = "app/control/best_pid_gains_per_dist.json"
DEFAULT_SWITCH_DIST = 20.0
MIN_GAIN_DIST = 1.0  # compute_gains_for_distance's log(dist/HIT_THRESHOLD) term blows up/goes
                       # negative for dist below its own HIT_THRESHOLD -- floor the distance it's
                       # solved for, never the PID's actual target (that stays exact).


def load_gains_by_dist(path: str = DEFAULT_GAINS_PATH) -> dict:
    """Loads the pre-baked ladder (best_pid_gains_per_dist.json) -- only
    needed if a caller explicitly wants nearest-neighbor-snapped gains
    instead of TwoPhaseAgent's default per-episode analytic derivation."""
    with open(path) as f:
        return json.load(f)


def _nearest_gains(gains_by_dist: dict, dist: float):
    key = min(gains_by_dist.keys(), key=lambda k: abs(float(k) - dist))
    return key, gains_by_dist[key]


def gains_for_distance(dist: float):
    """Closed-form PID gains for the exact distance dist (tune_pid.compute_gains_for_distance),
    not a nearest-ladder-snap -- works past the ladder's 250m ceiling too.
    Returns (label, gains_dict)."""
    solved_dist = max(dist, MIN_GAIN_DIST)
    gains, diag = compute_gains_for_distance(solved_dist)
    label = f"analytic@{solved_dist:.1f}m (wn_pos={diag['wn_pos']:.3f}, sat_ratio={diag['saturation_ratio']:.2f})"
    return label, gains


class TwoPhaseAgent:
    """Stateful get_action(obs, env) — call reset(env) right after env.reset()."""

    def __init__(self, model, gains_by_dist: dict | None = None, switch_dist: float = DEFAULT_SWITCH_DIST,
                 residual_scale: float = 0.0, device: str = "cpu", log=None):
        """gains_by_dist: pass the loaded best_pid_gains_per_dist.json dict to use
        nearest-neighbor-snapped, pre-tuned gains instead (fast, matches whatever
        a checkpoint's own training/demo data used). Default None: derive gains
        analytically per-episode for the exact sampled distance -- see module
        docstring. residual_scale: > 0 if model was trained in PID-residual mode
        -- see module docstring's Phase RL section, this changes how the RL
        phase composes its action, must match the value training used."""
        self.model = model
        self.gains_by_dist = gains_by_dist
        self.switch_dist = switch_dist
        self.residual_scale = residual_scale
        self.device = device
        self.log = log or (lambda msg: None)

        self.pid: PIDController | None = None
        self.phase: str | None = None
        self.gain_key = None
        self.switch_step: int | None = None
        self._step = 0

    def reset(self, env):
        dist = env.prev_distance
        if self.gains_by_dist is not None:
            self.gain_key, gains = _nearest_gains(self.gains_by_dist, dist)
        else:
            self.gain_key, gains = gains_for_distance(dist)
        self.pid = PIDController(**gains)
        self.pid.reset()
        self._step = 0
        # switch_dist gates engagement for BOTH modes -- PID alone flies the midcourse
        # leg (it extrapolates fine well past a checkpoint's trained range; see
        # gains_for_distance/compute_gains_for_distance), and the model only engages
        # once within switch_dist, exactly the correction's actual trained envelope.
        # This trades a small state-distribution mismatch at hand-off (training never
        # gated -- see _select_pid_teacher, correction ran continuously) for a much
        # bigger win: the correction never has to operate at distances it was never
        # trained on at all. Caller should still set switch_dist inside (or near) the
        # checkpoint's trained distance range for residual_scale > 0 -- handing off at
        # a distance the model itself never trained near is a second, separate mismatch.
        self.phase = "PID"
        self.switch_step = None
        self.log(f"[PID]    start   dist={dist:8.2f}m  gains={self.gain_key}  target={env.target_pos.tolist()}")

    def get_action(self, obs, env):
        dist = env.prev_distance

        if self.phase == "PID" and dist <= self.switch_dist:
            self.phase = "RL"
            self.switch_step = self._step
            mode = "PID+residual" if self.residual_scale > 0 else "pure RL"
            if self.residual_scale > 0:
                # Re-select gains for the hand-off distance, not the episode's original
                # (possibly much larger) start distance -- a training episode only ever
                # spanned Uniform(3,30) or similar, so gains chosen once at episode start
                # always roughly matched the CURRENT distance too (short episodes).
                # Reusing gains solved for a long start distance well past hand-off leaves
                # the PID baseline loose/slow right when the correction engages -- nothing
                # like the tight, appropriately-tuned baseline it trained against.
                if self.gains_by_dist is not None:
                    self.gain_key, gains = _nearest_gains(self.gains_by_dist, dist)
                else:
                    self.gain_key, gains = gains_for_distance(dist)
                self.pid = PIDController(**gains)
                self.pid.reset()
                self.log(f"[SWITCH] PID -> {mode}  dist={dist:6.2f}m  threshold={self.switch_dist}m  "
                         f"step={self._step}  re-selected gains={self.gain_key}")
            else:
                self.log(f"[SWITCH] PID -> {mode}  dist={dist:6.2f}m  threshold={self.switch_dist}m  step={self._step}")

        self._step += 1

        # residual_scale > 0: the PID keeps flying all the way through the RL phase too
        # (re-selected gains from the switch above) -- it's the baseline the correction
        # is added to, not something that gets switched off once the model engages.
        if self.phase == "PID" or self.residual_scale > 0:
            pid_action = self.pid.compute_action(env.drone_state, env.target_pos, env.target_yaw, dt=env.dt)
            if self.phase == "PID":
                return pid_action
        else:
            pid_action = None

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = self.model.forward(obs_t)
            model_action = self.model.scale_action(mean).squeeze(0).cpu().numpy()

        if self.residual_scale > 0:
            return np.clip(pid_action + self.residual_scale * model_action,
                            env.action_space.low, env.action_space.high)
        return model_action
