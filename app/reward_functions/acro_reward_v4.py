"""Acro reward v4 = v3 with the four fixes that make the POLICY fly the straight path at the airframe limit.

v3 added the pace term, but the reward-v3 runs still flew a bowed path: the first ran at ~34 m/s and 38.3 s, the
tuned one at ~36 m/s and 36.3 s with a ~33 m mean cross-track (a flight mean; the terminal aim was ~0.5 m). Four
reward-shape bugs explain it, and each one is fixed here:

1. **The hit test punished speed.** `compute_acro_reward` scored a hit from the point distance at the 60 Hz sample. At
   50 m/s a step covers 0.83 m, so a 1 m sphere is a 1.2-2.4 sample window: flying faster made the hit *less* likely
   even with a perfect path. `hit_swept` (passed through to v1) tests the whole segment flown in the step instead.
   `zone_time` must stay 0 for a speed run (a swept miss is one step long by construction).
2. **Cross-track was charged per STEP, not per metre.** A per-step penalty is discounted by going faster, exactly the
   wrong gradient. `line_path_leg` charges `lateral / line_ref` per metre flown, so the total over a leg of length D is
   `line_path_leg` at `line_ref` off the line, whether it is flown at 20 or 50 m/s. `line_leg` (v3) is forced to 0.
3. **Cross-track was a P term with no derivative.** Being on the line but drifting off it cost nothing this step, and
   correcting at 50 m/s (no lift margin at ~70 deg tilt) is expensive later. `line_lead_s` scores the offset the
   velocity is heading for `lead` seconds ahead, so starting to drift is charged immediately.
4. **Nothing terminal scored the path.** `closing_k` / `progress_k` telescope to a constant, so a 5 % longer path and a
   straight one ended the same. `r_path_bonus` (auto-scaled from `path_leg`) pays once on a hit, scaled by how much
   longer than the straight line the flown path was: full for a straight line, 0 at `path_slack` extra fraction.

The LOS teacher also gets an isotropic tilt clamp (`app/control/los.py`): v3 distilled from a teacher that bowed on
diagonal targets because the old clamp limited roll and pitch PER AXIS, so the pitch axis saturated first and rotated
the commanded direction. Distil with `--distill-end 0` from a v3 checkpoint: holding 0.02 re-imposes its offset.

Everything else is v3 unchanged. Tests: `app/test/test_reward_v4.py`.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, fields

import torch

from app.reward_functions.acro_reward_v3 import AcroRewardV3Config, cross_track, install_acro_reward_v3
from app.reward_functions.proportion import _is_zero, resolve_proportion


@dataclass
class AcroRewardV4Config(AcroRewardV3Config):
    # ---- swept hit (v1's compute_acro_reward) ----
    hit_swept: bool = True            # score a hit when the segment flown in one step passes through the sphere
    # ---- cross-track per METRE flown (v3's line_leg is per step and is forced to 0) ----
    line_path_leg: float = 0.0        # per-leg cost of flying a whole leg `line_ref` off the line (0 = off)
    line_lead_s: float = 0.0          # s of look-ahead: score the offset the current velocity is heading for (0 = off)
    # ---- terminal straightness bonus (auto-scaled from proportion.path_leg) ----
    r_path_bonus: float = 1.0         # nonzero marker; auto_scale replaces it with path_leg (0 = off)
    path_slack: float = 0.10          # extra path length, as a fraction of the straight line, that cancels the bonus

    @staticmethod
    def from_overrides(overrides: dict | None) -> "AcroRewardV4Config":
        cfg = AcroRewardV4Config()
        known = {f.name for f in fields(AcroRewardV4Config)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown acro reward field {k!r}; known: {sorted(known)}")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


def cross_track_lead(rel: torch.Tensor, vel: torch.Tensor, rel0: torch.Tensor, lead_s: float) -> torch.Tensor:
    """(N,) cross-track error `lead_s` seconds ahead: the perpendicular offset of `pos + vel * lead_s` from the
    episode's start->target line. With lead 0 it is v3's `cross_track(rel, rel0)`. On the line but drifting off it,
    the future offset is non-zero, so the term charges the drift before it happens (a derivative on the lateral)."""
    if lead_s <= 0.0:
        return cross_track(rel, rel0)
    return cross_track(rel - vel * float(lead_s), rel0)


def compute_line_path_term(*, rel: torch.Tensor, vel: torch.Tensor, rel0: torch.Tensor, active: torch.Tensor,
                           cfg: AcroRewardV4Config, ref_dist: torch.Tensor, step_dt: float) -> torch.Tensor:
    """(N,) per-step cross-track cost of the metres flown in this step:

        -(line_path_leg / ref_dist) * clamp(lateral_lead / line_ref, 0, 1) * |vel| * dt

    Summed over a leg of length ``ref_dist`` flown at `line_ref` off the line it is exactly `line_path_leg`, whatever
    the speed: the same offset costs the same per METRE instead of per sample (v3's per-step form paid a fast flight
    less for the same detour). `ref_dist` is the live leg length used by the rest of the reward (handoff h or the
    start distance). 0 where inactive, `line_path_leg` is 0, or `line_lead_s` is 0 and the drone is on the line."""
    if _is_zero(cfg.line_path_leg):
        return torch.zeros_like(torch.linalg.norm(rel, dim=-1))
    lateral = cross_track_lead(rel, vel, rel0, float(cfg.line_lead_s))
    pen = (lateral / max(float(cfg.line_ref), 1e-6)).clamp(0.0, 1.0)
    metres = torch.linalg.norm(vel, dim=-1) * float(step_dt)
    return -(float(cfg.line_path_leg) / ref_dist.clamp_min(1e-6)) * pen * metres * active.float()


def path_ratio(rel0: torch.Tensor, path_len: torch.Tensor) -> torch.Tensor:
    """(N,) straight-line distance / flown path length, clamped to [0, 1]: 1 = the optimal path, 0.9 = 11 % longer."""
    d0 = torch.linalg.norm(rel0, dim=-1)
    return (d0 / path_len.clamp_min(1e-6)).clamp(0.0, 1.0)


def compute_path_bonus(*, rel0: torch.Tensor, path_len: torch.Tensor, hit: torch.Tensor, active: torch.Tensor,
                       cfg: AcroRewardV4Config, r_path_bonus) -> torch.Tensor:
    """(N,) one-off bonus on a hit: `r_path_bonus * clamp(1 - extra / path_slack, 0, 1)`, where
    `extra = path_len / |rel0| - 1` is how much longer the flown path was than the straight line. Full on a straight
    path, 0 once the path is `path_slack` (fraction) longer. `r_path_bonus` is the auto-scaled value (0 = off)."""
    if _is_zero(r_path_bonus):
        return torch.zeros_like(torch.linalg.norm(rel0, dim=-1))
    d0 = torch.linalg.norm(rel0, dim=-1).clamp_min(1e-6)
    extra = ((path_len - d0) / d0).clamp_min(0.0)
    frac = (1.0 - extra / max(float(cfg.path_slack), 1e-6)).clamp(0.0, 1.0)
    return float(r_path_bonus) * frac * hit.float() * active.float()


def install_acro_reward_v4(unwrapped_env, cfg: AcroRewardV4Config | None = None) -> AcroRewardV4Config:
    """Install v3, then replace the per-step line term with the per-metre + lead form and add the terminal path bonus.
    Statistics: `r_line` (the per-metre term), `r_path` (`rew_path`), `train_perp_miss_m`, `train_hit_speed`."""
    cfg = cfg or AcroRewardV4Config()
    if not _is_zero(cfg.line_leg):
        print("[acro reward v4] line_leg is a per-STEP penalty (discounted by speed) and is replaced by "
              "line_path_leg (per metre); setting line_leg = 0")
        cfg.line_leg = 0.0
    install_acro_reward_v3(unwrapped_env, cfg)
    v3_compute = unwrapped_env._compute_dones_and_reward
    u = unwrapped_env
    step_dt = float(getattr(u, "step_dt", 1.0 / 60.0))
    u._v4_path = torch.zeros(u.num_envs, device=u.device)          # metres flown on this leg (policy steps only)
    if not getattr(u, "_acro_v4_reset_wrapped", False):
        prev_reset = u._reset_idx

        def _reset_idx(self, env_ids):
            prev_reset(env_ids)
            if env_ids is None or len(env_ids) == self.num_envs:
                self._v4_path.zero_()
            else:
                self._v4_path[env_ids] = 0.0

        u._reset_idx = types.MethodType(_reset_idx, u)
        u._acro_v4_reset_wrapped = True

    def _compute_dones_and_reward(self):
        terminated = v3_compute()                      # v3 sets the reward, episode sums and its own statistics
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        rel = (self._desired_pos_w - self._terrain.env_origins) - pos
        vel = self._robot.data.root_lin_vel_w
        handoff_h = getattr(self, "_handoff_h", None)
        if getattr(self, "_handoff_snap_enabled", False) and handoff_h is not None:
            ref_dist = torch.where(handoff_h > 0, handoff_h, self._start_dist)
        else:
            ref_dist = self._start_dist
        handed = getattr(self, "_handed_off", None)
        if handed is not None:
            policy_on = handed.clone()
        else:
            prefix = getattr(self, "_prefix_left", None)
            policy_on = (prefix <= 0) if prefix is not None else torch.ones_like(rel[:, 0], dtype=torch.bool)
        # odometer: every metre flown while the policy is in control (the hit step flies too, so not ~terminated)
        self._v4_path = self._v4_path + (torch.linalg.norm(vel, dim=-1) * step_dt) * policy_on.float()
        # v3 captured the leg's straight line on its first policy step; the bonus needs the same line and its length
        on_line = policy_on & self._v3_rel0_valid
        line = compute_line_path_term(rel=rel, vel=vel, rel0=self._v3_rel0, active=on_line, cfg=self._acro_cfg,
                                      ref_dist=ref_dist, step_dt=step_dt)
        cfg_eff = resolve_proportion(self._acro_cfg, ref_dist, step_dt)   # one leg = one unit, so path_leg -> r_path_bonus
        hit = self.extras["term_reasons"]["hit"]
        bonus = compute_path_bonus(rel0=self._v3_rel0, path_len=self._v4_path, hit=hit, active=on_line,
                                   cfg=self._acro_cfg, r_path_bonus=cfg_eff.r_path_bonus)
        extra = line + bonus
        self._cached_reward = self._cached_reward + extra
        self._episode_sums["reward"] += extra
        acc = self._acro_acc
        acc["r_line"] += line.sum()
        acc["r_path"] += bonus.sum()
        acc["r_total"] += extra.sum()
        return terminated

    unwrapped_env._compute_dones_and_reward = types.MethodType(_compute_dones_and_reward, unwrapped_env)
    return cfg
