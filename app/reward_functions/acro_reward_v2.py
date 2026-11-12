"""Acro reward v2 = acro_v1 with the progress terms of angle_reward_rc (total-distance closing + per-axis progress, both with
closing_lambda on retreating) and its line-of-sight penalty (same formulas, same defaults). v1's own progress term is off (progress_k 0).

The term compares the velocity direction with the direction to the target: -los_k * m(angle), m = los_aligned when the
velocity points at the target, +1 more by los_b (sideways), +los_far more by los_c (straight away). It is only paid once the
policy has taken over and only while dist > los_gate_frac * (distance at handoff).
Two anti-loop terms (after the takeover): A) every fly-by that does not hit costs k_miss * its closest distance;
B) more than near_limit_s in total inside near_radius ends the episode with -r_near. Everything else is v1 unchanged."""

from __future__ import annotations

import types
from dataclasses import dataclass, fields

import torch

from app.reward_functions.acro_reward_v1 import AcroRewardConfig, install_acro_reward
from app.reward_functions.angle_reward_rc import _smoothstep
from app.reward_functions.proportion import _col, _fill, _is_zero, resolve_proportion


@dataclass
class AcroRewardV2Config(AcroRewardConfig):
    progress_k: float = 0.0             # v1's per-axis log1p progress is replaced by the two rc terms below (set > 0 to add it back)
    # ---- closing + per-axis progress (same fields and defaults as RewardRCConfig) ----
    closing_k: float = 2.0              # k*(ln(prev_dist+eps) - ln(dist+eps)), telescopes to k*ln(d0/d_end); 0 = off
    closing_eps: float = 0.5
    closing_lambda: float = 1.0         # weight of moving away; > 1 punishes retreating harder than approaching pays
    axis_k: float = 0.5                 # axis_k * sum_i (ln(|prev_rel_i|+eps) - ln(|rel_i|+eps)) over x, y, z; 0 = off
    axis_eps: float = 0.5
    # ---- closing SPEED (linear): pays for flying fast toward the target at every distance. The log closing term above pays
    # k * v / dist, ~nothing when far (where speed matters), and a hit 30 s away is invisible at gamma 0.99-0.995 (horizon 1.7-3 s).
    # Per step: closing_speed_k * v_toward / closing_speed_ref, v_toward = drone velocity along the line to the target (the target's
    # own motion does not count, the policy cannot change it); flying away pays closing_lambda times that (negative) ----
    closing_speed_k: float = 0.0        # per policy step at v_toward == closing_speed_ref; 0 = off
    closing_speed_ref: float = 20.0     # m/s
    closing_speed_min_dist: float = 0.0 # m; only paid while farther than this (0 = all the way to the target)
    # ---- los (same fields and defaults as RewardRCConfig) ----
    los_k: float = 0.02                 # per-step weight of the line-of-sight term; 0 = off (then this is exactly v1)
    los_mode: str = "penalty"           # "penalty": -los_k * m(angle), m = 1 aligned, 2 sideways, 5 straight away; "cos": +los_k * cos
    los_a: float = 35.0                 # m stays los_aligned up to this angle (deg)
    los_b: float = 90.0                 # the first band reaches its full +1 here
    los_c: float = 135.0                # the far band reaches its full +los_far here
    los_far: float = 3.0                # extra multiplier past los_b
    los_aligned: float = 1.0            # m when the velocity points straight at the target; 0 = free until los_a
    los_min_speed: float = 0.5          # m/s; slower than this the direction is noise, term = 0
    los_gate_frac: float = 0.5          # los term only while dist > this * (dist at handoff); 0 = all the way to the target
    # ---- miss + hover
    stall_speed: float = 0.3  # m/s; slower than this while off-target counts as "stalled"
    hover_grace_s: float = 1.0  # stalled this long before the per-step penalty starts
    hover_limit_s: float = 3.0  # stalled this long -> terminate with r_hover
    c_hover: float = 0.05  # per policy step after the grace (0.05/step * 60 Hz * 2 s = 6)
    r_hover: float = 5.0  # one-off terminal cost; > r_crash on purpose (docs.md "qv2")
    # ---- A: missed-pass penalty (docs.md "acro_v2"): a fly-by that does not hit costs k_miss * closest distance of that pass ----
    k_miss: float = 1.0                 # per metre of miss distance, paid once per pass; 0 = off
    miss_radius: float = 15.0           # only passes whose closest point is inside this radius (m) count
    pass_speed_eps: float = 0.5         # m/s: "approaching" needs closing speed > +eps, the pass fires at closing speed < -eps
    # ---- B: near-miss timer: too long near the target without a hit ends the episode ----
    near_radius: float = 15.0           # m
    near_limit_s: float = 6.0           # TOTAL time (not consecutive) inside near_radius after the handoff; 0 = off
    r_near: float = 5.0                 # one-off terminal cost; below r_crash so crashing is never the cheaper way out


    @staticmethod
    def from_overrides(overrides: dict | None) -> "AcroRewardV2Config":
        cfg = AcroRewardV2Config()
        known = {f.name for f in fields(AcroRewardV2Config)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown acro reward field {k!r}; known: {sorted(known)}")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


def compute_los_term(*, vel: torch.Tensor, rel: torch.Tensor, gate: torch.Tensor | None, cfg: AcroRewardV2Config) -> torch.Tensor:
    """vel: (N,3) drone velocity, rel: (N,3) target minus drone, gate: (N,) bool or None. Returns the (N,) reward part."""
    if _is_zero(cfg.los_k):
        return torch.zeros(vel.shape[0], device=vel.device)
    speed = torch.linalg.norm(vel, dim=-1)
    cos_a = (vel * rel).sum(dim=-1) / (speed * torch.linalg.norm(rel, dim=-1) + 1e-6)
    moving = (speed > cfg.los_min_speed).float()
    if cfg.los_mode == "penalty":
        deg = torch.rad2deg(torch.acos(torch.clamp(cos_a, -1.0, 1.0)))
        m = cfg.los_aligned + _smoothstep(cfg.los_a, cfg.los_b, deg) + cfg.los_far * _smoothstep(cfg.los_b, cfg.los_c, deg)
        los = -cfg.los_k * m * moving
    elif cfg.los_mode == "cos":
        los = cfg.los_k * cos_a * moving
    else:
        raise ValueError(f"los_mode {cfg.los_mode!r}")
    return los if gate is None else los * gate.float()


def compute_progress_terms(*, rel: torch.Tensor, prev_rel: torch.Tensor, cfg: AcroRewardV2Config) -> torch.Tensor:
    """rc's closing + per-axis terms. rel / prev_rel: (N,3) target minus drone now / last step. Returns the (N,) reward part."""
    dist, prev_dist = torch.linalg.norm(rel, dim=-1), torch.linalg.norm(prev_rel, dim=-1)
    closing = cfg.closing_k * (torch.log(prev_dist + cfg.closing_eps) - torch.log(dist + cfg.closing_eps))
    closing = torch.where(closing < 0, closing * cfg.closing_lambda, closing)
    axis = _col(cfg.axis_k) * (torch.log(prev_rel.abs() + cfg.axis_eps) - torch.log(rel.abs() + cfg.axis_eps))
    axis = torch.where(axis < 0, axis * cfg.closing_lambda, axis).sum(dim=-1)
    return closing + axis


def compute_closing_speed_term(*, vel: torch.Tensor, rel: torch.Tensor, active: torch.Tensor, cfg: AcroRewardV2Config) -> torch.Tensor:
    """vel: (N,3) drone velocity, rel: (N,3) target minus drone, active: (N,) bool. Returns the (N,) reward part:
    closing_speed_k * v_toward / closing_speed_ref (x closing_lambda when negative), 0 where inactive or inside closing_speed_min_dist."""
    if _is_zero(cfg.closing_speed_k):
        return torch.zeros(vel.shape[0], device=vel.device)
    dist = torch.linalg.norm(rel, dim=-1)
    v_toward = (vel * rel).sum(dim=-1) / (dist + 1e-6)
    term = cfg.closing_speed_k * v_toward / cfg.closing_speed_ref
    term = torch.where(term < 0, term * cfg.closing_lambda, term)
    on = active & (dist > cfg.closing_speed_min_dist)
    return term * on.float()


def update_pass_state(*, dist: torch.Tensor, closing_speed: torch.Tensor, active: torch.Tensor, approaching: torch.Tensor,
                      d_min: torch.Tensor, cfg: AcroRewardV2Config):
    """Missed-pass detector (A). approaching: (N,) bool, d_min: (N,) closest distance since the approach started.
    A pass = approaching, then closing speed < -pass_speed_eps (flying away). Returns (penalty (N,), passed (N,) bool,
    new approaching, new d_min). Envs with active False are frozen out (no pass, state cleared)."""
    inf = torch.full_like(d_min, float("inf"))
    d_min = torch.where(approaching, torch.minimum(d_min, dist), inf)
    passed = approaching & (closing_speed < -cfg.pass_speed_eps) & (d_min < cfg.miss_radius) & active
    penalty = -cfg.k_miss * torch.where(passed, d_min, torch.zeros_like(d_min))
    leaving = approaching & (closing_speed < -cfg.pass_speed_eps)          # pass done (counted or too far out to count)
    starting = ~approaching & (closing_speed > cfg.pass_speed_eps)
    approaching = ((approaching & ~leaving) | starting) & active
    d_min = torch.where(starting & active, dist, torch.where(approaching, d_min, inf))
    return penalty, passed, approaching, d_min


def install_acro_reward_v2(unwrapped_env, cfg: AcroRewardV2Config | None = None) -> AcroRewardV2Config:
    """Install v1, then add the los part, the missed-pass penalty (A) and the near-miss timer (B) on top of its reward,
    episode sums and statistics (`rew_los`, `rew_miss`, `rew_near`, `train_near_miss_rate`, `train_passes_per_ep`)."""
    cfg = cfg or AcroRewardV2Config()
    install_acro_reward(unwrapped_env, cfg)
    v1_compute = unwrapped_env._compute_dones_and_reward
    u = unwrapped_env
    step_dt = float(getattr(u, "step_dt", 1.0 / 60.0))
    near_limit_steps = int(round(cfg.near_limit_s / step_dt)) if cfg.near_limit_s > 0 else 0
    u._v2_approach = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    u._v2_dmin = torch.full((u.num_envs,), float("inf"), device=u.device)
    u._v2_near = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
    if not getattr(u, "_acro_v2_reset_wrapped", False):
        prev_reset = u._reset_idx

        def _reset_idx(self, env_ids):
            prev_reset(env_ids)
            if env_ids is None or len(env_ids) == self.num_envs:
                self._v2_approach.fill_(False); self._v2_dmin.fill_(float("inf")); self._v2_near.zero_()
            else:
                self._v2_approach[env_ids] = False; self._v2_dmin[env_ids] = float("inf"); self._v2_near[env_ids] = 0

        u._reset_idx = types.MethodType(_reset_idx, u)
        u._acro_v2_reset_wrapped = True

    def _compute_dones_and_reward(self):
        prev_pos = self._prev_position.clone()          # v1 moves it to this step's position
        terminated = v1_compute()                       # sets self._cached_reward, _episode_sums and the v1 statistics
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        rel = (self._desired_pos_w - self._terrain.env_origins) - pos
        dist = torch.linalg.norm(rel, dim=-1)
        # same gate as the rc reward: only after the policy takes over, and only in the far part of the handoff distance.
        # ref_dist is also the per-leg reference: v1 used the same expression inside its own closure, so both parts of
        # the reward are scaled by the same N.
        handoff_h = getattr(self, "_handoff_h", None)
        if getattr(self, "_handoff_snap_enabled", False) and handoff_h is not None:
            ref_dist = torch.where(handoff_h > 0, handoff_h, self._start_dist)
        else:
            ref_dist = self._start_dist
        cfg_eff = resolve_proportion(self._acro_cfg, ref_dist, step_dt)
        handed = getattr(self, "_handed_off", None)
        if handed is not None:
            policy_on = handed.clone()
        else:
            prefix = getattr(self, "_prefix_left", None)
            policy_on = (prefix <= 0) if prefix is not None else torch.ones_like(dist, dtype=torch.bool)
        gate = (dist > cfg_eff.los_gate_frac * ref_dist) & policy_on
        # stall / hover is v1's job (acro_reward_v1: stall_count, hover_term); it must not gate the LOS term (stalled = speed < stall_speed,
        # the LOS term needs speed > los_min_speed: together they are never true, which silenced the LOS penalty completely)
        prev_rel = (getattr(self, "_prev_target_w", self._desired_pos_w) - self._terrain.env_origins) - prev_pos   # runner: last step's target
        prog = compute_progress_terms(rel=rel, prev_rel=prev_rel, cfg=cfg_eff) * (~terminated).float()
        self._acro_acc["r_progress"] += prog.sum()
        vclose = compute_closing_speed_term(vel=self._robot.data.root_lin_vel_w, rel=rel, active=policy_on & ~terminated, cfg=cfg_eff)
        self._acro_acc["r_vclose"] += vclose.sum()
        los = prog + vclose + compute_los_term(vel=self._robot.data.root_lin_vel_w, rel=rel, gate=gate, cfg=cfg_eff) * (~terminated).float()
        # ---- A: missed pass. closing speed from the same positions as the progress terms ----
        alive = ~terminated
        closing_speed = (torch.linalg.norm(prev_rel, dim=-1) - dist) / step_dt
        miss_pen, passed, self._v2_approach, self._v2_dmin = update_pass_state(
            dist=dist, closing_speed=closing_speed, active=policy_on & alive, approaching=self._v2_approach,
            d_min=self._v2_dmin, cfg=cfg_eff)
        if _is_zero(cfg_eff.k_miss):
            miss_pen = torch.zeros_like(miss_pen)
        # ---- B: near-miss timer (total steps inside near_radius since the takeover); never on v1's ending / timeout step ----
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        self._v2_near = self._v2_near + ((dist < cfg_eff.near_radius) & policy_on & alive).long()
        if near_limit_steps > 0:
            near_term = (self._v2_near >= near_limit_steps) & alive & ~timed_out
        else:
            near_term = torch.zeros_like(terminated)
        near_pen = torch.where(near_term, -_fill(dist, cfg_eff.r_near), torch.zeros_like(dist))

        extra = los + miss_pen + near_pen
        self._cached_reward = self._cached_reward + extra
        self._episode_sums["reward"] += extra
        acc = self._acro_acc
        acc["r_los"] += (los - prog - vclose).sum()
        acc["r_miss"] += miss_pen.sum()
        acc["r_near"] += near_pen.sum()
        acc["passes"] += passed.sum()
        acc["near_misses"] += near_term.sum()
        acc["episodes"] += near_term.sum()                     # v1 counted only its own endings
        acc["final_dist_sum"] += (dist * near_term.float()).sum()
        acc["r_total"] += extra.sum()
        self.extras["term_reasons"]["near_miss"] = near_term
        # v1 cleared its counters only for its own endings (no .any() check: that would sync the GPU every step)
        self._acro_stall = torch.where(near_term, torch.zeros_like(self._acro_stall), self._acro_stall)
        self._acro_away = torch.where(near_term, torch.zeros_like(self._acro_away), self._acro_away)
        return terminated | near_term

    unwrapped_env._compute_dones_and_reward = types.MethodType(_compute_dones_and_reward, unwrapped_env)
    return cfg
