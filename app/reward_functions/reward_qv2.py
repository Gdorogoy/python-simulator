"""Isaac reward qv2: qv1 plus a stall ("miss and hover") penalty and termination. See docs.md "qv1" and "qv2"."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, asdict, fields
from typing import NamedTuple

import torch

from app.reward_functions.rewards import (
    ATTITUDE_PITCH_DEG,
    ATTITUDE_ROLL_DEG,
    HIT_THRESHOLD,
    OOB_RADIUS,
)


@dataclass
class RewardQV2Config:
    hit_threshold: float = HIT_THRESHOLD   # keep == PyBullet eval's radius
    r_hit: float = 10.0
    r_crash: float = 5.0
    r_timeout: float = 3.0
    progress_k: float = 2.0
    progress_eps: float = 0.5
    c_step: float = 0.002
    c_tumble: float = 5e-4
    tumble_cap: float = 0.05
    c_tilt: float = 2.0
    tilt_soft_rad: float = 0.8             # ~46 deg; the hard cliffs are 65 (roll) / 80 (pitch) deg
    attitude_roll_deg: float = ATTITUDE_ROLL_DEG
    attitude_pitch_deg: float = ATTITUDE_PITCH_DEG
    oob_min_radius: float = OOB_RADIUS
    oob_start_dist_mult: float = 3.0
    # ---- miss + hover (new in qv2) ----
    stall_speed: float = 0.3               # m/s; slower than this while off-target counts as "stalled"
    hover_grace_s: float = 1.0             # stalled this long before the per-step penalty starts
    hover_limit_s: float = 3.0             # stalled this long -> terminate with r_hover
    c_hover: float = 0.05                  # per policy step after the grace (0.05/step * 60 Hz * 2 s = 6)
    r_hover: float = 5.0                   # one-off terminal cost; > r_crash on purpose (docs.md "qv2")
    # Diagnostic zone for the "stalled near the target" statistic (not used in the reward itself).
    stall_inner: float = 0.25
    stall_outer: float = 0.75
    diag_stall_speed: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_overrides(overrides: dict | None) -> "RewardQV2Config":
        cfg = RewardQV2Config()
        known = {f.name for f in fields(RewardQV2Config)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown reward_qv2 field {k!r}; known: {sorted(known)}")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


class RewardOut(NamedTuple):
    reward: torch.Tensor        # (N,)
    terminated: torch.Tensor    # (N,) bool: crash | hit | hover (timeouts are reported by the env separately)
    hit: torch.Tensor           # (N,) bool
    oob: torch.Tensor           # (N,) bool
    roll_bad: torch.Tensor      # (N,) bool
    pitch_bad: torch.Tensor     # (N,) bool
    hover_term: torch.Tensor    # (N,) bool
    parts: dict                 # name -> (N,) tensor, each already masked to where it applies


def potential(dist: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log1p(dist / eps)


def compute_reward_qv2(*, pos: torch.Tensor, dist: torch.Tensor, prev_dist: torch.Tensor,
                       start_dist: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor,
                       ang_vel: torch.Tensor, timed_out: torch.Tensor, stall_count: torch.Tensor,
                       grace_steps: int, limit_steps: int, cfg: RewardQV2Config) -> RewardOut:
    """pos: (N,3) env-local position; dist/prev_dist/start_dist: (N,); roll/pitch: (N,) radians;
    ang_vel: (N,3); timed_out: (N,) bool; stall_count: (N,) consecutive stalled steps INCLUDING this one."""

    oob_radius = torch.clamp(start_dist * cfg.oob_start_dist_mult, min=cfg.oob_min_radius)
    oob = torch.isnan(pos).any(dim=-1) | (pos[:, 2] < 0.0) | (torch.linalg.norm(pos, dim=-1) > oob_radius)
    roll_bad = roll.abs() > math.radians(cfg.attitude_roll_deg)
    pitch_bad = pitch.abs() > math.radians(cfg.attitude_pitch_deg)
    crash = oob | roll_bad | pitch_bad
    hit = (dist < cfg.hit_threshold) & ~crash
    hover_term = (stall_count >= limit_steps) & ~crash & ~hit

    zero = torch.zeros_like(dist)
    progress = cfg.progress_k * (potential(prev_dist, cfg.progress_eps) - potential(dist, cfg.progress_eps))
    step_cost = torch.full_like(dist, -cfg.c_step)
    tumble = -torch.clamp(cfg.c_tumble * ang_vel.pow(2).sum(dim=-1), max=cfg.tumble_cap)
    tilt = torch.maximum(roll.abs(), pitch.abs())
    tilt_pen = -cfg.c_tilt * torch.relu(tilt - cfg.tilt_soft_rad).pow(2)
    timeout_pen = torch.where(timed_out & ~crash & ~hit & ~hover_term, torch.full_like(dist, -cfg.r_timeout), zero)
    hover_pen = torch.where(stall_count > grace_steps, torch.full_like(dist, -cfg.c_hover), zero)
    hover_end = torch.where(hover_term, torch.full_like(dist, -cfg.r_hover), zero)

    live = ~crash
    parts = {
        "progress": torch.where(live, progress, zero),
        "step": torch.where(live & ~hit, step_cost, zero),
        "tumble": torch.where(live, tumble, zero),
        "tilt": torch.where(live, tilt_pen, zero),
        "hit": torch.where(hit, torch.full_like(dist, cfg.r_hit), zero),
        "crash": torch.where(crash, torch.full_like(dist, -cfg.r_crash), zero),
        "timeout": timeout_pen,
        "hover": torch.where(live & ~hit, hover_pen, zero) + hover_end,
    }
    reward = sum(parts.values())
    return RewardOut(reward=reward, terminated=crash | hit | hover_term, hit=hit, oob=oob, roll_bad=roll_bad,
                     pitch_bad=pitch_bad, hover_term=hover_term, parts=parts)


# --------------------------------------------------------------------------------------------------
# Isaac glue
# --------------------------------------------------------------------------------------------------

_STAT_KEYS = (
    "steps", "episodes", "hits", "oob", "roll", "pitch", "timeouts", "hovers", "hit_steps_sum", "final_dist_sum",
    "speed_sum", "tilt_sum", "near_steps", "near_speed_sum", "stall_steps",
    "r_progress", "r_step", "r_tumble", "r_tilt", "r_hit", "r_crash", "r_timeout", "r_hover", "r_total",
)


def _new_acc(device) -> dict:
    return {k: torch.zeros((), device=device) for k in _STAT_KEYS}


def pop_reward_stats(unwrapped_env) -> dict:
    """Per-chunk training-env statistics accumulated since the last call (python floats); resets them.
    One GPU->CPU sync per call, not per step."""
    acc = unwrapped_env._qv2_acc
    vals = torch.stack([acc[k] for k in _STAT_KEYS]).tolist()
    for k in _STAT_KEYS:
        acc[k].zero_()
    s = dict(zip(_STAT_KEYS, vals))
    steps = max(s["steps"], 1.0)
    eps = s["episodes"] if s["episodes"] > 0 else float("nan")   # no finished episode -> rates are NaN, not a fake 0.000
    dt = float(getattr(unwrapped_env, "step_dt", 1.0 / 60.0))
    out = {
        "train_episodes": s["episodes"],
        "train_hit_rate": s["hits"] / eps,
        "train_oob_rate": s["oob"] / eps,
        "train_roll_rate": s["roll"] / eps,
        "train_pitch_rate": s["pitch"] / eps,
        "train_timeout_rate": s["timeouts"] / eps,
        "train_hover_rate": s["hovers"] / eps,
        "train_hit_time_s": (s["hit_steps_sum"] / s["hits"] * dt) if s["hits"] > 0 else float("nan"),
        "train_final_dist": s["final_dist_sum"] / eps,
        "train_mean_speed": s["speed_sum"] / steps,
        "train_mean_tilt_rad": s["tilt_sum"] / steps,
        "train_near_frac": s["near_steps"] / steps,
        "train_near_speed": (s["near_speed_sum"] / s["near_steps"]) if s["near_steps"] > 0 else float("nan"),
        "train_stall_frac": s["stall_steps"] / steps,
    }
    for name in ("progress", "step", "tumble", "tilt", "hit", "crash", "timeout", "hover", "total"):
        out[f"rew_{name}"] = s[f"r_{name}"] / steps          # mean per step per env
    return out


def install_reward_qv2(unwrapped_env, cfg: RewardQV2Config | None = None) -> RewardQV2Config:
    """Swap this env instance's _compute_dones_and_reward for qv2; adds the `_qv2_stall` counter."""
    cfg = cfg or RewardQV2Config()
    import isaaclab.utils.math as math_utils  # here so the module stays importable without Kit

    unwrapped_env._qv2_cfg = cfg
    unwrapped_env._qv2_acc = _new_acc(unwrapped_env.device)
    unwrapped_env._qv2_stall = torch.zeros(unwrapped_env.num_envs, dtype=torch.long, device=unwrapped_env.device)
    step_dt = float(getattr(unwrapped_env, "step_dt", 1.0 / 60.0))
    grace_steps = max(1, int(round(cfg.hover_grace_s / step_dt)))
    limit_steps = max(grace_steps + 1, int(round(cfg.hover_limit_s / step_dt)))

    def _compute_dones_and_reward(self):
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        roll, pitch, _yaw = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        target_local = self._desired_pos_w - self._terrain.env_origins
        dist = torch.linalg.norm(target_local - pos, dim=-1)
        vel = self._robot.data.root_lin_vel_w
        speed = torch.linalg.norm(vel, dim=-1)
        ang_vel = self._robot.data.root_ang_vel_b
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        stalled = (dist > cfg.hit_threshold) & (speed < cfg.stall_speed)
        handed = getattr(self, "_handed_off", None)     # handoff stage: only count stalls once the policy flies
        if handed is not None:
            stalled = stalled & handed
        else:
            prefix = getattr(self, "_prefix_left", None)
            if prefix is not None:
                stalled = stalled & (prefix <= 0)
        self._qv2_stall = (self._qv2_stall + 1) * stalled.long()

        out = compute_reward_qv2(pos=pos, dist=dist, prev_dist=self._prev_distance, start_dist=self._start_dist,
                                 roll=roll, pitch=pitch, ang_vel=ang_vel, timed_out=timed_out,
                                 stall_count=self._qv2_stall, grace_steps=grace_steps, limit_steps=limit_steps,
                                 cfg=cfg)

        self.extras["term_reasons"] = {"hit": out.hit, "oob": out.oob, "attitude_roll": out.roll_bad,
                                       "attitude_pitch": out.pitch_bad, "hover": out.hover_term}

        ended = out.terminated | timed_out
        self._qv2_stall = torch.where(ended, torch.zeros_like(self._qv2_stall), self._qv2_stall)

        continuing = ~out.terminated
        self._prev_position = torch.where(continuing.unsqueeze(-1), pos, self._prev_position)
        self._prev_distance = torch.where(continuing, dist, self._prev_distance)

        self._cached_reward = out.reward
        self._episode_sums["reward"] += out.reward
        self._episode_sums["distance_to_goal"] += dist

        # ---- statistics (GPU accumulators; popped once per chunk) ----
        acc = self._qv2_acc
        timeout_only = timed_out & ~out.terminated
        near = dist < 1.0
        diag_stall = (dist > cfg.stall_inner) & (dist < cfg.stall_outer) & (speed < cfg.diag_stall_speed)
        acc["steps"] += dist.numel()
        acc["episodes"] += ended.sum()
        acc["hits"] += out.hit.sum()
        acc["oob"] += out.oob.sum()
        acc["roll"] += (out.roll_bad & ~out.oob).sum()
        acc["pitch"] += (out.pitch_bad & ~out.oob & ~out.roll_bad).sum()
        acc["timeouts"] += timeout_only.sum()
        acc["hovers"] += out.hover_term.sum()
        acc["hit_steps_sum"] += (self.episode_length_buf.float() * out.hit.float()).sum()
        acc["final_dist_sum"] += (dist * ended.float()).sum()
        acc["speed_sum"] += speed.sum()
        acc["tilt_sum"] += torch.maximum(roll.abs(), pitch.abs()).sum()
        acc["near_steps"] += near.sum()
        acc["near_speed_sum"] += (speed * near.float()).sum()
        acc["stall_steps"] += diag_stall.sum()
        for name, part in out.parts.items():
            acc[f"r_{name}"] += part.sum()
        acc["r_total"] += out.reward.sum()

        return out.terminated

    unwrapped_env._compute_dones_and_reward = types.MethodType(_compute_dones_and_reward, unwrapped_env)
    return cfg
