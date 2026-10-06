"""Isaac reward qv1: hit / log-potential progress / step / timeout / crash / tumble / tilt. See app/reward_functions/docs.md "qv1"."""

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
class RewardQV1Config:
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
    # Diagnostic zone for the "stalled near the target" statistic (not used in the reward itself).
    stall_inner: float = 0.25
    stall_outer: float = 0.75
    stall_speed: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_overrides(overrides: dict | None) -> "RewardQV1Config":
        cfg = RewardQV1Config()
        known = {f.name for f in fields(RewardQV1Config)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown reward_qv1 field {k!r}; known: {sorted(known)}")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


class RewardOut(NamedTuple):
    reward: torch.Tensor        # (N,)
    terminated: torch.Tensor    # (N,) bool: crash | hit  (timeouts are reported by the env separately)
    hit: torch.Tensor           # (N,) bool
    oob: torch.Tensor           # (N,) bool
    roll_bad: torch.Tensor      # (N,) bool
    pitch_bad: torch.Tensor     # (N,) bool
    parts: dict                 # name -> (N,) tensor, each already masked to where it applies


def potential(dist: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log1p(dist / eps)


def compute_reward_qv1(*, pos: torch.Tensor, dist: torch.Tensor, prev_dist: torch.Tensor,
                       start_dist: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor,
                       ang_vel: torch.Tensor, timed_out: torch.Tensor, cfg: RewardQV1Config) -> RewardOut:


    """pos: (N,3) env-local position; dist/prev_dist/start_dist: (N,); roll/pitch: (N,) radians;
    ang_vel: (N,3); timed_out: (N,) bool (episode hits its length cap on this step)."""


    oob_radius = torch.clamp(start_dist * cfg.oob_start_dist_mult, min=cfg.oob_min_radius)
    oob = torch.isnan(pos).any(dim=-1) | (pos[:, 2] < 0.0) | (torch.linalg.norm(pos, dim=-1) > oob_radius)
    roll_bad = roll.abs() > math.radians(cfg.attitude_roll_deg)
    pitch_bad = pitch.abs() > math.radians(cfg.attitude_pitch_deg)
    crash = oob | roll_bad | pitch_bad
    hit = (dist < cfg.hit_threshold) & ~crash

    progress = cfg.progress_k * (potential(prev_dist, cfg.progress_eps) - potential(dist, cfg.progress_eps))
    step_cost = torch.full_like(dist, -cfg.c_step)
    tumble = -torch.clamp(cfg.c_tumble * ang_vel.pow(2).sum(dim=-1), max=cfg.tumble_cap)
    tilt = torch.maximum(roll.abs(), pitch.abs())
    tilt_pen = -cfg.c_tilt * torch.relu(tilt - cfg.tilt_soft_rad).pow(2)
    timeout_pen = torch.where(timed_out & ~crash & ~hit, torch.full_like(dist, -cfg.r_timeout),
                              torch.zeros_like(dist))

    live = ~crash
    parts = {
        "progress": torch.where(live, progress, torch.zeros_like(dist)),
        "step": torch.where(live & ~hit, step_cost, torch.zeros_like(dist)),
        "tumble": torch.where(live, tumble, torch.zeros_like(dist)),
        "tilt": torch.where(live, tilt_pen, torch.zeros_like(dist)),
        "hit": torch.where(hit, torch.full_like(dist, cfg.r_hit), torch.zeros_like(dist)),
        "crash": torch.where(crash, torch.full_like(dist, -cfg.r_crash), torch.zeros_like(dist)),
        "timeout": timeout_pen,
    }
    reward = sum(parts.values())
    return RewardOut(reward=reward, terminated=crash | hit, hit=hit, oob=oob, roll_bad=roll_bad,
                     pitch_bad=pitch_bad, parts=parts)


# --------------------------------------------------------------------------------------------------
# Isaac glue
# --------------------------------------------------------------------------------------------------

_STAT_KEYS = (
    "steps", "episodes", "hits", "oob", "roll", "pitch", "timeouts", "hit_steps_sum", "final_dist_sum",
    "speed_sum", "tilt_sum", "near_steps", "near_speed_sum", "stall_steps",
    "r_progress", "r_step", "r_tumble", "r_tilt", "r_hit", "r_crash", "r_timeout", "r_total",
)


def _new_acc(device) -> dict:
    return {k: torch.zeros((), device=device) for k in _STAT_KEYS}


def pop_reward_stats(unwrapped_env) -> dict:

    """Returns the per-chunk training-env statistics accumulated since the last call (as python floats)
    and resets them. One GPU->CPU sync per call, not per step."""

    acc = unwrapped_env._qv1_acc
    vals = torch.stack([acc[k] for k in _STAT_KEYS]).tolist()
    for k in _STAT_KEYS:
        acc[k].zero_()
    s = dict(zip(_STAT_KEYS, vals))
    steps = max(s["steps"], 1.0)
    eps = max(s["episodes"], 1.0)
    dt = float(getattr(unwrapped_env, "step_dt", 1.0 / 60.0))
    out = {
        "train_episodes": s["episodes"],
        "train_hit_rate": s["hits"] / eps,
        "train_oob_rate": s["oob"] / eps,
        "train_roll_rate": s["roll"] / eps,
        "train_pitch_rate": s["pitch"] / eps,
        "train_timeout_rate": s["timeouts"] / eps,
        "train_hit_time_s": (s["hit_steps_sum"] / s["hits"] * dt) if s["hits"] > 0 else float("nan"),
        "train_final_dist": s["final_dist_sum"] / eps,
        "train_mean_speed": s["speed_sum"] / steps,
        "train_mean_tilt_rad": s["tilt_sum"] / steps,
        "train_near_frac": s["near_steps"] / steps,
        "train_near_speed": (s["near_speed_sum"] / s["near_steps"]) if s["near_steps"] > 0 else float("nan"),
        "train_stall_frac": s["stall_steps"] / steps,
    }
    for name in ("progress", "step", "tumble", "tilt", "hit", "crash", "timeout", "total"):
        out[f"rew_{name}"] = s[f"r_{name}"] / steps          # mean per step per env
    return out


def install_reward_qv1(unwrapped_env, cfg: RewardQV1Config | None = None) -> RewardQV1Config:
    """Swap this env instance's _compute_dones_and_reward for qv1 (same bookkeeping as the original)."""
    cfg = cfg or RewardQV1Config()
    # Imported here so this module stays importable without Kit (unit tests).
    import isaaclab.utils.math as math_utils

    unwrapped_env._qv1_cfg = cfg
    unwrapped_env._qv1_acc = _new_acc(unwrapped_env.device)

    def _compute_dones_and_reward(self):
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        roll, pitch, _yaw = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        target_local = self._desired_pos_w - self._terrain.env_origins
        dist = torch.linalg.norm(target_local - pos, dim=-1)
        vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        out = compute_reward_qv1(pos=pos, dist=dist, prev_dist=self._prev_distance, start_dist=self._start_dist,
                                 roll=roll, pitch=pitch, ang_vel=ang_vel, timed_out=timed_out, cfg=cfg)

        self.extras["term_reasons"] = {"hit": out.hit, "oob": out.oob, "attitude_roll": out.roll_bad,
                                       "attitude_pitch": out.pitch_bad}

        continuing = ~out.terminated
        self._prev_position = torch.where(continuing.unsqueeze(-1), pos, self._prev_position)
        self._prev_distance = torch.where(continuing, dist, self._prev_distance)

        self._cached_reward = out.reward
        self._episode_sums["reward"] += out.reward
        self._episode_sums["distance_to_goal"] += dist

        # ---- statistics (GPU accumulators; popped once per chunk) ----
        acc = self._qv1_acc
        speed = torch.linalg.norm(vel, dim=-1)
        ended = out.terminated | timed_out
        timeout_only = timed_out & ~out.terminated
        near = dist < 1.0
        stall = (dist > cfg.stall_inner) & (dist < cfg.stall_outer) & (speed < cfg.stall_speed)
        acc["steps"] += dist.numel()
        acc["episodes"] += ended.sum()
        acc["hits"] += out.hit.sum()
        acc["oob"] += (out.oob).sum()
        acc["roll"] += (out.roll_bad & ~out.oob).sum()
        acc["pitch"] += (out.pitch_bad & ~out.oob & ~out.roll_bad).sum()
        acc["timeouts"] += timeout_only.sum()
        acc["hit_steps_sum"] += (self.episode_length_buf.float() * out.hit.float()).sum()
        acc["final_dist_sum"] += (dist * ended.float()).sum()
        acc["speed_sum"] += speed.sum()
        acc["tilt_sum"] += torch.maximum(roll.abs(), pitch.abs()).sum()
        acc["near_steps"] += near.sum()
        acc["near_speed_sum"] += (speed * near.float()).sum()
        acc["stall_steps"] += stall.sum()
        for name, part in out.parts.items():
            acc[f"r_{name}"] += part.sum()
        acc["r_total"] += out.reward.sum()

        return out.terminated

    unwrapped_env._compute_dones_and_reward = types.MethodType(_compute_dones_and_reward, unwrapped_env)
    return cfg
