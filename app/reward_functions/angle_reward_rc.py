"""Angle-style "remote controlled drone" reward: log closing, constant step cost, zone entry + hold, -15 caught/crash.

Per step: terminal checks -> intercepted (-15) -> closing term -> step cost -> hover/timeout -> zone entry -> hit."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, asdict, fields
from typing import NamedTuple

import torch


from app.reward_functions.proportion import RewardProportionMixin, _col, _fill, _is_zero, resolve_proportion


@dataclass
class RewardRCConfig(RewardProportionMixin):
    tilt_limit_deg: float = 60.0        # angle between body-z and world-up
    roll_limit_deg: float = 85.0
    pitch_limit_deg: float = 120.0
    intercept_radius: float = 0.25      # interceptor within this of the drone = caught
    hit_radius: float = 0.75          # drone within this (total 3D dist) of the base = in zone
    r_hit: float = 45.0                 # TOTAL for entry + hold; completion pays r_hit - r_zone_entry
    r_intercepted: float = 15.0         # paid as -15
    r_crash: float = 15.0               # tilt / roll / pitch / oob / NaN; same cost as being caught
    # "potential": k*(ln(prev+eps) - ln(dist+eps)); sums to k*ln(d0/d_end), no gain from dithering.
    # "log_step":  k*(ln1p(relu(d)/eps) - lambda*ln1p(relu(-d)/eps)) per step; concave, so many small steps pay more.
    closing_mode: str = "potential"
    closing_k: float = 2.0
    closing_eps: float = 0.5
    closing_lambda: float = 1.0         # weight of moving away; > 1 punishes retreating harder than approaching pays
    # per-axis progress: axis_k * sum_i (ln(|prev_rel_i|+eps) - ln(|rel_i|+eps)) over x, y, z of (target - drone).
    # Telescopes per axis (no gain from dithering), and an axis that is already lined up is punished for drifting off it.
    # Retreating on an axis is weighted by closing_lambda. 0 = off (the old closing-only reward).
    axis_k: float = 0.5
    axis_eps: float = 0.5
    c_step: float = 0.01                # constant cost per policy step
    # ---- los ----
    los_k: float = 0.02                 # per-step weight of the line-of-sight term; 0 = off
    los_mode: str = "penalty"           # "penalty": -los_k * m(angle), m = 1 aligned, 2 sideways, 5 straight away; "cos": +los_k * cos
    los_a: float = 35.0                 # m stays 1 up to this angle (deg)
    los_b: float = 90.0                 # the first band reaches its full +1 here, m = 2
    los_c: float = 135.0                # the far band reaches its full +los_far here, m = 5
    los_far: float = 3.0                # extra multiplier past los_b
    los_aligned: float = 1.0            # m when the velocity points straight at the target; 1 = the old flat cost (a step cost in disguise), 0 = free until los_a
    los_min_speed: float = 0.5          # m/s; slower than this the direction is noise, term = 0
    los_gate_frac: float = 0.5          # los term only while dist > this * (dist at handoff); 0 = all the way to the target


    # ---- hover / timeout ----
    stall_speed: float = 0.5            # m/s; slower than this while outside hit_radius = stalled
    hover_grace_s: float = 1.0
    hover_limit_s: float = 2.5          # stalled this long -> terminate with -r_hover
    c_hover: float = 0.05               # per step after the grace
    r_hover: float = 20.0               # > r_crash on purpose: parking must never beat crashing
    r_timeout: float = 10.0
    # ---- zone ----
    zone_time: float = 5 / 60           # consecutive seconds inside hit_radius -> hit; 5 policy steps (was 2 s, which made it brake to hold)
    r_zone_entry: float = 7.0           # paid once per episode on first entry
    # After the hit is scored the episode keeps running this many more policy steps (0 = end on the hit, the old behaviour). In that
    # window closing / axis / los / step cost / hover / timeout are off; crash, out of bounds and being caught still end it with
    # their penalty, so the policy learns to stay flyable after the pass instead of only learning up to the hit.
    post_hit_steps: int = 30
    oob_min_radius: float = 50.0
    oob_start_dist_mult: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_overrides(overrides: dict | None) -> "RewardRCConfig":
        cfg = RewardRCConfig()
        known = {f.name for f in fields(RewardRCConfig)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown reward_rc field {k!r}; known: {sorted(known)}")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


TIMER_FEATURES = 2     # policy inputs appended by the env when cfg.timer_obs: in_zone, zone_count / limit_steps_hit


def zone_timer_features(dist: torch.Tensor, zone_count: torch.Tensor, hit_radius: float, limit_steps_hit: int) -> torch.Tensor:
    """(N, 2): [inside hit_radius now (0/1), consecutive in-zone steps / steps needed for the hit (0..1)]. The hit needs
    the second one to reach 1, so this is the clock the policy has to wait out."""
    in_zone = (dist < hit_radius).float()
    progress = (zone_count.float() / limit_steps_hit).clamp(max=1.0)
    return torch.stack([in_zone, progress], dim=-1)


class RewardRCOut(NamedTuple):
    reward: torch.Tensor        # (N,)
    terminated: torch.Tensor    # (N,) bool: intercepted | crash | hit | hover
    hit: torch.Tensor
    intercepted: torch.Tensor
    oob: torch.Tensor
    tilt_bad: torch.Tensor
    roll_bad: torch.Tensor
    pitch_bad: torch.Tensor
    hover_term: torch.Tensor
    parts: dict                 # name -> (N,) tensor


def _smoothstep(a, b, x):
    """0 below a, 1 above b, S-shaped in between: t = clamp((x - a) / (b - a), 0, 1), S = t * t * (3 - 2t)."""
    t = torch.clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tilt_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Angle (rad, 0..pi) between body-z and world-up."""
    cos_t = 1.0 - 2.0 * (quat_wxyz[:, 1].pow(2) + quat_wxyz[:, 2].pow(2))
    return torch.acos(torch.clamp(cos_t, -1.0, 1.0))


def compute_reward_rc(*, pos: torch.Tensor, dist: torch.Tensor, prev_dist: torch.Tensor, start_dist: torch.Tensor,
                      tilt: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor,
                      interceptor_rel: torch.Tensor | None, timed_out: torch.Tensor, stall_count: torch.Tensor,
                      zone_count: torch.Tensor, first_entry: torch.Tensor,
                      grace_steps: int, limit_steps: int, limit_steps_hit: int,
                      cfg: RewardRCConfig, rel: torch.Tensor | None = None,
                      prev_rel: torch.Tensor | None = None, vel: torch.Tensor | None = None,
                      hit_done: torch.Tensor | None = None, post_hit_count: torch.Tensor | None = None,
                      los_gate: torch.Tensor | None = None) -> RewardRCOut:
    """pos: (N,3) env-local; dist/prev_dist/start_dist: (N,); tilt/roll/pitch: (N,) radians;
    interceptor_rel: (N,3) interceptor minus drone position, or None when there is no interceptor;
    stall_count: (N,) consecutive stalled steps INCLUDING this one;
    zone_count: (N,) consecutive in-zone steps INCLUDING this one;
    first_entry: (N,) bool, first in-zone step of the episode;
    rel / prev_rel: (N,3) target minus drone position now / last step, the axis term is skipped when they are None;
    vel: (N,3) drone linear velocity, the los term is skipped when it (or rel) is None;
    hit_done: (N,) bool, the hit was scored on an EARLIER step (the episode is in its post-hit window), None = never;
    post_hit_count: (N,) steps since the hit was scored, INCLUDING this one (the step after the hit is 1)."""

    # ---- terminal checks ----
    oob_radius = torch.clamp(start_dist * cfg.oob_start_dist_mult, min=cfg.oob_min_radius)
    oob = torch.isnan(pos).any(dim=-1) | (pos[:, 2] < 0.0) | (torch.linalg.norm(pos, dim=-1) > oob_radius)
    tilt_bad = tilt > math.radians(cfg.tilt_limit_deg)
    roll_bad = roll.abs() > math.radians(cfg.roll_limit_deg)
    pitch_bad = pitch.abs() > math.radians(cfg.pitch_limit_deg)
    crash = oob | tilt_bad | roll_bad | pitch_bad

    # ---- intercepted: caught beats everything below ----
    if interceptor_rel is None:
        intercepted = torch.zeros_like(crash)
    else:
        intercepted = torch.linalg.norm(interceptor_rel, dim=-1) <= cfg.intercept_radius
    caught = intercepted & ~crash
    dead = intercepted | crash


    # ---- closing ----
    delta = prev_dist - dist                        # > 0: got closer
    if cfg.closing_mode == "potential":
        closing = cfg.closing_k * (torch.log(prev_dist + cfg.closing_eps) - torch.log(dist + cfg.closing_eps))
        closing = torch.where(closing < 0, closing * cfg.closing_lambda, closing)
    elif cfg.closing_mode == "log_step":
        closing = cfg.closing_k * (torch.log1p(torch.relu(delta) / cfg.closing_eps)
                                   - cfg.closing_lambda * torch.log1p(torch.relu(-delta) / cfg.closing_eps))
    else:
        raise ValueError(f"closing_mode {cfg.closing_mode!r}")


    # ---- per-axis progress ----
    if not _is_zero(cfg.axis_k) and rel is not None:
        axis = _col(cfg.axis_k) * (torch.log(prev_rel.abs() + cfg.axis_eps) - torch.log(rel.abs() + cfg.axis_eps))
        axis = torch.where(axis < 0, axis * cfg.closing_lambda, axis).sum(dim=-1)
    else:
        axis = torch.zeros_like(dist)

    # ---- los: angle between velocity and the line of sight to the target; 0 below los_min_speed ----
    if not _is_zero(cfg.los_k) and rel is not None and vel is not None:
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
        if los_gate is not None:                  # only once handed off and in the far half (see the glue)
            los = los * los_gate.float()
    else:
        los = torch.zeros_like(dist)

    # ---- zone: entry (once), hold, then the post-hit window ----
    if hit_done is None:
        hit_done = torch.zeros_like(crash)
    if post_hit_count is None:
        post_hit_count = torch.zeros_like(zone_count)
    after = hit_done & ~dead                                     # in the window: only crash / out of bounds / caught still matter
    hit = (zone_count >= limit_steps_hit) & ~hit_done & ~dead    # the hit itself, scored once
    window_over = after & (post_hit_count >= cfg.post_hit_steps)
    finished = (hit | window_over) if cfg.post_hit_steps <= 0 else window_over
    zone_entry = first_entry & ~dead & ~hit & ~hit_done
    hover_term = (stall_count >= limit_steps) & ~dead & ~hit & ~after
    live = ~dead & ~hit & ~after                                 # hover / timeout only judge a flight that has not hit yet

    zero = torch.zeros_like(dist)
    off = dead | after                                           # progress terms stop at death and once the hit is scored
    parts = {
        "closing": torch.where(off, zero, closing),
        "axis": torch.where(off, zero, axis),
        "step": torch.where(after, zero, -_fill(dist, cfg.c_step)),
        "zone_entry": torch.where(zone_entry, _fill(dist, cfg.r_zone_entry), zero),
        "hit": torch.where(hit, _fill(dist, cfg.r_hit) - _fill(dist, cfg.r_zone_entry), zero),
        "intercepted": torch.where(caught, -_fill(dist, cfg.r_intercepted), zero),
        "crash": torch.where(crash, -_fill(dist, cfg.r_crash), zero),
        "hover": torch.where(live & (stall_count > grace_steps), -_fill(dist, cfg.c_hover), zero)
                 + torch.where(hover_term, -_fill(dist, cfg.r_hover), zero),
        "timeout": torch.where(timed_out & live & ~hover_term, -_fill(dist, cfg.r_timeout), zero),
        "los": torch.where(off, zero, los),
    }
    reward = sum(parts.values())
    return RewardRCOut(reward=reward, terminated=dead | finished | hover_term, hit=hit, intercepted=caught, oob=oob,
                       tilt_bad=tilt_bad, roll_bad=roll_bad, pitch_bad=pitch_bad, hover_term=hover_term, parts=parts)


# --------------------------------------------------------------------------------------------------
# Isaac glue
# --------------------------------------------------------------------------------------------------

def install_reward_rc(unwrapped_env, cfg: RewardRCConfig | None = None) -> RewardRCConfig:
    """Swap this env instance's _compute_dones_and_reward for the rc reward. The nearest threat comes from
    `env.nearest_threat_rel()` (swept closest approach, None without threats). Training statistics go into qv2's
    accumulators, so the angle trainer's pop_reward_stats works unchanged (progress = closing, crash = crash + caught)."""
    cfg = cfg or RewardRCConfig()
    import isaaclab.utils.math as math_utils
    from app.reward_functions.angle_reward_qv2 import _new_acc

    unwrapped_env._qv2_acc = _new_acc(unwrapped_env.device)   # names shared with qv2: the angle trainer reads _qv2_acc / _qv2_stall

    unwrapped_env._rc_cfg = cfg
    unwrapped_env._qv2_stall = torch.zeros(unwrapped_env.num_envs, dtype=torch.long, device=unwrapped_env.device)
    # _zone_count / _zone_entered live on the env (zeroed in its _reset_idx, so the terminal observation of a truncated
    # episode still shows the timer); the observation reads them together with _zone_radius / _zone_limit
    unwrapped_env._zone_count.zero_()
    unwrapped_env._zone_entered.zero_()
    unwrapped_env._hit_done.zero_()           # post-hit window state; zeroed per env in the env's _reset_idx too
    unwrapped_env._post_hit.zero_()

    step_dt = float(getattr(unwrapped_env, "step_dt", 1.0 / 60.0))   # seconds per step

    # hover: seconds -> steps
    grace_steps = max(1, int(round(cfg.hover_grace_s / step_dt)))
    limit_steps = max(grace_steps + 1, int(round(cfg.hover_limit_s / step_dt)))

    # zone: seconds -> steps
    limit_steps_hit = max(1, int(round(cfg.zone_time / step_dt)))
    unwrapped_env._zone_radius = cfg.hit_radius
    unwrapped_env._zone_limit = limit_steps_hit

    def _compute_dones_and_reward(self):
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        quat = self._robot.data.root_quat_w
        roll, pitch, _yaw = math_utils.euler_xyz_from_quat(quat)
        roll = math_utils.wrap_to_pi(roll)
        pitch = math_utils.wrap_to_pi(pitch)
        target_local = self._desired_pos_w - self._terrain.env_origins
        dist = torch.linalg.norm(target_local - pos, dim=-1)
        speed = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=-1)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        # reference leg for auto_scale (and the LOS gate): the handoff distance h, else the start distance
        handoff_h = getattr(self, "_handoff_h", None)
        if getattr(self, "_handoff_snap_enabled", False) and handoff_h is not None:
            ref_dist = torch.where(handoff_h > 0, handoff_h, self._start_dist)
        else:
            ref_dist = self._start_dist
        cfg_eff = resolve_proportion(self._rc_cfg, ref_dist, step_dt)      # one leg = one unit of reward


        # hover counter (only outside the zone)
        stalled = (dist > cfg_eff.hit_radius) & (speed < cfg_eff.stall_speed)
        handed = getattr(self, "_handed_off", None)     # handoff stage: only count stalls once the policy flies
        if handed is not None:
            stalled = stalled & handed
        self._qv2_stall = (self._qv2_stall + 1) * stalled.long()

        # zone counter: read the entry flag BEFORE updating it
        in_zone = dist < cfg_eff.hit_radius
        first_entry = in_zone & ~self._zone_entered
        self._zone_count = (self._zone_count + 1) * in_zone.long()
        self._zone_entered = self._zone_entered | in_zone

        icpt_rel = self.nearest_threat_rel() if hasattr(self, "nearest_threat_rel") else None

        # post-hit window: `_hit_done` was set on an earlier step; `post_hit` counts the steps since, including this one
        hit_done = self._hit_done
        post_hit = self._post_hit + hit_done.long()

        # line-of-sight penalty: only after the policy takes over, and only in the far part (cfg.los_gate_frac) of the distance it had at handoff
        los_gate = dist > cfg_eff.los_gate_frac * ref_dist
        if handed is not None:
            los_gate = los_gate & handed

        out = compute_reward_rc(pos=pos, dist=dist, prev_dist=self._prev_distance, start_dist=self._start_dist,
                                tilt=tilt_from_quat(quat), roll=roll, pitch=pitch, interceptor_rel=icpt_rel,
                                timed_out=timed_out, stall_count=self._qv2_stall,
                                rel=target_local - pos, prev_rel=target_local - self._prev_position,
                                vel=self._robot.data.root_lin_vel_w,
                                zone_count=self._zone_count, first_entry=first_entry,
                                grace_steps=grace_steps, limit_steps=limit_steps,
                                limit_steps_hit=limit_steps_hit, cfg=cfg_eff, hit_done=hit_done, post_hit_count=post_hit,
                                los_gate=los_gate)
        self._hit_done = hit_done | out.hit
        self._post_hit = post_hit

        # tilt crashes are reported as "roll" so the trainer's eval (which only knows roll/pitch) classifies them
        self.extras["term_reasons"] = {"hit": out.hit, "intercepted": out.intercepted, "oob": out.oob,
                                       "attitude_roll": out.roll_bad | out.tilt_bad,
                                       "attitude_pitch": out.pitch_bad, "hover": out.hover_term}
        ended = out.terminated | timed_out
        self._qv2_stall = torch.where(ended, torch.zeros_like(self._qv2_stall), self._qv2_stall)
        continuing = ~out.terminated
        self._prev_position = torch.where(continuing.unsqueeze(-1), pos, self._prev_position)
        self._prev_distance = torch.where(continuing, dist, self._prev_distance)
        self._cached_reward = out.reward
        self._episode_sums["reward"] += out.reward
        self._episode_sums["distance_to_goal"] += dist

        acc = self._qv2_acc
        acc["steps"] += dist.numel()
        acc["episodes"] += ended.sum()
        acc["hits"] += out.hit.sum()
        acc["caught"] += out.intercepted.sum()
        if getattr(self, "_K", 0):          # by chaser geometry, over the episodes that end now (bucket -1 = ended before the chasers spawned)
            bucket = self.threat_bucket()
            for b in range(len(self.BUCKET_NAMES)):
                in_b = bucket == b
                acc[f"eps_b{b}"] += (ended & in_b).sum()
                acc[f"hits_b{b}"] += (out.hit & in_b).sum()            # counted when the hit happens, like acc["hits"]: the episode ends post_hit_steps later
                acc[f"caught_b{b}"] += (out.intercepted & in_b).sum()
                acc[f"att_b{b}"] += ((out.roll_bad | out.tilt_bad | out.pitch_bad) & ~out.oob & in_b).sum()
                acc[f"oob_b{b}"] += (out.oob & in_b).sum()
                acc[f"timeouts_b{b}"] += (timed_out & ~out.terminated & in_b).sum()
                acc[f"hovers_b{b}"] += (out.hover_term & in_b).sum()
        acc["oob"] += out.oob.sum()
        acc["roll"] += ((out.roll_bad | out.tilt_bad) & ~out.oob).sum()
        acc["pitch"] += (out.pitch_bad & ~out.oob & ~out.roll_bad & ~out.tilt_bad).sum()
        acc["timeouts"] += (timed_out & ~out.terminated).sum()
        acc["hovers"] += out.hover_term.sum()
        acc["hit_steps_sum"] += (self.episode_length_buf.float() * out.hit.float()).sum()
        acc["final_dist_sum"] += (dist * ended.float()).sum()
        acc["speed_sum"] += speed.sum()
        acc["tilt_sum"] += tilt_from_quat(quat).sum()
        acc["near_steps"] += (dist < 1.0).sum()
        acc["near_speed_sum"] += (speed * (dist < 1.0).float()).sum()
        acc["stall_steps"] += stalled.sum()
        acc["r_progress"] += (out.parts["closing"] + out.parts["axis"] + out.parts["los"]).sum()
        acc["r_step"] += out.parts["step"].sum()
        acc["r_hit"] += (out.parts["hit"] + out.parts["zone_entry"]).sum()
        acc["r_crash"] += (out.parts["crash"] + out.parts["intercepted"]).sum()
        acc["r_timeout"] += out.parts["timeout"].sum()
        acc["r_hover"] += out.parts["hover"].sum()
        acc["r_total"] += out.reward.sum()
        return out.terminated

    unwrapped_env._compute_dones_and_reward = types.MethodType(_compute_dones_and_reward, unwrapped_env)
    return cfg