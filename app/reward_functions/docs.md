# reward_functions

## Purpose
Every reward the project trains or evaluates with. The current training reward is **qv2**. `rewards.py` still
provides the shared termination geometry (hit radius, roll/pitch limits, out-of-bounds radius) and the numpy
`reward_func` used by the numpy env for replay and evaluation.

## Files

| File | What it holds | Used by |
|---|---|---|
| `reward_qv2.py` | **Current reward.** qv1 plus a stall ("miss and hover") penalty and termination. | `training/rtl_train_isaac.py` |
| `rewards.py` | Termination constants, numpy `reward_func`, and the older `RewardConfig` / `make_reward_fn` curriculum. | numpy env, Isaac env defaults, eval |

The previous reward, qv1, now lives in `deprecated/reward_functions/reward_qv1.py`; qv2 is a standalone copy of
it plus the stall terms, so its design is still documented below. The qv rewards are pure torch, so the maths can be unit-tested without Isaac. Only `install_reward_qv*()` touches
Isaac: it replaces `_compute_dones_and_reward` **on one env instance**, so the env class and other scripts are
unaffected.

```python
from app.reward_functions.reward_qv2 import RewardQV2Config, install_reward_qv2, pop_reward_stats
install_reward_qv2(env.unwrapped, RewardQV2Config())
...
stats = pop_reward_stats(env.unwrapped)   # python floats accumulated since the last pop, one GPU sync
```

## qv1 (base design, inherited by qv2)
**Why it exists.** With the old `reward_func`, a policy that parks 0.3-0.5 m from the target and times out was
barely punished (about -0.2 discounted). Meanwhile, moving near the target was punished about 60x more than
approaching was rewarded. PPO drifted from "hit" toward "stop and hover", and the hit rate decayed chunk by chunk.

Terms (per policy step, 60 Hz):

| Term | Value | Notes |
|---|---|---|
| hit | `+r_hit` (10), terminal | `dist < HIT_THRESHOLD` (0.25 m). Dominant on purpose: the task is only "hit". |
| progress | `k * (log1p(prev/eps) - log1p(dist/eps))` | Potential difference **without gamma**. The gamma-scaled form pays for standing still near the target. `log1p` makes the last 0.3 m worth as much as the first metres. Total progress available from 10 m is about 5.3, less than `r_hit`. |
| step | `-c_step` | Living cost: arriving sooner is better and hovering is worse. There is no speed or stability penalty near the target, because that is what taught the old policy to stop. |
| timeout | `-r_timeout`, once | Timeouts are true terminals in `rtl_train_isaac` (no bootstrap), so never hitting has a real cost. |
| crash | `-r_crash`, terminal | Out of bounds / underground / NaN / roll > 65 deg / pitch > 80 deg (same geometry as `rewards.py`). |
| tumble | `-min(c_tumble * abs(w)^2, cap)` | Tiny. Discourages spinning without limiting agility. |
| tilt | `-c_tilt * relu(tilt - tilt_soft)^2` | Soft wall before the 65/80 deg cliff, so there is a gradient before termination. |

If a crash and a hit happen on the same step, the crash wins.

## qv2
**Observed failure it fixes.** Handed off at 4-7 m/s, the policy overshot, braked, then parked 0.91 m from the
target with zero action and zero speed for 180 s. That parked state was a fixed point, and qv1 didn't punish it
enough.

Added on top of qv1:
- **stalled** means `dist > hit_threshold and speed < stall_speed (0.3 m/s)`. It is based on state, not action:
  a zero action is a legitimate brake.
- After `hover_grace_s` (1 s) stalled, there is a cost of `-c_hover` per step. After `hover_limit_s` (3 s), the
  episode terminates with `-r_hover`. `r_hover` (5) is deliberately larger than `r_crash`: parking must be worse
  than gambling on a fast approach.
- The counter resets as soon as the drone moves. Brake-and-return is never punished, and moving away from the
  target isn't either, because the drone must be free to turn around after an overshoot.
- In the handoff stage, steps flown by the controller (`_handed_off == False`) never count as stalled.
- A hover termination is a true terminal (no bootstrap), like a crash or a hit.

**Known limitation.** Progress is potential-based, so its total depends only on the start and end distance, not
on the path. A detour such as climbing first and then diving costs only step cost and discounting. That is why
handoff policies can climb before approaching.

## reward_func (rewards.py)
- A flat potential-based reward (Ng et al.), with phi = negative L1 distance divided by `start_dist`. Every knob is
  a module constant.
- `HIT_REWARD` was 1000, about 100-1000x every other term. That produced heavy-tailed advantages that blew up KL
  and collapsed training, so it is now 50.
- Milestone bonuses (+10/+15/+20 at 25/50/75% progress) are calibrated with `APPROACH_MILESTONE_BUDGET`, which was
  measured with the tuned PID. Recalibrate with `control.tune_pid.calibrate_approach_milestone_budget()` if the
  gains or the reward change.
- Anti-oscillation zone terms inside 1 m and 0.5 m are derived in `info/PROJECT_DEFENSE_GUIDE.md` Part 2.4.1.
- `terminal_checks` reports **every** condition that tripped on a step, not only the first one. `reward_func`
  passes a distance-scaled out-of-bounds radius, because a fixed 30 m sphere makes targets beyond about 10 m
  unreachable.
- `GAMMA` (0.97) is exported so the PPO gamma in `training/config.py` can't drift from the shaping gamma.

## Depends on
`numpy`, `scipy`, `torch`. Isaac Lab only inside `install_reward_qv*`.
