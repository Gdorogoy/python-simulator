# reward_functions

## Purpose
Every reward the project trains or evaluates with. The angle trainer's default is **qv2** (`--reward qv2`); the recent
angle runs (for example `angle_los_fixed_v5_15_50`) use **rc** (`--reward rc`). The acro trainer defaults to **acro_v1**;
the 1000-1500 m speed runs use **acro_v3** (`--reward v3`, v1 + rc progress/LOS + the speed / cross-track / attitude
terms) or **acro_v4** (`--reward v4`, v3 + the swept hit test, cross-track per metre with look-ahead, and the terminal
straightness bonus). `rewards.py` still
provides the shared termination geometry (hit radius, roll/pitch limits, out-of-bounds radius) and the numpy
`reward_func` used by the numpy env for replay and evaluation.

## Files

| File | What it holds | Used by |
|---|---|---|
| `angle_reward_qv2.py` | **Current reward.** qv1 plus a stall ("miss and hover") penalty and termination. | `training/angle_rtl_train_isaac.py` |
| `angle_reward_rc.py` | **Remote-controlled drone reward** for the angle drone with threats (interceptors): log-potential closing, +45 hit inside 0.75 m (then 30 more steps of flight), -15 caught / crash, hover + timeout penalties. Select with `--reward rc`. | `training/angle_rtl_train_isaac.py` |
| `acro_reward_v1.py` | **Acro reward.** qv2's task terms with a quaternion tilt limit that is widened stage by stage. | `training/acro_rtl_train_isaac.py` |
| `acro_reward_v2.py` | **Acro v2.** v1 plus the angle rc progress terms and the line-of-sight penalty (see "acro_v2"). | `training/acro_rtl_train_isaac.py` (`--reward v2`) |
| `acro_reward_v3.py` | **Acro v3.** v2 plus the terms that pay for speed and a straight line: brake-limited target-speed tracking, cross-track, soft attitude wall (see "acro_v3"). | `training/acro_rtl_train_isaac.py` (`--reward v3`) |
| `acro_reward_v4.py` | **Acro v4.** v3 plus the swept hit test, cross-track charged per METRE with a look-ahead, and the terminal straightness bonus (see "acro_v4"). | `training/acro_rtl_train_isaac.py` (`--reward v4`) |
| `proportion.py` | **Per-leg normalisation** shared by every reward (`auto_scale`): one leg = one unit of reward at any distance. See "Reward proportions" below. | the `install_*` functions in the reward modules |
| `rewards.py` | Termination constants, numpy `reward_func`, and the older `RewardConfig` / `make_reward_fn` curriculum. | numpy env, Isaac env defaults, eval |

The previous reward, qv1, now lives in `deprecated/reward_functions/reward_qv1.py`; qv2 is a standalone copy of
it plus the stall terms, so its design is still documented below. The qv rewards are pure torch, so the maths can be unit-tested without Isaac. Only `install_reward_qv*()` touches
Isaac: it replaces `_compute_dones_and_reward` **on one env instance**, so the env class and other scripts are
unaffected.

```python
from app.reward_functions.angle_reward_qv2 import RewardQV2Config, install_reward_qv2, pop_reward_stats
install_reward_qv2(env.unwrapped, RewardQV2Config())
...
stats = pop_reward_stats(env.unwrapped)   # python floats accumulated since the last pop, one GPU sync
```

## Reward proportions (per-leg normalisation)
**Why.** Every absolute reward number (`c_step` 0.002, `r_hit` 10, `r_crash` 7, ...) was tuned on one curriculum
stage and silently changes meaning on another. A 250 m leg and a 1500 m leg paid the same `r_hit` and `r_crash`, so a
crash was 6x cheaper in "per-metre" terms on the long leg, while the step cost either dominated (long leg, ~50 for a
hit's worth of flying) or vanished (short leg). That is one reason `train_hit` plateaued around 0.78 with ~15 % of
episodes ending far from the target: the reward told the policy that giving up was cheap. `RewardProportionMixin`
(`proportion.py`) defines every reward in per-leg units instead, and the `install_*` glue derives the absolute fields
from the live leg length, so the same config is meaningful at 250 m or 1500 m.

**One leg = one unit.** `reference_speed` is the physical limit of the acro drone (55 m/s), so

    N(D) = clamp(D / (reference_speed * dt), min_leg_s / dt, max_leg_s / dt)      # steps in one leg

- a **per-step** coefficient becomes `*_leg / N`, so a term that pays its coefficient on every step of a flown leg
  totals exactly `*_leg`;
- a **terminal / one-off** term becomes the `*_leg` value itself, because it is paid once;
- a coefficient that is **0 stays 0** ("0 = off"), so optional shaping terms are never switched on by the scaling;
- `k_miss` is per metre: `k_miss = miss_leg / miss_radius`, so a pass at the edge of `miss_radius` costs `miss_leg`
  and a perfect pass costs nothing.

`D` is the policy's own leg: the handoff distance `h` in `--handoff` mode, else the start distance. It is read every
env step from the env, so a mixed-distance run gets per-env coefficients. `auto_scale=False` returns the config
untouched (the old absolute numbers), for reproducing old runs.

Because `N` is the flight time at the **physical limit** (55 m/s), the step cost only breaks even for a near-limit
approach: at 1500 m `N = 1636` steps ≈ 27 s, so a 30 s run already pays 3 steps' worth of extra living cost per second.
The `min_leg_s` floor keeps short legs from getting an impossibly tight budget (250 m would be 4.5 s at the limit; it is
scored against 7.5 s of settling time instead).

**The time bonus and its clock.** `r_time_bonus` is paid on a hit, scaled by the fraction of its budget
left. With `auto_scale` that budget is `N`, which no real flight beats (the drone starts at rest, so a 1250 m leg takes
~2500 steps against `N = 1364`) — `time_left` clamps to 0 and the bonus was **always 0**. Two fields replace the clock:

- `time_bonus_budget_s`: one fixed clock for every episode (e.g. 62.5, the episode cap). A hit at 30 s pays ~52 % of
  `r_time_bonus`, a hit at the cap pays 0.
- `time_bonus_speed`: a **distance-dependent** clock, `budget_s = clamp(start_distance / time_bonus_speed, min_leg_s,
  max_leg_s)`, recomputed per episode. Use it with a distance curriculum: a fixed clock saturates at short range (a
  150 m hit pays nearly the full bonus however it flies). `time_bonus_speed = 30` reproduces a fixed 40 s at ~1200 m
  and gives a 150 m leg 7.5 s (the floor). It takes priority over `time_bonus_budget_s`.

Default 0 / 0 keeps the per-leg budget (the old behaviour), so nothing changes unless you opt in.

**Defaults.** Per-step accruals: `step_leg` 5, `hover_accrual_leg` 5, `away_leg` 3, `decel_leg` 3, `tumble_leg` 2,
`tilt_leg` 5, `progress_leg` 10, `closing_leg` 15, `axis_leg` 15, `los_leg` 3, `close_speed_leg` 10. One-off
terminals: `hit_leg` 50, `hover_leg` 45, `crash_leg` 45, `timeout_leg` 45, `escape_leg` 45, `near_leg` 45,
`intercepted_leg` 45, `zone_entry_leg` 7, `path_leg` 0, `miss_leg` 30. Acro v3 only: `speed_leg` 25 (on), `line_leg` 0
and `attitude_leg` 0 (off until enabled). Acro v4: `line_path_leg` 0, `line_lead_s` 0, `path_slack` 0.10 (with
`path_leg` 0 = off). Leg window: `reference_speed` 55 m/s, `min_leg_s` 7.5
(settling floor, same as `step_budget.MIN_EPISODE_SECONDS`), `max_leg_s` 62.5 (the default episode). Time-bonus clock:
`time_bonus_budget_s` 0 (fixed seconds) and `time_bonus_speed` 0 (distance-dependent, m/s; takes priority) — both off.

**Why `c_step` is there.** `c_step` is the **per-step living cost**, not a terminal, so it is the one term that has to
be divided by `N`: flying the whole leg costs `step_leg` (5) whatever the distance. Without it a policy can park/loiter
for free and the only cost of a timeout is the timeout terminal; with it, arriving sooner is strictly better and a
slow lap is paid for in proportion to its length. It also sets the "failed attempt" price together with `miss_leg`.

**Attempt beats giving up.** A hit pays `hit_leg` (50). A failed attempt costs `step_leg + miss_leg` (5 + 30 = 35).
Every give-up ending costs more than a failed attempt (`hover`, `timeout`, `crash`, `near_miss`, `caught`, `escape` all
45), and a hit beats parking (50 > 45). `install_acro_reward` warns if an override breaks either rule.

**Where it is applied.** Only the `install_*` glue (`install_acro_reward`, `install_acro_reward_v2`,
`install_acro_reward_v3`, `install_acro_reward_v4`, `install_reward_rc`, `install_reward_qv2`) resolves the config, once per env step. The pure
`compute_*` functions keep using the fields exactly as given, which is what the unit tests call — `auto_scale` is
deliberately inert there. The acro time bonus (`r_time_bonus * time_left_frac`) is measured against `N` when
`auto_scale` is on, instead of the old `steps_for_dist` budget (which was about 1 m/s and made a normal approach score
`time_left ≈ 0.95`); `time_bonus_budget_s` replaces `N` with a fixed clock, because `N` is unreachable in practice,
and `time_bonus_speed` replaces it with a per-episode `start_distance / speed` clock for a distance curriculum.

Because the absolute fields are derived, when `auto_scale=True` the `*_leg` fields win and the old absolute fields
(`c_step`, `r_hit`, `r_crash`, `r_hover`, `r_timeout`, `los_k`, ...) are ignored — set a leg field instead. To turn a
single term off, set its absolute field to 0 (the scaling preserves it). Example:

```json
{"auto_scale": true, "reference_speed": 55.0, "hit_leg": 55.0, "hover_leg": 50.0,
 "crash_leg": 50.0, "timeout_leg": 50.0, "step_leg": 6.0, "miss_leg": 30.0}
```

`to_dict()` / `run_config.json` still record the fields as configured (the leg totals), not the per-env resolved
values; a viewer that wants the exact per-step pay should resolve them with `proportion.resolve_proportion(cfg, dist, dt)`.
Tests: `app/test/test_reward_proportion.py`.


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
| timeout | `-r_timeout`, once | Timeouts are treated as true terminals (the trainer does not bootstrap them), so never hitting has a real cost. |
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

**auto_scale.** The numbers above are the `auto_scale=False` values. By default `install_reward_qv2` derives them from
the per-leg totals (hit 50, crash/hover/timeout 45, step 5 per leg, ...); see "Reward proportions".

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

## rc (remote-controlled drone, with threats)

Per step, in this order: terminal checks (tilt > 60 deg, roll > 85, pitch > 120, out of bounds, NaN: -15) -> caught by a
threat (swept closest approach <= 0.25 m: -15, beats a hit) -> closing -> step cost -> hover/timeout -> hit (< 0.75 m of the
base for `zone_time`: +45 in total, 7 on entry and 38 on completion) -> post-hit window. The reward is the sum.

- **Closing** (`closing_mode`): `potential` (default) is `k*(ln(prev+eps) - ln(dist+eps))`, which sums to `k*ln(d0/d_end)` and
  cannot be farmed by dithering. `log_step` is the per-step `ln1p` form; it is concave, so many small steps pay more than one
  fast move. Retreating is weighted by `closing_lambda`.
- **Per-axis progress** (`axis_k`, default 0.5; 0 = off): the same log potential on each component of `target - drone` separately,
  `axis_k * sum_i (ln(|prev_i|+eps) - ln(|now_i|+eps))`. Closing speed along one axis pays on its own, an axis that is already
  lined up is punished for drifting off it, and out-and-back on an axis nets zero. Retreating is weighted by `closing_lambda`.
  It adds to the scalar closing term (`closing_k` 2); `--reward-json '{"axis_k": 0}'` gives the old closing-only reward.
- **Line of sight** (`los_mode`, default `penalty`; `los_k` 0.02; 0 = off): the drone pays `-los_k * m(angle)` on each step it flies
  faster than `los_min_speed` (0.5 m/s), where `angle` is between its velocity and the direction to the target.
  `m = 1 + S(35, 90, angle) + 3 * S(90, 135, angle)`, with `S(a, b, x) = t*t*(3 - 2t)` and `t = clamp((x - a) / (b - a), 0, 1)`:
  m is 1 aligned up to 35 deg, 2 sideways (90 deg), 5 from 135 deg on, with no jumps between them. Aligned flight costs
  `-0.02` per step (-1.2/s at 60 steps/s), straight away `-0.10` (-6/s).
  Gated: only once the policy has taken over (`_handed_off`), and only while `dist > 0.5 x` the distance at handoff (`_handoff_h`,
  falling back to `start_dist`). Zero after the hit and on death, like the other progress terms.
  `los_mode: cos` keeps the old `+los_k * cos(angle)`, which pays every aligned step and made the policy dawdle at the target.
  Logged inside `r_progress`.
- **Post-hit window** (`post_hit_steps`, default 30 = 0.5 s; 0 = end on the hit as before): the hit is scored once, then the episode
  keeps running for that many more policy steps and ends as a success. Inside the window closing, axis, los, the step cost, hover and
  timeout are all off (the drone is flying away from the target, which they would punish and teach it to stop at the target again),
  so the reward there is 0. Crash, out of bounds and being caught still end it with their usual penalty (-15 and the step cost),
  and the trainer's eval still counts the flight as a hit because it takes the first outcome per env. The policy gets no "already hit"
  input, so past the target it only learns to stay flyable. State lives on the env (`_hit_done`, `_post_hit`, zeroed in `_reset_idx`).
  The numpy viewer env still ends the flight at the hit.
- **Hover:** outside the hit radius and slower than 0.5 m/s for 1 s costs -0.05/step; 2.5 s (`hover_limit_s`) ends the episode
  at -20 (worse than a crash on purpose). Timeout costs -10. Stalls only count after the handoff.
- **Zone hold:** the hit needs `zone_time` consecutive seconds inside `hit_radius` (default 5/60 s = 5 policy steps; the recent
  runs set 0.0167 s = 1 step). The trainer's `--reward` help text still says "+30 hit"; the code default is `r_hit` 45 in total.
- **Threats:** read from `env.nearest_threat_rel()`, the swept closest approach during the step (two drones closing at 36 m/s
  move 0.6 m per 60 Hz step and would tunnel through a point check).
- Training stats reuse qv2's accumulators, so the trainer's logging is unchanged; `train_caught_rate` is new.

**auto_scale.** The numbers above are the `auto_scale=False` values. By default `install_reward_rc` derives them from
the per-leg totals: a completed hit pays `hit_leg` (50) in total (7 at zone entry + 43 at completion), a crash or a
catch costs 45, the per-step terms are divided by `N`; see "Reward proportions".

## Drone styles (angle vs acro)
The rewards work unchanged on both styles (`dynamics/docs.md` "Drone styles"), but some terms assume angle-style:
- **Tilt termination.** `terminal_checks` / qv1's crash term end the episode at roll > 65 deg or pitch > 80 deg,
  and qv1's soft `tilt` term pushes back before that. For an acro drone that makes every flip a crash, so acro uses
  `acro_v1` instead, which measures tilt from the quaternion (no gimbal singularity).
- **Imitation term.** `reward_func`'s `imitation_coef` term compares the action with `env.pid_teacher`, which
  outputs torques. In acro mode the numpy env has no teacher (`pid_teacher=None`), so that term can't be used
  until a rate-output PID exists. The qv rewards have no imitation term.
- The `tumble` term (`c_tumble * |w|^2`, cap 0.05) costs about 0.018 per step at a 6 rad/s roll, which is paid
  on every step of a flip. Lower `c_tumble` if an acro policy refuses to spin.


## acro_v1 (acro-style drone)
qv2's hit / progress / step / timeout / crash / stall terms, with the roll/pitch cliffs replaced by a tilt limit.
Install with `install_acro_reward(env.unwrapped, AcroRewardConfig(...))`, read stats with `pop_acro_stats`, change the
limit with `set_tilt_limit`.

| Term | Value | Notes |
|---|---|---|
| tilt measure | `arccos(1 - 2(qx^2 + qy^2))` | Angle between body-z and world-up, from the quaternion. No Euler singularity, so 120 deg means 120 deg. |
| tilt (soft) | `-c_tilt * relu(tilt - L)^2` | `L = tilt_limit_deg` (default 120). Nothing below L. |
| tilt (hard) | crash `-r_crash`, terminal | At `L + tilt_hard_margin_deg` (default 30). Never trips once that reaches 180 deg. |
| hit | `r_hit + r_time_bonus * time_left_frac` | Faster hits pay more. This is what makes acro worth having. |
| tumble | `-min(c_tumble * abs(w)^2, cap)` | Same as qv2. Lower `c_tumble` as L grows if the policy refuses to spin. |
| braking | `-c_decel * relu(prev_speed - speed - deadband)` per step | Only inside the zone `dist < decel_zone_frac * ref` (default 0.33; `ref` is the handoff distance `h` in `--handoff` mode, else the start distance, so 10 m gives a 3.3 m zone), only while closing on the target, and only for speed that is LOST (a drop of 10 m/s costs about `10 * c_decel` = 1 at the default 0.1; accelerating and being slow are free; the deadband is 0.05 m/s per step). After an overshoot braking is free, so the drone can turn around. Attitude is not limited by this term. It cannot stop the drone braking *before* the zone; watch `train_near_speed` and the eval miss speed. |
| hit zone | `hit_threshold` (default 0.25 m) for `zone_time` s | The hit is scored once the drone has been inside `hit_threshold` for `zone_time` seconds in a row (policy steps; default 0 = the first step inside; rc uses 5/60 s). The count resets the moment it leaves the zone and on every env reset. Flags: `--hit-radius` (alias `--hit-zone`), `--zone-time`, or `--reward-json`. The viewer's acro plant uses the run's values from `run_config.json`. |
| away | `-c_away` per step | While distance > `away_factor` (1.3) x start distance for more than `away_grace_s` (2 s). Stops the moment the drone is back inside. |

**Progress is per axis.** `(k/3) * sum over x,y,z of (log1p(|prev_err_i|/eps) - log1p(|err_i|/eps))`, with `err` = target - position.
Each axis keeps its own steep log gradient, so a small sideways or altitude error is rewarded for shrinking even when another axis
dominates the distance (qv2's Euclidean form is blind to it). `k/3` per axis keeps the total from 10 m below `r_hit`. Example
(error [3.0, 0.3, 0] -> [2.9, 0.2, 0]): Euclidean +0.062, per-axis +0.108.

Why a soft penalty with a hard cap, not termination at L: terminating on the limit teaches the policy that dying early is
cheap, and it gives no gradient before the wall.

**Crash must cost more than stalling.** With qv2's values (`r_crash` 5, `c_hover` 0.05, `r_hover` 5) a stalled drone pays
about 11 in total, while flying off and crashing out of bounds pays about 7. A policy that cannot hit yet then learns to
turn away sharply and leave (observed: OOB rate 0.33 -> 0.93 while the hover rate fell 0.65 -> 0.06). acro_v1 therefore
started with `r_crash` 20 and `c_hover` 0.01 (full stall cost about 6). With `--handoff` that left parking much cheaper than the crash, and the
policy learned to park (hover endings 1 % -> 10 % while the eval stayed flat), so `r_hover` is now 18.8: a full stall costs
`r_hover + c_hover * 120 steps` = 20.0, **the same as a crash**. `install_acro_reward` prints a warning only if
`r_crash < r_hover + c_hover * stalled_steps` (the stall costs more than a crash, which pushes the policy to fly off instead).
With `auto_scale` (the default) these absolute numbers are derived from the per-leg totals and the rule becomes
`give_up_leg > step_leg + miss_leg`; see "Reward proportions". Set `auto_scale: false` to use the numbers above verbatim.

**Tilt limit.** `L` is widened stage by stage (120 -> 140 -> 160 -> 180 deg). The reward changes with L, so the critic has to
be re-fit after each change. How a stage is run and when to move on is in the training docs. A policy can reach a high hit
rate by flying slowly, so judge it by hit time as well (`time_left_frac` in the hit term is what pays for speed).

New stats (`rew_decel` is the per-step braking penalty): `train_tilt_rate` (replaces roll/pitch), `train_mean_tilt_deg`, `train_max_tilt_deg`,
`train_over_limit_frac` (steps above L: shows whether the policy uses the extra tilt). The stall counter is
`env._acro_stall` (qv2 uses `_qv2_stall`).

`install_acro_reward` also wraps `env._reset_idx` to keep the episode age: a full reset records `_acro_age_off`, so the
reward's age is `episode_length_buf - _acro_age_off`, and the first, deliberately shortened episode is marked `_acro_partial`
(a timeout there is not penalised and not counted). `zero_acro_clock` clears all of this.

## acro_v2 (acro_v1 with the angle rc progress terms + line-of-sight penalty)
`acro_reward_v2.py`: `install_acro_reward_v2(env.unwrapped, AcroRewardV2Config(...))`. It installs v1, then adds one part,
`los`, on top of the v1 reward, the episode sums and `rew_los`. Everything else (hit, braking, tilt, hover, away) is v1.
v1's own progress term is replaced (`progress_k` 0) by the rc ones, same formulas and defaults, summed into `rew_progress`: the
total-distance closing `closing_k` (2) * (ln(prev+eps) - ln(dist+eps)), and the per-axis term `axis_k` (0.5) * sum over x, y, z of
(ln(|prev_rel|+eps) - ln(|rel|+eps)); a negative value (moving away, on an axis or overall) is multiplied by `closing_lambda` (1).
The term is the one from the angle **rc** reward (`angle_reward_rc.py`; qv2 has no LOS term), same formula and defaults:
`-los_k * m(angle)` per step (`los_k` 0.02), `angle` = velocity vs the direction to the target, `m = los_aligned` (1) up to
`los_a` (35 deg), +1 by `los_b` (90 deg, smoothstep), +`los_far` (3) more by `los_c` (135 deg), so aligned costs 0.02, sideways
0.04 and straight away 0.10 per step. Zero below `los_min_speed` (0.5 m/s), zero on the ending step, and gated: only after the
handoff and only while `dist > los_gate_frac (0.5) * distance at handoff`. `los_mode "cos"` pays `+los_k * cos` instead;
`los_k 0` gives exactly v1. Select it with `acro_rtl_train_isaac.py --reward v2`; tune with `--reward-json '{"los_k": 0.05}'`.

### Anti-loop terms (A: missed pass, B: near-miss timer)
Why: a trained policy can settle into a stable back-and-forth line that passes ~2.5 m beside the target every ~7 s until
the timeout. The closing and per-axis terms are differences of logs, so they cancel over each lap; hover never fires
(speed > 0.3 m/s); the LOS term is small. The loop is a zero-risk ~0 reward, so PPO keeps it. Both terms below act only
after the takeover (`_handed_off`, or `_prefix_left <= 0`), and never on a step v1 already ended.

- **A, missed pass** (`update_pass_state`). Per env: `approaching` turns on when the closing speed (from the same
  positions as the progress terms) is above `+pass_speed_eps` (0.5 m/s); `d_min` tracks the closest distance since then.
  When the closing speed drops below `-pass_speed_eps` the pass is over: if `d_min < miss_radius` (15 m) it costs
  `-k_miss * d_min` (`k_miss` 1.0) once. A pass at 2.5 m costs 2.5, at 1.2 m costs 1.2, so the penalty gets smaller as
  the miss gets smaller. A hit ends the episode before the pass fires, so hits pay nothing. The deadband keeps a slow
  wobble from counting. `k_miss 0` = off. Stats: `rew_miss`, `train_passes_per_ep`.
- **B, near-miss timer.** `_v2_near` counts the steps inside `near_radius` (15 m) since the takeover, in TOTAL, not
  consecutive (a loop that dips out of the radius each lap still adds up). At `near_limit_s` (6 s) the episode ends with
  `-r_near` (5.0, kept below `r_crash` so crashing is never the cheaper way out). A normal braking approach from 15 m takes
  ~3 s. Not on the timeout step. Reported as `extras["term_reasons"]["near_miss"]` (v1 always puts a zeros tensor there),
  eval outcome `near_miss` (code 6 in `acro_rtl_train_isaac.EVAL_CODES`), stats `rew_near`, `train_near_miss_rate`.
  `near_limit_s 0` = off.

- **Closing speed (linear, off by default)**: `closing_speed_k * v_toward / closing_speed_ref` per policy step, where `v_toward`
  is the drone's own velocity along the line to the target (a runner's motion does not count, the policy cannot change it).
  Flying away costs `closing_lambda` times that. Only after the takeover, and only while `dist > closing_speed_min_dist`.
  Why: the log closing term pays `closing_k * v / dist`, almost nothing far away, and a hit 30 s away is invisible to the critic
  at gamma 0.99-0.995 (horizon 1/(1-gamma) steps = 1.7-3 s at 60 Hz), so nothing else rewards flying faster. Logged as `rew_vclose`.
  It is linear in `v_toward`, though, so its per-leg total does not depend on the speed at all (see "acro_v3" for the term that does).
  Start with `{"closing_speed_k": 0.02, "closing_speed_ref": 20}`; too large and it trades hits for speed (fly-bys), which `k_miss` pays for.

Tune with e.g. `--reward-json '{"miss_leg": 40.0, "near_limit_s": 8.0}'` (with `auto_scale`, the `*_leg` fields win;
setting `k_miss` / `r_near` to 0 still switches the term off). Tests: `app/test/test_acro_reward_v2.py`.

The absolute numbers in this section are the `auto_scale=False` values; by default `install_acro_reward_v2` derives
them from the per-leg totals (hit 50, crash/hover/timeout 45, near-miss 45, miss 30 at the radius edge, step 5 per
leg). See "Reward proportions".

## acro_v3 (acro_v2 with the speed and straight-line terms)
`acro_reward_v3.py`: `install_acro_reward_v3(env.unwrapped, AcroRewardV3Config(...))`, select with
`acro_rtl_train_isaac.py --reward v3`. It installs v2 and adds three parts on top of its reward, episode sums and
statistics (`rew_speed`, `rew_line`, `rew_att`, plus `train_mean_lateral_m` / `train_speed_short_frac`; the per-step
values and the two physical diagnostics also print as `spd=` / `line=` / `lat=` / `short=` in the console line).
Everything else is v2 unchanged.

**Why v2 could not make the policy fast.** Every shaping term in v1/v2 is the integral of a function of position, so it
pays for the *path* and not the *pace*. The log potentials (`closing_k`, `axis_k`, `progress_k`) telescope to a log of
the distances, and the linear closing-speed term is explicit about it:

    sum_t closing_speed_k * v_toward / closing_speed_ref  =  close_speed_leg * reference_speed / closing_speed_ref

A 26 m/s flight and a 50 m/s flight over the same path earn exactly the same `vclose`. Measured on the 1000-1500 m run:
`rew_vclose` +0.0118/step against `rew_step` -0.0037/step, i.e. doubling the cruise paid ~8 units per leg while a crash
cost 45 and the hit (50) sat ~2500 steps away (`0.995 ** 2500 ≈ 3e-6`). Flying at 26 m/s was the risk-optimal choice.

A per-step reward can only pay for speed in three ways, and v3 uses all of them: the constant living cost (`c_step`,
already in v1), a **non-linear function of speed** (below), and the terminal bonus (`r_hit` + `r_time_bonus`, which
needs `time_bonus_budget_s > 0` to be non-zero — see "Reward proportions"). There is deliberately **no `dv/dt` bonus**:
a reward for accelerating pays for oscillating, and the target-speed profile already pays for getting up to speed.

- **Speed, `r_speed`.** The target is the LOS teacher's brake-limited profile with a non-zero touchdown speed,
  `v* = clamp(sqrt(speed_end_speed^2 + 2 * speed_accel * dist), speed_end_speed, speed_target)`: fast while far,
  braking to `speed_end_speed` at the target. The term is

      -(speed_leg / N) * clamp((v* - v_toward) / speed_target, 0, 1)     per step, 0 where not active or dist <= speed_min_dist

  where `v_toward` is the velocity along the line of sight. So it charges for being slow, clamps at one leg's worth when
  flying away, and pays nothing for flying sideways: **speed and straightness in one term**. It is a *lower* bound:
  flying faster than `v*` is never punished. `speed_leg 25` costs ~12 units per leg at a 26 m/s cruise and ~2.5 at
  45 m/s, on top of `step_leg` 5. Defaults `speed_target` 50 (the airframe cap is ~52 m/s at ~70 deg tilt, full thrust;
  55 is not reachable), `speed_accel` 16 m/s², `speed_end_speed` 0.
  `speed_end_speed` matters at the *end*: with 0 the profile allows ~0 m/s from ~80 m out, so nothing charges the final
  creep and the policy is free to arrive at walking pace. Set it to the speed you want at the hit sphere (20-25 m/s):
  at 50 m/s that brakes over the last ~63 m and a 5 m/s creep costs `speed_leg * 15/50` per leg instead of ~0. It costs
  no accuracy — the miss distance is set by the straightness of the path hundreds of metres earlier.
- **Cross-track, `r_line`** (off by default). `-(line_leg / N) * clamp(lateral / line_ref, 0, 1)`, where `lateral` is the
  distance from the straight line through the drone's position on the leg's **first policy step** and the target. It
  charges a curved-but-fast approach that the speed term cannot see (a sideways detour flown at 50 m/s earns full
  `r_speed`). Captured lazily on the first policy step after a reset (cleared in a chained `_reset_idx`), so it works
  with `--handoff` and a moving runner target. Defaults `line_leg` 0 (off), `line_ref` 50 m.
  **`line_ref` is a saturation, not just a scale**: the term is linear in `lateral` up to `line_ref` metres and *flat*
  beyond it, so any offset larger than `line_ref` has no gradient and is not worth correcting. The first reward-v3
  run used `line_ref 50`, `line_leg 8`; its `rew_line` was only -0.0038/step, small next to the speed term, and the
  tuned run still averaged ~33 m of cross-track. The terminal aim was never 33 m off (v4 later measured `perp` p50
  0.49 m), so read this as a midcourse bow, not a parallel pass. Set `line_ref` above every offset you want corrected
  (150-300 m at 1000-1500 m legs) and size `line_leg` relative to the hit (a 150 m offset should cost about as much
  as one leg's pay).
- **Soft attitude wall, `r_att`** (off by default). `-(attitude_leg / N) * clamp((tilt - attitude_soft_deg) /
  (tilt_limit_deg + tilt_hard_margin_deg - attitude_soft_deg), 0, 1)`. With `tilt_limit_deg 180` the policy flew an
  eval max tilt of 100 deg (inverted) for free; with `attitude_soft_deg 85` and `tilt_limit_deg 80 + margin 45` the
  ~70 deg max-thrust cruise is free and the 100-125 deg regime is charged. `attitude_leg 0` (off) by default.

**Reading the two diagnostics.** `train_mean_lateral_m` is the mean cross-track offset in metres (independent of
`line_ref` / `line_leg`; a flight mean, not the terminal offset), and `train_speed_short_frac` is the mean shortfall as
a fraction of `speed_target` (0 = tracking the profile, 1 = standing still or flying away). The tuned reward-v3 run
(`..._v3`) ended at **33.2 m / 0.313** (best-hit checkpoint: 40.6 m / 0.292), which is what "not the optimal path, and
slow" looks like in numbers: a bowed path through the midcourse, converging only near the target, not a 33 m parallel
pass. `short` falling without `lat` means the speed term is winning by flying a longer path, so `line_leg` has to go up.

Recommended starting point for a 1000-1500 m speed run: `{"speed_target": 50, "speed_accel": 16, "speed_end_speed": 22,
"speed_leg": 50, "line_leg": 30, "line_ref": 150, "attitude_soft_deg": 85, "attitude_leg": 8, "los_leg": 25,
"step_leg": 8, "miss_leg": 30, "time_bonus_budget_s": 62.5, "r_time_bonus": 20}` with `--tilt-limit 80 --tilt-margin 45`;
watch `eval_hit_time_s` (want it falling) and keep `eval_hit` within ~0.03. A failed attempt (`step_leg + miss_leg`) must
stay below every give-up terminal (45), so raise `miss_leg` only with that in mind. If hits drop, lower `speed_leg`
before touching `line_leg`. Tests: `app/test/test_reward_v3.py`.

## acro_v4 (acro_v3 with the swept hit, per-metre cross-track and the path bonus)
`acro_reward_v4.py`: `install_acro_reward_v4(env.unwrapped, AcroRewardV4Config(...))`, select with
`acro_rtl_train_isaac.py --reward v4`. It installs v3 and adds four fixes. The first reward-v3 speed run (`..._v2`) is
the evidence for all four: `eval_hit` 0.954, `eval_hit_time_s` 38.3 s, mean speed 34.1 m/s, max tilt 74.5 deg,
`eval_min_dist_p90` 0.984 m; the tuned run (`..._v3`) reached 0.958 / 36.3 s / 36.1 m/s but still averaged 33 m of
cross-track and 0.31 short. The 0.95 hit rate came from a bowed path that converged late and grazed the *sampled*
sphere — not a straight run at 50 m/s.

**1. The hit test punished speed.** `compute_acro_reward` decided the hit from the point distance at the 60 Hz sample.
At 50 m/s one step is 0.83 m, so a 1 m sphere is a 1.2-2.4 sample window; flying faster made the hit *less* likely even
on a perfect path. `hit_swept` (a v1 config field, default False there, True in v4) tests the whole segment
`prev_err -> err` and uses the closest point to the target. It is disabled when `prev_speed < 0` (the first step after
a reset has a stale previous position) and it cannot satisfy `zone_time > 0`: a swept crossing is one step long, so
**keep `zone_time` 0 for a speed run**. The old sampled hit remains the gate for the zone timer.

**2. Cross-track was charged per STEP, not per metre.** A per-step penalty is discounted by flying faster, the exact
opposite of what a speed run needs. `line_path_leg` charges the metres flown:

    -(line_path_leg / ref_dist) * clamp(lateral_lead / line_ref, 0, 1) * |v| * dt

Summed over a leg of length `ref_dist` at `line_ref` off the line, it is exactly `line_path_leg`, at 20 or at 50 m/s.
`ref_dist` is the live leg length the rest of the reward uses (the handoff distance `h` in `--handoff`, else the start
distance). `line_leg` (v3, per step) is **forced to 0** with a warning: keeping both would pay the same detour twice
and the per-step half would still reward speeding up. `line_path_leg` is still a per-LEG total (`proportion.py`
contract), not an absolute per-step number; `line_ref` is still a saturation, so set it above any offset you want
corrected (150-300 m at 1000-1500 m legs).

**3. Cross-track had no derivative.** `line` was a P term on position, so being exactly on the line but drifting off it
cost nothing this step, and the correction at 50 m/s (there is no lift margin at ~70 deg tilt) is expensive later.
`line_lead_s > 0` scores the offset the velocity is heading for, `cross_track(rel - vel * lead_s, rel0)`: on the line
and moving 50 m/s sideways, `lead 0.5` sees a 25 m offset immediately. It is the same term, evaluated ahead; it does
not change a stationary offset. `0` = off.

**4. Nothing terminal scored the path.** `closing_k` / `progress_k` telescope to a constant, so a straight hit and a
5 % longer path paid the same. `r_path_bonus` (auto-scaled from the new `path_leg` in `proportion.py`) is paid once on
a hit:

    r_path_bonus * clamp(1 - (path_len / |rel0| - 1) / path_slack, 0, 1)

`path_len` is the metres flown since the first policy step (the `_v4_path` odometer, reset with the episode),
`|rel0|` is the straight-line distance to the target on that same step. Full bonus on the optimal line, half at
`path_slack / 2` extra path, 0 once the path is `path_slack` (default 0.10 = 10 %) longer. It is non-negative, so it
cannot make a hit worse than a failed attempt; it only separates "hit straight" from "hit after a detour".
`path_leg` 0 = off; with `auto_scale=False`, `r_path_bonus` is the absolute value.

**LOS teacher: isotropic tilt clamp.** `velocity_to_action` used `clip(roll, +-max_tilt)` and the same for pitch, i.e.
two axes each up to `max_tilt`. On a diagonal target the pitch axis saturated first, the commanded direction rotated,
and the teacher bowed out ~90 m on a 30 deg azimuth (and went out of bounds in 2.2 s at 60 m/s). It now scales the
`(roll, pitch)` vector to the cone `|roll, pitch| <= tan(max_tilt)`, so the total tilt is the cap, the direction is
preserved, and a 45 deg diagonal gets the same tilt budget as a cardinal one. The same change is in
`los_np.velocity_to_action` (`test_los_np.py` diffs the two). **Distil with `--distill-end 0`**: holding 0.02 from a
v3 checkpoint re-imposes its bowed behaviour on the student.

**Diagnostics.** `train_perp_miss_m` is the mean perpendicular distance from the target to the velocity line
(`|rel x vel| / |vel|`); `train_hit_speed` is the mean speed on the hit step; `isaac_eval` reports
`eval_perp_miss_mean/p50/p90` (the perp at the closest sample) and `eval_hit_speed`, printed as `perp=` / `hitv=`.
`rew_line` is now the per-metre term and `rew_path` the terminal bonus; both are in the plot and the console line.
A large `perp` with a large `min_dist` means a parallel fly-by; a small `perp` with a large `min_dist` means the drone
stopped short. Both must fall for the path to be straight and fast.

Recommended starting point for a 1000-1500 m run (v3's numbers plus the new levers):
`{"speed_target": 50, "speed_accel": 16, "speed_end_speed": 22, "speed_leg": 50, "line_path_leg": 30,
"line_ref": 150, "line_lead_s": 0.5, "path_leg": 30, "path_slack": 0.10, "attitude_soft_deg": 85,
"attitude_leg": 8, "los_leg": 25, "step_leg": 8, "miss_leg": 30, "time_bonus_budget_s": 62.5,
"r_time_bonus": 20}` with `--tilt-limit 80 --tilt-margin 45 --reward v4`. Watch `eval_hit` stay within ~0.03 of v3,
`eval_hit_time_s` fall toward ~30 s, `eval_perp_miss_mean` fall below ~2 m and `eval_hit_speed` rise toward 45-47 m/s
(the episode mean; a perfect 50 m/s cruise cannot read 50 there, because the mean includes the spool-up from rest).
If hits drop, lower `speed_leg` before `line_path_leg`; raising `line_path_leg` alone gives the LOS@35 failure mode
("straight but slow"), so pace must keep coming from `speed_leg` / `time_bonus_budget_s`. Tests:
`app/test/test_reward_v4.py`.

**First full run (`runs/acro_speed_s4_1000_1500_v4`, 300 M steps, 8192 envs).** Best chunk 105: `eval_hit` 0.966,
`eval_hit_time_s` 35.1 s, mean speed 37.1 m/s, max tilt 72.8 deg, min_dist p50/p90 0.840/0.980 m, perp p50/p90
0.49/0.89 m. Final chunk 142: 0.953, 34.5 s, 37.5 m/s, 75.1 deg, 0.845/0.985 m, 0.52/0.92 m. Against the tuned
reward-v3 run (0.958 / 36.3 s / 36.1 m/s) that is +0.007 hit, -1.2 s, +1.0 m/s; against the first reward-v3 run
(0.954 / 38.3 s / 34.1 m/s) it is +3.0 m/s and -3.2 s. The four v4 fixes did their job on the path (final mean lateral
33.2 -> 28.1 m, `perp` p50 0.49 m), but the speed stopped well short of `speed_target 50`. The run is **converged, not
under-trained** (actor LR at the `3e-6` floor, `kl 0.0017`, `clip_frac` 0.4 %, 5 epochs per chunk) and **not
thrust-bound** (`sat_frac_d0` 0.074; a level 38 m/s cruise is ~56 deg tilt and 0.40 thrust, 50 m/s is ~69 deg and
0.89, and 52.1 m/s is 70.5 deg and 1.00). At gamma 0.995 the whole 38->50 speed stream is only ~2.3 units of discounted
value, while 2.8-4.1 % of eval episodes already end in a tilt crash, so the policy sits at a risk/accuracy optimum. The
next experiment is the fine-tune card in `guide/05_rewards.md` (runnable command included): from `best.pt`, constant
`lr_actor 1e-4` (`--lr-min-ratio 1.0`), `speed_leg` 75, `time_bonus_budget_s` 40, no teacher/distillation (the teacher
is at 36 m/s), 25 chunks. If hits drop, centre the pass with `line_path_leg` 60-90 and `line_lead_s` 0.8; if the mean
plateaus at ~40-42 m/s, use a 42->46->50 speed-target curriculum or `step_leg` 8->10.
`eval_mean_speed` is the per-step mean over the episode, so the ramp from rest caps it: a perfect 50 m/s cruise reads
~45-47 there, and 52 m/s reads ~47-48. `train_hit_speed` (`hitv`) is the hit-step speed; `eval_hit_speed` is the
episode mean over hit episodes. Do not lower the drag to chase 55 m/s: 0.0153 is the physical lumped value and the
3 mg thrust bound gives 52.1 m/s; raise `THRUST_UP_FACTOR` to ~2.3 and retrain from the teacher / scratch (the old
checkpoint's thrust bounds no longer match) if a faster airframe is wanted.

## Retired rewards
Moved to `deprecated/reward_functions/` (see `deprecated/README.md`). Nothing outside `deprecated/` imports them.

| File | What it was | Replaced by |
|---|---|---|
| `reward_qv1.py` | Previous training reward; qv2 is a standalone copy plus the stall terms. | `angle_reward_qv2.py` |
| `phase1_rewards.py` | Phase 1.0-1.3 curriculum: single-axis approach rewards built on `rewards.RewardConfig`. | `angle_reward_qv2.py` |

## Depends on
`numpy`, `scipy`, `torch`. Isaac Lab only inside `install_reward_qv*`.
