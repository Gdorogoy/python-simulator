# PID gains and reward budget: the math

Two separate derivations that feed into each other: `tune_pid.py` computes
PID gains per distance by closed-form pole-placement (no search), and
`rewards.py`'s `reward_func` uses that *same* tuned PID to measure its own
`APPROACH_MILESTONE_BUDGET` constant. Both are worked through below with the
actual numbers currently in the repo, so this doc goes stale the moment
either file's constants change -- it's a snapshot of *how* to recompute
them, not a promise the numbers below still match `git diff`.

## Part 1 -- PID gains (`app/control/tune_pid.py`)

### The plant

`PIDController.compute_action` treats the position loop's output directly
as commanded acceleration:

```
accel_cmd = kp_pos * pos_err - kd_pos * vel
```

and maps it to a tilt angle (`accel_cmd / g`, clamped to `max_tilt_rad`).
Treated as mass-normalized, that's a plain double integrator
(`pos'' = accel_cmd`), so standard 2nd-order pole placement applies
directly:

```
kp_pos = wn_pos**2
kd_pos = 2 * zeta_pos * wn_pos
```

The attitude (roll/pitch) and yaw loops obey the same identity, but through
`I * angular_accel = torque` instead (`angular_acceleration()` in
`dynamics/methods.py` divides net torque by inertia per axis):

```
kp_att = I_xx * wn_att**2      kd_att = 2 * zeta_att * wn_att * I_xx
kp_yaw = I_zz * wn_yaw**2      kd_yaw = 2 * zeta_yaw * wn_yaw * I_zz
```

### Choosing `wn_pos`: settling time, not the generic 2%-band rule

The naive rule of thumb "settling time `t_s = 4 / (zeta * wn)`" targets the
error decaying to **2% of the initial error** -- e.g. 5m of slack left over
at `dist=250m`. That's nowhere near this controller's actual target: reach
`HIT_THRESHOLD = 0.05m`, regardless of how far `dist` is. Modeling the
closed-loop error envelope as `error(t) ~= dist * exp(-zeta * wn * t)` (valid
for a near-critically-damped 2nd-order system) and solving
`error(settle_time) = HIT_THRESHOLD` for `wn` instead:

```
wn_pos = ln(dist / HIT_THRESHOLD) / (zeta_pos * settle_time)
```

where `settle_time = steps_for_dist(dist) * DT * SETTLE_TIME_FRACTION` --
half of the per-distance step budget (`SETTLE_TIME_FRACTION = 0.5`), leaving
the other half as margin for jitter/off-axis approach/imitation noise rather
than tuning to the exact deadline.

The number of time-constants this needs **grows with distance** --
`ln(3/0.05) ~= 4.1` at 3m vs `ln(250/0.05) ~= 8.5` at 250m -- which is why
the old fixed "4" constant under-budgeted `wn` at long range (verified: at
250m it landed at 0.073m, just short of the 0.05m target, right at the step
budget).

### Inner-loop bandwidth separation

Roll/pitch and yaw both run at a fixed multiple of `wn_pos` (cascade-control
rule: an inner loop has to track its setpoint much faster than the outer
loop moves it, or the two fight each other):

```
wn_att = BANDWIDTH_SEP_ATT * wn_pos     (BANDWIDTH_SEP_ATT = 5.0)
wn_yaw = BANDWIDTH_SEP_YAW * wn_pos     (BANDWIDTH_SEP_YAW = 2.0)
```

Yaw's multiplier is deliberately smaller: yaw torque authority comes from
`k_m = kf_km_ratio * k_f` (`create_quad_rotors`), only 2% of the thrust
coefficient, so demanding the same bandwidth as roll/pitch would need torque
the rotors can't actually deliver without saturating `max_rpm`.

### Worked example -- `dist = 3m`

Constants: `G = 9.81`, `MAX_TILT_RAD = 0.3`, `HIT_THRESHOLD = 0.05`,
`I_xx = I_yy = 0.02`, `I_zz = 0.04` (from `create_quad_config(mass=1.5,
inertia=(0.02,0.02,0.04), ...)`), `zeta_pos = zeta_att = 0.85`,
`zeta_yaw = 0.9`, `DT = 1/240`.

```
steps_for_dist(3)  = max(1800, int(750*3/3))  = 1800
settle_time         = 1800 * (1/240) * 0.5     = 3.75 s
wn_pos               = ln(3/0.05) / (0.85*3.75) = 4.0943 / 3.1875 = 1.2846 rad/s

kp_pos = 1.2846**2                     = 1.6502
kd_pos = 2*0.85*1.2846                 = 2.1838

wn_att = 5.0 * 1.2846                  = 6.4230
kp_att = 0.02 * 6.4230**2              = 0.8251
kd_att = 2*0.85*6.4230*0.02            = 0.2184

wn_yaw = 2.0 * 1.2846                  = 2.5692
kp_yaw = 0.04 * 2.5692**2              = 0.2640
kd_yaw = 2*0.9*2.5692*0.04             = 0.1850

saturation_ratio = kp_pos*3 / (9.81*0.3) = 4.9506 / 2.943 = 1.68
```

`saturation_ratio > 1` means the position loop commands more tilt than
`max_tilt_rad` at the initial (full-distance) error -- it starts saturated
and coasts at max tilt before desaturating on approach. Expected and fine at
short range; `compute_gains_for_distance()` prints it per distance so a run
that saturates unexpectedly hard is easy to spot.

The same formulas at every distance in `DISTANCES = (3, 10, 50, 100, 150,
250)` produce `best_pid_gains_per_dist.json`; the middle entry (currently
50m) is also written to `best_pid_gains.json` as the one generic fallback
gain set `BaseDroneEnv`'s own default `pid_teacher` loads.

`verify_gains()` then runs each derived gain set through the real sim (every
axis, a couple of target yaws) purely as a sanity check -- it's diagnostic
logging to catch a derivation bug, not a search: nothing about the gains
themselves is adjusted based on the result.

## Part 2 -- reward budget (`app/reward_functions/rewards.py`)

### The pieces of `reward_func`

```
reward_func(env):
    terminal_checks -> oob / attitude-ROLL / attitude-PITCH penalties (summed, can co-occur)
    hit_target(dist) -> HIT_REWARD if dist < HIT_THRESHOLD            (currently 50)
    phi-shaping:  phi(now) - phi(prev)   where phi = -L1_dist / start_dist
    step_penalty: -(TARGET_FRACTION * APPROACH_MILESTONE_BUDGET) / steps_for_dist(start_dist)
    milestone_bonus: one-time 10 / 15 / 20 at 25% / 50% / 75% progress toward the target
```

`HIT_REWARD`, `TARGET_FRACTION`, and the milestone fractions/bonuses are
picked, not derived -- `HIT_REWARD` specifically had to be *small enough*
that it doesn't dominate PPO's advantage estimates (see the comment next to
it in `rewards.py`: 1000 was ~100-1000x every other term and wrecked
training; 50 keeps it in the same order of magnitude as the milestones).
`APPROACH_MILESTONE_BUDGET`, unlike those, **is measured**, not picked.

**The shaping term is deliberately `phi_now - phi_prev`, not
`GAMMA * phi_now - phi_prev`.** Using PPO's own `GAMMA<1` here looks natural
(it's the textbook potential-based-shaping formula `F = γΦ(s')-Φ(s)`) but
leaves a residual `(GAMMA-1)*phi_now` every step even when nothing moves --
and since `phi` is always ≤0, that residual is *positive*. Confirmed by
hand: with `GAMMA=0.97` a drone commanded to do nothing at a 3m axis-aligned
target earned **+0.013 reward every step** (`(1-0.97)*1.0 - step_penalty`),
~+23 over a full 1800-step non-progressing episode -- nearly half of
`HIT_REWARD`, entirely undoing `step_penalty`'s purpose. Plain
`phi_now - phi_prev` is exactly the per-step *change* in distance-to-target:
zero when nothing changes, positive only on real progress -- reconfirmed by
hand after the fix: the same idle drone now earns exactly `step_penalty`
(negative) every step, nothing more.

### What `APPROACH_MILESTONE_BUDGET` means and how it's measured

The idea: `step_penalty` should cost the episode a *fraction*
(`TARGET_FRACTION = 0.25`) of the best reward a controller could realistically
earn on this task -- not an arbitrary constant. "The best reward a
controller could realistically earn" is operationalized as: run the tuned
PID (the same gains from Part 1) through the real `reward_func`, with
`step_penalty` itself forced to zero (so the measurement isn't contaminated
by the very quantity it's used to derive), and take the max cumulative
reward observed. That's `calibrate_approach_milestone_budget()` in
`tune_pid.py`:

```
for dist in (3, 10):
    for a few random directions/yaws at that distance:
        run the tuned PID for one full episode against reward_func
        (APPROACH_MILESTONE_BUDGET patched to 0.0 for the duration)
        keep the max cumulative reward seen
```

`APPROACH_MILESTONE_BUDGET` = that max. It is **not** applied automatically
-- rerun the function and hand-copy the printed number into `rewards.py`
whenever `HIT_REWARD`, the milestone bonuses, `best_pid_gains_per_dist.json`,
or the shaping formula itself change. (The zero-patch is one subtle part:
`reward_func` reads `APPROACH_MILESTONE_BUDGET` as a live module constant,
so measuring "the max reward, budget included" while the CURRENT --
possibly stale -- budget is still driving `step_penalty` would contaminate
the number being derived. Confirmed by hand: skipping the patch measured
21.56 instead of the correct 122.18 at the time, a 5.7x error, entirely from
stale `step_penalty` eating into the measurement.)

### Worked example, current constants

`HIT_REWARD = 50`, `APPROACH_MILESTONE_BUDGET = 96.72` (measured, hit bonus
included, step_penalty excluded, measured *after* the `phi_now - phi_prev`
fix above -- the earlier 122.18 was measured under the buggy
`GAMMA*phi_now - phi_prev` formula, which also inflated the PID's own
measured trajectory reward via the same standing-still residual, just less
severely since the PID is actually moving), `TARGET_FRACTION = 0.25`:

```
step_penalty(dist=3)  = -(0.25 * 96.72) / steps_for_dist(3)   = -24.18 / 1800 = -0.0134 / step
step_penalty(dist=10) = -(0.25 * 96.72) / steps_for_dist(10)  = -24.18 / 2500 = -0.0097 / step
```

So a full, un-terminated episode at 3m "spends" at most
`0.25 * 96.72 = 24.18` total reward on time pressure (that's the point of
`TARGET_FRACTION`: it's the fraction of the measured max reward the episode
is allowed to lose purely to elapsed time, spread evenly over the step
budget) -- against a max achievable reward of `96.72` (phi-shaping +
milestones + hit), or `46.72` if it never hits at all (no `HIT_REWARD`, just
milestones + shaping). Milestones alone (`10 + 15 + 20 = 45`) are worth
almost as much as the hit bonus (`50`) precisely because `HIT_REWARD` was
deliberately shrunk into their range rather than dwarfing them.

### Recalibrating after a future change

```python
from app.control.tune_pid import calibrate_approach_milestone_budget
budget = calibrate_approach_milestone_budget(distances=(3, 10))
print(budget)  # hand-copy into rewards.APPROACH_MILESTONE_BUDGET
```
