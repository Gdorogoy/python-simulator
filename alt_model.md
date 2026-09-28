# Alternative model: acro/rate-based control

Planning notes from the 2026-09-28 discussion. This is a proposal, not a
built thing — nothing described here exists in the codebase yet. Separate
from `PROJECT_DEFENSE_GUIDE.md`, which documents the current, actually-built
system. Cross-references to that guide (`0.X`) assume its numbering as of
2026-09-27.

## Motivation

The current controller (cascaded position PID → small-angle tilt → attitude
PID, `PROJECT_DEFENSE_GUIDE.md` 0.3) has a **hard mathematical ceiling** on
maneuverability: max commandable acceleration is `g·tan(max_tilt)`, capped at
`max_tilt_rad=0.3` (~17°) regardless of the airframe's real thrust-to-weight
ratio. It also cannot invert — the small-angle approximation `accel≈g·tilt`
breaks down past ~45-90°, and Euler-angle extraction hits gimbal lock exactly
at 90° pitch (0.2.B), which the current design avoids only by terminating
episodes at 65°/80°.

For an **interceptor** specifically, this is a real problem: if the evader
can use its full physical envelope (inversion, high-rate maneuvers), the
current interceptor is kinematically incapable of matching it — not a
tuning gap, a structural ceiling. Reference: *Champion-Level Drone Racing
using Deep Reinforcement Learning* (Kaufmann, Bauersfeld, et al.) — a
real, working, full-envelope RL drone controller, used as the model for this
proposal. Notably, even that paper does **not** have the policy output raw
torque — it outputs **collective thrust + body rates**, handed to a
low-level rate controller. That's the architecture below.

## Decided: architecture

```
NN outputs: [thrust, p_des, q_des, r_des]   (collective thrust + desired body rates)
        |
        v
Rate controller (NEW, hand-derived)  -->  torque [tau_roll, tau_pitch, tau_yaw]
        |
        v
Existing mixer_inversion (UNCHANGED) --> rotor speeds --> motor lag --> physics
```

**Proportional navigation (ProNav) was considered and dropped.** ProNav
would add a second hand-derived layer (guidance: desired acceleration
direction) on top of the rate controller (attitude), and the two solve
different sub-problems (guidance vs. control) — adding it back doubles the
scaffolding without a clear win. Current decision: the network handles
guidance implicitly (it outputs thrust+rates directly); only the low-level
rate-to-torque step is hand-derived. This can be revisited if the from-scratch
guidance problem turns out to be intractable.

**Why a rate controller instead of the current attitude PID:** rate error
(`omega_des - omega`) is well-defined at *any* orientation, upside-down
included — no tilt/angle geometry, no small-angle approximation, no gimbal
lock. This is what actually removes the ceiling, not just a bigger number.

## Decided: rate controller derivation

Physics: `tau = I * (domega/dt)` (0.1.B, Newton's law for rotation). Control
law: `tau = Kp * (omega_des - omega)`. Substituting:

```
I * d(omega_error)/dt = -Kp * omega_error
```

A plain first-order decay, `omega_error(t) = omega_error(0) * exp(-(Kp/I)*t)`.
Pick a desired response time `tau_c` (real rate loops run fast, ~20-50ms):

```
Kp = I / tau_c
```

*Worked example:* `I=0.02` kg*m^2 (roll axis), `tau_c=0.03s`: `Kp ≈ 0.667`.
Add a small `Ki` term for steady-state disturbance rejection, same
anti-windup pattern as the existing PID (0.3.B) — clamp the integral, don't
let it wind up while saturated.

This is simpler to derive than the existing attitude-loop pole placement
(0.3.B) — first-order, not second-order — and gives a second, independently
defensible control-law derivation for the project book alongside the
existing one.

## Decided: what does NOT need to change

- **Mixer** (`mixer_inversion`) — unchanged. It already only clips at
  `max_rpm`, a real physical limit, not an artificial one.
- **Observation** — unchanged. `base_drone_env.py:53` already feeds the raw
  quaternion directly into the network, not Euler angles. No gimbal-lock
  problem on the network's *input* side; the problem was always confined to
  (a) the PID's own control law and (b) the reward's attitude-termination
  check, both of which this architecture sidesteps.

## Decided: no imitation warm-start is possible for this action space

The existing PID never produced body-rate commands, and has no control law
for full-envelope maneuvers anyway (0.3 is derived under the small-angle
assumption throughout). There is no expert to imitate here — training starts
from scratch (or from the curriculum below), not from a BC/DAgger checkpoint.
The residual-policy trick (0.13) is also unavailable for the same reason: no
valid base action exists to bound a correction against.

## Decided: curriculum via auto-retiring test scenarios

Generate `N` fixed `(start, target)` scenarios at difficulty tiers (easy /
medium / hard). Train against the current tier; once the policy succeeds on
roughly half of a tier's scenarios, retire those and generate new
(harder) ones to replace them. This is a real, named technique (automatic /
success-based curriculum, e.g. prioritized level replay), not something
invented from scratch — de-risked by precedent.

**Reuse existing infrastructure:** `app/training/eval_matrix.py`
(`build_eval_pairs`/`run_eval_matrix`) already does fixed-scenario evaluation
— the per-scenario success tracking and retirement logic should build on
this rather than a new harness from zero.

**Scope check, flagged explicitly:** for a *static* target, "shortest path"
is a straight line — already what the potential-based reward computes (0.7),
nothing new to build. If the real target is a genuinely *moving* evader,
computing a reference optimal path is a real trajectory-optimization
sub-problem (comparable in effort to the whole PID gain-ladder derivation,
likely more) and needs its own separate scoping — do not fold it silently
into "just generate N examples".

## Decided: keep the current model running as a fallback

Continue training/maintaining the current (small-angle + residual PID)
model in parallel while experimenting with the rate-controller model. Noted
explicitly during discussion: **this limits downside risk, it does not make
the new model's training any easier.** The new model still has to solve
guidance from scratch on its own; the fallback is a deployment/risk hedge,
not a technical unlock.

## Open questions (unresolved as of this writing)

1. **Reward function.** Not yet decided for this setup. Does not
   auto-inherit the old one: `stability_penalty` (rewards.py) currently
   penalizes velocity/tilt near the target, which would punish legitimate
   aerobatic maneuvers under this architecture. Needs an explicit design
   pass before training starts.
2. **Attitude termination / "what counts as a crash".** `ATTITUDE_ROLL_DEG`/
   `ATTITUDE_PITCH_DEG` (65°/80°) can't mean "crashed" anymore once inversion
   is a legitimate maneuver — needs a new definition (e.g. loss of altitude
   past a floor, out-of-bounds, exceeding a time budget with no progress).
3. **Curriculum scenario generation for moving targets** — real
   trajectory-optimization work, not yet scoped (see above).
4. **`Ki` gain and anti-windup limit for the rate controller** — not yet
   chosen; `Kp` derivation above only covers the proportional term.
5. **Training abort criterion.** Not yet defined. Should exist *before*
   the experiment starts, same discipline as the existing promotion gate
   (`SOLID_*`, 0.11.C) — e.g. "no sign of learning after N chunks on the
   easiest curriculum tier -> stop, log as a negative result / future work",
   rather than an open-ended commitment.
6. **What "the new model" in step 3 of the plan below actually means** —
   a refined version of this rate-controller model after more reading, or a
   separate design. Not yet specified.

## Plan / sequence (as stated)

1. Finish training the current model until stable through 150m (extends the
   existing curriculum: 3-10 -> 3-30 -> ... -> 150m; needs the PID gain
   ladder extended to cover new distances, same as the 3-30m step, 0.3.C).
2. Write the rate controller (derivation above) and attempt training on it.
3. Write "the new model" after reading `PROJECT_DEFENSE_GUIDE.md` and the
   references below (open question 6).

## References

Same four sources added to `PROJECT_DEFENSE_GUIDE.md`'s References section
— see that file for full citations: PPO [R1], potential-based reward shaping
[R2], DAPG / BC-regularized RL [R3], DAgger [R4]. Plus, specific to this
document: Kaufmann, Bauersfeld, et al., *Champion-Level Drone Racing using
Deep Reinforcement Learning* — the source of the collective-thrust +
body-rate action-space idea used above.
