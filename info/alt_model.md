# Alternative model: two-phase guidance (controller -> RL policy) on a handoff-state pool

Rewritten 2026-10-05. **Status: a design, not built.** What *is* built (and measured) is described in
`PROJECT_DEFENSE_GUIDE.md` Part 14 and `app/training/docs-rtl-handoff.md`: `app/training/rtl_train_isaac.py`
(PPO + PID teacher + handoff stage + Isaac-only evaluation) and `app/reward_functions/reward_qv2.py`.
This document says what I would build *next*, why, and in what order, using what those runs taught us.

**Status update 2026-10-05 (late):** three small pieces of this design already exist in `rtl_train_isaac.py`: time-uniform *staggered starts* (a stop-gap for the lock-step problem the state pool removes), a guard
driven by the *eval* hit rate (4.3), and `--anchor-checkpoint` (a frozen previous best as the distillation teacher, 4.3; currently an MSE loss, not the proposed KL penalty). Everything else below is still design.

**Measured motivation for 4.2 (2026-10-06):** the training-time position re-centring is an observation transform that must also be applied at deployment; a numpy two-phase replay without it gave 9 % hits, with it 24/24 on the same scenarios
(`TwoPhaseAgent(recentre_z=20)`, a stop-gap built from this finding). A target-relative observation removes the need for the transform altogether.

**Built (2026-10-06):** the position part of that is now `--obs-position-mode height` (no absolute x/y, altitude clipped at 30 m; same 23-dim layout). Starting from the old `full`-mode best, it scored 0.935 in Isaac eval with no training (0.875 in its own mode). The rest of 4.2 (bearing/range from a tracker, noise, latency, visibility flag) is still design.

The earlier version of this file proposed a different alternative (acro/rate control, a policy that outputs thrust and
body rates). That proposal is not dead, it is a separate axis (action space), and it is kept unchanged in
**Appendix A**. Everything above the appendix is about the *problem structure*: who flies which part of the flight, and
what the policy is trained on.

## 1. Goals (what the system has to do)

1. **Distance, cheaply.** Long distances are flown by a controller. The learned policy never learns to cruise.
2. **Speed.** Arrival speed is a priority, so the controller cruises fast (it must not brake early) and the policy
   handles the high-speed terminal phase.
3. **Hit, and recover if it misses.** After a miss the drone turns around and comes back (no parking).
4. **Camera later.** The policy must be trainable now on a target estimate that a camera + tracker can supply later.
5. **PPO, RTL, teacher-student stay.** Same learning machinery, different data and a better-matched teacher.

## 2. What we measured that drives this design

| Finding (all from runs in `runs/`, details in Part 14 case study) | Consequence |
|---|---|
| A 35 m/s top speed with 3-10 m of room is unwinnable: crash rate by speed bucket ~0 / ~0.6 / ~0.9, PID hits ~0.2 | handoff distance and arrival speed must be chosen together from braking distance v^2/(2a) |
| 20 s episodes (long spawn leg) synchronise 8192 envs; chunk-level `train_hit` swung 0.36-0.90 with a frozen policy | the guard misfired, the teacher weight hit its cap (5.0), eval fell 0.57 -> 0.35; envs must not start in lock-step |
| The real controller leg handed off **20.6 m too far out on average** (p90 +38.8 m) at a lower speed than planned (6.8 vs 10.5 m/s); the ramp acceleration (3.0) was at the tilt limit | the handoff must be triggered by position, and the profile must be flyable (g*tan(tilt)*0.5) |
| Same start policy: eval 0.56 with a 50-80 m spawn, **0.09** with 150-250 m; **0.82** after re-centring the world so the handoff is at the origin | absolute position in the observation is a liability |
| The hover PID scores 0.1-0.65 on this task; anchoring the student to it (guard boost 20x) pulled the student toward parking | the PID is the wrong teacher for "hit"; the student is already better |
| Parking at 0.9-3.5 m with action ~0 appeared in a replay tool that has no hover termination and a different simulator | evaluate end-to-end inside Isaac with the same terminations as training |

## 3. Architecture

```
            +----------------------- phase 1 (controller) ------------------------+     +-- phase 2 (policy) --+
spawn  ---> | cruise controller: ramp -> cruise -> ramp down to arrival speed     | --> | PPO policy: hit,     |
(far)       | (velocity tracking along the line, NOT a hover PID)                 |  ^  | recover on a miss    |
            +----------------------------------------------------------------------+  |  +----------------------+
                                                                         handoff trigger: real distance <= h
                                                                         (h from the POLICY's range, not the spawn range)
```

Training does **not** simulate phase 1 in every episode. See 4.1.

## 4. Components

### 4.1 Handoff-state pool (replaces "fly the leg in every episode")
- Run the real cruise controller from long distances **once**, at many speeds, and log the state at the handoff point:
  position, velocity, attitude, angular rate, rotor speeds, target vector. Re-centre so the handoff point is near the
  origin (we saw why: eval 0.09 vs 0.82).
- Training episodes start from a random pool sample plus noise, plus hand-made hard cases:
  **very low speed and standstill, overshoot starts (already past the target, moving away), large lateral velocity,
  large tilt and angular rate.**
- Wins: every step is policy data (no masked controller steps), no 20 s leg per episode, and the **lock-step problem
  disappears** because each env starts from a different state at a different moment.
- Cost: the pool is only as good as the controller that made it. Mitigation: 4.7 (end-to-end test).
- Automatic curriculum: sample more from the (distance x speed) bins with the lowest eval hit rate.

### 4.2 Observation (target-relative, camera-ready)
- Target bearing and range (or a relative vector in a yaw-aligned frame), closing speed and lateral velocity, own
  attitude (quaternion), angular rate, rotor speeds.
- **No absolute position.** (Same reason as 4.1; also a camera drone does not have it.)
- A defined **estimator interface**: the policy reads `(estimate, uncertainty, visible)`. In simulation the estimate is
  the true value plus noise that mimics the sensor chain: bearing noise small, range noise growing with distance^2
  (stereo), 1-3 frames of latency, random dropouts, and "target out of view" after an overshoot. When the camera
  arrives it replaces the noise generator; the policy is not retrained from pixels.
- Tracker (outside the network): CNN gives bearing (+ apparent size), stereo gives range only when close, a filter
  fuses them with own position/velocity. After a miss the target leaves the field of view; the filter keeps predicting a
  static target from own motion, so the policy can still turn around.

### 4.3 Teacher: stop distilling toward a hover PID
- **Main anchor: a frozen copy of the previous best policy** (KL penalty = trust region). The student is already better
  than the PID here, so the best available teacher is its own earlier self.
- **Optional scripted floor:** a pursuit/intercept controller (aim at the target, hold speed, no braking). It hits at any
  speed, so its labels point the right way for "hit". The hover PID is kept only as a tiny-weight floor or dropped.
- **Weight by teacher quality:** use the teacher only on state bins where it succeeds, and decay the weight.
- **Guard:** driven by held-out eval (not by `train_hit`), with a low cap on the boost. (In the 3-30 m run the boost
  reached 20x and the coefficient 0.45; in the 50-80 m runs it reached the cap 5.0.)

### 4.4 Reward (qv2 plus two terms)
- Keep: hit (+10, terminal), potential-based progress (log1p distance), small step cost, crash, tumble/tilt walls,
  **hover rule** (stalled = off-target and slower than a threshold for 1 s -> penalty, 3 s -> terminate).
- Add: a **time-to-hit** term (speed is a priority) and a **re-attack** term for reducing distance after a first-pass miss.
- Keep timeouts as true terminals, and keep policy-phase episodes short so that is consistent.

### 4.5 PPO / RTL machinery (mostly unchanged)
- Separate critic (own MLP), critic warmup before the actor trains, head-only before full unfreeze, GAE, clip + KL early stop.
- **New:** save and resume the critic and optimizer (`--resume`), value normalisation, advantage normalisation per
  minibatch, a std with a lower floor and its own schedule, `best.pt` chosen on the held-out grid.

### 4.6 Evaluation (Isaac only)
- **Policy-phase grid:** fixed seeds, hit and time-to-hit per (distance x speed) bin; recovery rate after a first-pass miss.
- **Per-distance time budgets**, first-episode scoring, no lock-step bias (burn-in or independent eval envs).
- Never select on the same numbers you report. Keep a held-out set.

### 4.7 End-to-end test (guards the handoff)
Every N chunks: the **real** controller from 250 m into the **real** policy. Log the real handoff distance and speed error
(already done in `isaac_eval` as "handoff check"). If this number drifts from what the pool assumed, the pool is stale.

### 4.8 Cruise controller (speed priority)
- Velocity-tracking along the line (the `_carrot_action` idea), ramp planned at `0.5*g*tan(max_tilt)`.
- Handoff distance from physics: `h >= v_arrival^2 / (2*a_brake)`, plus margin. This is the number that ties speed to distance.
- Closed-loop cross-track correction (the measured handoff error was +2 m mean, +5 m p90, from sideways drift).

## 5. What changes in the code

| Today (`rtl_train_isaac.py`, ~1400 lines) | Proposed |
|---|---|
| handoff wrapper flies the leg in every episode, masks controller steps (`pm`) | state pool; no mask needed |
| `PidTeacher` labels every step, guard on `train_hit` | **partly built:** `--anchor-checkpoint` (frozen best, MSE) and an eval-driven guard; proposed: KL form, held-out eval |
| observation has absolute position | target-relative + estimator interface |
| eval resets all envs together, burn-in to desynchronise (+ time-uniform staggered training starts, added 2026-10-05) | independent eval slice, plus end-to-end test |
| ~80 CLI flags | one config dataclass; modules: pool/sampler, teacher, ppo core, eval, logging |
| no tests for the sampler | pure-torch unit tests (like the reward) |

## 6. Order of work

1. Target-relative observation + handoff-state pool (biggest measured win).
2. Eval grid, held-out selection, end-to-end test.
3. Frozen-best KL anchor and guard on held-out eval.
4. Time-to-hit and re-attack reward terms; low-speed and overshoot starts.
5. `--resume`, refactor, tests.
6. Camera estimator (noise model first, real tracker later).

## 7. Open questions
1. Is the target static? (Everything above assumes yes. A moving target changes the tracker and the recovery logic.)
2. Real top speed and sustainable deceleration (sets `h` and the speed buckets).
3. Camera frame rate and field of view (sets the update rate the policy must be trained at).
4. Pool size/coverage vs. keeping the controller in the loop: how stale can the pool get?
5. Abort criterion for each stage, decided **before** the run (same discipline as the promotion gate, 0.11.C).

## 8. Risks
- A pool made by controller A is wrong if deployment uses controller B (hence 4.7).
- Camera estimate noise in simulation may not match reality; the sensor model needs measuring, not guessing.
- The student can never be better than a bad anchor; a frozen-best anchor can lock in its own mistakes (keep the weight small and decaying).

---

# Appendix A — Earlier proposal (2026-09-28): acro/rate-based control (kept unchanged)

*The text below is the previous content of this file, unchanged. It concerns the action space (thrust + body rates and a
hand-derived rate controller), not the two-phase structure above. The two ideas are compatible.*

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
