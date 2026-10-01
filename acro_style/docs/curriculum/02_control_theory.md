# 2. Control Theory: PID vs Body-Rate / Geometric Control

*Curriculum topic 2/5 — acro-style drone project. Starting from zero on this topic.*

## Feedback control in plain terms

A **PID controller** is the standard way to make something track a target value. You measure an
**error** (target − actual), and you compute a correction from three terms:
- **P (proportional):** correction ∝ how big the error is right now. Bigger error → bigger push.
- **I (integral):** correction ∝ accumulated error over time. Fixes small persistent errors a
  pure-P controller would leave uncorrected (e.g. a constant offset from gravity/wind).
- **D (derivative):** correction ∝ how fast the error is changing. Damps overshoot — it's what
  keeps a P controller from oscillating wildly around the target.

`app/control/pid.py`'s `PIDController` does exactly this, with separate gains for position
(`kp_pos, kd_pos`) and attitude (`kp_att, kd_att`).

## The control hierarchy on a drone

A quadrotor is controlled in **nested loops**, each one commanding the loop inside it:

```
outer:  position error  →  desired lean angle (or desired velocity)
inner:  angle/rate error →  torque command  →  motor speeds
```

The outer loop runs slower and reasons about where the drone should go. The inner loop runs
faster (it has to react before the drone tips over) and reasons about attitude. Right now this
repo's PID computes a **desired lean angle** as its inner-loop target (`des_roll`, `des_pitch` in
`pid.py`), clamped to `max_tilt_rad = 0.3` (~17°) — the outer position loop is never allowed to
*ask* for more than 17° of lean, no matter how urgently it wants to move.

## Angle control vs rate control — the actual point of this topic

This is the real distinction, and it maps directly onto something you may have already run into
if you've ever flown (or read about) an FPV drone:

- **Angle mode (self-leveling):** the stick position maps to a *target lean angle*. Let go of the
  stick, and the drone levels itself back to 0° automatically. This is safe and easy to fly, but
  it has a hard ceiling — you physically cannot command more lean than the mode's max angle
  (commonly ~30-45° on a real FPV drone; this repo's PID uses ~17°). You **cannot flip or invert**
  in angle mode — "upside down" isn't a reachable target angle in a bounded self-leveling scheme.
- **Acro mode (rate mode):** the stick position maps to a *target rotation rate* (e.g. "spin at
  400°/s about this axis"), not a target angle. There is no self-leveling and no angle ceiling —
  hold the stick over and the drone keeps rotating indefinitely, straight through upside-down and
  back around. Full flips, rolls, and inverted flight are only possible in rate mode, because
  nothing in the controller has any notion of "maximum angle" at all — it only ever tracks a rate.

That's exactly why this project is called **acro style**: Phase 1 changes the policy's output
from "desired lean angle" (angle-mode-like, capped by `max_tilt_rad`) to **"desired body
rate"** ω = (ω_x, ω_y, ω_z) — acro-mode-like, no angle ceiling anywhere in the control loop.
Swift's paper uses exactly this: the policy's action is `[collective thrust, ω_x, ω_y, ω_z]`.

**Why *body* rate and not Euler rate** (ties back to topic 1): body rate is angular velocity
measured in the drone's own frame — it's literally what a gyroscope measures directly, has no
gimbal-lock singularity, and composes cleanly with quaternion integration. Euler angle rates have
the coordinate-singularity problem from topic 1 baked in.

## What has to be built: the missing inner loop

Right now there is **no rate-control inner loop anywhere in this repo** — the PID's torque output
goes straight into `mixer_inversion`. Phase 1 needs one new piece sitting between the policy and
`mixer_inversion`: a controller that takes `(ω_desired − ω_actual)` and produces a torque command,
e.g. a simple proportional law `torque = Kp · (ω_desired − ω_actual)`, or with a feedforward term
for accuracy: `torque = I·α_desired + ω × (I·ω)` (that cross term is the gyroscopic coupling from
the plan's Phase 1 physics addition — it matters more here than in angle-mode control, because
rate-mode commands can produce much larger ω than a bounded lean angle ever would).

## What removing `max_tilt_rad` actually implies

Once the policy commands rates instead of angles, `max_tilt_rad` doesn't get "raised" — it stops
existing as a concept. There is no lean-angle setpoint left to clamp. Safety doesn't disappear,
though; it just moves location — from a hard clamp inside the controller to a **soft penalty in
the reward** (the tilt-from-vertical metric from topic 1), which is a deliberate choice: the
controller stays capable of anything physically possible, and the *policy* learns, through
reward, what's actually worth doing.

## Summary

| | Angle mode (current PID) | Rate mode (acro, Phase 1) |
|---|---|---|
| Command | desired lean angle | desired body rate ω |
| Self-levels? | yes | no |
| Max tilt | hard-clamped (`max_tilt_rad`) | none — reward-shaped instead |
| Can flip/invert? | no | yes |
| Inner loop needed | angle→torque (exists) | rate→torque (**new, Phase 1**) |
