# Curriculum Log

One entry per topic: level found (via binary search), tutorial file, test result, and any
design decisions the topic settled. Feeds `../design.md` once topic 5 closes out.

## 1. Rotation math & quaternions
- Level found: novice-to-intermediate. Strong physical/applied intuition going in (correctly
  reasoned through gimbal lock and identity rotation unprompted). Formal mechanics (unit-norm
  check, axis-angle formula, which pair of axes merges at gimbal lock) needed the tutorial;
  unit-norm check took 3 tries (kept summing raw components instead of squaring first — root
  cause turned out to be reading given quaternion values as pre-squared, not a math error) but
  ended with a correct, self-stated general rule.
- Tutorial: `01_rotation_math.md` (includes a self-correction: an inaccurate example about
  "fast yaw while upright" was caught during grading and fixed in the file).
- Test result: Q1 wrong (unit-norm arithmetic), Q2 correct (180° roll quaternion), Q3 unanswered
  then explained, Q4 half-right (right shape of idea, wrong axis pair). Remedial R1 wrong (same
  unit-norm slip), R2 correct (roll+yaw merge). Final restatement of the unit-norm rule: correct.
- Decisions locked in: Phase 1's quaternion-native tilt-from-vertical termination
  (`arccos(dot(body_up, world_up))`) is confirmed as the plan going forward — replaces the
  roll/pitch-split hard termination in `rewards.py`. No objections raised once the mechanics were
  understood.

## 2. Control theory (PID vs body-rate/geometric control)
- Level found: true beginner going in ("idk this topic at all," calibration skipped, went
  straight to a from-scratch tutorial) — picked it up fast via the angle-mode/acro-mode framing.
- Tutorial: `02_control_theory.md`
- Test result: 3/4. Q1 wrong (described P as predicting plant state — that's model-predictive
  control's job; P is model-free, reacts only to the current error). Q2-Q4 correct (action
  vectors, why angle mode can't flip, why body rate not Euler rate).
- Decisions locked in: Phase 1's body-rate action space `[collective_thrust, ω_x, ω_y, ω_z]` is
  confirmed as the framing to build toward — the "acro mode vs angle mode" analogy is the mental
  model to keep using going forward. `max_tilt_rad` doesn't get raised, it stops applying — no
  objection raised to safety moving from a hard controller clamp to a reward-level soft penalty
  (topic 1's tilt-from-vertical term).

## 3. Drone aerodynamics & motor modeling
- Level found: _pending_
- Tutorial: `03_aerodynamics.md`
- Test result: _pending_
- Decisions locked in: _pending_

## 4. Perception & state estimation
- Level found: _pending_
- Tutorial: `04_perception.md`
- Test result: _pending_
- Decisions locked in: _pending_

## 5. RL/PPO reward design & sim-to-real
- Level found: _pending_
- Tutorial: `05_reward_and_sim2real.md`
- Test result: _pending_
- Decisions locked in: _pending_
