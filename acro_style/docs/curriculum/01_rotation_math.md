# 1. Rotation Math & Quaternions

*Curriculum topic 1/5 — acro-style drone project. Pitched at: solid physical intuition already
there (gimbal lock, identity rotation); this fills in the actual quaternion mechanics.*

## Why this matters for the acro project

Right now the reward function (`app/reward_functions/rewards.py`) hard-terminates an episode by
splitting orientation into separate roll and pitch angles and checking each against a degree
limit. That split-angle approach is exactly what causes gimbal lock and axis coupling — and it's
the reason the drone currently can't act "acro" (invert, flip, roll hard while yawing fast). Phase
1 of the plan replaces it with a quaternion-native check. This tutorial is what makes that
replacement make sense instead of being a black box.

## Euler angles and why they break

An Euler angle triple (roll, pitch, yaw) describes an orientation as three *sequential* rotations
about fixed axes, applied in some order (this repo uses `"xyz"` — see
`Rotation.from_quat(...).as_euler("xyz")` in `rewards.py`). The problem: at pitch = ±90°, the
first and third rotation axes become parallel — you're now rotating about the same physical axis
twice, and one degree of freedom has vanished. That's **gimbal lock**, and it's not a numerical
glitch, it's structural — it happens at exactly that orientation *no matter how precisely you
compute it*.

There's a second, related problem, and it's the one your repo's own docs already flag
(`docs-demo.md`): even *away* from pitch=90°, Euler angle *rates* (d(roll)/dt, d(pitch)/dt,
d(yaw)/dt) are **not the same thing** as the drone's actual angular velocity ω in its body frame.
They're related by a matrix that depends on the current pitch angle, and that matrix's
determinant goes to zero as pitch approaches 90° — so the faster the drone yaws while pitched
over, the more Euler rates and true angular velocity diverge. This is exactly why Swift's policy
(and Phase 1 of this plan) commands **body rates** (ω_x, ω_y, ω_z — true angular velocity in the
body frame) rather than Euler rates: body rates have no such singularity, ever.

## Quaternions: the actual mechanics

A quaternion `q = (x, y, z, w)` represents a single rotation by angle θ about a unit axis vector
`â = (a_x, a_y, a_z)`:

```
x = a_x · sin(θ/2)
y = a_y · sin(θ/2)
z = a_z · sin(θ/2)
w = cos(θ/2)
```

Two things fall directly out of this that you got tripped up on in the quiz:

1. **Which component is nonzero depends on the axis, not the angle.** `(x,y,z)` together *are*
   the axis vector (scaled by `sin(θ/2)`). Rotate about x → x carries it. About z → z carries it.
   `w` never encodes the axis — it only encodes how much rotation, via `cos(θ/2)`.
2. **The unit-norm constraint is what makes 4 numbers represent 3 degrees of freedom.**
   `x² + y² + z² + w² = 1` — because `a_x²+a_y²+a_z² = 1` (unit axis) and
   `sin²(θ/2) + cos²(θ/2) = 1` (trig identity), so it's forced. This is a genuine algebraic
   constraint, not a range restriction on any one component — check it whenever an answer looks
   wrong, it catches most quaternion mistakes instantly.

**Worked examples (the two from the quiz, done right):**
- Identity (no rotation): θ=0 → `sin(0)=0`, `cos(0)=1` → `(0,0,0,1)` regardless of axis choice
  (makes sense — with zero rotation, "which axis" is meaningless).
- 90° yaw (axis = (0,0,1)): θ=90° → θ/2=45° → `sin(45°)=cos(45°)≈0.707` →
  `(0, 0, 0.707, 0.707)`. Check: `0²+0²+0.707²+0.707² ≈ 0.5+0.5 = 1.0` ✓.

**Composing rotations:** quaternions chain via quaternion multiplication (`q_total = q2 * q1`
applies q1 first, then q2) — this is what `Rotation.from_quat(...) * Rotation.from_rotvec(...)`
does in `app/dynamics/methods.py`'s orientation integration. No gimbal lock risk anywhere in this
operation, which is why the whole dynamics/state layer already uses quaternions exclusively
(confirmed in exploration — no `from_euler`/`from_matrix` calls anywhere in `app/`). Euler angles
only ever get computed transiently, when something needs a human-readable or scalar angle value.

## The gimbal-lock-free tilt metric (what Phase 1 actually implements)

Instead of splitting into roll and pitch and checking each against a limit, compute a single
"how far from upright" angle directly from the quaternion:

```
body_up = q applied to (0, 0, 1)      # rotate the body's local up-vector into world frame
tilt = arccos( dot(body_up, world_up) )   # world_up = (0, 0, 1)
```

This is the angle between where the drone's "up" is actually pointing and true vertical — a
single number, valid all the way to fully inverted (tilt=180°), with **no singularity anywhere**.
The roll/pitch split's actual failure mode is near gimbal lock itself: as pitch approaches ±90°
(i.e. the drone approaches 90° of tilt from vertical — well within acro territory), roll and yaw
become non-unique per the gimbal-lock argument above, so the *same* physical tilt can report a
wildly different "roll" value depending on yaw alone. `arccos(dot(body_up, world_up))` never
decomposes into roll/pitch/yaw at all, so it has nothing to become non-unique — it's stable and
well-defined all the way to fully inverted.

## Cheatsheet

| Want to know | Formula |
|---|---|
| Rotation by θ about axis â | `(â·sin(θ/2), cos(θ/2))` |
| Is this quaternion valid? | `x²+y²+z²+w² ≈ 1` |
| Chain q1 then q2 | `q2 * q1` (quaternion multiplication, not addition) |
| "How far from upright" (gimbal-lock-free) | `arccos(dot(q·(0,0,1), (0,0,1)))` |
| Body rate vs Euler rate | Different things; body rate (ω) has no singularity, Euler rate does |
