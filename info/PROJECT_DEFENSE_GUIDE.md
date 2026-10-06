# Project Defense Guide — Drone RL Simulator (numpy oracle + Isaac Lab port)

Built from the actual code — first as of 2026-09-16, extended through
2026-09-27, with **Part 14 added 2026-10-05** for the two-phase (controller -> policy) RTL work
(not from docs-*.md, which may be stale, and not from memory).
Every claim below cites a real `file:function`.

**Theory is a ladder, for every topic: P (prerequisites) → B (basics) → C
(what this project actually does today).** Part 0 has all three rungs for each
topic (0.1-0.14); Parts 1-8 are the code walkthroughs that implement the C rung;
Part 12 is the chronological case study of the 2026-09-19 → 09-27 work (what
broke, how it was diagnosed, what was changed, what the numbers were).
Part 13 checks this project against the actual K14 grading rubric.
Scope: every active `.py` file under `app/`, excluding `__pycache__`,
`deprecated/`, `app/test/`, and run-artifact folders (`app/z_final_version_*`).

## How to use this

1. Read **Part 0** once — it's the theory every later part assumes, organized
   per topic as **P → B → C**: if you're missing background, start at the P
   rung; if the general idea is fine but you can't say what THIS project does,
   jump to the C rung. If a term in Parts 1-12 is unfamiliar, it's almost
   certainly defined there. Part 12 tells the story of the recent work in order
   and is the best thing to read before a defense conversation about "what did
   you actually do and why".
2. Read each Part's **file walkthrough** before its **practice questions**.
3. For practice: cover the "Model answer" and answer from memory first
   (closed-book, like a real defense), then check yourself. Score yourself
   0-3 per the rubric in Part 11 — don't be generous.
4. Part 10 has cross-file "trace the data" scenarios — these are the
   hardest and most realistic defense questions (a real panel will ask "and
   then what happens to that number", not "define GAE").
5. **Part 14** (2026-10-05) covers the newest pipeline: a full-network policy trained with PPO plus a PID teacher, the
   controller-to-policy handoff stage, the `reward_qv2` hover rule, and Isaac-only evaluation. Read it after Part 12 if
   the defense is about the *current* training approach; the next-version design is in `alt_model.md`.
6. Part 13 is administrative, not code theory — read it once before
   submission and again before the oral defense (the rubric's "מעמד הצגת
   הפרויקט" section is literally an oral exam on this codebase).

## References

Every `[Rn]` marker in Part 0 points here. The K14 rubric requires citing
sources for theoretical background (procedure PDF, "רקע תיאורטי / ספרות
מקצועית — יש לציין מקורות") — this list is that citation set; copy it
directly into the project book's bibliography section.

- **[R1]** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
  (2017). *Proximal Policy Optimization Algorithms*. OpenAI. — the PPO
  algorithm itself: the clipped surrogate objective and the trust-region
  motivation behind it (0.4).
- **[R2]** Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy Invariance
  Under Reward Transformations: Theory and Application to Reward Shaping*.
  — the potential-based shaping theorem and its telescoping-sum proof (0.7),
  and this project's two deliberate departures from it.
- **[R3]** Rajeswaran, A., Kumar, V., Gupta, A., Vezzani, G., Schulman, J.,
  Todorov, E., & Levine, S. (2018). *Learning Complex Dexterous Manipulation
  with Deep Reinforcement Learning and Demonstrations*. — DAPG, cited as the
  "BC-regularized RL" alternative to this project's residual-policy fix
  (0.13); not implemented here, referenced for context only.
- **[R4]** Ross, S., Gordon, G., & Bagnell, D. (2011). *A Reduction of
  Imitation Learning and Structured Prediction to No-Regret Online
  Learning*. — the `O(εT²)` vs. `O(εT)` regret bound motivating DAgger over
  plain behavior cloning (0.5).

---
# PART 0 — Prerequisite theory

## 0.1 Rigid-body dynamics

**Ladder: P (prerequisites) → B (basics) → C (this project, current).**

### 0.1.P Prerequisites

**Vectors.** A vector is just a list of numbers that also has a direction in
space — `(3, 0, 0)` means "3 meters along x, 0 along y, 0 along z". You
already add/subtract them like you'd add lists element-by-element:
`(1,2,0) + (0,1,5) = (1,3,5)`. Two operations on vectors show up constantly:

- **Dot product** `a·b = a_x b_x + a_y b_y + a_z b_z` — multiply matching
  components and add them up. It equals `|a||b|cosθ` (`θ` = angle between
  them), so it measures "how much do these two vectors point the same way":
  `+` = same-ish direction, `0` = perpendicular, `−` = opposite-ish.
  *Example:* `a=(1,0,0)`, `b=(0,1,0)` (perpendicular): `a·b = 1·0+0·1+0·0 = 0`. ✓.
- **Cross product** `a×b` — a *new vector*, perpendicular to both `a` and
  `b`, with length `|a||b|sinθ`. Formula:
  `a×b = (a_y b_z − a_z b_y,  a_z b_x − a_x b_z,  a_x b_y − a_y b_x)`.
  *Why you need it here:* if you push on a rigid body at some offset `r`
  from its center with force `F`, the amount it starts to SPIN (the torque)
  is `τ = r×F`. *Example:* push with `F=(0,0,10)` N (straight up) at offset
  `r=(1,0,0)` m (1m to the side): `τ = (0·10−0·0, 0·0−1·10, 1·0−0·0) = (0,−10,0)`
  N·m — a torque around the y-axis, which is exactly right: pushing up on
  one side of something makes it rotate around the axis running front-to-back.

**Derivatives and integrals (just "rates" and "running totals").** Velocity
is "how fast position is changing" — if position is `x(t)`, velocity is
`v(t) = dx/dt`, its derivative. Acceleration is the derivative of velocity.
Going backward (from acceleration to velocity to position) is *integration*
— literally "add up all the little changes". A computer can't do a true
continuous integral, so it approximates: pick a small time step `dt`
(here, `dt = 1/240` s), and repeat `v_new = v_old + a·dt`,
`x_new = x_old + v_new·dt`, one tick at a time. *Example:* starting at rest
(`v=0`) with `a = 2 m/s²` and `dt = 0.1` s: after step 1, `v = 0 + 2·0.1 = 0.2`
m/s; after step 2, `v = 0.2 + 2·0.1 = 0.4` m/s — velocity climbs by `a·dt`
every tick, exactly matching the calculus answer `v(t)=a·t` in the limit of
tiny steps.

**Rotation matrices, briefly.** A 3×3 matrix `R` can rotate a vector:
`v_world = R·v_body` turns a vector written in the drone's own frame into
the same vector written in world coordinates. You don't need to compute one
by hand for this project (quaternions do the actual work, 0.2) — just know
that "rotating a vector" is a matrix multiply, and that rotating twice in a
row is two matrix multiplies **in a specific order** (rotating by A then B
is not generally the same as B then A).

**Units, and the one bug this always causes.** Everything here is meters,
seconds, kilograms, Newtons, Newton-meters, and **radians** (not degrees) —
`radians = degrees · π/180`. This project's tilt limit is `0.3 rad ≈ 17°`,
while its crash-detection limits are written in degrees
(`ATTITUDE_ROLL_DEG = 65`) — mixing the two silently without converting is
the single most common bug in any project like this.

### 0.1.B Basics

**Degrees of freedom (DoF) and "state".** A rigid body floating in space can
move 3 independent ways (along x, y, z) and rotate 3 independent ways (roll,
pitch, yaw) — 6 numbers total describe how it *can* move (6 DoF). To know
where it actually IS right now, you need position (3 numbers), velocity (3),
orientation (a quaternion, 4 numbers — 0.2 explains why 4 and not 3), and
angular velocity (3) — 13 numbers total, and that whole list is called the
**state**.

**Newton's law, and its rotational twin.** You already know
`F = m·a`, i.e. `a = F/m` — force divided by mass gives acceleration. The
rotational version replaces each piece with its rotational analogue: force
`F` → torque `τ` (how hard you're twisting), mass `m` → moment of inertia
`I` (how hard it is to START it spinning, defined just below), acceleration
`a` → angular acceleration `α` (how fast the spin rate itself is changing).
That gives `α = τ/I`, i.e. **`τ = I·α`** — this is literally `F=ma` with
every symbol relabeled for rotation, not a new law. *Example:* `I = 0.02`
kg·m² (this drone's roll axis) and you apply `τ = 0.1` N·m of torque:
`α = 0.1/0.02 = 5` rad/s² — the spin rate increases by 5 rad/s every second
this torque is held.

The version above ignores one subtlety that appears when a body is spinning
around several axes *at once*: extra "gyroscopic" cross-terms
(`ω×(Iω)`, ignore the exact formula) can appear, the same way a spinning
bicycle wheel resists being tilted. This project's code drops that term
(each axis is treated independently, `α_i = τ_i/I_i`) — a deliberate,
named simplification (0.1.C), not a bug.

**Moment of inertia, concretely.** `I = Σ mᵢ·rᵢ²` — for every little piece of
mass `mᵢ` making up the object, multiply its mass by the *square* of its
distance `rᵢ` from the spin axis, and add them all up. Mass far from the axis
counts a lot more (squared!) than mass near it — this is why a figure skater
spins faster when they pull their arms in. This project's drone has inertia
`(Ixx, Iyy, Izz) = (0.02, 0.02, 0.04)` kg·m²: `Ixx = Iyy` because the drone
looks the same from the front and the side (roll and pitch are equally
"hard"), and `Izz` is bigger because the mass — the four rotors — sits
farthest from the *vertical* (yaw) axis, at the ends of the arms.

**How a quadrotor actually makes force: one rotor at a time.** Spin rotor `i`
at angular speed `ωᵢ` (rad/s) and it produces thrust
`Tᵢ = k_f·ωᵢ²` (`k_f` = a constant of that specific motor+propeller — thrust
grows with the SQUARE of spin speed) straight up along the body, plus a
reaction torque `Qᵢ = k_m·ωᵢ²` (the propeller pushes air one way, so by
Newton's third law the air pushes the propeller's frame the other way — the
same reason a helicopter needs a tail rotor). Add up all four rotors' thrust
for total lift; add up all four rotors' *positions crossed with* their
thrust (that's the `τ = r×F` from above) for roll/pitch torque, and all four
reaction torques for yaw torque. Going the OTHER direction — "I want this
much total thrust and this much roll/pitch/yaw torque, what should each
rotor's speed be?" — is called the **mixer**, and since everything above is
just squares of `ωᵢ`, solving it is: solve a linear system for the four
`ωᵢ²` values, clip any negative ones to 0 (a rotor can't spin backward here),
then take the square root of what's left to get real speeds.

**Underactuation, i.e. "why tilt at all?"** The drone has 4 controls
(thrust + 3 torques) but 6 DoF — it CANNOT push directly sideways. The trick:
tilt the whole body a little, so part of the "up" thrust now points
sideways. For small tilt angles, `sideways acceleration ≈ g · tilt_angle`
(`g=9.81` m/s²) — *example:* tilt 0.1 rad (≈5.7°): sideways accel
`≈ 9.81 · 0.1 = 0.981` m/s². This is why the whole controller (0.3) is built
as position-error → desired TILT → torque to achieve that tilt, instead of
one flat step.

**Motor lag.** A rotor can't jump to a new speed instantly — it ramps toward
its target: `ω_new = ω_old + (ω_target − ω_old)/τ_m · dt` (`τ_m` = the
motor's own time constant, how many seconds it roughly takes to catch up).
*Example:* `ω_old=0`, `ω_target=1000` rad/s, `τ_m=0.05` s, `dt=1/240` s:
`ω_new = 0 + (1000−0)/0.05 · (1/240) ≈ 83.3` rad/s after just one tick — it's
climbing toward 1000 but hasn't gotten there, and won't for several more
ticks. A controller that assumes instant response will overshoot on a real
(laggy) motor.

**Drag and wind.** Air resistance grows with the SQUARE of speed and opposes
motion: `F_drag = −½·ρ·C_d·A·|v|·v` (`ρ`=air density, `C_d`=a shape constant,
`A`=cross-section area, `v`=velocity) — double your speed, quadruple the
drag. Wind just adds an extra push proportional to how fast the air itself
is moving relative to the drone.

**Why "semi-implicit" Euler, not the naive version.** The naive
("explicit") way to integrate is `x_new = x_old + v_old·dt` — using the OLD
velocity. This project instead updates velocity FIRST, then uses the NEW
velocity for position: `v_new = v_old + a·dt`, then `x_new = x_old + v_new·dt`.
This one-line change (semi-implicit Euler) is dramatically more numerically
stable for anything oscillating (like a controller correcting position
error) — the explicit version can spiral out to infinity over many steps
even when the real physics is stable. Because everything is tuned against
one fixed `dt = 1/240`, this project never changes `dt` to speed things up —
it uses **decimation** instead (0.6): running physics at full speed/accuracy
but only asking the AI for a new decision every few ticks.

**Hover, and why "action=0" means "just float".** To hover, total thrust
must exactly cancel gravity: `ΣT = m·g`. *Example:* `m=1.5` kg:
`ΣT = 1.5 · 9.81 = 14.715` N — that's this project's `_HOVER_THRUST`
constant. Rather than have the neural network output the full 14.715 N from
scratch every single time (and have to relearn "hovering" from zero), its
thrust output is a *delta* added on top: `real_thrust = hover_thrust +
action[0]`. So `action[0] = 0` already means "hover exactly", and the
network only has to learn small corrections around that baseline.

### 0.1.C In this project (current)
- **Plant:** mass 1.5 kg, inertia `(0.02, 0.02, 0.04)` kg·m², arm 0.22 m, drag
  coefficient 0.035 (cross-section 0.05 m², air density 1.225), max 12,000 rpm,
  motor time constant 0.05 s, physics step `dt = 1/240 s`.
- **Two implementations of the same physics.** The numpy "oracle"
  (`methods.py`, Part 1) integrates everything itself. In Isaac Lab, **PhysX**
  integrates the rigid body, while the mixer, motor lag, thrust/torque and
  drag/wind are custom torch code applied each physics tick as an external
  force and torque (`_apply_action`). The two are diff-tested against each
  other (Part 8).
- **Control rate differs between the two:** the numpy oracle queries the policy
  every physics tick (240 Hz); Isaac queries it every 4th tick (60 Hz,
  `decimation=4`) and holds the action in between. This is a known
  simulator-to-simulator gap (see Part 12, open items).
- **Detail and simplifications (original notes):**

A rigid body's motion splits into **translational** (position/velocity,
governed by `F = ma`) and **rotational** (orientation/angular velocity,
governed by `τ = Iα`) halves, which are coupled only through where forces are
*applied* (a force off-center produces both linear acceleration AND torque).

- **Newton's second law**: `a = F_net / m`. Semi-implicit (symplectic) Euler
  integration — used everywhere in this project — updates velocity first,
  then position from the *new* velocity: `v' = v + a·dt`, `x' = x + v'·dt`.
  This is NOT the same as explicit Euler (`x' = x + v·dt`, using the *old*
  velocity) — semi-implicit is unconditionally more stable for oscillatory
  systems (a spring-like restoring force won't blow up numerically the way
  explicit Euler eventually does).
- **Euler's rotation equation**: `τ = I·α` per axis, where `I` is the moment
  of inertia (resistance to angular acceleration, kg·m²) and `α` is angular
  acceleration (rad/s²). For a **diagonal inertia tensor** (`Ixx, Iyy, Izz`,
  true when the body's mass distribution is symmetric about its principal
  axes — assumed everywhere in this project), each axis decouples:
  `α_i = τ_i / I_i` — no cross terms. This is why `angular_acceleration()`
  can just divide element-wise.
  - **This is a simplification, not the general case.** The full rigid-body
    rotation equation (Euler's equation, body frame) is
    `I·dω/dt = τ - ω×(Iω)`, with a **gyroscopic coupling** term `ω×(Iω)`
    that only vanishes exactly when `I` is a scalar multiple of the
    identity (spherically symmetric mass) or when `ω=0`. For a diagonal
    but *unequal* `(Ixx,Iyy,Izz)` spinning on more than one axis at once,
    `methods.py:angular_acceleration`'s per-axis `τ_i/I_i` silently drops
    this term — a documented-by-omission approximation, not an oversight
    you need to defend as a bug, but you should be able to name it as a
    simplification if asked "is this exact?".
- **Body frame vs. world frame**: state can be expressed in either frame.
  Position/velocity are natural in world frame (where is it, where is it
  going, in absolute space). Angular velocity/torque/inertia are natural in
  **body frame** (a drone's own left/right rotor thrust asymmetry always
  produces the same roll torque *relative to itself*, regardless of which
  way it's currently facing in the world). Converting a body-frame vector to
  world frame requires the current orientation (rotation matrix or
  quaternion): `v_world = R · v_body`.

## 0.2 Quaternions and rotation composition

**Ladder: P → B → C.**

### 0.2.P Prerequisites

**Why not just use roll/pitch/yaw everywhere?** Roll/pitch/yaw ("Euler
angles") are 3 numbers describing "rotate around x, then around y, then
around z, in that order" — intuitive, and great for telling a controller
"tilt 0.1 rad forward". Problem: at pitch = ±90°, the roll axis and the yaw
axis end up pointing the same direction — you lose the ability to
independently control one of the three rotations (this is called **gimbal
lock**). A drone that ever pitches to vertical would break this
representation. Quaternions are a 4-number alternative with no such
breaking point.

**The 2D warm-up: rotating with a single complex number.** In 2D, multiplying
a point `z` (written as a complex number) by `e^{iθ} = cosθ + i·sinθ` rotates
it by angle `θ` — one multiplication, no matrix needed. *Example:* rotate
`z=1` (the point `(1,0)`) by `θ=90°`: `e^{i·90°} = cos90°+i·sin90° = 0+i·1
= i`. `z·i = i`, i.e. the point `(0,1)` — quarter-turn counterclockwise,
correct. A **quaternion** is the 3D generalization of this same trick (one
mathematical object, one multiplication rule, rotates a vector directly).

**3D rotations don't commute — order matters.** Rotate 90° about x, then 90°
about y, and you land somewhere different than doing y-then-x. Any
representation of 3D rotation has to respect this, including quaternions —
composing two rotations is *multiplication*, and `q1*q2 ≠ q2*q1` in general.

### 0.2.B Basics

| Representation | Numbers | Pros | Cons |
|---|---|---|---|
| Euler angles | 3 | intuitive, good for setpoints | gimbal lock, awkward composition |
| Rotation matrix | 9 | direct vector rotation | redundant, drifts from orthonormal |
| Axis–angle / rotation vector | 3 | compact, natural for `ω·dt` | composing is awkward |
| Unit quaternion | 4 | no singularity, cheap composition | double cover, needs unit norm |

**What the 4 numbers mean.** A quaternion for "rotate by angle `θ` around
unit axis `n̂`" is built as `q = (n̂·sin(θ/2), cos(θ/2))` — the first 3
numbers are the rotation axis scaled by `sin(θ/2)`, the 4th is `cos(θ/2)`.
*Example:* rotate 180° around the z-axis (`n̂=(0,0,1)`, `θ=180°`):
`sin(90°)=1`, `cos(90°)=0`, so `q = (0,0,1,0)`. Notice `q` and `−q` (here,
`(0,0,−1,0)`) describe the exact SAME rotation — every rotation has two
valid quaternions representing it ("double cover"), which is why you can't
just compare quaternions with `==` to check "are these the same rotation".

**Composing and applying rotations.** Quaternions have their own
multiplication rule (the "Hamilton product", `⊗`) that composes rotations —
`q2 ⊗ q1` means "rotate by `q1` first, then by `q2`", same non-commuting
order rule as above. To actually rotate a vector `v`, you compute
`v' = q ⊗ (v,0) ⊗ q*` where `q*` is `q`'s conjugate (flip the sign of the
axis part) — for a *unit* quaternion (length exactly 1), the conjugate is
also the inverse, the same way rotating backward undoes rotating forward.

**Turning a spin rate into a rotation, one tick at a time.** The drone's
angular velocity `ω` (body-frame, rad/s per axis) tells you how fast it's
currently spinning, not its orientation directly. Each physics tick, you need
to turn "it's spinning at `ω` for `dt` seconds" into "here's the small extra
rotation that happened" and then apply it. That small rotation is built
exactly like the construction above, using axis = direction of `ω`
and angle = `|ω|·dt`: `Δq = exp(ω·dt/2)` (a compact way of writing "build a
quaternion from this rotation vector"). *Example:* `ω=(0,0,10)` rad/s
(spinning around z at 10 rad/s) and `dt=1/240` s: rotation this tick is
`|ω|·dt = 10/240 ≈ 0.0417` rad (about 2.4°) around z — a small step, applied
240 times a second. Because `ω` is measured in the BODY's own frame, this
small rotation must be composed **on the right**: `q_new = q_current ⊗ Δq`
— composing on the left only happens to give the same (wrong-in-general)
answer when `q_current` is very close to "no rotation", which is why this
particular bug can hide for a while before showing up as a slowly-wrong
simulation once the drone has actually turned.

**Why no explicit "renormalize" step is needed here.** In principle,
repeatedly multiplying quaternions could drift so the result is no longer
exactly unit-length (floating-point rounding, accumulated over thousands of
steps) — a broken invariant that would make the "rotate a vector" formula
above subtly wrong. This project never calls an explicit renormalize because
the quaternion library it uses (`scipy.spatial.transform.Rotation`)
guarantees every rotation it returns is already unit-length internally — the
safety net is built into the library, not missing from this project's code.

**Converting back to angles.** Since the controller (0.3) thinks in
roll/pitch/yaw, the quaternion is converted to Euler angles every step for
that purpose. This re-introduces the gimbal-lock risk from 0.2.P in
principle, but this project's episodes terminate at 65°/80° roll/pitch
(well before ±90°), so the singularity is never actually reached in
practice.

**Naming conventions that WILL trip you up if you don't check them.**
Different libraries write the same 4 numbers in different orders — `(x,y,z,w)`
in scipy vs `(w,x,y,z)` in Isaac Lab — so copying a quaternion between the two
without converting silently scrambles it into a different (wrong) rotation.
Always check which convention a library uses before passing quaternions
across a library boundary.

### 0.2.C In this project (current)
- The numpy side stores orientation as a scipy `Rotation` (unit by
  construction, so no renormalize call is needed). The Isaac side reads
  `root_quat_w` (wxyz) and converts with `isaaclab.utils.math.euler_xyz_from_quat`
  inside `base_training_isaac._read_kinematics`, feeding roll/pitch/yaw to the
  torch PID.
- Episodes terminate at roll 65° / pitch 80° (`rewards.py`), so the Euler
  extraction never reaches its singularity in practice.
- **Detail (original notes):**

A **quaternion** `q = (x, y, z, w)` (scipy/this-project convention) or
`(w, x, y, z)` (Isaac Lab convention — **the two libraries disagree on
component order**, a real bug source, see Part 1's frame-convention warning)
represents a 3D rotation with 4 numbers instead of a singular-prone 3×3
matrix or gimbal-locking Euler angles.

- **Composition**: rotating by `q1` then `q2` is `q2 * q1` (quaternion
  multiplication, not commutative — order matters).
- **Body-frame angular velocity integration**: given body-frame angular
  velocity `ω` (rad/s) and timestep `dt`, the incremental rotation is
  `Δq = exp(ω·dt/2)` (the quaternion exponential map, exactly what
  `Rotation.from_rotvec(ω·dt)` computes). Because `ω` is body-frame, this
  increment must compose **on the right**: `q_new = q_current * Δq`.
  Composing on the left is only correct when `q_current` is near-identity —
  it silently gives wrong answers for large rotations. This exact
  left-vs-right distinction is called out in `methods.py:timestamp_update`.
- **Quaternion drift and renormalization**: composing rotations by
  multiplication only stays a valid unit quaternion if every intermediate
  result is itself renormalized — accumulated floating-point roundoff can
  otherwise leave `|q|` drifting away from 1 over many steps. This project
  never calls an explicit normalize on the drone's orientation, and that's
  not a missing step: `scipy.spatial.transform.Rotation` stores and
  returns a unit quaternion by construction (`.from_quat`/`.from_rotvec`
  and `*` composition all renormalize internally), so
  `current_rot * delta_rot` in `methods.py:timestamp_update` can never
  hand back a denormalized result. Know this as the REASON there's no
  renormalize call, not as a gap.
- **symlog / atanh / tanh** (used pervasively, not rotation-specific but
  grouped here as "numeric transforms you'll be asked about"):
  - **symlog(x, k)** = `x/k` for `|x|≤k`, `sign(x)·(1+ln(|x|/k))` beyond —
    linear near zero (preserves fine detail at short range), log beyond
    threshold `k` (compresses large values so a 250m error and a 3m error
    don't blow out the observation's scale). Used in `build_observation`.
  - **tanh-squash**: an unbounded Gaussian sample `raw_action ~ N(μ,σ)` is
    mapped through `tanh` (into `[-1,1]`) then affine-rescaled into
    `[action_low, action_high]`. Needed because PPO's Gaussian policy has
    unbounded support but real actuators don't. Requires a **change-of-
    variables log-prob correction** (see 0.4).

## 0.3 Classical control: PID

**Ladder: P (prerequisites) → B (basics) → C (this project, current).**

### 0.3.P Prerequisites

**Feedback vs. feedforward, in one sentence each.** Feedforward: compute the
right input from a MODEL ("I know I need exactly this much thrust to hover")
— fast, but wrong the moment your model is a little off. Feedback: MEASURE
what's actually happening and correct the error — robust to a wrong model or
surprises (wind), but if it corrects too hard or too late it can overshoot
or oscillate instead of settling. A PID controller is pure feedback.

**The error signal.** `e(t) = target − measured`. Everything below is about
turning this one number (or vector, here: a 3D position error) into a
control output.

**Why "second-order" system, and what `ζ`/`ωₙ` mean.** Many physical
systems, once you write down their equation of motion, take the form
`ẍ + 2ζωₙẋ + ωₙ²x = ωₙ²·r` (dots mean "derivative": `ẋ`=velocity,
`ẍ`=acceleration). This is the same equation as a mass on a spring with a
damper — `ωₙ` (rad/s) is roughly "how fast it wants to oscillate on its
own", and `ζ` (no units) is "how much damping fights that oscillation":
`ζ<1` overshoots and rings a bit before settling, `ζ=1` is the fastest
settle with NO overshoot, `ζ>1` is slow and sluggish. You'll see below that
a PD-controlled drone axis is EXACTLY this equation, with `ωₙ` and `ζ`
literally set by the gains you choose.

**Stability, the short version.** A linear system's behavior over time is
governed by the roots of its "characteristic polynomial" (a small algebra
step you do on the ODE, shown below). If every root has a negative real
part, every disturbance dies out over time (stable); if any root has a
positive real part, some disturbance grows without bound (unstable). For
`s² + 2ζωₙs + ωₙ² = 0`, the quadratic formula gives roots
`s = −ζωₙ ± ωₙ√(ζ²−1)` — the real part is `−ζωₙ`, which is negative for any
positive `ζ` and `ωₙ`. Conclusion used constantly below: **as long as your
gains are positive, this kind of controller is automatically stable** — the
only question is how FAST and how SMOOTHLY it settles, not whether it does.

**Settling time and overshoot, as formulas you'll actually use.** For
`ζ<1`, the error overshoots the target before settling, by a fraction
`≈ exp(−πζ/√(1−ζ²))` of the initial jump — *example:* `ζ=0.85`:
`exp(−π·0.85/√(1−0.85²)) = exp(−2.671/0.527) = exp(−5.07) ≈ 0.006`, i.e. about
0.6% overshoot, barely noticeable. Settling time (time to stay within 2% of
the target) is roughly `4/(ζωₙ)` for the generic case — this project uses a
more exact version of this idea below, because 2% of a 250m flight is 5m of
slack, way looser than what this controller actually needs.

**Actuator saturation and integrator windup.** Every real actuator has a
limit (a motor can only spin so fast). While a command is stuck at that
limit, the system is temporarily behaving open-loop (feedback can't push any
harder) — and if a controller has an **integral** term that keeps
accumulating error the whole time it's stuck, that integral can grow huge
("wind up") and cause a big overshoot once the limit is no longer hit. Fixes:
clamp the integral to a max size (what this project does), stop
accumulating while saturated, or more advanced schemes.

**Discrete time — the same formulas, just run every tick.** A real
controller doesn't get a continuous signal, it runs once every `dt` seconds.
An integral becomes a running sum (`I ← I + e·dt`, add a little bit each
tick); a derivative becomes a measured rate or a difference between ticks.

### 0.3.B Basics

**The three PID terms, and what each one is FOR.**
```
u(t) = Kp·e(t)  +  Ki·∫e(t)dt  +  Kd·(de/dt)
       ^^^^^^^     ^^^^^^^^^^^     ^^^^^^^^^^
       P: react    I: remove       D: react to
       to error    leftover        the RATE the
       right now   steady offset   error is
                                   changing at
```
- **P alone** pushes toward the target proportional to how wrong you are —
  simple, but against a constant disturbance (like gravity) it settles with
  some permanent leftover error, because it needs *some* error to keep
  producing a nonzero push.
- **I** fixes exactly that: it accumulates error over time, so even a tiny
  leftover error eventually accumulates into a big enough push to cancel it.
  Cost: it reacts slowly and can make the system overshoot/oscillate more.
- **D** looks at how fast the error is changing (basically, velocity toward
  or away from the target) and pushes against that — it "sees a crash coming"
  and brakes early, reducing overshoot. Cost: it reacts to noise as hard as
  it reacts to real motion, so a noisy sensor makes a D term jittery.

**Derivative on the measurement, not on the error ("derivative kick").** If
the target ever jumps instantly, `de/dt` spikes to a huge number for one
tick — a real controller measures the plant's own rate of change instead
(here: velocity), which for a fixed target is just `d(target−pos)/dt =
−velocity`, so this project's D term is written as `−Kd·vel` directly —
mathematically the same thing, but immune to target jumps.

**Why this project uses PD only (`Ki=0`).** An integral term exists to
cancel a constant disturbance the P term alone can't remove — but gravity
(the main constant disturbance here) is already cancelled directly: the
thrust output is a delta ON TOP OF a precomputed hover thrust (0.1.B), not
learned from scratch. With that disturbance already handled, a plain PD
loop is enough, and skipping the integral avoids windup entirely.

**Choosing gains: three ways.** (1) Manual tuning or classic
recipes like Ziegler–Nichols — trial and error, guided by rules of thumb.
(2) **Pole placement** — write down the system's equation, decide what
`ζ`/`ωₙ` you WANT, and solve algebraically for the gains that produce them
(no trial and error at all — this is what this project does, derived
below). (3) Numerical search (try many gain sets, measure which works best)
— this repo tried Optuna for this early on; the closed-form approach below
replaced it because it's exact and instant instead of a search.

**The full derivation used by `tune_pid.py`, step by step.** Start from the
real physics: for one axis, treating "commanded acceleration" as the
control input, the plant is a plain double integrator
`ẍ = u` (acceleration `u` in, position `x` out — literally Newton's second
law with mass normalized away). The control law is PD on the position error
`e = target − x`: `u = Kp·e − Kd·ẋ`. Two algebra steps turn this into the
standard second-order form from 0.3.P:

1. Since the target is fixed, `ė = −ẋ`, so `ẍ = −ë` and `ẋ = −ė`. Substitute
   into `ẍ = u = Kp·e − Kd·ẋ`:
   `−ë = Kp·e − Kd·(−ė) = Kp·e + Kd·ė`.
2. Multiply by `−1` and rearrange:
   **`ë + Kd·ė + Kp·e = 0`**.

Compare this directly to `ẍ + 2ζωₙẋ + ωₙ²x = 0` (the no-disturbance version
of 0.3.P's formula) term by term — they're the same equation with `e` in
place of `x`, so:
```
Kd = 2·ζ·ωₙ        Kp = ωₙ²
```
This is the whole trick: **pick the `ζ` and `ωₙ` you want, and the gains
fall straight out — no search.** *Worked example:* want `ζ=0.85`,
`ωₙ=1.2` rad/s: `Kp = 1.2² = 1.44`, `Kd = 2·0.85·1.2 = 2.04` — closely
matching this project's actual 10m gains (`kp_pos=1.43`, `kd_pos=2.04`,
0.3.C shows exactly how `ωₙ=1.2` itself gets chosen for a given distance).

**Turning "how fast do I need this to settle" into a required `ωₙ`.** With
`ζ` reasonably close to 1, the error's decay is dominated by
`e(t) ≈ e₀·exp(−ζωₙt)` (the real solution has an extra oscillating factor
that matters less as `ζ→1`; ignoring it here is what makes this step
approximate rather than exact — good enough for choosing a gain). Set a
target: "shrink from `e₀` (the starting distance) down to a tolerance `ε` by
time `t_s`", i.e. `ε = e₀·exp(−ζωₙt_s)`. Solve for `ωₙ`:
```
ε/e₀ = exp(−ζωₙt_s)
ln(ε/e₀) = −ζωₙt_s
ωₙ = ln(e₀/ε) / (ζ·t_s)
```
(flipped the sign by taking `ln(e₀/ε)` instead of `ln(ε/e₀)`, since
`e₀>ε` makes that the positive, easier-to-read form). *Example:* `e₀=10`m,
`ε=0.05`m, `ζ=0.85`, `t_s=5.21`s (0.3.C shows where 5.21 comes from):
`ωₙ = ln(10/0.05)/(0.85·5.21) = ln(200)/4.43 = 5.298/4.43 ≈ 1.196` rad/s —
matches the worked example above almost exactly (small rounding). Notice
`ln(e₀/ε)` (how much decay is needed) grows only slowly as distance grows,
while a longer flight naturally gets a longer time budget `t_s` too (0.3.C)
— time budget growing faster than the required decay is *why* longer
distances end up needing SMALLER `ωₙ` (a gentler, slower loop), which is the
opposite of the naive guess that "far away needs a stronger push".

**Cascade control: chaining two loops.** This project doesn't control
tilt-angle-to-torque and position-to-tilt as one combined loop — it chains
two separate PD loops, the first loop's OUTPUT becoming the second loop's
TARGET (its "setpoint"). This only works if the inner loop reacts much
faster than the outer one changes its mind (5× faster for roll/pitch here,
2× for yaw) — otherwise the inner loop is still chasing yesterday's target
when the outer loop already moved it again, and the two end up fighting.

**The full quadrotor cascade, step by step:**
1. **Position PD** (derived above) → desired horizontal acceleration.
2. Small-angle relation from 0.1.B (`accel ≈ g·tilt`) inverted → desired
   tilt angle `= accel/g`, clamped to `±max_tilt_rad` (a hardware/safety
   limit, not a math step).
3. **Attitude PD** on tilt error → roll/pitch torque. Since real physics is
   `τ = I·α` (0.1.B) rather than a mass-normalized `ẍ=u`, the SAME
   derivation above applies but with an extra factor of `I`:
   `Kp_att = I·ωₙ_att²`, `Kd_att = 2ζ·I·ωₙ_att`.
4. **Altitude PD** (same math as position, on the vertical axis alone) →
   thrust delta.
5. **Yaw PD** (same math again) → yaw torque, independently of the other
   four steps.

**Saturation is why the gain ladder exists.** Step 2's clamp caps
commandable horizontal acceleration at `g·max_tilt_rad = 9.81·0.3 ≈ 2.94`
m/s², no matter how far the target is. If you reused the 3m gains
(`Kp_pos=1.65`) at 30m, the position loop would initially try to command
`Kp·e = 1.65·30 = 49.5` m/s² — about **17× the actuator's real ceiling**
(`49.5/2.94 ≈ 17`), so it just pins at max tilt and coasts, badly mismatched
to the actual settle-time math above. That's why gains have to shrink with
distance rather than being one fixed set (0.3.C's gain-ladder table).

### 0.3.C In this project (current)
- **Two implementations of the same law:** numpy `pid.py` and the batched torch
  `TorchPIDController` in `torch_pid.py` (per-env gain tensors), diff-tested.
- **Gain ladder.** `tune_pid.DISTANCES = (3, 10, 20, 30, 50, 100, 150, 250)`
  (20 and 30 were added on 2026-09-25 for the 3-30 m stage; before that the
  10 m gains served the whole 10-30 m span through nearest-neighbour lookup).
  Position gains `(kp_pos, kd_pos)` currently in
  `app/control/best_pid_gains_per_dist.json`:

  | dist (m) | 3 | 10 | 20 | 30 | 50 | 100 | 250 |
  |---|---|---|---|---|---|---|---|
  | `kp_pos` | 1.65 | 1.43 | 0.458 | 0.232 | 0.097 | 0.029 | 0.006 |
  | `kd_pos` | 2.18 | 2.04 | 1.15 | 0.82 | 0.53 | 0.29 | 0.13 |

- **Worked example (10 m).** `steps_for_dist(10) = max(1800, 750·10/3) = 2500`
  physics ticks = 10.42 s; settle time = `SETTLE_TIME_FRACTION (0.5) × 10.42 =
  5.21 s`; `ωₙ = ln(10/0.05) / (0.85·5.21) = 5.298/4.427 = 1.197 rad/s`; so
  `kp_pos = ωₙ² = 1.43`, `kd_pos = 2·0.85·1.197 = 2.03` — exactly the table.
  Attitude: `ωₙ_att = 5·1.197 = 5.98`, `kp_att = 0.02·5.98² = 0.716`,
  `kd_att = 2·0.85·0.02·5.98 = 0.203`. Yaw uses `2×` and `ζ=0.9`.
- **Design tolerance ≠ scoring tolerance.** The PID is designed to reach
  `ε = 0.05 m` (`tune_pid.HIT_THRESHOLD`) but the reward counts a hit at
  `0.25 m` (`rewards.HIT_THRESHOLD`). That is why every successful episode's
  `final_dist` reads about 0.25 — the episode ends the moment it is within
  0.25 m, so **0.25 is the floor of `avg_final_dist`**, not an error.
- **Gain selection.** Nearest key to the episode's start distance
  (`assign_gains_by_distance`), re-selected on every reset.
- **Measured performance** on the 3-10 m diagnostic (30 episodes per chunk):
  success 0.97-0.98, final distance ≈ 0.26 m, hit time ≈ 3.7 s (≈ 220 policy
  steps at 60 Hz). Gains fall steeply with distance (30 m: `kp_pos` is 6× softer
  than at 10 m), so long-range episodes are slow by design (the step budget is
  31 s at 30 m).
- **Three roles in this repo:** imitation teacher (BC/DAgger labels), reference
  baseline in every diagnostic (`pid_*` metrics), and — since 2026-09-26 — the
  **base controller under the residual policy** (`PidResidualEnv`, 0.13): the
  network now learns only a correction on top of it.
- **Control rate:** the policy runs at 60 Hz (action held for 4 physics ticks);
  `compute_action` is called with `dt = 1/240`.
- **Detail (original notes — formula, anti-windup, frames, pole placement):**

A **PID controller** computes a control signal from an error `e(t) = target
- current`:

```
u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·(de/dt)
```

- **P (proportional)**: pushes toward the target now, proportional to how
  far off you are. Alone, it leaves steady-state error against any constant
  disturbance (gravity, drag) — it needs *some* residual error to keep
  pushing.
- **I (integral)**: accumulates past error over time; eliminates steady-
  state offset P alone leaves behind. Must be **clamped** ("integral limit")
  or it "winds up" — keeps growing while the system is saturated elsewhere,
  then overshoots wildly once it's finally released.
- **D (derivative)**: reacts to the *rate of change* of error (here,
  velocity, since `d(position error)/dt = -velocity` when the target is
  fixed) — damps oscillation/overshoot. Too much D amplifies noise.
- **Cascaded (nested) PID**: this project's controller is two loops in
  series — an outer **position loop** whose output (desired acceleration →
  desired tilt angle) becomes the **setpoint** for an inner **attitude
  loop** whose output is torque. This only works if the inner loop is
  *much faster* than the outer loop (bandwidth separation) — otherwise the
  inner loop can't actually track its setpoint before the outer loop moves
  it again, and the two fight each other. `tune_pid.py` enforces this with
  an explicit `BANDWIDTH_SEP` multiplier.
- **Anti-windup, concretely**: `pid.py`'s `PIDController` clamps every
  integral accumulator (`integral_pos`, `integral_att`, `integral_yaw`) to
  `±integral_limit_*` on EVERY `compute_action` call
  (`np.clip(self.integral_pos + pos_err*dt, -limit, limit)`), not just at
  reset — this per-step clamp *is* the anti-windup mechanism, not a
  separate saturation-detection flag.
- **Which frame is the position error in?** The outer loop computes
  `accel_cmd = Kp·pos_err - Kd·vel + Ki·integral` in WORLD frame (both
  `pos_err` and `vel` are world-frame vectors), but a roll/pitch tilt
  command only means something relative to the drone's OWN (yaw-rotated)
  horizontal axes — "tilt forward" points a different world-frame direction
  depending on current yaw. `pid.py` rotates `accel_cmd`'s x/y component by
  `-yaw` before turning it into a roll/pitch setpoint for the inner loop.
  Forgetting this rotation is a classic cascaded-PID bug: the controller
  looks perfect at yaw=0 and increasingly mis-steers as the drone yaws
  away from that.
- **2nd-order pole placement**: a mass-normalized double integrator
  (`position'' = u`, i.e. P+D control with no separate mass term) has
  closed-loop dynamics `error(t) ≈ error(0)·exp(-ζ·ωn·t)` for a target
  damping ratio `ζ` and natural frequency `ωn`, when `Kp = ωn²`,
  `Kd = 2·ζ·ωn`. Solving `error(t_settle) = target_precision` for `ωn` gives
  you an *analytically derived* gain, no trial-and-error search needed —
  exactly what `tune_pid.py` does.

## 0.4 Reinforcement learning: MDP, policy gradients, PPO, GAE

**Ladder: P → B → C.**

### 0.4.P Prerequisites

**The reinforcement-learning setup, in plain words.** An **agent** (here, the
neural network) sits in some **state** `s` (what it currently observes), picks
an **action** `a`, and gets back a **reward** `r` (a number saying how good
that was) plus a new state. One full run from start to done is an **episode**.
The goal: pick actions that maximize the total reward collected, not just
the reward on this one step — a small sacrifice now (fly slightly off
target) can be worth it if it sets up a bigger reward later.

**Expectation, in one line.** `E[X]` just means "the average value of `X`
if you repeated this random process many times". If you flip a coin and win
$1 on heads, $0 on tails, `E[winnings] = 0.5·1 + 0.5·0 = 0.5` — the
long-run average payout, even though no single flip ever actually pays $0.50.
**Why bigger batches help:** averaging `n` independent noisy samples shrinks
the error of that average by a factor of `1/√n` — average over 4× more
rollouts, and your estimate is roughly 2× more reliable (error halves, not
quarters), which is why more parallel environments give cleaner gradients.

**The Gaussian, and why the policy uses one.** `N(μ,σ²)` is the familiar bell
curve: most likely value `μ`, spread `σ`. Sampling from it is
`x = μ + σ·ε` where `ε` is a random number drawn from a STANDARD
(`μ=0,σ=1`) bell curve — you can generate any Gaussian sample by scaling and
shifting a standard one. This project's policy network doesn't output one
fixed action per state; it outputs `(μ,σ)` and SAMPLES an action from
`N(μ,σ)` — bigger `σ` means more exploration (trying varied actions), smaller
`σ` means more deterministic (always close to `μ`).

**Gradient descent, the mechanism all of this rides on.** To minimize some
loss `L(θ)` (`θ` = all the network's weights, one big vector), compute its
gradient `∇L` (which direction makes `L` go up fastest) and step the
opposite way: `θ ← θ − lr·∇L` (`lr` = learning rate, how big a step). Neural
networks are just very large parametrized functions `f(x;θ)`; "training"
IS repeating this update, over and over, on different batches of data.

**The log-derivative trick — the one algebra step that makes policy gradients
possible.** We eventually need the gradient of an EXPECTATION (average
reward over all the random action choices the policy could make), but you
can't easily differentiate "the average over an infinite number of
random dice rolls" directly. This one identity fixes that:
```
∇p(a)  =  p(a) · ∇log(p(a))          (chain rule on log, rearranged: ∇log(p) = ∇p/p, so ∇p = p·∇log(p))
```
Plugging that into an expectation over actions turns "gradient of an
average" into "average of (something) times (gradient of a log-probability
you CAN compute)" — this is exactly the shape of every policy-gradient
formula below, so it's worth recognizing as one specific, derivable algebra
trick, not a separate law of its own.

### 0.4.B Basics

**Return and discount — "total reward from here on", with distant rewards
worth a little less.** `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...` (`γ`, the
discount, between 0 and 1) — a reward `k` steps in the future counts
`γᵏ` as much as an immediate one. *Example:* `γ=0.97`, a reward 33 steps
away: `0.97^33 ≈ 0.37` — already discounted to about a third of its face
value. The **effective horizon** (roughly "how far ahead the agent actually
cares about") is `≈ 1/(1−γ)`: at `γ=0.97` that's `1/0.03 ≈ 33` steps
(≈0.56s at this project's 60Hz control rate) — much shorter than a typical
3.7s flight, which is exactly why the reward needs dense, every-step
progress signal (0.7) instead of relying on one distant "you win" bonus that
this discount would barely see coming.

**Value function: "how good is this state, on average, going forward?"**
`V(s) = E[G_t | currently in state s]` — the expected return if you start
from `s` and keep following the current policy. Nobody computes this by
brute-force averaging every possible future; instead, a second neural
network (the "critic") learns to PREDICT it, trained the ordinary supervised
way: regress its output toward observed returns, exactly like fitting any
other numeric prediction.

**Advantage: "was this particular action better or worse than average from
here?"** `A(s,a) = Q(s,a) − V(s)`, where `Q(s,a)` is "expected return if you
take action `a` in state `s`, then follow the policy after". Subtracting the
state's average value `V(s)` turns "how good was this" into "how much
BETTER than typical was this specific choice" — a positive-vs-negative
signal that's much less noisy to learn from than the raw return.

**The policy-gradient theorem — using the trick from 0.4.P.** We want to
increase the probability of actions that led to good outcomes, and decrease
it for bad ones. Applying the log-derivative trick to the expected return
gives:
```
∇J(θ) = E[ ∇log π(a|s;θ) · A(s,a) ]
```
Read the two halves separately: `∇log π(a|s;θ)` is "which direction in
weight-space makes THIS action more likely" (pure calculus on the network,
computable exactly); `A(s,a)` is "was this action actually good" (a single
number, positive or negative). Multiplying them and averaging over many
samples: nudge weights toward actions that turned out good, away from ones
that turned out bad, scaled by HOW good or bad. Using the advantage here
instead of the raw return `G_t` doesn't change what this converges to (you
can subtract any state-only baseline without changing the expectation's
value — the subtracted term integrates to zero over the action
distribution), it only reduces the noise in the estimate.

**GAE (Generalized Advantage Estimation) — a better way to estimate that
advantage.** Define the one-step "surprise" `δ_t = r_t + γ·V(s_{t+1}) −
V(s_t)` — actual reward-plus-next-value, minus what the critic predicted
BEFORE seeing this step; positive means "better than the critic expected".
The simplest advantage estimate is just `A_t = δ_t` — cheap, but only looks
one step ahead, so it's very dependent on the critic being accurate right
now. GAE instead adds up a whole DECAYING SERIES of future surprises:
```
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```
`λ∈[0,1]` controls how far ahead you trust: `λ=0` collapses back to the
one-step version (low variance since it's a short sum, but biased toward
whatever the critic currently believes); `λ=1` sums the FULL future (no
bias from the critic at all, but noisier, since it's built from a long chain
of noisy individual rewards). **Computing that infinite-looking sum
efficiently** is a small derivation of its own: notice the SAME series,
shifted one step, reappears starting from `A_{t+1}`:
```
A_t = δ_t + (γλ)·[δ_{t+1} + (γλ)·δ_{t+2} + ...] = δ_t + (γλ)·A_{t+1}
```
That's it — a clean recursive formula, `A_t = δ_t + γλ·A_{t+1}`, computed
backward through a rollout starting from the LAST step (where there's no
`A_{t+1}` yet, so it starts at just `δ_t`) and working toward the first.
This is why `compute_gae` loops backward through time, not forward.

**Actor-critic, put together.** "Actor" = the policy (picks actions).
"Critic" = the value function (grades states, used to compute the advantage
above). They can share a network trunk since both need similar features
from the same observation (0.10). A critic that's biased (systematically
wrong) biases the actor's whole training signal; a critic that's merely
noisy (right on average, wrong per-sample) adds variance but not bias.

**Trust regions and PPO — stopping one bad batch from wrecking the policy.**
Because you collect a batch of experience with the OLD policy but then
update it, and possibly reuse that same batch for several gradient steps,
there's a risk the policy changes enough that the batch no longer reflects
what it would actually do — the update becomes based on stale data. PPO
guards against this two ways: (1) it computes the probability RATIO between
new and old policy for the actions actually taken,
`r(θ) = π_new(a|s)/π_old(a|s)`, and *clips* it to `[1−ε, 1+ε]` before
multiplying by the advantage — so one single sample can't yank the policy
arbitrarily far in one step. (2) if the measured KL divergence (0.9 — a
formal "how different are these two probability distributions" number)
between old and new policy exceeds a threshold partway through, it just
stops updating on that batch early, rather than continuing to trust
increasingly stale data.

**Advantage normalization, and its hidden cost near a good policy.** Every
batch's advantages are rescaled to have zero mean and a spread of exactly
1 (subtract the batch mean, divide by the batch's own spread) — this keeps
the update's SIZE consistent regardless of the raw reward's scale (a reward
scaled in the thousands vs. the tens shouldn't need a totally different
learning rate). The hidden cost: once a policy is ALREADY good, the true
advantage of any action is close to zero (nothing much left to improve on),
so what's actually being measured is mostly estimation noise — and
normalizing rescales that noise back up to full size regardless. Combined
with an optimizer (Adam, 0.8) that takes similarly-sized steps whether the
underlying signal is real improvement or pure noise, the weights can
**random-walk away from a good solution** even with zero true gradient to
follow — exactly the mechanism behind the training-stage slide seen in runs
v1-v4 (Part 12).

**Exploration and entropy.** Since actions are sampled from `N(μ,σ)`, `σ`
directly controls how much exploration happens. Left alone, gradient descent
tends to shrink `σ` early (a more confident/deterministic policy usually
scores a little better in the short term) — the **entropy bonus** adds a
small reward for KEEPING `σ` larger, fighting that premature narrowing so
the policy keeps trying varied actions long enough to find better ones,
before that bonus is decayed away later in training (0.8.C) so the policy
can commit to what it's learned.

**Failure modes worth being able to name:** reward hacking (maximizing the
reward function in a way that doesn't match the actual intended behavior),
policy collapse (a sudden, hard-to-recover performance drop), critic lag (the
value function hasn't caught up to a recently-changed policy yet, so its
advantage estimates are stale), catastrophic forgetting (fine-tuning erases
previously-good behavior), high-variance gradients (noisy updates that don't
reliably point the same direction run to run), and an eval metric that
differs from the training objective (the policy optimizes REWARD, which may
not track whatever number you're using to judge "success" — 0.14).

### 0.4.C In this project (current)
- **Hyperparameters** (from `app/training/best_hparams_isaac.json`, Optuna
  trial 137 with the changes documented in Part 12): `γ = 0.97` (must equal
  `rewards.GAMMA`), `λ = 0.9066`, clip `ε = 0.2048`, `target_kl = 0.008642`,
  10 epochs × 64 minibatches, `vf_coef = 0.2087`, `max_grad_norm = 1.347`,
  entropy coefficient `0.01 → 0.002`, `lr = 1.451e-4`.
- **Loop shape.** One *chunk* = `num_steps_per_chunk × num_envs` env-steps
  (`128 × 16384 = 2,097,152`); one `isaac_ppo_train` call per chunk collects
  that rollout and calls `ppo_update`, whose minibatch is
  `2,097,152 / 64 = 32,768` samples.
- **Advantage normalization** is `train.py:175`
  (`(A − mean)/(std + 1e-8)`); the KL early-stop is checked once per *epoch*
  (`train.py:240`), on the epoch's mean approximate KL.
- **Observed behaviour.** Direct-policy runs early-stopped after epoch 1 on
  almost every chunk (KL 0.01-0.03 against a 0.0086 target); residual-mode runs
  had KL 0.002-0.003 and ran all 10 epochs.
- **Detail (original notes):**

- **MDP (Markov Decision Process)**: state `s`, action `a`, reward `r(s,a)`,
  transition `s' ~ P(·|s,a)`, discount `γ ∈ [0,1)`. The agent's job: learn a
  policy `π(a|s)` maximizing expected discounted return
  `E[Σ γ^t r_t]`.
- **Policy gradient**: `∇J(θ) = E[∇log π(a|s;θ) · A(s,a)]` — increase the
  probability of actions that turned out better than expected (positive
  advantage `A`), decrease it for worse-than-expected ones. `A(s,a)` (the
  **advantage**) is "how much better was this action than the average
  action from this state" — using advantage instead of raw return reduces
  variance without introducing bias (baseline subtraction).
- **Value function / critic**: `V(s) = E[Σ γ^t r_t | s_0=s]`, the expected
  return from state `s` under the current policy. Learned by regression
  toward observed **returns** (`V(s) ≈ target`, MSE loss). The **actor**
  (policy) and **critic** (value function) can share a network trunk
  (`ActorCritic` in this project does) since both need to "understand" the
  state similarly.
- **GAE (Generalized Advantage Estimation)**: a way to estimate `A(s,a)`
  that trades off bias vs. variance via `λ ∈ [0,1]`. The TD-residual
  `δ_t = r_t + γ·V(s_{t+1}) - V(s_t)` measures one-step surprise; GAE is an
  exponentially-weighted sum of these:
  `A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...`, computed efficiently
  backward through a trajectory: `A_t = δ_t + γλ·(1-done_t)·A_{t+1}`.
  `λ=0` → pure one-step TD (low variance, high bias); `λ=1` → full Monte
  Carlo return minus baseline (high variance, low bias, since it never
  bootstraps off a possibly-wrong value estimate).
- **PPO (Proximal Policy Optimization)** (Schulman et al. 2017 [R1]): policy gradient with a **clipped
  surrogate objective** that prevents any single update from moving the
  policy too far from what collected the data (off-policy correction ratio
  `r(θ) = π_new(a|s)/π_old(a|s)` gets clipped to `[1-ε, 1+ε]`):
  `L = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)`. Why clip: without it, a large
  advantage estimate could drive one gradient step to make the policy
  wildly different from the one that collected the rollout, invalidating
  the importance-sampling ratio's validity and destabilizing training.
- **Entropy bonus**: `-ent_coef · H(π)` added to the loss encourages
  exploration (a peaked/deterministic policy has low entropy; adding an
  entropy-maximizing term keeps some randomness alive, decayed over
  training as the policy needs to commit to good behavior).
- **`target_kl` early stopping**: if the *measured* KL divergence between
  old and new policy exceeds a threshold mid-epoch, stop that update early
  — a direct trust-region safety check on top of the clip (the clip bounds
  the *objective's* sensitivity to any one sample, KL early-stop bounds the
  *whole batch's* actual policy drift).
- **Why multiple epochs makes PPO "slightly off-policy"**: a strictly
  on-policy method would take exactly one gradient step per collected
  batch — any later step optimizes against a policy that no longer
  collected the data. PPO instead takes `num_epochs` passes over the SAME
  rollout for better sample efficiency, which is only safe *because* the
  clip and `target_kl` early-stop bound how far `π_new` can drift from
  `π_old` within those epochs. Without them, later epochs would keep
  trusting an importance-sampling ratio that no longer reflects a small,
  reliable correction.
- **tanh-squash log-prob correction, worked out**: if action
  `a = tanh(z)·scale + shift` for `z ~ N(μ,σ)`, then `log π(a) ≠ log
  N(z;μ,σ)` — a change of variables (`a=g(z)`) always requires subtracting
  `log|dg/dz|` from the density in the ORIGINAL variable: `log p(a) = log
  p(z) - log|dg/dz|`. Here `da/dz = scale·(1-tanh(z)²)` (chain rule through
  `tanh`, whose derivative is `1-tanh(z)²`, times the constant `scale`), so:
  `log π(a) = log N(z;μ,σ) - log(scale) - log(1-tanh(z)²)`. Concretely: near
  `z=0`, `tanh(z)²≈0` so the correction is small (~`-log(scale)`); as `|z|`
  grows, `tanh(z)²→1` and `log(1-tanh(z)²)→-∞`, meaning the correction term
  `-log(1-tanh(z)²)→+∞` — the density genuinely blows up near the tanh's
  saturation boundary (a small range near `a=±scale` maps back to a HUGE
  range of `z`, so probability mass gets compressed into an
  infinitesimally thin slice of `a`-space right at the boundary). Skipping
  this silently biases the policy gradient (the density used for PPO's
  importance-sampling ratio wouldn't be the density of the action space
  actually being optimized).
- **Truncation vs. termination**: **termination** means the episode ended
  because the MDP itself ended (a true absorbing state — crashed, hit the
  target). **Truncation** means an artificial time limit cut the episode
  short — the "true" MDP would have continued. Bootstrapping matters here:
  on truncation, you should still add `γ·V(s_final)` to the last reward
  (there WOULD have been future reward, you just don't get to observe it);
  on true termination, you should NOT (there is no future).

## 0.5 Imitation learning: behavior cloning & DAgger

**Ladder: P → B → C.**

### 0.5.P Prerequisites

**Supervised regression, the ordinary kind.** You have pairs `(input,
correct_output)`, and you train a function `f(x;θ)` to output something
close to `correct_output` when fed `x` — the usual gradient-descent training
loop from 0.4.P, no reinforcement-learning trial-and-error needed, because
you're TOLD the right answer directly at every training example.

**Likelihood, and why "negative log-likelihood" is a loss you minimize.** A
probabilistic model doesn't just predict one number — it predicts a whole
PROBABILITY DISTRIBUTION over possible outputs (here: `N(μ,σ)`), and
"likelihood" is how much probability-density that distribution assigns to
the actually-observed correct answer. A good model should assign HIGH
probability to what actually happened, so training maximizes likelihood —
equivalently (since `log` is increasing, and it turns products of
probabilities across many samples into sums, easier to work with),
MINIMIZES the negative log of that likelihood (NLL). This is a completely
general recipe, and 0.5.B applies it to a Gaussian output specifically.

**Expert vs. student.** Here the "expert" is the hand-tuned PID controller
(0.3) — it already knows a working policy. The "student" is the neural
network being trained to copy it. Once trained, the student can (in
principle) run without the PID at all.

**Why "which states you visit" matters.** A policy doesn't just get graded
in a vacuum — it generates its OWN sequence of states by acting, and only
gets more experience in states it (or whoever generated the training data)
actually reaches. If the student ever drifts to a state the expert rarely
visited, there's no training data telling it what to do there — the seed of
the problem in 0.5.B.

### 0.5.B Basics

**The plain BC objective.** Regress the expert's action on the state: show
the network `(state, expert_action)` pairs and train it to predict
`expert_action` from `state`. If you fix `σ` to some constant and only ever
train `μ`, the Gaussian NLL from 0.5.P mathematically REDUCES to ordinary
mean-squared error (`(a*−μ)²`) — a Gaussian NLL with fixed spread is just a
fancier way of writing MSE. This project instead lets `σ` be LEARNED too,
which is where the next point comes from.

**Learned-`σ` NLL, derived from scratch, and its collapse.** For one action
dimension, the Gaussian's negative log-density at the true target `a*` is:
```
L = ½·[ ((a*−μ)/σ)²  +  2·ln σ  +  ln(2π) ]
```
(this comes directly from writing out `−ln(N(a*; μ,σ))` and simplifying —
take it as the standard formula for "how surprised is a Gaussian model by
seeing `a*`"). Training adjusts BOTH `μ` and `σ` to shrink `L`. What does the
optimal `σ` look like? Take the derivative of `L` with respect to `σ` and set
it to zero (the standard "find the minimum" move from calculus):
```
dL/dσ = −(a*−μ)²/σ³ + 1/σ = 0
      →  1/σ = (a*−μ)²/σ³
      →  σ² = (a*−μ)²
```
(for a batch of many samples, this generalizes to `σ² = mean((a*−μ)²)` — the
best-fit variance simply equals however much residual error is actually left
over). **Now the problem:** the PID "expert" is a deterministic function — same
state always gives the same action, zero true randomness. If the network's
`μ` fits it well, the residual `(a*−μ)` shrinks toward 0, and the formula
above says the LOSS-MINIMIZING `σ` shrinks right along with it, toward 0.
But look at what happens to `L` itself as `σ→0`: the `2·ln σ` term goes to
`−∞` — the loss keeps getting BETTER (more negative) the smaller `σ` gets,
forever, with no floor. Nothing in this loss on its own stops `σ` from
collapsing all the way to whatever floor the code happens to impose
elsewhere (here, a `tanh` clamp, 0.10.B) — this exact mechanism is the root
cause behind the std-collapse bug (5.1a, Part 12).

**Covariate shift — the problem this project's imitation stage exists to
avoid.** Plain BC only ever sees states the EXPERT visited. Once the trained
student makes even a small mistake, it can end up in a state the expert
rarely or never visited — no training signal there, so the student's next
action there is closer to a guess, which can lead to an even less familiar
state, and so on. This compounds: a classic result (Ross et al. 2011 [R4]) shows plain
BC's total error can grow QUADRATICALLY with how long the episode runs
(`O(εT²)`, `ε`=per-step error rate, `T`=episode length), versus only
LINEARLY (`O(εT)`) for the fix below — the difference between errors adding
up and errors snowballing.

**DAgger (Dataset Aggregation) — the fix.** Instead of only training on the
expert's own trajectory, run the STUDENT'S current policy, but at every
state it visits, ask the expert "what would you have done here" and use
THAT as the label; keep aggregating all rounds of this together and
retraining. This directly generates corrective training data for exactly
the states the student's own mistakes lead it into — attacking the
covariate-shift problem at its source instead of hoping it doesn't happen.

**BC/DAgger as a starting point for RL, not the end goal.** Pretraining a
policy this way buys immediate competence (a policy that already does
something reasonable, rather than starting from random weights) — but it
carries four specific hazards forward into the RL stage: (i) the collapsed
`σ` derived above; (ii) a critic (0.4.B) that was never trained at all
during imitation and starts essentially random, needing its own warmup
period before it can be trusted (0.10); (iii) fine-tuning can erode what was
learned (0.13); (iv) the student can never exceed what the expert itself
could do, unless the RL stage that follows actually improves on it.

**Alternatives to plain BC-then-finetune, for context.** Residual learning
(train the network to output only a small CORRECTION on top of the expert,
0.13 — this project's actual fix); BC-regularized RL (Rajeswaran et al. 2018
[R3]'s DAPG: add a penalty that keeps the policy close to the BC solution
while still training with RL);
or KL-to-reference regularization (a softer version of the same idea,
measured in probability-distribution distance rather than raw action
distance).

### 0.5.C In this project (current)
- **Pipeline:** `collect_demo_isaac.py` → `pretrain_bc.py` (Gaussian NLL) →
  `dagger_isaac.py` → `app/control/pretrained_bc_dagger.pt`.
- **In training:** the direct-policy path runs an on-policy imitation stage
  (`_run_imitation_stage_isaac`, 8% of the budget since 2026-09-24) after
  loading that checkpoint, then **resets `actor_log_std` to 0** (5.1a).
  Residual mode (0.13) skips the stage entirely and uses the checkpoint only
  for its trunk features (the actor head is zeroed).
- **Quality of the copy.** The cloned policy matches the PID almost exactly on
  success (≈ 0.95-0.98 grade in warmup) and on training reward (`recent_reward`
  ≈ 62-67 for the clone vs ≈ 68 for the pure PID in the residual run's warmup)
  — so imitation on 3-10 m is essentially lossless, and there is little left for
  RL to add there.
- **Detail (original notes):**

- **Behavior cloning (BC)**: supervised regression — collect
  `(state, expert_action)` pairs from a working controller (here, tuned
  PID), train a network to predict `expert_action` from `state` via MSE or
  (as this project does) **Gaussian negative log-likelihood**
  (`-log N(target; μ, σ)`) so each action dimension is normalized by its own
  learned variance — a large-magnitude dimension (thrust) doesn't dominate
  a small one (yaw torque) in the loss just because of scale.
- **Distributional shift / covariate shift**: BC's core failure mode — the
  network is only trained on states the EXPERT visits. Once the trained
  policy makes a small mistake, it enters a state the expert rarely
  visited, has no training signal there, makes a bigger mistake, and
  errors compound (a state distribution mismatch between training and
  deployment).
- **DAgger (Dataset Aggregation)**: fixes covariate shift by iterating:
  drive the environment with the CURRENT (student) policy, but label every
  visited state with what the EXPERT would have done there, then retrain
  on the growing aggregate of all rounds' data. This directly collects
  corrective labels for exactly the states the student's own mistakes lead
  it into — first-order fixing the mismatch BC has.
- **Recency weighting**: later DAgger rounds correct more-refined mistakes
  than round 0's baseline; weighting the retrain loss and buffer-eviction
  probability by `decay^(rounds_old)` lets old data fade in influence
  without being wholly discarded (which would waste already-collected
  coverage).
- **This project's DAgger has β=0, always** — no mixture-policy rollout.
  The original DAgger algorithm (Ross, Gordon & Bagnell 2011 [R4]) rolls out a
  MIXTURE policy (`β·expert + (1-β)·student`, annealing `β→0` across
  rounds) so early rounds still collect expert-quality trajectories. This
  project's on-policy imitation stage
  (`base_training._run_imitation_stage`) always drives the env with the
  CURRENT student's deterministic action and only queries the PID expert
  to LABEL that same visited state (`vec_env.step_with_pid_actions`) — the
  simplest DAgger variant, safe here specifically because the student
  starts from an already-competent BC/DAgger warm-start checkpoint rather
  than a random policy, so an early pure-student rollout is imperfect, not
  degenerate.

## 0.6 Physics engines: PhysX / Isaac Sim / Isaac Lab basics

**Ladder: P → B → C.**

### 0.6.P Prerequisites

**The simulation loop, as a repeating recipe.** Every fixed timestep `dt`:
look at the current state, work out what forces are acting, integrate them
into a new state (0.1.P), repeat. Everything in this project's physics (both
the numpy version and the Isaac Lab version) is this same 4-step recipe run
240 times a second.

**Why GPUs make MANY copies of this fast, but not one copy faster.** A CPU
has a few cores, each of which can run a DIFFERENT sequence of instructions
very fast and flexibly. A GPU instead has thousands of small, simple cores
that all run the SAME instruction at the same time, just on different data
— like thousands of calculators all doing "multiply these two numbers"
simultaneously, one pair per calculator. This is a great fit whenever you
have "the exact same computation, repeated many times over independent
copies of data" — which is exactly what "simulate 16,384 separate drones,
all running the identical physics equations" looks like. It does nothing to
speed up ONE single drone's simulation, though — that's still just one
sequential chain of steps.

**Tensors and batching — how you actually write "16,384 copies at once".**
Instead of a Python `for` loop calling the physics function 16,384 times
(slow — each call has overhead), you stack all 16,384 drones' states into
one big array (a "tensor") with an extra dimension for "which drone", and
write the physics math ONCE using array operations that apply to every
drone simultaneously. `position + velocity·dt` on a `(16384, 3)` array does
all 16,384 updates in a single instruction.

**Why Python's own overhead matters here.** Every line of plain Python code
(a loop iteration, reading a value out of a GPU tensor with `.item()`,
waiting for the GPU to sync up with the CPU) costs somewhere between
microseconds and milliseconds — tiny on its own, but if it happens once per
physics tick, 240 times a second, per env, it can easily become the actual
bottleneck, slower than the GPU math it's wrapped around. This is why "how
many times does Python code run per second of simulated time" matters as
much as raw physics speed (0.6.C, decimation).

### 0.6.B Basics

**Rigid-body solver pipeline, one tick.** Integrate forces into a new
velocity → check for collisions → resolve any contact/joint constraints (not
much of this here — this project's drone is a single free body, no joints)
→ integrate the new velocity into a new position. Splitting a tick into
several smaller "substeps" internally can make this more numerically
accurate/stable, at the cost of more compute per tick.

**Vectorized RL environments — running many independent episodes side by
side.** `num_envs` copies of the same environment step together, in
lockstep (all 16,384 advance one tick, then all 16,384 advance the next
tick). Each copy tracks its OWN episode progress independently — one env can
finish its episode (hit the target, or time out) while its neighbors are
still mid-flight. When an env finishes, it's typically reset immediately
within the same tick, which is why the OBSERVATION you get back for a
just-finished env may already belong to its brand-new NEXT episode, not the
one that just ended — a detail that matters for correctly crediting the
final reward (0.4.B's truncation/termination bootstrap). Total throughput is
`num_envs × (ticks per second per env)`.

**Sample-budget arithmetic — the numbers you actually configure a training
run with.** One "rollout" (batch of collected experience) is
`num_envs × num_steps` total environment-steps (`num_steps` = how many
ticks each env contributes before you stop and train on what you've got).
That rollout gets split into `num_minibatches` pieces for the actual
gradient updates, and you typically pass over the whole rollout
`num_epochs` times, so the total number of gradient steps per rollout is
`num_epochs × num_minibatches`. *Worked example, this project's actual
numbers:* `num_envs=16384`, `num_steps=128` → rollout =
`16384 × 128 = 2,097,152` steps; with `num_minibatches=64`, each minibatch
is `2,097,152/64 = 32,768` samples; with `num_epochs=10`, that's
`10 × 64 = 640` gradient steps per rollout. **The classic mistake:** some
constants elsewhere in a project are defined PER-ENV (e.g. "1,000,000 steps
per environment") while others are GRAND TOTAL across every parallel env —
treating one as the other is an instant factor-of-`num_envs` error (here,
possibly 100×+ off), which is why this project's own constants explicitly
label which kind each one is.

**Where the wall-clock time actually goes.** Roughly three things compete
for time each tick: the GPU physics itself, any remaining per-tick Python
code, and (separately) whatever evaluation/diagnostic code runs periodically.
Amdahl's law, informally: however much of the total time is spent in the
part that DOESN'T get faster when you add more parallel envs (e.g. a fixed
per-chunk Python evaluation loop) puts a hard ceiling on how much adding
more envs can actually help overall.

**The sim-to-sim gap.** Two different simulators of "the same" physical
system (here: the numpy oracle vs. Isaac Lab/PhysX) rarely behave IDENTICALLY
— they may use different integration schemes, different effective control
rates, or handle edge cases (like contacts) differently. A policy trained
against one can end up subtly specialized to that particular simulator's
quirks rather than to the true underlying physics, which is why this project
diff-tests the two against each other (0.1.C, Part 8) rather than assuming
they automatically agree.

### 0.6.C In this project (current)
- **Production shape:** `--num_envs 16384 --num-steps-per-chunk 128` →
  2,097,152 steps per chunk, ~61 chunks for 128 M steps, about 1.6 min per
  chunk. Chunk *count* matters: every fixed-chunk-count floor
  (`MIN_WARMUP_CHUNKS = 16`, reset windows, the promotion gate at chunk 40+)
  silently changes its share of the run if `num_envs` is raised without
  lowering `num_steps_per_chunk` (Part 12, entry 1).
- **Per-chunk evaluation is CPU-side numpy** (30 model + 30 PID episodes on
  the numpy oracle), not Isaac — see the sim-to-sim caveat and the control-rate
  difference (240 vs 60 Hz) in 0.1.C.
- **Episode length:** `episode_length_s = MAX_POLICY_STEPS × PHYSICS_DT = 62.5 s`.
- **Detail (original notes):**

- **Isaac Sim / Kit / PhysX**: NVIDIA's GPU-accelerated physics+rendering
  stack. **PhysX** is the actual rigid-body solver (collision, contact,
  integration). **Isaac Sim** is the simulation application built on Kit
  (NVIDIA's app framework) that hosts PhysX plus rendering/sensors.
  **Isaac Lab** is a thin RL-oriented framework on top of Isaac Sim
  providing the `DirectRLEnv`/gym-style API, vectorized (`num_envs`
  parallel scenes) stepping, and task registration.
- **Decimation, concretely**: this project's physics step is `dt=1/240`
  (240 Hz). With `decimation=4` (this project's current value, raised from
  1), the policy only sees a new observation and issues a new action every
  4th physics tick — an effective 60 Hz control rate — while every
  intermediate tick still integrates motor lag and rigid-body dynamics at
  the full 240 Hz. Raising decimation trades control responsiveness for
  training throughput (fewer policy forward passes and PPO bookkeeping
  calls per second of simulated time) WITHOUT touching physics fidelity —
  motor lag's time constant `τ` and the rigid-body integrator are still
  resolved every 1/240s regardless of what decimation is set to.
- **RigidObject vs. Articulation**: a `RigidObject` is a single free rigid
  body with 6 degrees of freedom (position+orientation), no internal
  joints. An `Articulation` is a kinematic tree of rigid bodies connected
  by joints (like a robot arm), solved with a reduced-coordinate solver
  that maintains its own generalized mass matrix. This project **switched
  from Articulation to RigidObject** for the drone body — see Part 8's bug
  writeup for exactly why (the Articulation solver cached a mass-dependent
  quantity from the asset's *original* USD-authored mass that a post-hoc
  override didn't fully invalidate).
- **`decimation`**: how many PhysX substeps (each `sim.dt` long) run per
  *policy* step. Policy decisions happen once every `decimation` physics
  ticks — this is how you make training faster (fewer policy-network
  forward passes and PPO-relevant bookkeeping per second of simulated
  time) WITHOUT changing the physics timestep itself (which would change
  numerical behavior and invalidate anything tuned against the original
  `dt`, like PID gains or step-budget constants).
- **`num_envs`**: Isaac Lab clones the whole scene `num_envs` times and
  steps every clone in lockstep on the GPU — the entire point of moving off
  a CPU-bound Python for-loop over environments (this project's numpy
  `SubprocVecBaseDroneEnv`/`VecBaseDroneEnv`).

## 0.7 Potential-based reward shaping

**Ladder: P → B → C.**

### 0.7.P Prerequisites

**Sparse vs. dense rewards.** A "sparse" reward gives feedback only rarely —
e.g. a big bonus only at the exact moment the drone reaches the target, and
0 every other step. A "dense" reward gives SOME feedback every single step
(here: a little credit just for having moved closer since last tick).

**A potential function.** Just a rule that assigns one number to every
possible state — a "how good is this situation, ignoring how I got here"
score. This project's potential is `φ(s) = −(distance to target)`: farther
away scores lower (more negative), closer scores higher, and exactly at the
target scores 0.

**Telescoping sums — the algebra trick this section's proof rests on.** If
you have a series of terms that are each "next value minus previous value",
almost everything cancels when you add them up:
```
(a1−a0) + (a2−a1) + (a3−a2) = a3 − a0
```
*Why:* `−a1` from the first term cancels `+a1` from the second; `−a2` from
the second cancels `+a2` from the third — everything in the MIDDLE cancels,
leaving only the very first and very last terms. This one pattern is the
entire proof behind the theorem in 0.7.B.

### 0.7.B Basics

**The sparse-reward problem, concretely.** If the ONLY reward is a bonus for
hitting the target, then early in training (random, untrained actions) the
drone essentially never reaches it — so almost every rollout returns nothing
but zeros, giving the learning algorithm no signal at all about which random
flailing was "closer to working" than which other flailing. Exploration and
credit assignment (0.4.B) become extremely hard with no partial credit.

**Reward shaping — adding partial credit, and the risk it creates.** The
obvious fix is to add a bonus for PROGRESS (e.g. "+1 for every meter you got
closer this step"). The danger: an arbitrarily-chosen bonus can change what
the OPTIMAL behavior actually is — the agent might learn to farm the bonus
in a way that doesn't match the real goal (e.g., wiggling back and forth to
repeatedly collect a badly-designed "moved" reward).

**Potential-based shaping — a specific recipe proven not to have that
problem.** Ng, Harada & Russell (1999 [R2]) showed that if your extra shaping
reward, added on top of the real reward, has this exact form:
```
F(s, s') = γ·φ(s') − φ(s)
```
(`s`=state before the step, `s'`=state after, `γ`=discount from 0.4.B), then
the shaping is *provably* transparent to the optimal policy — whatever
sequence of actions was optimal WITHOUT the shaping term is still exactly
optimal WITH it. **Here's the actual proof**, which is just the telescoping
trick from 0.7.P applied to the full discounted sum of shaping rewards over
an episode `s_0, s_1, ..., s_T`:
```
Σ_{t=0}^{T-1} γᵗ·F(s_t,s_{t+1})
  = Σ γᵗ·(γ·φ(s_{t+1}) − φ(s_t))
  = Σ γ^{t+1}·φ(s_{t+1})  −  Σ γᵗ·φ(s_t)
  = [γφ(s_1) + γ²φ(s_2) + ... + γᵀφ(s_T)]  −  [φ(s_0) + γφ(s_1) + ... + γ^{T-1}φ(s_{T-1})]
  = γᵀ·φ(s_T) − φ(s_0)                (every middle term appears once with + and once with −, and cancels)
```
The result, `γᵀφ(s_T) − φ(s_0)`, depends ONLY on the very first and very
last state — never on which path of actions got you from one to the other.
*Worked numeric check:* `T=2`, `γ=0.9`, states with `φ(s_0)=−10`,
`φ(s_1)=−6`, `φ(s_2)=−2` (getting closer each step): shaping rewards are
`F_0 = 0.9·(−6)−(−10) = 4.6` and `F_1 = 0.9·(−2)−(−6) = 4.2`. Discounted sum:
`F_0 + γ·F_1 = 4.6 + 0.9·4.2 = 4.6+3.78 = 8.38`. Formula's shortcut:
`γ²φ(s_2)−φ(s_0) = 0.81·(−2) − (−10) = −1.62+10 = 8.38` — matches, confirming
the whole messy middle really did cancel.

**This project deliberately breaks the theorem, twice — on purpose, not by
accident.** (1) It uses a PLAIN difference `φ(s')−φ(s)` (no `γ` multiplying
the first term) instead of the exact `γφ(s')−φ(s)` the theorem requires —
because with PPO's own `γ<1` in there, standing PERFECTLY STILL at a
non-target state (so `φ(s')=φ(s)`) would still yield a reward of
`(γ−1)·φ(s)`, which is POSITIVE (since `φ` is negative and `γ−1` is
negative, their product is positive) — a free reward for doing nothing,
which the exact formula would reward every single step forever. Losing the
strict "provably optimal-policy-invariant" guarantee is judged worth it to
remove that standing-still exploit. (2) It normalizes `φ` by that episode's
OWN starting distance (`φ(s) = −distance/start_dist`), so a 3m task and a
10m task produce comparably-sized shaping signals — but this means `φ` is no
longer a single, fixed function shared identically across every episode in
the theorem's strict sense (it still IS one fixed function within any single
episode, since `start_dist` doesn't change mid-episode — just not literally
identical across different episodes' distributions).

**Non-potential reward terms, i.e. the rest of the reward function.** Not
everything has to be (or should be) potential-based — a one-time bonus for
reaching a milestone, a penalty for going out of bounds, a small per-step
time penalty: these are deliberate INCENTIVES layered on top, not attempts
to preserve the theorem's guarantee.

**Reward hacking / Goodhart's law.** Any reward function you write down is a
PROXY for what you actually want. If maximizing the proxy ever diverges from
achieving the real goal, a sufficiently good optimizer will find and exploit
that gap — worth actively watching for whenever training behavior looks
"technically correct but clearly not what was intended".

**Reward scale interacts with everything else.** How big the reward numbers
are affects the critic's value-loss scale (bigger rewards → bigger squared
errors to regress, 0.4.B), how sensitive training is to the learning rate,
and even how far ahead the effective discount horizon (0.4.B) reaches in
practice — reward magnitude isn't a cosmetic choice.

**The reward is not the same thing as whatever number you use to judge
success.** PPO optimizes the training reward, full stop — it has no idea a
separate "grade" metric even exists. If that grade doesn't track the reward
closely, a policy can genuinely improve its reward while looking flat or
worse on the grade (0.14) — always check whether the two actually agree
before trusting either one alone.

### 0.7.C In this project (current)
- **Terms** (`rewards.py`): hit reward `HIT_REWARD = 50` at
  `HIT_THRESHOLD = 0.25 m`; milestone bonuses `(10, 15, 20)` at fractions
  `(0.25, 0.5, 0.75)` (their total plus the hit is `APPROACH_MILESTONE_BUDGET
  = 96.72`, the max reward excluding time pressure, used to scale the per-step
  time penalty); out-of-bounds `−1.5` (radius `OOB_RADIUS`), attitude `−1.0`;
  zone-exit penalties `−1.0` / `−2.0` at radii 1.0 / 0.5 m; a stability penalty
  `−0.1·(|vel|+|roll|+|pitch|)` inside the outer zone; `GAMMA = 0.97`.
- **A finished successful episode returns roughly 65-90 reward** (the logged
  `recent_reward`, a 10-episode window, is noisy — read 5+ chunk averages).
- **The grade is not the reward.** `grade = success − 0.3·error_ratio −
  0.1·grad_ratio` (`w_time = 0`), so speed is rewarded by the training reward
  but ignored by the grade (0.14).
- **Detail (original notes):**

A shaping term `F(s,s') = γ·φ(s') - φ(s)` added to a reward is
**policy-invariant** (Ng, Harada & Russell 1999 [R2]) — the optimal policy under
the shaped reward is provably identical to the optimal policy under the
original reward alone, for any potential function `φ`. This project's
`reward_func` uses a **plain difference** `φ(s') - φ(s)` (not
`γ·φ(s') - φ(s)`) — deliberately deviating from the strict theorem, because
using PPO's own `γ<1` there left a residual `(γ-1)·φ(s)` reward every step
even standing perfectly still (since `φ` is negative, that residual is
positive — see `rewards.py`'s in-file comment). This is a real, documented
design choice worth understanding: strict policy-invariance was traded for
removing a standing-still exploit.
- **The actual potential function here**: `rewards.py`'s `reward_func` sets
  `φ(s) = -L1_distance(pos, target) / start_dist` — negative L1 distance to
  target, normalized by THAT EPISODE's own starting distance (a constant
  fixed at reset, not a function of the live state). A second deviation
  from the textbook theorem, alongside the plain-difference one above:
  normalizing by a per-episode constant keeps shaping magnitude comparable
  whether the target started 3m or 10m away, at the cost of `φ` no longer
  being a function of state alone in the purest sense across DIFFERENT
  episodes — the invariance guarantee still holds WITHIN one episode
  (`start_dist` is fixed once sampled), just not as a single global `φ`
  shared across the whole training distribution.

## 0.8 Optimization: SGD, Adam/AdamW, learning-rate schedules, clipping, batch size

**Ladder: P → B → C.**

### 0.8.P Prerequisites

**The loss as a landscape you're trying to walk downhill on.** Picture the
loss `L(θ)` as a hilly terrain, where `θ` (all the network's weights) is
your current position and `L` is your altitude. The gradient `∇L` points in
the UPHILL direction, so `−∇L` points downhill — that's the direction every
training step moves in. Real networks have huge, bumpy landscapes (many
valleys, flat plateaus, saddle points) — training only ever finds A good
valley, never provably THE best one.

**Why a minibatch's gradient is noisy.** You compute the gradient using only
a SAMPLE of data (a batch), not the entire dataset/all possible experience —
so it's an estimate, not the exact true gradient, and (0.4.P) that
estimate's error shrinks like `1/√batch_size` as the batch gets bigger.

**Exponential moving average (EMA) — "a running average that forgets old
data gradually".** `m ← β·m + (1−β)·x` (`β` close to 1, e.g. 0.9) mixes in
a little of the newest value `x` each step while keeping most of the old
running average `m`. This behaves like an average over roughly the last
`1/(1−β)` samples — *example:* `β=0.9`: `1/(1−0.9)=10`, so it's roughly
tracking the last 10 values; `β=0.999` tracks roughly the last 1000.

### 0.8.B Basics

**Plain gradient descent (SGD).** `θ ← θ − lr·g` (`g`=this step's gradient,
`lr`=learning rate, a small constant you choose). The step size is directly
PROPORTIONAL to the gradient — a big gradient gives a big step, a tiny
(possibly mostly-noise) gradient gives a tiny step. **Momentum** is one fix
for how noisy this can get: instead of stepping by the raw gradient, keep an
EMA of past gradients and step by THAT — noise that points different
directions from step to step partially cancels out in the average, while a
consistent signal accumulates.

**Adam — momentum on both the gradient AND its typical size.** Adam keeps
TWO running averages: `m` (an EMA of the gradient `g` itself, `β₁=0.9`) and
`v` (an EMA of `g²`, the gradient SQUARED, `β₂=0.999` — this tracks roughly
how big the gradient typically is, ignoring its sign). The update (after a
small correction for `m`/`v` starting at zero, "bias correction", omitted
here for brevity) is:
```
θ ← θ − lr · m/√v
```
Two consequences that matter directly for this project:
1. **The step size is roughly `lr`, no matter how small the true gradient
   is.** Since you're dividing the gradient's running average `m` by an
   estimate of its own typical SIZE `√v`, a consistently-tiny-but-nonzero
   gradient still produces a step of roughly size `lr` (the "how big" cancels
   out on both top and bottom) — unlike plain SGD, where a tiny gradient
   automatically means a tiny step.
2. **Pure noise still produces movement.** If the true gradient is exactly
   0 (an optimum) and all you're measuring is symmetric random noise, `m`
   and `v` still settle to characteristic nonzero values, and it works out
   that `|m|/√v ≈ √((1−β₁)/(1+β₁)) = √(0.1/1.9) ≈ 0.23` — i.e. Adam still
   takes steps of around a QUARTER of `lr`, in a random direction, every
   single update, purely from noise. Repeated many times, this is a random
   walk AWAY from the optimum, not a settling-down — the mechanism directly
   behind the training-stage slide observed in this project (0.4.B, 0.13).

**AdamW — fixing an interaction with weight decay.** "Weight decay"
(shrinking weights slightly toward 0 every step, to discourage overly large
weights) is normally added directly into the loss as an extra penalty term.
But that penalty then gets fed through Adam's own `m`/`v` normalization
along with everything else, distorting how much decay actually happens.
AdamW instead applies the decay as its own SEPARATE step,
`θ ← θ − lr·wd·θ`, outside Adam's normalization — decoupled, hence the name.

**Learning-rate schedules — changing `lr` over the course of training.**
Common patterns: warmup (start `lr` low, ramp it up over the first few
steps, to avoid a big destabilizing jump before the network has settled at
all), step decay (drop `lr` by a fixed factor at set points), and **cosine
decay**, used here:
```
lr = lr_min + ½·(lr_max − lr_min)·(1 + cos(π·p))
```
`p∈[0,1]` is "how far through training you are" (0=start, 1=end). *Worked
example:* `lr_max=1.451e-4`, `lr_min_ratio=0.1621` (so `lr_min =
0.1621·1.451e-4 ≈ 2.35e-5`), at `p=0.5` (halfway):
`cos(π·0.5)=cos(90°)=0`, so `lr = 2.35e-5 + ½·(1.451e-4−2.35e-5)·(1+0) =
2.35e-5 + 6.14e-5 ≈ 8.5e-5` — roughly the midpoint between max and min, as
you'd expect halfway through a smooth decay. **A subtlety worth knowing:**
`cos` is a PERIODIC function — if you ever feed `p>1` into this formula
without clamping it first, the schedule doesn't just keep decreasing, it
starts climbing back up as `cos` cycles around again. Any code using this
formula needs to clamp `p` to `[0,1]` itself; the formula won't do it for you.

**Gradient clipping — capping how big any single step can be.** Compute the
gradient's overall size (its "norm", `‖g‖` — think Pythagorean-theorem
length across all its numbers combined), and if it exceeds some limit `c`,
scale the WHOLE gradient down by `c/‖g‖` so its new size is exactly `c`
(direction unchanged, just shorter). This caps rare huge spikes from
blowing up training. Combined with Adam's own normalization (above), clipping
mostly changes how different parameters' updates are weighted RELATIVE to
each other during a spike, more than it changes the overall step size, since
Adam was already roughly normalizing step size to `lr` regardless.

**Batch size and learning rate — why "bigger batch, bigger lr" doesn't
automatically apply here.** In ordinary supervised learning, common
heuristics say to scale `lr` up (roughly linearly, or by `√batch_size`) when
you increase the batch size, since a bigger batch's gradient is less noisy
(0.8.P) and can support a bigger step. In THIS kind of RL training, the
actual limiting factor is different: it's how far the policy's probability
distribution is allowed to drift per update (measured by KL divergence,
0.9), not gradient noise directly. A bigger batch gives a MORE COHERENT
gradient (same underlying reasoning as 0.8.P: less noise) — and a more
coherent gradient at the SAME `lr` pushes the policy further in a consistent
direction, not less — so the usual "bigger batch → bigger lr is safe"
heuristic can actively point the wrong way here (0.13).

### 0.8.C In this project (current)
- **Optimizer:** `AdamW(lr = 1.451e-4, weight_decay = 3.828e-6)`. At that
  weight decay the decay term is ~5e-10 of a weight per step — negligible.
- **Schedule** (`base_training_isaac.scheduled_lr`): flat `1.451e-4` through
  critic warmup (the critic was still improving at chunk 16), then a cosine
  over the warmup+PPO span down to a floor of `0.1621×`, times a 3-chunk ramp
  (`1/3, 2/3, 1`) after each unfreeze (`UNFREEZE_LR_RAMP_CHUNKS`). Realized
  values in the residual run: 4.2e-5 at the first training chunk → 1.2e-4
  → 4.5e-5 at chunk 44.
- **Clipping:** `max_grad_norm = 1.347`; pre-clip `grad_norm` averaged 8.0 in
  the residual run (up to 18), so clipping was active every chunk.
- **Steps per chunk:** direct-policy runs early-stopped after epoch 1 → 64
  Adam steps per chunk; residual runs ran all 10 epochs → 640.
- **Batch:** production minibatch 32,768 vs the Optuna search's 1,024 (32×).
- **Evidence on lr** (Optuna, 100 trials, minibatch 1,024; `delta` =
  post-unfreeze mean grade minus the trial's own baseline):

  | lr bucket | n | mean delta |
  |---|---|---|
  | < 1e-5 | 28 | −0.37 |
  | 1e-5 … 1e-4 | 59 | −0.26 |
  | 1e-4 … 1e-3 | 8 | −0.23 |
  | ≥ 1e-3 | 5 | ≈ −0.99 (collapsed) |

  Inside the working band, lower lr did *worse* (Spearman lr-vs-delta +0.55;
  likely because critic warmup is lr-limited); above 1e-3 everything
  collapsed. Hence trial 137's `2.9e-4` was halved to `1.45e-4`, **not**
  scaled up by the batch/step ratios (which would have put it near 1e-2).

## 0.9 Probability and statistics for training and evaluation

**Ladder: P → B → C.**

### 0.9.P Prerequisites

**Random variable, mean, variance.** A random variable is just "a number
that comes out a bit differently each time you measure it" (a die roll, a
hit-or-miss episode outcome). Its **mean** (`E[X]`, 0.4.P) is the long-run
average; its **variance** measures how spread out the results are:
`Var(X) = E[(X−mean)²]` — average squared distance from the mean. A useful,
easy-to-derive fact used below: for a constant `c`, `Var(c·X) = c²·Var(X)`
(scaling the variable by `c` scales its SPREAD-SQUARED by `c²`, since every
`(X−mean)` term inside the square also gets scaled by `c`).

**Independence, and why variances of independent things just ADD.** Two
random variables are independent if knowing one tells you nothing about the
other (two separate coin flips). For independent `X` and `Y`,
`Var(X+Y) = Var(X)+Var(Y)` — no cross-terms to worry about, unlike with
means of correlated things. This one fact is the entire engine behind the
standard-error derivation below.

**The binomial distribution.** `n` independent yes/no trials, each with
success probability `p` (like `n` episodes, each either a "hit" or not).
Each single trial, treated as a 0/1 number, has `Var = p(1−p)` — you don't
need to derive this one, just recognize it (it peaks at `p=0.5`, where
outcomes are most unpredictable, and shrinks to 0 as `p→0` or `p→1`, where
the outcome is nearly certain either way).

**Logarithms, briefly.** `ln(ab) = ln(a) + ln(b)` (turns multiplication into
addition — this is exactly why "negative log-likelihood", 0.5.P, turns a
PRODUCT of per-sample probabilities into a SUM). `ln` of a number close to 0
is a large NEGATIVE number (`ln(0.01) ≈ −4.6`), which is why a model that's
very rarely surprised (assigns high probability to what happens) achieves a
loss close to 0, while one that's frequently very surprised (assigns tiny
probability to what happens) can have a loss growing arbitrarily large.

### 0.9.B Basics

**Binomial noise in an evaluation — derived, not just quoted.** Say you run
`n` episodes, each a Bernoulli trial (hit=1, miss=0) with true success rate
`p`. Your MEASURED hit rate is `p̂ = (1/n)·(sum of the n outcomes)`. Using
the two facts from 0.9.P: the sum of `n` independent trials has variance
`n·p(1−p)` (variances add); dividing by `n` to get the average scales
variance by `(1/n)² = 1/n²` (the scaling-by-constant rule), so
`Var(p̂) = n·p(1−p)/n² = p(1−p)/n`. Taking the square root (variance → the
more interpretable "typical spread" number, called standard error):
```
standard error(p̂) = √(p(1−p)/n)
```
*Worked examples:* `n=5, p=0.8`: `√(0.8·0.2/5) = √0.032 ≈ 0.179` — huge; on
only 5 episodes, one single flipped outcome moves your measured rate by
`1/5=0.2`, comparable to the whole noise band. `n=30, p=0.9`:
`√(0.9·0.1/30) = √0.003 ≈ 0.055` — much tighter. Worst case is always
`p=0.5` (variance `p(1−p)` peaks there): at `n=30`, `√(0.25/30) ≈ 0.091`.
**The practical rule this justifies:** if two measured hit-rates differ by
less than roughly this standard error, you cannot conclude one is really
better — the gap could easily be pure sampling luck.

**Averaging shrinks noise, but only removes NOISE, not real drift.** The
mean of `k` independent, equally-noisy measurements has standard error
`1/√k` times a single measurement's — same reasoning as above, generalized.
But this only applies to measurements that are actually independent samples
of the SAME underlying quantity; if the underlying quantity is itself
changing over time (a policy genuinely getting better or worse), averaging
those changing values together doesn't "clean up noise", it blurs together
a real trend with whatever noise is also present. To measure pure noise, you
need a period where the underlying thing you're measuring truly isn't
changing (e.g. a frozen policy).

**Entropy of a Gaussian — what it means, and the formula (stated, not fully
derived — the derivation needs an integral beyond this section's scope).**
Entropy measures "how spread out / unpredictable" a distribution is. For a
1-D Gaussian `N(μ,σ)`, working through its definition gives:
```
H = ½·ln(2π·e·σ²) = ln(σ) + ½·ln(2πe) ≈ ln(σ) + 1.419      (in "nats", the natural-log unit of information)
```
Notice `μ` doesn't appear anywhere — entropy only depends on `σ` (how spread
the bell curve is), never on WHERE it's centered, which makes sense: shifting
a bell curve left or right doesn't change how "spread out" it is. *Numeric
check:* `σ=0.0837` gives four independent dimensions each contributing
`ln(0.0837)+1.419 ≈ −2.481+1.419 = −1.062`; four dimensions add up (entropy
of independent things just adds, same logic as variance):
`4·(−1.062) = −4.245`. **A frozen `σ` therefore mathematically GUARANTEES a
frozen entropy** — if you ever see entropy dead flat over many training
steps, `σ` itself must be dead flat too, whatever else is changing.

**KL divergence between two Gaussians — derived.** KL divergence measures
"how different is distribution `Q` from distribution `P`" (used by PPO's
trust-region check, 0.4.B). For two Gaussians with the SAME `σ` but
different means `μ1` (under `P`) and `μ2` (under `Q`), the general
definition `KL(P‖Q) = E_P[ln(p(x)/q(x))]` simplifies with pure algebra:
```
ln(p(x)/q(x)) = −(x−μ1)²/(2σ²) − (−(x−μ2)²/(2σ²)) = [(x−μ2)² − (x−μ1)²] / (2σ²)
```
Expand both squares: `(x−μ2)²−(x−μ1)² = 2x(μ1−μ2) + (μ2²−μ1²)`. Now take the
expectation under `P` (mean `μ1`), using only the fact that `E_P[x]=μ1` by
definition of what "mean" means:
```
KL(P‖Q) = [2μ1(μ1−μ2) + (μ2²−μ1²)] / (2σ²)
        = [μ1² − 2μ1μ2 + μ2²] / (2σ²)          (regroup the terms)
        = (μ1−μ2)² / (2σ²)
```
So, writing `Δμ = μ1−μ2`: **`KL = Δμ²/(2σ²)`.** Notice `σ` is in the
DENOMINATOR — a SMALLER `σ` makes the SAME mean shift produce a BIGGER KL.
*Worked example:* `σ=0.0707`, `Δμ=0.02`: `KL = 0.02²/(2·0.0707²) =
0.0004/0.01 = 0.04` — already about 4.7× over a `target_kl=0.0086` trust-
region limit, from a genuinely tiny mean shift. This is exactly why small
exploration noise makes PPO's KL check hypersensitive: the same actual
change in behavior registers as a much bigger "distance moved" when `σ` is
small.

**Change of variables (used by the tanh-squash log-prob correction, 0.4).**
When you transform a random variable through some function (`a=g(z)`), its
probability DENSITY doesn't just carry over unchanged — it picks up a
correction factor `|dg/dz|⁻¹` (how much the transformation stretches or
compresses space at that point). Skipping this correction silently uses the
wrong density wherever `g` isn't perfectly evenly-stretching.

**Rank correlation (Spearman) — correlation that only cares about ORDER.**
Instead of correlating the raw numbers, first replace every value with its
RANK (1st smallest, 2nd smallest, ...), then correlate those ranks. This
makes it robust to any relationship that's consistently increasing or
decreasing but not necessarily a straight line. On noisy real-world data with
around 100 points, a value with `|ρ|<0.2` is weak enough to be
indistinguishable from noise — don't read much into it.

**Correlation is not causation — and time is the classic trap.** If two
things both steadily trend in the same direction over time (e.g. both
increase as training progresses), they'll show a strong correlation with
each other even if NEITHER one is causing the other — both are just riding
the same underlying trend (here: elapsed training time). *Example from this
project:* `value_loss` (falling as the critic learns) correlated with grade
at `ρ = +0.67/+0.84/+0.92` across three runs — tempting to read as "a better
critic causes a better grade", but both are simply trending together over
the SAME training run, which isn't proof of either causing the other.

**Selection bias, a.k.a. "winner's curse".** If you run many noisy trials
and pick the single best-scoring one, that measured score is inflated by
BOTH genuine quality AND simple luck — the trial that happened to get lucky
on top of being decent will usually beat one that was slightly better but
had average luck. The "winner's" TRUE quality is usually a bit lower than
its measured score suggested.

### 0.9.C In this project (current)
- `N_DIAGNOSTIC_EPISODES = 30` (raised from 10 on 2026-09-24 because 10 was
  noisy enough to false-trigger resets); the Optuna objective used **5**, so
  its grades were quantized in steps of 0.2 (Part 12, entry 6).
- **Measured eval noise:** the pooled spread of grade over frozen-actor chunks
  (same policy every chunk) is ≈ 0.06 per chunk, so means of 12 chunks resolve
  differences of only ≈ 0.04. `analyze_optuna_isaac.py` prints this number and
  counts how many trials are within it of the best (10 of 100).
- **Check on the entropy formula:** at `σ = 0.0837` four dimensions give
  `4 × (1.419 + ln 0.0837) = −4.245`, and the logged `entropy_loss`
  (`−entropy`) is `+4.244`. ✔
- **Grade noise budget:** a 30-episode grade moves by one episode = 0.033 of
  success — read trends over 5+ chunks.

## 0.10 Neural-network design for actor-critic

**Ladder: P → B → C.**

### 0.10.P Prerequisites

**A linear layer, spelled out.** `y = W·x + b` — `x` is the input vector,
`W` a matrix of learnable weights, `b` a learnable offset vector, `y` the
output. This is nothing more than "a weighted sum of the inputs, plus an
offset", computed for every output number at once via matrix multiplication.

**Why you need a nonlinearity between layers.** Stacking two linear layers
back to back, with NOTHING in between, is mathematically still just ONE
linear layer (`W2·(W1·x+b1)+b2` is itself of the form `W'x+b'`) — no matter
how many you stack, you'd never get anything more expressive than a single
linear function. Inserting a nonlinear function (`tanh`, `ReLU`) between
layers is what actually lets a deep stack represent CURVED, non-linear
relationships.

**Backpropagation, in one sentence.** It's just the chain rule from
calculus, applied layer by layer backward through the network, to compute
how the FINAL loss depends on every single weight — however deep the
network, this is one mechanical, repeatable algebra procedure, not a
different technique per layer.

**Universal approximation, informally.** A wide-enough network with one
hidden layer can, in principle, approximate any reasonably smooth function
arbitrarily well. In practice, depth (more layers) and width (bigger layers)
trade off — more capacity to represent complex functions, at the cost of
being harder to train.

### 0.10.B Basics

**The MLP trunk.** A stack of Linear+nonlinearity layers that turns the raw
observation (a 23-number vector describing the drone's state) into
"features" — a transformed representation the later heads (below) actually
use to make decisions.

**Deriving `tanh`'s derivative, and why saturation matters.**
`tanh(x) = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)`; its derivative works out (a standard calculus
result) to `tanh'(x) = 1 − tanh(x)²`. Plugging in a few values shows the
shape: `tanh'(0)=1` (steepest, most sensitive to change), `tanh'(2)≈0.071`,
`tanh'(3)≈0.0099`, `tanh'(5)≈1.8e-4` — the derivative collapses toward 0 as
`|x|` grows ("saturation"). **Why this matters for training:** a parameter's
gradient gets MULTIPLIED by this derivative on its way back through the
`tanh` (chain rule, 0.10.P) — so a parameter sitting several units deep in
the flat tail has its incoming gradient shrunk by that same tiny factor,
making it move extremely slowly. Combine this with Adam's behavior (0.8.B:
its step size is roughly `lr` regardless of how small the underlying
gradient is, so a smaller gradient just means MORE steps needed, not zero
movement) — escaping a deep tail can take tens to hundreds of training
chunks with a perfectly consistent gradient direction, and considerably
longer with a NOISY gradient direction (which is the realistic case). Worse:
whatever quantity is computed FROM the saturated parameter (here, the policy's
`σ`, next paragraph) is itself multiplied by that same near-zero derivative,
so it can look completely frozen in the logs even while Adam is technically
still nudging the underlying raw parameter. This exact chain is the
mechanism behind the `actor_log_std` collapse (0.5.B, 5.1a).

**How this project's policy `σ` is actually computed.** Rather than have
the network directly output a std value (which would need to be forced
positive and bounded some other way), it keeps one learnable raw number per
action dimension and squashes it:
```
log σ = log_std_min + ½·(log_std_max − log_std_min)·(tanh(raw) + 1)
```
Since `tanh(raw)` is always between −1 and +1, `(tanh(raw)+1)` is always
between 0 and 2, and the whole expression is always between `log_std_min`
and `log_std_max` — the squash guarantees `σ` stays in a safe, bounded range
no matter what value the raw parameter takes, including deep in a saturated
tail. *Worked example:* `raw=0` (right in `tanh`'s most sensitive region):
`tanh(0)=0`, so `log σ = log_std_min + ½·(log_std_max−log_std_min)` — exactly
the MIDPOINT of the allowed range. With this project's `[−3.0,−2.3]`:
midpoint `= (−3.0+−2.3)/2 = −2.65`, giving `σ = e^{−2.65} ≈ 0.0707` — matching
the observed reset value exactly, and landing in `tanh`'s live, fast-moving
middle rather than a saturated tail.

**Sharing one trunk between the actor and critic — a trade-off, not a free
lunch.** Using the SAME trunk features for both the action-choosing head
and the value-predicting head saves parameters and lets both benefit from
whatever useful features either one discovers. The cost: gradients from
BOTH heads flow back into that same shared trunk and reshape it — a critic
that's learning aggressively (often a much bigger raw loss than the actor's)
can push the shared features in directions that help ITS job at some cost to
the actor's. Using fully separate networks, or having the critic read the
trunk's output through a `.detach()` (a command that says "use this value,
but don't let gradients flow backward through it") removes that
interference, at the cost of a weaker critic (it no longer benefits from
whatever the actor's gradient would have taught the shared features).

**"Freezing" a parameter, precisely.** Setting `requires_grad_(False)` on a
parameter tells the framework "don't bother computing a gradient for this
one" — the optimizer, seeing no gradient (`.grad is None`) for it, simply
skips updating it that step. Important edge case: if you freeze
EVERY SINGLE parameter in the network, there's nothing left with a gradient
anywhere in the whole computation, so calling backward() on the loss
actually errors out (there's no path for gradients to flow through at all) —
which is why a genuinely "no updates happen at all, not even to the critic"
period needs a different mechanism (`skip_update`: just don't call the
update function at all that chunk) rather than freezing every parameter.

**Why initialization is a real design choice, not an afterthought.**
Starting a layer's weights at exactly 0 (rather than the network's usual
small-random initialization) makes that layer's OUTPUT exactly 0 regardless
of its input — a deliberate "start as a no-op, let training build up from
there" choice, used directly by this project's residual-policy mode (0.13):
zeroing the action-output head means the policy starts as EXACTLY the
baseline controller, with zero risk of an untrained random output making
things immediately worse.

**Checkpoints are just named tensors.** Saving a model's weights produces a
"state dict" — essentially a lookup table from layer-name strings to weight
tensors. Loading it back into a DIFFERENT but related architecture (e.g. one
with a dropout layer inserted, shifting every later layer's internal index)
requires matching keys up carefully by their actual role, not blindly by
position — mismatched key names silently load garbage or nothing at all
into the wrong layers.

### 0.10.C In this project (current)
- **Architecture** (`train.ActorCritic`): observation 23-d → 4 × `Linear(64)`
  with `Tanh` between the first three → ReLU → heads `actor_mean`
  (64→4), `critic_head` (64→1), plus `actor_log_std` (4 free numbers).
  Parameter count = `1,536 + 3·4,160 + 260 + 65 + 4 = 14,345`.
- **Std parametrization values:** `log_std_min = −3.0` (hard-coded floor,
  `σ = 0.0498`), `log_std_max = −2.3` (tuned; `σ = 0.1003`); raw 0 sits at the
  midpoint `σ = 0.0707`.
- **`detach_critic`** (flag, default off): `critic_head(x.detach())`.
- **Freeze phases:** run start (`skip_update` for 3 chunks), critic warmup (16
  chunks: `shared`, `actor_mean`, `actor_log_std` frozen), post-reset warmup (2
  chunks).
- **Residual mode** zero-initializes `actor_mean` after loading the
  checkpoint; `--residual-resume` keeps a previously learned head.

## 0.11 Curriculum learning, staging and promotion gates

**Ladder: P → B → C.**

### 0.11.P Prerequisites

You need the RL training loop (0.4) and one new idea: instead of one fixed
task, this project trains against a WHOLE RANGE of tasks at once — the
target is placed somewhere between `distance_low` and `distance_high` meters
away, a different random distance every episode. "Widening the range" later
just means changing those two numbers and continuing training from the same
weights, rather than starting over.

### 0.11.B Basics

**Curriculum learning, the idea.** Train on an EASY version of the task
first (here: close targets, 3-10m), then widen to a harder range (3-30m,
then further), continuing from the same weights each time instead of
starting from scratch. This helps whenever the hard version alone would give
the network too little useful learning signal to make progress from a
random start — the easy version bootstraps a reasonable starting point.

**Transfer and forgetting — the two things that can happen when you widen
the range.** Usually, weights that already work well on the easy range
"transfer" — they're a good starting point for the harder range too, since
the underlying skill (fly toward a target) is similar. But if training then
runs for a long time almost entirely on FAR targets, the network can slowly
lose whatever made it good at NEAR targets, simply because it's no longer
seeing enough of that kind of example — this is called catastrophic
forgetting, and it's a real risk any time you fine-tune a network on a
narrower slice of what it originally learned.

**Deciding when a stage is actually "done" — the promotion gate.** You can't
just check the grade once and call it good — a single evaluation is noisy
(0.9: a 30-episode grade can easily wobble ±0.05-0.09 just from luck). This
project instead requires a WINDOW: several CONSECUTIVE checkpoints, all
above a threshold on average, with a floor so even the single worst one in
the window isn't a disaster — much harder to satisfy by pure luck than one
good number.

**Why stop training right when the gate fires, instead of continuing.**
Continuing to train past a genuinely good, stable point risks exactly the
slow drift described in 0.13 (a well-trained policy random-walking away from
where it was) — stopping there and moving on to the next, harder stage both
saves compute and locks in the checkpoint before that drift has a chance to
happen.

**Why the baseline controller has to keep up with the curriculum.** This
project's PID teacher/base-controller only has hand-tuned gains for specific
DISTANCES (0.3.C's gain ladder). Widening the training range to include
distances the PID has never been tuned for means it gives bad guidance (or,
in residual mode, a bad BASE action) for exactly the new, harder part of the
range — so widening the curriculum and extending the gain ladder have to
happen together, not the RL range alone.

### 0.11.C In this project (current)
- **Stages:** 3-10 m → 3-30 m → later 3 m-3 km. The PID gain ladder has to
  reach each stage (currently to 250 m; 3 km needs new entries).
- **Promotion gate** (`base_training_isaac`, constants `SOLID_*`): 5
  consecutive chunks, every one at loop-chunk index ≥ 40, averaging ≥ 0.8 with
  none below 0.6 (the same avg/min shape as the plotting leaderboard).
  Training starts at chunk 16, so 40 means ≥ 24 PPO chunks; the earliest firing
  is chunk 44. On firing it writes `<checkpoint-dir>/SOLID.json` (window
  grades, the best-graded checkpoint in the window) and stops the stage
  (`--no-stop-when-solid` to continue). If it never fires it prints the best
  qualifying window. Flags: `--solid-grade / --solid-min-chunk / --solid-window`.
- **Result:** the residual 3-10 m run fired at chunk 44 (window grades 0.880,
  0.921, 0.921, 0.951, 0.922 → mean 0.919), promoting `model_92274688.pt`;
  none of the direct-policy runs (v1-v4) ever passed it.
- **Carrying the correction forward:** `--residual-resume` keeps the loaded
  actor head so the next stage starts from the learned correction instead of
  zero.

## 0.12 Hyperparameter optimization

**Ladder: P → B → C.**

### 0.12.P Prerequisites

You need: evaluation noise (0.9 — a single measured grade can be wrong by
±0.05-0.09 from luck alone) and selection bias/winner's curse (0.9 — picking
the best of many noisy trials overstates how good it really is). Also worth
knowing before this section: a full production training run here takes about
1.5 hours; a single hyperparameter SEARCH trial (smaller scale, fewer steps)
takes only minutes — the entire point of a search is to try many
combinations cheaply before committing to one at full scale.

### 0.12.B Basics

**Grid search vs. random search — why random usually wins.** Grid search
tries every combination on a fixed grid of values (e.g. every combination of
5 learning rates × 5 batch sizes = 25 trials) — wasteful if, say, only the
learning rate actually matters, since you're repeating the same few
meaningfully-different LR values 5 times over for no benefit. Random search
instead samples every hyperparameter independently at random each trial —
for the same trial budget, it tries more DISTINCT values of whichever
hyperparameter actually matters, since it's not wasting trials repeating
values of the ones that don't.

**Bayesian search / TPE — using past trials to guide future ones.** Rather
than sampling blindly, keep a model of "which hyperparameter values tend to
score well" and bias new trials toward those. This project's search uses TPE
(Tree-structured Parzen Estimator): split all COMPLETED trials into a "good"
group (best-scoring) and a "bad" group (the rest), fit a separate rough
probability model of each group's hyperparameter values, and propose new
trials where the ratio `p(good)/p(bad)` is highest — i.e., "values that look
like what good trials tend to have, and unlike what bad trials tend to
have". Crucially, TPE only learns from trials that actually FINISHED with a
real score — a trial that crashed or errored contributes literally nothing
to this model, good or bad.

**Pruning — stopping bad trials early to save time.** If a trial partway
through is already scoring below the median of previous trials AT THE SAME
POINT in training, a pruner can just kill it early, rather than waste the
rest of its budget on something already likely to lose. This only makes
sense if the objective is smooth/stable enough that "behind at the midpoint"
reliably predicts "behind at the end" — a wildly noisy objective would prune
trials that would have recovered.

**Why the objective function you choose matters more than the search
algorithm.** Whatever number the search is told to maximize, it WILL find
ways to maximize — including ways you didn't intend, if the number is noisy
or gameable. *Concretely, from this project's own search:* the objective
originally returned the grade of just the LAST evaluated checkpoint. Since
each evaluation used only 5 episodes (0.9: quantized in steps of 0.2, easy
to get a lucky 5/5), many trials scored well simply by ending on a lucky
final measurement, not by actually training a policy that reliably held up.
The fix: score something that actually reflects what you care about —
here, "does the grade HOLD relative to where this trial's own policy started",
averaged over several checkpoints, not one lucky snapshot.

**Hyperparameters found at small scale don't automatically transfer to a
bigger run.** A search run with fewer parallel environments and a shorter
budget has a genuinely different EFFECTIVE BATCH SIZE (0.6.B) than the real
production run — anything whose right value depends on batch size (`lr`,
gradient-clipping threshold, value-loss weighting) needs to be re-checked,
not just copied over (0.8.B explains exactly why bigger batch does NOT
simply mean "safe to raise `lr`" in this setting). Hyperparameters that are
expressed as FRACTIONS of the schedule (e.g. "decay to 16% of the peak by
the end") or as trust-region limits (`target_kl`) tend to transfer more
reliably, since they're less tied to the raw step-count/batch-size details
of the search run.

**Winner's curse, applied here.** Report a hyperparameter search's best
measured trial as an OPTIMISTIC estimate, not a guarantee — some of its
apparent advantage over the next-best trials is real, and some is simply
that it got a luckier evaluation than an equally-good trial might have
(0.9).

### 0.12.C In this project (current)
- **Study:** 100 completed trials (a separate 100 failed instantly on a
  mlflow soft-deleted-experiment error and carry no information), 256 envs,
  2 M steps, 28 chunks per trial (16 frozen-actor + 12 PPO), 5 diagnostic
  episodes, phase split pulled live from `base_training_isaac`'s constants.
- **What went wrong with the objective:** it returned the *last* chunk's grade
  from 5 episodes (steps of 0.2). 14 of 100 trials got a lucky final 5/5
  (0.94-0.955); trial 162, reported best at 0.955, has a post-unfreeze mean of
  0.70 and ranks **27th** by robust delta.
- **What the study really showed** (`app/guidance/analyze_optuna_isaac.py`):
  mean baseline (frozen actor) 0.926 → mean post-unfreeze 0.599; **0 of 100
  trials held within −0.05 of their baseline**; the top trials share a tight
  `log_std_max` (−1.7…−2.1) and `lr` in the flat 1e-4…1e-3 band; the ranking
  among the top 10 is within measurement noise. Trial 137 (baseline 0.974,
  post-mean 0.861, 67% of post-unfreeze chunks ≥ 0.9) was the most consistent.
- **Transfer decisions:** `lr` halved (not scaled up by the 32× batch ratio);
  `ent_coef_end_frac` (0.83) converted to an absolute `ent_coef_end`.
- **Lessons recorded:** aggregate over chunks, use ≥ 20 eval episodes, score
  relative to baseline, keep the search's phase split identical to production
  by importing the same constants/functions (`entropy_coef_at`,
  `scheduled_lr`), and give the search enough post-unfreeze chunks (12 was too
  few to see learning).

## 0.13 Fine-tuning a good initialization: drift, anchoring and residual policies

**Ladder: P → B → C.**

### 0.13.P Prerequisites

This section leans on several earlier ones directly: PPO and advantage
normalization rescaling noise to full size (0.4.B); Adam still taking steps
from pure noise near an optimum (0.8.B); BC's std collapse (0.5.B); shared
trunks and gradient interference (0.10.B); and how small `σ` makes KL
hypersensitive (0.9.B). If any of those feel shaky, that's the section to
revisit first — this one is about combining them into "why does a GOOD
policy get worse during further training, and what do you do about it".

### 0.13.B Basics

**Putting the pieces together: why fine-tuning a near-optimal policy can
make it WORSE.** Once a policy is already good at a task, the TRUE
advantage of any action is close to 0 (0.4.B — nothing much left to
improve). But advantage NORMALIZATION rescales whatever's left — mostly
estimation noise at this point — back up to full size (0.4.B), and Adam
converts that noise into fixed-size random steps regardless of how tiny the
real signal is (0.8.B). The result: the weights take a genuine RANDOM WALK
away from a good solution, even though there's no real improvement signal
pulling them anywhere in particular — and for a precision task (hit within
0.25m), any wander at all costs real performance. Other things that can make
this worse (though which one dominates in any given case is something you'd
have to check empirically, not assume): a shared trunk being reshaped by the
critic's own (often much larger) gradient (0.10.B); exploration noise (`σ`)
sitting bigger than the task's actual precision needs; and a training
reward that isn't perfectly aligned with whatever metric you're using to
judge success (0.7.B/0.14).

**Remedies, from mild to strong.**
- **Lower the learning rate.** Shrinks EVERY step, including the useless
  noise-driven ones — helps, but doesn't remove the underlying random walk,
  just slows it down.
- **More or cleaner data.** A cleaner (less noisy) advantage estimate has
  less pure-noise component to accidentally amplify.
- **Freeze or detach parts of the network** (0.10.B) — stops specific
  gradients from reaching specific weights at all.
- **Checkpoint selection / early stopping** (0.11.B) — since drift happens
  over time, just don't keep training (or don't keep the LATEST checkpoint)
  past the point where it was actually good.
- **Anchoring** — add an extra penalty term to the loss that punishes the
  policy's action for straying too far from some REFERENCE action (e.g. what
  the original BC-trained policy, or the PID, would have done):
  ```
  L = L_PPO + β · E[ ((a_π − a_ref) / range)² ]
  ```
  Read it as: normal PPO loss, PLUS `β` times the average SQUARED, scaled
  distance between the policy's action `a_π` and the reference action
  `a_ref`. This is a SOFT leash — nothing stops the policy from straying far
  if PPO's own gradient pushes hard enough against the penalty, but the
  bigger `β` is, the more it costs to stray, and the tether length works out
  to roughly `1/β` (double `β`, the penalty for the same deviation doubles,
  so the policy settles at roughly half the deviation for the same PPO
  gradient strength). Not implemented in this project (yet) — see Part 12.
- **Residual policy learning** — instead of a soft penalty, change the
  ACTION ITSELF structurally: `a = a_base + s·π(s)` (`a_base` = the
  reference/baseline controller's action, `s` = a small fixed scale, `π(s)`
  = the network's output). This is a HARD leash: no matter what the weights
  do, the executed action can never be more than `s` times the action
  range away from `a_base` — not a preference, an actual mathematical bound.

**Residual policy learning's properties, worked through.** (1) If the
network's output is exactly 0, the executed action IS `a_base` exactly — so
the WORST case (an untrained or drifted network outputting near-0) is "as
good as the baseline controller", not "broken". (2) The deviation from the
baseline is bounded by `s × action_range` no matter how far training drifts
the weights — *this project's actual numbers:* `s=0.1`, so the correction
can never exceed 10% of the full action range, whatever the network learns.
(3) Since sampled actions include the SAME exploration noise `σ` as always,
but it now only affects the SMALL correction term, its real-world effect on
the executed action is also scaled down by `s` — a bonus reduction in how
much exploration noise disturbs behavior. (4) The cost: total achievable
performance is capped somewhere near `baseline + s·range` — if the true
optimal action needs a bigger correction than `s` allows, this architecture
literally cannot reach it, and everything now depends on the baseline
controller being decent to begin with (its tuning, and whether it even
covers the task range at all, 0.11.B).

**Why zero-initializing the correction head specifically matters.** Setting
the output head's weights to exactly 0 at the START of residual training
makes property (1) above true from the very first step, not just
eventually — training begins EXACTLY at baseline performance, with nothing
to lose, and only gradually adds a correction as it finds one that helps. As
a side effect, since the head's output starts at 0, the trunk's influence on
the actual executed action starts at 0 too (it only matters once the head's
weights grow away from 0), which also blunts how much the critic's
gradients through the shared trunk can disturb the actor early on (0.10.B).

**Anchoring and residual learning aren't rivals — they compose.** Anchoring
is a soft PENALTY added to the loss; residual learning is a hard structural
CONSTRAINT on what the action can even be. You could add an anchoring-style
penalty ON TOP of a residual architecture — e.g. penalizing the correction's
own size, `β·E[(s·π(s))²]` — which is just anchoring applied specifically to
the correction term rather than to the whole action.

### 0.13.C In this project (current)
- **Direct-policy runs v1-v4 (2026-09-24 → 09-26)** all show the same shape:
  grade ≈ 0.95 for 5-10 chunks after unfreeze, then a slide to 0.4-0.7 with
  resets only partially recovering it; the last (v4) went 0.96 → 0.5 by 77 M
  steps. Ruled out as sole causes: frozen `log_std` (fixed), entropy/std creep
  (v3's std stayed 0.071 → 0.072 yet drifted anyway), the LR profile.
- **Residual mode** (`--residual-scale 0.1`; `base_training_isaac.PidResidualEnv`):
  `env_action = pid_action + 0.1 × policy_action`; head zeroed; imitation
  stage skipped; diagnostics apply the same composition; PID gains re-selected
  per episode exactly as the imitation stage does. Smoke test: with a zero
  correction the wrapped env hit in 98% of episodes vs a raw zero-action
  reward of −3.1.
- **Result (`res_v1`, 3-10 m):** grade 0.971 in warmup (pure PID) → 0.927
  over 29 training chunks (min 0.841; 26 of 29 ≥ 0.9; 0 resets); success
  0.987 → 0.983, final distance 0.258 → 0.260, hit time 3.70 → 3.64 s. The
  0.04 grade gap is the grad-norm term of the grade formula (`grad_norm`
  averaged 8.0 ⇒ about −0.05), not lower success. It held, but did **not
  improve on the PID** — on 3-10 m the PID is at the ceiling.
- **Also implemented:** `--detach-critic` (untested as a fix on its own),
  `--residual-resume`. **Not implemented:** anchoring, a separate critic
  network.

## 0.14 Experiment tracking and reading training curves (MLflow)

**Ladder: P → B → C.**

### 0.14.P Prerequisites

You need two things from earlier sections: how noisy a single measurement
is (0.9 — a single chunk's grade can wobble ±0.05-0.09 from luck alone, so
don't over-read one data point), and what each logged quantity actually
means (grade/success from 0.7, loss/gradient-norm/KL from 0.4 and 0.8,
std/entropy from 0.9 and 0.10) — a graph is only as useful as your
understanding of the number it's plotting.

### 0.14.B Basics

**Three different JOBS a graph can be doing — and you need to know which.**
1. **Outcome graphs** answer "is the policy actually good?" (grade, success
   rate, final distance). These are what you ultimately care about.
2. **Diagnostic graphs** answer "is the TRAINING PROCESS healthy?" (loss
   values, gradient norm, KL divergence). These don't matter for their own
   sake — they help EXPLAIN why an outcome graph looks the way it does.
3. **Schedule graphs** answer "did the plan I configured actually execute?"
   (learning rate over time, entropy coefficient over time). These should
   just match what you intended — if they don't, something's misconfigured,
   before you even get to judging the training itself.
Always look at outcome graphs FIRST to know if there's a problem at all, then
use diagnostic/schedule graphs to figure out WHY.

**Never read a number in isolation — compare it to a reference.** A grade of
"0.93" means nothing on its own. Is that close to the best this task can
possibly score? Compare against the PID teacher's own measured grade on the
identical scenarios — "0.93 vs. the PID's 0.95" tells you the policy is
nearly matching a hand-tuned expert, while "0.93 vs. the PID's 0.98" would
tell a different story.

**Look at TRENDS across several points, not one point.** Since any single
measurement is noisy (0.9), average 5+ consecutive chunks before drawing a
conclusion, and when comparing two different runs, plot them on the SAME
graph (overlaid) rather than eyeballing two separate graphs side by side —
small offsets are much easier to misjudge when they're not directly next to
each other.

**Use a log scale for anything that spans a huge range.** If a quantity
(like gradient norm, or a loss value) can be anywhere from 0.1 to 50 across
a run, a normal (linear) axis squashes all the small, actually-interesting
variation down near zero and makes big spikes look like the ONLY thing
happening. A log-scale axis instead makes each MULTIPLICATIVE change (e.g.
"halved" or "doubled") look like the same-sized visual step no matter where
on the scale it happens, which usually matches what actually matters.

**Some jumps are supposed to happen — know which ones.** When training
switches from one phase to another (e.g. the actor unfreezing at chunk 16),
several graphs will jump on purpose, by design — the question isn't "did it
jump", it's "did it jump the RIGHT way and by roughly the expected amount".

**A real failure usually shows up as a COMBINATION of graphs moving
together, not one graph alone.** A single graph looking a bit off could just
be noise; several graphs telling a consistent story (e.g. grade falling
WHILE gradient norm is rising WHILE success rate is falling) is much stronger
evidence of a real, specific problem — see the combination table in 0.14.C.

### 0.14.C In this project (current)
Metrics are logged with `step = timesteps` (the UI's x-axis) from the training
loop (`log_metrics_safe`), mirrored to `<checkpoint-dir>/metrics.csv` and the
shared `runs/isaac_training_metrics.json`; `stage` is a tag. MLflow UI:
`python -m mlflow ui --backend-store-uri sqlite:///mlflow_isaac.db` (run from
the project root). Warmup lasts 16 chunks (~33 M steps); eval = 30 episodes.

**Is the policy good?** (these decide)

| Graph | Good | Bad |
|---|---|---|
| `grade` | flat 0.92-0.97 (ceiling ≈ 0.96-0.99) | slides (v4: 0.96 → 0.5) |
| `success_rate` | ≥ 0.95, ≈ `pid_success_rate` (≈ 0.98) | < 0.8 |
| `outcome_hit` | 29-30 of 30 | falling |
| `outcome_oob/_attitude-*/_timeout` | 0-1 | rising counts show *how* it fails |
| `avg_final_dist` | ≈ 0.25 (the floor: every episode hit, 0.3.C) | > 0.30 = misses |
| `avg_hit_time_sec` | ≈ 3.5-3.7 s; lower is good only if success holds | dropping while success falls = rushing (v4) |
| `pid_*` | the reference to match, not a health signal | – |
| `best_grade` | a staircase that only rises | – |
| `reset_triggered` | 0 | any |

**Is learning healthy?** (diagnostics)

| Graph | Good | Bad |
|---|---|---|
| `value_loss` | falls through warmup, flattens; may fall more after unfreeze (res_v1: 39 → 7) | jumps (resets do this) or oscillates |
| `approx_kl` | 0.002-0.015 (target 0.0086) | steadily ≥ 3× target (0.02-0.07 in old direct runs) |
| `grad_norm` | ≈ 0 in warmup, jumps at unfreeze, then eases (clip = 1.35, so always clipped) | keeps rising while grade falls (v2: 2 → 5) |
| `recent_reward` | flat or slightly up | falling > 10%; **noisy** (10-episode window) |
| `policy_loss` | tiny (≈ ±0.005), mean-zero (advantages are normalized) | NaN / huge |
| `entropy_loss` | drifts slowly | exactly flat in training (frozen std) |
| `effective_std_mean` | slow smooth drift inside 0.05-0.10 | flat in training = frozen; 0.0498 = collapsed; sawtooth = resets |
| `total_param_norm` | smooth growth | jumps = big updates or resets |

**Schedules:** `lr` flat 1.45e-4 in warmup then ramp + cosine; `ent_coef` flat
0.01 in warmup then decays to 0.002.

**Grade components:** `error_ratio` ≈ 0.01 (higher = misses); `grad_ratio`
costs up to −0.1 (it explains res_v1's 0.97 → 0.93); `time_ratio` is logged
but weighted 0.

**Failure signatures:** grade ↓ with `grad_norm` ↑ = drift; `hit_time` ↓ with
success ↓ = rushing; `effective_std_mean` dead flat in training = the
saturated-std bug; `value_loss` spikes = a reset just happened.

**Companion tools:** `app/guidance/analyze_optuna_isaac.py` (rank Optuna trials
by *holding* performance, prints the noise floor, writes a CSV) and
`plot_grad_norm_3d` (7.1) for the gradient-norm landscape.

## Part 0 practice questions

1. **(own words, ≤5 sentences)** Explain what "semi-implicit Euler"
   integration means and why this project uses it instead of explicit
   Euler.
2. **(trace a scenario)** A rigid body has inertia `(Ixx,Iyy,Izz) =
   (0.02, 0.02, 0.04)` kg·m² and angular velocity `ω=(5,0,3)` rad/s, with
   zero applied torque. What does `methods.py:angular_acceleration`
   actually return? What does the TRUE (full Euler equation) angular
   acceleration equal, and why do the two disagree?
3. **(what breaks)** If `methods.py:timestamp_update` composed
   `delta_rot * current_rot` (left) instead of `current_rot * delta_rot`
   (right), what would go wrong — and why would the bug be invisible for
   the first several steps and only show up later?
4. **(why this over the alternative)** Why does this project's PID use a
   CASCADED position→attitude structure instead of one loop mapping
   position error directly to rotor commands?
5. **(trace a scenario)** GAE's advantage estimate at `λ=1` telescopes down
   to one specific dependency on the critic; at `λ=0` it has a different
   (more direct) dependency. Which is more robust to a badly miscalibrated
   critic, and what does `base_training.py` actually do about this problem
   instead of just picking a "safe" `λ`?
6. **(cross-file)** This project's imitation stage never anneals a mixing
   parameter β the way the original DAgger paper does. What replaces that
   role here, and why is skipping β-annealing safe in THIS pipeline
   specifically (not in general)?
7. **(what breaks)** If `decimation` were raised from 4 to 40 with nothing
   else changed, what breaks first — physics fidelity, motor lag accuracy,
   or control responsiveness — and why?
8. **(why this over the alternative)** Why does this project's reward
   shaping use a plain potential DIFFERENCE instead of the textbook
   `γ·φ(s')-φ(s)` form, and what concrete failure mode did that avoid?
9. **(math)** `symlog(x, k)` is linear for `|x|≤k` and logarithmic beyond.
   Compute `symlog(6, 3)` and `symlog(300, 3)` by hand. Why would using RAW
   (un-transformed) position/distance values in the observation instead be a
   problem for a policy trained across BOTH a 3m task and a 250m task?
10. **(math)** For `z ~ N(μ,σ)` squashed through `a=tanh(z)·scale`, write
    the correction term `-log|da/dz|` explicitly, then evaluate it at `z=0`
    and at `z=3` for `scale=1`. Which `z` value makes the correction blow up
    faster, and what does that tell you about training stability if the
    raw (pre-tanh) policy output is allowed to grow very large?
11. **(trace a scenario)** A drone is hovering with `yaw=90°` (facing along
    world +y). The position-loop PID computes `accel_cmd=(2,0,0)`
    (world-frame: "accelerate toward world +x"). What roll/pitch-frame
    command should this become after the `-yaw` rotation, and what would go
    wrong if the yaw rotation were skipped entirely?
12. **(math)** Given a 2-step trajectory with rewards `r_0=1, r_1=1`, values
    `V(s_0)=0.5, V(s_1)=0.5, V(s_2)=0` (terminal), `γ=0.9`, compute the
    TD-residuals `δ_0, δ_1`, then GAE's `A_0` for both `λ=0` and `λ=1`. Which
    one equals the plain (undiscounted-baseline) Monte-Carlo advantage, and
    why does that make sense given 0.4's telescoping explanation?

<details><summary>Model answers</summary>

1. Semi-implicit (symplectic) Euler updates velocity first using the
   current acceleration, then updates position using the NEW velocity
   (`v' = v + a·dt`, `x' = x + v'·dt`) — unlike explicit Euler, which
   updates position from the OLD velocity. This ordering is unconditionally
   more stable for oscillatory/restoring-force systems, which matters for
   a drone whose attitude-control dynamics are inherently oscillatory. —
   0.1.
2. `Iω = (0.1, 0, 0.12)`. `ω×(Iω) = (0·0.12-3·0, 3·0.1-5·0.12, 5·0-0·0.1) =
   (0, -0.3, 0)`. True Euler equation with τ=0: `dω/dt = -ω×(Iω)/I =
   (0, 0.3/0.02, 0) = (0, 15, 0)` rad/s² — real gyroscopic precession on
   the y-axis. `angular_acceleration` returns exactly `τ_i/I_i = (0,0,0)`
   since `net_torque=0` and the function never computes `ω×(Iω)` at all —
   the code and the true physics disagree specifically because of the
   omitted coupling term. — 0.1, `methods.py:angular_acceleration`.
3. `Δq` was derived from a BODY-frame angular velocity, so it must be
   applied "in body frame" (right composition) to mean what it's supposed
   to mean. Composing on the left instead applies it as if it were a
   WORLD-frame increment. For a small rotation, world and body axes are
   still nearly aligned, so the error per step is second-order tiny; it
   only becomes visible once the drone has actually turned enough that its
   body axes have diverged substantially from world axes — the bug is a
   growing directional drift, not an instant, obvious failure. — 0.2,
   `methods.py:timestamp_update`.
4. Cascaded structure lets a fast inner attitude loop track its setpoint
   well before the slower outer position loop changes that setpoint again
   (bandwidth separation) — a real quadrotor's attitude dynamics genuinely
   operate on a much faster timescale than its translational dynamics. One
   loop mapping position error straight to rotor commands would need pole
   placement simultaneously fast enough for attitude stabilization and
   smooth enough for position tracking — conflicting requirements that
   fight each other. — 0.3, `tune_pid.py`'s `BANDWIDTH_SEP`.
5. At `λ=1`, GAE's sum of discounted TD residuals telescopes to `(full
   discounted Monte-Carlo return) - V(s_t)` — every intermediate `V(s)`
   estimate cancels out, leaving only ONE dependency on the critic (a
   baseline subtraction). At `λ=0`, the advantage IS
   `r_t + γV(s_{t+1}) - V(s_t)` — both terms depend directly on the critic,
   so a bad critic corrupts it more immediately. But `base_training.py`
   doesn't dodge this by picking `λ=1` (it uses `λ=0.97`, close to the
   textbook default) — it runs a CRITIC-ONLY warmup stage first (actor
   frozen, `WARMUP_DURATION_STEPS`) so the critic is already reasonably
   calibrated before any real advantage estimate is used to move the
   policy at all. — 0.4, `base_training.py`'s warmup-stage comment.
6. `base_training._run_imitation_stage` always drives the environment with
   the CURRENT student policy's deterministic action, and only queries the
   PID expert to LABEL that same visited state
   (`vec_env.step_with_pid_actions`) — β=0 from round one, the simplest
   DAgger variant. It's safe here specifically because the student starts
   from an already-competent BC/DAgger warm-start checkpoint rather than a
   random policy (`load_bc_checkpoint`), so an early pure-student rollout
   is imperfect, not degenerate — the original paper's β-annealed mixture
   exists to protect against exactly the random-start case this project
   doesn't have. — 0.5.
7. Motor lag and rigid-body integration would NOT degrade — they're
   resolved every physics tick (1/240s) regardless of decimation. Control
   RESPONSIVENESS collapses instead: the policy would only act once every
   40 ticks, an effective 6 Hz control rate, far too slow to correct a
   quadrotor's fast attitude dynamics — the drone would likely tumble or
   oscillate uncontrollably between increasingly rare policy decisions,
   independent of physics correctness. — 0.6.
8. The strict `γ·φ(s')-φ(s)` form leaves a residual `(γ-1)·φ(s)` reward
   every step even standing perfectly still — since `φ` is negative and
   `γ<1`, that residual is POSITIVE, so doing nothing at all still nets
   reward, undermining `step_penalty`'s entire purpose. The plain
   difference `φ(s')-φ(s)` makes standing still net exactly zero shaping
   reward. Measured concretely: about +0.013/step at a 3m target, nearly
   half of `HIT_REWARD` accumulated over a stalled 1800-step episode. —
   0.7, `rewards.py`'s in-file comment.
9. `symlog(6,3)`: `|6|>3`, so `sign(6)·(1+ln(6/3)) = 1+ln(2) ≈ 1.693`.
   `symlog(300,3)`: `1+ln(300/3) = 1+ln(100) ≈ 5.605`. Raw values differ by
   50x (6 vs 300); symlog-transformed values differ by only ~3.3x
   (1.693 vs 5.605). A policy seeing raw values would face observation
   magnitudes varying ~100x across a 3m-vs-250m curriculum — badly scaled
   gradients (large-task magnitudes would numerically swamp small-task
   precision) with no single normalization constant working well for both
   regimes. symlog keeps small distances LINEAR (full precision where it
   matters most) while compressing large ones logarithmically. — 0.2.
10. `-log|da/dz| = -log(scale) - log(1-tanh(z)²)`; with `scale=1` this is
    `-log(1-tanh(z)²)`. At `z=0`: `tanh(0)=0`, so `-log(1-0)=-log(1)=0` — no
    correction. At `z=3`: `tanh(3)≈0.9951`, `1-tanh(3)²≈0.00987`,
    `-log(0.00987)≈4.62` — a large correction. `z=3` blows up far faster
    (the correction grows without bound as `|z|→∞`, `z=0` stays exactly
    zero). This means an UNCLAMPED, saturating raw policy output produces
    huge, poorly-behaved log-prob corrections — real motivation for keeping
    the raw (pre-tanh) output from growing unbounded, since the
    log-probability used in PPO's ratio becomes numerically unstable right
    where the tanh saturates. — 0.2/0.4.
11. Rotating world-frame `(2,0)` by `-yaw=-90°`:
    `R(-90°)=[[0,1],[-1,0]]`, giving `(0·2+1·0, -1·2+0·0)=(0,-2)` — the
    command becomes purely a body-Y (lateral) component, zero body-X
    (forward). At `yaw=90°`, achieving a world+x acceleration REQUIRES
    tilting sideways (roll), not forward (pitch) — exactly what the
    rotation produces. Skipping the rotation would apply `(2,0)` directly
    as if body-x were still aligned with world-x, commanding PITCH instead
    of ROLL — the drone would tilt the wrong axis, and the steering error
    would grow the further yaw diverges from 0. — 0.3.
12. `δ_0 = r_0+γV(s_1)-V(s_0) = 1+0.9·0.5-0.5 = 0.95`.
    `δ_1 = r_1+γV(s_2)-V(s_1) = 1+0.9·0-0.5 = 0.5`. With `A_2=0` (terminal):
    `λ=0`: `A_1=δ_1=0.5`, `A_0=δ_0=0.95`.
    `λ=1`: `A_1=δ_1+γ·A_2=0.5`, `A_0=δ_0+γ·A_1=0.95+0.9·0.5=1.4`.
    Check against plain Monte Carlo: `G_0-V(s_0) = (r_0+γr_1)-V(s_0) =
    1.9-0.5 = 1.4` — matches `λ=1` exactly, confirming 0.4's telescoping
    claim (all intermediate `V` estimates cancel at `λ=1`, leaving only a
    baseline subtraction). `λ=0`'s `0.95` depends directly on `V(s_1)` too,
    not just `V(s_0)` — more exposed to a miscalibrated critic. — 0.4.

</details>

## Part 0b practice questions (theory 0.8-0.14)

1. **(own words, ≤5 sentences)** Why does Adam random-walk near an optimum
   even though the true gradient is ≈ 0? Give the size of a pure-noise step.
2. **(trace a scenario)** Peak `lr = 1.451e-4`, floor ratio `0.1621`, cosine
   over the PPO stage only (progress `p = 0.5`). What is `lr`? What would an
   unclamped `p = 1.5` give, and why does that matter?
3. **(why this over alternative)** Optuna's best trial had `lr = 2.9e-4` at
   minibatch 1,024; production uses minibatch 32,768. Why was `lr` halved
   instead of scaled up by 32× (or 64×)?
4. **(numbers)** Standard error of a hit-rate from 5 episodes at `p = 0.8`,
   and from 30 episodes at `p = 0.9`. What does the first imply about the
   Optuna objective?
5. **(trace)** Policy `σ = 0.0707`; PPO moves the mean by `Δμ = 0.02` (raw
   units). KL under equal-σ Gaussians? Compare with `target_kl = 0.0086`.
6. **(own words)** Across v1/v2/v3, `value_loss` correlates +0.67/+0.84/+0.92
   with grade. Why is that *not* proof that critic learning damages the actor?
7. **(trace)** `actor_log_std` raw = −5 with `log_std ∈ [−3, −2.3]`. What is
   `σ`, what is `tanh'(−5)`, and why does the logged `effective_std` stay
   bit-identical for a whole run even though Adam is updating the parameter?
8. **(what breaks)** If every parameter were frozen (`requires_grad_(False)`)
   and `isaac_ppo_train` still ran a normal update, what would happen — and
   what does the code do instead for the run-start "no update" chunks?
9. **(trace the gate)** Window grades (chunks 40-44):
   (a) `[0.82, 0.78, 0.81, 0.79, 0.84]`, (b) `[0.95, 0.95, 0.95, 0.95, 0.55]`,
   (c) chunks 38-42. Does the promotion gate fire in each? Why is 40 (not 20)
   the minimum chunk?
10. **(own words)** Optuna reported trial 162 as best (0.955). Give three
    reasons that is misleading and one better ranking metric.
11. **(what breaks)** A hundred trials fail instantly with "Cannot set a
    deleted experiment". What does TPE learn from them, and what is the state
    of the study afterwards?
12. **(numbers)** Residual mode: `residual_scale = 0.1`, thrust range ±14.715
    N, torque range ±0.5 N·m. The PID commands thrust delta `+2.0 N`. What is
    the range of the executed thrust and the largest torque correction? What
    is the executed thrust if the policy outputs 0?
13. **(why this over alternative)** Residual policy vs anchoring loss vs
    just lowering the lr: what does each buy, and what does each cost?
14. **(own words)** `res_v1` held its grade (0.927 vs 0.971 baseline). What
    does that result show, what does it *not* show, and what explains the
    0.04 gap?
15. **(read the graph)** Over 15 chunks: grade 0.95 → 0.6, success 0.98 → 0.6,
    `avg_hit_time` 3.7 → 3.1 s, `recent_reward` roughly flat, `grad_norm`
    2 → 5. Name the failure signature and two hypotheses to test.
16. **(numbers)** Grade for success = 1.0, `error_ratio = 0.01`,
    `grad_norm = 8` (`grad_norm_ceiling = 50`, weights 1.0/0.3/0.1). Then for
    `grad_norm = 0.15`.
17. **(PID)** Derive `kp_pos, kd_pos` for a 20 m target from first principles
    (ζ = 0.85, ε = 0.05 m, `SETTLE_TIME_FRACTION = 0.5`,
    `steps_for_dist(20)`). Then compute the saturation ratio at 30 m for the
    30 m gains and for the 3 m gains (`g·max_tilt = 2.943`).
18. **(cross-file)** Trace which flags and code paths change when you add
    `--residual-scale 0.1 --residual-resume` to `train_isaac.py`, in order.

<details><summary>Model answers</summary>

1. At an optimum the *expected* gradient is ≈ 0 but each minibatch's gradient
   is still noise of unchanged size. Adam normalizes by the gradient's own
   scale (`m̂/√v̂`), so the step doesn't shrink with the signal: for zero-mean
   noise `|m̂|/√v̂ ≈ √((1−β₁)/(1+β₁)) = √(0.1/1.9) ≈ 0.23`, i.e. random steps of
   ~0.23·lr per update, which accumulate as a random walk. Advantage
   normalization (`train.py:175`) makes the noise full-size to begin with. — 0.8,
   0.4.
2. `factor = 0.1621 + ½·(1−0.1621)·(1+cos(π·0.5)) = 0.1621 + 0.41895·1 =
   0.5811` → `lr ≈ 8.43e-5`. Unclamped `p = 1.5`: `cos(1.5π) = 0`, factor 0.581
   again — cosine is periodic, so an un-clamped progress makes the LR
   *oscillate back up* instead of decaying. Hence `training_progress_at`
   clamps to `[0,1]`. — 0.8.
3. Because the search's ≥ 1e-3 trials all collapsed (mean grade ≈ −0.02) and
   lr < 1e-5 was worst; the flat band is 1e-4…1e-3. With Adam the step per
   update is ≈ lr regardless of batch, and a cleaner large-batch gradient makes
   each step *more coherent*, so the same lr moves the policy further per
   chunk — bigger batch does not license a bigger lr in PPO. Scaling by 32-64×
   would have landed at ~1e-2, deep in the collapsed zone. 1.45e-4 sits at the
   low edge of the flat band (also ~3× more post-unfreeze steps in production).
   — 0.8, 0.13.
4. `√(0.8·0.2/5) = 0.179`; `√(0.9·0.1/30) = 0.055`. With 5 episodes one failed
   episode moves success by 0.2, so the last-chunk grade the objective returned
   is dominated by luck (grades cluster at ≈ 0.96 / 0.76 / 0.35). — 0.9, 0.12.
5. `KL = Δμ²/(2σ²) = 0.0004 / (2·0.004998) = 0.040` — about 4.7× the 0.0086
   target from a shift of only 0.02 raw units. Small σ makes the KL check
   hypersensitive, which is why early-stopping fires after one epoch. — 0.9.
6. Both curves trend monotonically with time (value_loss falls as the critic
   fits, grade falls as PPO drifts), so any two such curves correlate strongly
   (time is a confounder). Resets add non-monotone events (value_loss jumps up,
   grade recovers) but they also restore the actor's weights, so that is not an
   isolated critic effect either. It is a hypothesis to test (e.g. with
   `--detach-critic`), not a finding. — 0.9, 0.13.
7. `log σ = −3 + 0.35·(tanh(−5)+1) ≈ −3 + 0.35·(9.1e-5) ≈ −3.00003` → `σ ≈ 0.0498`
   (the floor). `tanh'(−5) = 1 − tanh² ≈ 1.8e-4`. Adam still updates the raw
   value (≈ lr per step at most), but σ depends on it only through `tanh'`, so
   the *logged* σ changes by ~1e-8 relative and looks bit-identical; getting
   out would need several raw units of movement at ≤ lr per step. — 0.10.
8. `loss.backward()` fails ("element 0 of tensors does not require grad and
   does not have a grad_fn") because nothing in the graph needs gradients. The
   run-start chunks therefore use `isaac_ppo_train(skip_update=True)`, which
   collects the rollout and never calls `ppo_update`. — 0.10.
9. (a) mean 0.808 ≥ 0.8, min 0.78 ≥ 0.6 → fires at chunk 44. (b) mean 0.87 but
   min 0.55 < 0.6 → does not fire (one crash disqualifies the window).
   (c) chunks 38-42 start below 40 → the rule requires *every* chunk of the
   window at index ≥ 40, so it cannot fire there. 40 = at least 24 PPO chunks
   after the unfreeze at 16; 20 would promote a checkpoint that is just the
   warmup (imitation/PID) policy plus a few PPO steps. — 0.11.
10. (i) It returned only the last chunk from 5 episodes (quantized ±0.2, luck);
    (ii) its post-unfreeze mean was 0.70 and last-7 mean 0.55 — the final 5/5
    was a spike; (iii) it hardly differs from 13 other trials with the same lucky
    5/5. Better: rank by delta (post-unfreeze mean − own frozen baseline) or the
    trailing-5 mean; 162 ranks 27th, 137 ranks near the top. — 0.12.
11. Nothing: failed trials have no objective value, TPE only models completed
    ones. Cause: a soft-deleted mlflow experiment name can't be reused; the
    study's trial counter still advanced (trial numbers started at 100), and the
    study.db kept the 100 FAIL rows harmlessly. Fix: purge the experiment or
    restore it. — 0.12.
12. Correction range = `0.1 × 14.715 = ±1.47 N` thrust and `0.1 × 0.5 = ±0.05
    N·m` torque, so executed thrust ∈ [0.53, 3.47] N; if the policy outputs 0
    the executed thrust is exactly 2.0 N (pure PID). — 0.13.
13. Residual: hard bound on deviation, starts as the PID, noise ×0.1; costs a
    ceiling near `base + s·range` and dependence on the base controller.
    Anchoring: soft penalty `β·E[(a−a_pid)²]` — flexible, but needs a teacher
    action at every state, tuning β, and doesn't bound drift. Lower lr: slows
    drift but starves the critic warmup and doesn't remove it (the drift
    continued while lr fell to 2e-5). — 0.13.
14. Shows: with the correction capped at ±10% and starting at the PID, PPO did
    not slide (26/29 chunks ≥ 0.9, no resets) and passed the gate at chunk 44.
    Does not show improvement over the PID: success 0.987 → 0.983, final
    distance 0.258 → 0.260, hit time 3.70 → 3.64 s (within noise). The 0.04
    gap is the grad-norm penalty (mean `grad_norm` 8.0 → `grad_ratio` ≈ 0.53 →
    −0.053), not lower success. — 0.13, 0.14.
15. "Rushing": speed up while success and precision fall, with a flat reward.
    Hypotheses: (1) the training reward includes time pressure while the grade
    ignores time (`w_time = 0`), so PPO trades success for speed; (2) an
    Isaac-vs-numpy mismatch — the policy specializes to Isaac physics
    (60 Hz control, PhysX) while the numpy oracle evaluates it at 240 Hz.
    Test (2) by evaluating the same checkpoints in Isaac
    (`scripts/isaac_lab/diagnose_model_isaac.py`). — 0.14, 0.6.
16. `grad_ratio = ln(1+8)/ln(1+50) = 2.197/3.932 = 0.559` → grade =
    `1.0 − 0.3·0.01 − 0.1·0.559 = 0.941`. For `grad_norm = 0.15`:
    `ln(1.15)/ln(51) = 0.140/3.932 = 0.0356` → `1 − 0.003 − 0.0036 = 0.993`.
    — 0.14.
17. `steps_for_dist(20) = max(1800, 750·20/3) = 5000` ticks = 20.83 s;
    `t_s = 0.5·20.83 = 10.42 s`; `ωₙ = ln(20/0.05)/(0.85·10.42) = 5.991/8.854 =
    0.677`; `kp = ωₙ² = 0.458`, `kd = 2·0.85·0.677 = 1.15`. Saturation at 30 m:
    30 m gains `0.232·30/2.943 = 2.4×` (starts saturated, desaturates near the
    target); 3 m gains `1.65·30/2.943 = 16.8×` (violently over-demanding).
    — 0.3.
18. (1) `train_isaac.py` parses both flags and passes `residual_scale=0.1,
    residual_resume=True` to `train()`; (2) `train()` loads the `--bc-checkpoint`
    (a residual checkpoint, e.g. `SOLID.json`'s `model_…pt`) with
    `load_bc_checkpoint`; (3) because `residual_scale > 0` it wraps the env in
    `PidResidualEnv` and, because `residual_resume`, **keeps** `actor_mean`
    instead of zeroing it; (4) `imitation_timesteps_grand = 0` so the imitation
    stage is skipped; (5) `actor_log_std` is still reset to 0; (6) each chunk's
    rollout runs through the wrapper (`pid + 0.1·action`), and
    `diagnose_with_model` applies the same composition with per-episode gains;
    (7) the promotion gate applies as usual. — 0.13, 0.11.

</details>

---

# PART 1 — Block: Dynamics (the physics oracle)

**Files**: `app/dynamics/drone.py`, `app/dynamics/methods.py`,
`app/dynamics/torch_methods.py`

## 1.1 `drone.py` — data structures

Plain dataclasses, no physics: `Vector3D`, `Quaternion` (xyzw, scipy
convention), `RotorConfig` (per-motor: position, `spin_dir` ±1, `k_f`
thrust coeff, `k_m` torque coeff, `max_rpm`, `motor_tau`), `QuadConfig`
(per-drone: mass, `inertia` tuple, `arm_length`, `drag_coeff`, 4 rotors),
`QuadState` (per-tick: position, velocity, orientation, angular_velocity,
`rotor_rpm[4]`).

**`create_quad_rotors`** places 4 rotors at 45/135/225/315° (X
configuration), **alternating `spin_dir`** so reaction torques cancel at
hover (two CW, two CCW — a real quadrotor design constraint, not arbitrary).
Solves `k_f` so that `hover_rpm_fraction * max_rpm` (default 50% of max)
exactly balances gravity:

```
k_f = (mass·g/4) / (max_rpm·hover_rpm_fraction)²
```

(from `thrust = k_f·ω²` summed over 4 equal rotors `= mass·g` at hover).

## 1.2 `methods.py` — the numpy oracle's per-step physics (flow.md's spine)

This is the file everything else in the project is calibrated against. Full
per-step pipeline, `timestamp_update`:

1. **`mixer_inversion(config, desired=[thrust,roll,pitch,yaw])`**: builds a
   4×4 matrix `M` where row `i`, column `j` is rotor `j`'s contribution to
   output `i`:
   ```
   M[0,j] = k_f_j                        (thrust)
   M[1,j] = position_y_j · k_f_j         (roll torque, x-axis)
   M[2,j] = -position_x_j · k_f_j        (pitch torque, y-axis)
   M[3,j] = k_m_j · spin_dir_j           (yaw torque, z-axis)
   ```
   (rows 1/2 come from `cross([x,y,0],[0,0,F]) = [y·F, -x·F, 0]` — the
   moment-arm torque a body-z force produces at an offset position). Solves
   `M @ w² = desired` for `w²` (rotor speed squared, since thrust ∝ ω²),
   clips negative solutions to 0 (a torque/thrust combo the mixer can't
   physically realize), takes `sqrt`.
2. **`motor_lag(w_current, w_target, τ, dt)`**: first-order lag,
   `w_new = w_current + ((w_target-w_current)/τ)·dt` — a rotor can't jump
   speed instantly; `τ` is the spin-up/down time constant.
3. **Forces**: `net_combining_thrust` sums `k_f·ω²` over 4 rotors (scalar,
   body-z). `net_combining_torque` sums moment-arm cross products PLUS
   reaction torque `k_m·ω²·spin_dir` per rotor (3-vector, body-frame).
   `drag_force`: `0.5·ρ_air·|v|²·Cd·A`, opposing velocity direction, zero
   below 0.01 m/s (avoids a `0/0` direction at rest). `wind`:
   `wind_vector·mass·k_wind_coeff` (currently disabled — see 1.2.1 below).
4. **Integration**: thrust rotated body→world via the current orientation
   quaternion, summed with drag+wind+gravity → `total_force`. Semi-implicit
   Euler (0.1) for position/velocity. Angular: `alpha = torque/inertia`
   (diagonal, per-axis), `new_ω = ω + α·dt`, incremental rotation
   `Rotation.from_rotvec(new_ω·dt)` composed **on the right** of the
   current orientation (0.2's body-frame composition rule).

**1.2.1 — currently-disabled features worth knowing about**: domain
randomization (`sample_wind_conditions` in `enviorment.py`, wind vector +
mass_scale) is wired up but the actual call in `base_drone_env.py:reset` is
commented out, hardcoded to `wind_vector=[0,0,0], mass_scale=1`. If asked
"does the drone experience wind" — the honest answer is "the mechanism
exists and is exercised in dynamics/methods.py's math, but is not currently
active in training."

## 1.3 `torch_methods.py` — batched torch mirror (migration step 3)

Line-for-line the same math as `methods.py`, but every function takes
`(num_envs, ...)` tensors instead of a single `QuadState`, and the rotor
config becomes precomputed tensors (`QuadConfigTorch`, with `mixer_inv`
precomputed once via `torch.linalg.inv`, not rebuilt every call). Verified
**bit-exact** against the numpy oracle (`atol=rtol=1e-9`) over 64 random
cases × 5 functions — see `scripts/isaac_lab/diff_test_physics.py`. Pure
torch, no `isaaclab` import, so it runs standalone (no Kit boot needed) —
this is deliberate: the force/torque MATH is validated completely
independently of whether PhysX's integration is trusted.

**Frame convention** (memorize this, it's a recurring gotcha across the
whole Isaac port): torque and angular velocity are **body-frame**
throughout (rotor positions are body-frame, inertia is diagonal in body
frame); linear force/velocity are **world-frame**. Isaac Lab's own
quaternion convention is `(w,x,y,z)`; the oracle/scipy convention is
`(x,y,z,w)` — anywhere the Isaac port builds an observation or calls the
oracle for comparison, this reordering has to happen explicitly (see
`base_drone_env_isaac.py:_get_observations`).

## Part 1 practice questions

1. **(own words, ≤5 sentences)** Explain what `timestamp_update` does, start
   to finish, without reading the code.
2. **(trace a scenario)** A rotor's `motor_lag` state is currently 6000
   rad/s, `mixer_inversion` just computed a target of 8000 rad/s,
   `motor_tau=0.05`, `dt=1/240`. What's the new rotor speed after exactly
   one call to `motor_lag`? Show the arithmetic.
3. **(what breaks)** If you deleted the `np.clip(speeds, 0.0, None)` line in
   `mixer_inversion`, what specific runtime failure would you expect, and
   under what commanded action would it first trigger?
4. **(why this over the alternative)** Why does `net_combining_torque` sum
   a moment-arm cross product AND a separate reaction-torque term per
   rotor, instead of just one or the other?
5. **(cross-file)** `torch_methods.py` claims to be diff-tested "bit-exact"
   against `methods.py`. Which script proves this, and why can that script
   run without ever booting Isaac Sim?
6. **(cross-file)** Which project file first told you that motor lag must
   be integrated at `PHYSICS_DT` regardless of `decimation`, and why does
   that requirement exist (what would decimation-scaled motor lag get
   wrong)?
7. Why does `create_quad_rotors` alternate `spin_dir` between adjacent
   rotors instead of, say, having rotors 0-1 spin one way and 2-3 the
   other?
8. What does the `<0.01 m/s` branch in `drag_force` exist to prevent,
   numerically?

<details><summary>Model answers</summary>

1. Given the current drone state, a commanded `[thrust,roll,pitch,yaw]`,
   wind, and `dt`: invert the mixer to get target rotor speeds, lag each
   rotor toward that target, sum the lagged speeds into net thrust/torque,
   add drag/wind/gravity, integrate linear and angular state one step
   (semi-implicit Euler + quaternion exponential map), return the new
   `QuadState`. — `methods.py:timestamp_update`
2. `w_dot = (8000-6000)/0.05 = 40000`. `w_new = 6000 + 40000·(1/240) ≈
   6166.7` rad/s. — `methods.py:motor_lag`
3. `sqrt` of a negative number → `nan` propagating into every rotor speed,
   then into thrust/torque, then into position — silent NaN corruption, not
   an exception. Triggers whenever a commanded torque/thrust combination
   the 4-rotor mixer can't physically realize (e.g. a huge yaw demand with
   near-zero thrust) inverts to a negative squared-speed for some rotor.
4. Moment-arm torque comes from thrust applied off-center (roll/pitch);
   reaction torque comes from the rotor's own spin resisting the air
   (drag reaction, causes yaw) — two physically distinct torque sources a
   real quadrotor has, both needed or yaw control (which depends ONLY on
   reaction torque, since all 4 rotors are equidistant from center for a
   symmetric X-frame's net moment-arm-in-z... actually the moment arm term
   is zero contribution to yaw since F is along z and r is in the xy-plane
   — cross(r,[0,0,F]) has no z-component) would be entirely missing.
5. `scripts/isaac_lab/diff_test_physics.py` — pure numpy/torch/scipy, no
   `isaaclab` import anywhere in the file, so nothing requires Kit/PhysX to
   even be installed correctly, let alone booted.
6. `base_drone_env_isaac.py`'s `_pre_physics_step`/`_apply_action` comments
   (and flow.md section 8): motor lag is a real ODE with time constant `τ`
   calibrated against the true physics rate; integrating it at a coarser
   (decimated) rate would make the rotor spin-up/down dynamics wrong —
   slower or faster than the real motor — independent of what `decimation`
   happens to be chosen for training speed.
7. So reaction torques cancel at hover: CW and CCW rotors' yaw reaction
   torques are opposite in sign; alternating ensures the four don't all add
   in the same direction, matching a real quadrotor's requirement for
   yaw-neutral hover.
8. Division by (near-)zero speed when normalizing the drag direction
   (`-v/speed`) — without the floor, a resting drone would produce a NaN or
   wildly noisy drag direction from floating-point noise near `v=0`.

</details>

---

# PART 2 — Block: Environment & Reward

**Files**: `app/environmental/base_drone_env.py`, `enviorment.py`,
`vec_base_drone_env.py`, `subproc_vec_base_drone_env.py`,
`app/reward_functions/rewards.py`, `app/control/step_budget.py`

## 2.1 `base_drone_env.py` — the numpy Gym environment

`BaseDroneEnv(gym.Env)`. `build_observation` (23-dim, module-level function
also reused by the Isaac port for numerical parity):

```
symlog(pos,15)[3] + vel/10[3] + quat_xyzw[4] + ang_vel/20[3]
+ rotor_rpm/12000[4] + symlog(target-pos,15)[3] + symlog(dist)[1]
+ [sin(yaw_err),cos(yaw_err)][2]  = 23
```

`sin/cos(yaw_err)` instead of raw radians avoids the `±π` discontinuity
showing up as a huge jump in the observation (a policy would otherwise see
"turn left 179°" and "turn right 179°" as wildly different inputs for
nearly the same physical heading).

`action_space`: `Box([-hover_thrust,-0.5,-0.5,-0.5], [hover_thrust,0.5,0.5,
0.5])` — thrust is a **delta around hover**, not absolute (the policy
learns to push harder/softer than "just hovering", not to reinvent hover
from scratch every step). `step()`: clip action → add hover offset to index
0 → `timestamp_update` → build obs → call the injected reward function →
truncate at `max_steps`.

**`reset(start_pos, target_pos, target_yaw, ...)`**: if none given and
`target_pairs` was supplied at construction, samples uniformly at random
from that pool (this exact mechanism is what migration step 6 ported to the
Isaac side as `set_target_pairs`). Otherwise applies `spawn_offset_range`/
`target_offset_range` (x-only jitter) around a fixed `(0,0,5)`.

## 2.2 `enviorment.py` — environment-sampling utilities

Now just `sample_wind_conditions(np_rand)` (uniform wind ±2 m/s per axis,
mass_scale 0.92-1.08) — the pybullet-specific `spawn_drone` (built a
PyBullet multi-body for GUI visualization) was removed this session; Isaac
Sim is the only visualization path now.

## 2.3 `vec_base_drone_env.py` / `subproc_vec_base_drone_env.py` — numpy
vectorization (the thing Isaac Lab's `num_envs` replaces)

`VecBaseDroneEnv`: a plain Python for-loop over N `BaseDroneEnv` instances,
single process — batches the POLICY forward pass (worth doing, since a GPU
needs a batch to be worth the transfer) but not the physics stepping
itself (still one core, sequential). `SubprocVecBaseDroneEnv`: shards envs
across `multiprocessing.get_context("spawn")` worker processes (one per
core) — **spawn, not fork**, because the main process may have already
initialized CUDA (model on GPU) by the time the vec-env is built, and
forking after CUDA init is unsafe (driver-internal locks aren't fork-safe).
Reward-fn closures from `chain_reward_fns` aren't picklable (nested
functions) — each worker rebuilds its own envs from plain picklable data
(a factory function reference + params + target_pairs), never receives
constructed env/reward-fn objects directly.

**`_select_pid_teacher`**: after every reset, swaps the env's `pid_teacher`
to whichever `best_pid_gains_per_dist.json` entry has the nearest distance
key to that episode's actual `start_dist` — a single fixed gain set is a
much worse imitation teacher across a wide distance curriculum than
per-distance-matched gains (this exact mechanism is what the Isaac port's
`assign_gains_by_distance` in `torch_pid.py` mirrors).

Both auto-reset a just-terminated env before returning its obs, and stash
the TRUE pre-reset terminal observation in `info["terminal_observation"]`
— needed for correct truncation-bootstrap (0.4's termination-vs-truncation
distinction) since the returned `next_obs` for a truncated env already
belongs to its NEXT episode.

## 2.4 `rewards.py` — two reward systems, one actually used

**`reward_func`** (module-constant-driven, no config class) — **the one
actually used by current training**:
- `terminal_checks`: NaN position / `z<0` / `|pos|>oob_radius` (scaled per
  episode: `max(30, start_dist·3)`, so a far target isn't structurally
  unreachable), attitude limits (65°/80° roll/pitch).
- `hit_target`: `dist < 0.25`.
- Shaping: `phi_now - phi_prev`, `phi = -L1(target-pos)/start_dist` — a
  **plain difference**, deliberately not `γ·phi_now - phi_prev` (0.7's
  policy-invariance trade-off).
- `step_penalty`: `-(TARGET_FRACTION·APPROACH_MILESTONE_BUDGET) /
  steps_for_dist(start_dist)` — flat per-step cost scaled to the episode's
  own distance budget.
- `milestone_bonus`: one-time bonuses at 25/50/75% progress toward target
  (tracked in `env.milestones_hit`, a set reset every episode).
- `stability_penalty`/`exit_penalty` (added 2026-09-19): anti-oscillation
  terms -- see 2.4.1.

**`RewardConfig`/`make_reward_fn`/`base_reward_fn`** — an OLDER,
config-driven curriculum system (optional `phase_imitation_fn` chained via
`chain_reward_fns` for N steps before permanently switching to
`base_reward_fn`) — **not what current training uses**, and explicitly NOT
ported to the Isaac side (the Isaac port only implements `reward_func`).

`HIT_REWARD = 50` — was `1000` once; the comment explains why that broke
training: 100-1000x every other term gave the critic a target range it
couldn't fit, producing heavy-tailed advantages that blew up KL divergence
even with gradient clipping and advantage normalization (neither bounds the
update DIRECTION, only magnitude).

## 2.4.1 Anti-oscillation terms: `stability_penalty` / `exit_penalty` (2026-09-19)

**Observed problem**: trained policies would approach the target, then
retreat before committing, then approach again -- repeating instead of
converging.

**Why it's NOT simple reward-farming**: `reward_func`'s shaping term is a
pure potential DIFFERENCE, `phi_now - phi_prev`. This **telescopes** over
any path: advance 2m then retreat 2m and the shaping reward nets to exactly
zero, regardless of how many times you do it (this is the same telescoping
property 0.4's GAE model-answer #5 and 0.7 both rely on). Pure back-and-forth
literally cannot farm free reward from that term -- so the oscillation had
to be coming from somewhere else.

**Actual cause (two, compounding)**:
1. `HIT_REWARD` fires on a single instantaneous `dist < HIT_THRESHOLD` check
   and ends the episode immediately -- zero reward for *staying* near the
   target, and a fast/imprecise approach can tunnel straight through that
   thin threshold band between two sampled physics steps without ever
   registering.
2. Committing to a precise final stop requires a sharper braking maneuver
   than approaching loosely; if that risks tipping past the attitude limit
   (instant terminal crash), then NOT fully committing -- hovering in a
   "close enough, still safe" oscillation -- can be the genuinely rational
   choice under the existing incentive structure. Not the policy cheating;
   the policy correctly reading the risk/reward as designed.

**Fix, two new terms** (both additive -- the existing instant-hit mechanic
is completely untouched, so nothing that worked before gets harder; this
only penalizes the specific unwanted behavior):
- **`stability_penalty`**: `-STABILITY_COEF*(|vel|+|roll|+|pitch|)`, active
  only when `dist < OUTER_ZONE_RADIUS` (1.0m) -- costs more to fly fast/
  tilted once close, rewarding settling instead of flying through.
- **`exit_penalty`**: comparing `prev_dist` (already tracked) against
  `dist` each step -- if the env was inside a zone LAST step and is outside
  it THIS step, charge a flat penalty (`INNER_ZONE_EXIT_PENALTY=-2.0` for
  the 0.5m inner zone, `OUTER_ZONE_EXIT_PENALTY=-1.0` for the 1.0m outer
  zone, `elif` so a single step exiting both only pays the worse one).
  Since shaping alone telescopes a round trip to exactly zero, adding this
  flat exit cost makes retreating **strictly worse than neutral** instead
  of neutral -- advance-then-retreat now costs more than never advancing.

**Worked example** (`start_dist=5m`, `max_episode_length=3750`, illustrative
assumptions stated per column):

| term | best (clean, ~222 steps) | avg (2 hesitant retreats, ~500 steps) | worst (oscillates whole episode, never hits) |
|---|---|---|---|
| shaping (telescoped, path-independent) | +0.95 | +0.95 | ~0 |
| step_penalty | -1.43 | -3.22 | -24.18 (bounded ceiling, 8.6's fix) |
| milestones | +45 | +45 | +10 |
| stability | -0.40 | -3.00 | -168.75 |
| exit | 0 | -2.00 | -112.00 |
| hit | +50 | +50 | 0 |
| **total** | **≈+94.1** | **≈+87.7** | **≈-294.9** |

Note the shaping row is IDENTICAL between best/avg despite the avg case
taking a longer, detour-laden path -- direct proof-by-example of the
telescoping property: only the non-shaping terms change with behavior.

**Not yet empirically tuned**: unlike `step_penalty` (derived via
`calibrate_approach_milestone_budget()`), the zone radii/penalty magnitudes
here are reasoned starting points, not analytically calibrated -- expect to
need real-run tuning.

## 2.5 `step_budget.py`

`steps_for_dist(dist) = max(MIN_STEPS, int(750·dist/3))` — scales the
episode length with target distance but floors at a settling-time minimum
(`MIN_EPISODE_SECONDS=7.5s`), since hitting `hit_threshold` precision takes
roughly the same settling time regardless of distance — a short-distance
task still needs real time to *stabilize*, not just travel.

## Part 2 practice questions

1. **(own words)** Explain `reward_func`'s reward at a high level, in your
   own words, in ≤5 sentences.
2. **(trace)** An episode has `start_dist=10`. At step 500, `dist=6`
   (`prev_distance` was 6.2 the step before). Roughly what does the
   shaping term contribute this step (sign and rough magnitude, not exact
   decimals)?
3. **(what breaks)** If `SubprocVecBaseDroneEnv` used `fork` instead of
   `spawn`, under what specific circumstance would that be dangerous, and
   why?
4. **(why this over alternative)** Why is `reward_func`'s shaping a plain
   `phi_now - phi_prev` instead of the textbook `γ·phi_now - phi_prev`
   potential-based form?
5. **(cross-file)** `terminal_observation` is stashed in `info` by the vec
   envs. Which downstream training code actually reads it, and what would
   go wrong for a truncated (not terminated) episode if it didn't exist?
6. **(cross-file)** `_select_pid_teacher` picks gains by nearest distance
   key in a JSON file. Which OTHER file computed the actual numbers in that
   JSON file, and by what method (search vs. closed-form)?
7. Why does `build_observation` use `sin(yaw_err), cos(yaw_err)` instead of
   the raw yaw error in radians?
8. What's the functional difference between `reward_func` and
   `RewardConfig`/`base_reward_fn`, and which one does current training
   actually use?
9. **(why this over alternative)** Explain why "the policy is farming the
   shaping term by oscillating" is provably NOT what was happening, using
   the telescoping property. If not that, what were the two actual causes?
10. **(trace a scenario)** An episode's drone enters the outer zone
    (1.0m), then exits it two steps later without ever entering the inner
    zone (0.5m) or hitting. Which penalty fires, and does the shaping term
    from those two steps get "taken back" by anything?
11. **(what breaks)** If `exit_penalty`'s `elif` were changed to two
    separate `if`s (no longer mutually exclusive), what's the one scenario
    where behavior would differ, and would it be a bigger or smaller
    penalty than intended?

<details><summary>Model answers</summary>

1. Every step, reward = change in (negative, normalized) distance to
   target since last step, minus a flat time-pressure penalty, plus
   one-time bonuses for crossing 25/50/75% progress and for hitting the
   target; large penalties and episode termination on going out of bounds,
   flipping over, or drifting away too long.
2. `diff = prev_distance - dist = 6.2-6 = 0.2` (positive, moved closer).
   `phi_now - phi_prev = -(6/10) - (-(6.2/10)) = 0.02` — small positive
   shaping reward, consistent with meaningful but modest per-step progress.
3. If the parent process has already initialized CUDA (e.g. the model is on
   GPU) before spawning workers — forking after CUDA init can inherit
   corrupted/locked driver state in the child process, since CUDA's
   internal locks aren't fork-safe.
4. Using PPO's own `γ<1` there left a residual `(γ-1)·φ(s)` reward every
   step even when standing still (φ is negative, so the residual is
   positive) — a near-free reward for not moving, undermining
   `step_penalty`'s entire purpose. A plain difference is exactly zero when
   nothing changes.
5. `app/guidance/train.py`'s `vec_ppo_train`/`isaac_ppo_train` (truncation
   bootstrap: `reward += γ·V(terminal_observation)`). Without it, a
   truncated episode's bootstrap value would be computed from the WRONG
   state (the next episode's fresh-spawn observation after auto-reset),
   valuing an unrelated state instead of the actual cut-off moment.
6. `app/control/tune_pid.py`, via closed-form 2nd-order pole placement (not
   a search) — `compute_gains_for_distance`.
7. Avoids the `±π` wraparound showing up as a discontinuous jump — 179° and
   -179° error are nearly the same physical heading but very different
   raw numbers; sin/cos are continuous across the wrap.
8. `reward_func` is flat, module-constant-driven, no phases, used by
   current training. `RewardConfig`/`base_reward_fn` is a config-class-based
   curriculum system with an optional chained imitation-then-base phase
   (`chain_reward_fns`) — an older system, not what current training uses,
   and not ported to Isaac.
9. The shaping term is a pure potential DIFFERENCE, `phi_now-phi_prev`,
   which telescopes over any path: the sum across a round trip (advance
   then return to the same distance) collapses to `phi(end)-phi(start) = 0`
   when start==end — no matter how many oscillations happen in between, the
   NET shaping reward from a round trip is always exactly zero, never
   positive. So oscillation can't be "farming" that term. The two real
   causes: (1) a single-instant `HIT_REWARD` check with zero reward for
   staying near the target, easy to overshoot/tunnel through between
   sampled steps; (2) committing to a precise final stop risks tipping past
   the attitude limit (instant crash), making cautious non-commitment a
   locally rational strategy under the existing incentives.
10. `OUTER_ZONE_EXIT_PENALTY` (-1.0) fires on the step it crosses back
    outside 1.0m (the inner zone was never entered, so `INNER_ZONE_EXIT_
    PENALTY`'s `elif` branch never triggers). The shaping reward from those
    two steps is NOT taken back by anything — exit_penalty is purely
    additive on top; shaping's own telescoping already nets that mini
    round-trip to ~zero on its own, independent of exit_penalty existing at
    all.
11. Without the `elif`, a single step that exits BOTH zones at once (a very
    fast movement jumping from inside the inner zone straight past the
    outer boundary in one step) would trigger BOTH penalties, stacking to
    -3.0 (`-2.0 + -1.0`) instead of just `-2.0` — a BIGGER penalty than
    intended. The design intent was "worse retreat gets the bigger single
    penalty," not "stack every threshold crossed in one step."

</details>

---

# PART 3 — Block: Control / PID

**Files**: `app/control/pid.py`, `torch_pid.py`, `tune_pid.py`,
`verify_pid.py`

## 3.1 `pid.py` — the numpy cascaded controller

`PIDController.compute_action(drone_state, target_pos, target_yaw, dt)`:
1. Position error → `accel_cmd = Kp_pos·err - Kd_pos·vel + Ki_pos·integral`
   (world-frame acceleration command).
2. **Yaw-rotate into body horizontal frame**: `accel_cmd` is world-frame
   x/y, but roll/pitch tilt the drone in its OWN (yaw-rotated) frame —
   rotating by `-yaw` before mapping to tilt is required or any nonzero
   `target_yaw` makes the drone tilt toward the wrong world direction once
   it actually turns.
3. `des_roll = clip(-accel_body_y/g, ±max_tilt_rad)`,
   `des_pitch = clip(accel_body_x/g, ±max_tilt_rad)` — small-angle
   approximation: horizontal acceleration ≈ `g·tan(tilt) ≈ g·tilt`.
4. Inner attitude loop: `roll_torque = Kp_att·att_err - Kd_att·ang_vel_x +
   Ki_att·integral`, same for pitch/yaw.
5. `thrust_delta = Kp_pos·pos_err_z - Kd_pos·vel_z + Ki_pos·integral_z`
   (z-axis reuses the position loop's own gains, not a separate one).

All 3 integral terms are clamped (`integral_limit_pos/att`) — anti-windup.

## 3.2 `torch_pid.py` — batched mirror (migration step 6)

Same math, `(num_envs,...)` tensors, per-env integral state. **Diff-tested
bit-exact** over 20 steps × 32 envs of *stateful* recursion (integral terms
carried across steps, not just one-shot) — `scripts/isaac_lab/diff_test_pid.py`.
`assign_gains_by_distance` mirrors `subproc_vec_base_drone_env.py`'s
`_select_pid_teacher` — nearest-distance gain lookup, per-env.

## 3.3 `tune_pid.py` — closed-form gain derivation (0.3's pole placement,
applied)

`compute_gains_for_distance(dist)`:
```
settle_time = steps_for_dist(dist)·DT·0.5
wn_pos = ln(dist/HIT_THRESHOLD) / (ζ_pos · settle_time)
kp_pos = wn_pos²;  kd_pos = 2·ζ_pos·wn_pos
wn_att = 5·wn_pos;  kp_att = Ixx·wn_att²;  kd_att = 2·ζ_att·wn_att·Ixx
wn_yaw = 2·wn_pos;  kp_yaw = Izz·wn_yaw²;  kd_yaw = 2·ζ_yaw·wn_yaw·Izz
```
Two deliberate departures from textbook settling-time formulas, both worth
being able to explain:
- **Not** the generic `t_settle ≈ 4/(ζ·ωn)` (2%-of-initial-error) rule —
  that targets 2% of `dist` (e.g. 5m of slack at 250m), far looser than
  this controller's real `HIT_THRESHOLD=0.05m`. Instead solves
  `error(t)=dist·exp(-ζωn·t)` for `error(settle_time)=HIT_THRESHOLD`
  directly — the "4" in the generic rule is really
  `ln(1/0.02)≈3.9`, constant regardless of target precision; this project's
  own `ln(dist/HIT_THRESHOLD)` grows with distance (~4.1 at 3m, ~8.5 at
  250m) instead.
- **Yaw's bandwidth separation (2x) is smaller than attitude's (5x)**:
  because yaw torque authority comes from `k_m = 0.02·k_f` — only 2% of the
  thrust coefficient — demanding the same aggressive `wn` as roll/pitch
  would need torque the rotors can't deliver without saturating `max_rpm`.

`verify_gains`: one-shot sanity check (not a search) — runs the derived
gains through the real sim across every axis/sign and two yaws, reports
worst-case hit rate. `calibrate_approach_milestone_budget`: measures the
tuned PID's max achievable `reward_func` total (temporarily patching
`APPROACH_MILESTONE_BUDGET` to 0 to avoid a self-referential measurement) —
the number to hand-copy into `rewards.APPROACH_MILESTONE_BUDGET`.

**Update (2026-09-25).** `DISTANCES` is now `(3, 10, 20, 30, 50, 100, 150, 250)`.
`20` and `30` were added for the 3-30 m curriculum stage: nearest-neighbour
lookup (`assign_gains_by_distance`) puts every target on the closest key, and
with only `10` and `50` between 3 and 100 the midpoint was exactly 30 m, so
almost the whole 10-30 m span used the 10 m gains — tuned for a ~3× smaller
step budget. Re-running `python -m app.control.tune_pid` regenerates both
`best_pid_gains_per_dist.json` and the generic `best_pid_gains.json` (whose key
is the middle element of `DISTANCES`, now `50`, was `100`). Gains at 30 m are
6× softer than at 10 m (`kp_pos` 0.232 vs 1.43) — long-range episodes are slow
by design (0.3.C).

## 3.4 `verify_pid.py`

Standalone script: loads `best_pid_gains_per_dist.json`, runs
`run_eval_matrix` across the full `DISTANCES` ladder, prints hit/fail per
pair, plots via `plot_eval_matrix_distance`/`plot_eval_matrix_pairs`. Exit
message tells you whether it's safe to proceed to `collect_demonstrations.py`.

## Part 3 practice questions

1. **(own words)** Explain the outer-to-inner data flow of
   `PIDController.compute_action` in ≤5 sentences.
2. **(trace)** `target_yaw=0`, drone currently yawed 90° (`yaw=π/2`),
   `accel_cmd=[2,0,0]` (world-frame, 2 m/s² in +x). What are
   `accel_body_x`/`accel_body_y` after the yaw rotation? (Hint: at yaw=90°,
   world +x becomes which body axis?)
3. **(what breaks)** If you deleted the integral clamp
   (`integral_limit_pos`), what specific failure mode would you expect
   during a large, sustained position error?
4. **(why this over alternative)** Why does `tune_pid.py` solve gains
   analytically instead of using Optuna/a search, given a search was
   already available elsewhere in this project's history?
5. **(cross-file)** `compute_gains_for_distance` reads `_CONFIG.inertia`.
   Where does that number ultimately come from, and what happens to the
   derived gains if that mass/inertia ever drifts from what
   `base_drone_env.py:reset()` actually builds?
6. **(cross-file)** `torch_pid.py`'s diff-test checks 3 separate internal
   state tensors beyond just the output action. Name them and explain why
   checking them (not just the action) matters for a STATEFUL controller.
7. Why is yaw's bandwidth-separation multiplier smaller than roll/pitch's?
8. What single value does `calibrate_approach_milestone_budget` produce,
   and which module-level constant does it feed?

<details><summary>Model answers</summary>

1. Position error → outer-loop acceleration command → rotate into body
   frame by `-yaw` → map to a desired tilt angle (clamped to `max_tilt_rad`)
   → attitude error → inner-loop torque command; z-axis reuses the
   position loop's own gains for thrust.
2. At yaw=90°, `cos(yaw)=0, sin(yaw)=1`:
   `accel_body_x = 2·0 + 0·1 = 0`; `accel_body_y = -2·1 + 0·0 = -2`. World
   +x has become body -y (a 90° yaw rotates the body frame so what was
   "forward" in world is now "left/right" in body).
3. Integral windup: the term keeps growing unbounded while error persists,
   then causes a large overshoot once the error finally starts closing
   (the accumulated integral term keeps commanding hard even after the
   error is gone).
4. Because the physics/gain relationship is analytically tractable (a
   standard 2nd-order pole-placement identity given known mass/inertia) —
   a search is unnecessary machinery for a problem with a closed-form
   solution, and gives an exact answer instead of an approximate one from
   sampling.
5. `create_quad_config` in `base_drone_env.py:reset()` (mass=1.5,
   inertia=(0.02,0.02,0.04)) — `tune_pid.py`'s own `_CONFIG` is built with
   the SAME hardcoded numbers, with an explicit comment that gains derived
   from any other mass/inertia would be tuned for the wrong plant. If they
   ever drift apart, the derived gains would no longer match the actual
   simulated drone.
6. `integral_pos`, `integral_att`, `integral_yaw`. A stateful controller's
   correctness depends on its accumulated internal state evolving
   identically over multiple steps, not just producing the right action
   from a given state — checking only the output action on a single call
   could hide a divergence that compounds over an episode.
7. Yaw torque authority (`k_m = 0.02·k_f`) is only 2% of the thrust
   coefficient — an aggressive yaw bandwidth would demand torque the
   rotors physically can't deliver without saturating `max_rpm`.
8. The max achievable `reward_func` total (hit bonus included, step_penalty
   excluded via the temporary patch) across the tested distances — feeds
   `rewards.APPROACH_MILESTONE_BUDGET`.

</details>

---

# PART 4 — Block: RL Core (PPO)

**File**: `app/guidance/train.py` only (plus its Isaac-native additions
from this session — noted inline, cross-referenced fully in Part 8)

## 4.1 `ActorCritic` — shared-trunk network

`shared` = N `Linear→Tanh(+Dropout)` layers. `actor_mean` (Linear head),
`actor_log_std` (a learned but STATE-INDEPENDENT parameter — one value per
action dim, not a function of the observation), `critic_head` (Linear→
scalar). `forward` applies a final `ReLU` after the shared trunk, then
splits into `(action_mean, action_std, state_value)`.

**`log_std` clamping**: `log_std_min + 0.5·(log_std_max-log_std_min)·
(tanh(actor_log_std)+1)` — squashes the raw learnable parameter through
`tanh` into a fixed `[log_std_min, log_std_max]` range, so std can never
collapse to exactly 0 (would make log-prob computation degenerate) or
explode arbitrarily large.

**`scale_action`** (deterministic/eval path): `tanh(raw)` then affine
rescale into `[action_low, action_high]`. **`get_action_and_value`**
(stochastic/training path): samples `raw_action ~ N(mean,std)` (or reuses a
given `raw_action` — needed so `ppo_update`'s re-evaluation pass computes
log-prob against the SAME pre-squash sample the rollout buffer stored, not
a fresh sample), squashes+rescales, and applies the **change-of-variables
log-prob correction** (0.4):
```python
log_prob = dist.log_prob(raw_action).sum(-1) - torch.log(half_range·(1-squashed²)+1e-6).sum(-1)
```

**Update (2026-09-27).** `ActorCritic(..., detach_critic=False)` — when True the
value head reads `x.detach()`, so value-loss gradients never reach the trunk the
actor also reads (0.10). It is a plain Python flag (not saved in checkpoints).
Parameter count of the production network: 14,345 (0.10.C). `log_std_max` is
now tuned (−2.3) while `log_std_min = −3.0` stays the hard-coded floor.

## 4.2 `RolloutBuffer` / `compute_gae`

Flat `(num_steps, ...)` tensors: obs, actions (pre-squash raw), log_probs,
rewards, values, dones. `compute_gae`: backward recursion (0.4's formula),
also handles the **batched `(num_steps, num_envs)` case** via plain
broadcasting — the same function serves both single-env and vectorized
training without special-casing.

## 4.3 `ppo_update` — the epoch/minibatch loop

For `num_epochs`, shuffled minibatches of size `batch_size`: recompute
`log_prob`/`entropy`/`value` under the CURRENT policy for the SAME stored
`(obs, raw_action)`, compute the clipped surrogate policy loss + value MSE
+ entropy bonus, backprop, clip grad norm, step. **Early-stops** the epoch
loop once the batch's average `approx_kl` exceeds `target_kl` (0.4).

**This session's additions** (flag-switchable, off by default, still
present in this same function — see Part 8 for how the Isaac path exercises
them): `value_clip_eps` (PPO2-style clipped value loss — clips the value
update to `old_value ± ε`, takes the max of clipped/unclipped squared
error, same idea as the policy ratio clip but for the critic), and
`distill_target_actions`/`distill_coef` (adds `coef·MSE(scaled_action,
target)` to the loss — a generic student-teacher distillation term,
agnostic to what produced the target actions).

## 4.4 `ppo_train` / `vec_ppo_train` — the numpy-side rollout loops

`ppo_train`: single env, numpy `<->` torch conversion every step.
`vec_ppo_train`: same algorithm, batched across `vec_env.num_envs` —
**one batched forward pass instead of `num_envs` separate ones**, which is
"what actually gives a GPU something to chew on" (per the in-file comment)
— collects rollouts in lockstep, truncation-bootstraps via
`infos[i]["terminal_observation"]` per env.

## 4.5 `warmup_critic`

Freezes nothing itself (relies on the CALLER having frozen the actor, e.g.
`base_training.py`'s own actor-freeze) — collects rollouts with the current
actor, then for 4 direct gradient steps per round, regresses `critic_head`
toward GAE-computed returns via plain MSE, with a cosine-decayed
per-round learning rate. **NOT what `base_training.py`'s critic-warmup
phase actually calls** — that phase freezes the actor and just runs normal
`vec_ppo_train` (see Part 6) — `warmup_critic` is a separate, standalone
tool that exists in this file but isn't wired into the main training
orchestration. (This exact "which mechanism does the orchestrator ACTUALLY
call" distinction was a real bug found this session on the Isaac side —
Part 8.)

## 4.6 `evaluate` / `load_bc_checkpoint`

`evaluate`: N deterministic episodes (mean action, no sampling), counts
`info["reason"]=="Hit"`. `load_bc_checkpoint`: remaps a BC/DAgger
checkpoint's `shared.*` keys by **Linear-layer POSITION**, not raw
`nn.Sequential` index — `ActorCritic` only inserts a `Dropout` sublayer
when `dropout>0`, which shifts every later Linear's raw index; every BC/
DAgger checkpoint is saved at `dropout=0`, so loading into a model built
WITH dropout would silently mismatch under a naive `load_state_dict` (the
weights wouldn't error, they'd just land on the wrong layers).

## Part 4 practice questions

1. **(own words)** Explain why `ActorCritic` needs a log-prob correction
   when actions are tanh-squashed, in ≤5 sentences.
2. **(trace)** A minibatch's `approx_kl` after epoch 3 (of 10) is 0.03,
   `target_kl=0.02`. What happens next in `ppo_update`, concretely?
3. **(what breaks)** If `get_action_and_value` were called with
   `raw_action=None` during the `ppo_update` re-evaluation pass (instead of
   the buffer's stored raw_action), what would go wrong with the resulting
   policy gradient?
4. **(why this over alternative)** Why is `actor_log_std` a learned
   parameter independent of the observation, rather than a third output
   head of the network (state-dependent std)?
5. **(cross-file)** `load_bc_checkpoint` exists because of an interaction
   between two specific things. Name both.
6. **(cross-file)** `compute_gae` is used identically by both single-env
   `ppo_train` and batched `vec_ppo_train`. What property of its
   implementation makes that possible without any special-casing?
7. What's the actual difference between `warmup_critic` and what
   `base_training.py` calls its "critic warmup" phase?
8. Why does `vec_ppo_train` need `infos[i]["terminal_observation"]`
   specifically, instead of just using `next_obs[i]`?

<details><summary>Model answers</summary>

1. The policy's actual output distribution is over the SQUASHED action
   (what the environment receives), not the raw Gaussian sample — scoring
   log-prob against the raw Gaussian directly would use the wrong density,
   biasing the policy gradient; the tanh+affine Jacobian's log-determinant
   must be subtracted to correct for the change of variables.
2. This epoch's loop breaks early (`early_stopped=True`) — the batch
   already drifted past the trust-region KL threshold, so further epochs
   on this SAME batch are skipped (KL isn't reset until the next round's
   fresh rollout).
3. A fresh sample would be drawn from the CURRENT policy instead of
   reusing the exact sample the rollout buffer's stored `log_prob`/reward
   correspond to — the importance-sampling ratio `π_new/π_old` would then
   be computed against mismatched samples, invalidating PPO's whole
   off-policy correction.
4. Simplicity/stability — a state-dependent std head can collapse toward
   zero on states it's seen often (premature exploitation) faster than a
   single global parameter that decays more uniformly across the whole
   state space over training.
5. `ActorCritic.__init__` only inserts a `Dropout` sublayer into `shared`
   when `dropout>0` (shifting later `Linear` layers' raw `nn.Sequential`
   index); every BC/DAgger checkpoint is saved with `dropout=0`.
6. `compute_gae`'s backward recursion is written with plain tensor
   broadcasting, not explicit per-env loops — it works identically whether
   `rewards`/`values`/`dones` are shape `(num_steps,)` or
   `(num_steps, num_envs)`.
7. `warmup_critic` is a standalone function (its own rollout collection +
   direct 4-step MSE regression loop). `base_training.py`'s actual "critic
   warmup" phase just freezes the actor and runs NORMAL `vec_ppo_train`
   chunks — the two are architecturally different mechanisms with the same
   goal, and `warmup_critic` is not what the orchestrator calls.
8. A truncated env auto-resets before `vec_ppo_train` gets its return value
   — `next_obs[i]` for that env already belongs to its NEXT episode
   (fresh spawn), not the true cut-off state; bootstrapping off the wrong
   state would value an unrelated situation entirely.

</details>

---

# PART 5 — Block: Imitation Learning

**Files**: `app/control/pretrain_bc.py`, `collect_demonstrations.py`,
`dagger.py`

## 5.1 `pretrain_bc.py`

`pretrain_behavior_cloning(model, obs, actions, weights=None, epochs, lr,
weight_decay)`: converts `(obs, actions)` (numpy or already-loaded) to
tensors, per-epoch shuffled minibatches, maps target actions into the
RAW pre-tanh space (`atanh` of the normalized target, clamped to
`±0.999` to keep `atanh` finite) so they can be scored against the
network's raw Gaussian output via **Gaussian NLL** (0.5): each action
dimension is normalized by its OWN learned variance, so thrust (large
magnitude) doesn't dominate yaw torque (small magnitude) in the loss.
`weights`, if given, is a per-sample loss weight (DAgger's recency
weighting plugs in here) — `None` reproduces plain uniform BC.

## 5.1a `actor_log_std` collapses to the floor under this loss (found 2026-09-24)

**Symptom**: `runs/base_training_isaac_3_10_fix/metrics.csv`'s
`effective_std_mean` is bit-identical (`0.049787...`) on EVERY logged row,
from the very first `critic_warmup` checkpoint through the last `training`
row 126M steps later. `entropy_loss` is likewise frozen at exactly `6.3242`
the whole run — for a Gaussian, differential entropy is a function of `std`
alone (`0.5*log(2*pi*e*sigma^2)`), independent of the mean, so a frozen std
necessarily means frozen entropy regardless of anything the mean/actor is
doing. `0.049787...` is exactly `exp(DEFAULT_LOG_STD_MIN) = exp(-3.0)` (Part
4.1's `log_std` clamp floor) — i.e. `tanh(actor_log_std) ≈ -1`, fully
saturated at the LOW end of the clamp.

**Root cause, traced end to end**: `pretrain_behavior_cloning`'s Gaussian NLL
(5.1 above) scores BOTH the mean and `actor_log_std` against the PID
teacher's action. The teacher is a deterministic controller — same state,
same action, every time — so the imitation targets carry no actual variance
for a Gaussian to represent. Minimizing NLL against a target with no spread
is minimized by shrinking `std` toward zero (tighter density around a
correctly-predicted point = lower loss), and nothing in `pretrain_bc.py` or
`dagger.py` counteracts that pull. Over enough BC/DAgger epochs,
`actor_log_std`'s raw (pre-tanh) value gets driven deeply negative until
`tanh` saturates at `-1`. `load_bc_checkpoint` (Part 4, `app/guidance/
train.py`) then loads that saturated value into the Isaac PPO run verbatim,
same as every other weight — nothing about loading a BC checkpoint
special-cases `actor_log_std`.

**Why this wrecks PPO once loaded**: once `actor_log_std` is pinned at the
floor, its raw (pre-`tanh`) value sits several units deep in the saturated
tail, where `tanh' ≈ 1e-4`. (Precision, added 2026-09-27: Adam moves each
parameter by at most ~`lr` per step *whatever* the gradient's size, so it is
not that the gradient is exactly zero — it is that covering several units of
raw distance at ≤ `lr` per step takes tens to hundreds of chunks even with a
consistent gradient sign, far more with the noise the sign actually has, while
`σ`'s sensitivity to the raw value is `tanh'`, so the logged `effective_std`
looks bit-identical.) In practice PPO inherits a policy that can never adjust
its own exploration noise, up or down, for the rest of the run. Worse, a near-zero std makes the PPO probability ratio
(`π_new/π_old`) hypersensitive to tiny mean shifts (dividing by a tiny std
amplifies any log-prob difference), so the very first real gradient step
after ANY actor-unfreeze overshoots `target_kl` badly and craters the grade.
This is directly visible at both unfreeze points in that run's `metrics.csv`:
critic-warmup-end unfreeze (`approx_kl` jumps `0.075 -> 0.235` against
`target_kl=0.0245`, grade `0.834 -> 0.403`) and the post-reset unfreeze later
in the same run — the same signature both times, because it's the same
frozen-std cause both times, not two unrelated instabilities.

**This is NOT the same knob as `PARAMS['log_std_max'] = -0.9`.** That's the
CEILING of the clamp, deliberately tightened (from a looser default) to stop
a fully-trained policy from lingering with too much exploration noise near a
missed target instead of committing to a final corrective approach (see
`base_training.py`'s `ENT_COEF_START`/`END` comment, 2026-09-17 discussion).
`DEFAULT_LOG_STD_MIN = -3.0` (the FLOOR) was never touched by that decision
and isn't meant to be a resting value the parameter starts pinned at before
training even begins — the floor exists so std can't collapse to literal
zero (degenerate log-prob), not so a pre-trained checkpoint arrives already
sitting on it with no gradient able to move it.

**Fix, and a first attempt that didn't survive**: the first fix reset
`actor_log_std` right after `load_bc_checkpoint`. That alone doesn't hold —
`train()` runs its OWN on-policy imitation stage next
(`_run_imitation_stage_isaac`, `IMITATION_FRACTION` of the budget), which
calls the SAME `pretrain_behavior_cloning` Gaussian NLL against the SAME
deterministic PID teacher, and re-collapses `actor_log_std` right back down
before `critic_warmup` ever starts — confirmed in
`runs/base_training_isaac_3_10`, where `effective_std_mean` was still
pinned at `~exp(-3.0)` for the entire run despite that first reset. The reset
has to happen AFTER `_run_imitation_stage_isaac`, immediately before the
`actor_frozen`/critic-warmup freeze locks the value in for good: reset
`actor_log_std` to `torch.zeros_like(...)` — its original
`ActorCritic.__init__` value. `tanh(0) = 0` sits at the MIDPOINT of
`[log_std_min, log_std_max]` (`≈ exp(-1.95) ≈ 0.142` for the `[-3.0, -0.9]`
band above), squarely in `tanh`'s near-linear, gradient-carrying region. This
keeps the (legitimately good) `shared`/`actor_mean`/`critic_head` weights —
both the BC-loaded ones and whatever the on-policy imitation stage further
refined them to — and only discards the collapsed `actor_log_std`, which was
never a meaningful calibrated exploration setting to begin with, just an
artifact of imitating a variance-free teacher, twice over. It does not touch
or loosen `log_std_max=-0.9` — PPO can still only ever move `actor_log_std` within
that same deliberately-narrow band; this fix just stops it from starting the
run already glued to the wrong end of it.

## 5.2 `collect_demonstrations.py`

`collect_demonstrations`: sequential, one drone, iterates
`build_eval_pairs` (axis-aligned lattice) per distance, jitters the spawn,
records `(obs, pid_action)` before every `env.step`. `collect_demonstrations
_omni`: the RICHER version — `sample_omni_target` draws a random
single/pair/triple-axis category (weighted `CATEGORY_WEIGHTS = {"single":
0.2, "pair":0.3, "triple":0.5}`), with elevation/azimuth jitter around that
category's nominal direction, instead of a fixed lattice — greedily fills
whichever category is furthest below its target row share each episode
(live-tracked balancing). **This session's addition**,
`collect_demonstrations_base_drone_isaac`: batched, `num_envs` parallel
episodes on the Isaac env, using the SAME `sample_omni_target` sampler
(not the fixed lattice) but with a **periodically-refreshed pool**
(proportional to `CATEGORY_WEIGHTS`, not live-tracked per-episode) — a
statistical, not exact, match to the numpy version's balancing (see Part 8
for why exact per-env balancing was tried and reverted as buggy).

## 5.3 `dagger.py`

`dagger(gains_by_dist, n_rounds, ...)`: per round, per distance — the
CURRENT policy (deterministic mean) drives the env, the distance-matched
PID only supplies the corrective label (`build_omni_eval_pairs`, the fixed
lattice covering every axis-combo, not the continuous sampler here).
Raw per-distance pair counts are **subsampled down to the smallest
distance's count** before aggregating (long-distance episodes run more
steps, would otherwise dominate the BC retrain loss). Aggregate buffer
capped at 500,000 pairs with **recency-weighted eviction**
(`RECENCY_DECAY=0.85`): `keep_probability ∝ decay^(rounds_old)`.

**This session's addition**, `dagger_base_drone_isaac`: batched,
`num_envs` parallel on-policy episodes, same recency-weighted
aggregate/eviction/retrain structure, now ALSO using the continuous
`sample_omni_target` pool (upgraded from `build_omni_eval_pairs`'s fixed
lattice this session, per direct request). Known, documented gap: no
per-distance row balancing (the pool-based collection doesn't track/
subsample per-distance the way the numpy version explicitly does).

## Part 5 practice questions

1. **(own words)** Explain why BC alone (no DAgger) tends to degrade over a
   long deployed episode, in ≤5 sentences.
2. **(trace)** DAgger round 3 collects 50,000 new pairs; the aggregate
   before this round was 480,000 pairs; the cap is 500,000. What happens to
   the aggregate, and by what rule are pairs chosen to keep/evict?
3. **(what breaks)** If `pretrain_behavior_cloning` were given targets
   without first mapping them into the raw pre-tanh space (scored directly
   against the squashed mean via plain MSE), what would go wrong for
   targets near the action bounds?
4. **(why this over alternative)** Why does `dagger()` subsample every
   distance's raw pair count down to the smallest before aggregating,
   instead of just using everything collected?
5. **(cross-file)** Both `collect_demonstrations_omni` and
   `dagger_base_drone_isaac` use `sample_omni_target`. Which file actually
   defines that function, and is it Isaac-specific?
6. **(cross-file)** `dagger()`'s retrain call passes `weights=train_w`.
   Which OTHER file's function receives and uses that weight, and how?
7. What's the one thing `dagger_base_drone_isaac` does NOT do that the
   numpy `dagger()` does, and why was it dropped rather than ported?
8. In `collect_demonstrations_base_drone_isaac`, why is the omni-target
   pool refreshed periodically instead of drawing a genuinely fresh sample
   for every single episode?

<details><summary>Model answers</summary>

1. BC only ever sees states the EXPERT visits during data collection; once
   the trained policy makes even a small mistake, it drifts into a state
   the expert rarely visited, has no training signal there, and the error
   compounds — a train/deployment state-distribution mismatch (covariate
   shift).
2. `agg_obs`/`agg_actions` grow to 530,000 pairs (temporarily over cap),
   then get subsampled back down to 500,000 via
   `np.random.choice(..., p=keep_w/keep_w.sum())`, where
   `keep_w = RECENCY_DECAY^(rounds_old)` — older pairs are more likely
   (not certain) to be evicted, not uniformly random and not strictly
   chronological.
3. Near the bounds, `tanh` saturates — its derivative approaches 0, so the
   network's raw pre-squash output would need to go to `±∞` to hit a
   bound-adjacent target under plain MSE against the squashed value,
   producing exploding gradients / an ill-conditioned loss landscape;
   mapping into raw space via `atanh` first avoids this entirely.
4. Long-distance episodes run far more steps than short ones, so raw pair
   counts are wildly unequal — without subsampling, the BC retrain loss
   would be dominated by whichever distance happened to produce the most
   steps, not weighted by task importance.
5. `app/control/collect_demonstrations.py` — not Isaac-specific, pure
   numpy, reused unchanged (imported with a lazy `isaaclab` import
   elsewhere in the same file, but `sample_omni_target` itself has zero
   isaaclab dependency).
6. `pretrain_bc.py:pretrain_behavior_cloning` — used as a per-sample loss
   weight in the Gaussian NLL: `loss = (per_sample_nll · w).sum()/w.sum()`,
   so older, lower-weight samples matter less to the retrain gradient.
7. Per-distance row balancing (subsampling every distance to the smallest
   count before aggregating) — dropped because the shared-pool API
   (`set_target_pairs`, uniform random draw per reset) has no per-env
   "assign this env exactly this distance" hook to balance against
   cleanly; flagged as a known gap, not silently dropped.
8. A genuinely fresh per-episode sample would need PER-ENV target
   assignment at exactly the right moment relative to that env's own
   reset — an earlier attempt at this had a real staleness bug (category
   bookkeeping updated before the new target actually took effect); the
   periodic-refresh pool sidesteps that entirely at the cost of being
   statistical rather than exact.

</details>

---

# PART 6 — Block: Orchestration

**Files**: `app/training/base_training.py`, `base_training_isaac.py`,
`diagnostics.py`, `eval_matrix.py`

## 6.1 `base_training.py` — the numpy full-pipeline orchestrator

Budget split (1,000,000 per-env steps, `NUM_ENVS=128` workers):
**15% on-policy imitation → 1% critic-only warmup → 84% PPO**, in that
order.
- **Imitation** (`_run_imitation_stage`): the actor (warm-started from a
  BC/DAgger checkpoint) drives every worker, each worker's distance-matched
  PID supplies the correction label via `step_with_pid_actions`. Every
  `IMITATION_RETRAIN_EVERY` steps, folds fresh pairs into a recency-weighted
  aggregate (capped, evicted same as `dagger()`) and runs one
  `pretrain_behavior_cloning` pass. Training only on each block's OWN fresh
  pairs (discarding afterward) was tried first and produces catastrophic
  forgetting — aggregating is what DAgger's convergence argument actually
  depends on.
- **Critic warmup**: freezes `shared` + `actor_mean` + `actor_log_std`
  (NOT the critic head), then runs the SAME `vec_ppo_train` used for real
  training — `optimizer.step()` naturally skips any param with `grad=None`.
  This is NOT `warmup_critic()` (Part 4.5) — a real, easy-to-miss
  distinction.
- **PPO**: chunked (`CHUNK_TIMESTEPS=20,000` per-env steps/chunk), with
  cosine-decayed lr and linearly-decayed entropy coefficient per chunk.
  Every chunk: save checkpoint (`.pt`+`.onnx`), run `diagnose_with_model`
  (N deterministic episodes) AND `diagnose_with_pid` (the teacher on the
  SAME target pairs, so the plots show whether RL is converging toward the
  PID ceiling or diverging from it), log everything to MLflow + a CSV row.

**Curriculum staging**: NOT an auto-widening curriculum within one run —
chain multiple `train()` calls with increasing `(distance_low,
distance_high)`, each warm-starting `bc_checkpoint_path` from the previous
stage's final checkpoint (`load_bc_checkpoint`, remapped by Linear
position).

## 6.2 `base_training_isaac.py` — this session's Isaac-native counterpart

Same 15/1/84 split and order, imported directly from `base_training.py`
(not reinvented — see Part 8 for the two real bugs found while porting
this). Budget expressed as **grand-total env-steps across ALL parallel
envs** (not per-env like numpy's constant), since `NUM_ENVS_ISAAC` scales
with GPU capacity rather than CPU core count — a fixed per-env constant
reused as-is would make the grand total explode or shrink arbitrarily with
whatever `--num_envs` is passed.

Diagnostics/PID-baseline comparison and plotting **reuse the numpy oracle
unchanged** — `diagnose_with_model`/`diagnose_with_pid`/`plot_training_run`
run against a plain `BaseDroneEnv`, not the Isaac env, since both are
policy-quality checks (N deterministic episodes), not physics throughput,
and the numpy oracle is diff-tested to track the Isaac physics closely.

**Update (2026-09-27) — what `base_training_isaac.train()` does now** (see
Part 12 for the story behind each item):
- **Budget split:** imitation `IMITATION_FRACTION = 0.08` (was 0.15; skipped
  entirely in residual mode), critic warmup = `max(1%, 16 chunks)`, PPO = the
  rest. Chunk size is `--num-steps-per-chunk × num_envs`.
- **Run start:** the first `RUN_START_FROZEN_CHUNKS = 3` chunks are
  rollout-only (`skip_update=True`).
- **`actor_log_std` reset to 0** after the imitation stage, before the warmup
  freeze locks it in (5.1a).
- **Schedules** live in shared helpers so the Optuna objective can't drift:
  `training_progress_at`, `entropy_coef_at` (held through warmup, then linear
  start → end), `scheduled_lr` (flat through warmup, then the original
  full-span cosine × the unfreeze ramp).
- **Soft reset** (8.7) now `RESET_DEGRADE_MARGIN = 0.4`, `RESET_BLEND_ALPHA =
  0.75`, then `RESET_WARMUP_CHUNKS = 2` critic-only chunks.
- **Promotion gate** `SOLID_*` → `SOLID.json` + early stop (0.11.C).
- **Residual mode** `--residual-scale S` (`PidResidualEnv`), `--residual-resume`,
  `--detach-critic` (0.13.C).
- **Logging:** `metrics.csv` per run, the shared `runs/isaac_training_metrics.json`
  (one key per run, `_append_json_row`), MLflow to `mlflow_isaac.db`.
- **Diagnostics** use 30 episodes and, in residual mode, apply the same
  PID+correction composition as training (`diagnose_with_model(residual_scale,
  gains_by_dist)`).

## 6.3 `diagnostics.py`

`_run_diagnostic_episodes`: shared rollout/outcome-counting loop.
`diagnose_with_model`: policy drives (deterministic mean action).
`diagnose_with_pid`: the PID teacher drives instead (optionally swapping to
nearest-distance gains per reset, matching `_select_pid_teacher`) — same
counting logic, different action source, giving an apples-to-apples
RL-vs-PID comparison on identical episodes.

## 6.4 `eval_matrix.py`

Deterministic `(start, target, target_yaw)` pair builders, for reproducible
scoring independent of whatever random target `env.reset()` would sample:
`build_eval_pairs` (one axis at a time, both signs), `build_omni_eval_pairs`
(every single/pair/triple-axis combo, equal-magnitude components, fixed
lattice), `build_random_omni_eval_pairs` (full-sphere random directions,
fixed distance ladder), `build_uniform_omni_eval_pairs` (full-sphere random
direction AND continuous `Uniform(low,high)` distance — what
`base_training.py` actually trains against). `run_eval_matrix`: runs any
`get_action(obs,env)` function through every pair `n_repeats` times,
returns per-pair hit-rate/final-distance summary stats.

## Part 6 practice questions

1. **(own words)** Explain, in order, the three phases `base_training.py`
   runs and roughly why each exists, in ≤5 sentences.
2. **(trace)** `WARMUP_DURATION_STEPS=10,000`, `CHUNK_TIMESTEPS=20,000`.
   During chunk 0 (`ppo_timesteps_done` goes from 0 to 20,000), at what
   point does the actor get unfrozen, and does chunk 0 run with a MIX of
   frozen/unfrozen actor or one or the other throughout?
3. **(what breaks)** If `_run_imitation_stage` trained only on each
   block's own fresh pairs (discarding them after each retrain), what
   specific failure mode was observed, and why does aggregating fix it?
4. **(why this over alternative)** Why does `base_training_isaac.py`
   express its budget as a GRAND TOTAL across all envs instead of reusing
   `base_training.py`'s per-env `TOTAL_TIMESTEPS` constant directly?
5. **(cross-file)** `diagnose_with_pid` can optionally take `gains_by_dist`.
   Which function does it internally call to actually swap gains per
   episode, and where does THAT function normally get called from in
   non-diagnostic code?
6. **(cross-file)** `base_training.py`'s PPO chunk loop calls
   `export_onnx_model` every chunk. Which file defines that function, and
   what does it need to infer about a model it wasn't given directly?
7. Which eval-matrix builder does `base_training.py`'s actual training use
   for TARGET SAMPLING (not just deterministic eval scoring), and how does
   it differ from the fixed-lattice builders?
8. What two things does `verify_gains`/`compute_pid_baseline`-style
   diagnostic logging protect you from, conceptually, that a training curve
   alone would not reveal?

<details><summary>Model answers</summary>

1. Imitation first refines a BC/DAgger-warm-started policy on-policy
   against the PID teacher, before RL's reward-only objective ever touches
   it. Critic warmup next calibrates the (randomly initialized) value
   function against the now-good policy, since an untrained critic would
   otherwise inject garbage advantage estimates into the first real PPO
   updates. PPO last is the main phase, free to exceed what PID/BC could
   do since it's optimizing the actual task reward.
2. The unfreeze check happens at the START of each chunk
   (`if actor_frozen and ppo_timesteps_done >= WARMUP_DURATION_STEPS`), not
   mid-chunk — so chunk 0 (0→20,000) runs ENTIRELY frozen (since
   `ppo_timesteps_done=0 < 10,000` at its start), and gets unfrozen only
   before chunk 1 begins. No chunk mixes frozen and unfrozen actor
   mid-chunk.
3. Catastrophic forgetting — the network fit whatever narrow slice of
   state-space the latest rollout visited, with nothing anchoring it to
   what earlier blocks already taught it. Aggregating (with recency
   weighting) keeps earlier competence represented in the retrain data
   instead of discarding it every block.
4. `NUM_ENVS_ISAAC` scales with GPU capacity (thousands), not CPU core
   count (128) — reusing a per-env constant as-is would make the grand
   total scale arbitrarily (up or down) with whatever `--num_envs` happens
   to be passed, instead of representing a deliberately chosen amount of
   total experience.
5. `_select_pid_teacher` (from `subproc_vec_base_drone_env.py`) —
   normally called automatically on every worker's episode reset during
   real (non-diagnostic) rollout collection.
6. `app/guidance/export_onnx.py`. It infers the architecture
   (obs_dim/hidden/num_hidden_layers/action_dim) straight from the
   state_dict's tensor shapes, since a bare state_dict doesn't carry those
   hyperparameters — but here it's called on an already-built, in-memory
   model, so `export_onnx_model` (not `export_onnx_checkpoint`) is used,
   skipping that inference step entirely.
7. `build_uniform_omni_eval_pairs` — continuous `Uniform(distance_low,
   distance_high)` magnitude with a random full-sphere direction, unlike
   the fixed lattices (`build_eval_pairs`/`build_omni_eval_pairs`), which
   only cover discrete distance rungs and equal-magnitude axis combos.
8. Whether a good-looking training curve reflects real task competence vs.
   a fluke (comparing against a known-good PID baseline on IDENTICAL
   configs), and whether a "success" is sustained rather than a lucky
   single checkpoint (the streak-window logic in `plotting.py`).

</details>

---

# PART 7 — Block: Tooling

**Files**: `app/guidance/plotting.py`, `mlflow_utils.py`, `utils.py`,
`export_onnx.py`

## 7.1 `plotting.py`

`plot_training_run`: 5 figures from a `metrics.csv` — `training_error.png`
(policy/entropy loss + log-scale value_loss/grad_norm), `policy_std.png`
(exploration decay, log scale — should trend DOWN, not flat-high or
climbing), `distance_distribution.png` (RL mean±std final distance overlaid
with the PID-teacher reference on identical pairs — a flat/low PID line
next to a flat/high RL line means the task is solvable, the gap is the
policy's), `success_and_outcomes.png` (hit-rate + stacked outcome-count
breakdown, with **streak-window shading**: `_streak_windows` finds runs of
≥5 consecutive checkpoints at a hit-rate tier, so a single lucky checkpoint
doesn't read as "converged"), `vs_pid_baseline.png` (bar chart, RL final
checkpoint vs. classical PID on 4 metrics). `plot_dagger_history`: raw
pair-count imbalance + per-distance hit-rate across DAgger rounds.

**Update (2026-09-27).** `plot_grad_norm_3d` is now a **landscape-style plot**:
x = timesteps, y = `grade` (was `lr`, which is a fixed function of time and so
cannot define a surface), z = `grad_norm`; a jet-coloured surface with mesh
lines, contour lines projected on the floor, the run's path as a black line
with a start dot and an arrowhead. **The surface is interpolated** (a
Gaussian-kernel smooth of the logged chunks that relaxes to the mean away from
any logged point); only the path is data — the title says so. Implementation
notes worth knowing: the start marker and arrowhead are `Line3D` artists
because matplotlib depth-sorts patches/collections against each other (a
surface polygon can hide a scatter marker or a patch arrow) whereas lines draw
last; the arrowhead is built in unit-cube coordinates so it looks the same size
despite timesteps ~1e8 vs grade ~1. New helper script
`app/guidance/analyze_optuna_isaac.py` ranks Optuna trials by holding
performance (0.12.C); reading every MLflow graph is 0.14.C.

## 7.2 `mlflow_utils.py`

Thin wrappers: `start_run` (sets experiment then starts a run),
`log_params_safe`/`log_metrics_safe` (stringify/truncate params to survive
MLflow's type restrictions; drop non-numeric/None metrics, cast bools to
0/1 — MLflow itself would raise on bad types otherwise). Uses the default
local `sqlite:///mlflow.db` unless `MLFLOW_TRACKING_URI` is set — this
session's Isaac orchestrator sets that env var to a SEPARATE store
(`mlflow_isaac.db`) specifically to avoid a schema-version conflict against
the numpy venv's older mlflow-created database (Part 8).

## 7.3 `utils.py`

`compute_grade`: single scalar combining success rate (rewarded) minus
normalized penalties for final-distance error, time-to-hit, and gradient
norm — for ranking checkpoints/trials on one number. `calc_drone_state`:
averages the last N `QuadState`s (position/velocity/orientation/rotor_rpm)
— a smoothing utility.

## 7.4 `export_onnx.py`

`infer_architecture`: reads `obs_dim`/`hidden`/`num_hidden_layers`/
`action_dim` straight from a state_dict's tensor shapes (a bare state_dict
carries no architecture metadata). `export_onnx_model`: exports an
already-loaded model (what training loops use, no checkpoint round-trip).
`export_onnx_checkpoint`: loads a `.pt` from disk first, for standalone CLI
use.

## Part 7 practice questions

1. **(own words)** What does the streak-window shading in
   `success_and_outcomes.png` protect you from misreading, in ≤5 sentences?
2. **(trace)** A checkpoint's hit-rate history is `[0.1, 0.3, 0.8, 0.9,
   0.85, 0.95, 0.4, 0.9, 0.9, 0.9, 0.9, 0.9]` (12 checkpoints,
   `STREAK_LEN=5`). Which 75% tier window, if any, gets shaded?
3. **(what breaks)** If `log_metrics_safe` didn't drop non-numeric values
   before calling `mlflow.log_metrics`, what would happen?
4. **(why this over alternative)** Why does `export_onnx_model` (used by
   training loops) skip `infer_architecture` entirely while
   `export_onnx_checkpoint` (standalone CLI) needs it?
5. **(cross-file)** `mlflow_utils.start_run` is called by both
   `base_training.py` and `base_training_isaac.py`. What environment
   variable makes them write to DIFFERENT databases, and why was that
   necessary?
6. **(cross-file)** `plot_training_run`'s `distance_distribution.png` plots
   `pid_avg_final_dist` alongside the RL curve. Which orchestrator function
   computes that number every checkpoint, and against what env?
7. What does `compute_grade`'s `time_ratio` default to when
   `avg_hit_time_sec is None`, and why does that make sense?

<details><summary>Model answers</summary>

1. A single lucky checkpoint hitting a high tier by chance, without the
   policy actually having converged — the shading only appears once a tier
   held for `STREAK_LEN` (5) CONSECUTIVE checkpoints, filtering out noise.
2. The 75% window: checkpoints 8-11 (0.9,0.9,0.9,0.9 — only 4 in a row from
   index 8, need to check index 7 too: values from idx 7 are 0.9,0.9,0.9,
   0.9,0.9 = indices 7-11, 5 consecutive ≥0.75) — shaded window covers
   indices 7 through 11 (the last 5 checkpoints, all ≥0.75, but NOT ≥0.9's
   90% tier since it only shades the HIGHEST tier actually achieved with a
   qualifying streak, and 90% only has a run starting at index 7 too if all
   5 are ≥0.9, which they are — so 90% tier shades instead of 75%).
3. `mlflow.log_metrics` would raise an exception (it rejects None/
   non-numeric values), likely crashing the training loop at a logging
   call — exactly what the drop-and-cast guard prevents.
4. `export_onnx_model` receives an ALREADY-BUILT, already-loaded model
   object directly from the training loop (architecture is already known,
   no state_dict round-trip needed); `export_onnx_checkpoint` only has a
   `.pt` file on disk with no architecture metadata attached, so it must
   reverse-engineer the shapes first.
5. `MLFLOW_TRACKING_URI` — set by `base_training_isaac.py` to
   `sqlite:///mlflow_isaac.db` because installing mlflow into the Isaac
   venv (a newer version) hit a schema-version mismatch against the
   numpy venv's existing `mlflow.db`; using a separate store avoided
   running a schema migration on the numpy side's live experiment history.
6. `base_training.py`'s (or `base_training_isaac.py`'s) `train()`, via
   `diagnose_with_pid(eval_env.pid_teacher, eval_env, ...)` — against the
   plain numpy `BaseDroneEnv` `eval_env`, even in the Isaac orchestrator.
7. `1.0` (the worst/maximum ratio) — if the policy/PID never hit the
   target that episode, there's no time-to-hit to reward, so it should
   count as the WORST case for that term, not be silently ignored (which
   would make "never succeeding" score better than "succeeding slowly").

</details>

---

# PART 8 — Block: Isaac Lab Port

**Files**: `app/environmental/base_drone_env_isaac.py`, plus the
Isaac-native additions inside `app/guidance/train.py`
(`isaac_ppo_train`, `isaac_warmup_critic`, `adaptive_kl_lr_step`),
`app/control/collect_demonstrations.py`
(`collect_demonstrations_base_drone_isaac`), `app/control/dagger.py`
(`dagger_base_drone_isaac`), `app/training/base_training_isaac.py`.
Cross-references Parts 1/3/4/5/6 heavily — this Part focuses on what's
**specifically** different/new about running on Isaac Lab, and the real
bugs found doing it.

## 8.1 `base_drone_env_isaac.py` — the DirectRLEnv port

`BaseDroneEnvIsaac(DirectRLEnv)` registers gym id
`Isaac-Base-Drone-Direct-v0` on import. Mirrors `BaseDroneEnv` exactly in
*intent* but everything is `(num_envs, ...)` tensors, GPU-resident, and
PhysX does the actual rigid-body integration — **we only compute forces/
torques** (`app.dynamics.torch_methods`, Part 1.3), PhysX turns them into
motion. Key hooks (`DirectRLEnv`'s own step-loop order:
`_pre_physics_step` once → `_apply_action` × `decimation` → `_get_dones` →
`_get_rewards` → `_reset_idx` for done envs → `_get_observations`):

- `_pre_physics_step`: clip, hover-offset, `mixer_inversion` — ONCE per
  policy step (the rotor TARGET doesn't change within a decimated
  substep-group).
- `_apply_action`: `motor_lag` — EVERY substep (must integrate at
  `PHYSICS_DT` regardless of `decimation`, 1.2.1's requirement), recompute
  thrust/torque/drag/wind from the freshly-lagged rotor speeds, apply via
  `self._robot.permanent_wrench_composer.set_forces_and_torques(...)`.
- `_get_observations`: matches `build_observation`'s 23-dim layout exactly,
  including reordering Isaac's `(w,x,y,z)` quaternion to the oracle's
  `(x,y,z,w)` (0.2's frame-convention warning, made concrete).
- `_compute_dones_and_reward`: ports `reward_func` exactly, as per-env
  tensors instead of per-env Python attributes; also stashes
  `self.extras["term_reasons"]` (boolean masks: hit/oob/attitude_roll/
  attitude_pitch — batched equivalent of the oracle's joined-string
  `info["reason"]`) and `self.extras["terminal_observation"]` (Part 2.3's
  pattern, ported).
- `set_target_pairs`: added this session — resolves the "target_pairs
  deferred" gap from earlier in the migration; mirrors
  `BaseDroneEnv.reset()`'s pool-sampling semantics exactly. Target direction
  is drawn by `collect_demonstrations.sample_full_sphere_target` (Part 5) --
  a truly uniform-random point on the FULL sphere (`rng.normal(size=3)`
  normalized to a unit vector, the standard unbiased method), not the
  earlier category-anchored jitter sampler this project started with.
  Distance is drawn continuously from `Uniform(distance_low, distance_high)`
  per target (not a discrete pick from a fixed list) -- PID gains are
  assigned per-env by NEAREST-distance match against whatever continuous
  distance the sampled target actually landed at
  (`torch_pid.assign_gains_by_distance`, Part 3), since `best_pid_gains_
  per_dist.json` is only tuned at a handful of discrete distances.

## 8.2 The robot asset: a real, instructive bug chain

**Original placeholder**: `CRAZYFLIE_CFG`, IsaacLab's stock small-drone
`ArticulationCfg` (4 dummy rotor joints), mass/inertia overridden
post-hoc via `root_physx_view.set_masses`/`set_inertias` to match
`QuadConfig`.

**Bug found** (`scripts/isaac_lab/debug_force_isolation.py`): applying a
known force/torque directly via the wrench composer, bypassing all of our
own physics, still under-delivered — a hover-thrust-only test (should net
~0 acceleration against gravity) only recovered ~20% of the expected
correction; torque, ~76%. Isolated by:
1. Cross-checking against IsaacLab's OWN unmodified `Isaac-Quadcopter-
   Direct-v0` example — exact, confirming the wrench-composer API itself
   isn't broken.
2. Testing with the mass override REMOVED (native ~25g Crazyflie mass):
   delivery jumped to ~89%. **The shortfall scaled with how far the mass
   override diverged from the asset's native/cooked value** — pointing at
   a PhysX Articulation-solver quantity cached from the asset's ORIGINAL
   mass at cook time that a post-hoc override doesn't fully invalidate.

**Fix**: stopped fighting a mismatched placeholder — replaced the
Crazyflie `ArticulationCfg` with a plain procedural `RigidObjectCfg`
(`sim_utils.CuboidCfg`, no joints, since nothing in this project ever used
joint dynamics — everything is analytic single-rigid-body force/torque),
sized `(0.4, 0.4, 0.02)` m so its OWN natural uniform-density inertia
already matches `QuadConfig`'s `(0.02,0.02,0.04)` to within ~0.25%
(`I = m/12·(edge_a²+edge_b²)`), mass authored directly at spawn. **No
override of any kind afterward** — nothing to fight. Result: both isolation
tests went to ~100% delivered.

**Why this matters for a defense**: this is the single best "trace a real
debugging story" example in the whole project — a plausible-sounding
hypothesis (wrench-composer API bug) was DISPROVEN by a control experiment
(stock example works fine), and the actual root cause (asset mismatch, not
API bug) was found by varying ONE variable (mass-override magnitude) and
observing a clean proportional relationship.

## 8.3 `isaac_ppo_train` / `isaac_warmup_critic` (in `train.py`)

`isaac_ppo_train`: same algorithm as `vec_ppo_train` (Part 4.4), but drives
a live `DirectRLEnv` directly — torch tensors in/out, no numpy round-trip,
no Python per-env stepping loop (`env.step()` is already GPU-batched).
Truncation bootstrap reads `extras["terminal_observation"]` (8.1's
addition) instead of a Python list of per-env info dicts. `adaptive_kl_lr_
step`: an off-by-default lr schedule (halve/double lr based on measured KL
vs. `target_kl` each round — same scheme as rl_games' AdaptiveScheduler),
hand-written (no RL library), satisfying a "no rsl_rl/skrl" constraint.

**Real bug found**: `BaseDroneEnvIsaacCfg.action_space` was originally a
plain int (`=4`) — `DirectRLEnvCfg` auto-generates an UNBOUNDED
`Box(-inf,inf,(4,))` for a bare int. `ActorCritic`'s tanh-squash rescale
(`action_low + (squashed+1)·0.5·(action_high-action_low)`) then computes
`-inf + finite·inf = NaN` on the very first action, before any `env.step()`
even runs. Fixed with an explicit bounded `Box` matching `BaseDroneEnv`'s
real action bounds. **Why this is a good defense example**: it shows a
failure mode invisible from reading the physics code at all — the bug was
in a CONFIGURATION default's interaction with a downstream consumer that
assumed a property (boundedness) the config silently didn't guarantee.

`isaac_warmup_critic`: batched port of `warmup_critic` (Part 4.5) — but
this session ALSO found that `base_training_isaac.py` was initially calling
this function for its critic-warmup phase, when it should have mirrored
`base_training.py`'s REAL mechanism (frozen-actor PPO chunks, Part 6.1) —
fixed to merge phases 2+3 into one chunk loop, matching exactly.

## 8.4 Two more bugs, found by the user pushing back rather than accepting
the first pass (a real "defend your design choices" example)

1. **Imitation retrain cadence was a fixed per-env step count.** Silently
   shrinks the number of "collect, retrain" ROUNDS as `NUM_ENVS_ISAAC`
   grows (round count, not raw pair volume, is what avoids catastrophic
   forgetting — Part 6.1). Fixed: cadence is now DERIVED from a minimum
   round-count floor (`MIN_IMITATION_RETRAIN_ROUNDS=8`, matching
   `base_training.py`'s own ~7.5 rounds), guaranteeing enough rounds
   regardless of env count.
2. **Critic warmup called the wrong mechanism** — see 8.3's last
   paragraph.

Both were caught not by testing, but by the user asking "is this actually
a good idea" instead of accepting the initial implementation — worth being
able to articulate as a real example of design review catching a
correctness bug that ran, produced plausible-looking output, and would
have silently under-trained at real scale.

## 8.5 Three more bugs, found reviewing a completed real training run (not
testing in isolation this time -- reading `metrics.csv`/plots critically)

All three surfaced from the SAME symptom: `dagger_base_drone_isaac`'s own
printed `hit_rate` collapsed to ~0.00 across every round even starting from
a freshly, cleanly BC-converged checkpoint, and (separately) a full 128M-
timestep `train_isaac.py` run's `success_rate` collapsed to ~0 the moment
the actor unfroze after critic warmup and never recovered for the remaining
~100 chunks.

1. **`env.reset()` called on every round/every chunk, not once.** Both
   `dagger_base_drone_isaac` (top of every DAgger round) and `isaac_ppo_train`
   (top of every call -- and `base_training_isaac.train()` calls it once per
   PPO chunk, ~every 256 env-steps) did a FULL `env.reset()`. `DirectRLEnv`'s
   full-reset path deliberately randomizes every env's `episode_length_buf`
   uniformly across `[0, max_episode_length)` (standard desync-on-reset
   behavior, so parallel envs don't all terminate in lockstep). With a
   rollout window far shorter than `max_episode_length` (DAgger:
   `rows_per_round/num_envs`≈122 steps vs. 3750-step episodes; PPO chunks:
   256 vs. the same 3750), most envs never got a real episode before being
   reset again, and a rollout-window-sized slice of envs landed within
   reach of their (fake, randomly-assigned) ceiling and instantly "timed
   out" having barely acted -- manufacturing a burst of spurious timeouts
   every round/chunk regardless of actual policy quality.
   **Isolated by**: two new standalone diagnostics
   (`scripts/isaac_lab/diagnose_pid_isaac.py`,
   `scripts/isaac_lab/diagnose_model_isaac.py`) that reset ONCE and run many
   episodes straight through -- these showed the PID teacher at 90.6% hit
   rate and the SAME checkpoint `dagger_base_drone_isaac` scored ~0% for at
   85.9%/90.0% (pre/post-DAgger). The checkpoints were fine the whole time;
   the measurement was broken.
   **Fix**: `isaac_ppo_train` gained an `initial_obs` param (`None` = old
   single-call behavior) and now returns `(model, optimizer, episode_rewards,
   last_losses, final_obs)`; `base_training_isaac.train()` threads a
   persistent `obs` across its chunk loop instead of discarding it.
   `dagger_base_drone_isaac`'s reset moved from inside the round loop to
   once before it. **This means the PPO phase (~84% of the real training
   budget) had never actually been run correctly before this was found** --
   caught by reviewing metrics before trusting a checkpoint, not after.
2. **`step_penalty` over-accumulated, making a quick crash reward-preferred
   over floundering.** `reward_func`'s `step_penalty` is calibrated so a full
   non-converging episode costs exactly `TARGET_FRACTION*APPROACH_MILESTONE_
   BUDGET=24.18` total, by construction -- the numpy oracle enforces this by
   truncating each episode at exactly `steps_for_dist(dist)` PHYSICS steps.
   `BaseDroneEnvIsaac` ported the formula literal (divided by a per-env
   `steps_for_dist(dist)` value) but applies it once per POLICY step, and
   -- per 8.6's own already-documented simplification -- has ONE FIXED
   `max_episode_length` (3750 policy steps) for every distance, not a
   per-distance truncation. At dist=3 (`steps_for_dist(3)=1800`,
   `-24.18/1800≈-0.01343/step`), a full 3750-step timeout accumulates
   `3750×0.01343≈50.4` -- roughly equal to `HIT_REWARD=50`, versus a
   one-time crash costing only `-1.0`/`-1.5`. Once a trajectory wasn't
   cleanly converging, the reward function was telling the policy crashing
   immediately was ~35x cheaper than continuing to try -- directly
   explaining why the PPO collapse never recovered on its own.
   **Fix**: `step_penalty` now divides by `self.max_episode_length` (the
   REAL, fixed, actually-enforced cap) instead of the mismatched
   per-distance value -- restores the intended 24.18 ceiling regardless of
   distance. `self._steps_for_dist` (now unused) and its import were
   removed rather than left dangling.
   **Found by the user pushing back** on bug #1 alone being a sufficient
   explanation, citing `rewards.py`'s own `HIT_REWARD` comment describing
   an earlier, structurally identical "collapses right at unfreeze, never
   recovers" failure (fixed then by dropping `HIT_REWARD` from 1000 to 50)
   -- a second real example (after 8.4) of design-review pushback catching
   something a first pass missed.
3. **Critic warmup's `WARMUP_FRACTION=0.01` collapsed to only 2 chunks at
   `num_envs=4096`** (`chunk_grand_steps=NUM_STEPS_PER_CHUNK×num_envs≈1.05M`,
   so `1%×128M=1.28M` grand-total steps ÷ that chunk size ≈ 1.2 → 2 whole
   chunks) -- the SAME fraction-collapses-at-large-`NUM_ENVS` pattern as
   8.4's imitation-retrain-round bug, just for critic warmup instead.
   `metrics.csv` showed `value_loss` still ~14 (started ~27) at the exact
   chunk the actor unfroze -- nowhere near converged, feeding the actor
   garbage advantage estimates the moment it started listening to them.
   **Fix**: `MIN_WARMUP_CHUNKS=8` (same floor-not-fraction pattern as
   `MIN_IMITATION_RETRAIN_ROUNDS`) -- `warmup_timesteps_grand =
   max(WARMUP_FRACTION*total_timesteps, MIN_WARMUP_CHUNKS*chunk_grand_steps)`.

All three fixes are independently justified and likely compound: bug #3
gives the actor a bad initial push right at unfreeze; bug #2 is what
prevented recovery afterward by actively rewarding staying crashed; bug #1
is what made the DAgger-stage symptom (a completely healthy checkpoint
reading as ~0% hit rate) look like a policy-quality problem when it never
was. None of the three are yet re-verified against a fresh full training
run at the time of writing.

## 8.6 Budget/config differences from the numpy version (know these cold)

- `decimation=4` (raised from 1 once the diff-test passed at decimation=1 —
  0.6's speed-knob concept, applied): 240/4 = 60Hz effective control rate.
- `TOTAL_TIMESTEPS_ISAAC` is a GRAND TOTAL (across all envs), unlike
  numpy's per-env constant — see 6.2/8.4.1.
- `NUM_ENVS_ISAAC`: not yet empirically benchmarked on the actual hardware;
  estimated comfortably 8192+, since this scene (single free rigid body, no
  joints, no sensors) is lighter than IsaacLab's own Cartpole example
  (commonly 4096-8192 envs on 8-12GB cards).
- Target sampling defaults to Uniform(3,10) placeholder unless
  `set_target_pairs` is called (8.1) — distances beyond ~50-60m would need
  a per-env episode-length fix first (the env has ONE fixed
  `max_episode_length` for every env, unlike the numpy oracle's
  per-distance `max_steps`).

## 8.7 Soft periodic reset toward best checkpoint (2026-09-22)

**Motivation**: every real training run observed so far (Optuna hparams,
confirmed genuine baseline hparams, old and new reward function alike)
degrades monotonically the further PPO runs past the end of imitation —
the imitation-phase checkpoint is consistently the peak of all results (see
8.5's analysis for why: no anchor back to the BC policy, a critic that's
never fully trustworthy, and an asymmetric risk where a near-optimal BC
starting point has little to gain and a lot to lose from noisy PPO
gradients). Rather than a KL-penalty term added to the PPO loss (which
would require reworking `isaac_ppo_train`'s loss function itself and a new
frozen reference-policy forward pass every update), this is a lighter,
outside-the-loss mechanism: periodically check whether the policy has
degraded, and if so, pull it back toward the best weights seen so far.

**Mechanism** (`base_training_isaac.py`, chunk loop, active only once the
actor is unfrozen — `stage=="training"`):
```
every chunk:
    if grade > best_grade: best_grade, best_ckpt_path = grade, this_chunk's_saved_ckpt
    grade_history.append(grade)
    if len(grade_history) >= RESET_SUSTAIN_CHUNKS=5:
        trailing_avg = mean(grade_history[-5:])
        if best_grade - trailing_avg >= RESET_DEGRADE_MARGIN=0.15:
            # ONE-TIME soft blend, not a repeated pull every chunk
            weights = RESET_BLEND_ALPHA=0.5 * best_ckpt_weights + 0.5 * current_weights
            reinitialize optimizer (AdamW) -- stale Adam momentum from the
            drifted region is meaningless once the weights themselves jump
            grade_history = []  # cooldown: need a fresh 5-chunk window before firing again
```
`best_ckpt_path` reuses the checkpoint the loop already saves every chunk
(`model_{timesteps}.pt`) — no extra weights held in memory. `reset_triggered`
and `best_grade` are logged to `metrics.csv`/mlflow every chunk.

**Why blend (`alpha=0.5`) instead of a hard reset (`alpha=1.0`)**: a hard
reset would discard 100% of whatever PPO learned since `best_ckpt_path`,
including anything genuinely better that just hasn't shown up in the grade
yet (grade is a noisy per-chunk diagnostic over only `N_DIAGNOSTIC_EPISODES`
rollouts). A blend keeps half the drifted weights, giving PPO a chance to
recover on its own before the NEXT window; a hard reset would make repeated
firings look identical to simply never leaving `best_ckpt_path`, defeating
the point of running PPO past imitation at all (8.5's own stated goal:
PPO exists to let the policy exceed what BC/PID could do, not just match it).

**Why a 5-chunk trailing AVERAGE, not a single bad chunk**: grade is noisy
chunk-to-chunk (same reasoning as `_print_suitable_weights`'s
`sustain_window` in `plotting.py`, which uses the same 5-chunk pattern for
the same reason) — resetting on one unlucky diagnostic batch would fire
constantly on noise instead of on real, sustained degradation.

**Cooldown**: clearing `grade_history` after a reset prevents the SAME
degraded window from re-triggering the very next chunk (the history would
otherwise still read as degraded from the same decline). It also means a
big single-chunk grade crash doesn't stack multiple resets back to back.

**Known limitation, stated plainly**: this treats a symptom (degrading
grade) rather than a cause (unbounded policy drift, unreliable critic
estimates). It doesn't stop PPO from drifting between checks, and a KL-to-
BC-policy loss term (discussed but not yet implemented) would prevent the
drift continuously rather than correcting it after the fact every 5 chunks.
Not yet verified against a fresh full training run at the time of writing.

## Part 8 practice questions

1. **(own words)** Explain, in ≤5 sentences, the FULL bug chain that led
   from "Articulation placeholder" to "procedural RigidObject" — what was
   observed, what was ruled out, what was the actual cause.
2. **(trace)** `debug_force_isolation.py` applies exactly `mass·9.81` N of
   thrust and zero torque to a stationary drone. What should
   `root_lin_vel_w` be after one step, and what did it actually read
   BEFORE the fix (roughly, as a fraction of expected)?
3. **(what breaks)** If `BaseDroneEnvIsaacCfg.action_space` were reverted
   to a plain int `4`, what's the FIRST place in the training pipeline
   that would fail, and with what error?
4. **(why this over alternative)** Why does the RigidObject fix use a
   THIN FLAT cuboid `(0.4, 0.4, 0.02)` specifically, rather than some other
   box shape?
5. **(cross-file)** `isaac_ppo_train`'s truncation bootstrap needs
   `extras["terminal_observation"]`. Which specific method in
   `base_drone_env_isaac.py` populates it, and why THAT method specifically
   (not `_get_observations`, not `_reset_idx`)?
6. **(cross-file)** The imitation-retrain-round-floor fix and the
   critic-warmup-mechanism fix both live in the same file. Name it, and
   explain why fixing the second one required also changing where
   `isaac_warmup_critic` is imported from.
7. Why was `decimation` kept at 1 during the diff-test phase, and only
   raised to 4 afterward — what would raising it earlier have cost you?
8. What's the practical difference between "8192 envs is likely fine" (this
   guide's current claim) and an actual verified answer — what would you
   need to run to turn the estimate into a fact?
9. **(trace a scenario)** `dagger_base_drone_isaac` resets the whole env at
   the top of every round, with `rows_per_round=500,000` and `num_envs=4096`.
   Roughly how many POLICY steps does that give each env before the round
   ends, and why does that make a large fraction of "episodes" completing
   within the round pure measurement artifacts rather than real outcomes?
10. **(why this over alternative)** Why does fixing bug 8.5.1 require
    `isaac_ppo_train` to RETURN its final `obs` (not just accept one), when
    the earlier fix to `dagger_base_drone_isaac` didn't need an equivalent
    return value?
11. **(math)** At `dist=3`, `steps_for_dist(3)=1800` and
    `TARGET_FRACTION*APPROACH_MILESTONE_BUDGET=24.18`. Compute
    `step_penalty` per step under the ORIGINAL (buggy) formula, and the
    worst-case total over a full `max_episode_length=3750`-step timeout.
    Compare that total to `HIT_REWARD=50` and to a one-time crash penalty of
    `-1.5` — what does the comparison tell you about the policy's incentive
    once a trajectory stops converging cleanly?
12. **(cross-file)** Name the THREE places in this codebase that now use the
    "floor a minimum count instead of a raw fraction" pattern (two from 8.4,
    one from 8.5.3). What's the one sentence explaining why a raw fraction
    is unsafe in all three, despite them protecting different things
    (imitation retrain rounds, critic warmup chunks)?
13. **(trace a scenario, 8.7)** `best_grade=0.80`. Chunks 41-45 log grades
    `0.68, 0.64, 0.70, 0.66, 0.62`. Does a soft reset fire after chunk 45?
    Show the trailing average and compare it to `RESET_DEGRADE_MARGIN`. If
    it fires, what happens to `grade_history` right after, and why does
    that matter for chunk 46?
14. **(why this over alternative, 8.7)** Why does 8.7 reinitialize the
    optimizer instead of just blending the model weights and leaving Adam's
    momentum buffers untouched?

<details><summary>Model answers</summary>

1. A hover-thrust-only isolation test (should net ~0 acceleration) only
   recovered ~20% of expected force on the Articulation placeholder.
   Cross-checking against IsaacLab's own unmodified example ruled out the
   wrench-composer API itself. Removing the mass override (using the
   asset's native mass) raised delivery to ~89%, showing delivery scaled
   inversely with how far the override diverged from native — pointing at
   a PhysX Articulation-solver quantity cached at cook time from the
   original mass. Switching to a procedural RigidObject with NATIVE mass/
   inertia (no override needed) fixed it to ~100%.
2. Expected: ~0 m/s (thrust exactly cancels gravity). Before the fix: only
   about 20% of the needed correction was actually applied, leaving a
   large residual downward-canceling-but-not-quite velocity (specifically,
   using the numbers from this session: roughly a -0.033 m/s residual
   against a near-zero expectation).
3. `ActorCritic.__init__`/`get_action_and_value`, the very first action
   computed before `env.step()` ever runs — `action_low`/`action_high`
   would be `-inf`/`inf`, and the tanh-squash rescale produces NaN
   immediately (`-inf + finite·inf`).
4. Its uniform-density inertia, computed from `I=m/12·(edge_a²+edge_b²)`,
   ALREADY matches `QuadConfig`'s target ratio (`Ixx=Iyy=0.02, Izz=0.04`,
   i.e. `Izz=2·Ixx`) almost exactly for a flat square shape — no separate
   inertia override is needed on top of the mass override, since the
   geometry itself produces the right numbers.
5. `_get_rewards` — because the DirectRLEnv step order is `_get_dones` →
   `_get_rewards` → `_reset_idx` → `_get_observations`; `_get_rewards` is
   the LAST hook that runs before a just-terminated env's `_reset_idx`
   overwrites its state, so it's the only correct place to snapshot the
   true pre-reset observation.
6. `app/training/base_training_isaac.py`. Once critic warmup was changed
   to reuse `isaac_ppo_train` (the same function phase 3 uses) instead of
   calling the separate `isaac_warmup_critic`, the `isaac_warmup_critic`
   import became unused and was removed from that file's import list
   (the function itself still exists in `train.py` as a separate tool).
7. So the trajectory diff-test (Part 1's oracle comparison) would be a
   strict 1:1 comparison — one Isaac policy step advancing exactly one
   physics tick, directly comparable to one oracle `timestamp_update` call.
   Raising decimation earlier would have meant each Isaac step covered
   MULTIPLE physics ticks while the oracle comparison script only advanced
   one, making the diff-test invalid without first modifying it to loop
   the oracle call `decimation` times (which was in fact needed and done
   once decimation was actually raised).
8. You'd need to actually run training at increasing `--num_envs` (e.g.
   1024/2048/4096/8192/16384) and watch VRAM usage and wall-clock
   steps/second directly on the RTX 5060 Ti, rather than reasoning by
   analogy to IsaacLab's Cartpole example — the estimate is a reasoned
   guess, not a measurement.
9. `500,000/4096≈122` policy steps per round. `max_episode_length=3750`, so
   no env can naturally reach a hit or a genuine timeout within 122 steps
   unless it crashes almost immediately. Any env whose FULL-RESET-randomized
   `episode_length_buf` happened to land within ~122 steps of the ceiling
   (~3.3% chance per env, ≈133 of 4096 — matching what was actually
   observed) will hit `time_out` almost immediately regardless of how it's
   actually flying, which is exactly what gets counted as a completed
   "episode" in that round's `hit_rate` denominator.
10. `isaac_ppo_train` is called REPEATEDLY, once per PPO chunk, by
    `base_training_isaac.train()`'s own chunk loop, and needs to hand the
    CURRENT env state to the NEXT call so that call can skip its reset —
    the caller (train()) owns the loop and must be able to pass what the
    previous call left off with. `dagger_base_drone_isaac`'s fix only needed
    a SINGLE reset moved from inside its own round loop to before it — the
    function owns its whole loop internally, so there's no cross-call
    boundary that needs an explicit obs hand-off.
11. `step_penalty = -24.18/1800 ≈ -0.01343` per step. Worst case:
    `3750 × 0.01343 ≈ 50.4` total — almost exactly equal to `HIT_REWARD=50`,
    and about 33x a one-time crash's `-1.5`. Once a trajectory isn't
    cleanly converging, the reward function is telling the policy that
    ending the episode via a quick crash is far cheaper than continuing to
    try — crashing becomes locally reward-maximizing, not a mistake the
    policy is making.
12. `MIN_IMITATION_RETRAIN_ROUNDS` (imitation retrain cadence, 8.4.1),
    `MIN_WARMUP_CHUNKS` (critic warmup duration, 8.5.3), and the
    `rows_per_round`/round-boundary structure in `dagger_base_drone_isaac`
    that bug 8.5.1's fix implicitly depends on staying meaningful relative
    to `max_episode_length`. One sentence: a raw fraction of a GRAND-TOTAL
    step budget divides down to a raw step (or chunk) COUNT that shrinks as
    `NUM_ENVS_ISAAC` grows, and below some threshold count the mechanism it
    was meant to guarantee (enough retrain rounds, enough critic updates,
    enough real episode time) silently stops happening at all — a floor on
    the COUNT itself is the only thing that's invariant to `NUM_ENVS_ISAAC`.
13. Trailing average = `(0.68+0.64+0.70+0.66+0.62)/5 = 0.66`. Gap =
    `best_grade - trailing_avg = 0.80-0.66 = 0.14`, which is LESS than
    `RESET_DEGRADE_MARGIN=0.15` — no reset fires (a deliberately close call:
    the mechanism only trips once the sustained gap clears the threshold,
    not merely gets close to it). Since it doesn't fire, `grade_history`
    is NOT cleared — chunk 46's grade gets appended on top of these five,
    and the next check uses chunks 42-46 (still a rolling 5-chunk window,
    no cooldown in effect).
14. Adam's momentum/variance buffers (`exp_avg`, `exp_avg_sq`) were
    accumulated from gradients computed AT the drifted (pre-blend) weights.
    After the blend jumps the weights to a different point in parameter
    space, those buffers describe a gradient trajectory that no longer
    corresponds to where the model now sits — reusing them would apply
    updates sized/directed for the wrong region, likely re-injecting the
    same kind of instability (recall `MIN_WARMUP_CHUNKS`, 8.5.3: this
    codebase already has direct evidence that stale/mismatched optimizer
    state right after a weight discontinuity causes a grad_norm spike).
    Starting AdamW fresh at the blended weights costs only a few chunks of
    slightly less-informed step sizing, which is cheap next to that risk.

</details>

---

# PART 9 — Standalone / not wired into the main pipeline

**File**: `app/navigation/kalmans.py`

A standard **linear Kalman filter** over a 6-element state
`[x,y,z,vx,vy,vz]`, constant-velocity motion model `F`, no control input
(`B`/`u` unused — "RL actions aren't known accelerations here"). Standard
predict/update cycle:
```
predict:  x_hat_est = F @ x_hat;  P_est = F@P@Fᵀ + Q
update:   K = P_est@Hᵀ@inv(H@P_est@Hᵀ + R)
          x_hat = x_hat_est + K@(z - H@x_hat_est)
          P = (I - K@H)@P_est
```
`Q`/`R` are process/measurement noise (scalars expanded to `q·I`/`r·I`);
`H` maps state to the 3 directly-measured position components (velocity is
inferred, not measured). **This module is NOT called anywhere else in the
project** — it's a standalone utility/exercise, not part of the training or
environment pipeline. If asked "where is this used" in a defense, the
honest answer is "it isn't currently integrated."

## Part 9 practice questions

1. Why does `Kalman` have no `B`/`u` (control input) term, given the
   drone clearly IS being controlled?
2. What would `K` (the Kalman gain) look like in the limit `R → 0`
   (perfect sensor) — does the filter trust the prediction or the
   measurement more?

<details><summary>Model answers</summary>

1. Because this filter estimates state from noisy MEASUREMENTS assuming a
   constant-velocity model, independent of what's driving the underlying
   motion — the actual RL policy's action isn't a known/measured
   acceleration this filter has access to, so there's no control-input term
   to include.
2. As `R→0`, `K→H⁻¹`-like behavior (trusts the measurement almost
   entirely) — the update essentially replaces the prediction with the
   measurement, since a perfect sensor should dominate an uncertain model
   prediction.

</details>

---

# PART 10 — Cross-file data flow ("trace it end to end")

These are the hardest, most realistic defense questions — a real panel
asks "and then what happens to that number", not definitions.

## 10.1 One policy step, full pipeline (numpy path)

`BaseDroneEnv.step(action)` →clip→ add hover offset →
`methods.timestamp_update` (mixer_inversion → motor_lag → net thrust/
torque → drag/wind/gravity → semi-implicit Euler + quaternion exp-map) →
`build_observation` → `reward_func(env)` (terminal checks → hit check →
shaping+step_penalty+milestones) → truncate check → return.

## 10.2 One policy step, full pipeline (Isaac path)

`env.step(action)` → `BaseDroneEnvIsaac._pre_physics_step` (clip, hover
offset, `torch_methods.mixer_inversion`) → `decimation`×
`_apply_action` (`torch_methods.motor_lag`, thrust/torque, drag/wind
rotated body↔world via `isaaclab.utils.math`, `permanent_wrench_composer.
set_forces_and_torques`) → PhysX integrates → `_get_dones`
(`_compute_dones_and_reward`: reward_func ported to tensors) →
`_get_rewards` (returns cached reward, stashes `terminal_observation`) →
`_reset_idx` for done envs → `_get_observations` (23-dim, quat reordered).

## 10.3 Demo collection → BC → DAgger → PPO, one thread

1. `tune_pid.py` derives gains analytically → `best_pid_gains_per_dist.json`.
2. `verify_pid.py` sanity-checks those gains hit reliably across the full
   distance ladder.
3. `collect_demonstrations_omni` (or `..._base_drone_isaac`) uses those
   gains as a teacher to produce `(obs, pid_action)` pairs →
   `demonstrations_omni.npz`.
4. `pretrain_bc.py:pretrain_behavior_cloning` regresses an `ActorCritic`
   onto those pairs → `pretrained_bc.pt`.
5. `dagger.py` (or `..._base_drone_isaac`) loads that checkpoint, drives
   the env WITH the policy, labels with PID, aggregates+retrains for
   `n_rounds` → `pretrained_bc_dagger.pt`.
6. `base_training.py` (or `..._isaac`) loads THAT checkpoint via
   `load_bc_checkpoint`, runs imitation→warmup→PPO.

## Part 10 practice questions

1. A drone is hovering (`rotor_rpm` all at hover target), then the policy
   commands `[thrust=+2, roll=0.1, pitch=0, yaw=0]`. Trace this through
   EVERY function call in `timestamp_update`, in order, naming each one,
   until you reach the new `QuadState`.
2. Same scenario, but on the Isaac path — name every function/method
   called between `env.step(action)` and the updated `root_lin_vel_w`,
   including which ones happen once vs. once-per-substep.
3. A DAgger round just finished; the aggregate buffer is now over its
   500,000-pair cap. Trace exactly what happens to the data, in order,
   including which numpy/torch calls are involved and what determines
   which pairs survive.
4. `pretrained_bc_dagger.pt` is about to be loaded into
   `base_training.py`'s `train()`. Trace what `load_bc_checkpoint` actually
   does to the state_dict keys, and explain the ONE scenario where a plain
   `model.load_state_dict(torch.load(path))` would have silently failed
   here instead.
5. A checkpoint just finished a PPO chunk in `base_training_isaac.py`.
   Trace everything that happens between that chunk's `isaac_ppo_train`
   call returning and the NEXT chunk starting — name every function called
   and what each one is checking/writing.

*(No model answers provided for Part 10 — these require synthesizing
material across multiple Parts. If you can answer all 5 confidently without
opening the code, you're ready.)*

---

# PART 11 — Scoring rubric (self-assessment)

Per question: **0** wrong/blank · **1** vague/partially right · **2**
correct "what" only · **3** correct "what" + "why" + downstream
consequences. If you open the code to correct yourself mid-answer, cap that
question at **2** regardless of how right the final answer was — closed-book
first, always.

**Core technologies checklist** — rate yourself 0-3 on each before
considering yourself defense-ready:

- [ ] Rigid-body dynamics: torque/inertia, quaternion composition
      (body-frame right-composition rule)
- [ ] Mixer matrix construction + inversion, motor lag as a first-order ODE
- [ ] PID cascade structure + anti-windup + pole-placement gain derivation
- [ ] `gymnasium.Env` API, termination vs. truncation, auto-reset +
      `terminal_observation`
- [ ] Potential-based reward shaping (and why this project deviates from
      the strict theorem)
- [ ] Shared-trunk `ActorCritic`, tanh-squash + log-prob correction
- [ ] GAE derivation and why it generalizes across single/vectorized envs
      for free
- [ ] PPO's clipped objective + `target_kl` early stop + why both exist
- [ ] Behavior cloning as Gaussian NLL regression in raw (pre-tanh) space
- [ ] DAgger's aggregation + recency weighting, and WHY (covariate shift)
- [ ] The imitation → critic-warmup → PPO phase order and WHY that order
- [ ] Isaac Lab: RigidObject vs. Articulation, decimation, num_envs
      batching
- [ ] The Articulation mass-override bug chain (Part 8.2) — a real,
      nameable debugging story
- [ ] Every place this project's Isaac port differs numerically from the
      numpy oracle, and why each difference is either provably harmless
      (diff-tested) or an explicitly flagged, unresolved gap

If any box is below a 2, go reread that Part's file walkthrough before
your next self-quiz pass.

---

# PART 12 — Case study: what we actually did, in order (2026-09-19 → 2026-09-27)

Every entry is **symptom → evidence → diagnosis → change → status**. The point
of reading this is not the individual fixes but the *method*: each one was
found by checking a number, and several first explanations were wrong.

## 12.1 The one-paragraph summary
The goal was a 3-10 m controller that stays good, so the curriculum can move to
3-30 m and later 3 km. Direct-policy PPO fine-tuned from a PID-imitating
network reached ~0.95 grade and then **slid to 0.4-0.7 in every run** (v1-v4).
Along the way we found and fixed a genuine bug (the BC/DAgger NLL collapsed
`actor_log_std` to its floor, twice), fixed schedule bugs (entropy and LR were
decaying during a warmup where they cannot act), showed that the Optuna
"best trial" was luck, ruled out several causes of the slide, and finally
changed the *architecture of the problem*: the network now learns a small
correction on top of the PID (residual mode). That run (`res_v1`) held its
grade and passed a new promotion gate at chunk 44. It did not beat the PID —
on 3-10 m the PID is already at the ceiling — so the payoff is expected on
3-30 m.

## 12.2 Entries

**1. Chunk arithmetic (2026-09-24).** *Symptom:* raising `--num_envs` to 16384
gave only ~30 chunks. *Evidence:* `chunk_grand_steps = NUM_STEPS_PER_CHUNK ×
num_envs` (256 × 16384 = 4,194,304); `n_chunks = total / chunk`; and
`MIN_WARMUP_CHUNKS = 16` is a fixed *count*, so warmup alone was over half the
run. *Change:* `--num-steps-per-chunk` (128 → 2,097,152/chunk, ~61 chunks).
*Side traps found on the way:* a duplicate `--num-envs` flag aliasing
`--num_envs` (argparse maps both to one dest), a dead `num_envs` parameter that
`train()` immediately overwrote with `unwrapped.num_envs`, and a shell command
whose missing trailing `\` silently dropped the last flag. *Lesson:* floors
expressed in chunk counts must be re-derived whenever chunk size changes.

**2. PID gain ladder (2026-09-25).** Added `20` and `30` to
`tune_pid.DISTANCES` and regenerated the gains (3.3 update, 0.3.C).

**3. Diagnostics, resets and warmup (2026-09-24).** `N_DIAGNOSTIC_EPISODES`
10 → 30 (10 false-triggered resets); reset margin 0.15 → 0.25 (later 0.4);
`RESET_WARMUP_CHUNKS = 2` (a reset also drags the critic back, so it needs its
own warmup); `RUN_START_FROZEN_CHUNKS = 3` rollout-only chunks via a new
`skip_update` flag on `isaac_ppo_train` (freezing *every* parameter breaks
`backward()`, 0.10.B); a shared `runs/isaac_training_metrics.json`;
`IMITATION_FRACTION` 0.15 → 0.08.

**4. The frozen exploration std (2026-09-24/25) — the real bug.**
*Symptom:* in the retrained "fix" run the grade peaked (0.929) but *only during
critic warmup*, then degraded; `effective_std_mean` was `0.049787` (= `exp(−3)`
exactly) on **every** row and `entropy_loss` was frozen at `6.3242`.
*Diagnosis:* `pretrain_bc.py` scores `actor_log_std` with Gaussian NLL against a
deterministic PID, whose optimum is `σ → 0` (0.5.B); `load_bc_checkpoint`
loaded the collapsed value; PPO could then neither move it nor explore.
*First attempt* (reset right after loading the checkpoint) **did not work** —
the run's own on-policy imitation stage runs the same NLL and re-collapsed it
before the warmup freeze. *Fix that worked:* reset to 0 *after* the imitation
stage, just before the freeze (`σ` starts at the clamp midpoint and moves
freely). Full write-up: 5.1a. *Status:* fixed (first row now `σ = 0.0837` →
later `0.0707` with `log_std_max = −2.3`).

**5. Optuna (2026-09-25).** *Symptom:* trial 162 reported best (0.955) but its
curve looked odd. *Evidence:* 0/100 trials held their grade after unfreeze;
`analyze_optuna_isaac.py`; details in 0.12.C. *Change:* the search objective now
mirrors production (imports the same constants/schedules); a robust ranking
script; trial 137's hyperparameters ported with `lr` halved and
`ent_coef_end_frac` converted to an absolute value. *Also:* a soft-deleted
mlflow experiment made 100 trials fail instantly; the experiment's rows were
purged (backup kept) — TPE ignores failed trials, so nothing was lost.

**6. Schedules (2026-09-26).** *Symptom:* `ent_coef` sat at 0.01664 for the
whole training stage. *Diagnosis:* the schedule counted warmup chunks in its
progress and, with `end/start = 0.83`, hit its floor ~10 chunks into a warmup
where the actor is frozen. *Change:* `entropy_coef_at` (held through warmup, then
linear over the PPO stage). LR was made flat through warmup; that shifted the
training-stage LR up (mid-run +45%) and v2 drifted faster, so `scheduled_lr`
keeps the flat warmup but returns to the original full-span cosine afterwards.

**7. The direct-policy runs v1-v4 (2026-09-24 → 26).**

| run | config change | training mean grade | shape |
|---|---|---|---|
| v1 | std fix, reset margin 0.25/α 0.5 | 0.765 | 0.90 → 0.7; 4 resets |
| v2 | flat-warmup LR, entropy schedule, margin 0.4/α 0.75 | 0.723 | first 11 chunks 0.92, then 0.65; 2 resets |
| v3 | `log_std_max −2.3`, `ent 0.01→0.002` | slide to 0.44 by chunk ~35 | std stayed 0.071 → 0.072 |
| v4 | same as v3 (turned out `residual_scale = 0`) | 0.96 → 0.5 by 77 M | slide repeats |

*What was ruled out:* the frozen std (fixed), std/entropy creep (v3), the LR
profile (v1↔v2 differ, both slide). *What was checked:* PPO's own training
reward also falls late (v2: 66 → 48), so it is not just a grade-vs-reward
mismatch; `grad_norm` jumps 0.13-0.15 → 2.1-2.4 at unfreeze in all three
runs; `value_loss` vs grade correlates +0.67/+0.84/+0.92 (time-confounded).
*Hypotheses not yet separated:* (a) noise-driven random walk via normalized
advantages + Adam (0.13.B), (b) critic gradients reshaping the shared trunk,
(c) a numpy-oracle-vs-Isaac gap (v4's `avg_hit_time` fell while success fell —
"rushing" — with the eval running at 240 Hz vs 60 Hz training control).
Test (c) with `scripts/isaac_lab/diagnose_model_isaac.py` (not run yet).

**8. Promotion gate (2026-09-26).** Built after the goal was clarified as "move
to 3-30 m only when 3-10 m is solid (≥ 0.8) at chunk ≥ 40" (0.11.C). Replay on
the saved runs: v1's best qualifying window averaged 0.754, v2's 0.638 —
neither would have fired.

**9. Residual mode (2026-09-26/27).** `--residual-scale S` wraps the env
(`PidResidualEnv`): `action = pid + S × policy`, head zeroed, imitation skipped,
eval composed identically (0.13). Tested first in isolation: a smoke test in
real Isaac (64 envs) — raw zero-action reward −3.1 vs 86.8 wrapped with zero
correction (98% hit) vs 88.1 with a ±30% random correction — and then an
end-to-end 20-chunk run through the whole `train()` loop (no errors; the gate
correctly reported "cannot fire"). `--detach-critic` was implemented alongside.
*Result `res_v1` (3-10 m, scale 0.1, 45 chunks, 94 M steps):* grade 0.971
(warmup, pure PID) → 0.927 (training), 26/29 chunks ≥ 0.9, min 0.841, 0 resets,
gate fired at chunk 44 (window 0.880-0.951, mean 0.919), promoting
`model_92274688.pt`. Success 0.987 → 0.983, hit time 3.70 → 3.64 s, final
distance 0.258 → 0.260 — held, not improved. *Design flaw found on review and
fixed:* residual mode zeroed the actor head after loading *any* checkpoint,
which would have erased the correction learned at 3-10 m when seeding 3-30 m;
`--residual-resume` keeps the loaded head.

**10. Tooling.** Landscape-style `plot_grad_norm_3d` (7.1 update),
`analyze_optuna_isaac.py`, and the MLflow reading guide (0.14.C).

**11. `record_run.py`/`serve_run.py` had no idea residual mode exists
(2026-09-27).** *Symptom:* every checkpoint from `3_10_res_v1`, `3_30_v1` and
`3_50_v1` (all residual, confirmed via mlflow `residual_scale=0.1`) scored
0/75 in the "Run Full Evaluation" tool — every episode timed out. *Diagnosis:*
`run_episode` fed the model's raw output straight to the drone
(`action = model.scale_action(mean)`); a residual checkpoint's actor head only
ever learned a small correction (zero-initialized, ×0.1 at training time), so
alone it means "hover, don't steer" — confirmed by the numpy env sharing the
same delta-from-hover action convention as Isaac (0.1.C). Nothing was wrong
with the checkpoints. *Fix:* `run_episode` (and `record_checkpoint`/
`evaluate_checkpoint`/`rank_checkpoints`, the CLI, and the web API/UI) gained
a `residual_scale`/`gains_by_dist` path that composes
`pid_action + residual_scale·model_action` exactly like `PidResidualEnv`,
re-selecting PID gains by distance and resetting integrators each episode
(reusing `_select_pid_teacher`). The web UI also gained a same-page hint that
fires when the selected checkpoint's path looks like a residual run and the
scale field is still 0. *Status:* implemented, syntax-checked; not yet run
against a real checkpoint (this tool's deps live in the WSL-native `.venv`,
unreachable from the Windows-side session that made the edit) — verify with
`evaluate_checkpoint(..., residual_scale=0.1)` before trusting a "0/75" or
"75/75" reading from this tool again. *Lesson, same shape as entry 6:* a
second, independent code path (an eval/demo tool, not the training loop)
silently assumed the OLD architecture; whenever the action interface changes,
every consumer of a checkpoint has to be checked, not just the training loop.

## 12.3 Things that were wrong along the way, and how they were caught
1. **A false "LR spike" alarm.** A diagnostic print truncated values to 7
   characters and cut the exponent (`1.3e-05` → `1.3`). Caught by re-reading the
   raw column. *Rule:* never truncate numbers you are about to interpret.
2. **"The fix worked" declared before checking.** The first std-reset run still
   showed a flat `σ` from the very first (frozen-actor) row — impossible if the
   reset had held — first suspected stale code, then traced to the imitation
   stage re-collapsing it (entry 4).
3. **A misleading magnitude.** The critic term in the loss was ~1,450× the policy
   loss; that is a *loss value*, not a *gradient* (advantages are zero-mean, so
   the policy loss value is ≈ 0 by construction). Discarded.
4. **"Saturated tanh has no gradient."** Imprecise with Adam (step ≈ lr
   regardless of gradient size); the accurate statement is about *distance*
   (0.10.B, 5.1a note).
5. **Reward claims.** "PID reward ≈ 87 vs the clone's 62-67" came from a biased
   sample (only episodes that finished inside a short window). In runs, the
   pure-PID warmup `recent_reward` is ≈ 68 vs ≈ 62-67 for the clone — nearly
   equal — and the predicted 85-90 was wrong. The residual mode's advantage is
   the hard bound on drift, not a better start.
6. **A run that wasn't the run being discussed.** v4 was assumed to be residual
   mode; mlflow's logged params (`residual_scale = 0.0`) and its first timestep
   (12.3 M instead of 2.1 M — the imitation stage ran) showed it wasn't.
   *Rule:* verify a run's identity from its logged params before analysing it.
7. **Unclamped cosine progress** (a periodic function) briefly made the LR
   oscillate; progress is now clamped (`training_progress_at`).
8. **Matplotlib depth sorting** hid the plot's arrow and start marker behind the
   surface; both are drawn as lines.

## 12.4 Numbers to know cold
| Quantity | Value |
|---|---|
| Plant | 1.5 kg, I = (0.02, 0.02, 0.04), arm 0.22 m, `dt = 1/240`, `decimation = 4` (60 Hz) |
| Network | 23 → 4×64 tanh MLP, 14,345 params, `σ ∈ [0.0498, 0.1003]`, start 0.0707 |
| Production shape | 16,384 envs × 128 steps = 2,097,152 / chunk; ~61 chunks; minibatch 32,768 |
| Phases | run-start 3 chunks rollout-only; warmup 16 chunks; PPO after; gate at chunk ≥ 44 |
| PPO | γ 0.97, λ 0.9066, clip 0.2048, target_kl 0.0086, 10 epochs × 64 minibatches, `vf_coef` 0.2087, clip-norm 1.347 |
| Optimizer | AdamW, lr 1.451e-4 (flat warmup → cosine to 0.1621×), wd 3.8e-6 |
| Entropy | 0.01 → 0.002 (over the PPO stage only) |
| Reset (safety net) | margin 0.4 below best, blend 0.75, 2 critic-only chunks after |
| PID (3-10 m) | success ≈ 0.98, final dist ≈ 0.26 m, hit time ≈ 3.7 s (220 policy steps) |
| Hit threshold | 0.25 m (so `avg_final_dist` floor ≈ 0.25); PID designed to 0.05 m |
| Grade | `success − 0.3·err − 0.1·grad_ratio`; ceiling ≈ 0.96-0.99 |
| Eval noise | 30 episodes → ±0.05-0.09 per chunk; 12-chunk means resolve ≈ 0.04 |
| `res_v1` | grade 0.971 → 0.927; 26/29 chunks ≥ 0.9; gate at chunk 44; checkpoint `model_92274688.pt` |

## 12.5 Open items and what to do next
- **Superseded in part (2026-10-05):** the residual-mode route above was replaced by a full-network RTL pipeline with a
  two-phase handoff; see Part 14 for what was built and measured, and `alt_model.md` for the proposed next version.
- **3-30 m stage** with `--residual-scale 0.1 --residual-resume --bc-checkpoint
  <SOLID.json checkpoint>`; per-chunk eval will be slower (30 m episodes are
  31 s long).
- **3 m-3 km:** extend the PID gain ladder past 250 m; re-check saturation
  behaviour (0.3.B) at those ranges.
- **Unresolved:** the cause of the direct-policy slide (hypotheses a-c above).
  Cheapest decisive test: evaluate a late v4 checkpoint in Isaac.
- **Known mismatch:** the numpy evaluation runs the PID at 240 Hz while Isaac
  runs it at 60 Hz with a held action; the residual grade is evaluated on the
  numpy side.
- **Not implemented:** anchoring loss, a separate critic network. **Deployment
  note:** the ONNX export contains only the policy; a residual policy needs the
  PID too.
- **Gate default:** it stops the stage at the first solid window; use
  `--no-stop-when-solid` to keep training.

## Part 12 practice questions

1. **(trace)** 16,384 envs, 256 steps/env, 128 M total steps, `MIN_WARMUP_CHUNKS
   = 16`. How many chunks, and what fraction is warmup? What single flag fixes
   it, and to what value do you set it?
2. **(own words)** State the root cause of the frozen exploration std and explain
   why the first fix (reset after `load_bc_checkpoint`) failed.
3. **(numbers)** What did the Optuna study actually show, in three numbers?
4. **(own words)** How can you tell from artifacts alone whether a run used
   residual mode?
5. **(why)** Why does the promotion gate require chunk ≥ 40 rather than the first
   good window?
6. **(own words)** What is established and what is only hypothesized about why
   direct-policy runs slide? Name the cheapest decisive test.
7. **(trace)** `res_v1`: baseline 0.971, post-unfreeze mean 0.927. Which delta
   bucket is that, and why isn't the result evidence of *learning*?
8. **(own words)** Name three claims made during this work that turned out
   wrong and how each was caught.
9. **(what breaks)** You start 3-30 m from `res_v1`'s checkpoint with
   `--residual-scale 0.1` but *without* `--residual-resume`. What is lost?
10. **(design)** What would you need to change to go from 3-30 m to 3 km?

<details><summary>Model answers</summary>

1. `chunk = 256·16384 = 4,194,304`; `128 M / 4.19 M ≈ 30` chunks; warmup =
   16 of 30 (53%). `--num-steps-per-chunk 128` gives 2,097,152 per chunk: ≈ 61 chunks when
   the whole budget goes through the chunk loop (residual mode), 56 in direct mode
   because 8% of the budget goes to the imitation stage first. — 12.2(1).
2. BC/DAgger's Gaussian NLL against a deterministic teacher has its optimum at
   `σ → 0`, so the loaded checkpoint's `actor_log_std` sat at the clamp floor
   (`σ = exp(−3) = 0.0498`), where the tanh parametrization is flat. The first
   fix reset it right after loading, but `train()` then ran its own on-policy
   imitation stage — the same NLL — which re-collapsed it before the warmup
   freeze. Resetting after the imitation stage, immediately before the freeze,
   works. — 12.2(4), 5.1a.
3. 0/100 trials held within −0.05 of their own baseline; mean baseline 0.926 →
   mean post-unfreeze 0.599; the reported best (162) ranks 27th (and 10 trials
   are within measurement noise of the true best). — 12.2(5), 0.12.C.
4. The first `metrics.csv` timestep (≈ 2.1 M = one chunk, not ≈ 12.3 M, because
   no imitation stage ran); ~61 chunks; mlflow params `residual_scale` and
   `detach_critic`; the log line `PID-residual mode: env_action = pid + …`.
   — 12.3(6).
5. The first chunks after unfreeze are essentially the warmup (imitation/PID)
   policy plus a few PPO steps and always look good; requiring ≥ 24 PPO chunks
   (chunk ≥ 40, training starts at 16) makes the promoted checkpoint one PPO
   has actually shaped and that has survived the period where the slide
   appears. — 0.11.
6. Established: PPO's own reward falls late, `grad_norm` jumps at unfreeze,
   std/entropy creep and the LR profile are not the sole cause. Hypothesized:
   noise-driven drift, trunk interference by the critic, and a numpy-vs-Isaac
   eval gap. Cheapest decisive test: evaluate an early and a late v4
   checkpoint directly in Isaac (`diagnose_model_isaac.py`); if the late one
   still succeeds there, the "slide" is an evaluation-simulator effect.
   — 12.2(7).
7. Delta = 0.927 − 0.971 = −0.044 → the "held (≥ −0.05)" bucket, which no Optuna
   trial reached. Not learning: success (0.987 → 0.983), final distance
   (0.258 → 0.260) and hit time (3.70 → 3.64 s) are unchanged within noise; the
   PID is at the ceiling on 3-10 m, so "held" is the achievable result there.
   — 12.2(9), 0.13.C.
8. E.g. (1) a "spike" that was string truncation of `1.3e-05` — caught by
   re-reading the raw column; (2) "saturated tanh has no gradient" — refined
   because Adam's step is ≈ lr regardless of gradient size; (3) "PID reward 87
   vs clone 62-67" — caught by comparing with the run's own warmup reward
   (~68) and finding the smoke test's sample was biased. — 12.3.
9. The learned correction (the actor head) is zeroed, so training restarts from
   a pure PID plus the loaded trunk/critic; the 3-10 m corrections are lost.
   `--residual-resume` keeps the head. — 12.2(9), 0.13.
10. Extend `tune_pid.DISTANCES` and regenerate gains past 250 m (settling time
    and saturation change regime), check the step budget
    (`steps_for_dist(3000) = 750,000` ticks ≈ 52 min sim), raise
    `distance_high`/OOB radius (`max(20, 3·d)`), re-derive chunk/floor
    arithmetic, expect much slower per-chunk evaluation, and possibly widen the
    curriculum in more stages (3-30 → 3-100 → …). — 12.5, 0.11.

</details>

---

# PART 13 — K14 grading-rubric compliance (checked 2026-09-27)

Source: the two official documents at
`meyda.education.gov.il/files/CSIT/K14/procedure-for-project-K14.pdf` (the
procedure — project domains, requirements, proposal format, defense format)
and `.../project-criterias-K14-714918.pdf` (the grading rubric — point table
+ automatic-failure conditions), both current as of 2026-09-27. Quotes below
are translated from the original Hebrew; re-check the source PDFs directly
for anything decision-critical — **this section is not a substitute for
your advisor's sign-off**, only a code-fact check against the rubric text.

## 13.1 Which track this project is
The procedure lists three project domains; this project is **מידת מכונה**
(machine learning) — collect data, build/train a deep-learning model, tune
it, deploy it, analyze errors. (The other two domains — OS/networks/security,
and "algorithmic problem in advanced engineering technology" — have their own
separate requirement lists; this section only checks the ML-track ones, since
that is what this codebase actually is.)

## 13.2 Checked against the ML-track requirement list (procedure PDF, p.4)
Every bullet is "must include ALL of the following stages":

| Requirement | Status | Where |
|---|---|---|
| Collect, prepare and analyze data | done | `collect_demonstrations.py`, `collect_demo_isaac.py` |
| Data reliability check (heterogeneous data, bias) | partial | target pairs are sampled uniformly (`build_uniform_omni_eval_pairs`); no explicit bias/heterogeneity analysis written up |
| Build and train a deep-learning model | done | `ActorCritic` + hand-written PPO (0.4.C, 0.10.C) — no RL library used |
| Monitor and track performance metrics | done | MLflow + `metrics.csv` + the plotting suite (0.14) |
| **Deployment on >= 2 machines with a logical/parallel/serial pipeline** (transfer learning alone doesn't count; a multi-output single model doesn't count either) | **not done** | see 13.4 |
| Efficiency: hyperparameter tuning, callbacks, justification -- **"do not settle for automatic tools such as Optuna"** | partial | Optuna search exists (0.12.C) but its results were shown to be noise-dominated and the ported hyperparameters were hand-justified (0.8.C) -- that reasoning needs to be **written up in the project book**, not left in chat history |
| Error analysis and success metrics | done | grade formula (0.7.C), `analyze_optuna_isaac.py`, Part 12's per-run tables |
| Layered architecture: interface / logic / data (MVVM/MVP/MVC) | unclear | the codebase is organized by module (`environmental`/`control`/`training`/`guidance`), not explicitly in this pattern -- map it explicitly in the book or restructure |
| System-limits performance analysis, load analysis, multi-user handling | **not done** | no load/limits analysis exists; `serve_run.py` is explicitly documented as single-request, no-auth, dev-only |
| Smart use of OOP | done | `ActorCritic`, `TorchPIDController`, `PidResidualEnv`, `BaseDroneEnv`, etc. |
| "A student who uses ONLY black boxes will lose points" | clear | PPO/GAE/reward/PID are all hand-written; PyTorch/Isaac Lab are the numerical/physics substrate, not the core algorithm |

## 13.3 Checked against the automatic-failure conditions (criteria PDF, bottom table)
Quoted, then the finding:

1. *"If the project has no algorithm meeting the required criteria -- failing
   grade."* -- Not applicable; PPO/GAE/ActorCritic is a real, substantial,
   hand-written algorithm.
2. *"If the project doesn't match the approved project proposal -- failing
   grade."* -- **Unverified here** -- I do not have your submitted proposal
   text. The procedure says the proposal states only the PROBLEM, not the
   solution method ("בהצעת הפרוייקט תוגדר מה הבעיה אך לא כיצד או איך פותרים
   אותה"), so a residual/PID architecture is very unlikely to conflict with
   a problem-only proposal -- but confirm your specific wording.
3. *"If the project contains only library functions with no White-Box work
   -- deduct 30%."* -- Not applicable (13.2's OOP/black-box row).
4. *"If the whole project is one API for the core solution, with no
   integration -- failing grade."* -- Not applicable; this is a from-scratch
   simulator, not an API wrapper.
5. *"If the student cannot explain the core algorithm and/or doesn't master
   the project -- failing grade."* -- Mitigated by this guide's existence, not
   guaranteed -- the P->B->C ladders (Part 0) and Part 12's case study exist
   specifically so you can explain PPO/GAE/the residual design and its
   trade-offs, and say plainly what you tried that didn't work and why.
6. *"Lack of generalization, heavy `if` use, deep nesting -- deduct 15%."* --
   Not audited in this pass; a normal code-quality review would catch this.
7. *"Lack of documentation -- deduct 10%."* -- Partially mitigated: this guide
   and the module docstrings are substantial, but the rubric is grading the
   **project book**, which is a separate document you still have to write.
8. **"The project must include ONE of: (a) two programming languages with
   real functional integration between them, or (b) at least one compiled
   (non-interpreted) language (C#, Java, C++, Go, Rust, ...) -- otherwise,
   failing grade."** -- **Not met.** Every student-written file in this
   repo is Python (64 `.py` files; zero `.cpp`/`.cs`/`.java`/`.go`/`.rs`,
   confirmed by directory listing). This is the one finding in this whole
   section with a hard, unambiguous automatic-failure consequence as
   currently written.

## 13.4 The recommended fix for 13.3's item 8 (and 13.2's deployment gap)
Both gaps share one fix. `export_onnx.py` already exports trained checkpoints
to ONNX; `serve_run.py` already has a serving layer. If the actual
**inference/serving** step -- loading the ONNX model and running it -- were
implemented as a real second component in a compiled language (e.g. ONNX
Runtime's C++ or C# API) instead of Python `torch.load`, with real functional
integration (the Python training/evaluation side calling it, not two
disconnected demos):
- it satisfies rubric item 8(a) directly, reusing work already done;
- it can double as the missing ">=2 machines with a pipeline" deployment
  item (13.2) if the compiled inference component runs as its own service
  the Python side talks to over a socket/HTTP, rather than in-process.

This is scoped but not trivial -- it is new code in a new language, not a
one-line change, and should be planned as its own piece of work, not done
casually alongside training experiments.

## 13.5 What to actually do with this section
- Confirm 13.3 item 8's wording and consequence with your advisor before
  treating it as certain -- the source PDF text is unambiguous, but you are
  the one accountable for how it is currently graded/updated.
- If confirmed, treat the ONNX/compiled-inference component (13.4) as its
  own planned task, not an afterthought.
- Write up the Optuna-insufficiency reasoning (0.8.C, 0.12.C) explicitly in
  the project book, in the student's own words -- the rubric names Optuna
  specifically as an insufficient tool on its own.
- Map the module layout onto interface/logic/data (or restructure) and state
  that mapping explicitly in the architecture section of the book.
- Re-read your own submitted project proposal and confirm this architecture
  (residual PID+RL, Isaac Lab, hand-written PPO) doesn't contradict its
  wording, per automatic-failure condition 2.


---
# PART 14 — Two-phase (controller -> policy) RTL training, Isaac-only evaluation (2026-10-02 -> 2026-10-05)

Everything in 14.1-14.7 is **built and was run** (files: `app/training/rtl_train_isaac.py`,
`app/reward_functions/reward_qv2.py`, plus `reward_qv1.py` which it copies). Numbers come from `runs/*/metrics.csv`,
`run_config.json` and short Isaac measurement runs. 14.8 is the **proposal** for the next version; the full design is
in `alt_model.md`. A shorter operational summary lives in `app/training/rtl-handoff.md`.

**Scope note.** Parts 0-13 describe the older pipeline (`base_training_isaac.py`, residual mode, numpy eval) and are still
correct about *that* pipeline. This Part describes the pipeline that replaced it for the "controller then policy" goal:
a **full network** (no PID residual) trained with PPO plus a PID teacher, started from a pretrained actor.

## 14.1 Two-phase guidance and the handoff distribution

### 14.1.P Prerequisites
- **Braking distance.** Slowing from speed `v` at deceleration `a` takes `d = v^2 / (2a)` metres. At 18 m/s and
  `a = 1.5 m/s^2` that is 108 m; at 35 m/s with 3-10 m of room there is no way to stop at all.
- **Tilt-limited acceleration.** A multirotor accelerates horizontally by tilting: `a_max = g * tan(tilt_max)`. With the
  PID's `max_tilt_rad = 0.3`, `a_max = 9.81 * tan(0.3) = 3.0 m/s^2` before any lag (0.3.B).
- **Distribution shift.** A network is only reliable on inputs like those it trained on. If deployment feeds it states it
  never saw, its output is arbitrary (it may output ~0 = "hover").

### 14.1.B Basics
Split the flight into two phases: a **controller** flies the long, easy part; a **policy** takes over at a *handoff point*
and does the terminal part (brake or aim, hit, and recover after a miss). Reasons: cruise is already solved by a
controller; a network over long distances is expensive (long episodes, no learning signal in straight flight, observation
scales tuned for tens of metres); speed is dominated by the cruise, so the cruise controller must not brake early.
The one hard requirement: **the states the policy meets at the handoff in deployment must be the states it trained on.**
That means matching distance, speed, lateral velocity, tilt, and even where the world origin is.

### 14.1.C In this project
Implemented by `install_handoff_stage` (`rtl_train_isaac.py`), an instance-level wrapper around `u._reset_idx`.
- **Handoff distance `h`** is drawn from the **model's** range `U(--distance-low, --distance-high)`, *not* from the spawn
  range. (A first version tied it to the spawn range; the user's rule is: "the low-high of the model, not the overall spawn".)
- **Speed buckets** (`SPEED_BUCKETS`, `SPEED_BUCKET_PROBS`): speed = `frac * --max-speed`, `frac` in 10-25 % (20 % of
  episodes), 25-75 % (40 %), 75-100 % (40 %).
- **Legacy mode** (no `--spawn-dist-*`): spawn already moving at the bucket speed, random tilt and sideways velocity; the PID
  teacher flies a short 2-5 m prefix.
- **Spawn-leg mode** (`--spawn-dist-low/high`): the drone spawns *at rest*, `spawn_dist ~ U(low, high)` from the target, and a
  scripted **carrot-chase controller** (`_carrot_action`) flies a trapezoid speed profile along the line: ramp up at
  `teacher_accel`, cruise, ramp down to the env's bucket speed exactly at the handoff point. Horizontal motion is
  *velocity tracking* (`accel = gain * (v_cmd - v)`), because a position-hold PID's `kd` term damps the cruise.
- **Handoff trigger (position, not time).** `_refresh_handoff` latches `_handed_off` when the real distance to the target
  is `<= h`, or the drone has flown the whole planned leg along the line (`s_along >= prefix_m`, covers a sideways miss of
  the handoff sphere), or the **fallback timer** (2.5x the ideal time + 10 s) runs out. Once handed off, an env stays handed off.
- **Handoff perturbation** (`_handoff_snap_obs`): at the handoff step velocity is set to the bucket speed, plus a random
  sideways component (`--lateral-frac`) and a random tilt (`--handoff-tilt-deg`); the observation is patched to match
  (velocity, quaternion, angular velocity zeroed, yaw-error terms).
- **Ramp acceleration** `--teacher-accel` defaults to `0.5 * g * tan(max_tilt)` ~ 1.52 m/s^2 (half the physical limit, so
  the drone can follow it).
- **Leg elevation clamp** `--leg-max-elev-deg` (default 15): a 200 m leg aimed down would end underground (the 0.5 m floor
  clamp then moved the target off the planned line); steep climbs crawl because the altitude channel is a plain hold.
- **Re-centring** (`--no-recenter` disables): the spawn is shifted back along the leg so the *handoff point* is at x=y=0. **Deployment must apply the same transform**: `TwoPhaseAgent(recentre_z=20)` / `record_run.py --recentre-z 20` (14.5 #13). Or avoid the transform altogether with `--obs-position-mode height` (14.5 #14).
  The observation contains absolute position (`symlog(pos)` in `_get_observations`, 8.1) and the policy trained near the origin.
- **Masking.** Controller-flown steps are excluded from the policy gradient (`batch["pm"] == 0`): PPO compares
  `pi_new(a|s)/pi_old(a|s)`, which is meaningless for an action the policy did not sample. The critic still learns from them.
- **Measured quality of the handoff** (512 envs, 150-250 m spawn, 18 m/s max, after the fixes): distance error vs `h`
  mean +2.0 m, p90 +5.2 m; speed just before the perturbation ~2.2 m/s above the bucket on average; no drone died before handoff.
  Before the fixes (3.0 m/s^2 ramp, timer trigger, 50-80 m spawn): mean +20.6 m, p90 +38.8 m, max +63.5 m; speed 6.8 vs a planned 10.5 m/s.

## 14.2 RTL as built: teacher-student, phases, guard

### 14.2.P Prerequisites
**Distillation:** train a student to match a teacher's outputs with an extra loss. **KL / MSE anchor:** a penalty for moving
away from a reference. **Trust region:** limit how far each update may move the policy (PPO's clip is a cheap one, 0.4.B).
**Advantage:** how much better than expected an action was (0.4.B).

### 14.2.B Basics
RTL here = start from a pretrained student, keep a teacher as an anchor for a while, and unfreeze the network in stages so
the first (noisy) gradients cannot wreck it. The *critic must be trained first*: PPO with a random critic produces garbage
advantages, and the BC checkpoint never had a critic (0.13).

### 14.2.C In this project (`rtl_train_isaac.py`)
- **Model:** `RTLActorCritic(ActorCritic)` adds a **separate critic MLP** (`value_net`, default 2x128); the inherited
  `critic_head` is unused. `actor_state_dict()` omits the critic, so **`best.pt` / `model_*.pt` are actor-only**; the critic
  and optimizer are in `latest_full.pt`, which `--bc-checkpoint` cannot load (a `--resume` flag does not exist yet).
- **Exploration:** `--init-std` sets the starting std; the learned log-std is clamped to `[-3.5, -2.0]` (std 0.030-0.135).
  `--ent-coef` defaults to 0 so nothing grows the std. (A tiny std cannot get a parked policy moving: 14.5 entry 6.)
- **Teacher:** `PidTeacher` runs the batched per-distance PID on the *student's* states each step; labels go into the loss
  `distill_coef * mean(w * (tanh(mean) - teacher_action)^2)` where `w = clamp(1 - speed/--distill-fade-speed, 0, 1)`
  (a hover PID commands braking at high speed, which fights "hit"). The student always flies; the PID only flies the handoff leg.
- **Phases** (`phase_of`, `apply_phase`): `settle` (rollouts only) -> `critic_warmup` (actor frozen, critic trains, 8 epochs) ->
  `head_only` (action head + log-std) -> `full`; each unfrozen group gets an LR ramp.
- **Guard (final form):** driven by the **eval** hit rate, judged at eval chunks only. The settle-chunk eval of the untouched start policy sets the bar
  (`best_eval_hit`); if eval falls `--guard-margin` below the best, the boost is multiplied (cap `--guard-boost-max`, coefficient cap `--distill-max`),
  otherwise it decays 0.9x and the bar follows new bests. (The first version used the 3-chunk mean of the *training* hit rate; see 14.5 #7 and #11 for why that failed.)
- **Known weakness (original):** the guard assumed the teacher is good and the train hit rate is a stable signal. Neither held here (14.5 entries 5, 7, 11). The signal is now eval; the teacher problem is addressed by `--anchor-checkpoint` (next bullet).
- **`--anchor-checkpoint PATH`** (`load_anchor`, `collect_rollout(..., anchor=)`): a frozen actor is the distillation teacher instead of the PID (label `tanh(mean)`, speed fade off); the PID is
  still queried for its integrators, the legacy prefix leg and the eval baseline. Scale note: a close anchor gives a tiny squared-error loss, but a measured test (14.5 #12) found the *weak* weight (0.25 -> 0.1) better than a strong one (5.0, the `--distill-max` cap): eval 0.787 -> 0.832 against 0.793 -> 0.764.
- **Transfer in this setup** = the initial weights (BC or a previous `best.pt`) + distillation. Within a run, each chunk is just continued training.

## 14.3 `reward_qv2.py`

`reward_qv1` plus one mechanism. Per policy step (60 Hz):
| Term | Value |
|---|---|
| hit | +10, terminal, distance < 0.25 m |
| progress | `2 * (log1p(prev/0.5) - log1p(dist/0.5))` (potential difference, not gamma-scaled) |
| step | -0.002 |
| timeout | -3, true terminal (no bootstrap; `--bootstrap-timeouts` restores the usual) |
| crash | -5 (oob, underground, NaN, roll > 65 deg, pitch > 80 deg) |
| tumble / tilt | small angular-velocity penalty (capped) and a soft wall at ~46 deg |
| **hover (new)** | stalled = `dist > 0.25` AND `speed < 0.3`. After 1 s: -0.05/step. After 3 s: terminate with -5 |

- **Why state, not action:** a zero action is also a legitimate brake mid-flight; "slow and not on the target" is the symptom.
  The counter resets the moment the drone moves, so **brake-and-return is free, only parking is punished**. Moving away is not penalised.
- **Why potential-based:** the sum telescopes, so oscillating between two distances earns nothing (0.7).
- **Incentive check:** parked = -5 minus the per-step cost, worse than a crash (-5). It is never chosen deliberately; a policy that parks is *stuck*, not rewarded.
- `install_reward_qv2` swaps `_compute_dones_and_reward` on the env **instance**, keeps the bookkeeping, and gates stalls on `_handed_off`
  so the controller leg (starting at rest) never counts as parking. `pop_reward_stats` returns per-chunk stats (hit/oob/roll/pitch/timeout/hover rates, per-term rewards).

## 14.4 Evaluation methodology (Isaac only)

The old numpy-simulator evaluation (and `record_run.py`, which is still numpy with the *old* reward and no hover termination)
does not test the task being trained. `isaac_eval` replaces it:
- Reset **all** envs, run the deterministic policy (mean action) and score each env's **first** episode. (`use_pid=True` runs the
  PID teacher the whole way as a baseline.)
- **Per-episode budget** from the start distance (`steps_for_dist`: ~1 s/m, 7.5 s floor); running out = `timeout` (a failure).
- Outcomes: hit / hover / oob / roll / pitch / timeout / unfinished; hit and crash rate **per speed bucket** (b0/b1/b2).
- **Handoff check** (`ho_*` keys): real distance and speed at the moment of handoff vs the sampled `h` and bucket.
- `best.pt` is the checkpoint with the highest eval hit rate (eval chunks only).

**Biases found and handled**
| Bias | Cause | Fix |
|---|---|---|
| Spikes in `train_hit` after each eval (hit 0.984 vs 0.910, hover 0.000 vs 0.066, hit time 3.2 s vs 5.2 s) | resetting all envs puts them in lock-step, so the next chunk only holds fast-ending episodes | `--eval-burnin` steps run and discarded |
| Fake timeouts / -3 penalties after a full `env.reset()` | `_reset_idx` randomises `episode_length_buf` when *all* envs reset | zero the counter after both resets |
| `eval` 0.77 vs `train_hit` 0.90 | different denominators: `train_hit` counts only episodes that *ended* in the chunk, eval counted 17 % unfinished as failures | per-distance budget; compare `hit/(1-unfinished)` |
| Chunk-level swing 0.36-0.92 with a frozen policy, `train_hit=0.000` in 4 of 5 chunks, saw-tooth in episodes ended per chunk tied to the eval period | long episodes (~23 s = ~5 chunks), synchronised starts, and every eval resetting all envs | time-uniform staggered starts at every training full reset + guard driven by eval (14.5 #7, #11); the state pool in `alt_model.md` 4.1 removes it structurally |
| Selection bias | `best.pt` = max over many evals on the same distribution | held-out set (not yet built) |

## 14.5 Case study, in order (symptom -> evidence -> diagnosis -> change -> status)

1. **Parking at 0.9 m (first replay).** *Evidence:* action ~0, speed 0, distance 0.91 m for 180 s (hit radius 0.25 m). *Diagnosis:* the
   policy overshot at 4-7 m/s (never seen in training, which started from rest), braked, and the policy's output at a stationary
   off-target state was ~0. *Change:* `--handoff` (varied-speed starts) + qv2 hover rule. *Status:* hover 0.0-0.7 % in later runs.
2. **35 m/s with 3-10 m (`rtl_handoff_v1_3_10`).** *Evidence:* `train_hit` 0.39 -> 0.42 over 122 chunks, oob 0.45, roll 0.14, std 0.060 -> 0.057;
   a 64-env smoke test showed crash rate by speed bucket 0.00 / 0.50-0.74 / 0.79-1.00 and the PID hitting ~0.20. *Diagnosis:* the fast
   buckets are physically unwinnable at that distance (braking distance, 14.1.P). *Change:* lower `--max-speed` (12-18). *Status:* hit ~0.9.
3. **Spikes** (14.4 table). *Status:* burn-in + counter zeroing.
4. **Eval reads 0.77, train reads 0.90.** *Diagnosis:* 17 % unfinished counted as failures. *Change:* per-distance budget. *Status:* done.
5. **3-30 m run from the 3-10 m `best.pt` (`rtl_handoff_v1_10_30`).** *Evidence:* eval 0.77 -> 0.67, eval hover 0.116 -> 0.20,
   `ep_return_mean` 14.4 -> 12.1, guard boost 20x (coefficient 0.45), `agree` 0.14 -> 0.085. *Diagnosis:* **not reward hacking** (return fell
   too). Likely: range extrapolation (hover 11.6 % at chunk 0), std at its floor so a parked policy could not explore out, an anchor that
   decayed then re-tightened. `best.pt` was effectively the starting policy. *Status:* wider std, higher anchor floor, staged distance ranges (advice).
6. **Parking in the replay tool** (1.4 m and 3.5 m, action exactly 0). *Evidence:* `record_run.py` runs the numpy `BaseDroneEnv` with the old
   reward and no hover termination; handoff speed 1.6 m/s was below the training minimum (10 % of max speed = 3.5 m/s at 35 m/s).
   *Diagnosis:* out-of-distribution state and/or simulator gap, **not** an exploit of qv2. *Status:* still to test by replaying the same
   checkpoint inside Isaac from the same state.
7. **Spawn-leg runs `rtl_handoff_*_2p_50_80_3_50m`.** *Evidence:* eval 0.566 at chunk 0 -> 0.353 at chunk 60 (v2; v1: 0.57 -> 0.43); eval hover
   0.235 -> 0.379; PID baseline 0.11; with the actor frozen, `train_hit` ran 0.894, 0.705, 0.359, 0.605, 0.697, 0.900, 0.709, 0.369 (period 5 chunks);
   `distill_boost` hit 20 and `d_coef` the cap 5.0 by chunk 20; `agree` 0.25 -> 0.083. *Diagnosis:* **synchronisation wave** (all envs start together,
   an episode is ~20 s ~ 5 chunks of 4.3 s) fooled the guard; the maxed anchor then dragged the student toward a PID that scores 0.11 here.
   *Change (2026-10-05):* **staggered starts** (training's full resets put each env at a random point of its controller leg; `u._stagger_next`) and the guard/baseline/abort check now only
   count chunks with enough finished episodes (`--guard-min-episodes`, default 4 % of envs); chunks with no finished episode report NaN rates instead of a fake 0.000. *Status:* this first version was **not enough** (entry 11); a better teacher is entry 12. In the user's next 8192-env run the same failure showed up first as `train_hit=0.000`
   in 4 of every 5 chunks and a guard boost of x2 then x4 within two chunks after the actor unfroze.
8. **The handoff was not at the handoff distance.** *Evidence* (`measure_handoff.py`, 512 envs): actual distance minus `h` mean +20.6 m (p90 +38.8, max +63.5);
   for `h` in [3,10) mean +30.3 m; actual speed 6.8 vs bucket 10.5 m/s. *Diagnosis:* ramp acceleration 3.0 = the physical limit, the drone lagged the
   timeline, and the trigger was a *timer* from the ideal profile. *Change:* position trigger, ramp from `0.5*g*tan(max_tilt)`, fallback timer. *Status:* mean +2.0 m after.
9. **150-250 m spawn: 30 % of legs never arrived.** *Evidence:* fallback handoffs concentrated at high `|unit_z|` (mean 0.69), handoff altitude ~156 m, leg time ~55 s;
   a first fix (vertical velocity tracking) made it worse (156/512). *Diagnosis:* downward legs end underground (floor clamp moved the target off the line); steep climbs crawl.
   *Change:* `--leg-max-elev-deg` 15 and a second trigger (`s_along >= prefix_m`). *Status:* fallback < 5 %, none died before handoff.
10. **Same policy, 0.09 vs 0.82.** *Evidence:* chunk-0 eval 0.09 (150-250 m) vs 0.56 (50-80 m); handoff accuracy equal in both. *Diagnosis:* absolute position in the
    observation (handoff 100-250 m from the origin). *Change:* re-centring is **on by default** (`--no-recenter` disables it): the spawn is shifted so the handoff point is at x=y=0. *Status:* 0.816.

11. **The waves came back after the first stagger (`rtl_handoff_v1_2p_150_250_3_50m`, 8192 envs).** *Evidence:* the run started after the stagger edit (`run_config.json` contains
    `guard_min_episodes`), yet episodes ended per chunk read 2475, 1941, 1046, 452, 226 then 2470, 1940, 1078, 392, 221 ... (period 5 = the eval period, jumps right after each eval);
    `d_coef` reached the cap 5.0 and the boost 20x by chunk 29; eval fell 0.852 -> 0.823 -> 0.711 (chunk 30). *Diagnosis:* (a) every eval resets **all** envs, re-synchronising them every 5 chunks;
    (b) the first stagger was uniform in *distance* along the leg, but the drone covers ground fast while cruising, so most envs had little time left and ended together; (c) the guard still read
    the chunk-level `train_hit` (0.5-0.92 depending on cycle phase). Comparing chunks of the *same* phase (2/12/22: 0.78, 0.80, 0.87) showed training was in fact improving, as eval was (0.796 -> 0.852).
    *Change:* stagger **uniform in time** (inverse of the trapezoid time function, `_reset_idx`), applied at every training full reset; guard driven by the **eval** hit rate against the start policy's eval.
    *Status:* smoke run (512 envs, 22 chunks, three eval resets): episodes ended per chunk 32-97 (no saw-tooth), `train_hit` 0.74-0.96 in every chunk, no GUARD line, `d_coef` on schedule, eval 0.79-0.81.
12. **`--anchor-checkpoint`: use the previous best as the teacher instead of the PID.** *Why:* the PID scores 0.1-0.65 here and the student ~0.8-0.9, so distilling toward it can only pull the student
    toward braking and parking; the BC checkpoint is itself a PID clone. *Change:* a frozen copy of any actor file supplies the distillation label (`tanh(mean)`, no speed fade). *Check:* a 256-env,
    12-chunk test ran cleanly (`agree` 0.000 -> 0.012). I first expected that to make the PID-era weight 0.25 too weak, so I tested two strengths (1024 envs, 150-250 m spawn, 23 chunks, one seed each, eval +-1.2 points):
    weight 0.25 -> 0.1: eval **0.787 -> 0.832**, eval hover 0.179 -> 0.130; weight 20 -> 5 (capped by `--distill-max` to a flat 5.0): eval 0.793 -> 0.764, hover 0.167 -> 0.207, `agree` 0.007
    (stayed closer). *Diagnosis:* the strong anchor froze the student at its starting behaviour, including the start policy's ~17 % hover, and RL could not improve it; the weak one let RL fix part
    of it. *Status:* keep the weak weight for a frozen-best anchor; one seed, to be confirmed on the full-size run. (My prior of "weak = no constraint, so raise it" was wrong.)

13. **The numpy two-phase replay showed 9 % hits; the cause was the position input, not the policy.** *Evidence:* `record_run.py --two-phase` (numpy `BaseDroneEnv`, PID leg, 250 m spawn, switch 40 m) on the new `best.pt`
    (Isaac eval 0.875): 75 scenarios -> 7 hit (9 %), 28 parked 0.3-0.9 m away, 6 other timeouts, 31 roll/pitch crashes (41 %), 3 PID-leg out-of-bounds (targets at 1-3 m altitude: the PID leg flew into the ground, not the policy).
    The first RL action was ~14.0 (the top of the thrust range) in nearly every episode and peak speeds reached 17-40 m/s (trained to 18). *Diagnosis:* the tool feeds the **absolute** position (`obs[0:3]`); at handoff the
    drone is ~210 m from the origin, up to 250 m high, while Isaac training re-centres the world so the handoff point is at x=y=0 (14.1.C). *Test:* same checkpoint, same 24 scenarios (seed 0, step cap 25000), stock vs
    position re-centred at the handoff (`pos' = pos - pos_handoff + (0,0,20)`): **3/24 hits stock (8 attitude, 13 timeouts, 10 parked) vs 24/24 hits re-centred** (0 crashes, 0 timeouts). *Change:* `TwoPhaseAgent(recentre_z=...)`
    and `record_run.py --recentre-z 20`. *Status:* verified end to end from the CLI (4/4). Caveats: numpy simulator, one seed, `z0 = 20` untuned, and the 3 low-altitude PID-leg failures are a separate problem.
    *Lesson:* a training-time observation transform (re-centring) is part of the policy and must be applied at deployment too; the proper fix is a target-relative observation (`alt_model.md` 4.2).

14. **Option C built: a `height` observation mode.** *Why:* #10 and #13 showed the policy's dependence on absolute position (0.09 vs 0.82 in Isaac, 3/24 vs 24/24 in the numpy replay) and that a training-time re-centring has to be mirrored at deployment.
    *Change:* `--obs-position-mode height` (default `full`): `obs[0:3] = (0, 0, symlog(clip(z, 0, 30)))`, same 23-dim layout; implemented in the numpy and Isaac envs, recorded in `run_config.json`, read by `record_run.py`, ignored by `recentre_z`.
    *Evidence:* the existing `full`-mode `best.pt`, with no training, scored **0.935** in the new mode at chunk 0 (1024 envs) against 0.875 in its own mode, with eval hover 0.036 against 0.11-0.16; a 10-chunk run from it stayed at 0.89-0.93.
    *Status:* a full retrain in `height` mode is the next step; numpy replay parity check (same 24 scenarios, 250 m / switch 40 m, seed 0, the same `full`-trained `best.pt`, no `recentre_z`) in `height` mode: **24/24 hits** (0 crashes, 0 timeouts), the same as the re-centred `full` replay and matching the Isaac eval. Caveat: the height clip (30 m) is a design choice, not tuned.

**Wrong first explanations along the way:** "the policy parks / crashes at handoff, so it needs finer inputs or a better reward" (#13: the deployment replay never re-centred the position, 3/24 -> 24/24 once it did); "reward hacking" (14.5 #5, #6: return fell, and the replay had no qv2); "the stagger fixed the waves" (#11: it did not, the evals re-synchronised the envs); "a close anchor is too weak, raise its weight" (#12: the stronger weight did worse); "the best checkpoint is good" (#5: it was the start);
"timeouts are the main failure" (#4: artifact of the reset); a vertical-tracking fix (#9) that regressed and was reverted.

## 14.6 Reading the log line
`train_hit` = share of episodes that *ended* in the chunk that were hits (noisy policy, training envs). `eval` = deterministic hit rate on freshly reset envs.
`[b0/b1/b2]` = eval hit rate per speed bucket. `pid` = PID baseline in the same eval. `hover`, `timeout` = training-episode shares. `agree` = mean student-teacher
action gap. `d_coef` = effective teacher weight (schedule x guard boost). `kl`, `vloss`, `ev` = PPO and critic health. Below it: `eval endings ...` and `handoff check ...`.
If `train_hit` and `eval` disagree, check in order: `unfinished`/`timeout` share, noisy vs deterministic, how the envs were reset.

## 14.7 Numbers to know cold
| Quantity | Value |
|---|---|
| Hit radius | 0.25 m; policy rate 60 Hz (`decimation 4`, physics 240 Hz) |
| `a_max` | `g*tan(0.3) = 3.0 m/s^2`; ramp uses half (~1.52) |
| Chunk | 256 steps (4.3 s); 8192 envs -> 2.1 M steps; 128 M steps = 61 chunks |
| qv2 | hit +10, crash -5, timeout -3, progress k=2 eps=0.5, step -0.002, hover grace 1 s / limit 3 s, c_hover 0.05, r_hover 5 |
| Std range | log-std in [-3.5, -2.0] -> 0.030-0.135 |
| Handoff accuracy (after fixes) | distance error mean +2.0 m / p90 +5.2 m; before: +20.6 m / +38.8 m |
| Start-policy eval | 0.09 (150-250 m, not re-centred) vs 0.816 (re-centred) |
| 35 m/s at 3-10 m | crash by bucket ~0 / ~0.6 / ~0.9; PID ~0.2 |

## 14.8 Proposed next version -> see `alt_model.md`
**Already built (2026-10-05):** a frozen-best anchor (`--anchor-checkpoint`, a simple MSE form, not yet the proposed KL form) and a guard driven by eval (against the start policy, not yet a held-out set); time-uniform staggered starts.
**Still only a design:** handoff-state pool instead of flying the leg in every episode; target-relative observation with a sensor-noise interface (camera and tracker later);
time-to-hit and re-attack reward terms; low-speed and overshoot starts; `--resume` (critic + optimizer); a held-out eval grid plus an end-to-end test. Order of work is in `alt_model.md` section 6.

## 14.9 Formula sheet

Every formula the training script computes, in one place (added 2026-10-05). PPO theory: 0.4; potential shaping: 0.7; PID and tilt-limited acceleration: 0.3.

Notation: `s` = state/observation, `a` = env action, `mu`, `sigma` = actor mean and std, `eps` = noise, `r` reward, `d` done flag, `V` critic, `g` = 9.81.

**1. Action and its probability** (`ActorCritic.scale_action`, `log_prob_raw`)
```
sigma      = exp( log_std_min + 0.5 * (log_std_max - log_std_min) * (tanh(p) + 1) )     p = learned parameter; range [-3.5, -2.0] -> sigma in [0.030, 0.135]
raw        = mu + sigma * eps ,   eps ~ N(0, 1)                                             (eval: raw = mu)
a          = low + (tanh(raw) + 1)/2 * (high - low)                                         (squash into the allowed range)
log pi(a)  = sum_k [ -(raw_k - mu_k)^2 / (2 sigma_k^2) - log sigma_k - 0.5 log(2 pi) ]
             - sum_k log( 0.5 (high_k - low_k) (1 - tanh(raw_k)^2) + 1e-6 )                (change of variables for the tanh + rescale)
entropy    = sum_k ( 0.5 + 0.5 log(2 pi) + log sigma_k )
```

**2. Advantage and return** (`compute_gae`; gamma 0.99, lambda 0.95)
```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
A_t     = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}          (computed backwards in time; d_t cuts the chain at an episode end)
R_t     = A_t + V(s_t)
A_hat   = (A - mean(A)) / (std(A) + 1e-8)                          (normalised per batch)
```

**3. The loss** (`ppo_distill_update`)
```
rho        = exp( log pi_new(a|s) - log pi_old(a|s) )
pg_each    = -min( rho * A_hat , clip(rho, 1-eps_c, 1+eps_c) * A_hat )                (eps_c = --clip-eps, 0.2)
L_pg       = sum( pg_each * pm ) / max(sum(pm), 1)                                    pm = 1 if the POLICY chose the action, 0 for controller-flown steps
L_v        = mean( (V(s) - R)^2 )
L_distill  = mean( tw * mean_k ( tanh(mu_k) - a_teacher_k )^2 )                      a_teacher in [-1, 1]
L          = L_pg + vf_coef * L_v - ent_coef * entropy + coef_eff * L_distill
approx KL  = mean( (rho - 1) - log rho )        explained variance = 1 - Var(R - V) / Var(R)
```
In critic warmup only `vf_coef * L_v` is used (the actor receives no gradient).

**4. Teacher label and distillation weight** (`PidTeacher.label`, `distill_scheduled`, the loop)
```
a_teacher   = clamp( 2 (a_pid - low)/(high - low) - 1 , -1, 1 )                       (PID)   or   tanh(mu_frozen)   (--anchor-checkpoint)
tw          = clamp( 1 - speed / --distill-fade-speed , 0, 1 )                         (PID teacher; speed read from the observation)   tw = 1   (anchor)
ds          = start + (end - start) * min(1, idx / decay_chunks)                       idx = number of chunks in which the actor was trainable
coef_eff    = min( --distill-max , ds * boost )     while the actor trains,  0 otherwise      <- the CAP: --distill-start 20 with the default --distill-max 5 acts as 5.0
```
**Guard** (at eval chunks only): `best` is set by the untouched start policy's eval. If `eval < best - margin`: `boost <- min(--guard-boost-max, boost * --guard-gain)`; else `best <- max(best, eval)`, `boost <- max(1, 0.9 * boost)`.

**5. Reward** (`reward_qv2`, per policy step)
```
progress = k * ( log(1 + d_prev/e) - log(1 + d/e) )         k = 2, e = 0.5 (potential difference; sums telescope)
step = -0.002 ; hit = +10 (d < 0.25) ; crash = -5 ; timeout = -3 ; tumble = -min(5e-4 |omega|^2, 0.05) ; tilt = -2 * relu(max(|roll|,|pitch|) - 0.8)^2
stalled_t = [ d > 0.25 ] and [ speed < 0.3 ] and handed_off          c_t = (c_{t-1} + 1) * stalled_t
hover     : -0.05 per step while c_t > grace (1 s) ;  terminate with -5 when c_t >= limit (3 s)
```

**6. Controller leg** (`install_handoff_stage`, `_carrot_action`; `L` = leg length = spawn_dist - h, `v_h` = bucket speed)
```
a_ramp   = 0.5 * g * tan(max_tilt) = 0.5 * 9.81 * tan(0.3) = 1.52 m/s^2                (default --teacher-accel; the physical limit is g*tan(0.3) = 3.04)
v_peak   = min( v_max , sqrt( a_ramp * L + 0.5 * v_h^2 ) )
d_up     = v_peak^2 / (2 a_ramp)         d_down = (v_peak^2 - v_h^2) / (2 a_ramp)       d_cruise = max(0, L - d_up - d_down)
t_up     = v_peak / a_ramp               t_cruise = d_cruise / v_peak                   t_down = (v_peak - v_h) / a_ramp
v_cmd(s) = sqrt(2 a_ramp s)                                  s < d_up
           v_peak                                            d_up <= s < d_up + d_cruise
           sqrt( max( v_peak^2 - 2 a_ramp (s - d_up - d_cruise) , v_h^2 ) )          after that           (s = distance flown along the line, read from the drone's position)
accel_xy = gain * ( u_xy * v_cmd - v_xy )                    (velocity tracking, gain = --teacher-cruise-gain)
fallback timer = 2.5 * (t_up + t_cruise + t_down) + 10 s
braking distance of any approach: d = v^2 / (2 a)
```
**Staggered start** (training full resets): `tau ~ U(0, t_up + t_cruise + t_down)`; then
`s0 = 0.5 a tau^2, v0 = a tau` (ramp-up), `s0 = d_up + v_peak (tau - t_up), v0 = v_peak` (cruise), `s0 = d_up + d_cruise + v_peak w - 0.5 a w^2, v0 = max(v_peak - a w, v_h)` with `w = tau - t_up - t_cruise` (ramp-down).

**7. Geometry and handoff**
```
handoff distance  h ~ U(--distance-low, --distance-high)         leg  L = spawn_dist - h ,  spawn_dist ~ U(--spawn-dist-low, --spawn-dist-high), at least h + 0.5
line direction    u = (u_xy, u_z) ,  u_z clamped to [ -(z_spawn - 0.5)/D , sin(--leg-max-elev-deg) ]   (D = h + L ; target stays above 0.5 m), then u_xy rescaled to keep |u| = 1
re-centre         x_spawn,xy = -u_xy * L      (so the handoff POINT is at x = y = 0)
handed_off  <=  [ |x_target - x| <= h ]  OR  [ (x - x_spawn) . u >= L ]  OR  [ timer <= 0 ]        (latched until the episode ends)
speed bucket      speed = frac * v_max ;  frac ~ U(0.10, 0.25) w.p. 0.2 ; U(0.25, 0.75) w.p. 0.4 ; U(0.75, 1.0) w.p. 0.4
handoff error     d_actual - h   and   speed_actual - bucket speed   (both logged in `handoff check`, before the random perturbation)
```

**8. Evaluation**
```
episode budget (physics steps at 240 Hz) = max(1800, floor(750 * d_start / 3)) ;  policy steps = that / 4      (steps_for_dist; ~1 s per metre, 7.5 s floor)
hit among finished = hit / (1 - unfinished)            e.g. 0.765 / (1 - 0.172) = 0.924
teacher normalisation of the PID action, observation scale VEL_SCALE = 10 m/s (obs[3:6] = velocity / 10)
```

## Part 14 practice questions

1. **(own words)** Why is the policy trained only on the terminal phase, and what is the one condition the two phases must satisfy?
2. **(trace)** An env samples `h = 12 m`, speed bucket 0.6 x 18 m/s, spawn 200 m. Trace which controller flies when, what triggers the handoff, and what is changed at that step.
3. **(why)** Why is the hover rule based on state, and why can a parked drone not be "rewarded"?
4. **(numbers)** The ramp used 3.0 m/s^2. Why was that a mistake, and what did the measurement show?
5. **(why)** Why are controller-flown steps masked from the policy gradient but not from the critic?
6. **(diagnose)** `train_hit` goes 0.89, 0.71, 0.36, 0.61, 0.70 while the actor is frozen. What is happening and what did it trigger?
7. **(diagnose)** The same start policy scores 0.09 and 0.56 for two spawn ranges, with equal handoff accuracy. What is the cause and how was it confirmed?
8. **(own words)** What is wrong with distilling toward the hover PID in this task?
9. **(compare)** `eval=0.77`, `train_hit=0.90`, `unfinished=0.172`. Reconcile them.
10. **(what breaks)** You start a new run from `best.pt` with a fresh critic and `--distill-start 1.0`. Name two things that go wrong.
11. **(design)** Describe the handoff-state pool and the two problems it removes.
12. **(design)** What must the observation look like for a camera drone, and why not train on pixels?
13. **(trace)** You pass `--distill-start 20 --distill-end 5` and nothing else. What is the effective distillation weight in the first and in the last chunk, and why?
14. **(diagnose)** A numpy two-phase replay of a checkpoint with Isaac eval 0.875 hits only 9 % of the time, with a saturated first thrust action and 41 % roll/pitch crashes. What do you check first, and what experiment confirms it?

<details><summary>Model answers</summary>

1. Cruise is solved by a controller; a network over long distances costs a lot and learns nothing in straight flight; speed comes from the cruise. The condition: the
   states (distance, speed, lateral velocity, tilt, origin) at the handoff in deployment must match training. 14.1.B.
2. The carrot-chase controller flies from rest: ramp up, cruise, ramp down to the bucket speed (10.8 m/s). Handoff fires when the real distance <= 12 m, or the drone has flown
   the whole planned leg along the line, or the fallback timer runs out. At that step velocity is set to the bucket speed plus a random sideways component and a random tilt, and the
   observation is patched to match; from then on the policy acts. 14.1.C.
3. A zero action is also a valid brake mid-flight; "slow and off-target for 1-3 s" is the symptom. Parked = -5 minus the per-step hover cost, worse than a crash (-5), so no policy
   chooses it on purpose; one that parks is stuck. 14.3.
4. 3.0 equals `g*tan(0.3)`, the physical limit, so the drone lagged the profile. The trigger was a timer from the ideal profile, so handoff fired 20.6 m too far on average (p90 +38.8)
   at 6.8 m/s instead of 10.5. 14.5 #8.
5. PPO's ratio compares the new and old probability of the action; for an action the policy did not sample that is meaningless. The critic needs only states and returns, which are valid
   whoever acted. 14.1.C.
6. A synchronisation wave: all envs start together, episodes last ~20 s (~5 chunks), so each chunk sees a different slice; the policy is constant. The guard read it as a hit-rate
   drop and raised the teacher weight to its cap (5.0), which dragged the student toward a PID scoring 0.11. 14.5 #7.
7. Absolute position in the observation: the 150-250 m spawn puts the handoff 100-250 m from the origin, outside training. Confirmed by re-centring the spawn so the handoff point
   is at x=y=0: eval went 0.09 -> 0.816. 14.5 #10.
8. It is a hover controller: it brakes and stops, which is the opposite of "hit". Its measured hit rate here is 0.1-0.65 and below the student's, so anchoring pulls the student toward
   braking and parking (agree fell, hover rose). 14.2.C, 14.5 #7.
9. The 17 % unfinished episodes are counted as failures in eval and ignored by `train_hit` (which counts only episodes that ended in the chunk). `0.765 / (1 - 0.172) = 0.924`, close
   to 0.90. 14.4.
10. The critic is untrained (needs warmup: `ev` low, garbage advantages) and the optimizer is fresh; and a strong anchor to the PID drags a policy that may already beat it back toward
    the PID's behaviour. 14.2.C.
11. Run the real controller once from long distances, log the handoff states, re-centre them, and spawn training episodes from the pool plus noise and hard cases. It removes the
    cost of flying a 20 s leg every episode and the lock-step problem (each env starts from a different state). `alt_model.md` 4.1.
12. Target-relative bearing/range with uncertainty and a visible flag, plus own velocity and attitude; no absolute position. Pixels are expensive and fragile to train on; a separate
    perception module outputs the estimate and the policy trains on a noise model of it. `alt_model.md` 4.2.
13. `coef_eff = min(--distill-max, ds * boost)`; `--distill-max` defaults to 5.0, so the weight is 5.0 in both the first chunk (ds = 20) and the last (ds = 5), the cap binds throughout. To get 20 you must also pass `--distill-max 20`. 14.9 section 4.
14. Whether the replay feeds the policy the same observation it trained on: the Isaac training re-centres the world so the handoff point is at x=y=0, the replay gave absolute positions 200+ m from the origin. Confirm by re-running the same checkpoint on the same scenarios with the position re-centred at the handoff: 3/24 -> 24/24 hits. 14.5 #13.

</details>
