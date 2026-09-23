# Project Defense Guide — Drone RL Simulator (numpy oracle + Isaac Lab port)

Built from the actual code as of 2026-09-16 (not from docs-*.md, which may be
stale, and not from memory). Every claim below cites a real `file:function`.
Scope: every active `.py` file under `app/`, excluding `__pycache__`,
`deprecated/`, `app/test/`, and run-artifact folders (`app/z_final_version_*`).

## How to use this

1. Read **Part 0** once — it's the theory every later part assumes. If a term
   in Parts 1-9 is unfamiliar, it's almost certainly defined there.
2. Read each Part's **file walkthrough** before its **practice questions**.
3. For practice: cover the "Model answer" and answer from memory first
   (closed-book, like a real defense), then check yourself. Score yourself
   0-3 per the rubric in Part 11 — don't be generous.
4. Part 10 has cross-file "trace the data" scenarios — these are the
   hardest and most realistic defense questions (a real panel will ask "and
   then what happens to that number", not "define GAE").

---
# PART 0 — Prerequisite theory

## 0.1 Rigid-body dynamics

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
- **PPO (Proximal Policy Optimization)**: policy gradient with a **clipped
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
  The original DAgger algorithm (Ross, Gordon & Bagnell 2011) rolls out a
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

A shaping term `F(s,s') = γ·φ(s') - φ(s)` added to a reward is
**policy-invariant** (Ng, Harada & Russell 1999) — the optimal policy under
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
