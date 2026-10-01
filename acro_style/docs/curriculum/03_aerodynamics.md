# 3. Drone Aerodynamics & Motor Modeling

*Curriculum topic 3/5 — acro-style drone project. Expanded to first-course-in-aerodynamics
depth, from rotational-motion fundamentals up through where the repo's physics model actually
comes from and what's missing from it.*

---

## 0. Prerequisite: circular/rotational motion from scratch

Everything below builds on five quantities. If any of these are shaky, this is the section to
reread before continuing.

**Angular position, velocity, acceleration** — the rotational analogs of position/velocity/
acceleration, exactly as linear motion has x, dx/dt, d²x/dt²:

```
θ  = angular position (radians)
ω  = dθ/dt   = angular velocity   (rad/s)   -- this is the ω you've been using all along
α  = dω/dt   = angular acceleration (rad/s²)
```

**Linear-to-angular dictionary** (every linear-motion concept has a rotational twin):

| Linear | Rotational | Relationship |
|---|---|---|
| position x | angle θ | — |
| velocity v | angular velocity ω | — |
| mass m | moment of inertia I | I = Σ mᵢrᵢ² (mass, weighted by *distance from the spin axis, squared*) |
| force F | torque τ | τ = r × F (a force applied at a distance r from the axis creates torque) |
| momentum p = mv | angular momentum L = Iω | — |
| Newton's 2nd law: F = ma | rotational 2nd law: τ = I·α | — |

**Why "distance squared" shows up in I:** a mass far from the spin axis is both harder to
accelerate rotationally (more leverage needed) *and* contributes more inertia per unit mass than
one close to the axis — both effects scale with r, so together they scale with r². This
r²-appears-from-two-effects pattern is worth remembering; a very similar thing happens with
thrust in section 1.

**Torque from a force at radius r, tangential to the rotation:** `τ = r·F` (magnitude; the
cross-product handles direction/sign in 3D — a force pushing "around" the axis at distance r
produces torque proportional to both r and F).

**Angular momentum and why spinning things resist having their axis changed:** `L = I·ω`, a
vector pointing along the spin axis. Changing L requires torque (`τ = dL/dt`), same as changing
linear momentum requires force. A fast-spinning object has a large L even from a small applied
torque's perspective, which is *why* gyroscopic effects (section 3) exist at all — it's the same
"changing momentum needs force, over time" idea, just applied to rotation.

---

## 1. Where `T = k_f·ω²` and `M = k_m·ω²` actually come from

The repo's formulas aren't arbitrary curve-fits — they fall out of **blade element theory**
(BET), the standard first-course derivation for propeller thrust.

**Setup:** a propeller blade is a thin airfoil spinning about the shaft axis. Take a small slice
of the blade at radius r from the axis, of width dr:

```
        axis of rotation
             |
             |  ← shaft
   ==========o==========   ← blade, spinning at ω
             |
        <-r->|<-dr->        one small blade element, at radius r, width dr
```

That element moves in a circle, so its **tangential (airspeed) velocity is `v = ω·r`** — directly
from the ω = dθ/dt definition in section 0 (arc length = r·θ, so speed along the arc = r·dθ/dt =
r·ω).

**Aerodynamic force on that element** comes from **dynamic pressure**, the fundamental quantity
behind essentially all aerodynamic forces:

```
q = ½·ρ·v²        (dynamic pressure: ρ = air density, v = local airspeed)
```

Force on any small aerodynamic surface is `dF = q·C·(area) = ½·ρ·v²·C·(area)`, where C is a
lift or drag coefficient depending on the airfoil shape and angle of attack. For our blade
element, `v = ω·r`, so:

```
dLift(r) ∝ ρ·(ω·r)²·c(r)·C_l·dr = ρ·ω²·r²·c(r)·C_l·dr
```

**Integrate over the whole blade length** (from root to tip) and every element contributes a
term proportional to ω² (ω is the same for every element, r varies) — so the *total* thrust from
one blade, and hence one rotor, comes out proportional to ω²:

```
T = k_f · ω²          where k_f bundles up ρ, blade shape, C_l, and the r²-weighted integral
```

That's exactly `methods.py`'s formula, now derived rather than asserted — and it's the same
"r² from two stacking effects" pattern from section 0: velocity itself scales with r (one
factor), and dynamic pressure is velocity *squared* (another factor of the same r), so integrating
a velocity-squared aerodynamic force over the blade naturally produces the ω² scaling once you
factor ω back out.

**Reaction torque works the same way**, but from *drag* instead of lift: each blade element also
experiences drag `dDrag(r) ∝ ρ·ω²·r²·c(r)·C_d·dr`, and drag applied at radius r produces torque
`dM(r) = dDrag(r)·r ∝ ρ·ω²·r³·c(r)·C_d·dr`. Integrate over the blade → `M = k_m·ω²`, matching
the repo's reaction-torque formula, and explaining *why* it's also ω²-scaled and not some
independent relationship — it's the same underlying blade-element drag, just with an extra
factor of r from the torque arm.

---

## 2. The nonlinearity's control consequence (recap, now grounded)

Because `T = k_f·ω²`, the *sensitivity* `dT/dω = 2·k_f·ω` grows with ω itself — the higher the
current rotor speed, the more thrust a small speed change produces. Near hover (moderate ω) this
gain is modest; near max RPM (acro-scale rate commands) the same size of motor-speed change
produces a much larger thrust swing. A fixed-gain controller tuned at hover doesn't automatically
stay well-tuned across that whole range — deep RL can learn to compensate for this state-dependent
gain directly (it doesn't need an explicit gain schedule the way classical control would), but the
underlying physics reason for the effect is exactly the blade-element derivation above.

---

## 3. Rigid-body rotation and where the gyroscopic term comes from — Euler's equations

This is the formal version of the `ω × (I·ω)` term from the Phase 1 plan.

**Setup:** angular momentum is `L = I·ω` (I is the moment-of-inertia tensor — for a drone with
symmetric mass distribution about its principal axes, this is just a diagonal matrix of
`[I_x, I_y, I_z]`, one moment of inertia per axis). Newton's rotational law says **torque equals
the rate of change of angular momentum**, but only when that rate of change is measured in a
**non-rotating (inertial) frame**:

```
τ = (dL/dt)_inertial
```

The catch: everything the drone measures (IMU, the sim's own integration) is naturally expressed
in the **body frame**, which is *itself rotating*. The rate of change of a vector as seen from a
rotating frame differs from its rate of change in the inertial frame by a standard result called
the **transport theorem**:

```
(dL/dt)_inertial = (dL/dt)_body + ω × L
```

(Intuition: even if L isn't changing at all in the body frame — e.g. it's fixed along the shaft
of a spinning gyroscope — an outside inertial observer still sees L *sweeping around* just
because the frame itself is rotating; that sweeping shows up as the `ω × L` term.)

Substituting `L = I·ω` and rearranging for the body-frame angular acceleration:

```
τ = I·(dω/dt)_body + ω × (I·ω)
  ⇒  I·(dω/dt)_body = τ − ω × (I·ω)
```

This is **Euler's rotation equation**. In component form, for principal axes (diagonal I):

```
I₁·ω̇₁ = τ₁ + (I₂ − I₃)·ω₂·ω₃
I₂·ω̇₂ = τ₂ + (I₃ − I₁)·ω₃·ω₁
I₃·ω̇₃ = τ₃ + (I₁ − I₂)·ω₁·ω₂
```

Each axis's angular acceleration depends not just on its own torque, but on the *product* of the
other two axes' angular velocities. That's the coupling: **spin fast enough about two axes at
once and you get an apparent torque on the third axis, with no motor commanding it.** This is
exactly why gyroscopes precess, why a spinning top's axis traces a cone instead of just falling
over, and — for this project — why a controller that only uses `τ = I·α` (ignoring the cross
term) will be systematically wrong at high ω, which is precisely the acro flight regime.

`app/dynamics/methods.py`'s current `angular_acceleration = torque/inertia` (line 105-109) is the
**simplified version with the cross term dropped** — valid when ω is small (hover, gentle
approach — this repo's original use case), invalid once ω gets large (acro). Phase 1 adds the
`ω × (I·ω)` term back in, i.e. implements the full Euler equation instead of the simplified one.

---

## 4. Motor dynamics: why the lag is first-order

A brushless motor doesn't reach a new speed instantly. Simplified reasoning: current through the
motor windings creates torque, but the motor spinning also generates a **back-EMF** (voltage)
opposing the driving voltage, proportional to the motor's own speed — so as it speeds up, the net
driving voltage (and hence torque) available shrinks, until it settles at the speed where
driving-voltage and back-EMF balance. Combined with the rotor+propeller's own moment of inertia
resisting speed changes, this produces the classic first-order response:

```
dω/dt = (ω_target − ω_current) / τ
```

`τ` (motor_tau, already in the repo, `=0.05s`) is the time constant of that electromechanical
settling process — smaller τ means a snappier motor (less winding inductance/rotor inertia
relative to its torque authority). This is the one actuator-dynamics effect beyond the ideal
already in the sim, and it's the physical reason the rate-control inner loop (topic 2) has
something real to compensate for — if motors responded instantly, there'd be no lag to fight.

---

## 5. What's modeled vs. what's missing, now precisely

**Modeled**, with the derivation now attached:
- Quadratic thrust/torque `T,M ∝ ω²` — blade element theory, §1.
- Motor first-order lag — back-EMF/inertia settling, §4.
- Simple isotropic quadratic drag `F = ½ρv²C_dA` — same dynamic-pressure idea as §1, applied to
  the whole airframe with one direction-independent C_d.
- Wind force (present in code, currently disabled/zeroed).

**Not modeled**, now precisely locatable in the theory above:
- **Gyroscopic coupling `ω×(Iω)`** — the dropped cross term from Euler's equation, §3. In scope
  for Phase 1: cheap to add (it's a few lines — literally computing that cross product each
  step), and directly relevant once ω gets large, which acro maneuvers guarantee.
- **Anisotropic drag** — real C_d isn't a single constant; it depends on the airframe's shape
  facing the airflow (edge-on vs. flat-on), i.e. on **angle of attack** relative to the body
  frame, not just speed. The isotropic model in §5 assumes C_d is the same from every direction.
- **Rotor-rotor interaction** — blade element theory (§1) assumes each rotor's induced airflow is
  undisturbed; in reality, closely-spaced rotors' downwash overlaps (and near a wall/floor,
  ground effect increases apparent thrust) — a correction to the "undisturbed inflow" assumption
  underlying §1's derivation.
- **Battery voltage sag** — the back-EMF/torque-speed picture in §4 assumed a constant supply
  voltage; under heavy current draw a real battery's voltage droops, shrinking the torque-speed
  curve dynamically. This is an *electrical* model on top of the mechanical one in §4.

## The scope call this topic sets up

- **In scope (Phase 1):** gyroscopic coupling — cheap, and directly derived above as the thing
  the current sim silently drops that acro-scale ω makes matter.
- **Worth a second look (Phase 3):** anisotropic drag, if the maneuver set ends up dominated by
  high-speed edge-on flight (racing-style) rather than pure aggressive rotation.
- **Provisionally out of scope:** rotor-rotor interaction and battery voltage sag — real effects,
  but both would mean fitting correction terms with no physical airframe to measure them against.

## Summary

| Effect | Modeled? | Where it comes from | Matters for acro? |
|---|---|---|---|
| T,M ∝ ω² | yes | blade element theory, §1 | yes — sets the nonlinear control gain |
| Motor lag (first-order) | yes | back-EMF/inertia settling, §4 | yes — what the rate loop fights |
| Isotropic drag | yes | dynamic pressure, §1/§5 | partially — direction-independent |
| Gyroscopic coupling ω×(Iω) | **no** | dropped cross term, Euler eq. §3 | **yes, significantly — Phase 1** |
| Anisotropic drag | no | angle-of-attack-dependent C_d | maybe, depends on maneuver style |
| Rotor-rotor interaction | no | broken undisturbed-inflow assumption | minor, no airframe to validate |
| Battery voltage sag | no | electrical model on top of §4 | no physical drone yet — deferred |
