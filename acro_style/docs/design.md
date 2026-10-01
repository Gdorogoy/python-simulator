# Acro-Style Drone — Capability Spec

Status: **open** — written after the curriculum (see `curriculum/capability_notes.md`) closes
out topic 5. Until then this holds the open questions the curriculum needs to answer, plus the
provisional architecture from the plan so nothing is decided twice.

## Open questions to resolve via the curriculum

- **Topic 1 (rotation math):** confirmed direction — quaternion-native tilt-from-vertical
  termination replacing the Euler roll/pitch split. Any edge cases in the switchover to watch for?
- **Topic 2 (control theory):** exact form of the inner-loop rate controller — plain P, PD, or
  feedforward `I·α + ω×(Iω)`? Depends on how aggressive the target maneuvers are.
- **Topic 3 (aerodynamics):** is anisotropic/body-frame drag worth the complexity for this
  project, or does the existing isotropic quadratic drag suffice? Battery voltage sag and
  rotor-rotor interaction are provisionally out of scope (no physical drone to validate against)
  — confirm or overturn.
- **Topic 4 (perception):** camera resolution / update rate target, and the realistic `num_envs`
  budget on the RTX 5060 Ti once rendering is on (needs an empirical sweep, not just a guess).
- **Topic 5 (reward/sim-to-real):** exact command-penalty weighting; whether phase 7
  (sim-to-real residual) is even reachable without a physical airframe, or should be
  descoped/simulated-only.

## Provisional architecture (from the approved plan — see phases 0-7)

See `C:\Users\Pc-Egor\.claude\plans\ok-what-if-i-cheeky-waterfall.md` for the full phase
breakdown. Summary once curriculum confirms it gets copied here as the final spec.
