# Pursuit–Evasion Roadmap

| | |
|---|---|
| **Status** | Draft v0.2, planning |
| **Created** | 2026-11-01 |
| **Scope** | Chaser and evader policy development, from the current baseline to multi-agent racing |
| **Owner** | Georgiy Dorogoy |

---

## 1. Objective

Deliver chaser and evader policies that both operate in the following regime:

| Parameter | Target |
|---|---|
| Engagement distance | ~300 m |
| Agent speed | 35 m/s |
| Disturbances | Wind, mass variation |
| Formats | 1v1 racing and many-vs-many |

## 2. Workstreams

### 2.1 Chaser

| ID | Item | Description |
|---|---|---|
| C1 | Slowing-down fix | Retrain with the evader as teacher, or fine-tune the current policy to remove the slowing behaviour. The choice depends on the root cause. |
| C2 | Speed curriculum | Train against the evader at 15 → 25 → 35 → 40 m/s (or the corresponding max speed). The evader is passed as the `7*k` observation block. At 40 m/s the evader starts moving only after the model takes over the flight. |
| C3 | Range expansion | Model-piloted distance of at least 250–400 m. Ideal: 500–750 m. |
| C4 | Randomization | Wind and mass. |

### 2.2 Evader

| ID | Item | Description |
|---|---|---|
| E1 | Speed curriculum | Train against the chaser at 15 → 25 → 35 → 40 m/s (or the corresponding max speed). The chaser starts moving only after the model takes over the flight. Chaser spawn distance: `U(33% of the model range, 99%)`. |
| E2 | Multi-chaser training | One evader vs N chasers. |
| E3 | Range expansion | Extend the operating range, in step with C3. |
| E4 | Randomization | Wind and mass. |

## 3. Phases

| Phase | Goal | Items | Exit criterion |
|---|---|---|---|
| **P0** | Chaser baseline repaired | C1 | Chaser holds speed through the pursuit. Metric and threshold to be defined. |
| **P1** | Low-speed co-development | C2 (15 → 25 m/s), E1 (15 → 25 m/s) | Both policies pass the success metric at 25 m/s. |
| **P2** | High-speed regime | C2 and E1 (35 → 40 m/s) | Both policies pass the success metric at 35 m/s. |
| **P3** | Range and disturbances | C3, C4, E3, E4 | Policies hold at 300 m or more with wind and mass randomization. |
| **P4** | Multi-agent | E2, then many-vs-many | Final dual goal met (section 1). |

Phase order is proposed and open to revision.

## 4. Dependencies

```
C1 ──> C2 ──> E1 ──> P2 ──> P3 ──> P4
                 └──> E2 ─────────────┘
```

The chaser needs a moving evader, and the evader needs a competent chaser. P1 is therefore a joint stage.

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Policy drift after unfreezing (seen in earlier PPO runs) | Regression between stages | Anchoring to a teacher; checkpoint gating between stages |
| Teacher invalid at long range | Bad distillation signal | Verify the teacher at each new range before distilling |
| Observation layout does not scale to N agents | Retraining from scratch | Fix `k` with padding and masks; decide before E2 |
| Curriculum co-adaptation (one side overfits to the other) | Brittle policies | Alternate freezes; evaluate against held-out opponents |
| 40 m/s not reachable with current drone parameters | Stage cannot be completed | Confirm the achievable max speed before P2 |

## 6. Open decisions

1. **C1 approach:** retrain with evader as teacher, or fine-tune. Requires a root-cause check first.
2. **Training order:** which side leads in P1, and whether training is joint or alternating.
3. **Multi-agent observation:** how `7*k` extends to N chasers and many-vs-many.
4. **Long-range teacher:** whether 500–750 m needs new PID/LOS gains.
5. **Wind model:** constant, gusts or turbulence, and whether the policy observes it.
6. **Mass range:** bounds, and whether the teacher is retuned for it.
7. **Success metrics:** per-stage definition, and the threshold to advance.

## 7. Change log

| Date | Change |
|---|---|
| 2026-11-01 | Initial roadmap created from planning notes |
