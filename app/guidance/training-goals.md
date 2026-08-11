
---

*DONE*


**Phase 0** 
description: teaching the drone how to hover in place without falling or drifting
goal: the drone can stay within 0.3m of spawn point `(0,0,0.5)` for the full episode length, velocity near 0
why: every later phase assumes the drone can already resist gravity — without this, all movement rewards get drowned out by free-fall, like you saw with the flat 165k-step plateau
additional info: target = spawn position exactly. No `moving_away_cap` needed really — replace with a "drift radius" termination (e.g. `dist > 1.0` = fail). Wind=0, mass fixed.

---

**Phase 1.0 (left)**
description: teaching the drone to translate in -x (or +y, pick one axis as "left" and be consistent) while holding altitude
goal: reach a target 2-5m to the left of spawn, e.g. `(-3, 0, 0.5)`
why: isolates roll-only control (or pitch, depending on your axis convention) from the rest — single degree of freedom, cleanest possible learning signal
additional info: target_range widened in 3 bands per your earlier plan: `(0.5,1.5)`, `(1.5,3)`, `(3,5)` meters. Warm-start from Phase 0 checkpoint.

**Phase 1.1 (right)**
description: teaching the drone to translate in +x, mirror of 1.0
goal: reach a target 2-5m to the right of spawn, e.g. `(3, 0, 0.5)`
why: right isn't guaranteed to be symmetric with left — motor asymmetries, mixer matrix quirks, or drag coefficients could make one direction genuinely harder; don't assume symmetry, verify it
additional info: warm-start from Phase 1.0, not Phase 0 — likely converges much faster since roll-control fundamentals transfer

**Phase 1.2 (backward)**
description: teaching the drone to translate in -y while holding altitude
goal: reach a target 2-5m behind spawn, e.g. `(0, -3, 0.5)`
why: pitch-axis control, the other half of horizontal movement — forward/backward uses a different torque channel than left/right, so this is a genuinely new skill, not just a relabeled version of 1.0
additional info: warm-start from Phase 1.1

**Phase 1.3 (forward)**
description: teaching the drone to translate in +y, mirror of 1.2
goal: reach a target 2-5m ahead of spawn, e.g. `(0, 3, 0.5)`
why: same asymmetry-verification logic as 1.1 — don't assume forward = backward difficulty
additional info: warm-start from Phase 1.2. **Merge candidate:** consider combining 1.2+1.3 into one stage with `target_mode="random"` restricted to the y-axis only, since it's the same skill in both directions — may save you a full training run.

---

**Phase 2.0 (northeast — +x,+y)**
description: teaching the drone to combine roll+pitch simultaneously to move diagonally
goal: reach a target at `(3, 3, 0.5)`
why: this is the first stage requiring the policy to coordinate two action dimensions at once instead of one — if 1.0-1.3 worked but this fails immediately, that tells you something about how your action space or reward handles simultaneous multi-axis commands, not a totally new skill from scratch
additional info: warm-start from **both** 1.1 (right) and 1.3 (forward) if you want to test transfer — practically, just pick 1.3's checkpoint since it's your most recent. Watch whether this converges nearly as fast as a single-axis stage; if yes, you can likely collapse the remaining 2.x sub-phases into one "diagonal, randomized quadrant" stage instead of 4 separate ones.

**Phase 2.1 (northwest — -x,+y)**
description: same as 2.0, mirrored quadrant
goal: reach a target at `(-3, 3, 0.5)`
why: verify coordination generalizes across quadrants, not just the one trained
additional info: warm-start from 2.0

**Phase 2.2 (southeast — +x,-y)**
description: same as 2.0, mirrored quadrant
goal: reach a target at `(3, -3, 0.5)`
why: same verification purpose as 2.1
additional info: warm-start from 2.1. **If 2.0-2.2 all converge fast (few checkpoints), stop here — skip 2.3 and go straight to a single randomized-quadrant stage**, since you'll have already demonstrated the skill generalizes.

**Phase 2.3 (southwest — -x,-y)**
description: same as 2.0, last mirrored quadrant
goal: reach a target at `(-3, -3, 0.5)`
why: completes quadrant coverage
additional info: warm-start from 2.2

---

**Phase 3.0 (up)**
description: teaching the drone to increase thrust to climb while holding x/y position
goal: reach a target directly above spawn, e.g. `(0, 0, 3.0)`
why: pure vertical control was implicitly covered by the hover-offset fix, but this stage tests *commanded* altitude change, not just holding still — different from Phase 0 because now `progress` reward depends on thrust magnitude changing deliberately, not staying constant
additional info: warm-start from Phase 0 (not 2.x — this is a separate branch of the skill tree, doesn't need horizontal skills first)

**Phase 3.1 (down)**
description: teaching the drone to decrease thrust to descend in a controlled way (not free-fall) while holding x/y
goal: reach a target below current altitude but above ground, e.g. `(0, 0, 0.2)`
why: descent is a genuinely different failure mode than ascent — overshoot here means crashing into the ground (`pos[2] < 0.0` = instant "oob" termination), so the risk profile is asymmetric even though the mechanics look similar
additional info: warm-start from 3.0. Watch `oob` count specifically in diagnostics here — if it's nonzero, that's the ground-crash failure mode surfacing for the first time.

---

**Phase 4.0 (horizontal + z combined, e.g. northeast-and-up)**
description: teaching the drone to combine lateral movement with simultaneous altitude change
goal: reach a target like `(3, 3, 2.5)` — diagonal in x/y AND changed in z, all at once
why: first stage exercising all 4 action dimensions together — full 3D movement is qualitatively different from "2D plane + separate altitude," since thrust now has to serve both "stay up" and "change altitude" roles at the same time as roll/pitch are doing translation
additional info: warm-start from whichever of 2.x/3.x converged best — likely 2.0 (or your merged randomized-quadrant version) plus 3.0

**Phase 4.x (all directions, combined)**
description: same as 4.0, randomized across all octants (8 combinations of ±x, ±y, ±z)
goal: reach targets randomly sampled across a full 3D shell around spawn, e.g. radius 3-5m, any direction
why: generalizes 4.0 the same way 2.1-2.3 generalized 2.0 — but here, given 2.x likely collapses into one randomized stage, I'd recommend skipping straight to this rather than writing out 8 discrete octant sub-phases; you have enough evidence by this point (from 2.x's generalization) that direction-specific sub-phases add little
additional info: warm-start from 4.0. **This effectively is your "phase 4.1 through 4.7" — one randomized 3D stage rather than 7 more discrete ones.**

---

**Phase 5.0 (chase, near-range params — from your 1.x scale)**
description: teaching the drone to reach a target using the full learned movement repertoire, at close range (2-5m), still static
goal: reliably hit (`dist < 0.3`) a randomly placed static target within 2-5m of spawn, in any direction/altitude
why: this is the first "real" chase stage — everything before was isolated skill-building, this is the actual composed task, just still easy-range
additional info: warm-start from 4.x. This is close to what you're already running now, just with your target now properly randomized instead of fixed at `(1,1,1)`

**Phase 5.1 (chase, mid-range — from your phase 2 diagonal scale)**
description: same chase task, wider range band
goal: reliably hit a target within 5-12m of spawn
why: tests whether the learned policy generalizes to distances it hasn't directly trained on, or whether it needs explicit exposure at this range
additional info: warm-start from 5.0

**Phase 5.2 (chase, far-range — your phase 3 vertical-inclusive scale)**
description: same chase task, wider range band, now including larger altitude differences
goal: reliably hit a target within 12-20m of spawn
why: pushes into range where your earlier flagged obs-scaling concerns start to matter — watch `POS_SCALE`/`DIST_SCALE` headroom here
additional info: warm-start from 5.1. If reward curves start flattening oddly here, that's your first real signal to reconsider the normalization approach (log-distance) before pushing further.

**Phase 5.3 (chase, extended range — your phase 4 combined scale)**
description: same chase task, full range up to whatever ceiling you set before long-range (pre-800m) work begins
goal: reliably hit a target within 20-40m of spawn
why: last "cheap" range extension before you hit the long-range problems (reward sparsity, obs saturation) flagged earlier — good checkpoint to pause and evaluate before going further
additional info: warm-start from 5.2

---
**Phase 6.0 (wind only)**
description: teaching the drone to chase successfully under wind disturbance, mass/inertia still fixed
goal: maintain roughly Phase 5.3's hit-rate (some degradation expected) across the 20-40m range, with sample_wind_conditions active and mass_scale still pinned to 1
why: isolates disturbance-rejection as its own skill before mixing it with the mass-uncertainty problem — if performance drops here, you know it's specifically wind response, not confused with mass effects
additional info: warm-start from 5.3. Start with a narrow wind band (e.g. low magnitude only) in an early sub-pass, then widen it — same expanding-range pattern as your distance curriculum. Watch whether oob/attitude terminations start appearing that weren't present in 5.3 — that's wind pushing the drone into failure modes it never had to handle before.

**Phase 6.1 (mass/inertia only)**
description: teaching the drone to chase successfully with randomized mass and inertia, wind still off
goal: maintain roughly Phase 5.3's hit-rate across 20-40m range, with mass_scale randomized per episode, wind fixed at 0
why: isolates mass-uncertainty as its own skill — this is a fundamentally different problem than wind (wind is an external force to counteract, mass uncertainty changes the drone's own response to its actions, i.e. its effective hover_thrust and control authority shift per episode)
additional info: warm-start from 5.3 (not 6.0 — this is a separate branch, same logic as Phase 3 not depending on Phase 2). Note your hover-thrust offset fix depends on self.config.mass computed fresh each episode — verify that's still correctly recalculated per-reset() here, since this is the first stage that actually exercises that per-episode variability.

**Phase 6.2 (wind + mass combined)**
description: teaching the drone to chase successfully with both disturbances active simultaneously
goal: maintain acceptable hit-rate (expect the largest single drop yet) across 20-40m range, both sample_wind_conditions and randomized mass_scale active together
why: this is the first stage where two independently-learned robustness skills have to compose at the same time — real-world conditions never isolate one variable, so this is your actual target operating condition, but it's only diagnosable now because you already know what "wind alone" and "mass alone" look like from 6.0/6.1
additional info: warm-start from whichever of 6.0/6.1 is stronger, or optionally train two lineages and pick the better one empirically. If 6.2 collapses badly despite 6.0 and 6.1 each working — that combination effect itself is useful information, tells you the failure is interaction-specific, not either disturbance alone.


[//]: # ( ~somethingf ~ do mark as done)