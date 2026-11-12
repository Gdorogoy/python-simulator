## Revised design

**1. One Isaac env, two drones, two models.**
- **Bodies per env:** one `evader` and one `threat`, both with the full rotor model. One env class hosts them.
- **Per-drone state:** the env today has one `_acro` flag, one rate controller and one set of rotor state. I'd move that into a small per-drone object: style (angle or acro), rotor state, controller and perception.
- **Learner and driver:** `learner_role` picks which drone the trainer controls. The other drone gets its actions from a **driver** inside `_pre_physics_step`, the same place the threat drones are driven now.
- **Observation per role:** each role builds its own observation, so a frozen policy driver can compute its own inputs. The trainer only receives the learner's.

**2. Training stages.**
- **Stage 1:** each role trains against a dummy.
  - The evader trains against the LOS threat, as now.
  - The acro threat trains against a scripted evader. That dummy doesn't exist yet.
- **Stage 2:** each role trains against a weaker checkpoint of the other. The driver loads a frozen policy from a pool of older checkpoints.

**3. Perception: placeholders only, torch, no numpy.**
- **Interface:** `estimate(observer, observed) -> (pos, known, source)`.
- **Stub:** the placeholder returns the truth with `source = FOV`. Priority (FOV, then base, then prediction) is a documented slot, with no filter written.
- **No numpy twin:** the viewer therefore can't use a numpy env for these runs. Playback would come from tracks recorded out of Isaac. That's a later decision.

**4. Jam.**
- **Injection:** the env overwrites the target coordinates with something like `(0,0,0)` or a random far point, for a random span of time, in random envs.
- **Detector:** a plain rule in torch. If the target estimate jumps by more than a threshold in one step, the jam flag goes to 1.
- **Threshold:** the drone moves at most about 1 m per 60 Hz step, so anything above roughly 20 m can only be a jam.
- **Autonomous mode:** the target slot holds the last good value (placeholder for prediction), and the mode flag goes to 1.

**5. Allies.**
- **Entries:** each ally is `[x, y, z, alive, id]`, self-reported with the true position.
- **1v1:** capacity is 0 for now, so the section is zero-width and nothing changes.
- **Later:** raising the capacity just adds columns.

**6. Observation with fixed-capacity sections.**
```
angle: [23 base][2 timer][threat slot: pos3 + known1 + source][ally x A: 5 each][flags: link, jam, autonomous]
acro : [23 base, target = assigned evader's estimate][ally x A][flags]
```
This is the same one-time conversion as before, with the loader column map.

**7. Kill rule:** when a threat takes down its target, it disappears from the sim.

## Order of work
1. Perception stubs and the jam rule, in torch, with tests.
2. The layout conversion and the loader column map (it also fixes the timer-column mismatch).
3. Move per-drone state into the per-drone object; the existing angle behaviour must not change.
4. The driver interface: scripted LOS, scripted evader, frozen policy.
5. The acro threat role against the scripted evader.
6. Stage 2: the checkpoint pool.
