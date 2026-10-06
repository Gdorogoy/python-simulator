# RTL handoff training: full guide

Covers the drone "hit the target" project as it stands: the model, PPO, the PID teacher (RTL), the reward (qv2),
the controller-to-policy handoff stage, evaluation, how to read the logs, what each run taught us, and the plan.
Numbers quoted as "default" are the script defaults at the time of writing (`app/training/rtl_train_isaac.py`);
check `--help` if in doubt.

---

## 1. The big picture

**Task:** a drone must hit a target (come within **0.25 m**). The final system is **two-phase**:

1. A **controller** (PID) flies the long, boring part (most of the distance).
2. At a **handoff point** the **learned policy** (a neural net) takes over and does the last stretch:
   braking/aiming at speed and hitting. If it misses, it must **turn around and come back**.

**Why two-phase and not one network for 2 km:** a straight cruise is already solved by a controller. Training a
network over long distances is expensive (long episodes, almost no learning signal in straight flight, observation
scales tuned for tens of metres) and risks being worse at it. The policy only needs to be good at the terminal phase.
Speed is dominated by the cruise leg, so the lever for speed is a **fast cruise controller that doesn't brake early**,
plus a handoff distance chosen from physics (braking distance is about v^2 / (2a)).

**Files that matter**

| File | What it is |
|---|---|
| `app/training/rtl_train_isaac.py` | the training script (PPO + PID teacher + handoff stage + Isaac eval) |
| `app/reward_functions/reward_qv2.py` | the reward (qv1 + miss-and-hover penalty/termination) |
| `deprecated/reward_functions/reward_qv1.py` | the previous reward (deprecated, kept for reference) |
| `app/guidance/train.py` | `ActorCritic`, `compute_gae`, checkpoint loading |
| `app/environmental/base_drone_env_isaac.py` | the Isaac env (`_reset_idx`, observations, actions) |
| `app/control/torch_pid.py`, `app/control/two_phase.py` | batched PID (the teacher) and the two-phase deployment agent |
| `app/guidance/record_run.py` | replay tool; runs in the **numpy** `BaseDroneEnv`, NOT Isaac, old reward |

---

## 2. Vocabulary (the things that got mixed up)

- **Step:** one policy decision. The policy runs at 60 Hz; each step the sim advances 4 physics substeps (240 Hz).
- **Episode:** one run, from spawn until it ends (hit, crash, hover-termination, or timeout).
- **Chunk:** a fixed number of steps per env (default 256) across ALL envs in parallel, followed by one PPO update.
  A chunk is NOT a set of episodes: episodes do not line up with chunk boundaries and may span several chunks.
- **Env / num_envs:** thousands of drones simulated in parallel on the GPU. One chunk = steps x num_envs samples.
- **Rollout:** the data collected in one chunk (observations, actions, rewards, log-probs, values).
- **Actor / policy:** the network that picks actions. **Critic / value function:** predicts expected future reward.
- **Return:** discounted sum of future rewards. **Advantage:** how much better an action was than expected.
- **Teacher / student:** teacher = PID; student = the network.
- **Distillation:** transferring the teacher's behaviour to the student (here with an extra loss term).
- **Handoff:** the moment control passes from the controller to the policy.
- **Prefix:** the first metres of a training episode flown by the PID (the policy "observes the controller").
- **Bucket:** a speed range used to sample the spawn speed (b0, b1, b2 in the logs).

---

## 3. The model

**Observation (23 numbers):** position (symlog scaled), velocity / 10 m/s, orientation quaternion, angular velocity,
rotor speeds, vector to target (symlog), distance (symlog), sin/cos of yaw error.
Note: it contains **absolute position**. That is a future problem (see section 13).

**Action (4 numbers):** a thrust delta and 3 torques, clamped to the env limits.

**`ActorCritic` (`app/guidance/train.py`)**
- `shared`: a small MLP (default 4 layers, 64 wide, tanh) giving features.
- `actor_mean`: a linear layer on those features -> the mean of the action distribution.
- `actor_log_std`: a learned parameter -> the noise size. It is squashed between a min and a max
  (the script uses log-std in [-3.5, -2.0], i.e. std about 0.030 to 0.135).
- The raw action goes through **tanh** and is rescaled into `[action_low, action_high]`. Reasons: the output is always
  inside the allowed motor range (the policy cannot command impossible values), and the change of variables is
  accounted for in the log-probability.

**`RTLActorCritic`** adds a **separate critic network** (`value_net`, default 2 layers x 128).
- Why separate: the value loss then never changes the actor's features, and the critic can train alone while the
  actor is frozen (critic warmup). The BC checkpoint also has no critic at all.
- The inherited `critic_head` is unused.

**Mean vs std**
- During rollouts the action is `mean + std * noise` (exploration). Larger std = more exploration (and higher entropy).
- At evaluation / deployment you use the **mean** only (deterministic).
- `--init-std` is the **starting noise size**. It is NOT an entropy value. (`--ent-coef` is the separate weight on an
  entropy bonus; default 0, so nothing makes the noise grow by itself.)

**What is saved in `best.pt` / `model_*.pt`:** the actor only (shared layers, mean head, log-std). NOT the critic and NOT
the optimizer. `latest_full.pt` has everything but the loader flag `--bc-checkpoint` cannot read it.
Consequence: a new run from `best.pt` needs **critic warmup again**, with a fresh optimizer.

---

## 4. PPO in plain words

PPO = collect experience with the current policy, then improve the policy a little using that experience.

**One chunk:**
1. Run the current policy in all envs for N steps (with noise). Store everything.
2. Compute **advantages** with GAE (below).
3. Do several passes (epochs) over the data in minibatches, updating actor and critic.
4. Throw the data away and repeat with the new policy (on-policy).

**Actor loss:** make actions with positive advantage more likely, and negative ones less likely, but only within a
**clip** (`--clip-eps`). **Critic loss:** regress the value prediction toward the actual discounted return.
The critic is NOT a grader: the reward comes from the environment. It predicts expected future reward from a state,
and the advantage is "what really happened minus what the critic expected".

**Why several passes over the same data:** experience is expensive. **What stops it going too far:** the clip (and an
early stop when the measured KL gets too large). Without it the policy would move away from the one that collected the
data and the old data would stop being valid.

**GAE (`compute_gae`):** a hit comes many steps after the actions that caused it. Raw per-step rewards would only
credit the last action. GAE spreads credit backwards using `gamma` (discount) and `lambda` (bias/variance trade-off),
computed backwards in time.

**`done`:** cuts the chain at an episode boundary, so value and advantage do not leak from one episode into the next.
**Timeouts are treated as real terminals** in this script (no bootstrap), so "never hitting" has a real cost and the
timeout penalty actually matters. (`--bootstrap-timeouts` restores the usual time-limit bootstrap.)

**Advantage normalisation:** advantages are normalised per batch so the update size does not depend on the reward scale.

**Reading the PPO numbers in the log**
- `kl` / `clip_frac`: how much the policy moved in the update. Large = unstable. Early stop kicks in above the limit.
- `vloss`, `ev` (explained variance): how well the critic predicts returns. `ev` near 1 = good critic, near 0 = useless.
- `agree`: mean gap between student action and teacher action.

---

## 5. RTL: teacher-student with phases

RTL here = start from a pretrained student, keep a PID teacher as an anchor, and unfreeze the network in stages.

**Three sources of knowledge in the final policy**
1. **Start weights:** the BC/DAgger clone of the PID, or a previous run's `best.pt` (`--bc-checkpoint`).
2. **Teacher distillation during training:** every step the PID is asked what it would do in the state the student is
   in. That is a label. The actor loss gets `distill_coef * mean( weight * (tanh(mean) - teacher_action)^2 )`.
   The student always flies. The PID only really controls the drone in the **handoff prefix**.
3. **The policy's own reward-driven experience** (PPO).

**Distillation weight**
- `--distill-start` -> `--distill-end`: the coefficient decays linearly over `--distill-decay-chunks`
  (counted in chunks where the actor trains). High = follow the PID strongly. Low = mostly RL.
- Per-sample weight fades with speed (`--distill-fade-speed`, default 6 m/s): at high speed a hover PID commands
  **braking**, which fights "hit the target", so the teacher is ignored there.
- **Guard:** if the training hit rate falls a margin below its recent best, the coefficient is multiplied up (capped);
  otherwise the boost decays. Sensible when the teacher is good.
  **It can backfire when the teacher is worse than the student** (the PID gets ~0.65, the student ~0.9): it drags
  the student toward the PID's braking and parking. In the 3-30 m run the boost reached its max.

**Danger of a strong anchor for too long:** the student can't beat the teacher and inherits its flaws.

**Phases (by chunk)**

| Phase | What happens |
|---|---|
| settle | rollouts only, no updates. Measures the untouched start policy. |
| critic_warmup | actor FROZEN (still acts and collects data); only the critic learns. Needed because the critic has never been trained, and PPO with a random critic produces garbage advantages. |
| head_only | the action head and log-std train; the trunk stays frozen. |
| full | everything trains. |

Each newly unfrozen group gets a learning-rate ramp. `--warmup-chunks` (default 16) controls the critic phase. With a
good loaded actor, about 8 is enough; check that `ev` is above ~0.9.

**"Transfer"** in this setup is the initialisation (from BC or a previous best) and the teacher distillation.
It is not something that "happens every chunk": within a run, each chunk is simply continued training.

---

## 6. The reward (reward_qv2)

Per policy step:

| Term | Meaning |
|---|---|
| hit | +10, terminal, distance < 0.25 m |
| progress | `k * (log1p(prev_dist/eps) - log1p(dist/eps))`, potential difference (not gamma scaled) |
| step | small constant cost per step, so arriving sooner is better and hovering is worse than hitting |
| timeout | -3 one-off on the truncation step (a true terminal) |
| crash | -5 terminal: out of bounds, underground, NaN, roll > 65 deg, pitch > 80 deg |
| tumble | small penalty on angular velocity squared (capped) |
| tilt | soft wall in front of the roll/pitch limits |
| **hover (new)** | stalled = off-target (dist > hit radius) AND speed < 0.3 m/s. After 1 s of continuous stall: -0.05 per step. After 3 s: terminate with -5. |

**Why the hover rule exists:** the policy overshot, braked, and parked ~0.9 m from the target with zero action and
zero speed for minutes. Nothing punished it enough.

**Why a state check, not an action check:** a zero action is also a legitimate brake or hover in mid-flight.
"Slow and not on the target" is the real symptom. The counter resets as soon as the drone moves, so
**brake-and-return is free; only parking is punished.** Moving away from the target is NOT punished (the drone
must be free to turn around).

**Why progress is a potential difference:** the sum telescopes. Going back and forth between two distances gains
nothing, so there is no loop to farm. Paying "reward for being close" every step can be farmed by hovering.

Hits beat the other terms on purpose (progress available from 10 m is about 5; hit is 10). Hard terminations take
priority over a same-step hit.

Note the incentive: a parked drone (-5 plus the per-step hover cost) is worse than a crash (-5). It should never be
chosen deliberately. If it still parks, it is because it is stuck, not because it is rewarded.

---

## 7. The handoff stage (`--handoff`)

Every training episode starts the way a real handoff leaves the drone, not from rest.

- **Speed:** `frac * --max-speed` with frac drawn from buckets: 10-25% (20% of episodes), 25-75% (40%),
  75-100% (40%).
- **Velocity:** toward the target plus a random sideways component (up to `--lateral-frac` of the speed), and a random
  roll/pitch (up to `--handoff-tilt-deg`). A real handoff carried ~1.8 m/s lateral and +-20 deg tilt.
- **Distance:** the remaining distance at handoff is drawn from U(`--distance-low`, `--distance-high`). The target is
  placed that far plus the prefix length from the spawn. The PID covers a bit less than the nominal prefix (it brakes),
  so the real handoff is slightly farther than drawn.
- **Prefix:** the PID teacher flies the first 2-5 m at that speed (`--prefix-low/high`), then the policy takes over.
- **Why the prefix steps are masked from the policy gradient:** PPO compares the new and old probability of an
  action. Controller-chosen actions were not sampled from the policy, so those probabilities mean nothing. The critic
  still learns from them because it only needs states and returns.

**Deployment mismatch to remember:** in training the prefix is flown by the braking hover PID. If deployment uses a
different controller (e.g. a cruise controller), the policy sees a different handoff than it trained on. The prefix
controller should be the deployment controller.

**Speed vs distance (physics):** at high speed the policy needs room to brake. With 35 m/s and 3-10 m to go there is
under a second, so it is unwinnable (even the PID fails). Pick `--max-speed` and the handoff distance together.
Braking distance is about v^2 / (2a).

---

## 8. Evaluation

**Where:** inside Isaac only (`isaac_eval`). The old numpy `BaseDroneEnv` evaluation was removed. (The replay tool
`record_run.py` still uses the numpy env and the OLD reward: it is not a test of qv2 or the Isaac policy.)

**How:** reset ALL envs, run the deterministic policy (mean action), and score each env's FIRST episode.
- Each episode has its own **time budget from its start distance** (`steps_for_dist`: about 1 s per metre, 7.5 s floor).
  Running out of time = `timeout` (counted as a failure).
- `--eval-steps` is an optional hard cap (0 = none).
- Outcomes: hit, hover, oob, roll, pitch, timeout, unfinished. Plus hit and crash rate per speed bucket (b0/b1/b2).
- The PID baseline runs the same way every `--pid-eval-every` chunks.
- `best.pt` is the checkpoint with the highest eval hit rate (only on eval chunks).
- Eval runs every `--eval-every` chunks (default 5).

**The burn-in:** after an eval all envs are in lock-step. Without correction, the next chunk contains only fast-ending
episodes (e.g. hit 0.98 / hover 0 / hit time 3 s instead of 0.91 / 0.07 / 5 s) and produces spikes. So after each eval
the script runs `--eval-burnin` noisy steps (default 600) and discards them, plus it zeroes the episode counter.
**A full `env.reset()` randomises each env's elapsed-time counter**, which made fake early timeouts. That is fixed.

---

## 9. Reading the console line

```
[rtl] 59/61 full  train_hit=0.903 (base 0.851) eval=0.77 [b0/b1/b2 ...] pid=0.24 timeout=0.000 hover=0.028 stall=0.010 agree=0.125 d_coef=0.130 kl=0.0013 vloss=0.32 ev=0.95 183k step/s
[rtl]   eval endings: hit=0.765 hover=0.006 oob=0.022 roll=0.030 pitch=0.005 timeout=0.000 unfinished=0.172 | crash by bucket 0.04/0.03/0.09
```

| Field | Meaning |
|---|---|
| `59/61` | chunk number / total chunks |
| phase | settle / critic_warmup / head_only / full |
| `train_hit` | of the episodes that ENDED in this chunk, the share that hit. Noisy policy, training envs. Unfinished episodes are not counted. |
| `base` | average `train_hit` during critic warmup, i.e. the untouched start policy |
| `eval` | hit rate of the deterministic policy on freshly reset envs (first episode each) |
| `[b0/b1/b2]` | eval hit rate per speed bucket (slow / medium / fast) |
| `pid` | the PID baseline in the same eval |
| `timeout`, `hover` | share of training episodes ended by timeout / hover termination |
| `stall` | share of steps spent 0.25-0.75 m from the target, nearly stopped (diagnostic) |
| `agree` | mean student-teacher action gap |
| `d_coef` | current effective distillation weight (schedule x guard boost) |
| `kl`, `vloss`, `ev` | PPO / critic health |

**Why `train_hit` and `eval` differ:** different denominators and conditions. `train_hit` ignores unfinished episodes
and uses the noisy policy. `eval` counts unfinished as non-hits (before the per-distance budget, this was ~17%).
Rule: compute "hit among finished" = hit / (1 - unfinished) to compare them. Also compare `timeout`, then
noisy vs deterministic, then how the envs were reset.

---

## 10. What each run taught us (history)

1. **Parking at 0.91 m** (first test): the policy overshot, braked, and sat there with zero action. -> added qv2's hover
   rule, and the handoff stage.
2. **35 m/s with 3-10 m** (`rtl_handoff_v1_3_10`): train hit stuck at about 0.40, out-of-bounds 45%. The smoke test
   showed crash rate by bucket 0 / ~0.6 / ~0.9 and the PID only ~0.2: the fast buckets are physically unwinnable at that
   distance. -> lower max speed (12-18 m/s) or longer handoff distance.
3. **Spikes in the graphs:** caused by the eval reset (synchronised envs), see section 8. Fixed with the burn-in.
4. **Fake timeouts after eval:** the full reset randomised the elapsed-time counter. Fixed.
5. **3-30 m run from the 3-10 m `best.pt`:** eval fell from about 0.77 to 0.67 over training and hover rose from 11% to
   20%. It was NOT reward hacking: mean return fell too. Likely causes: extrapolating from a 3-10 m policy to 3-30 m
   (hover 11% at chunk 0 already), exploration std at its floor (0.03) so a parked policy cannot learn to leave, and the
   anchor decaying. `best.pt` was effectively the starting policy. -> use wider std, a higher anchor floor, and widen the
   distance range in stages.
6. **Replay parking at 1.4 m / 3.5 m:** came from `record_run`, which has no hover termination and uses the old reward.
   The likely reasons are an out-of-distribution state (handoff slower than anything seen in training) or a gap between
   the numpy sim and Isaac. Still to be tested by replaying the same checkpoint inside Isaac.
7. **Unfinished 17% in eval:** the fixed 600-step cutoff was too short for long distances. Now per-distance budgets.

---

## 11. Common hyperparameters and what they mean

| Flag | Role |
|---|---|
| `--max-speed` | 100% of the spawn speed distribution (m/s). Must be survivable at the chosen distance. |
| `--distance-low/high` | remaining distance at handoff |
| `--init-std` | starting exploration noise (action space), clamped to the log-std range |
| `--distill-start/end/decay-chunks` | teacher anchor weight schedule |
| `--distill-fade-speed` | speed above which the teacher is ignored |
| `--warmup-chunks`, `--head-chunks`, `--settle-chunks` | phase lengths |
| `--lr-actor`, `--lr-critic` | learning rates (actor much smaller) |
| `--bc-checkpoint` | actor weights to start from (BC or a previous best) |
| `--eval-every`, `--eval-burnin`, `--eval-steps`, `--pid-eval-every` | evaluation behaviour |
| `--num_envs`, `--total_timesteps` | with more envs each chunk is bigger, so the same total gives fewer updates |

**Starting a new run from `best.pt` is fine-tuning** (warm start). Take care with: selection bias (the checkpoint that
looks best on a noisy eval is biased upward; keep a held-out test), drift (re-check the older ranges), and the fact
that restarting is only useful if eval actually improves (compare to the chunk-0 eval).

Example (start from a previous best at a harder setting):
```
python -u app/training/rtl_train_isaac.py --headless --num_envs 8192 --handoff --max-speed 18 \
    --distance-low 3 --distance-high 50 --checkpoint-dir runs/<NEW_NAME> \
    --distill-start 0.25 --distill-end 0.1 --init-std 0.1 \
    --bc-checkpoint runs/<previous>/best.pt --warmup-chunks 8
```
Use a NEW checkpoint dir each time; reusing one overwrites metrics and weights.

---

## 12. Quick answers to the self-check (corrected)

1. Separate critic = its own layers, value loss never changes the actor's features; lets the critic train while the actor is frozen.
2. Mean = chosen action (used at eval); std = exploration noise size.
3. tanh keeps actions inside the motor range (and needs the log-prob correction); it is not mainly about gradient explosion.
4. `best.pt` = actor only. New run => critic warmup + fresh optimizer.
5. Actor loss: raise probability of positive-advantage actions (clipped). Critic loss: regress to the actual return.
6. Several passes reuse expensive data; the clip / KL stop keeps the policy near the one that collected it.
7. GAE spreads delayed credit back over earlier actions.
8. `done` cuts value/advantage across episode boundaries; timeouts are real terminals so never-hitting has a cost.
9. The PID labels the student's states each step (distillation); it only flies the handoff prefix.
10. Start weights + PID distillation + own RL experience.
11. A strong anchor caps the student at the teacher's level and copies its flaws.
12. Hover rule: state-based because zero action is also a valid brake.
13. Potential difference: no farmable loop.
14. Speed distribution and distance (plus lateral velocity and tilt).
15. Eval is deterministic, on freshly reset envs; train is noisy; the reset caused synchronised episodes, hence spikes.
16. Check `unfinished`/`timeout` share, then noisy vs deterministic, then the reset.

---

## 13. Plan / known gaps

- **Cruise controller + speed-gated handoff.** Replace the braking hover PID with a speed-holding controller (e.g. a
  carrot target) and hand off only when `dist <= switch_dist` AND `v_min <= speed <= v_max`. Use the same controller
  as the training prefix.
- **Low-speed bucket (0-10%, including standstill)** as a safety net. Not yet added.
- **Post-pass / overshoot starts** so the policy trains the turn-around after a miss. Not yet added.
- **Recovery metric:** of episodes that miss on the first pass, how many hit later, and how long it takes.
- **Isaac replay of a checkpoint** from a chosen handoff state, to separate "policy gap" from "numpy-sim gap".
- **Resume flag** (`--resume latest_full.pt`) to continue a run exactly (critic, optimizer, chunk counter). Not yet added.
- **Distance curriculum:** widen the range in stages (e.g. 3-10, then 3-20, then 3-30/50), each from the previous best.
- **Absolute position in the observation is a problem** for far handoffs and for a camera drone. Make the observation
  target-relative (bearing, range, own velocity and attitude) and re-centre states.
- **Camera plan:** CNN detects the target -> bearing (accurate at any range); stereo range only when close (error
  grows with distance squared); a tracker/filter fuses these with own position and velocity into a target estimate with
  an uncertainty and a "visible" flag (keeps predicting after the target leaves the view). The RL policy sees the
  estimate, not pixels. Train with noise matching the sensors (bearing noise, range noise growing with distance,
  1-3 frames of latency, dropouts).

---

## 14. Self-quiz (answer without looking)

1. What is the difference between a step, an episode, and a chunk? What is `train_hit` computed over?
2. Which parts of the model are saved in `best.pt`? What does a new run from it have to relearn?
3. What does the critic predict, and how does PPO use it?
4. What does GAE do and what does `done` do inside it?
5. Who controls the drone in a training episode, step by step? Where does the teacher's action go?
6. When is a strong teacher anchor good, and when bad?
7. Why is the hover check based on state and not on the action?
8. Why is progress a potential difference?
9. Why are prefix steps masked from the policy gradient but not from the critic?
10. Why can `train_hit` and `eval` differ even when nothing is wrong?

---

## 15. Updates after this guide was first written (2026-10-05)

These supersede the older descriptions of the handoff stage in sections 7 and 8. Full detail and numbers: `PROJECT_DEFENSE_GUIDE.md`
Part 14. Proposed next version: `alt_model.md`.

- **Spawn-leg mode** (`--spawn-dist-low/high`): the drone spawns at rest and a scripted carrot-chase controller flies a ramp -> cruise ->
  ramp-down profile along the line. The legacy short-prefix mode is used when those flags are omitted.
- **Handoff distance** `h` is drawn from the model's range (`--distance-low/high`), not from the spawn range.
- **Handoff trigger is position-based**: real distance <= h, or the whole planned leg has been flown along the line, with a generous
  fallback timer. Once handed off, an env stays handed off. (The timer-only trigger handed off 20.6 m too far on average.)
- **Ramp acceleration** `--teacher-accel` defaults to `0.5 * g * tan(max_tilt)` (~1.52 m/s^2), not 3.0 (the physical limit).
- **Leg elevation clamp** `--leg-max-elev-deg` (default 15): downward legs would end underground; steep climbs crawl.
- **Re-centring** (on by default, `--no-recenter` disables): the spawn is shifted back so the *handoff point* is at x=y=0, because the
  observation contains absolute position. Same start policy: eval 0.09 without it, 0.82 with it (150-250 m spawn).
- **Handoff perturbation** also patches the observation (velocity, quaternion, angular velocity, yaw error), not only the velocity.
- **Eval** now has per-distance time budgets (`--eval-steps` is only an optional cap), a post-eval burn-in (`--eval-burnin`), zeroes
  `episode_length_buf` after a full reset, and prints a **`handoff check`** line (real distance/speed at handoff vs the sampled h/bucket).
- **Staggered starts (final form, 2026-10-05).** In training, every *full* reset (the initial one and the one after each eval) starts each env at a
  random point of its controller leg, **uniform in time** (not in distance), already moving at the profile speed there (`u._stagger_next`). Eval does not
  stagger (it must score true first episodes from the spawn point). History: a first version staggered by *distance*; because the drone covers ground fast while
  cruising, most envs then had little time left and ended (and restarted) together, and since every eval resets all envs (every 5 chunks) the saw-tooth in
  episodes-ended-per-chunk (2465, 1878, 1097, 444, 257, then 2359 ...) came straight back. Time-uniform removes it (episodes ended per chunk stayed 32-97 across
  three evals in the smoke test).
- **Guard (changed):** it is now driven by the **eval hit rate**, not the chunk-level `train_hit`. The untouched start policy's eval (settle chunk) sets the bar
  (`best_eval_hit` in `metrics.csv`, formerly `best_train_hit_ma3`); at each eval chunk, if eval < best - `--guard-margin` the boost is multiplied (cap
  `--guard-boost-max`), otherwise it decays x0.9 and the bar rises with new bests. `--guard-min-episodes` (default 4 % of envs, min 50) only gates the baseline and the abort
  check now; chunks with no finished episode report NaN rates.
- **`--anchor-checkpoint PATH` (new):** a frozen copy of those actor weights is the distillation teacher INSTEAD of the PID (label = `tanh(mean)`, no speed fade).
  The PID is still queried each step (integrators, the legacy short prefix leg, and the PID baseline in eval). At the start `agree` is 0.000 (student = anchor).
  Scale note and a measured result: a frozen-best anchor stays close to the student (`agree` ~0.01 -> squared error ~1e-4), so its loss is tiny next to the old PID-era
  loss (`agree` ~0.3). I expected the weak weight to be "almost no constraint" and therefore too weak; **the test showed the opposite** (1024 envs, 150-250 m spawn, 23 chunks,
  start policy = anchor = `v5_3_50_18ms/best.pt`, one seed each, eval = first episodes on fresh envs, so +-1.2 points):

  | anchor weight (effective) | eval hit: start -> end | eval hover: start -> end | `agree` at the end |
  |---|---|---|---|
  | `--distill-start 0.25 --distill-end 0.1` (~0.2) | 0.787 -> **0.832** | 0.179 -> 0.130 | 0.014 |
  | `--distill-start 20 --distill-end 5` (capped by `--distill-max 5.0` to a flat 5.0) | 0.793 -> 0.764 | 0.167 -> 0.207 | 0.007 |

  The strong anchor held the student at its starting behaviour (including the start policy's ~17 % hover) and RL could not improve it; the weak one let RL fix part of it. Keep the
  weak weight for the frozen-best anchor. Caveat: one seed, 23 short chunks; confirm on the full-size run.
- **Deployment must re-centre the position too (measured 2026-10-06).** The training re-centring (handoff point at x=y=0) is part of what the policy was trained on.
  Replaying the new `best.pt` in the numpy two-phase tool with the raw absolute position gave 9 % hits (75 scenarios, 250 m spawn, switch 40 m; first thrust action saturated at ~14, 41 % roll/pitch
  crashes, 37 % parked); the same checkpoint on the same 24 scenarios with `pos' = pos - pos_handoff + (0,0,20)` gave **24/24** hits (3/24 stock). Caveats: numpy
  simulator, one seed, z0=20 untuned; 3 of 75 stock episodes were PID-leg out-of-bounds at 1-3 m target altitude (a separate problem). The structural fix is a target-relative observation (`alt_model.md` 4.2).
  - **Implemented (2026-10-06, later the same day).** The option described above had not actually landed in the code; it now has. What re-centring is: training does
    not steer the drone through x=y=0, it only *places the spawn* so that the straight controller leg reaches distance h at x=y=0. At deployment the same thing is
    done as a coordinate shift (a "mapper"): at the handoff step, `P0` = the drone's position; from then on the policy's `obs[0:3]` is `symlog(pos - offset)` with
    `offset = (P0.x, P0.y, max(0, P0.z - z0))`. So x,y read 0 at the handoff, and z is lowered to z0 **only if the handoff is higher than z0**. This differs from the
    measured formula above (`+ (0,0,20)`, i.e. always z = 20 at the handoff): a handoff at 3 m altitude would read as 20 m there, hiding the ground. Velocity,
    attitude and the vector to the target are unchanged by a shift, so nothing else in the observation is touched. The drone still flies (and is drawn) in real coordinates.
  - Where: `TwoPhaseAgent(..., recentre_z=z0)` (`app/control/two_phase.py`; ignored when the env is in `height` mode); `recentre_z` on `record_checkpoint_two_phase`,
    `trajectory_two_phase`, `evaluate_two_phase`; `python app/guidance/record_run.py CKPT --two-phase --evaluate --distance 250 --switch-dist 40 --recentre-z 20`
    (run with `PYTHONPATH=.`); the viewer option (section 17). The "rl takes over" log line shows the offset used.
  - Re-measured with this implementation: `v3_2p_..._3_50m/best.pt` (`full` mode), 250 m, switch 40 m, seed 0, **8 scenarios**: stock **2/8** hits (2 attitude, 4 timeouts),
    re-centred (z0 = 20) **8/8**.
- **Still open:** no `--resume` (critic + optimizer); no held-out set for checkpoint selection; the handoff-state pool, target-relative observation and time-to-hit reward are design only (`alt_model.md`).

---

## 16. Formula sheet

Added 2026-10-05 because several formulas were only described in words (and the distillation cap `--distill-max` was not written down anywhere).
The PPO background is also derived in `PROJECT_DEFENSE_GUIDE.md` Part 0.4; the potential-shaping proof is in 0.7.

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

---

## 17. The web viewer (2026-10-06)

`uv run uvicorn app.guidance.serve_run:app --port 8000` -> http://localhost:8000. The page is `app/guidance/viewer.html` (edit it as a normal HTML file; it is read on every request).
- **Tabs:** *Two-phase* (default), *Single policy*, *Rank checkpoints*.
- **Two-phase tab:** pick any checkpoint the server lists (`best.pt` files first, `latest_full.pt` hidden) or upload one; target distance, switch distance, cruise / handoff
  speed, test runs, seed; presets; and the **"Re-centre the position input at the handoff"** option (`recentre_z`, altitude cap default 20 m) described in section 15.
  Picking a checkpoint from an `rtl_train_isaac.py` run reads its `run_config.json` (`GET /api/checkpoint_info`) and shows the trained range, max speed, spawn leg and
  **position input mode**. The re-centre box sets itself: **on** for a `full`-mode run that used a spawn leg with re-centring (`recenter` true, the default), **off** for a run
  that did not re-centre, **disabled** for a `height`-mode run (nothing to re-centre). An uploaded `.pt` has no `run_config.json`, so the box is left as you set it.
- **"Run & play in 3D"** (replaced the mp4 player, 2026-10-06): runs one episode and plays it back in a three.js scene in the page (three.js 0.160 from cdn.jsdelivr.net,
  so the page needs internet access). The scene is z-up, the same frame as the simulator, and uses real world coordinates (re-centring only changes what the policy is fed).
  - **Camera:** drag = orbit, right-drag = pan, wheel = zoom toward the cursor. Modes: *Orbit* (free), *Follow* (the camera keeps its offset and travels with the drone),
    *Chase* (behind the drone along its velocity). One-shot views: *Fit*, *Top*, *Side*.
  - **Playback:** play/pause, step one frame back/forward, jump to start/end, loop, speed 0.1x-64x (defaults to whatever makes the replay ~30 s), a timeline slider, and a
    distance (log scale) + speed chart under the scene that you can click or drag to seek; the policy-phase part of the chart is shaded and "RL takes over" is marked.
  - **Keys:** Space play/pause, Left/Right step (Shift = 10 frames), Home/End, F follow, C chase, R fit, + / - speed. Double-click the scene = play/pause.
  - **Scene:** the drone model uses the simulator's real orientation (white nose = body +x), blue during the controller leg and orange once the policy flies; the trail is
    coloured the same way (plus an optional faint full path); velocity arrow, optional dashed line to the target, a drop line + shadow to the ground; the target is drawn at its
    true hit radius plus a wireframe marker that keeps a constant on-screen size so it can be found from far away; the faint green sphere is the switch distance; the grey
    marker is the start. The drone and markers are scaled up with camera distance so they stay visible when zoomed out.
  - **HUD:** phase (PID / RL policy), sim time, simulator step, distance, speed, and the outcome once the end is reached. Details and the timestamped log are under the player.
- **Batch:** "Run test batch (no video)" is unchanged: hit / roll-pitch crash / out-of-bounds / timeout cards (and how many timeouts were parked < 1 m), a per-scenario
  table, the summary text with a copy button, and the timestamped log.
- **Single policy / Rank tabs:** unchanged (the single-policy tab still renders an mp4).
- **API:**
  - `/api/two_phase/trajectory` (new): same form fields as `/api/two_phase/record`, no mp4/png. Returns the episode stats plus `track`: per-frame `pos`, `quat` (x,y,z,w),
    `phase` (0 controller / 1 policy), `dist`, `speed`, the simulator `steps` each frame came from, and the frame `dt`. Downsampled to <= 3000 frames (first and last step
    always kept); a 60 m episode is ~0.3 MB of JSON and ~2 s. Built from `run_episode_two_phase(..., track=True)` via `trajectory_two_phase` in `record_run.py`.
  - `/api/two_phase/record`, `/api/two_phase/trajectory` and `/api/two_phase/evaluate` take either an uploaded `file` or a listed `checkpoint` path (any other path is
    rejected) plus an optional `recentre_z`; record and trajectory return `recentre_z` and `position_mode`.
  - `/api/checkpoint_info` also returns `obs_position_mode` and `recentred` (spawn-leg run with `recenter` on).
- **Python changes need a server restart** (uvicorn is not started with `--reload`); the HTML is re-read on every request.
- **Not yet:** the viewer replays the *numpy* simulator, not Isaac. Recording the real Isaac eval (`isaac_eval`) and playing it here is planned, not built.

---

## 18. Observation position mode `height` (option C, 2026-10-06)

`--obs-position-mode height` (default `full` = the original). It changes only `obs[0:3]` (same 23-dim layout, so `full`-mode checkpoints still load):

| mode | `obs[0:3]` | consequence |
|---|---|---|
| `full` | `symlog(x, y, z)` of the absolute position | the policy can see where in the world it is; a handoff 200 m from the origin is out of range (needs `recentre_z` at deployment) |
| `height` | `(0, 0, symlog(clip(z, 0, 30 m)))` | no absolute horizontal position at all, altitude clipped at 30 m ("30 m up" = "200 m up"); nothing to re-centre anywhere |

Where it is implemented: `build_observation(..., position_mode)` / `BaseDroneEnv(position_mode=)` (numpy), `BaseDroneEnvIsaac.set_obs_position_mode()` (Isaac), recorded in `run_config.json` as
`obs_position_mode`; `record_run.py` reads it from the `run_config.json` next to a checkpoint (`_position_mode_for`) and builds the numpy env in the same mode. `TwoPhaseAgent.recentre_z` is ignored in
`height` mode, and the viewer shows the mode and does not turn on re-centring for such a checkpoint.

**Measured (1024 envs, 150-250 m spawn, start = `rtl_handoff_v2_2p_150_250_3_50m/best.pt`, which was trained in `full` mode; chunk-0 Isaac eval, ~+-1 point):**

| observation | eval hit | eval hover |
|---|---|---|
| `full` (as trained) | 0.875 (best checkpoint) / 0.80-0.85 typical | 0.11-0.16 |
| `height`, no training at all | **0.935** | **0.036** |

**Numpy replay parity** (same 24 scenarios as the re-centring test, 250 m / switch 40 m, seed 0, same checkpoint, no `recentre_z`): `height` mode **24/24 hits** (stock `full` replay: 3/24; `full` + `recentre_z=20`: 24/24).
A short run after that (10 chunks, `--obs-position-mode height`, frozen-best anchor 0.25 -> 0.1) held eval at 0.89-0.93 (too short to conclude more).
So removing the absolute x/y inputs did not hurt the existing policy, it helped. Retrain command: add `--obs-position-mode height`; a `full`-mode `best.pt` is a valid `--bc-checkpoint` / `--anchor-checkpoint`.
Caveat on the anchor: it is fed the same `height` observation as the student, so it anchors to the start policy *as it behaves in that mode*.

---

## 19. Implementation notes (moved from code comments, 2026-10-06)

Details of `install_handoff_stage` / `collect_rollout` / the main loop in `rtl_train_isaac.py` that used to live in comments.

**Handoff trigger.** Control passes to the policy as soon as the drone's **real** distance to the target is `<= h`, where
`h ~ U(dist_low, dist_high)`. That is the policy's own range, not the spawn range, so a BC checkpoint trained on that
range stays in-distribution regardless of how far the drone spawned. There is a second trigger, for when the drone flew
the whole planned leg along the line without entering the `h` sphere (small `h` plus cross-track error). `_prefix_left`
is only a fallback timer, so a drone that can't keep up isn't flown forever. Once handed off, an env stays handed off
until its episode ends; an overshoot does not give control back to the controller. Controller steps have `pm == 0` and
are excluded from the policy gradient.

**Legacy mode (no `--spawn-dist-*`).** A short prefix leg `U(prefix_low, prefix_high)`. The drone spawns already moving
at the bucket speed with a random tilt, the position-tracking PID flies the whole leg, and there is no velocity snap.

**Carrot leg (spawn-leg mode).**
- The drone spawns at rest and level. The real position-tracking PID brakes hard near *any* target, so over a long leg
  its speed overshoots and then collapses well before `h`. Instead, the carrot controller tracks a velocity profile
  along the straight spawn->target line: `accel = cruise_gain * (v_cmd(s) - vel)`, where `s` is read back from the
  drone's own position. The hover PID's `kd_pos` would fight a sustained cruise. Altitude is a position hold on the
  target z.
- **Ramp acceleration** is planned at half the tilt limit, `0.5 * g * tan(max_tilt)`. The old 3.0 m/s^2 sat right at
  the limit: the drone fell behind the profile, and the timer-based handoff fired 20-60 m too far out at a much lower
  speed than planned.
- **Gains are re-picked every step** from the live remaining distance (nearest key in `best_pid_gains_per_dist.json`).
  A fixed close-range gain set fed a long pursuit error overdrove the attitude loop into roll crashes, worse at higher
  speed.
- **Leg elevation is clamped** to `[-(spawn_z - 0.5) / L, +leg_max_elev_deg]`. A long leg aimed downward used to end
  underground. The floor clamp then moved the target off the planned line, and the drone flew into the ground. Steep
  climbs crawl, because altitude is only a position hold.
- **Re-centring (full mode).** The spawn is shifted back along the line so the handoff **point** lands at x = y = 0.
  Without this, a 150-250 m leg hands off 100-250 m from the origin, which is outside anything the policy saw: eval hit
  0.56 with a 50-80 m spawn vs 0.09 with 150-250 m. Not needed in `height` mode.
- **Fallback timer** comes from the ramp, cruise and ramp times, not `prefix_m / (handoff_speed * dt)`. Most of the leg
  is flown near `vpeak`, so that formula overestimated low-speed buckets and left the carrot frozen at the handoff point.
- **Handoff snap.** Tracking lags a little, so at the handoff step velocity and attitude are forced to the sampled
  bucket speed (direction to target plus a random lateral component and tilt: the "imperfect real handoff"). Position
  is untouched. The next observation is patched in place: velocity, quaternion, angular velocity (zeroed), yaw-error
  terms. `record_run`'s two-phase agent does the same snap.

**Staggered start.** On full resets in training only, each env starts at a random point of its leg, **uniform in time**,
already moving at the profile speed. Without this, a ~23 s leg spans about 5 chunks and episodes end in waves: chunks
with no finished episode (shown as `train_hit = 0.000`) alternate with chunks where nearly all finish. That fooled the
guard into boosting the PID anchor to its cap, and starved PPO of policy-phase data in 4 chunks out of 5. A
distance-uniform start brought the waves back, because the drone covers ground fast while cruising. Eval never
staggers.

**Eval burn-in.** An eval resets every env, which puts them in lock-step again. The next chunk would then contain only
fast-ending episodes (hit 0.98 / hover 0 / hit time 3 s, against a true 0.91 / 0.07 / 5 s): the spikes after each eval.
`--eval-burnin` runs the noisy policy with the data discarded until the episodes are desynchronised.

**Guard inputs.** Chunks with fewer than `--guard-min-episodes` finished episodes are ignored. The teacher guard is
driven by the **eval** hit rate (deterministic, fresh envs, first episodes, so unbiased), never by the chunk-level
`train_hit`, and is judged only at eval chunks.

**Eval.** Per-episode time budget from the start distance (`steps_for_dist`). Only each env's first episode counts, so
fast-ending episodes aren't over-counted. `use_pid=True` flies the whole episode with the PID as the baseline. The env
must be reset again before training continues.

**Shutdown.** The traceback is printed before `SimulationApp.close()`, because Kit can hard-exit during shutdown and
swallow it.
