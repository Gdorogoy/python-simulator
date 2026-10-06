# Why the old non-residual PPO collapsed, and what rtl_train_isaac.py does differently

Written 2026-10-02 from: `git show c5c8775:app/training/base_training_isaac.py` (old `train()`),
`mlflow_isaac.db` run `810c79` (old non-residual 3-30 m, same params as that function), and the
`runs/rtl_3_10_v1`, `runs/rtl_3_30_v1` results. Nothing here was verified by running the changes one at a time
(see "Caveat").

## 1. What happened in the old run (3-30 m, non-residual)

| step | success | approx_kl (target 0.0245) | actor grad norm | timeouts / 30 |
|---|---|---|---|---|
| 35.9M (last warmup chunk) | 0.90 | 0 | 0.6 | 2 |
| 37.0M (actor unfreezes) | 0.63 | 0.058 | 8.1 | 11 |
| 38.1M | 0.63 | 0.128 | 7.5 | 7 |
| 39.1M | 0.33 | 0.073 | 7.2 | 16 |
| 42.3M | 0.17 | 0.028 | 4.1 | 23 |

The collapse starts exactly at unfreeze. Failures are timeouts, not crashes (roll/pitch 0, oob 0-4):
the policy parks near the target (final distance 0.3-0.5 m = inside the 0.5 m inner zone of the old reward).
Same signature in the 3_10 and 3_50 non-residual runs.

## 2. Reward: old (`rewards.py`) vs `reward_qv1.py`

| Term | Old | qv1 |
|---|---|---|
| Hit | +50 | +10 |
| Progress | L1 distance potential / start_dist | log1p(dist/0.5) potential in metres, not divided by start_dist |
| Milestones | +10/+15/+20 once at 25/50/75% progress | none |
| Step cost | -0.0065 / step | -0.002 / step |
| Within 1 m of target | -0.1*(speed+tilt) every step | none |
| Leaving 0.5 m / 1.0 m zone | -2 / -1 | none |
| Timeout | no penalty; value bootstrapped | -3, true terminal, no bootstrap |
| Crash | -1 attitude, -1.5 oob | -5 |
| Tilt / spin | none | small soft tilt wall + angular-velocity penalty |

Note: the old `train()` calls `isaac_ppo_train` with the old reward only. Neither a timeout penalty nor a teacher
term exists in it (`distill_teacher_fn` is never passed; timeouts bootstrap from V(terminal_obs)).

## 3. Training: old `train()` vs `rtl_train_isaac.py`

| | Old | RTL |
|---|---|---|
| Teacher during PPO | none | PID labels on the student's own states as a loss, decaying + adaptive guard, gated by speed |
| Unfreeze | everything at once, 3-chunk LR ramp | head only for 8 chunks, then trunk, ramp on each group |
| Critic | linear head on the actor trunk, lr 5e-5 | separate MLP, lr 3e-4, 8 epochs in warmup |
| Update size / chunk | 10 epochs x 64 minibatches = 640 Adam steps, lr 5e-5 | 5 x 32 = 160 steps, lr 3e-5 |
| KL control | checked only after a full epoch, target 0.0245 | same style, target 0.01 |
| Safety net | soft reset toward "best" checkpoint | teacher-coefficient boost |
| Best-checkpoint tracking | only after unfreeze (BUG, below) | not used for reset |
| gamma / lambda | 0.97 / 0.97 | 0.99 / 0.95 |
| Exploration std | pinned at the floor (0.0498) | starts 0.06, can move |
| Eval targets | same 2000 pairs as training | separate fixed 500, training uses fresh targets |

Identical in both: the imitation stage (byte-identical old vs HEAD except constants) and the PPO core in `train.py`.

## 4. Likely causes, most to least confident

1. **Soft-reset safety net restored the wrong policy (certain, from code + log).** `best_grade` starts at -inf and is
   updated only under `if stage == "training"`. The first chunk after unfreeze (grade 0.576, already collapsed)
   became "best" and stayed 0.576 for the rest of the run. Resets at 42.3M and 47.5M blended 50% toward an already
   broken checkpoint. Same logic is still in `base_training_isaac.py` at HEAD.
   Fix: seed best_grade/best_ckpt_path from the warmup-stage chunks too.
2. **Each chunk moved the policy too far (strong).** KL 0.06-0.13 in the first chunks vs target 0.0245. One epoch
   of 64 sequential minibatch steps already overshoots, and the check only runs after the epoch.
   ~7x more Adam movement per chunk than RTL, with all params unfrozen at once.
3. **Reward made hovering the safe choice (strong, circumstantial).** Failures sat at 0.3-0.5 m, inside the 0.5 m
   inner zone: leaving it costs -2, moving costs -0.1*speed per step, a timeout costs ~-0.2 discounted and is
   bootstrapped. Stopping short is cheap, overshooting is expensive.
4. **No teacher anchor once PPO started (strong).** Any drift was unopposed.
5. **Critic unconverged at unfreeze (plausible).** value_loss still falling 42 -> 30 with no plateau; linear head on
   the shared trunk. Poor critic -> noisy advantages -> normalised up to full-size steps.
6. Ruled out / minor: imitation stage (identical), std magnitude (0.05 vs 0.06).

## 5. How the RTL run did

- 3-30 m (`runs/rtl_3_30_v1`, 122 chunks, no abort, held-out eval): training hit rate 0.88 -> 0.98, timeouts
  0.16-0.33 -> ~0.007, stall fraction 0.66 -> 0.05. Eval ~0.92 (30 episodes, +-5% per point), PID 0.87-0.97.
  Best checkpoint `best.pt` = chunk 99 (104.9M steps); a mild lingering rise (stall 0.03 -> 0.10) appears in the last
  ~40 chunks, so use best.pt, not the final one.
- `train_hit_rate` is lagged: a stalled env only counts as a failure when it times out ~3750 steps (~15 chunks) later.
  The frozen BC student's real hit rate was ~0.7, not the 0.93 the first chunks suggest.
- 3_10 started before the target pool was removed (2000 fixed pairs, memorisable); 3_30 was seeded from the LAST
  3_10 checkpoint, not its best.

## 6. Caveat and next steps

Reward, update size, anchor, critic and unfreeze order were all changed at once. We cannot say which mattered most.
1. Fix the best_grade bug in `base_training_isaac.py`.
2. Add `--reward old` to `rtl_train_isaac.py` and rerun 3-10 m: only the reward differs. If it still holds, the
   training changes did the work; if it collapses, the reward was key.
