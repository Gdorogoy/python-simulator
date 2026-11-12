# Reading the training logs (acro, reward v3/v4)

Standalone copy of the "Reading the training logs" section of `app/training/docs.md`. Same numbers, same meaning: the
console is the quick view, `metrics.csv` is the same data with one column per key.

Each chunk writes **one progress line** (plus two eval lines on eval chunks) to the console and **one row** to
`metrics.csv` / `metrics.json` / MLflow. `summary.png` is redrawn from the CSV.

## The progress line

One per chunk, wrapped over three lines, `[rtl]` prefix:

```
[rtl] 12/143 full          train_hit=0.93 (base 0.71) eval=0.95 [b0/b1/b2 0.99/0.94/0.88] caught=0.000 pid=0.97
      timeout=0.02 hover=0.01 cut=0.01 hit_all=0.90 stall=0.00 agree=0.08 spd=-0.010 line=-0.004 lat=6.2m
      short=0.12 perp=1.4m path=+0.003 hitv=44.1 d_coef=0.010 kl=0.012 vloss=0.42 ev=0.71 1800 steps/s
```

The example numbers are illustrative; the field list is what the trainer actually prints.

### Hit rate and how episodes ended

| field | meaning | how to use it |
|---|---|---|
| `train_hit` | hits / **finished** training episodes, windowed average | optimistic when the policy stops finishing episodes: the few that end are the easy ones |
| `hit_all` | hits / (finished + `cut`), i.e. the shortened first episode after each reset counted as a miss | read this one whenever `cut` is not ~0 |
| `cut` | first episodes whose randomly shortened clock ran out | not in `train_hit`'s denominator |
| `timeout`, `hover`, `escape` | how the finished episodes ended | `oob`, `tilt`, `near_miss` are CSV columns only |
| `base` | hit rate of a no-learning baseline on the same envs | `train_hit` below `base` means the policy is worse than nothing |
| `eval` | deterministic eval hit rate: first episode per env, `tanh(mean)` action | the number to trust; `BEST.json` scores `eval_hit - 1e-4 * eval_hit_time_s`, so hit rate dominates |
| `b0/b1/b2` | the same eval split by takeover-speed bucket | handoff runs only: "can it still hit when it takes over fast?" |
| `evalnoisy` | eval with training noise (CSV only) | a large gap to `eval` means the policy relies on its noise |

### Teacher, distillation, PPO health

| field | meaning | healthy |
|---|---|---|
| `pid` | the teacher flying alone, no policy | a policy below it is worse than its own teacher |
| `stall` | fraction of steps in the near-target window below `diag_stall_speed` | ~0; rising means it parks at the target |
| `agree` | teacher/student action MAE | falling = getting closer to the teacher |
| `d_coef` | effective distillation coefficient (`distill_scheduled * guard`) | falls to 0 by design in a v4 run (`--distill-end 0`) |
| `kl` | approximate KL between the old and new policy | 0.005-0.03 is healthy; spikes mean the step is too big |
| `vloss` | critic loss | falls; oscillation means the critic is chasing the policy |
| `ev` | explained variance of the critic | close to 1 is good; negative is worse than predicting the mean |
| `steps/s` | throughput | a sudden drop is usually a machine problem, not a training problem |

### v3/v4 shaping (the part that matters for the speed task)

| field | definition | read it as |
|---|---|---|
| `spd` | `rew_speed` per step | negative = paying the speed shortfall; closer to 0 = on the target profile |
| `line` | `rew_line` per step | negative = paying the cross-track cost |
| `lat` | `train_mean_lateral_m`, metres | mean distance off the straight spawn->target line |
| `short` | `train_speed_short_frac` | 0 = on the target speed profile, 1 = standing still / flying away |
| `perp` | `train_perp_miss_m`, metres (**v4**) | distance from the target to the velocity line: how parallel the approach is |
| `path` | `rew_path` per step (**v4**) | the terminal straightness bonus, amortised per step |
| `hitv` | `train_hit_speed`, m/s (**v4**) | mean speed on the hit step: the pace that actually got paid |

`spd` and `line` are per-step rewards, so they are tiny numbers. Compare them to each other and to `step_leg / N`, not
to 1.

## Eval chunks

Two extra lines (three in handoff runs):

```
[rtl]   eval endings: hit=0.950 caught=0.000 hover=0.010 oob=0.020 tilt=0.010 timeout=0.005 near_miss=0.005 escape=0.000 unfinished=0.000
[rtl]   closest approach: perpendicular miss mean 1.40 m (p50 1.10, p90 2.70) | hit-flight mean speed 44.1 m/s
```

- `eval endings` is the outcome split of the deterministic eval. `crash by bucket` (handoff only) splits `oob`+`tilt`
  by takeover speed.
- The closest-approach line (v4) separates the two failure shapes:
  - **large `perp`** -> it flew past the target on a parallel line;
  - **small `perp` + large `min_dist`** -> it stopped short.
- `eval_ho_*` (handoff check) is the state at takeover, *before* the perturbation. If its distance/speed errors are
  large, the scenario is broken, not the policy.

## `metrics.csv`

The console keys are columns directly (`train_*`, `eval_*`, `rew_*`), one row per chunk. The columns worth watching
over time:

- `eval_hit` / `success_rate` — the same number (deterministic hit rate).
- `eval_hit_time_s` — mean time of the hits. This is the **pace** metric.
- `eval_mean_speed` — mean speed over the eval episodes.
- `eval_min_dist_p10/p50/p90` — sampled centre distance at closest approach.
- `eval_perp_miss_mean/p50/p90`, `eval_hit_speed` — v4 only. `eval_hit_speed` is the episode mean over the
  episodes that hit (it includes the spool-up from rest), not the speed on the hit step.
- `train_mean_lateral_m`, `train_speed_short_frac`, `train_perp_miss_m`, `train_hit_speed`.
- `rew_*` — per-step contribution of each reward part.
- `best_eval_hit` — the running best (what `BEST.json` records, with the checkpoint path, chunk and timesteps).

`summary.png` plots hit rate, endings, tilt and the reward parts.

## How to judge a v4 speed run, in order

1. **`eval_hit` must stay within ~0.03 of the tuned reward-v3 run (0.958).** If it drops, the reward is
   charging too much: lower `speed_leg` first, not `line_path_leg`.
2. **`eval_hit_time_s` must fall** (36.3 s -> ~30 s for 1000-1500 m). If it does not move, the pace term is not
   the binding constraint.
3. **`lat` must fall without `short` rising, and `perp` must stay low.** The reward-v3 failure was a ~33 m
   mean cross-track (a bow) at `short 0.31`; v4's `perp` p50 is ~0.5 m, so the terminal aim was already good.
   Do not read the flight-mean `lat` as a terminal miss.
4. **`eval_hit_speed` must rise toward 45-47 m/s and `hitv` from ~30.** The eval mean includes the spool-up from
   rest, so 50 on that column is not reachable. `hitv` is the hit-*step* speed and stays ~7 m/s below the flight
   mean because `speed_end_speed 22` asks for a slow pass; raise it only with that profile (or a v5 `v_hit` term).
   The airframe limit is 52.1 m/s at ~70.5 deg tilt and full thrust; 55 m/s is not reachable.
5. **If `short` falls but `lat`/`perp` do not**, the speed term is being paid by flying a longer path. Raise
   `line_path_leg` (per metre flown), never `line_leg` (per step — it is forced to 0 in v4 and would only buy
   "straight but slow", the LOS@35 behaviour).

## Failure shapes -> lever

| symptom in the log | what it means | first lever |
|---|---|---|
| `eval_hit` falls, `short` falls | reward is charging too much for pace/path | lower `speed_leg` |
| hit rate holds, `eval_hit_time_s` flat | pace term not binding | raise `speed_leg`, tighten `time_bonus_budget_s` |
| `lat`/`perp` flat, `short` falls | speed bought with a longer path | raise `line_path_leg`, `line_lead_s` |
| `lat` falls, `perp` stays high | on the line late but arriving parallel | raise `line_lead_s` (charges the drift immediately) |
| `path` ~0 on hits | flown path is > `path_slack` longer than straight | raise `path_leg`, or fix `lat`/`perp` first |
| `eval_hit_speed` much below `eval_mean_speed` | the fast flights miss; only the slow ones hit | fix `lat`/`perp` first, then lower `speed_leg` |
| speed plateaus, `short` flat, `hit` stable | actor LR at its floor, or the speed term is too weak for the thrust corner | fine-tune with a constant `lr_actor 1e-4`, `speed_leg` 75-100 and `time_bonus_budget_s` 40; check `sat_frac_d0` |
| `stall`/`hover` rise near the target | it parks instead of committing | raise `hit_leg` / `miss_leg`, check `zone_time` is 0 |

A level 38 m/s cruise is ~56 deg tilt and 0.40 thrust; 50 m/s is ~69 deg and 0.89; 52.1 m/s is 70.5 deg and
1.00. If the policy sits at 38 with `sat_frac_d0` ~0.07, it is not thrust-limited: the reward / curriculum is,
not the plant.

## The fine-tune run (constant LR)

The card is in `guide/05_rewards.md`; this is how its log should read. Starting at `--lr-actor 1e-4` after a run
that ended at `3e-6` is a ~30x jump, so chunk 0 is expected to be noisy:

- `kl` jumps and `clip_frac` jumps. That is the LR, not a bug. The `--target-kl 0.01` early stop should hold
  `kl` near 0.01; if it *pins* at 0.01 and `clip_frac` stays above ~10% for several chunks, halve the LR to
  `5e-5` rather than abandoning the run.
- `spd` (more negative), `short` (0.31 -> below 0.2) and `eval_mean_speed` should move within 3-5 chunks. If
  `short` is flat while `kl` is healthy, the pace price is not binding — check that `speed_leg` really is 75.
- `sat_frac_d0` rising from 0.07 is the **success** signal: it means the policy is spending thrust on speed.
- `hitv` moves last. The hit-step speed only rises once the terminal aim holds at the higher cruise; a rising
  `eval_mean_speed` with a flat `hitv` means it is fast in the middle and slow at the pass.
- `eval_tilt` / tilt crash must stay ≤ 5%. 72-75 deg max tilt is already close to the 80 deg terminal.

Endpoint at 25 chunks, if it worked: `eval_hit` ≥ 0.95, `eval_hit_time_s` 30-31 s, `eval_mean_speed` 44-46,
`hitv` ≥ 38 (from 30.3 — the hit-*step* speed, not the flight mean), `short` < 0.15, `perp` p50 ≤ 0.6 m. Judge the
**last chunk** (and `--eval-every 5` gives you five
candidates); `BEST.json` is speed-blind, because its score is `eval_hit - 1e-4 * eval_hit_time_s` and 2 s of
flight time is worth only 0.0002 of hit rate.

## Numbers to keep in your head (1000-1500 m stage)

- Physical limit ~52 m/s at ~70.5 deg tilt, full thrust; the `velocity_to_action` clamp (`cos_tilt` >= 0.5) means
  total tilt above ~60 deg under-thrusts, so keep a distilled teacher's tilt flag <= 55 deg.
- At 50 m/s one 60 Hz policy step covers ~0.83 m, so a 1 m hit sphere is a 1.2-2.4 sample window. That is why v4 uses a
  **swept** hit test: without it, flying faster made the hit less likely even on a perfect path.
- `zone_time` must stay 0 for a speed run: a swept miss is one step long by construction and can never satisfy a timer.
- `line_ref` is a **clamp**, not a scale: the per-metre path cost is flat beyond `line_ref` metres of offset.
- The speed term is a lower bound on `v_toward` (velocity along the line of sight), so flying sideways counts as slow.
  `speed_end_speed` is the speed the profile brakes to at the target: 0 brakes to a stop, which turns the last ~80 m
  into a creep (that is why v3/v4 uses 20-25 instead).
- `closing_speed_k` pays per metre closed toward the target. It integrates to a constant, so it cannot score path
  *length*: a 5 % longer path earns the same `vclose`. Only `r_path_bonus` scores that, and only on a hit.
