# training

## Purpose
Training entry points and the evaluation plumbing around them. **The current trainer is `angle_rtl_train_isaac.py`.**
`rtl-handoff.md` is its full guide: model, PPO phases, PID teacher, reward, handoff stage, how to read the logs, and run
history.

## Files

| File | What it holds |
|---|---|
| `angle_rtl_train_isaac.py` | **Main trainer.** PPO with a separate critic, PID-teacher distillation with an adaptive guard, phased unfreezing, the controller->policy handoff stage, in-Isaac eval, MLflow and plots. |
| `acro_rtl_train_isaac.py` | **Acro trainer.** DAgger-BC from a teacher (the LOS controller by default, or a frozen angle-style policy), then PPO + fading distillation with `acro_reward_v1`. Same console line as the angle trainer. |
| `rtl-handoff.md` | Guide for the trainer above. |
| `base_training_isaac.py` | Earlier Isaac stage trainer (`train()`, PID-residual mode). The RTL trainer still imports its imitation stage, kinematics helper and JSON metric log. `train()` itself is only used by `deprecated/training/train_isaac.py`. |
| `config.py` | Shared PPO / imitation hyperparameters (`PARAMS`, imitation and entropy constants). |
| `diagnostics.py` | `diagnose_with_model` / `diagnose_with_pid`: outcome counts on fixed pairs in the numpy env. |
| `eval_matrix.py` | Target-pair builders (axis, omni lattice, random omni, uniform range) and `run_eval_matrix`. |

## Concepts
- **Step:** one policy decision. The policy runs at 60 Hz, with 4 physics substeps at 240 Hz.
- **Chunk:** `--num-steps-per-chunk` (256) steps of **every** env, followed by one PPO update. One chunk is
  `256 * num_envs` samples. Episodes don't line up with chunk boundaries.
- **`--total_timesteps`** counts samples across all envs. With 8192 envs, the default 128M gives about 61 chunks. Every
  chunk-count setting (warmup, eval cadence, decay) does **not** scale with `num_envs`.
- **Phases** (`PHASES`): `settle` (rollout only) -> `critic_warmup` (actor frozen) -> `head_only` (actor head + log_std)
  -> `full`.

## Running
See the README "Training" section for the exact commands. Every run writes `runs/<checkpoint-dir>/` (`best.pt`,
`latest_full.pt` for resume, `model_<steps>.pt` / `.onnx`, `metrics.csv`, `run_config.json`, plots) and logs to the
MLflow store `mlflow_isaac.db`.

Useful flags (all have `--help` text):

| Flag | What it does |
|---|---|
| `--handoff`, `--max-speed`, `--spawn-dist-low/high` | Handoff stage: speed buckets, and the carrot leg flown from spawn. |
| `--distance-low/high` | The policy's own range: handoff distance `h`. |
| `--obs-position-mode {full,height}` | `height` removes absolute x/y. Use it for new runs. |
| `--bc-checkpoint`, `--anchor-checkpoint` | Start weights, and the frozen policy used as the distillation anchor. |
| `--distill-start/end/decay-chunks`, `--guard-*` | Teacher-loss schedule and the eval-driven guard that boosts it. |
| `--settle/warmup/head-chunks`, `--lr-actor/critic` | Phase lengths and learning rates. |
| `--eval-every`, `--eval-burnin`, `--pid-eval-every` | In-Isaac evaluation cadence. |
| `--handoff` (acro) + `--spawn-dist-low/high`, `--max-speed`, `--lateral-frac`, `--handoff-tilt-deg`, `--teacher-accel`, `--leg-max-elev-deg`, `--no-recenter` | Two-stage episodes, same flags and meaning as the angle trainer: a carrot-chase controller flies from the spawn point (spawn-dist from the target) to the handoff point ~ U(`--distance-low`, `--distance-high`) from the target, then the acro policy takes over with a random sideways velocity and tilt. See "Handoff mode" in the acro section. |
| `--handoff-speed` (acro) | Staggered start on the trainer's full resets: random point on the spawn->target line (0-70% of the leg), moving toward the target at U(0, X) m/s; default 1.0, 0 = from rest. Eval is never staggered. |
| `--desync-steps` (acro) | Optional window after the reset in which each env is force-reset once at a random step (data discarded). Default 0 = off. |
| `--reward {qv2,rc}`, `--reward-json` | Pick the angle reward; override any field of its config (`RewardQV2Config` / `RewardRCConfig`), from a file or inline JSON. |
| `--teacher {auto,pid,policy,los}`, `--los-*`, `--teacher-bc-*`, `--scratch` | Who labels the distillation loss, the LOS controller's settings, and the supervised teacher-BC stage before PPO. |
| `--weight-decay` | AdamW weight decay for every optimiser (default `config.WEIGHT_DECAY` = 1e-4). |

## Reading the training logs
Standalone copy: [`logs_guide.md`](../../logs_guide.md) (repo root). Keep the two in sync.

Each chunk writes one progress line (plus eval lines on eval chunks) to the console and one row to `metrics.csv` /
`metrics.json` / MLflow. The console is the quick view; the CSV is the same numbers, one column per key.

**The progress line** (one per chunk, `[rtl]` prefix):
```
[rtl] 12/143 full          train_hit=0.93 (base 0.71) eval=0.95 [b0/b1/b2 0.99/0.94/0.88] caught=0.000 pid=0.97
      timeout=0.02 hover=0.01 cut=0.01 hit_all=0.90 stall=0.00 agree=0.08 spd=-0.010 line=-0.004 lat=6.2m
      short=0.12 perp=1.4m path=+0.003 hitv=44.1 d_coef=0.010 kl=0.012 vloss=0.42 ev=0.71 1800 steps/s
```
- `train_hit` — hits / **finished** training episodes, a windowed average. It is optimistic when the policy stops
  finishing episodes: the few that end are the easy ones. `hit_all` — hits / (finished + `cut`), i.e. the shortened
  first episode after each reset counted as a miss. Use `hit_all` when `cut` is not ~0.
- `cut` — first episodes whose randomly shortened clock ran out; they are not in `train_hit`'s denominator.
  `timeout`, `hover`, `escape`, ... — how the finished ones ended (`oob`, `tilt`, `near_miss` are CSV columns).
- `eval` — deterministic eval hit rate (first episode per env, `tanh(mean)` action). `b0/b1/b2` — the same split by
  takeover-speed bucket (handoff runs only); read it as "can it still hit when it takes over fast?".
  `evalnoisy` (CSV only) is the same eval with training noise: a large gap means the policy relies on its noise.
- `pid` — the teacher flying alone (no policy); a policy below it is worse than its own teacher.
- `stall` — fraction of steps in the near-target window at < `diag_stall_speed`; `agree` — teacher/student action MAE
  (lower = closer to the teacher); `d_coef` — effective distillation coefficient.
- PPO health: `kl` (0.005-0.03 is healthy), `vloss` (should fall, not oscillate), `ev` (explained variance of the
  critic: close to 1 is good, negative means it is worse than predicting the mean), `steps/s`.
- **v3/v4 shaping:** `spd` = `rew_speed` per step (negative = paying the speed shortfall), `line` = `rew_line`,
  `lat` = `train_mean_lateral_m` (metres off the straight line), `short` = `train_speed_short_frac` (0 = on the target
  profile, 1 = standing still / flying away). **v4 adds** `perp` = `train_perp_miss_m` (metres from the target to the
  velocity line: how parallel the approach is), `path` = `rew_path` per step (the terminal straightness bonus), and
  `hitv` = `train_hit_speed` (mean speed on the hit step).

**Eval chunks** add two or three lines:
```
[rtl]   eval endings: hit=0.950 caught=0.000 hover=0.010 oob=0.020 tilt=0.010 timeout=0.005 near_miss=0.005 escape=0.000 unfinished=0.000
[rtl]   closest approach: perpendicular miss mean 1.40 m (p50 1.10, p90 2.70) | hit-flight mean speed 44.1 m/s
```
`eval endings` is the outcome split; `crash by bucket` splits oob+tilt by takeover speed. The closest-approach line
(v4) separates the two failure shapes: **large `perp`** = flew past on a parallel line; **small `perp` + large
`min_dist`** = stopped short. `eval_ho_*` (handoff check) is the state at takeover, before the perturbation: if its
distance/speed errors are large, the scenario is broken, not the policy.

**`metrics.csv`.** The console keys are columns directly (`train_*`, `eval_*`, `rew_*`); one row per chunk. The ones
to watch over time: `eval_hit` / `success_rate` (same number), `eval_hit_time_s` (mean time of the hits: the pace
metric), `eval_mean_speed`, `eval_min_dist_p10/p50/p90` (sampled centre distance), `eval_perp_miss_mean/p50/p90` and
`eval_hit_speed` (v4), `train_mean_lateral_m`, `train_speed_short_frac`, `train_perp_miss_m`, `train_hit_speed`,
`rew_*` (per-step contribution of each reward part) and `best_eval_hit`. `summary.png` plots hit rate, endings, tilt
and the reward parts.

**How to judge a v4 speed run, in order.** 1) `eval_hit` must stay within ~0.03 of the tuned reward-v3 run (0.958; if
it drops, the reward is charging too much: lower `speed_leg` first). 2) `eval_hit_time_s` must fall (36.3 s -> ~30 s at
1000-1500 m). 3) `lat` must fall without `short` rising, and `perp` must stay low (the reward-v3 failure was a ~33 m
mean cross-track, a bow, with `short 0.31`; v4's `perp` p50 is ~0.5 m, so the terminal aim was already good).
4) `eval_hit_speed` must rise toward 45-47 m/s (the eval mean includes the spool-up from rest) and `hitv` from ~30.
`hitv` is the hit-*step* speed, ~7 m/s below the flight mean because `speed_end_speed 22` asks for a slow pass.
If `short` falls but `lat`/`perp` do not, the speed term is being paid by flying a longer path: raise
`line_path_leg`, not `line_leg`.

**`best.pt` is speed-blind.** It is chosen by `score = eval_hit - 1e-4 * eval_hit_time_s`: a 35 s time difference is
worth only 0.0035 of hit rate. A checkpoint 0.5-2 m/s faster with 1 % fewer hits is not kept (the v4 run: final was
0.46 m/s faster than best, 1.3 % lower hit, so `best.pt` is the slower one). For a speed run, evaluate the final
`model_<steps>.pt` too (`latest_full.pt` holds the optimiser and cannot be loaded for inference), or change the score
to include speed.

## Angle vs acro: why the evals are separate
The two trainers share the observation (23 numbers, same builder, same symlog scaling), the 60 Hz policy rate (decimation 4), the hit
radius (0.25 m), the per-distance time budget (`steps_for_dist`) and the network shape (23 -> 4). Everything else that an eval
depends on differs, so an angle policy cannot be scored by the acro eval, or the other way round. Nothing complains when you try:
the dimensions match, and the result is silently meaningless.

**1. The same four numbers mean different things.**

| | angle | acro |
|---|---|---|
| Action | `[thrust_delta, tau_x, tau_y, tau_z]` torques | `[thrust_delta, p, q, r]` body rates |
| Bounds (saved in the checkpoint as `action_low/high`) | thrust -14.7..+29.4 N, torque +-1.5, +-1.5, +-0.5 N*m (older checkpoints: thrust +-14.7, torque +-0.5) | thrust -14.7..+29.4 N, rate +-6, +-6, +-3 rad/s |
| What the plant does with it | `mixer_inversion` of `[thrust, torque]` once per policy step | a rate controller runs every physics substep (240 Hz) on the live body rate, then `mixer_inversion` |
| Env / gym id | `Isaac-Base-Drone-Direct-v0` | `Isaac-Acro-Drone-Direct-v0` (`AcroDroneEnvIsaacCfg`, `drone_style="acro"`) |

Run an angle policy on the acro plant and its torque-sized outputs (+-0.5) are read as rates: 0.5 rad/s at most, so the drone barely
tilts. Run an acro policy on the angle plant and its rate-sized outputs (up to 6) are clipped to +-0.5 N*m: every output
saturates and the drone spins. Both fly, neither is the policy that was trained. The env clamps actions to **its own** bounds, so
the policy's `scale_action` and the env must agree on them.

**2. The eval is fed different things.**

| | angle `isaac_eval` | acro `isaac_eval` |
|---|---|---|
| Start | the handoff stage: a PID / carrot controller flies the spawn leg, then the policy takes over at a sampled speed, with a random sideways velocity and tilt snapped in (`install_handoff_stage`) | clean start from rest at the spawn point, or launched toward the target at `--spawn-speed-*` (that applies to the eval too) |
| Who flies | controller until `_handed_off`, then the policy | the policy from step 0 |
| Baseline | a PID run on the same envs (`use_pid=True`) | none; with a teacher it can fly the converted labels (`teacher=`) |
| Endings read from `extras["term_reasons"]` | `hit`, `hover`, `oob`, `attitude_roll`, `attitude_pitch` | `hit`, `hover`, `oob`, `attitude_tilt`, `near_miss` (acro_v2 near-miss timer; always False with v1) |
| Extra per-step data | handoff state (`_handed_off`, `_prefix_left`, `_handoff_h`) | `extras["acro_step"]` (dist, speed, tilt) |
| Attitude rule | Euler roll > 65 deg or pitch > 80 deg is a crash | quaternion tilt from vertical > limit + margin (120 + 30 deg): a flip is allowed |
| Reward / stats module | `angle_reward_qv2` | `acro_reward_v1` |
| Extra stats | per speed bucket hit/crash, handoff distance/speed/tilt error | closest approach, final distance and speed of misses, tilt, over-limit fraction |

The two evals read different keys from the env (`attitude_roll` vs `attitude_tilt`, `_handed_off` vs `acro_step`) and from the reward
that is installed on it, so the wrong combination raises a `KeyError` / `AttributeError` instead of a score. The Euler roll/pitch
rule cannot be reused on acro anyway: a deliberate flip passes 90 deg of roll and would always count as a crash.

**3. The checkpoint tells you which one it is.** `action_high[1]` is at most 1.5 for angle (0.5 in older checkpoints), 6 for
acro; the viewer splits at `record_run.ACRO_MIN_RATE_BOUND` = 3. Acro checkpoints trained before the
bounds fix (v4, v5 and their `bc_best.pt`) carry +-0.5 instead; for those, the `run_config.json` next to the checkpoint decides (only the
acro trainer writes a `tilt_limit` key). The web viewer uses the same rule to choose the plant; an acro checkpoint
shown on the angle plant cannot reproduce the trainer's score, for the reason in point 1 (before this rule, v4/v5 were misread as angle).

**4. The one place the styles are bridged** is the BC label: `AngleTeacher` turns an angle torque into a rate command
(`cmd = w + tau / (I * kp)`), which is only approximate (see the acro trainer section). That makes an angle policy a *teacher*
for an acro student. It does not make it a drone that can fly in the acro env.

**Where each one is evaluated**

| Policy | In Isaac | In the viewer (numpy plant) |
|---|---|---|
| angle | `angle_rtl_train_isaac.py` `isaac_eval` (handoff stage) | Two-phase tab |
| acro | `acro_rtl_train_isaac.py` `isaac_eval` | Acro eval tab (`record_run --acro`) |

**Not built: both styles in one Isaac sim.** Flying an angle drone and an acro drone in the same simulation needs the plant pulled out of the
env into a per-drone class (the env keeps one `_acro` flag, one set of rotor / controller state and one action-bounds pair), a second robot in the scene,
a multi-agent env (or a frozen opponent run inside the step), and per-drone observations whose target is the other drone. The
checkpoints, observation layout and 60 Hz timing would not change. Without an opponent-velocity input a policy can only chase
the other drone's current position.

## acro_rtl_train_isaac
Trainer for `Isaac-Acro-Drone-Direct-v0`. It reuses the PPO update, phases and logging of the angle trainer: the task is
"reach a target at U(distance-low, distance-high) m", optionally behind the same handoff stage (`--handoff`).

**Teacher (`--teacher`, 2026-10-24).** `los` = `LosTeacher`: the angle trainer's line-of-sight controller (`make_los_label`) produces
an angle-style action `[thrust_delta, tau_x, tau_y, tau_z]`, and `angle_to_acro` turns it into a rate command with the same
first-substep torque. `checkpoint` = `AngleTeacher`, a frozen angle-style policy (`--teacher-checkpoint`). `none` = no teacher.
`acro` = `AcroTeacher` (added 2026-11-09), a frozen acro-style policy (e.g. a previous stage's `best.pt`): its output is already
`[thrust_delta, p, q, r]` in this env's units, so the labels are used as-is with no `angle_to_acro`. It refuses a checkpoint whose
action bounds differ from the env's, which catches an angle policy passed by mistake (angle rate bounds are +-1.5 / +-0.5, acro
+-6 / +-3). With `--teacher acro` and no `--bc-checkpoint`, the student starts as an exact copy of the teacher (head kept).
`auto` (default) picks `checkpoint` when `--teacher-checkpoint` is given, `none` when continuing from `--bc-checkpoint`, else `los`.
**Use `--teacher acro` explicitly for acro checkpoints.** `auto` treats any `--teacher-checkpoint` as angle-style; an acro file
then gets converted as if it were angle torques and the teacher flies at 0 %.
With the LOS teacher and no `--bc-checkpoint`, the student starts as a hover policy and runs 6 BC chunks by default.
Flags: `--los-stop-speed`, `--los-max-tilt-deg` (0 < cap <= 90 unless `--teacher-accel` is given), `--teacher-cruise-gain`,
`--teacher-accel`, `--max-speed`. `--pid-eval-every` flies the teacher alone every N chunks; that result is the `pid=` field of the
console line (acro has no PID baseline).

**Console output.** The per-chunk lines use the angle trainer's `[rtl]` format (see `rtl-handoff.md` section 9): one progress line, plus
`eval endings` and `handoff check` lines on eval chunks. Differences: `tilt=` replaces `roll=`/`pitch=` in the endings, `caught` is always
0 (no chasers), and `pid=` is the teacher alone. Every line uses the `[rtl]` prefix, and with a teacher the start-up prints the angle
trainer's `TEACHER CHECK` (the teacher flying the policy's part), `teacher flight` and a few `trace` lines, then exits if
`--teacher-check-only` is set; the `N chunks` banner ends with the reward config. The old `episodes:` / `reward/step:` / noisy-eval prints are gone from the console; the
same numbers are still in `metrics.csv` and MLflow (`rew_*`, `train_*`, `evalnoisy_*`). On `--reward v3` / `v4` the
progress line also carries `spd=` / `line=` / `lat=` / `short=` (speed-profile cost, cross-track cost, metres off the
line, speed shortfall fraction), and on `v4` `perp=` / `path=` / `hitv=` (perpendicular miss, path bonus, hit-step speed);
see "Reading the training logs" above.

**Stages (run by hand).**
1. *Stage 1*: the LOS teacher (default), or `--teacher-checkpoint <angle actor .pt>`, in which case the student starts from the teacher's
   trunk (same 23 -> 4 network). DAgger-BC runs first (up to `--bc-chunks`, default 6), then the usual PPO phases with a fading distillation term.
   BC aggregates every labelled state in a buffer (`--bc-buffer`), so states the student drifts into are kept (default cap 10M pairs, 250k per chunk).
   The teacher flies a fraction of the envs (`--bc-beta-start`) that decays to 0 over `--bc-beta-chunks` (default 4); after that the
   student flies alone and BC keeps training on its own mistakes. A deterministic eval runs every `--bc-eval-every`
   chunks; the best student is kept (`bc_best.pt`) and restored before PPO, and BC ends early at `--bc-target-hit`.
2. *Later stages*, `--bc-checkpoint <previous best.pt>` and a wider `--tilt-limit`, no teacher. The critic is fresh, so the
   critic warm-up runs again. Raise `--init-std` / `--ent-coef` per stage to re-open exploration.
3. `--stop-hit 0.9` ends a stage once eval hit stays >= 0.9 for `--stop-patience` evals and prints the next command.

**Handoff mode (`--handoff`).** Without it, an episode is the whole flight from the spawn point (from rest, or launched with `--spawn-speed-*`).
With it, the acro trainer uses the angle trainer's `install_handoff_stage` unchanged, so an episode has two stages: a scripted carrot-chase
controller flies the spawn leg (`--spawn-dist-low/high` from the target) with the trapezoid speed profile down to a handoff point
~ U(`--distance-low`, `--distance-high`) from the target, at a speed sampled from the buckets 10-25 % / 25-75 % / 75-100 % of `--max-speed`;
control then passes to the policy, with a random sideways velocity (<= `--lateral-frac` of the speed) and tilt (<= `--handoff-tilt-deg`) snapped in.
`--distance-*` is the *policy's* range, not the spawn range. What is acro-specific:
- The carrot controller outputs angle-style torques. `controller_action` turns them into rate commands with `angle_to_acro` (the same
  `cmd = w + tau/(I*kp)` conversion the teacher uses, now a shared function), so the leg is flown through the real acro plant.
- `collect_rollout` and `isaac_eval` let the controller fly while `_handed_off` is False. Those steps get `pm = 0` (no policy gradient); the
  handoff snap patches the observation of the envs that just took over. The eval records the real distance, speed and tilt at the takeover
  (`eval_ho_*`), hit / crash rate per takeover-speed bucket (`eval_hit_b0..2`, `eval_crash_b0..2`, `eval_n_b0..2`, in `metrics.csv`), and counts closest approach only after it.
  Pick `--max-speed` so the scenario is winnable: the sideways velocity (<= `--lateral-frac` of the speed) is about 0.3 * h at arrival whatever the speed, and the time to correct it is h / speed, so for h = 3-5 m and lateral 0.3 the top bucket must stay near 8 m/s (at 35 m/s the 26-35 m/s bucket hit 0.25 with 71 % crashes, at 8 m/s every bucket hit about 0.9).
- The staggered start comes from the handoff stage (random time along the leg), so `install_stagger` / `--handoff-speed` are not used.
- Hit time includes the controller leg. The eval start is the clean spawn point, so it is not comparable with from-rest or launched evals.
Checked in Isaac with a 15-chunk, 256-env smoke run only (wiring, not performance): all 256 eval envs reached the takeover point (distance error
+0.02 m, takeover speed about 9 m/s) and training, eval and the summary ran to the end.

**Teacher conversion.** The angle actor outputs `[thrust_delta, tau_x, tau_y, tau_z]`; the acro action is
`[thrust_delta, p, q, r]`. The rate loop outputs `I * kp * (cmd - w)`, so `cmd = w + tau / (I * kp)` reproduces the
teacher's first-substep torque. It is approximate: the rate error shrinks over the 4 substeps, so the real torque is
somewhat lower. Labels are taken on states the student visits, and are clamped to the acro bounds (6, 6, 3 rad/s).

**Gotchas.**
- A checkpoint from the angle style carries angle action bounds in its `action_low/high` buffers; the trainer overwrites
  them with the acro env's bounds after loading. **Bug fixed:** `ActorCritic` registered the bounds with `torch.as_tensor(numpy)`,
  which shares memory, so loading a checkpoint wrote its +-0.5 into the env's `single_action_space` arrays and the overwrite then
  copied them onto themselves. Every acro run that started from a checkpoint (`--teacher-checkpoint` or `--bc-checkpoint`: v4, v5,
  their `bc_best.pt`) therefore trained and saved with rate bounds of +-0.5 rad/s instead of 6, 6, 3 (thrust was fine); the env
  still clamped at 6, so the policy simply could not command more than 0.5 rad/s, and the BC student executed labels 12x weaker than
  the teacher's. Runs from scratch (`acro_c0`) were not affected. `--bc-checkpoint` now stops with an error when the checkpoint's
  bounds differ from the env's; `--keep-ckpt-bounds` continues such a checkpoint with its own scale.
- `--init-std` is in tanh-normalised space; the maximum is `exp(--log-std-max)` = 0.135.
- **Metrics CSV lock (fixed 2026-11-09).** `RunLog.add` (shared with the angle trainer) rewrites `metrics.csv` through a `.tmp`
  file and `os.replace`. On Windows that raises `PermissionError` while another program holds the CSV open (Excel, an IDE
  viewer, the web viewer), and the run died with it. Now it retries 10 times with a growing pause, and if the file is still
  locked it prints `WARNING ... locked by another program` and skips that write; rows stay in memory and the next chunk
  writes them all. Other OS errors print a warning and the run continues. Checkpoints are saved before the log write, so a
  lock never loses a model. Close the CSV in other programs while training to keep the log complete.
- Eval is deterministic, first episode per env, with a per-distance time budget. It also reports the mean of each
  episode's maximum tilt and the fraction of episodes that exceeded the limit.
- Outputs are the same as the angle trainer (`best.pt`, `model_*.pt/.onnx`, `metrics.csv`, `run_config.json`,
  `summary.json`, `summary.png`) plus `bc_best.pt` after the BC stage (the best BC student by deterministic eval hit).
  Best checkpoint = highest eval hit, ties broken toward faster hits.
- LOS teacher smoke test (512 envs, handoff 30-50 m, max speed 20, tilt cap 60): ran end to end with no NaN; the teacher flying alone
  hit only 0.22 with braking on (`--los-stop-speed 0.5`, 78 % timeouts). Check the teacher-alone line before a long run.

Example (stage 1):
```bash
/mnt/e/Isaac/env_isaaclab/Scripts/python.exe -u app/training/acro_rtl_train_isaac.py --headless \
      --num_envs 8192 --distance-low 3 --distance-high 50 --obs-position-mode height --tilt-limit 120 \
      --teacher-checkpoint stable_models/handoff_v5_3_50m_18/best.pt \
      --checkpoint-dir runs/acro_s1_L120_3_50m --init-std 0.07 --warmup-chunks 8 --stop-hit 0.9
```

### Fine-tuning a checkpoint (speed runs)

`--bc-checkpoint <run>/best.pt` loads the weights and sets `bc_chunks` to 0, so there is no DAgger stage; with no
`--teacher-checkpoint`, `--teacher auto` resolves to `none` and nothing distils back toward the teacher. That is
the mode for a targeted fine-tune: one reward change, 20-30 chunks, same env. The worked example (1000-1500 m,
v4 reward, constant LR) is "The fine-tune card" in `guide/05_rewards.md`.

Three flag traps:
- `--total_timesteps` and `--num_envs` use **underscores**; `--num-steps-per-chunk` uses **dashes**.
- The LR cosine floor is `--lr-min-ratio`; `1.0` is a constant LR. `--lr-ramp-chunks 0` removes the ramp.
- `--settle-chunks` + `--warmup-chunks` (actor frozen) and `--head-chunks` (trunk frozen) are counted in chunks.
  The defaults 2 + 8 + 8 would spend 18 of a 25-chunk run with a partly frozen network.

Non-default values the speed runs rely on (parser default in brackets): `--gamma 0.995` (0.99), `--lam 1.0`
(0.95), `--ent-coef 0.001` (0.0), `--log-std-max -1.5` (-2.0), `--desync-steps 256` (0). Each changes the
algorithm, not just the reward. (`--guard-gain 1` is also in those runs, but it only scales the distillation boost,
so it is inert without a teacher.) Rebuild a command by copying the previous run's `run_config.json`, not from
memory.

### Moving target (`--runner`)

The target becomes a **runner**: a drone with the learner's physics (its own `Runner` body, a sphere of `--hit-radius` with no collision)
flown by the shared LOS maths (`app/control/los.py`) in reverse. It hovers on the target point while the handoff controller flies, and
at the handoff it is released **already moving** straight away from the drone at its cap `U(--runner-speed-min, --runner-speed)`. It
never flees below `--runner-min-alt` (default 2 m). After that `_desired_pos_w` follows the runner body, so the observation, the reward,
the teacher and the hit check all chase it.

- **Win / loss.** It counts as a hit inside `--hit-radius`, as usual. If the **runner** leaves the sphere of radius `--runner-escape-dist`
  (default 100 m) around its **own spawn point**, the episode ends as `escape` with `-r_escape`. The spawn point is the target point it
  hovered on, e.g. spawned at (0, 50, 150) with radius 100: escaped once it is 100 m from there. The drone's position does not matter.
  `r_escape` defaults to 7 = `r_crash`, so letting it escape is never cheaper than crashing. It is an eval code and a train rate
  (`train_escape_rate`, `rew_escape`).
- **Inputs.** +3, appended last: the target's velocity relative to the drone, `/ VEL_SCALE`. A 23-input checkpoint loads with these
  columns at zero weight.
- **Rewards.** acro v1 and v2 progress, closing and missed-pass terms use the target's position from the **previous** step
  (`_prev_target_w`), so they measure the true change of distance. The out-of-bounds radius becomes at least `|spawn| + escape radius + handoff distance`.
- **LOS teacher.** It brakes in the target's frame: `v_des = v_runner + profile(rel)`, capped at `--max-speed`. Without the runner
  velocity, the braking profile would settle where `sqrt(2 a d) = v_runner` and trail the runner forever.
- **Eval budget.** With a runner the per-distance budget is replaced by the full episode length; the eval ends on hit, crash, escape or
  the episode cap.

Teacher check (LOS, 256 envs, handoff 20-50 m, runner 4-8 m/s, hit radius 1 m): hit 0.58 with v2 defaults, where the rest are
`near_miss` (the 6 s timer cuts off a second pass); hit 0.69 with `near_limit_s 0`, where the rest are timeouts. A pure-flee runner
sidesteps the last metre, so the LOS teacher overshoots and has to come round again. Consider `{"near_limit_s": 0}` or a longer timer
for runner stages.

## Threats and the rc reward (angle trainer)

`--reward rc` swaps qv2 for `angle_reward_rc` (see reward_functions/docs.md). `--num-threats K` allocates K threat slots and widens
the observation to `23 + 7K`; `--active-threats` (default K) switches them on; each threat is a real drone (own body, the drone's rotor
physics) flown by the limit-LOS controller, `--threat-speed/--threat-speed-min` its speed cap, `--threat-radius` its collision sphere. Existing checkpoints load unchanged through `--bc-checkpoint`: a narrower checkpoint is
zero-padded on its first layer and the frozen acro teacher reads only `obs[:, :23]`. Eval reports a new `caught` outcome.
Example: `--reward rc --num-threats 5 --active-threats 1 --bc-checkpoint runs/<old>/best.pt`, then raise `--active-threats`.

**Threat geometry (`--threat-spawn geometry`).** Chasers are not placed around the base at the start of the episode; they appear at the
handoff relative to the drone and start chasing at once: head-on (ahead, coming at it) or tail (behind, chasing).
`--threat-geometry-weights H T` (default `1 1`) draws each chaser's geometry; `--threat-reaction-s` (1.5) makes the head-on
spawn at least that many seconds of closing time ahead (and at least 20 m) so a dodge is possible; `--threat-tail-gap LO HI` (15 40) m
behind, `--threat-jitter` (2) m off the line for head-on. `--threat-release-after` and
`--threat-spawn-min/max` are ignored. The eval prints `by geometry: head-on n=.. h=.. c=.. ... | tail .. | none ..` and logs
`eval_{n,hit,caught,att,oob,timeout,hover}_g0..2` (g0 head-on, g1 tail, g2 no chaser; the geometry is the first chaser's); the train
line and the metrics CSV use `train_{...}_{headon,tail,none}`. There is no parallel geometry: a chaser spawned beside the drone either
caught it in the first ~0.05 s or flew off across its track, so it taught nothing. Raise the chaser
speed run by run (`--threat-speed`; `--threat-speed-min` keeps part of the old speed), continuing from the previous `best.pt`.

**Threat spawn, speed and release.** `--threat-spawn uniform` (alias `sphere`, default): every episode each threat is drawn at a
random point between `--threat-spawn-min` (default 15) and `--threat-spawn-max` (alias `--threat-spread`, default 30) metres from
the base, uniform in volume. `--threat-spawn base`: all start at the base (3 m high), slot 0 on it, the others on a ring with
`--threat-buffer` (default 0.5 m) clearance between bodies so they do not spawn inside each other. Speed cap per threat from
U(`--threat-speed-min`, `--threat-speed`); without `--threat-speed-min` it is fixed. It is the only chase limit (the airframe tilt
clamp is `--los-max-tilt-deg`). A threat hovers until the policy flies AND the drone has covered a fraction U(LO, HI) of the distance
it had left at the handoff (`--threat-release-after LO HI`, default `0.33 0.99`; equal values = fixed), then chases.
`--threat-release-frac F` is the old fixed form (fraction still LEFT) and replaces it. `--threat-radius R` makes each threat body a
sphere of R metres (default: the drone's own body). The `[rtl]` progress line shows `caught=` (eval caught rate, carried between
eval chunks like `eval=`).

**Zone timer inputs.** With `--reward rc` the observation gets 2 more numbers after the threat slots, `23 + 7K + 2`: `in_zone`
(inside `hit_radius` now) and `zone_count / limit_steps_hit` (0..1, the hold clock; the hit fires at 1). Older checkpoints
still load (zero-padded at the end). `--no-timer-obs` keeps the old size. The numpy viewer (`slots_from_obs_dim`) does not
know these inputs yet, so it refuses a checkpoint trained with them.

## Method reference: angle_rtl_train_isaac.py
Every function and class of the angle trainer, in file order: what it does in plain words, then pseudocode. Formulas with
symbols are in `rtl-handoff.md` section 16; this section explains the code flow. `obs` is always the 23-dim observation,
`a` the 4-dim action `[thrust_delta, tau_x, tau_y, tau_z]`, "normalised" means mapped to [-1, 1].

### Model
**`RTLActorCritic`**: an `ActorCritic` (actor trunk `shared` + `actor_mean` head + learned `actor_log_std`) with a second,
separate critic MLP `value_net`. The inherited `critic_head` is dead weight.
```
forward(obs):
    features = relu(shared(obs))
    return actor_mean(features), effective_std(), value_net(obs)      # critic sees obs directly, not the actor features
effective_std():  exp( log_min + 0.5*(log_max-log_min)*(tanh(p)+1) )  # p = actor_log_std; std is boxed to [0.030, 0.135]
actor_state_dict(): every weight except value_net.*                   # what best.pt / model_*.pt contain
actor_params():  shared + actor_mean + actor_log_std                  # optimiser group "trunk"+"head"
critic_params(): value_net
```
Plain words: the critic is separate so that (a) value-loss gradients never reshape the actor's features and (b) the
critic can learn alone while the actor is frozen. The std box stops exploration from collapsing to zero or exploding.

**`set_initial_std(model, std)`**: sets `actor_log_std` so that `effective_std()` equals `std` (clamped into the box).
```
target = clamp(log(std), log_min+1e-3, log_max-1e-3)
p = atanh( clamp(2*(target-log_min)/(log_max-log_min) - 1, -0.999, 0.999) )     # invert the squashing
actor_log_std.fill_(p)
```
Called after imitation/BC because BC against a deterministic teacher drives std to its floor; PPO needs noise to explore.

**`log_prob_raw(model, mean, std, raw)`**: log-probability of the action actually sent to the env. The env action is
`low + (tanh(raw)+1)/2*(high-low)`, so the Gaussian density of `raw` needs the change-of-variables term.
```
logp = sum_k( -(raw-mean)^2/(2 std^2) - log std - 0.5*log(2*pi) )
logp -= sum_k( log( half_range * (1 - tanh(raw)^2) + 1e-6 ) )
```
The rollout and the update both use this function, so the PPO ratio compares like with like.

**`gaussian_entropy(std)`**: `sum_k(0.5 + 0.5*log(2*pi) + log std_k)`. Used for the optional entropy bonus (`--ent-coef`, 0 by default).

### Teachers
**`PidTeacher`**: a batched per-distance PID that labels the states the *student* is in (DAgger-style).
```
__init__:  build a TorchPIDController for all envs; stack the per-distance gain rows from best_pid_gains_per_dist.json into a table
reset(env_ids):
    for each env: pick the table row whose key distance is nearest to env._start_dist      # vectorised argmin
    write those gains into the PID; reset its integrators
label():
    read pos, vel, ang_vel, roll/pitch/yaw, target from the Isaac state
    a = pid.compute_action(...)                      # raw env-unit action
    return clamp( 2*(a-low)/(high-low) - 1, -1, 1 )  # normalised like tanh(mean)
```
It is queried every step, even when its action is not flown, so its integrators stay in step with the drone.

**`make_los_label(env, teacher, max_speed, accel, cruise_gain, stop_speed, max_tilt)`** (`--teacher los`): a scripted controller that
flies the straight line of sight. It returns a function `obs -> label in [-1, 1]` (the obs argument is ignored; the state is read from
the sim).
```
accel  = given, else 0.5*g*tan(max_tilt)           # refuses max_tilt > 90 deg: tan < 0 would make every label NaN
v_cmd  = clamp(sqrt(2*accel*dist), stop_speed .. max_speed)      # braking-limited speed; stop_speed >= max_speed = never brake
accel_cmd = cruise_gain * (unit_to_target*v_cmd - vel)           # velocity tracking on all 3 axes
thrust = hover*(1 + accel_z/g)/cos(tilt) - hover ; des_roll/pitch = clamp(accel_body/g, +-max_tilt)
torques from the SHORT-range PID attitude gains ; normalise with teacher.low/high
```

**`load_anchor(args, env, device, teacher)`**: picks the distillation label source from `--teacher` (`resolve_teacher`): `pid` -> `None`
(use the PID's labels, speed-faded), `los` -> `make_los_label(...)`, `policy` -> a frozen previous actor (`--anchor-checkpoint`, else
`--bc-checkpoint`) returning `tanh(mean)`. `auto` = `policy` if a checkpoint was passed explicitly, else `los`. Non-PID teachers get no
speed fade (weight 1).

**Teacher check and teacher BC (in `run`).** With a non-PID teacher, `run` first flies the teacher alone through the handoff
(`isaac_eval(..., actor=anchor)`) and prints `TEACHER CHECK` (endings), `teacher flight` (speed, closest approach p10/p50/p90) and a
few `trace` lines; under 50 % hits it warns. `--teacher-check-only` stops there. Then `--teacher-bc-chunks` chunks of supervised DAgger:
the teacher flies a fraction beta of the envs (1 -> 0 over 70 % of the chunks), all policy-flown states are labelled, the buffer is
aggregated (cap 4M) and the actor regresses `tanh(mean)` onto the labels with AdamW at `--teacher-bc-lr`. A single non-finite label
makes the loss NaN and the weights NaN on the first step, so check the teacher first (see `rtl-handoff.md` section 20).

### Handoff stage
**`install_handoff_stage(u, ...)`**: monkey-patches the env so every episode starts like a real controller-to-policy
handoff. It wraps `u._reset_idx` and adds `_refresh_handoff`, `_handoff_snap_obs` and (spawn-leg mode) `_carrot_action`.
Full detail and history: `rtl-handoff.md` sections 7, 15, 19.

*Wrapped `_reset_idx(env_ids)`*
```
orig_reset(env_ids)                                     # the env's own spawn + random target
h      = U(dist_low, dist_high)                          # distance LEFT when the policy takes over
spawn  = max(U(spawn_low, spawn_high), h + 0.5)          # spawn-leg mode: PID always flies >= 0.5 m
L      = spawn - h                                       # leg length (legacy mode: U(prefix_low, prefix_high))
unit   = direction spawn -> target, elevation clamped to [-(z-0.5)/total, sin(leg_max_elev)]
if recenter: shift the spawn back so the HANDOFF POINT lands at x=y=0     # observation holds absolute x,y
target = spawn + unit*(h + L), z >= 0.5
speed  = bucket(10-25% / 25-75% / 75-100% of max_speed, probs .2/.4/.4)
spawn-leg mode:  spawn at rest + level; plan the trapezoid profile (ramp up, cruise, ramp down to `speed`)
                 with a = 0.5*g*tan(max_tilt); store vpeak, d_up, d_cruise, unit; set a fallback timer
                 staggered full reset (training only): start at a random TIME along the leg, already moving
legacy mode:     spawn already moving at `speed` + random lateral velocity + random roll/pitch; timer = L/speed
write pose + velocity into the sim; handed_off = False
```
*`_refresh_handoff()`*: after every env step, latch `_handed_off = True` for an env if the real distance to target <= its h,
or the drone has flown the whole planned leg along the line, or the fallback timer ran out. Once latched it stays latched
until the episode ends.
```
handed |= (timer <= 0)
if spawn-leg mode: handed |= (dist_to_target <= h) | (progress_along_line >= L)
```
*`_handoff_snap_obs(obs, ids)`*: at the handoff step, overwrite the real velocity (toward target at the bucket speed plus a
random sideways part) and attitude (random roll/pitch up to `--handoff-tilt-deg`), and patch the same numbers into `obs` so
the policy sees a consistent state. This is the "imperfect real handoff".
```
vel  = unit_to_target*speed + random_perpendicular*(lateral_frac*speed*U(0,1))
quat = from_euler(U(-tilt,tilt), U(-tilt,tilt), 0)
write vel, quat to sim
obs[3:6]=vel/VEL_SCALE ; obs[6:10]=quat(x,y,z,w) ; obs[10:13]=0 ; obs[21:23]=sin/cos(yaw_err)
```
*`_carrot_action(...)`* (spawn-leg mode): the scripted controller that flies the leg. A position-tracking PID would brake
near any target, so instead it tracks a *velocity profile* along the line.
```
s      = distance flown along the line (read from position)
v_cmd  = sqrt(2 a s) | vpeak | sqrt(max(vpeak^2 - 2a(s - d_up - d_cruise), v_h^2))      # ramp / cruise / ramp down, >= 0.1
gains  = nearest per-distance PID gain row for the live remaining distance (re-picked every step)
accel_xy = cruise_gain*(unit_xy*v_cmd - vel_xy)                      # velocity tracking
thrust_delta = kp_pos*z_err - kd_pos*vel_z                           # altitude hold
des_roll/pitch = clamp(accel_body / g, +-max_tilt)                   # accel -> tilt
torques = kp_att*(des - angle) - kd_att*ang_vel ; yaw torque to desired yaw
return clamp([thrust_delta, tau_x, tau_y, tau_z], action bounds)
```

### Evaluation
**`isaac_eval(env, model, teacher, steps, use_pid=False)`**: deterministic score of the real task on freshly reset envs.
Only each env's FIRST episode counts. `use_pid=True` flies the whole episode with the PID (baseline).
```
obs = env.reset(); zero episode counters; reset teacher + stall counters
budget[env] = steps_for_dist(start_dist) / 4                       # per-distance time budget in policy steps
for i in range(steps):
    a_student = scale_action(mean)                                 # no noise
    a = PID action if use_pid
        else where(env not handed off yet, carrot/PID action, a_student)       # controller flies the leg
    env.step(a); refresh handoff; at the handoff step record real distance/speed/tilt, then apply the snap
    code = hit | hover | oob | roll | pitch | timeout (first reason wins)
    outcome[env] = code the first time it is non-zero; budget exhausted -> timeout
    stop early when every env has an outcome
return fractions per outcome, per-speed-bucket hit/crash, and the "handoff check" statistics
```

### Rollout, advantages, update
**`collect_rollout(env, model, teacher, obs, num_steps, gamma, bootstrap_timeouts, ep_ret, anchor)`**: one chunk of
on-policy data.
```
for t in range(num_steps):
    mean, std, value = model(obs)
    raw = mean + std*noise ; logp = log_prob_raw(raw) ; a_student = scale_action(raw)
    pid_label = teacher.label()                                    # always queried
    teach     = anchor(obs) if anchor else pid_label               # label for the distillation loss
    if handoff stage:
        ctrl = not handed_off                                      # controller still flies this env
        a = where(ctrl, controller action, a_student) ; pm = (not ctrl)   # pm=0 -> no policy gradient on this step
    obs', r, terminated, truncated = env.step(a)
    refresh handoff; apply the handoff snap to envs that just handed off
    if bootstrap_timeouts: r += gamma*V(terminal_obs) for time-limit-only endings
    store (obs, raw, logp, r, value, done, teach, pm) ; track episode returns ; reset the teacher for ended envs
return batch (+ last_value = V(last obs)), stats
```
`done = terminated | truncated`. Timeouts are real terminals unless `--bootstrap-timeouts`.

**`compute_gae`** (in `app/guidance/train.py`): backwards sweep `delta_t = r_t + gamma*V_{t+1}*(1-d_t) - V_t`,
`A_t = delta_t + gamma*lambda*(1-d_t)*A_{t+1}`, `R_t = A_t + V_t`. `done` cuts the chain at episode ends.

**`explained_variance(pred, target)`**: `1 - Var(target-pred)/Var(target)`; NaN when the target has no variance. Computed on
the rollout's *old* values vs the returns, so it measures the critic before this update.

**`ppo_distill_update(...)`**: the learning step.
```
adv, ret = compute_gae(...) ; adv_n = normalise(adv)
flatten batch to T*N samples ; mb = n // num_minibatches
repeat `epochs` times (warmup_epochs when only the critic trains):
    shuffle ; for each minibatch:
        mean, std, v = model(obs) ; value_loss = mean((v-ret)^2)
        loss = vf_coef*value_loss                                   # only if train_critic
        if train_actor:
            ratio   = exp(log_prob_raw(raw) - logp_old)
            pg      = mean over pm==1 of -min(ratio*adv_n, clip(ratio,1-eps,1+eps)*adv_n)
            distill = mean( tw * mean_k (tanh(mean)-teach)^2 )
            loss   += pg - ent_coef*entropy + distill_coef*distill
        backward ; clip actor grads and critic grads separately to max_grad_norm ; optimiser step
    if mean KL of this epoch > 1.5*target_kl: stop early
return averaged losses, approx_kl, clip_frac, grad norms, epochs_run, explained_var, ...
```
With `train_actor=False` (critic warmup) the actor gets no gradient at all and the actor-side stats are NaN.

**`teacher_weight(obs, fade_speed, vel_scale)`**: per-sample weight `clamp(1 - speed/fade_speed, 0, 1)`, with the speed read back
from `obs[3:6]*VEL_SCALE`. A hover PID commands braking at speed, which fights "hit the target", so the teacher fades out
as the drone gets fast. Not used with an anchor (weight 1).

**`teacher_agreement(model, batch)`**: diagnostic only. Deterministic student action vs the stored teacher label on the rollout
states, in chunks of 262144 rows: mean absolute gap per action dimension (`agree_mae*`), fraction of saturated outputs
(`|tanh| > 0.98`, `sat_frac*`), and the mean absolute teacher label.

### Schedules and phases
```
phase_of(chunk): settle (chunk < settle) -> critic_warmup -> head_only -> full
apply_phase(model, phase):
    trunk.requires_grad  = (phase == full)
    head + log_std       = phase in (head_only, full)
    critic               = phase != settle
cosine_factor(progress, min_ratio) = min_ratio + (1-min_ratio)*0.5*(1+cos(pi*progress))   # LR decay for actor groups
distill_scheduled(idx, start, end, decay_chunks) = start + (end-start)*min(1, idx/decay_chunks)   # idx counts actor-trainable chunks
```

### Logging helpers
`_py(v)` converts numpy/torch scalars to plain Python for CSV/JSON. `RunLog.add(row)` keeps all rows in memory and rewrites
`metrics.csv` with the union of all keys, so a metric that exists only in some phases never breaks the file. `plot_summary`
draws the 3x2 `summary.png` (hit rate, episode endings, teacher agreement, lingering, reward components, optimisation).
`steps_str(sps)` formats steps/s.

### Setup helpers
**`load_student(args, env, device)`**: builds an `RTLActorCritic` with the checkpoint's architecture; if `--bc-checkpoint` exists,
loads the actor weights (the critic stays fresh); otherwise the student is random and an imitation stage is forced.

**`_run_imitation_stage_isaac`** (in `base_training_isaac.py`): the on-policy DAgger stage that runs before PPO when there is no
checkpoint (or `--imitation-fraction > 0`).
```
loop for `steps_per_env` steps:
    student acts deterministically ; PID labels the same states ; store (obs, pid_action)
    every `retrain_every` steps: add the block to the aggregate (older blocks down-weighted by RECENCY_DECAY^age,
                                 subsampled when over the cap) and retrain the actor by weighted BC (MSE)
```

### `run(env, args)`: the main loop
```
1  install reward_qv2 or reward_rc (+ threats) ; set obs-position mode ; clear target pairs ; set target distance range
2  load per-distance PID gains ; install handoff stage (--handoff) or constant-speed spawn
3  model = load_student ; imitation stage if needed ; set_initial_std
4  AdamW optimiser: groups trunk / head (lr_actor) and critic (lr_critic) ; PidTeacher ; anchor = load_anchor ; MLflow run ; run_config.json
5  non-PID teacher: TEACHER CHECK ; staggered full reset ; teacher.reset ; optional teacher-BC stage ; drop stats from reset/imitation
6  for chunk in range(n_chunks):
       phase = phase_of(chunk) ; apply_phase ; per-group LR = base * cosine (actor groups) * ramp-up since unfreeze
       coef_eff = min(distill_max, distill_scheduled * boost) while the actor trains, else 0
       batch = collect_rollout(...) ; tw = 1 (anchor) or teacher_weight(obs) ; agree = teacher_agreement
       if phase != settle: upd = ppo_distill_update(...)
       every eval_every chunks: isaac_eval (+ PID baseline every pid_eval_every) ; reset envs (staggered) ;
                                 burn in --eval-burnin noisy steps and discard them
       guard: untouched policy's eval sets the bar `best_ma`; at eval chunks
              eval < best_ma - margin -> boost *= gain (cap)   else best_ma = max(best_ma, eval), boost = max(1, 0.9*boost)
       abort if train hit rate (chunks with enough finished episodes) stays below --abort-hit-rate for the window
       save model_<steps>.pt/.onnx + latest_full.pt ; best.pt = highest eval hit (only while the actor trains)
       log row to CSV / JSON / MLflow ; print the console lines ; abort on non-finite losses
7  write summary.json, summary.png, legacy plots ; close MLflow
```
**`main()`** parses args, starts the Isaac app (`AppLauncher`), registers and creates `Isaac-Base-Drone-Direct-v0`, runs `run`,
prints the traceback before shutdown (Kit can hard-exit and swallow it), and closes the env.

## Method reference: acro_rtl_train_isaac.py
Every function and class of the acro trainer, in file order. It reuses from the angle trainer: `RTLActorCritic`, `RunLog`,
`apply_phase`, `phase_of`, `cosine_factor`, `distill_scheduled`, `log_prob_raw`, `ppo_distill_update`, `set_initial_std`,
`teacher_agreement`, `_py`, `steps_str` (see the section above). The reward is `acro_reward_v1.py` (see
`app/reward_functions/docs.md`). Action is `[thrust_delta, p, q, r]` (body rates in rad/s), 23-dim observation.

**`AngleTeacher(path, env, device)`**: a frozen angle-style policy used as the teacher for the acro student.
```
__init__:  infer the architecture from the checkpoint ; assert obs/action dims match the env ;
           load into a plain ActorCritic (dropout 0) ; freeze all parameters
label(obs):
    a = scale_action(mean(obs))                               # angle action: [thrust_delta, tau_x, tau_y, tau_z]
    w = current body angular velocity
    cmd = w + a[1:4] / (inertia * kp)                         # the rate loop outputs I*kp*(cmd-w), so this reproduces tau
    act = clamp([a[0], cmd], acro bounds)
    return normalise(act) in [-1, 1], act in acro units
```
Approximations: only the first physics substep's torque matches (the rate error shrinks over the 4 substeps); the rate loop
clips torque to its feasible limit and the label is clamped to the acro rate bounds, so very large teacher torques are
under-represented; the thrust delta is passed through unchanged (both styles use +-hover thrust).

**`angle_to_acro(u, a, low, high)` / `controller_action(env, u, low, high)`**: the torque -> rate conversion (shared by `AngleTeacher.label` and the handoff leg) and the
acro action the carrot controller wants right now (full batch).

**`collect_rollout(env, model, teacher, obs, num_steps, gamma, bootstrap_timeouts, ep_ret, beta=0)`**: one chunk of data. In `--handoff` mode the
controller's action replaces the policy's for envs not yet handed off (`pm = 0`), and the takeover snap is applied after the step.
```
for t in range(num_steps):
    mean, std, value = model(obs) ; raw = mean + std*noise ; logp = log_prob_raw(raw) ; a = scale_action(raw)
    pm = 1
    if teacher:
        label, a_teacher = teacher.label(obs) ; store label                       # label for every visited state
        if beta > 0: use_t = rand(N) < beta ; a = where(use_t, a_teacher, a) ; pm = (not use_t)   # DAgger mixing
    env.step(a) ; optional timeout bootstrap ; store (obs, raw, logp, r, value, done, pm) ; track returns
return batch, last obs, {ep_return_mean, teacher_step_frac}
```
`beta > 0` is used only in the BC stage; PPO chunks always run with `beta = 0`, so `pm = 1` everywhere there. If `beta` were
used in PPO, the teacher-flown steps would still enter GAE, and their rewards would leak into the neighbouring student
advantages.

**`install_stagger(u, speed)`**: wraps the env's `_reset_idx`. When `u._stagger_next` is set and the reset is a full one, every
env is moved to a random point 0-70% along the spawn->target line, moving toward the target at U(0, `--handoff-speed`) m/s.
Partial and eval resets are untouched.

**`reset_and_desync(env, model, teacher, args, ep_ret)`**: fresh reset after an eval (which resets every env). The env itself
staggers `episode_length_buf` on a full reset, so envs time out at different steps; this function keeps that stagger (an
earlier version zeroed the counter, which put every env in lock-step and made the train stats saw-tooth every eval).
It sets `u._stagger_next` around the reset: each env then begins at a random point on the spawn->target line
(0 to 70% of the way), already moving toward the target at U(0, `--handoff-speed`) m/s (default 1.0; 0 = from rest), like the
angle trainer's staggered full reset. Eval resets are never staggered. It clears the stall/away counters, optionally burns in
`--eval-burnin` noisy steps (default 0, discarded), and optionally runs a `--desync-steps` window (default 0 = off, discarded)
in which every env is force-reset once at a random step, which spreads episode *starts* further. Finally it clears the return tracker and the reward statistics. A timeout in the first, shortened episode is handled by
the reward (see `reward_functions/docs.md`, acro_v1). `isaac_eval` calls `zero_acro_clock` so it scores true first episodes.

**`DaggerBuffer(cap)`**: the "D" in DAgger.
```
add(batch, per_chunk):  pick a random per_chunk subset of this chunk's (obs, teacher label) pairs and append
                        while total > cap and more than 1 chunk: drop the oldest chunk
```
States the student reached by drifting stay in the set, so later updates cannot forget them.

**`bc_train(model, optimizer, buf, steps, minibatch)`**: supervised regression on the aggregate.
```
obs, label = concatenate all chunks
repeat `steps` times: idx = random minibatch ; loss = mean( (tanh(mean(obs[idx])) - label[idx])^2 ) ; clip grad 1.0 ; step
return (mean loss, per-dimension loss [thrust, p, q, r])
```
Only `tanh(mean)` is trained (log_std receives no gradient), so exploration noise is untouched.

**`isaac_eval(env, model, steps, tilt_limit_deg, noisy=False, teacher=None)`**: like the angle eval, without a handoff or PID baseline. With a `teacher` it flies the teacher's converted labels instead (the teacher-alone baseline printed before BC).
```
reset envs, zero counters ; budget[env] = steps_for_dist(start_dist)/4 policy steps
for i in range(steps):
    a = scale_action(mean)  (or mean + std*noise when noisy=True)
    env.step(a) ; read pre-reset values from extras["acro_step"] (dist, speed, tilt) for envs still alive
    track closest approach, last distance, mean speed, max tilt
    outcome = first of hit | hover | oob | tilt | timeout ; budget used up -> timeout
return outcome fractions, hit_time_s, tilt_max_mean_deg, over_limit_frac,
       for non-hits: mean closest approach, mean final distance, mean speed
```
`noisy=True` evaluates the policy with its training noise, to tell a policy that needs its noise from one that does not.

**`plot_summary(rows, out_path)`**: 2x2 `summary.png`: hit rates (train and eval), how episodes end, tilt statistics, reward
components per step.

**`load_student(args, env, teacher, device)`**: builds the acro student.
```
src = --bc-checkpoint if that file exists else (the teacher checkpoint if a teacher is given) else None
if src: take the architecture from it ; load the actor weights ; always overwrite action_low/high with THIS env's bounds
if src is the teacher: zero the actor head (tanh(0) = middle of every range = hover) and keep the trunk features
if no src: zero the actor head ; trunk stays random ; the student starts as a hover policy
critic is always fresh
```
A `--bc-checkpoint` path that does not exist stops the run with an error.

**`build_parser()`**: flags are grouped as: task (`--distance-*`, `--obs-position-mode`, `--spawn-speed-*`, `--episode-s`),
hit zone (`--hit-radius R` / `--hit-zone R` = `--reward-json '{"hit_threshold": R}'`, default 0.25 m, plus `--zone-time T` s inside it before the hit counts, default 0; the viewer / `record_run --acro` read it back from `run_config.json`, `--hit-radius` overrides), tilt curriculum (`--tilt-limit`, `--tilt-margin`, `--reward {v1,v2,v3,v4}` (v2 adds the rc LOS penalty, v3 the speed-profile / cross-track / attitude terms, v4 the swept hit / per-metre cross-track with lead / path bonus), `--reward-json`), weights (`--teacher-checkpoint`, `--bc-checkpoint`),
BC stage (`--bc-*`), phases, PPO, distillation + guard, abort / stage gate (`--stop-hit`), staggered start (`--handoff-speed`,
`--desync-steps`), eval and output.

**`run(env, args)`**
```
Setup
  install acro reward (tilt limit, margin, overrides) ; optional shorter episodes ; obs mode ; target distance range
  teacher = LosTeacher | AngleTeacher | None (--teacher) ; model = load_student
  bc_chunks = --bc-chunks, or 6 with a teacher and no --bc-checkpoint, else 0

Stage A: DAgger-BC (only if bc_chunks > 0)
  buf = DaggerBuffer ; set_initial_std ; AdamW on actor params
  for c in range(bc_chunks):
      beta = bc_beta_start * max(0, 1 - c/bc_beta_chunks)          # fraction of envs the TEACHER flies; decays to 0
      batch = collect_rollout(beta) ; buf.add(batch) ; loss = bc_train(buf)
      every bc_eval_every chunks: ev = isaac_eval (student alone) ; keep the best student (bc_best.pt) ;
                                  reset_and_desync
      stop early once beta == 0 and the best eval hit >= --bc-target-hit
  restore the best student's actor weights ; reset envs ; clear stats

Stage B: PPO + fading distillation (same shape as the angle loop)
  set_initial_std ; optimiser AdamW with groups trunk / head / critic
  for chunk in range(n_chunks):
      phase = settle -> critic_warmup -> head_only -> full ; per-group LR (cosine + ramp)
      coef_eff = min(distill_max, distill_scheduled * boost) while the actor trains AND a teacher exists
      batch = collect_rollout(beta=0) ; tw = 1 ; stats = pop_acro_stats ; agree = teacher_agreement
      if phase != settle: ppo_distill_update(...)
      every eval_every chunks: deterministic eval + noisy eval (+ teacher alone every pid_eval_every) ; reset_and_desync
      guard (teacher runs only): same rule as the angle trainer, driven by eval hit
      windowed success: pool the last chunks until >= --success-window finished episodes
      abort on low train hit (if enabled) ; stop the stage when eval hit >= --stop-hit for --stop-patience evals
      save model_<steps>.pt/.onnx, latest_full.pt ; best.pt = highest eval hit (ties -> faster hits)
      log row to CSV / JSON / MLflow ; [rtl] console lines (same layout as the angle trainer)
  write summary.json and summary.png
```
Stage workflow: stage 1 uses the LOS teacher (or `--teacher-checkpoint`); later stages use `--bc-checkpoint <previous best.pt>`, a wider
`--tilt-limit`, no teacher, and a re-opened `--init-std`.

**`LosTeacher(env, gains_by_dist, args)`**: builds a `PidTeacher` only for its short-range gain table and angle-bounds PID, and calls
`make_los_label` with a stand-in teacher whose `low/high` are the ANGLE bounds (the acro env's own bounds are rates). `label(obs)`
denormalises the angle label, converts it with `angle_to_acro`, and returns (normalised acro label, acro action), the same interface
as `AngleTeacher`.

**`main()`**: parses args, starts the Isaac app, creates `Isaac-Acro-Drone-Direct-v0`, runs `run`, prints the traceback before
shutdown, closes the env.

## base_training_isaac
These notes describe `train()`, the earlier stage trainer.
- **Imitation -> critic warmup -> PPO**, in one call on a pre-built env. `IMITATION_FRACTION` is 0.08; it was 0.15,
  which is 19.2M steps of pure imitation at 128M.
- **`actor_log_std` is reset after imitation.** BC's Gaussian NLL against a deterministic teacher drives std to the
  floor. Resetting only after loading the checkpoint didn't survive the on-policy imitation stage, which collapsed it
  again. `tanh(0)` puts std at the midpoint of its range, where PPO still has gradient.
- **PID-residual mode** (`residual_scale > 0`, `PidResidualEnv`): `env_action = pid + residual_scale * policy`. A policy
  that outputs about 0 *is* the PID, so PPO starts at PID performance, and drift and exploration are bounded by the
  scale. `actor_mean` is zeroed unless `residual_resume` is set, because a direct-policy head used as a correction would
  double the PID. Imitation is skipped.
- **Floors, not fractions.** `MIN_WARMUP_CHUNKS = 16` was raised from 8 after a run still collapsed post-unfreeze.
  `MIN_IMITATION_RETRAIN_ROUNDS = 8` keeps a large `num_envs` from shrinking the retrain rounds.
- **Schedules.** `training_progress_at` is 0 through warmup, then 0 -> 1 over PPO only. Previously, entropy hit its
  floor about 10 chunks into warmup and the LR had decayed 14% before unfreeze. The LR is flat through warmup and then
  cosine over the *full* warmup+PPO span: a training-only cosine ran about 45% higher LR mid-run and drifted more (mean
  grade 0.648 vs 0.731 over chunks 12-40).
- **Soft reset.** When the trailing grade drops `RESET_DEGRADE_MARGIN` below the best, the weights blend back toward the
  best checkpoint and the optimizer is reinitialised. That is followed by `RESET_WARMUP_CHUNKS` of critic-only training.
  A margin of 0.25 fired 4 resets in `base_training_isaac_3_10_v1`, each recovering only to 0.75-0.83 while pulling the
  critic back. 0.4 makes it a collapse-only safety net.
- **Promotion gate.** `SOLID_WINDOW` consecutive chunks, all at index `>= SOLID_MIN_CHUNK` (40, which counts warmup),
  must average at least `SOLID_GRADE` with none below the window minimum. The gate then writes `SOLID.json`. The index
  floor stops a checkpoint that is just the imitation policy (about 0.95 grade) from being promoted as if PPO had
  produced it.
- `N_DIAGNOSTIC_EPISODES = 30` here, because 10 was noisy enough to false-trigger the soft reset.

## config.py
- `PARAMS['dropout']` **must stay 0.** PPO's ratio and KL need old and new log-probs from the *same* deterministic
  function. With dropout on, every forward pass draws a new mask, and one run showed approx_kl of about 29 during
  critic warmup, while the actor was frozen and couldn't move at all.
- `gamma` comes from `rewards.GAMMA`, because potential-based shaping is only policy-invariant if the shaping gamma
  equals the PPO gamma.

## diagnostics / eval_matrix
- Outcome keys are the reward functions' exact reason strings (`"Hit"`). Compound reasons (`"attitude-ROLL+attitude-PITCH"`)
  are split, and each part is credited separately. `+` is also not allowed in MLflow metric names.
- `diagnose_with_pid` swaps in per-distance gains when `gains_by_dist` is given. One fixed gain set would understate
  the PID ceiling.
- `build_uniform_omni_eval_pairs` draws the distance once and only re-draws the **direction** when the target would be
  underground. Re-drawing both under-samples the far end of the range, because underground directions are more likely
  at long distances.
- The omni lattice scales each axis offset by `d / sqrt(len(axes))`, so the total distance is exactly `d`. The random
  omni builder covers skewed directions the lattice never produces.

## Depends on
`torch`, `numpy`, `mlflow`, Isaac Lab (trainers only). Internally: every other `app` package.
