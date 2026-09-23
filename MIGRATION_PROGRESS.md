# Isaac Lab migration progress

Tracker for the numpy -> Isaac Lab port. Update this file whenever work stops so
a resume doesn't need to re-derive status from scratch.

## 2026-09-16 reorg -- file paths below are historical, read this first

Everything Isaac-side was moved OUT of a top-level `app/isaac/` package and
into the folder each piece corresponds to, matching the existing
control/dynamics/environmental/guidance structure, and the drone task was
renamed from "interceptor drone" to match the numpy oracle's own name. Any
`app/isaac/...` path or `InterceptorDrone*`/`Isaac-Interceptor-Drone-Direct-v0`
name mentioned further down this file is the OLD location/name -- current:

| old | new |
|---|---|
| `app/isaac/tasks/interceptor_drone/interceptor_drone_env.py`, class `InterceptorDroneEnv`/`InterceptorDroneEnvCfg` | `app/environmental/base_drone_env_isaac.py`, class `BaseDroneEnvIsaac`/`BaseDroneEnvIsaacCfg` |
| `app/isaac/tasks/interceptor_drone/torch_physics.py` | `app/dynamics/torch_methods.py` |
| `app/isaac/tasks/interceptor_drone/torch_pid.py` | `app/control/torch_pid.py` |
| `app/isaac/train_isaac.py` (`isaac_ppo_train`) | merged into `app/guidance/train.py` (no isaaclab import -- safe merge) |
| `app/isaac/collect_demonstrations_isaac.py` | merged into `app/control/collect_demonstrations.py` as `collect_demonstrations_base_drone_isaac` (lazy isaaclab import inside the function, so the module still imports fine from the plain numpy venv) |
| `app/isaac/dagger_isaac.py` | merged into `app/control/dagger.py` as `dagger_base_drone_isaac` (same lazy-import treatment) |
| `app/isaac/scripts/*.py` | `scripts/isaac_lab/*.py` (repo root, not under `app/` -- these are launcher entrypoints, not library code) |
| gym id `Isaac-Interceptor-Drone-Direct-v0` | `Isaac-Base-Drone-Direct-v0` |

Also in this pass:
- **Continuous omni/multi-axis target sampling**, not just the fixed
  `build_omni_eval_pairs` lattice: both `collect_demonstrations_base_drone_isaac`
  and `dagger_base_drone_isaac` now draw from a pool refreshed periodically
  with `sample_omni_target` (single/pair/triple-axis category sampling,
  proportioned to `CATEGORY_WEIGHTS` 20/30/50) instead of the fixed lattice --
  matches `collect_demonstrations_omni`'s sampling richness, not
  `collect_demonstrations`'s. Note: this is a STATISTICAL match to the
  category ratio (pool refreshed every `pool_refresh_every_rows` rows), not
  the numpy version's exact per-episode greedy catch-up -- an earlier attempt
  at exact per-env category assignment had a real staleness bug (category
  bookkeeping updated before the new target actually took effect on that
  env's NEXT reset) and was replaced with this simpler, correct pool design.
- **`decimation` raised from 1 to 4** (240/4 = 60Hz effective control rate).
  Verified live: `random_agent_smoke.py` still runs clean (100 steps, 16
  envs), `train_smoke.py` still trains (0 NaN, 63 episodes completed in the
  same timestep budget -- more than before, since each policy step now
  covers 4x the sim time). `diff_test_trajectory.py` was updated to loop the
  oracle's `timestamp_update` `decimation` times per Isaac step and re-run:
  max pos error grew from 0.24mm (100 steps @ decimation=1) to 5.67cm (100
  steps @ decimation=4, i.e. 400 physics ticks of accumulated integrator
  divergence instead of 100) -- consistent with the same benign PhysX-vs-
  oracle integrator difference as before, just measured over 4x more
  simulated time (1.67s), not a regression. `rpm_err` (the pure torch-math
  layer) stayed at the same tiny magnitude regardless of decimation, as expected.
- **Distances stay (3, 10) for now.** Up to 250/3000m is explicitly deferred
  by the user, not a current blocker -- the per-env fixed
  `max_episode_length` gap noted in step 6 below only matters once distances
  beyond ~50-60m are actually used.
- **NUM_ENVS**: still not benchmarked live on this machine (RTX 5060 Ti
  16GB). Reasoning given to the user: this scene is about as light as PhysX
  gets (single free rigid body per env, no joints/articulation solver, no
  sensors, headless) -- lighter than IsaacLab's own Cartpole (which commonly
  runs 4096-8192 envs on 8-12GB cards) and far lighter than multi-link
  articulation examples (ANYmal/humanoid, which top out around 2048-4096 on
  similar cards). Estimate: likely comfortable at 8192, plausibly higher
  (12k-16k+) before VRAM is the limiter -- Python-level per-step loop
  overhead in `isaac_ppo_train` (not physics) is more likely to be the
  practical ceiling on throughput gains than memory. `scene.num_envs=4096` in
  `BaseDroneEnvIsaacCfg` is still just the unbenchmarked fallback default;
  every script passes its own `--num_envs`. Not yet empirically verified --
  offer to benchmark live (1024/2048/4096/8192/16384) if precision matters
  more than this estimate.

## 2026-09-16 (later same day): critic warmup + full phase pipeline + graphs + pybullet removed

**New file `app/training/base_training_isaac.py`** -- Isaac-native counterpart
to `app.training.base_training`, same proven phase split/order (imported
directly from that module, not reinvented): **15% imitation -> 1% critic
warmup -> 84% PPO**. Budget is expressed as a GRAND TOTAL env-steps across all
parallel envs (`TOTAL_TIMESTEPS_ISAAC`, default 128,000,000 -- matches
base_training.py's own grand total of 1,000,000-per-env * 128 workers, for a
directly comparable amount of experience), not per-env like the numpy
constant, since `NUM_ENVS_ISAAC` scales with GPU capacity rather than CPU
core count. Diagnostics/PID-baseline comparison and plotting reuse the numpy
oracle's existing `diagnose_with_model`/`diagnose_with_pid`/`plot_training_run`
**unchanged** -- both are policy-quality checks (N deterministic episodes),
not physics throughput, so no porting needed there; the numpy oracle is
diff-tested to track the Isaac physics closely (step 4).

**New in `app/guidance/train.py`**: `isaac_warmup_critic` (batched
counterpart to `warmup_critic`).

**Verified live** (`scripts/isaac_lab/full_pipeline_smoke.py`, 16 envs,
40,000-grand-total-step smoke budget, ~2 minutes): imitation stage collected
+ retrained (BC loss logged), critic warmup ran (value_loss logged), 7 PPO
chunks ran with a diagnostic + PID-baseline comparison every chunk (PID hit
rate 90-100% as expected from a tuned controller; the barely-trained policy
correctly shows 0% -- this is a mechanism check, not a convergence run), all
7 checkpoints saved as both `.pt` and `.onnx`, `metrics.csv` written, and all
**5 diagnostic plots generated**: `training_error.png` (policy/entropy/value
loss + grad norm), `policy_std.png` (exploration decay), `distance_
distribution.png` (RL vs PID-teacher final distance over training, converging-
or-not at a glance), `success_and_outcomes.png` (hit rate + failure-mode
breakdown), `vs_pid_baseline.png` (bar chart, RL final checkpoint vs tuned
PID, 4 metrics). mlflow tracked the run under experiment "base-training-isaac".

**Real bug found and fixed**: installing mlflow into the Isaac py3.11 venv
(it wasn't there before -- numpy-venv-only dependency) hit `mlflow.exceptions.
MlflowException: Detected out-of-date database schema` against the repo's
shared `mlflow.db` (created by an older mlflow version in the numpy venv).
**Did not run a schema migration on that shared file** -- risked corrupting
the numpy side's existing experiment history for a venv-version mismatch that
isn't really about the data. Fixed by pointing Isaac-side mlflow calls at a
separate store (`MLFLOW_TRACKING_URI=sqlite:///mlflow_isaac.db`, set at the
top of `base_training_isaac.py`, overridable). Also: `pip install mlflow` in
that venv bumped `starlette`/`prettytable` versions that conflict with
isaaclab's own pins (pip warns but doesn't block) -- verified live afterward
(`random_agent_smoke.py`) that Kit/training still runs correctly despite the
warning; not chased further since nothing actually broke.

**pybullet fully removed** (was only ever in the numpy oracle path -- the
Isaac side never used it): `app/environmental/enviorment.py`'s pybullet-only
`spawn_drone` deleted (kept `sample_wind_conditions`, which has no pybullet
dependency, updated the module docstring); `app/environmental/base_drone_env.py`'s
`render_mode="human"` PyBullet GUI path removed (`_init_render`,
`_spawn_visuals`, the pybullet `render()` body, the lazy `import pybullet`) --
`render_mode` param still accepted for signature compatibility, now always a
no-op; verified live afterward that the numpy oracle env still steps
correctly with no pybullet installed in the checking venv. Moved (not
deleted, per project convention) three already-retirement-flagged scripts
that depended on the PyBullet GUI: `app/guidance/watch_hover.py`,
`app/guidance/test_free_hover.py`, `app/demo/demo.py` -> `deprecated/guidance/`,
`deprecated/demo/`. Fixed `app/main.py`'s now-broken top-level
`from app.demo.demo import simulation` import (menu option 1 now prints a
pointer to `scripts/isaac_lab/*.py` instead of crashing on import).
`pyproject.toml`'s `pybullet; sys_platform == 'linux'` line removed.
**Not touched**: `uv.lock` still references pybullet -- it's an
auto-generated lockfile that should be regenerated via `uv lock`, not
hand-edited; not done here since `uv run`/`uv lock` are known broken on this
Windows machine for this project's venv (see project-isaac-lab-migration
memory: the uv venv is a WSL2 Linux venv, `uv lock` hits a symlink
`Access is denied` error on native Windows). Run `uv lock` from WSL2 to
actually refresh it.

## 2026-09-16 (evening): imitation-stage round floor + warmup mechanism fix

Two real bugs found via the user pushing back on "is this actually a good
idea" rather than accepting the first pass:

1. **Imitation retrain cadence was a fixed per-env constant (512 steps).**
   That silently shrinks the number of "collect on-policy, retrain" ROUNDS as
   `NUM_ENVS_ISAAC` grows (round count, not raw pair volume, is what avoids
   catastrophic forgetting -- see `_run_imitation_stage_isaac`'s docstring).
   Fixed: retrain cadence is now DERIVED from `MIN_IMITATION_RETRAIN_ROUNDS`
   (floor of 8, matching `base_training.py`'s own ~7.5 rounds), guaranteeing
   at least that many rounds regardless of env count.
2. **Critic warmup called the wrong mechanism.** `base_training_isaac.py`'s
   phase 2 was calling `isaac_warmup_critic` (a standalone MSE-regression
   loop) -- but `base_training.py` doesn't actually use `warmup_critic()` for
   this at all; it freezes the actor and runs NORMAL `vec_ppo_train` chunks
   (frozen params get skipped by `optimizer.step()`). Fixed: phase 2+3 now
   share ONE chunk loop using `isaac_ppo_train`, actor frozen for the first
   `warmup_timesteps_grand` worth of chunks then unfrozen -- genuinely the
   same mechanism as `base_training.py`, not just the same ratios.
   `isaac_warmup_critic` stays in `app/guidance/train.py` as a separate
   reusable tool, just isn't called by this orchestrator anymore.

**Verified live** (`full_pipeline_smoke.py`, 40,000-step budget): imitation
stage now runs 8 retrain rounds (was 1); PPO-chunk log correctly shows
`stage=critic_warmup` for the frozen-actor chunk(s) and `stage=training`
after, with an explicit "actor frozen"/"unfrozen" log line. No NaN, all
plots/checkpoints/mlflow still written correctly.

## Status

1. **flow.md** — DONE. Baseline numpy flow documented at repo root (action clip ->
   hover offset -> mixer inversion -> motor lag -> forces -> integrate -> obs ->
   reward -> done -> buffer -> GAE -> PPO update), plus constants that must survive
   the port (dt=1/240, obs scales, reward constants, drone physical params).
2. **Scaffold Isaac Lab env** — DONE (structure only; physics body is step 3).
   - Found existing install: `E:\Isaac\IsaacLab` (v2.3.2) + py3.11 venv `E:\Isaac\env_isaaclab`.
     `isaacsim` and `isaaclab` both import fine in that venv (install logs showed
     some noisy errors — EULA-prompt EOF, a `flatdict` build failure — but the
     actual packages ended up installed; verified live, not assumed).
   - New package `app/isaac/` (separate from `app/environmental`/`app/dynamics`,
     which stay alive as the numpy oracle): `app/isaac/tasks/interceptor_drone/`
     registers gym id `Isaac-Interceptor-Drone-Direct-v0` as a `DirectRLEnv`,
     modeled on IsaacLab's own `isaaclab_tasks/direct/quadcopter` template.
   - `InterceptorDroneEnvCfg`: `action_space=4`, `observation_space=23` (matches
     numpy `build_observation`), `sim.dt=PHYSICS_DT=1/240` fixed, `decimation=1`
     for now (kept at 1 so step 4's diff-test is 1:1 against the oracle;
     raise later as the speed knob per flow.md), `episode_length_s` derived
     from `MAX_POLICY_STEPS=15_000 * PHYSICS_DT` to match `BaseDroneEnv.max_steps`.
   - Robot asset started as a **placeholder** (`CRAZYFLIE_CFG`) — since replaced
     in step 3 with a procedural RigidObject sized to match `QuadConfig` natively
     (see step 3 detail below).
   - `_pre_physics_step` / `_apply_action` / `_get_observations` / `_get_rewards`
     / `_get_dones` / `_reset_idx` are stubbed (`raise NotImplementedError`),
     each with a TODO citing the exact flow.md section/step it must reproduce.
   - `app/isaac/scripts/zero_agent.py`: standalone launcher (AppLauncher boot
     -> gym.make -> zero-action step loop) to smoke-test the scaffold once
     step 3 fills in the physics. Adapted from IsaacLab's own zero_agent.py.
   - Verified so far: all 5 new files `py_compile` clean under the py3.11 venv;
     `interceptor_drone_env.py` imports up through `isaaclab.sim` and fails
     exactly at `pxr` (USD bindings only resolve inside a running Kit process,
     via `AppLauncher` — same as every IsaacLab task file), which is the
     expected/correct failure point, not a scaffold bug.
   - **Verified live** (headless, `--num_envs 2`, run directly via
     `E:\Isaac\env_isaaclab\Scripts\python.exe app/isaac/scripts/zero_agent.py`
     -- NOT `isaaclab.bat -p`, which on this machine picks the wrong `python`
     off PATH since the venv isn't conda-activated; call the venv's python.exe
     directly, or `set PYTHONPATH`/activate the venv first): Kit booted
     (RTX 5060 Ti, driver 591.86), scene created, printed
     `Gym observation space: Box(-inf, inf, (2, 23), float32)` /
     `Gym action space: Box(-inf, inf, (2, 4), float32)` (matches the numpy
     oracle's 23/4 dims), then hit our own `NotImplementedError("reset port
     pending -- migration step 3")` inside `_reset_idx` on `env.reset()` --
     the expected, correct stopping point. `zero_agent.py` needed one fix to
     get here: it inserts the repo root onto `sys.path` itself now (Kit's
     python.exe resolves `sys.path[0]` to the script's own directory, not the
     repo root, so `import app.*` failed until that was added).
3. **Port physics pipeline to torch/PhysX** (decimation, not bigger dt) — DONE.
4. **Diff-test against numpy oracle** — DONE for the math+trajectory layers
   (max pos error 0.24mm over 100 steps). Random Uniform(3,10) target-sampling
   is still the reset-time default, but `target_pairs` support was added in
   step 6 for deterministic/eval-matrix target assignment when needed.
5. **Port PPO tensor-native, batched across NUM_ENVS** — DONE. See detail below.
6. **Port PID teacher + DAgger + collect_demonstrations onto batched env** — DONE. See detail below.

## Steps 3-4 detail (2026-09-16) — DONE, diff-test passing

**New files:**
- `app/isaac/tasks/interceptor_drone/torch_physics.py` — batched torch mirror
  of `app.dynamics.methods` (mixer inversion, motor lag, thrust/torque, drag,
  wind). **Diff-tested bit-exact against the numpy oracle**: `python app/isaac/
  scripts/diff_test_physics.py` — 64 random cases x 5 functions, atol=rtol=1e-9,
  ALL PASS. No Kit boot needed (pure torch/numpy/scipy).
- `interceptor_drone_env.py` — all 6 DirectRLEnv hooks (`_pre_physics_step`,
  `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`,
  `_reset_idx`) implemented: mixer/motor-lag/thrust/torque/drag/wind wired to
  torch_physics.py, `reward_func` (flow.md section 4) ported to per-env tensors
  (NOT `RewardConfig`/`base_reward_fn` — that curriculum system isn't used by
  current training), observation matches `build_observation`'s 23-dim layout
  exactly (including reordering IsaacLab's wxyz quat to the oracle's xyzw).
  Target sampling is still a placeholder (Uniform(3,10) distance, random
  direction, z-clamped) — `target_pairs`/offset-range config isn't ported yet
  (deferred to steps 5-6, where DAgger/collect_demonstrations need it).
- `app/isaac/scripts/random_agent_smoke.py` — reset + N random-action steps,
  checks for exceptions/NaN. **Verified live**: 50 steps x 4 envs, no NaN, no
  exceptions, sane reward values.
- `app/isaac/scripts/diff_test_trajectory.py` — runs the same fixed action
  sequence through the oracle and the live Isaac env from matching initial
  state, reports per-step position/velocity/orientation/rpm divergence.
  **Verified live, PASSING**: over 100 steps, max pos error 2.45e-4 m
  (0.24mm), max vel error 1.37e-3 m/s, max ang_vel error 2.73e-3 rad/s, max
  rpm error 1.42e-3 rad/s. Errors grow smoothly/slowly, consistent with the
  expected (benign) difference between PhysX's rigid-body integrator and the
  oracle's exact semi-implicit-Euler + scipy rotvec-exponential-map
  integration — not a bug.
- `app/isaac/scripts/debug_force_isolation.py` / `debug_stock_quadcopter.py` —
  isolate force-only vs torque-only wrench-composer application against a
  fresh reset, and cross-check against IsaacLab's own unmodified quadcopter
  example. Kept for future debugging if this ever regresses.

**Bug #1 (fixed): mass override didn't reach dynamics.** Robot mass/inertia
needs to match `QuadConfig` (mass=1.5, inertia=(0.02,0.02,0.04)). A
**runtime-only** override (`root_physx_view.set_masses`) silently doesn't
propagate to an Articulation's translational dynamics (readback via
`get_masses()` looked correct, but applied force still integrated using the
placeholder Crazyflie's real ~25g mass) — fixed at the time by moving the
mass override to **spawn/config time** (`robot.spawn.mass_props`) instead.
Superseded by bug #2's fix below (no override needed at all anymore).

**Bug #2 (fixed): force/torque under-delivered by the Articulation solver.**
Even after bug #1's fix, `debug_force_isolation.py` showed applied
force/torque via `permanent_wrench_composer.set_forces_and_torques` was
under-delivered: hover-thrust-only only produced ~20% of the expected
correction; torque-only only ~76%. Cross-checked against IsaacLab's own
**unmodified** quadcopter example (`debug_stock_quadcopter.py`) — that one
was exact, ruling out the composer API itself. Then tested our env with the
mass override removed entirely (native ~25g Crazyflie mass): delivery jumped
to ~89%, i.e. **the shortfall scaled with how far the mass override diverged
from the asset's native/cooked value** (60x override -> ~20% delivered; no
override -> ~89% delivered) — a floating-base Articulation apparently caches
some mass-dependent solver quantity at cook time that a post-hoc
`set_masses`/`mass_props` override doesn't fully invalidate.

**Root fix:** stopped overriding a mismatched placeholder asset entirely.
Replaced the Crazyflie `ArticulationCfg` with a plain procedural
`RigidObjectCfg` (`sim_utils.CuboidCfg`, no joints — we never used them,
everything is analytic single-rigid-body force/torque) sized `(0.4, 0.4,
0.02)` m so its own natural uniform-density inertia already matches
QuadConfig's `(0.02, 0.02, 0.04)` to within ~0.25% (`I=m/12*(edge_a^2+
edge_b^2)`), with `mass_props=MassPropertiesCfg(mass=1.5)` authored directly
at spawn. Nothing to override, nothing cached from a different starting
value. Re-ran `debug_force_isolation.py`: hover-thrust-only now gives
`1.1e-9 m/s` (exact 0), torque-only gives `0.0104 rad/s` (expected
`0.010417`) — **both ~100% delivered**. Re-ran the trajectory diff-test:
0.24mm max position error over 100 steps (numbers above).

**Consequence for the visual/geometry gap noted in step 2:** moot now — the
asset is our own authored shape, not a mismatched Crazyflie, so there's no
separate "arm_length doesn't match" caveat anymore.

## Step 5 detail (2026-09-16) — DONE

**New file:** `app/isaac/train_isaac.py` — `isaac_ppo_train(env, ...)`, same
interface/semantics as `guidance.train.vec_ppo_train` but drives a live
`DirectRLEnv` directly: torch tensors in/out, GPU-resident, no numpy
round-trip, no Python per-env loop (env.step() is already batched over
`num_envs`). Reuses `ActorCritic`/`compute_gae`/`ppo_update` from
`app.guidance.train` unchanged. Also `adaptive_kl_lr_step` (same scheme as
rl_games' AdaptiveScheduler: halve/double lr based on measured KL vs
target_kl each round).

**Changed file:** `app/guidance/train.py`'s `ppo_update` gained two
off-by-default flags (flagged inline with a comment, rest of the file
untouched): `value_clip_eps` (PPO2-style clipped value loss) and
`distill_target_actions`/`distill_coef` (plain MSE distillation term against
externally-supplied target actions — teacher-agnostic, PID or a network).
Both no-op when unset, so `vec_ppo_train`'s existing numpy-path behavior is
byte-for-byte unchanged.

**Real bug found and fixed** (`app/isaac/scripts/train_smoke.py`'s first
round hit it immediately): `InterceptorDroneEnvCfg.action_space` was a plain
int (`= 4`), which makes `DirectRLEnvCfg` auto-generate an **unbounded**
`Box(-inf, inf, (4,))` action space. `ActorCritic`'s tanh-squash rescale
(`action_low + (squashed+1)*0.5*(action_high-action_low)`) then computes
`-inf + finite*inf = NaN` on the very first action, before any `env.step()`
even runs. Fixed: `action_space` is now an explicit bounded
`gym.spaces.Box` matching `BaseDroneEnv`'s real bounds
(`[-hover_thrust,-0.5,-0.5,-0.5]` to `[hover_thrust,0.5,0.5,0.5]`); `__init__`
now derives `self._action_low/_action_high` FROM `self.single_action_space`
instead of independently recomputing them, so there's one source of truth.

**Verified live:**
- `train_smoke.py` (16 envs, 4096 timesteps, num_steps=64): 4 PPO rounds
  completed, 15 episodes finished naturally, losses finite and sane
  (policy_loss=-0.005, value_loss=0.127, approx_kl=0.0074 < target_kl=0.02),
  no NaN in final model.
- Flag-switchable extras (`value_clip_eps=0.2`, `adaptive_kl_lr=True`,
  `distill_teacher_fn=<dummy>`, `distill_coef=0.1`) all together: ran clean,
  no NaN, KL-based early-stopping engaged correctly (`early_stopped=True`
  when a round's approx_kl exceeded target_kl).

## Step 6 detail (2026-09-16) — DONE

**New files:**
- `app/isaac/tasks/interceptor_drone/torch_pid.py` — `TorchPIDController`,
  batched mirror of `app.control.pid.PIDController` with per-env integral
  state and per-env gains (`set_gains`/`assign_gains_by_distance`, mirrors
  `subproc_vec_base_drone_env._select_pid_teacher`'s nearest-distance gain
  lookup). **Diff-tested bit-exact** against the numpy oracle over 20 steps x
  32 envs of *stateful* recursion (integral terms carried across steps, not
  just one-shot): `python app/isaac/scripts/diff_test_pid.py`, atol=rtol=1e-5,
  ALL PASS. No Kit boot needed.
- `app/isaac/collect_demonstrations_isaac.py` — `collect_demonstrations_isaac
  (env, gains_by_dist, distances, n_target_rows, ...)`: batched demonstration
  collection, `num_envs` parallel episodes instead of
  `app.control.collect_demonstrations`'s sequential per-pair-per-episode
  loop. Saves the same `(obs, actions)` .npz shape
  `app.control.pretrain_bc.pretrain_behavior_cloning` already expects — that
  function is reused **unchanged**. **Verified live**: 4000 rows, 16 envs, no
  NaN, and the PID actually tracks toward targets (mean symlog(dist) trended
  0.466 -> 0.363 over the run, a real behavioral check, not just "didn't crash").
- `app/isaac/dagger_isaac.py` — `dagger_isaac(env, gains_by_dist, n_rounds,
  ...)`: batched on-policy DAgger (policy drives the drone via
  `env.step(policy_action)`, PID only supplies the `(obs, pid_action)` label),
  same recency-weighted aggregate-buffer/eviction/retrain structure as
  `app.control.dagger.dagger`. Reuses `ActorCritic`/`pretrain_behavior_cloning`
  unchanged. **Verified live**: one round against the repo's real
  `pretrained_bc.pt` + `demonstrations_omni.npz` (153k existing pairs),
  collected 3008 new rows, retrained, saved checkpoint, no NaN.

**Extended file:** `interceptor_drone_env.py` gained `set_target_pairs()`
(resolves the "target_pairs deferred" TODO from migration steps 3-4) and
`self.extras["term_reasons"]` (per-env boolean masks for hit/oob/attitude,
batched equivalent of the oracle's `info["reason"]` string — needed for
DAgger's hit-rate logging).

**Known simplifications (flagged in-file, not fixed this pass):**
- `collect_demonstrations_isaac`/`dagger_isaac` don't rebuild a per-distance
  `max_episode_length` the way the numpy versions rebuild a fresh
  `BaseDroneEnv(max_steps=steps_for_dist(dist))` per distance —
  `InterceptorDroneEnvCfg` has one fixed episode length (currently 62.5s/
  15000 steps) for every env, so a long-distance (e.g. 250m, needs 62500
  steps) episode gets truncated well before completion. Still yields usable
  BC pairs along the truncated trajectory, just fewer near-target ones for
  long distances.
- `dagger_isaac` doesn't do `app.control.dagger`'s explicit per-distance row
  balancing (subsample every distance down to the smallest distance's raw
  count before aggregating) — envs draw targets uniformly from the combined
  pool and contribute rows for however long their episode runs, so longer
  (bigger-distance) episodes naturally contribute more rows, unrebalanced.
- `TorchPIDController.set_gains`/`assign_gains_by_distance` don't touch
  `max_tilt_rad`/`action_low`/`action_high` per-env (only the kp/kd/ki gains)
  — harmless today since every distance bucket in
  `best_pid_gains_per_dist.json` uses the same `max_tilt_rad=0.3` and default
  action bounds, but would need extending if that ever stops being true.

## Old-file disposition rule (per user instruction)

- File mostly rewritten for the port -> move original to `deprecated/` (no commit).
- File only partly changed -> keep in place, flag the changed block with a comment
  explaining what changed and why (not a full move).

## Per-round/per-chunk full env.reset() bug (2026-09-16, found post-migration)

After migration completed and real pipeline runs started (`collect_demo_isaac.py`
-> `pretrain_bc.py` -> `dagger_isaac.py`), `dagger_base_drone_isaac`'s own
printed `hit_rate` collapsed to ~0.00 across all 5 rounds even starting from a
freshly, cleanly BC-converged checkpoint. Root-caused via systematic
isolation, not guessed:

1. `scripts/isaac_lab/diagnose_full_sphere.py` (numpy oracle, PID vs. student,
   full-sphere Uniform(3,10) targets) showed the student at ~90% hit rate,
   nearly matching PID's ~95% -- but this comparison was itself flawed
   (`BaseDroneEnv` has no `decimation`, so it queries the model every physics
   step, a finer/easier control rate than the real Isaac `decimation=4` the
   model actually trains/deploys under).
2. `scripts/isaac_lab/diagnose_pid_isaac.py` (new): PID teacher driven
   directly in the REAL Isaac env, decimation=4, full-sphere Uniform(3,10) --
   90.6% hit rate, 0% oob/attitude, 9.4% timeout. Ruled out "decimation=4 is
   fundamentally too coarse" as an explanation.
3. `scripts/isaac_lab/diagnose_model_isaac.py` (new): both `pretrained_bc.pt`
   (pre-DAgger) and `pretrained_bc_dagger.pt` (post-5-rounds), driven
   directly in the same real Isaac env, decimation=4, model.eval(),
   deterministic mean action -- 85.9% and 90.0% hit rate respectively,
   0%/0% oob/attitude. Essentially matching PID. The checkpoints were fine
   the whole time; DAgger's own 5 rounds of retraining even improved things
   slightly (85.9% -> 90.0%).

Conclusion: the ~0% hit_rate `dagger_base_drone_isaac` was printing was a
measurement artifact of the function itself, not a policy-quality problem.

**Root cause:** both `dagger_base_drone_isaac` (`app/control/dagger.py`) and
`isaac_ppo_train` (`app/guidance/train.py`) called `env.reset()`
unconditionally on every call -- `dagger_base_drone_isaac` at the top of
every round, `isaac_ppo_train` at the top of every function call (and
`base_training_isaac.train()` calls it once per PPO chunk, ~every 256
env-steps). `BaseDroneEnvIsaac._reset_idx`'s full-reset path (`len(env_ids)
== self.num_envs`) deliberately randomizes every env's
`episode_length_buf` uniformly across `[0, max_episode_length)` (standard
DirectRLEnv desync-on-full-reset behavior, so parallel envs don't all
terminate in lockstep). With a rollout window far shorter than
`max_episode_length` (dagger: `rows_per_round/num_envs` ≈ 122 steps vs.
3750-step episodes; PPO: `NUM_STEPS_PER_CHUNK` = 256 steps vs. the same
3750), most envs never got a chance to run a real episode before being
reset again, and a rollout-window-sized slice of envs landed within reach of
their (fake, randomly-assigned) ceiling and immediately timed out having
barely acted -- manufacturing a burst of spurious timeouts every round/chunk
regardless of actual policy quality, and (for PPO specifically) also
discarding reward-shaping state (`start_dist`/`prev_distance`/
`milestones_hit`) and fragmenting the GAE horizon every chunk.

**Fix:**
- `isaac_ppo_train` gained an `initial_obs` param (default `None` = old
  behavior, fine for a single standalone call like `train_smoke.py`) and now
  returns `(model, optimizer, episode_rewards, last_losses, final_obs)` --
  one extra element. `base_training_isaac.train()` threads a persistent
  `ppo_obs` variable through its per-chunk loop instead of discarding it, so
  the env is only ever reset once for the whole PPO phase (individual envs
  still reset normally on their own natural termination).
- `dagger_base_drone_isaac`'s `env.reset()` (+ initial `assign_gains_by_distance`/
  `pid.reset()`) moved from inside the round loop to once before it, same
  reasoning.
- **This means the PPO training phase (`base_training_isaac.py`, ~84% of the
  budget) had this same bug and had never actually been run for real yet** --
  caught before any expensive training time was spent on it, not after.

## Insufficient critic warmup collapses the policy post-unfreeze (2026-09-16/17)

First full real run of `train_isaac.py` (128M timesteps, num_envs=4096,
Uniform(3,10), warm-started from the now-good `pretrained_bc_dagger.pt`,
after the env.reset() bug above was fixed) completed and was reviewed via its
`metrics.csv` and `plots/`. Result: RL `success_rate` collapses from matching
the BC/DAgger starting policy to ~0% within the first few chunks after the
actor unfreezes, and never recovers for the remaining ~100 chunks -- final
checkpoint: `avg_final_dist≈8m` / `success_rate=0` / `avg_reward≈0`, vs. the
PID baseline's `avg_final_dist≈0.25m` / `success_rate≈1.0` on the exact same
target set the whole run (see `runs/base_training_isaac_3_10/plots/
vs_pid_baseline.png`, `success_and_outcomes.png`). `outcome_hover_success` is
0 in literally every one of the ~103 logged chunks; `outcome_hit` is nonzero
only in the first ~14 (matching the good starting policy), then 0 for the
rest; `outcome_attitude-ROLL` climbs to dominate (5-9/10 diagnostic episodes)
and never improves.

**Root cause:** `WARMUP_FRACTION=0.01` (1% of 128M = 1.28M grand-total steps)
worked out to exactly 2 PPO chunks at `num_envs=4096`
(`chunk_grand_steps=NUM_STEPS_PER_CHUNK*num_envs≈1.05M`) -- the same
fraction-collapses-at-large-NUM_ENVS pattern as the imitation-retrain-round
bug above, just for critic warmup instead of imitation retraining. `metrics.csv`
shows `value_loss` still ~14 (started ~27) at the exact chunk the actor
unfroze -- nowhere near converged. This is precisely the failure mode critic
warmup exists to prevent (per this module's own docstring): a garbage,
under-trained critic feeds bad advantage estimates into the very first real
PPO update the moment the actor unfreezes, and with `effective_std_mean`
staying pinned near its floor the entire run (ruling out an entropy/noise
explanation), that bad gradient pushed the policy's MEAN action somewhere
bad and it stayed there -- a near-deterministic policy just keeps executing
whatever mean it learned.

**Fix:** added `MIN_WARMUP_CHUNKS = 8` (same floor-not-fraction pattern as
`MIN_IMITATION_RETRAIN_ROUNDS`) -- `warmup_timesteps_grand` is now
`max(WARMUP_FRACTION * total_timesteps, MIN_WARMUP_CHUNKS * chunk_grand_steps)`,
guaranteeing the critic gets at least 8 real chunks' worth of update passes
before the actor unfreezes, regardless of `NUM_ENVS_ISAAC`. At `num_envs=4096`
this raises warmup's share of the 128M budget from ~1% (1.28M) to ~6.6%
(8.39M) -- a small tradeoff against the PPO phase's budget for an actually-
converged critic.

## Second contributing cause: step_penalty over-accumulates in Isaac, making a
## quick crash reward-preferred over floundering (found same review, user pushback)

User correctly pushed back on the warmup-only explanation, pointing out this
exact same "collapses right at unfreeze, never recovers" symptom already has
history in this codebase -- `rewards.py`'s own `HIT_REWARD` comment describes
an earlier, structurally identical failure (HIT_REWARD=1000 producing
heavy-tailed advantages) that was only partially mitigated by dropping it to
50. That was the right instinct to chase down instead of accepting the
warmup fix as the whole story.

Found a second, more mechanistic, quantitatively-confirmed bug:
`reward_func`'s `step_penalty = -(TARGET_FRACTION * APPROACH_MILESTONE_BUDGET)
/ steps_for_dist(env.start_dist)` is calibrated on the numpy oracle's
invariant that an episode truncates at exactly `steps_for_dist(dist)` PHYSICS
steps -- so a full non-converging episode always accumulates exactly
`TARGET_FRACTION*APPROACH_MILESTONE_BUDGET` (24.18) total step_penalty,
regardless of distance, by construction.

`BaseDroneEnvIsaac` ported this formula literally
(`self._steps_for_dist`, set per-env from `steps_for_dist(dist_mag)` in
`_reset_idx`) but applies it once per POLICY step (not physics step,
decimation=4), AND -- per the already-documented "known simplification" --
this env has ONE FIXED `max_episode_length` (3750 policy steps) for every
distance, not a per-distance truncation. So the per-step magnitude stayed
calibrated for a short, distance-scaled cap that Isaac never actually
enforces, while the REAL cap (3750) is much longer for short/medium
distances. Worst case at dist=3 (`steps_for_dist(3)=1800`,
`step_penalty=-24.18/1800=-0.01343/step`): a full 3750-step timeout
accumulates `3750*0.01343≈50.4` total step_penalty -- roughly equal to
`HIT_REWARD=50`, and ~2x the intended 24.18 ceiling. Compare a one-time
crash: `-1.0` (attitude) or `-1.5` (oob). Once a trajectory isn't cleanly
converging, the reward function was telling the policy that crashing
immediately is far cheaper than continuing to try -- a direct, sufficient
explanation for both the initial collapse AND why it never recovered on its
own over the remaining ~100 chunks (PPO gradients rationally chase whichever
is actually reward-maximizing, and once drifted into a non-converging
regime, that was crashing).

**Fix:** `base_drone_env_isaac.py`'s `_compute_dones_and_reward` now divides
by `self.max_episode_length` (this env's real, fixed, actually-enforced
policy-step cap) instead of the per-env `self._steps_for_dist` -- restores
the intended `TARGET_FRACTION*APPROACH_MILESTONE_BUDGET` ceiling on
worst-case step_penalty accumulation regardless of distance, matching the
numpy oracle's actual design invariant instead of a mismatched proxy for it.
`self._steps_for_dist` (now fully unused -- was only ever read here) and its
`steps_for_dist` import were removed rather than left dangling.

Both fixes (warmup floor + step_penalty denominator) are independently
justified and likely compound: an undertrained critic gives the actor a bad
initial push right at unfreeze, and the step_penalty miscalibration is what
prevented any recovery afterward by actively rewarding staying crashed.
**Neither fix is yet re-verified with a fresh full run** -- the next real
`train_isaac.py` run should be checked the same way (metrics.csv + plots,
specifically whether `outcome_hit`/`success_rate` hold up well past the
warmup->training transition this time) before trusting its final checkpoint.
