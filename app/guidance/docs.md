# guidance

## Purpose
The learning side and the tools for looking at its results. It holds:
- the policy network and the PPO core;
- checkpoint replay and recording, with the local web viewer;
- training plots, ONNX export, and MLflow helpers.

## Files

| File | What it holds |
|---|---|
| `train.py` | `ActorCritic`, `RolloutBuffer`, `compute_gae`, `ppo_update`, single/vec/Isaac PPO loops, critic warm-up, `load_bc_checkpoint`. |
| `record_run.py` | Replays a checkpoint in the numpy env (two-phase mp4 / evaluation) and evaluates acro checkpoints (`evaluate_acro`). |
| `serve_run.py` | FastAPI app behind the web viewer: endpoints plus the static page. |
| `dyn_track.py` | `DynRecorder`: per-frame rotor thrusts (made / wanted), torques, drag, net force and drone-to-target angles for the 3D player's side panel and vectors. |
| `web/` | The viewer page: a small React app made of plain `.js` files (no build step). See "Web viewer" below. |
| `plotting.py` | Every training/eval plot (`plot_training_run`, grad-norm views, eval matrix, DAgger, per-worker). |
| `export_onnx.py` | `.pt` -> `.onnx`, with the architecture inferred from tensor shapes. |
| `mlflow_utils.py` | `start_run`, `log_params_safe`, `log_metrics_safe`. |
| `utils.py` | `compute_grade`: one scalar score per checkpoint. |

## ActorCritic
- **Trunk:** `num_hidden_layers` Linear layers with Tanh, width `hidden`. Production uses 64 x 4. It has an
  `actor_mean` head, a state-independent `actor_log_std`, and a `critic_head`.
- **std** is bounded: `log_std = min + (max - min) * (tanh(p) + 1) / 2`.
- **Actions** are sampled in raw space, squashed with tanh, and rescaled into `[action_low, action_high]`. These are
  buffers, so they're saved in the checkpoint. `log_prob` includes the tanh + affine change-of-variables correction,
  so it's the density of the action the env actually receives.
- `detach_critic` stops value gradients from reaching the trunk. It's a runtime flag and isn't saved.
- `load_bc_checkpoint` matches `shared.*` Linear layers **by position**, because an inserted Dropout layer shifts
  the raw Sequential indices. It raises an error on a real shape mismatch.

## PPO
- `ppo_update` returns the last epoch's averaged stats (policy/value/entropy loss, approx_kl, grad_norm,
  early_stopped). Two extras, both off by default: PPO2-style value clipping (`value_clip_eps`) and teacher
  distillation (`distill_target_actions`, `distill_coef`), which adds `coef * MSE(student, teacher)`.
- **Truncation bootstrap.** A time-limit cutoff isn't a real terminal state. The missing future value is added to
  the reward using `terminal_observation`, because envs auto-reset before returning, so `next_obs` already belongs
  to the next episode.
- `isaac_ppo_train` reuses the same network, GAE and update with GPU rollouts. Pass the previous call's `final_obs`
  as `initial_obs`: resetting on every call corrupts reward-shaping state and randomises `episode_length_buf`
  (`info/PROJECT_DEFENSE_GUIDE.md` Part 8.5.1). `skip_update` collects rollouts with no update at all, for a
  "settle" period. Freezing with `requires_grad` can't do this, because backward fails with every parameter frozen.
- `adaptive_kl_lr_step` (rl_games style) shrinks the lr when `approx_kl > 2 * target` and grows it when
  `approx_kl < target / 2`.
- `warmup_critic` / `isaac_warmup_critic` train only the critic while the actor is frozen. A BC-loaded actor
  starts with a random critic, and untrained-critic advantage noise can wreck a good warm start.

## record_run
- Replays a checkpoint **deterministically** (mean action) in the numpy `BaseDroneEnv`, with no Isaac needed, and
  renders a matplotlib 3D mp4 (at most 360 frames, 24 fps).
- The architecture is inferred from the checkpoint's tensor shapes, so any training script's `.pt` loads.
  `latest_full.pt` (from `angle_rtl_train_isaac`) contains the critic and optimizer and is not an inference checkpoint.
- Modes:

```bash
python -m app.guidance.record_run ckpt.pt --two-phase --distance 200 --switch-dist 20 --cruise-speed 18   # mp4
python -m app.guidance.record_run ckpt.pt --two-phase --evaluate --n-scenarios 50 --distance 200 [--action-repeat 4] [--hit-radius 1.0] [--hit-rule trained|acro]
python -m app.guidance.record_run acro_ckpt.pt --acro --n-scenarios 75 --distance-low 3 --distance-high 5 [--action-repeat 4] [--start auto|rest]
```

- The viewer picks the plant from the checkpoint (angle or acro), because a policy only means something on the plant it was trained on
  (`training/docs.md` "Angle vs acro"). The Two-phase tab is for angle checkpoints and the Acro eval tab for acro ones; both have the
  handoff option (below). The CLI `--two-phase` is angle only.
- **`--acro` / `evaluate_acro`** scores an acro checkpoint like the trainer's `isaac_eval`: clean start from rest at (0, 0, 5), random
  direction and yaw, deterministic `tanh(mean)`, first episode only, per-distance budget (`steps_for_dist`). Endings are hit /
  oob / tilt / hover / timeout with the `AcroRewardConfig` thresholds (tilt crash at `--tilt-limit` + 30 deg), not the numpy
  reward's roll 65 / pitch 80 deg crash. `--action-repeat` is how many 240 Hz physics steps each action is held: Isaac trains at
  60 Hz (4), the numpy env steps once per call (1 = a 240 Hz policy). `--start auto` (default) launches the drone toward the target at
  the run's `--spawn-speed-low/high` read from `run_config.json` (the trainer applies that to its eval too, so a run trained with it
  is scored launched); `--start rest` forces a start from rest. Note that at 16-35 m/s one policy step moves 0.27-0.58 m, so a pass
  through the 0.25 m hit sphere can step over it. The style is read from `run_config.json` (`tilt_limit` key =
  acro), then from the saved action bounds. Acro checkpoints saved before the bounds fix carry torque-style bounds (+-0.5
  rad/s); they are evaluated with the bounds they were trained with.

- **Handoff or just the model (web viewer, both tabs).** Each tab has a "Handoff" checkbox, and the Angle tab pre-sets it from the
  selected run's `run_config.json` (`handoff`). Off = the policy flies from the spawn point by itself. On = a scripted carrot controller flies from the spawn distance to the handoff point, then the policy takes over:
  - Angle (Two-phase tab): target distance = spawn distance, switch distance = handoff point, cruise speed = max speed, handoff speed,
    plus "sideways speed" (fraction of the handoff speed, random, like the trainer's `--lateral-frac`) and "tilt at handoff" (random
    roll/pitch within +-deg, `--handoff-tilt-deg`). Both default to 0 (the old behaviour: velocity along the line, attitude untouched)
    and are pre-set from `run_config.json` for a checkpoint trained with `--handoff`. Off sends `switch_dist = distance + 1` and
    no cruise speed, so the controller never runs.
  - Acro (Acro eval tab): the same form as the angle tab (pick/upload checkpoint, target distance, switch distance, cruise speed,
    handoff speed (blank = cruise / 2), sideways fraction, tilt, test runs, seed, presets), plus acro-only tilt limit, policy rate and
    the launch "Start" used when handoff is off. Distances are single values (the batch varies direction and yaw, like the angle batch).
    The controller starts from rest at the target distance; its torque is turned into a rate command with `cmd = w + tau/(I*kp)`
    (the trainer's `angle_to_acro`). Hover (stall) is counted and closest approach measured only after the takeover; the time budget
    is that of the spawn distance. The player colours the two phases.
  - The takeover snap itself is `app/control/two_phase.snap_handoff_state`, shared by both.
- **Two-phase** flags: `--handoff-speed` (default cruise/2), `--recentre-z` (full-mode policies only).
- With `trace_every > 0`, two-phase runs log one line per simulated second: phase, distance, speed, closing and
  lateral speed, tilt, and action. The final line carries a closest-approach summary, and `info["diag"]` holds the
  same numbers. This separates a near-miss from a parked policy or one that never arrives.
- `MAX_STEPS_CAP` is a wall-clock safety cap: `steps_for_dist(5000 m)`, about 1.25M steps. A 5 km episode takes
  minutes of real time.

## Web viewer (`serve_run.py` + `web/`)
- Run with `uv run uvicorn app.guidance.serve_run:app --port 8000` and open http://localhost:8000.
- The page is `web/index.html` plus plain ES-module files (React and htm come from a CDN through the importmap in `index.html`, so there is
  nothing to install or build; it needs internet on first load). Layout of `web/js/`:

  | File | What it holds |
  |---|---|
  | `main.js`, `App.js` | Mounts the app; title, the two tabs (both stay mounted so results survive a tab switch). |
  | `tabs/AngleTab.js`, `tabs/AcroTab.js` | One file per tab: its defaults, presets, drone-specific fields (angle: re-centring; acro: tilt limit, policy rate, start) and how the form becomes a request. The page layout is the same for both. |
  | `components/HandoffFields.js`, `CheckpointCard.js` | The scenario fields (incl. hit zone radius) and the checkpoint picker that both tabs share. |
  | `components/ChasersCard.js` | The "Chasing drones" card. The acro tab shows it switched off (the acro env has no chasers yet). |
  | `components/EvalResults.js` | The single-flight and batch result views shared by both tabs: same outcome counters (hit, attitude crash, intercepted, out of bounds, hover, timeout), same per-scenario table (a cell the env cannot fill shows `-`). |
  | `components/ui.js` | Small widgets: Card, NumberField, Kpi, DataTable, Status ... |
  | `components/Player.js` | Play bar, camera buttons and options of the 3D replay. The HUD label says "RL policy" when the frame is policy-flown; otherwise the controller's name (`meta.controller`, default "controller"). A track whose `phase` is all policy (a run with no handoff) always says "RL policy". Before 2026-11-09 the label was hardcoded "PID" for every non-policy frame, so acro runs showed PID. |
  | `player/PlayerScene.js` | The three.js scene itself (drone, paths, camera, chart); no React inside. |
  | `hooks.js` | `useStoredState` (remembers a value), `useRunner` (busy/error status), `useCheckpointChoice`, `useHandoffForm`. |
  | `api.js`, `storage.js`, `format.js`, `html.js` | fetch helpers, localStorage wrapper, number formatting, the `html` template tag. |

  Edit a file and refresh the browser; the server reads them from disk on every request. A new tab = a new file in `tabs/` plus one line in `App.js`.
- `serve_run.py` is organised as: settings, checkpoint helpers, the form-field classes (`CheckpointChoice`, `TwoPhaseSettings`, `AcroSettings`),
  the endpoints, and the static mount. The endpoints are plain `def` (FastAPI runs them in worker threads), and the settings classes turn the
  form into the arguments of `record_run.py`'s functions, including the "handoff off = just the model" rule.
- It is a local dev tool: no authentication, and synchronous (a request blocks while the episode runs).
- Checkpoints are listed from `runs/` and `app/control/` (`best.pt` first, then newest). Anything else, including
  `stable_models/`, can be **uploaded** from the page.
- API:

| Endpoint | What it does |
|---|---|
| `GET /api/checkpoints` | List of checkpoints. |
| `GET /api/checkpoint_info` | Stored run config for a checkpoint. |
| `POST /api/acro/evaluate` | Acro checkpoint scored like `isaac_eval` (form, same fields as `/api/two_phase/evaluate` plus `tilt_limit_deg`, `action_repeat`, `start`; `.pt` upload allowed). |
| `POST /api/acro/trajectory` | One acro flight as per-frame pose data for the 3D player (form, same fields as `/api/two_phase/trajectory` plus the acro ones). |
| `POST /api/two_phase/record` | Two-phase mp4. |
| `POST /api/two_phase/trajectory` | Two-phase 3D trajectory data. |
| `POST /api/two_phase/evaluate` | Two-phase stats. |

- **Geometry spawn** (angle tab): `threat_spawn=geometry` plus `threat_w_head_on / threat_w_tail`, `threat_reaction_s`,
  `threat_tail_lo/hi`, `threat_jitter` (the trainer's `--threat-geometry-weights` etc.; pre-filled from the run). The numpy
  `Threats.spawn_at_handoff` is called every step while the policy flies and acts once; before it the chasers are parked 300 m up and the track's
  `threat_dist` reads 10000. Hand copy of the Isaac `_spawn_geometry`: keep them in sync.
- **Chasing drones** (angle tab): the three `/api/two_phase/*` endpoints also take `threats` (how many chasers are on, 0 = none),
  `threat_speed` and `threat_speed_min` (commanded chase speed; each chaser draws its cap from U(speed_min, speed)),
  `threat_spawn` (`uniform` or `base`), `threat_spread` / `threat_spawn_min` (uniform: shell around the base), `threat_buffer` (base:
  clearance between neighbouring bodies), `threat_radius` (collision sphere of the run, empty = the drone's own body),
  `threat_release_lo` / `threat_release_hi` (the trainer's `--threat-release-after`: fraction of the handoff distance the drone has
  covered when a chaser is released, drawn per chaser) and `threat_max_tilt_deg` (`--los-max-tilt-deg`). Defaults as in the trainer; the
  page pre-sets them from the run's config. The slot count comes from the checkpoint
  (observation = 23 + 7 per slot, plus 2 when it was trained with the rc zone timer; `layout_from_obs_dim` tells the two apart and
  the numpy env then feeds in_zone and zone seconds / zone_time, using hit_radius and zone_time from run_config.json), so asking
  for more chasers than slots is a 400. Each chaser is a drone with the learner's own physics (`timestamp_update`), flown by the
  limit-LOS controller (`control/los_np.py`, the numpy twin of `control/los.py`): straight at the drone at its speed cap, no other chase
  limit. It hovers where it spawned until the policy has taken over AND the drone has covered its release fraction; one within the
  catch radius (the run's `reward_qv2.intercept_radius`, else 0.25 m, swept per tick) ends the flight as `Intercepted`. The viewer has
  no contact: a chaser does not bounce off the drone or other chasers, `threat_radius` only sizes the drawn body and the base spacing.
  The track carries `threat_radius`; the 3D player draws each chaser in its slot colour (the Isaac palette).
- **Hit zone** (angle tab): `/api/two_phase/*` take `hit_radius` (m, empty = the run's own radius). For a `--reward rc` checkpoint
  (run_config.json `reward == "rc"`) a Hit is scored by the selected hit rule (next bullet); the flight then keeps going for
  `post_hit_steps` (6) more policy steps (a crash in that window still counts as a Hit, as `isaac_eval` keeps the first outcome). The
  3D player draws the zone sphere at the effective radius and the logs, distance plot and batch summary use it. It does not change what
  the model sees: a checkpoint trained with the zone timer keeps the `hit_radius` / `zone_time` from its run_config.json for those two
  inputs. Checkpoints without an rc run config keep the old instant rule (`rewards.reward_func` ends the flight the moment the drone is
  inside `env.hit_radius`, empty = 0.25 m).
- **Hit rule** (angle tab, rc checkpoints): the "Hit rule" selector has two settings.
  - `trained` = the run's own rule: a Hit needs the drone to stay inside the zone for the run's `zone_time` (0.05 s = 3 consecutive
    60 Hz policy steps). This is exactly what `isaac_eval` scored, so batch numbers stay comparable to `eval_hit` / `eval_roll`.
  - `acro` (the tab's default) = the acro eval's rule: the first policy step inside the radius is a Hit, and a pass through the sphere
    between two policy samples also counts (the straight-segment test from `acro_reward_v1.segment_min_dist`, which was added because
    one 60 Hz step at speed can jump over the sphere). This is a viewer override, not what the rc checkpoint trained with; batch numbers
    then mean "reached the target", not the trained metric. The zone-timer observation always keeps the trained radius/time.
  The CLI exposes the same choice as `--hit-rule trained|acro` (default `trained` for the CLI, so scripted batches stay faithful).
- **Attitude limits** (angle tab): the numpy env ends a flight as `attitude-ROLL` / `attitude-PITCH` past a roll / pitch limit. It takes them from the run (`run_config.json` -> `reward_qv2.roll_limit_deg` / `pitch_limit_deg`; `--reward rc` runs often set 360 = off), else the numpy reward's own 65 / 80 deg. Before this, a run trained with the limits off still ended at 65 deg in the viewer.
- **rc endings** (angle tab, `--reward rc` checkpoints): the old numpy `reward_func` terminals are switched off (`hit_radius = -1`,
  attitude 360 deg, `oob_radius = inf`) because they would fire every 240 Hz step with the wrong rules. `_RCTerminalChecker` in
  `record_run.py` is the numpy twin of `angle_reward_rc.compute_reward_rc` and judges once per policy step instead:
  - **tilt**: angle between body-z and world-up (from the quaternion) above `tilt_limit_deg`; isaac_eval buckets it as `roll`, so the
    viewer reports `attitude-ROLL` (the batch's `attitude` total). Roll/pitch limits are separate and usually 360 (off).
  - **hover**: `hover_limit_s` seconds (after `hover_grace_s`) with distance > hit radius and speed < `stall_speed`, outside the zone.
  - **oob**: position outside `max(oob_min_radius, oob_start_dist_mult * start distance)`, z < 0, or NaN.
  - **timeout**: the per-distance budget `steps_for_dist(distance)`, rounded up to a whole number of policy steps.
  All thresholds come from the run (`/api/checkpoint_info` returns them as `rc_*` so the page can show them). The batch totals now
  include `hover`, and each scenario carries `tilt_max_deg`, `time_s`, `mean_speed`, `policy_steps`, `hit_step` and
  `post_hit_end_reason`.
- **Policy rate** (angle tab, both drone styles): Isaac evaluates at 60 Hz (`decimation = 4`), so each policy action is held for 4
  physics steps by default. The page's "Policy rate" selector also offers 240 Hz (1 step) for comparison; the timer inputs and every rc
  terminal advance per policy step either way. A batch and a single flight use the same rate.
- **Target yaw**: like the trainer's reset, each angle flight draws the target yaw uniformly from [-pi, pi]. A batch varies direction
  and yaw per scenario. Distances are still one chosen value per batch; training's eval samples U(distance_low, distance_high).
- **Line of sight** (every `/trajectory`, angle and acro; acro reads `los_*` from the run's `acro_reward`, else acro v2 defaults): `track` gains `los_deg` (angle between the velocity and the direction to the
  target; 0 = straight at it, 90 = sideways, 180 = away), `los_reward` (the rc reward's line-of-sight term, `-los_k * m(angle)` by default, 0 below `los_min_speed`,
  per 60 Hz policy step), `los_reward_sum` (running total, each 240 Hz sim step counts a quarter) and `los_meta` (`k`, `min_speed`, `mode`,
  from the run's `reward_qv2` in run_config.json, else the rc defaults, so older checkpoints show what the term would pay). The
  player HUD shows them. Only the line-of-sight term is shown; the numpy env's own step reward is still the old `reward_func`.
- **Forces panel**: every `/trajectory` result (angle and acro) carries `track.dyn` (per frame) and `track.dyn_meta`. The player draws a
  coloured arrow per rotor (made thrust; a pale arrow for what the commanded action asks of it, red and pointing down when the mixer asks for a
  push the rotor cannot make), whole-drone vectors (thrust, wanted thrust, weight, drag, net force) and two arcs at the drone: thrust axis to the
  target line (yellow-green) and velocity to the target line (blue). The side panel lists the same numbers, how close each command is to its
  bound (`AT LIMIT`), and the angles. "Wanted" is the action pushed through the mixer with the state at that frame.
  The trajectory's `track` gains `threats` (per-frame chaser positions) and `threat_dist`; batches report a `caught` total.
  `/api/checkpoint_info` returns `num_threats`/`active_threats`/`threat_*` so the page pre-fills them. Model: `app/environmental/threats.py`.
- **Moving target (acro `--runner` runs).** `threats.Runner` is the numpy twin of the Isaac runner. It hovers on the target until the policy
  takes over (at the handoff, or at the first step), then flees from the drone at its cap, already moving, never below `runner_min_alt`.
  `_acro_episode` makes it the target, appends the 3 runner inputs (target velocity - drone velocity, `/ VEL_SCALE`), and ends a flight
  as `escape` once the runner is farther than the escape radius from its own spawn point (`Runner.escaped()`). The player draws that sphere. It uses the Isaac episode cap instead of the per-distance budget, and the trainer's out-of-bounds
  bound. The run is detected from `run_config.json` (`runner`), or from a 26-input checkpoint. The Acro tab shows a "Moving target" card,
  pre-filled from `/api/checkpoint_info` (`runner`, `runner_speed`, `runner_speed_min`, `runner_escape_dist`; empty = the run's value).
  The track gains `target` (position per frame): the player moves the hit sphere and beacon along it and draws its path. Batches count
  `escape`. The Acro tab's hit-radius field now reaches the server (`AcroSettings.hit_radius`); it used to be dropped.

- Videos and JSON summaries go to `recordings/` (gitignored). For security, a picked checkpoint path must be one of
  the listed ones.

## Plots
- `plot_training_run` draws `training_error`, `policy_std`, `distance_distribution`, `success_and_outcomes` and
  `vs_pid_baseline`. The PID line is the PID on the **same target pairs**: a high RL line next to a low PID line
  means the task is solvable and the gap is the policy's.
- The 3D grad-norm "landscape" is an interpolated surface. Only the drawn path is real data.
- Sustained-checkpoint ranking accepts a checkpoint only if its trailing window passes both the average and the
  minimum grade. This filters out lucky spikes.

## MLflow
- The store is `MLFLOW_TRACKING_URI`, or `sqlite:///mlflow.db` if that's unset. Isaac runs write
  `mlflow_isaac.db`; see the README.
- When several processes first touch a fresh sqlite store, they race to create the schema. Create the experiment
  once before starting workers.

## Depends on
`torch`, `numpy`, `matplotlib`, `imageio[ffmpeg]`, `mlflow`, `onnx`, `fastapi` and `uvicorn` (viewer only).
Internally: `environmental`, `control` (`two_phase`, `step_budget`), `reward_functions.rewards`, `training.eval_matrix`.
