"""Training loop for app.reward_functions.rewards.reward_func -- the one flat
potential-based reward, used identically on every step of every episode. No
reward-fn phases/roadmap/config class.

Budget split (1,000,000 steps total):
  - 25% imitation: on-policy, run INSIDE this script (_run_imitation_stage) --
    not the same thing as the DAgger/BC checkpoint below. The actor is first
    warm-started from that checkpoint, then this stage keeps refining it
    on-policy against each worker's distance-matched PID teacher before PPO
    ever touches it.
  - 1% critic-only warmup (WARMUP_DURATION_STEPS): actor frozen so the
    (randomly-initialized) critic calibrates against the warm-started policy
    before PPO starts updating it -- untrained-critic advantage noise can
    otherwise immediately wreck a warm-started policy. Which reward is active
    during this stretch doesn't matter for the policy, only for the value
    targets, so it's the same reward_func throughout, not a different one.
  - 74% (rest of the 99% after warmup): standard PPO, actor+critic, against
    reward_func.

Targets are drawn from a CONTINUOUS Uniform(distance_low, distance_high)
range with a random 3D direction (build_uniform_omni_eval_pairs), not a
discrete distance ladder.

Curriculum staging: this script trains one FIXED distance range per run, not
an auto-widening curriculum. To teach longer range, run it multiple times
with an increasing (distance_low, distance_high) window, each stage
warm-starting from the previous stage's final checkpoint:

    python -m app.training.base_training --distance-low 3 --distance-high 10 \
        --checkpoint-dir runs/base_training_3_10
    python -m app.training.base_training --distance-low 3 --distance-high 50 \
        --checkpoint-dir runs/base_training_3_50 \
        --bc-checkpoint runs/base_training_3_10/model_1000000.pt
    python -m app.training.base_training --distance-low 3 --distance-high 100 \
        --checkpoint-dir runs/base_training_3_100 \
        --bc-checkpoint runs/base_training_3_50/model_1000000.pt
    python -m app.training.base_training --distance-low 3 --distance-high 200 \
        --checkpoint-dir runs/base_training_3_200 \
        --bc-checkpoint runs/base_training_3_100/model_1000000.pt

Goal of each stage:
  - 3-10m: learn clean close-range approach/hover against the PID teacher
    while the search space is small; this is also what the DAgger checkpoint
    (BC_CHECKPOINT_PATH-equivalent, --bc-checkpoint default) was built for.
  - 3-50m / 3-100m / 3-200m: reuse that close-range competence as a warm
    start and stretch it to longer straight-line flight + terminal approach,
    one distance-bucket step at a time, rather than forcing PPO to learn
    both "fly far" and "land precisely" from scratch at long range. Each
    step's own OOB_RADIUS auto-scales with --distance-high
    (max(20, distance_high*3)); best_pid_gains_per_dist.json already has
    tuned PID gains for the 3/10/50/100/150/250 buckets that
    SubprocVecBaseDroneEnv's per-episode teacher selection needs, so no
    extra tuning is required to add a stage within that ladder.
  - --checkpoint-dir MUST differ per stage -- timesteps_done restarts at
    IMITATION_DURATION_STEPS every run, so two stages sharing a directory
    will silently overwrite each other's model_<timesteps>.pt/metrics.csv.

Uses SubprocVecBaseDroneEnv (multiprocess, one worker per core) -- see
subproc_vec_base_drone_env.py; ~526 (vs ~92 single-process) per-env steps/sec.

reward_func's own knobs (TARGET_FRACTION, APPROACH_MILESTONE_BUDGET, HIT_REWARD,
milestone fractions/bonuses, OOB_RADIUS/attitude limits) are plain module-level
constants in rewards.py, not per-run hyperparameters -- see that module for
their values and APPROACH_MILESTONE_BUDGET's provenance (measured via the
tuned PID, hit bonus included; rerun that calibration and update the constant
if best_pid_gains_per_dist.json or the reward changes).

Every checkpoint is saved as both .pt and .onnx; metrics are appended to a
CSV row per checkpoint (one row per CHUNK_TIMESTEPS of training, not per raw
env-step -- 1,000,000/20,000 = 50 rows over a full run); diagnostics run
10 deterministic episodes per checkpoint.
