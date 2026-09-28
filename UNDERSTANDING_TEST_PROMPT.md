# Prompt: project understanding test

Paste everything below the line into a new Claude Code session opened in this repo.

---

Run the **project understanding test** on me.

## Scope
Every `.py` file under `app/` in this repo. Exclude `__pycache__`, `app/z_final_version_*` (run artifacts), `app/test/`, anything under `deprecated/`, and non-code files. **Before asking anything, read all of it** and build every question from the actual current code (real function names, real `file:line` references) — not from the `docs-*.md` files (they may be stale) and not from memory of earlier sessions. When you grade, verify my answers against the code, not against your recollection.

## Sections
Ask in this order (it follows the dependency chain).

MAJOR — 7 to 11 questions each:
1. **dynamics** — `app/dynamics/drone.py`, `app/dynamics/methods.py`
2. **environmental** — `app/environmental/base_drone_env.py`, `vec_base_drone_env.py`, `subproc_vec_base_drone_env.py`, `enviorment.py`
3. **reward_functions** — `app/reward_functions/rewards.py`, `reward_fn_phase1.py`, `reward_fn_phase2.py`
4. **ppo core** — `app/guidance/train.py` only
5. **control** — `app/control/pid_hover.py`, `tune_pid.py`, `collect_demonstrations.py`, `pretrain_bc.py`, `dagger.py`, `verify_pid.py`

STANDARD — 4 to 7 questions each:
6. **training loops** — `app/training/phase_0_training.py`, `phase_1_training.py`, `phase_2_training.py`, `diagnostics.py`, `eval_matrix.py`
7. **guidance tooling** — `app/guidance/optuna_search.py`, `plotting.py`, `mlflow_utils.py`, `utils.py`, `export_onnx.py`, `watch_hover.py`, `test_free_hover.py`

Start each section at its minimum count; add questions only where my answers reveal a gap worth probing. Never exceed the max. Total will be roughly 45–65 questions — if it looks like it won't fit in one session, say so and propose splitting by section.

## Question rules
- Real understanding only: what does X do, why does Y happen, what does Z affect downstream, why was this chosen over the obvious alternative, how does data flow from A to B, what breaks if W is changed or removed. No yes/no questions. No "what is the default value of X" trivia unless the value matters.
- Question 1 of every section: "Explain this part in your own words in at most 5 sentences." It counts toward the total.
- Every MAJOR section must include: (a) one "trace this concrete scenario through the code step by step" question, (b) one "what would break or change if…" question, (c) one "why this design instead of the alternative" question, and at least 2 questions that require connecting two different files.
- One question at a time. Wait for my answer. **Max 2 follow-ups per question in total** — a request for me to clarify my answer counts as one. Don't hint at the answer inside a follow-up.
- After a question is closed, give the correct answer in at most 3 sentences with the `file:function` it lives in, then move on. If I say "no feedback until the end", hold all answers for the final grading instead.
- Closed book by default: I answer from memory first. If I then open the file and correct myself, record it as **self-corrected** (capped at partial credit).
- Don't grade generously. If I hedge between two answers, score the worse one. If my answer is wrong, say so plainly.
- If I say "stop here", write the results so far and end. The results file must be resumable: "continue the understanding test" picks up at the next unanswered question.

## Scoring
- Per question 0–3: **0** wrong or blank · **1** vague or partially right · **2** correct "what" only · **3** correct "what" + "why" + consequences. Self-corrected answers cap at 2.
- Section score = points / max points, as %. Overall = weighted average, MAJOR sections 2x, STANDARD 1x.
- Per section, also grade each **core technology** 0–3. Derive the exact list from the code you read; starting point:
  - dynamics: rigid-body torque/inertia, quaternion kinematics via scipy `Rotation` (body-frame vs world-frame composition), mixer matrix + inversion, motor lag / drag / wind, integration scheme
  - environmental: gymnasium `Env` API, observation construction + symlog normalization, action semantics (thrust delta + body torques), termination vs truncation + auto-reset + `terminal_observation`, multiprocessing spawn + pickling constraints, PID-teacher selection per distance
  - reward_functions: shaping terms, curriculum chaining (`chain_reward_fns`), terminal/boundary checks, the streak mechanism, the imitation term, how phase2 differs from phase1 and why
  - ppo core: shared-trunk `ActorCritic`, tanh squash + log-prob correction, GAE, clipped objective, entropy bonus + `target_kl` early stop, truncation bootstrap, `load_bc_checkpoint` remapping
  - control: cascaded PID (outer position → desired tilt → inner attitude → torque), gain meaning + saturation (`max_tilt_rad`), per-distance gains, BC as regression on (obs, action), the DAgger loop, demonstration coverage (omni categories)
  - training loops: chunked loop + checkpoints/diagnostics, warm start + critic freeze/unfreeze, schedules (ent_coef, cosine lr, weight decay), what is logged and why
  - guidance tooling: Optuna objective + search space, MLflow logging, the grade metric, ONNX export + architecture inference
- Keep a **misconception log**: every wrong belief I state, quoted, with the correction.

## Output
Write to `reports/understanding-test-<YYYY-MM-DD>.md`, **incrementally after each section** (not only at the end):
1. Per section: each question, my answer (summarized), follow-ups, score, verdict (good / partial / wrong), the correct answer, and one concrete "how to improve" (which file/function to reread, or a small experiment to run).
2. Per section: overall %, technology scores, a 2–3 sentence assessment.
3. Overall: weighted %, top strengths, top weaknesses, the misconception log, and a prioritized study plan (top 5, each pointing at specific files/functions with a small hands-on exercise).

At the end, print the overall summary in chat too.
