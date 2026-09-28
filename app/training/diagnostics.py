"""Trained-model outcome-distribution diagnostic, split out of phase_0_training.py
so callers (phase_1_training.py, base_training.py) don't drag in that module's
import-time side effects (its own log-file setup prints "Logging this run to:
..." on every import -- harmless but noisy, especially once per worker process)."""
import numpy as np
import torch

from app.guidance.train import device


def _run_diagnostic_episodes(env, n_episodes, get_action, label, on_reset=None):
    """Shared rollout/outcome-counting loop for diagnose_with_model and
    diagnose_with_pid -- get_action(obs) supplies the per-step action, so the
    two differ only in how that action is produced. on_reset (e.g. a PID
    controller's .reset(), to clear its integral terms between episodes) runs
    right after each env.reset()."""
    # Keys must match the exact reason strings the reward fns return (_check_hit
    # returns "Hit", not "hit") -- callers that write it to a CSV consumed by
    # plot_training_run rename this back to lowercase "hit" for that one column.
    outcomes = {"oob": 0, "attitude-ROLL": 0, "attitude-PITCH": 0, "Hit": 0,
                "hover_success": 0, "moving_away_cap": 0, "drift": 0, "timeout": 0}
    steps_survived = []
    hit_times_sec = []
    final_dists = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if on_reset is not None:
            on_reset()
        done = False
        step_count = 0
        last_reason = None
        info = {}

        while not done:
            action = get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
            last_reason = info["reason"]

        steps_survived.append(step_count)
        final_dists.append(env.prev_distance)
        if info.get("hit_time_sec") is not None:
            hit_times_sec.append(info["hit_time_sec"])

        if env.hover_success_achieved:
            outcomes["hover_success"] += 1
        elif truncated and not terminated:
            outcomes["timeout"] += 1
        else:
            # last_reason can be a "+"-joined compound (rewards.terminal_checks
            # reports every condition that tripped the same step, e.g.
            # "attitude-ROLL+attitude-PITCH") -- credit each contributing
            # reason separately rather than minting a new compound key, which
            # would (a) dilute the single-reason counts and (b) break MLflow
            # metric logging downstream ("+" isn't a valid metric-name char).
            for reason in last_reason.split("+"):
                outcomes[reason] = outcomes.get(reason, 0) + 1

    avg_hit_time_sec = float(np.mean(hit_times_sec)) if hit_times_sec else None
    print(f"[diagnostic][{label}] outcomes over {n_episodes} eps:", outcomes)
    print(f"[diagnostic][{label}] avg steps survived: {np.mean(steps_survived):.1f}")
    print(f"[diagnostic][{label}] avg final dist: {np.mean(final_dists):.3f}  avg time-to-hit: {avg_hit_time_sec}")
    outcomes["avg_hit_time_sec"] = avg_hit_time_sec
    outcomes["avg_final_dist"] = float(np.mean(final_dists))
    outcomes["final_dists"] = final_dists  # raw per-episode list, for std/min plots
    return outcomes


def diagnose_with_model(model, env, n_episodes, residual_scale=0.0, gains_by_dist=None):
    """residual_scale > 0: the model's (deterministic mean) action is a CORRECTION on top of the per-distance
    PID -- executed action = clip(pid_action + residual_scale * model_action), the same composition
    base_training_isaac.PidResidualEnv applies during training -- so the grade measures the policy that is
    actually being trained. Needs gains_by_dist (same per-episode gain swap diagnose_with_pid does)."""
    if residual_scale > 0:
        from app.environmental.subproc_vec_base_drone_env import _select_pid_teacher

    def get_action(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, std, _ = model.forward(obs_t)
            action = model.scale_action(mean)
        action = action.squeeze(0).cpu().numpy()
        if residual_scale > 0:
            pid_action = env.pid_teacher.compute_action(env.drone_state, env.target_pos, env.target_yaw)
            action = np.clip(np.asarray(pid_action) + residual_scale * action,
                             env.action_space.low, env.action_space.high)
        return action

    def on_reset():
        _select_pid_teacher(env, gains_by_dist)

    return _run_diagnostic_episodes(env, n_episodes, get_action, label="model",
                                    on_reset=on_reset if residual_scale > 0 else None)


def diagnose_with_pid(pid, env, n_episodes, gains_by_dist=None):
    """Same rollout/outcome-counting as diagnose_with_model, but driven by a
    PIDController instead of the policy -- gives a per-checkpoint reference
    for what the hand-tuned teacher achieves on the exact same env/target_pairs,
    so plot_training_run can show whether the RL policy is converging toward
    that ceiling or diverging from it.

    `gains_by_dist` (app/control/best_pid_gains_per_dist.json, loaded), if given,
    swaps env.pid_teacher to the nearest-distance gain set after every reset --
    matching what SubprocVecBaseDroneEnv's workers do for the actual imitation
    teacher (see subproc_vec_base_drone_env._select_pid_teacher). Without it,
    `pid` is used unchanged for every episode; on a multi-distance omni curriculum
    (e.g. base_training's Uniform(distance_low, distance_high)) a single fixed
    gain set is a much worse controller than the per-distance-matched one, which
    makes this diagnostic understate what PID actually achieves."""
    from app.environmental.subproc_vec_base_drone_env import _select_pid_teacher

    def get_action(obs):
        teacher = env.pid_teacher if gains_by_dist is not None else pid
        return teacher.compute_action(env.drone_state, env.target_pos, env.target_yaw)

    def on_reset():
        if gains_by_dist is not None:
            _select_pid_teacher(env, gains_by_dist)
        else:
            pid.reset()

    return _run_diagnostic_episodes(env, n_episodes, get_action, label="pid", on_reset=on_reset)
