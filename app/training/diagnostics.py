"""Deterministic outcome diagnostics (policy or PID) on fixed target pairs in the numpy env."""
import numpy as np
import torch

from app.guidance.train import device


def _run_diagnostic_episodes(env, n_episodes, get_action, label, on_reset=None):
    """Shared rollout + outcome counting; get_action(obs) supplies the action, on_reset runs after each reset."""
    # keys = the reward fns' exact reason strings ("Hit"); CSV writers lowercase it
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
            # split "+"-joined compound reasons ("+" is also invalid in MLflow metric names)
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
    """residual_scale > 0: executed action = clip(pid + residual_scale * model), as in training (needs gains_by_dist)."""
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
    """Same diagnostic driven by the PID (per-distance gains if gains_by_dist is given): the reference ceiling."""
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
