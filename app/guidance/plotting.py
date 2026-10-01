import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# PID baseline -- what would a classical controller score on the same task?
# ---------------------------------------------------------------------------
def compute_pid_baseline(pid_gains_path="app/control/best_pid_gains.json",
                          n_episodes=20, n_steps=600,
                          oob_radius=7, hit_reward=10, attitude_penalty=-1.0,
                          oob_penalty=-1.5, streak_penalty_coef=-0.05,
                          hover_success_steps=200, streak_cap=30,
                          distances=None, gains_by_dist_path=None):
    """
    Runs the tuned PID controller through the same env/reward machinery used to score
    the RL policy, so the final checkpoint can be compared apples-to-apples. With
    `distances` set, loads per-distance gains from gains_by_dist_path and pools results
    across the whole curriculum instead of one fixed-distance baseline. Returns None
    (with a printed reason) if the gains file is missing, since this plot is optional.
    """
    from app.environmental.base_drone_env import BaseDroneEnv
    from app.control.pid import PIDController
    from app.reward_functions.rewards import RewardConfig, make_reward_fn

    final_dists, steps_survived, episode_rewards = [], [], []
    n_success = 0
    n_episodes_run = 0

    if distances is not None:
        from app.training.eval_matrix import build_eval_pairs

        if gains_by_dist_path is None or not os.path.isfile(gains_by_dist_path):
            print(f"[pid_baseline] no per-distance gains file at {gains_by_dist_path}, skipping PID comparison")
            return None
        with open(gains_by_dist_path) as f:
            gains_by_dist = json.load(f)

        n_per_pair = max(1, n_episodes // (len(distances) * 4))
        for dist in distances:
            gains = gains_by_dist[str(dist)]
            pid = PIDController(**gains)
            dist_oob_radius = max(oob_radius, dist * 3.0)

            reward_cfg = RewardConfig(
                oob_radius=dist_oob_radius, hit_reward=hit_reward, attitude_penalty=attitude_penalty,
                oob_penalty=oob_penalty, streak_penalty_coef=streak_penalty_coef,
                hover_success_steps=hover_success_steps, streak_cap=streak_cap,
            )
            env = BaseDroneEnv(make_reward_fn(reward_cfg), pid_gains_path=None)

            for start, target, target_yaw in build_eval_pairs(oob_radius=dist_oob_radius, distances=(dist,)):
                for _ in range(n_per_pair):
                    obs, _ = env.reset(start_pos=start.copy(), target_pos=target.copy(), target_yaw=target_yaw)
                    pid.reset()
                    done = False
                    step_count = 0
                    ep_reward = 0.0
                    reason = None
                    while not done and step_count < n_steps:
                        action = pid.compute_action(env.drone_state, env.target_pos, env.target_yaw)
                        obs, reward, terminated, truncated, info = env.step(action)
                        ep_reward += reward
                        step_count += 1
                        done = terminated or truncated
                        reason = info["reason"]
                    steps_survived.append(step_count)
                    episode_rewards.append(ep_reward)
                    final_dists.append(env.prev_distance)
                    n_episodes_run += 1
                    if reason == "Hit":
                        n_success += 1

    else:
        if not os.path.isfile(pid_gains_path):
            print(f"[pid_baseline] no gains file at {pid_gains_path}, skipping PID comparison")
            return None

        with open(pid_gains_path) as f:
            gains = json.load(f)
        pid = PIDController(**gains)

        reward_cfg = RewardConfig(
            oob_radius=oob_radius, hit_reward=hit_reward, attitude_penalty=attitude_penalty,
            oob_penalty=oob_penalty, streak_penalty_coef=streak_penalty_coef,
            hover_success_steps=hover_success_steps, streak_cap=streak_cap,
        )
        env = BaseDroneEnv(make_reward_fn(reward_cfg), pid_gains_path=None)

        for _ in range(n_episodes):
            obs, _ = env.reset()
            pid.reset()
            done = False
            step_count = 0
            ep_reward = 0.0
            reason = None
            while not done and step_count < n_steps:
                action = pid.compute_action(env.drone_state, env.target_pos, env.target_yaw)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step_count += 1
                done = terminated or truncated
                reason = info["reason"]
            steps_survived.append(step_count)
            episode_rewards.append(ep_reward)
            final_dists.append(env.prev_distance)
            n_episodes_run += 1
            if reason == "Hit":
                n_success += 1

    return {
        "success_rate": n_success / max(n_episodes_run, 1),
        "avg_final_dist": float(np.mean(final_dists)),
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_steps_survived": float(np.mean(steps_survived)),
    }


# ---------------------------------------------------------------------------
# Streak analysis -- is a good checkpoint a fluke or sustained?
# ---------------------------------------------------------------------------
STREAK_TIERS = (0.25, 0.50, 0.75, 1.00)
STREAK_LEN = 5  # consecutive checkpoints required at a given tier


def _hit_ratios(rows, n_diag_episodes):
    return [float(r.get("outcome_hit", 0) or 0) / n_diag_episodes for r in rows]


def _streak_windows(values, threshold, min_len=STREAK_LEN):
    """
    Indices [start, end] (inclusive) of every run where value >= threshold for at least
    min_len consecutive checkpoints -- filters out a single lucky checkpoint.
    """
    windows = []
    start = None
    for i, v in enumerate(values):
        if v >= threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                windows.append((start, i - 1))
            start = None
    if start is not None and len(values) - start >= min_len:
        windows.append((start, len(values) - 1))
    return windows


def _shade_streak_windows(ax, timesteps, rows, n_diag_episodes):
    """Overlay the highest tier's streak window(s) as shaded spans on a plot."""
    ratios = _hit_ratios(rows, n_diag_episodes)
    for tier in reversed(STREAK_TIERS):
        windows = _streak_windows(ratios, tier)
        if windows:
            for start_i, end_i in windows:
                ax.axvspan(timesteps[start_i], timesteps[end_i], color="g", alpha=0.12,
                           label=f"{int(tier*100)}% streak (>={STREAK_LEN} ckpts)")
            break  # only shade the highest tier actually achieved


# ---------------------------------------------------------------------------
# Plots -- 5 figures: training_error, policy_std, distance_distribution,
# success_and_outcomes, and vs_pid_baseline.
# ---------------------------------------------------------------------------
def plot_training_run(csv_path, output_dir="plots_final",
                       hover_success_steps=480, n_diag_episodes=20,
                       hit_threshold=0.3, streak_cap=30, max_steps=5000,
                       oob_radius=7, hit_reward=10, attitude_penalty=-1.0,
                       oob_penalty=-1.5, streak_penalty_coef=-0.05,
                       pid_gains_path="app/control/best_pid_gains.json",
                       pid_baseline_episodes=20,
                       pid_baseline_distances=None,
                       pid_gains_by_dist_path=None):
    """
    hover_success_steps, n_diag_episodes, hit_threshold, streak_cap, max_steps must match
    what the run was actually trained with -- they only draw reference lines and rebuild
    an equivalent RewardConfig for the PID baseline, not recompute anything from the CSV.
    Pass pid_baseline_distances/pid_gains_by_dist_path to compare against the PID
    baseline across the full distance curriculum instead of one fixed target.
    """
    rows = _read_csv(csv_path)
    if not rows:
        print(f"[plot_training_run] no rows found in {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    timesteps = np.array([float(r["timesteps"]) for r in rows])

    def col(name, default=0.0):
        return np.array([float(r.get(name, default) or default) for r in rows])

    # --- 1) training_error: policy/entropy loss + value_loss/grad_norm (log) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(timesteps, col("policy_loss"), label="policy_loss")
    ax1.plot(timesteps, col("entropy_loss"), label="entropy_loss")
    ax1.axhline(y=0.0, color="k", linestyle=":", alpha=0.5)
    ax1.set_ylabel("loss"); ax1.set_title("Policy & entropy loss"); ax1.legend()

    ax2.semilogy(timesteps, col("value_loss"), label="value_loss")
    ax2.semilogy(timesteps, np.maximum(col("grad_norm"), 1e-8), label="grad_norm (pre-clip)")
    ax2.set_xlabel("timesteps"); ax2.set_ylabel("log scale")
    ax2.set_title("Value loss & gradient norm (success = low & flat, not spiking)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_error.png")); plt.close(fig)

    # --- 2) policy_std: exploration decay ---
    plt.figure(figsize=(8, 4))
    plt.semilogy(timesteps, np.maximum(col("effective_std_mean"), 1e-8))
    plt.xlabel("timesteps"); plt.ylabel("effective_std_mean (log scale)")
    plt.title("Policy action std (success = decaying, not climbing or flat-high)")
    plt.savefig(os.path.join(output_dir, "policy_std.png")); plt.close()

    # --- 3) distance_distribution: mean +/- std band, min, target=0 ---
    avg_dist, std_dist, min_dist = col("avg_final_dist"), col("std_final_dist"), col("min_final_dist")
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, avg_dist, label="avg_final_dist")
    plt.fill_between(timesteps, avg_dist - std_dist, avg_dist + std_dist, alpha=0.2,
                      label="+/- 1 std across diagnostic episodes")
    plt.plot(timesteps, min_dist, label="min_final_dist", linestyle="--")
    if "pid_avg_final_dist" in rows[0]:
        # Per-checkpoint PID-teacher reference on the exact same target_pairs
        # (see diagnose_with_pid) -- a flat/low PID line next to a flat/high RL
        # line means the task itself is solvable, so the gap is the policy's.
        plt.plot(timesteps, col("pid_avg_final_dist"), label="pid_avg_final_dist",
                  linestyle=":", color="gray")
    plt.axhline(y=hit_threshold, color="g", linestyle="--", label=f"hit_threshold={hit_threshold}")
    plt.axhline(y=0.0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("timesteps"); plt.ylabel("distance (m)")
    plt.title("Final distance to target (success = converges to 0, band narrows)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "distance_distribution.png")); plt.close()

    # --- 4) success_and_outcomes: hit rate + outcome breakdown ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    hit_ratio = col("outcome_hit") / n_diag_episodes
    ax1.plot(timesteps, hit_ratio, marker=".", label="RL success_rate")
    if "pid_success_rate" in rows[0]:
        ax1.plot(timesteps, col("pid_success_rate"), linestyle=":", color="gray", label="PID success_rate")
    for tier, style in ((0.25, ":"), (0.50, "--"), (0.75, "-."), (1.00, "-")):
        ax1.axhline(y=tier, color="g", linestyle=style, alpha=0.5, label=f"{int(tier*100)}%")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("hit / n_diag_episodes")
    ax1.set_title("Hit rate per checkpoint")
    _shade_streak_windows(ax1, timesteps, rows, n_diag_episodes)
    ax1.legend(loc="lower right")

    outcome_keys = [k for k in rows[0].keys() if k.startswith("outcome_")]
    outcome_series = [col(k) for k in outcome_keys]
    ax2.stackplot(timesteps, *outcome_series, labels=[k.replace("outcome_", "") for k in outcome_keys])
    ax2.axhline(y=n_diag_episodes, color="g", linestyle="--",
                label=f"all {n_diag_episodes} eps -> hit")
    ax2.set_xlabel("timesteps"); ax2.set_ylabel("count")
    ax2.set_title("Outcome distribution (success = stack fills with hit only)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "success_and_outcomes.png")); plt.close(fig)

    print(f"[plot_training_run] wrote plots_final to {output_dir}/")
    _print_suitable_weights(rows, os.path.dirname(csv_path) or ".", n_diag_episodes)

    # --- 5) vs_pid_baseline: RL final checkpoint vs classical PID controller ---
    baseline = compute_pid_baseline(
        pid_gains_path=pid_gains_path, n_episodes=pid_baseline_episodes,
        oob_radius=oob_radius, hit_reward=hit_reward, attitude_penalty=attitude_penalty,
        oob_penalty=oob_penalty, streak_penalty_coef=streak_penalty_coef,
        hover_success_steps=hover_success_steps, streak_cap=streak_cap,
        distances=pid_baseline_distances, gains_by_dist_path=pid_gains_by_dist_path,
    )
    if baseline is not None:
        last = rows[-1]
        rl_stats = {
            "success_rate": float(last.get("outcome_hit", 0)) / n_diag_episodes,
            "avg_final_dist": float(last.get("avg_final_dist", 0.0)),
            "avg_reward": float(last.get("reward_mean", 0.0)),
            "avg_steps_survived": float(last.get("avg_steps_survived", 0.0)),
        }
        metrics = [
            ("success_rate", "higher is better"),
            ("avg_final_dist", "lower is better"),
            ("avg_reward", "higher is better"),
            ("avg_steps_survived", "higher is better"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, (key, direction) in zip(axes.flat, metrics):
            ax.bar(["RL (final ckpt)", "PID baseline"],
                   [rl_stats[key], baseline[key]],
                   color=["#2b6cb0", "#a0aec0"])
            ax.set_title(f"{key}\n({direction})")
        fig.suptitle("RL policy vs. tuned PID controller, same task/reward")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "vs_pid_baseline.png")); plt.close(fig)
        print(f"[plot_training_run] wrote {output_dir}/vs_pid_baseline.png "
              f"(RL {rl_stats}, PID {baseline})")


def plot_grad_norm(csv_path, output_dir="plots_final"):
    """Standalone grad_norm-over-time plot, shading the critic_warmup stage -- training_error.png
    already has grad_norm on a shared log-scale axis with value_loss, this is a clearer dedicated
    view for spotting a post-unfreeze spike."""
    rows = _read_csv(csv_path)
    if not rows:
        print(f"[plot_grad_norm] no rows found in {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    timesteps = np.array([float(r["timesteps"]) for r in rows])
    grad_norm = np.array([float(r.get("grad_norm", 0.0) or 0.0) for r in rows])
    stage = [r.get("stage", "") for r in rows]

    plt.figure(figsize=(9, 4.5))
    warmup_mask = np.array([s == "critic_warmup" for s in stage])
    if warmup_mask.any():
        plt.axvspan(timesteps[0], timesteps[warmup_mask][-1], color="tab:orange", alpha=0.12, label="critic_warmup")
    plt.plot(timesteps, grad_norm, marker=".", linewidth=1, color="tab:blue")
    plt.xlabel("timesteps"); plt.ylabel("grad_norm (pre-clip)")
    plt.title("Gradient norm over training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grad_norm.png")); plt.close()
    print(f"[plot_grad_norm] wrote {output_dir}/grad_norm.png")


def plot_grad_norm_3d(csv_path, output_dir="plots_final", y_col="grade"):
    """Landscape-style 3D view of grad_norm: x=timesteps, y=`y_col` (default grade), z=grad_norm, drawn as a
    jet-colored surface with contour lines projected on the floor and this run's path (start dot ... arrowhead)
    riding on top -- the way a loss-landscape descent is usually drawn.

    The SURFACE IS INTERPOLATED, not a measured landscape: a Gaussian-kernel smooth of the logged
    (timesteps, y, grad_norm) chunks that relaxes to the mean grad_norm away from any logged chunk, so
    only the path itself is data. (y=lr, the old choice, gives no surface at all: lr is a fixed function of
    timesteps, so the run only ever visits a single curve of that plane.)"""
    rows = _read_csv(csv_path)
    if len(rows) < 4:
        print(f"[plot_grad_norm_3d] need >= 4 rows in {csv_path}, found {len(rows)}")
        return

    os.makedirs(output_dir, exist_ok=True)
    col = lambda name: np.array([float(r.get(name, 0.0) or 0.0) for r in rows])
    t, y, z = col("timesteps"), col(y_col), col("grad_norm")

    def unit(v):
        return (v - v.min()) / (np.ptp(v) or 1.0)
    tn, yn = unit(t), unit(y)
    sig_t, sig_y, prior_w = 0.05, 0.10, 0.02

    def surface(px, py):
        w = np.exp(-0.5 * (((px[..., None] - tn) / sig_t) ** 2 + ((py[..., None] - yn) / sig_y) ** 2))
        return (w @ z + prior_w * z.mean()) / (w.sum(-1) + prior_w)

    g = np.linspace(0, 1, 80)
    GX, GY = np.meshgrid(g, g)
    GZ = surface(GX, GY)
    to_t = lambda u: t.min() + u * (np.ptp(t) or 1.0)
    to_y = lambda u: y.min() + u * (np.ptp(y) or 1.0)
    X, Y = to_t(GX), to_y(GY)

    dense = np.linspace(0, 1, len(t) * 8)
    order = np.argsort(tn)
    pt, py_ = np.interp(dense, tn[order], tn[order]), np.interp(dense, tn[order], yn[order])
    pz = surface(pt, py_)
    z_range = max(GZ.max() - GZ.min(), 1e-9)
    floor = GZ.min() - 0.35 * z_range
    lift = 0.02 * z_range

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, GZ, cmap="jet", rstride=1, cstride=1, linewidth=0.2,
                           edgecolor=(0, 0, 0, 0.25), antialiased=True, alpha=0.96)
    ax.contour(X, Y, GZ, zdir="z", offset=floor, cmap="jet", levels=8, linewidths=1.3)
    ax.plot(to_t(pt), to_y(py_), pz + lift, color="black", linewidth=2.4, zorder=10)
    ax.plot([to_t(pt[0])], [to_y(py_[0])], [pz[0] + lift], marker="o", markersize=10, color="black", linestyle="none", zorder=11)  # a Line3D, not scatter: scatter is depth-sorted with the surface and can vanish behind it
    # Arrowhead = two line segments, not a patch/quiver: Axes3D depth-sorts patches and collections
    # against each other, so the big surface polygon gets drawn over a patch arrow; lines (zorder) draw last.
    # Geometry is built in unit-cube coordinates (each axis scaled to 0..1) so the head looks the same size
    # despite timesteps ~1e8 vs grade ~1, then mapped back to data units.
    z_top = GZ.max() + 0.05 * z_range
    to_zn = lambda v: (v - floor) / (z_top - floor)
    from_zn = lambda u: floor + u * (z_top - floor)
    k = max(2, len(pt) // 25)
    tip = np.array([pt[-1], py_[-1], to_zn(pz[-1] + lift)])
    direction = tip - np.array([pt[-k], py_[-k], to_zn(pz[-k] + lift)])
    if np.linalg.norm(direction) > 1e-9:
        direction /= np.linalg.norm(direction)
        side = np.cross(direction, [0.0, 0.0, 1.0])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0.0, 0.0])
        for sign in (1, -1):
            wing = tip - 0.10 * direction + sign * 0.05 * side
            ax.plot([to_t(wing[0]), to_t(tip[0])], [to_y(wing[1]), to_y(tip[1])],
                    [from_zn(wing[2]), from_zn(tip[2])], color="black", linewidth=3.6,
                    solid_capstyle="round", zorder=12)

    ax.set_zlim(floor, z_top)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(y_col)
    ax.set_zlabel("grad_norm (pre-clip)")
    ax.set_title("Gradient-norm landscape (interpolated surface; black path = this run)", fontsize=11)
    ax.view_init(elev=27, azim=-62)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="grad_norm")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "grad_norm_3d.png"), dpi=130)
    plt.close(fig)
    print(f"[plot_grad_norm_3d] wrote {output_dir}/grad_norm_3d.png")


def plot_time_to_hit(csv_path, output_dir="plots_final"):
    """avg_hit_time_sec over training -- sim-seconds from episode start to
    hit/hover-success, averaged only over episodes that actually succeeded
    that checkpoint (blank/None when none did, see diagnose_with_model).
    Success = trending down as the policy gets faster, not just hitting more."""
    rows = _read_csv(csv_path)
    if not rows:
        print(f"[plot_time_to_hit] no rows found in {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    timesteps = np.array([float(r["timesteps"]) for r in rows])
    tth = np.array([float(r["avg_hit_time_sec"]) if r.get("avg_hit_time_sec") not in (None, "") else np.nan
                     for r in rows])

    if np.all(np.isnan(tth)):
        print(f"[plot_time_to_hit] avg_hit_time_sec is empty in every row of {csv_path} "
              "(no checkpoint has hit/hover-succeeded yet) -- skipping")
        return

    plt.figure(figsize=(8, 4))
    valid = ~np.isnan(tth)
    plt.plot(timesteps[valid], tth[valid], marker=".", label="avg_hit_time_sec")
    n_missing = int((~valid).sum())
    if n_missing:
        plt.scatter(timesteps[~valid], np.zeros(n_missing), marker="x", color="r",
                     label=f"no hit this checkpoint ({n_missing})")
    plt.xlabel("timesteps"); plt.ylabel("avg time-to-hit (sim seconds)")
    plt.title("Time to hit target (success = trending down, not just hitting more)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "time_to_hit.png")); plt.close()
    print(f"[plot_time_to_hit] wrote {output_dir}/time_to_hit.png")


def _print_suitable_weights(rows, checkpoint_dir, n_diag_episodes, top_n=5, recent_window=20,
                             sustain_window=5, avg_grade_min=0.75, min_grade_min=0.6):
    """A checkpoint only counts as a potential suitable weight if its OWN trailing
    sustain_window (ending at it) has avg grade >= avg_grade_min AND min grade >=
    min_grade_min -- sustained performance, not one lucky spike (grade is already
    computed and stored per-row during training, no re-evaluation needed). Prints the
    top_n qualifying checkpoints all-time, and the top_n qualifying within the last
    recent_window checkpoints -- same format as app.guidance.record_run.rank_checkpoints."""
    def f(row, name):
        return float(row.get(name, 0.0) or 0.0)

    def score(r):
        attitude = f(r, "outcome_attitude-ROLL") + f(r, "outcome_attitude-PITCH")
        return {
            "checkpoint": f"model_{r['timesteps']}.pt",
            "grade": f(r, "grade"),
            "hit": 100.0 * f(r, "outcome_hit") / n_diag_episodes,
            "oob": 100.0 * f(r, "outcome_oob") / n_diag_episodes,
            "attitude": 100.0 * attitude / n_diag_episodes,
            "timeout": 100.0 * f(r, "outcome_timeout") / n_diag_episodes,
        }

    def print_leaderboard(title, entries):
        print(f"\n{title} (from {checkpoint_dir}):")
        if not entries:
            print(f"  none -- no checkpoint's trailing {sustain_window}-checkpoint window met "
                  f"avg>={avg_grade_min}/min>={min_grade_min}")
            return
        for i, s in enumerate(entries, 1):
            print(f"{i}. {s['checkpoint']}  grade={s['grade']:.3f}  hit={s['hit']:.0f}%  "
                  f"oob={s['oob']:.0f}%  attitude={s['attitude']:.0f}%  timeout={s['timeout']:.0f}%")

    scored = [score(r) for r in rows]
    grades = [s["grade"] for s in scored]

    qualified = []
    for i in range(len(scored)):
        if i + 1 < sustain_window:
            continue
        window = grades[i + 1 - sustain_window: i + 1]
        if sum(window) / len(window) >= avg_grade_min and min(window) >= min_grade_min:
            qualified.append((i, scored[i]))

    crit = f"avg>={avg_grade_min}/min>={min_grade_min} over trailing {sustain_window} checkpoints"

    all_time = sorted((s for _, s in qualified), key=lambda s: s["grade"], reverse=True)[:top_n]
    print_leaderboard(f"Potential suitable weights (top {len(all_time)} of {len(qualified)} qualifying [{crit}], all time)",
                       all_time)

    recent_start = len(scored) - recent_window
    recent_qualified = [s for i, s in qualified if i >= recent_start]
    recent_top = sorted(recent_qualified, key=lambda s: s["grade"], reverse=True)[:top_n]
    print_leaderboard(f"Potential suitable weights (top {len(recent_top)} of {len(recent_qualified)} "
                       f"qualifying in last {recent_window} checkpoints)", recent_top)


# ---------------------------------------------------------------------------
# Eval matrix plots -- fixed (start,target) pairs, see app/training/eval_matrix.py
# ---------------------------------------------------------------------------
def plot_eval_matrix_distance(labeled_results, output_path="plots_final/eval_matrix_distance.png"):
    """
    Plots task distance (x) vs. mean final distance actually achieved (y, 0=perfect).
    labeled_results: {label -> list of run_eval_matrix() result dicts}, e.g.
    {"RL": rl_results, "PID": pid_results} to compare on one plot.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 6))
    for label, results in labeled_results.items():
        task_dists = [r["task_dist"] for r in results]
        final_dists = [r["mean_final_dist"] for r in results]
        stds = [r["std_final_dist"] for r in results]
        order = np.argsort(task_dists)
        task_dists = np.array(task_dists)[order]
        final_dists = np.array(final_dists)[order]
        stds = np.array(stds)[order]
        plt.errorbar(task_dists, final_dists, yerr=stds, marker="o", capsize=3, label=label)

    max_d = max(r["task_dist"] for results in labeled_results.values() for r in results)
    plt.plot([0, max_d], [0, 0], "k:", alpha=0.5, label="perfect (final_dist=0)")
    plt.xlabel("task distance -- ||target - start|| (m)")
    plt.ylabel("mean final distance to target (m)")
    plt.title("Correct distance vs. achieved distance across eval matrix")
    plt.legend()
    plt.savefig(output_path); plt.close()
    print(f"[plot_eval_matrix_distance] wrote {output_path}")


def plot_eval_matrix_pairs(labeled_results, output_path="plots_final/eval_matrix_pairs.png"):
    """One bar group per (start, target) pair, showing which specific configs fail."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    labels = list(labeled_results.keys())
    n_pairs = len(next(iter(labeled_results.values())))
    x = np.arange(n_pairs)
    width = 0.8 / len(labels)

    plt.figure(figsize=(max(8, n_pairs * 1.2), 5))
    for i, label in enumerate(labels):
        results = labeled_results[label]
        means = [r["mean_final_dist"] for r in results]
        stds = [r["std_final_dist"] for r in results]
        plt.bar(x + i * width, means, width=width, yerr=stds, capsize=3, label=label)

    pair_labels = [f"{r['start']}->{r['target']}" for r in next(iter(labeled_results.values()))]
    plt.xticks(x + width * (len(labels) - 1) / 2, pair_labels, rotation=45, ha="right", fontsize=7)
    plt.axhline(y=0.0, color="k", linestyle=":", alpha=0.5, label="perfect (final_dist=0)")
    plt.ylabel("mean final distance to target (m)")
    plt.title("Per-config final distance (which specific configs fail)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path); plt.close()
    print(f"[plot_eval_matrix_pairs] wrote {output_path}")


# ---------------------------------------------------------------------------
# DAgger per-distance balancing diagnostics -- see app/control/dagger.py
# ---------------------------------------------------------------------------
def plot_dagger_history(history, output_dir="plots_final"):
    """
    history: list of {"round", "raw_counts", "balanced_counts", "hit_rate"} dicts from
    dagger.py, one per round. Writes dagger_raw_counts.png (pre-balancing pair counts per
    distance, showing long-distance episodes' data volume advantage) and
    dagger_hit_rate.png (per-distance hit rate of the policy flying against PID labels).
    """
    if not history:
        print("[plot_dagger_history] empty history, nothing to plot")
        return
    os.makedirs(output_dir, exist_ok=True)
    rounds = [h["round"] for h in history]
    distances = sorted(history[0]["raw_counts"].keys(), key=lambda d: float(d))

    fig, ax = plt.subplots(figsize=(8, 5))
    for d in distances:
        ax.plot(rounds, [h["raw_counts"][d] for h in history], marker="o", label=f"{d}m")
    ax.set_xlabel("DAgger round"); ax.set_ylabel("pairs collected (raw, before balancing)")
    ax.set_title("Per-distance data volume per round -- imbalance BC was trained on")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dagger_raw_counts.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for d in distances:
        ax.plot(rounds, [h["hit_rate"][d] for h in history], marker="o", label=f"{d}m")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("DAgger round"); ax.set_ylabel("hit_rate (policy-flown episodes)")
    ax.set_title("Per-distance hit rate per round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dagger_hit_rate.png")); plt.close(fig)

    print(f"[plot_dagger_history] wrote dagger_raw_counts.png, dagger_hit_rate.png to {output_dir}/")


def plot_worker_metrics(history, counts, output_dir="plots_final"):
    """Per-worker breakdown of what SubprocVecBaseDroneEnv's parallel envs are
    actually doing over the course of training -- one worker_<i>/ folder per
    OS worker process (i in range(len(counts))), each with every env that
    worker owns plotted as its own line.

    history: {"timesteps": [t0, t1, ...],
              "start_dist": [arr0, arr1, ...],   # each arr shape (num_envs,)
              "live_dist":  [arr0, arr1, ...]}   # snapshotted once per chunk
    counts: SubprocVecBaseDroneEnv.counts -- envs-per-worker, in the same
    order the flat (num_envs,) arrays above are concatenated in, so a
    straight cumulative-sum split recovers each worker's own slice.

    Two plots per worker:
      - target_dist.png: start_dist per env over time -- confirms the
        sampler is actually spreading distances across every worker, not
        just in aggregate.
      - live_dist.png: live current distance-to-target per env over time --
        a diverging/stuck worker shows up here as a line that doesn't trend
        toward its own start_dist.
    """
    if not history["timesteps"]:
        print("[plot_worker_metrics] empty history, nothing to plot")
        return

    timesteps = np.asarray(history["timesteps"])
    start_dist = np.stack(history["start_dist"])  # (n_chunks, num_envs)
    live_dist = np.stack(history["live_dist"])  # (n_chunks, num_envs)

    offsets = np.cumsum([0, *counts])
    for w, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        worker_dir = os.path.join(output_dir, f"worker_{w}")
        os.makedirs(worker_dir, exist_ok=True)
        env_ids = range(lo, hi)

        fig, ax = plt.subplots(figsize=(8, 5))
        for env_idx in env_ids:
            ax.plot(timesteps, start_dist[:, env_idx], alpha=0.7, label=f"env {env_idx}")
        ax.set_xlabel("timesteps"); ax.set_ylabel("target spawn distance (m)")
        ax.set_title(f"worker {w} ({len(env_ids)} envs) -- target spawn distance")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(worker_dir, "target_dist.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for env_idx in env_ids:
            ax.plot(timesteps, live_dist[:, env_idx], alpha=0.7, label=f"env {env_idx}")
        ax.set_xlabel("timesteps"); ax.set_ylabel("live distance to target (m)")
        ax.set_title(f"worker {w} ({len(env_ids)} envs) -- live distance-to-target")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(worker_dir, "live_dist.png")); plt.close(fig)

    print(f"[plot_worker_metrics] wrote {len(counts)} worker_*/ folders to {output_dir}/")


if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "runs/1m_10epochs_v2/metrics.csv"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "plots_final"
    plot_training_run(csv_arg, out_arg)
