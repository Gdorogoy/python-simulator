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
    Runs the tuned PID controller through the same env/reward machinery
    used to score the RL policy, and returns the same summary stats
    diagnose_with_model would -- so the RL run's final checkpoint can be
    compared against it apples-to-apples.

    Two modes:
      - distances=None (default): old single-target behavior, using the one
        gains file at pid_gains_path against env's default target.
      - distances=(3, 10, 50, 150, ...): multi-distance mode -- loads
        gains_by_dist_path (a {dist: gains} json, e.g. best_pid_gains_per_dist.json)
        and runs each distance x all 4 cardinal directions with ITS OWN
        tuned gains and oob_radius (via build_eval_pairs), pooling results
        across the whole curriculum -- matches what phase_1_training.py
        actually trains/evaluates the RL policy on, instead of comparing
        against a single fixed-distance baseline.

    Returns None (with a printed reason) if the relevant gains file is
    missing, rather than raising -- this plot is a nice-to-have, not
    required to see the rest of the training dashboard.
    """
    from app.environmental.interceptor_drone import InterceptorDroneEnv
    from app.control.pid_hover import PIDHoverController
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
            pid = PIDHoverController(**gains)
            dist_oob_radius = max(oob_radius, dist * 3.0)

            reward_cfg = RewardConfig(
                oob_radius=dist_oob_radius, hit_reward=hit_reward, attitude_penalty=attitude_penalty,
                oob_penalty=oob_penalty, streak_penalty_coef=streak_penalty_coef,
                hover_success_steps=hover_success_steps, streak_cap=streak_cap,
            )
            env = InterceptorDroneEnv(make_reward_fn(reward_cfg), pid_gains_path=None)

            for start, target in build_eval_pairs(oob_radius=dist_oob_radius, distances=(dist,)):
                for _ in range(n_per_pair):
                    obs, _ = env.reset(start_pos=start.copy(), target_pos=target.copy())
                    pid.reset()
                    done = False
                    step_count = 0
                    ep_reward = 0.0
                    reason = None
                    while not done and step_count < n_steps:
                        action = pid.compute_action(env.drone_state, env.target_pos)
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
        pid = PIDHoverController(**gains)

        reward_cfg = RewardConfig(
            oob_radius=oob_radius, hit_reward=hit_reward, attitude_penalty=attitude_penalty,
            oob_penalty=oob_penalty, streak_penalty_coef=streak_penalty_coef,
            hover_success_steps=hover_success_steps, streak_cap=streak_cap,
        )
        env = InterceptorDroneEnv(make_reward_fn(reward_cfg), pid_gains_path=None)

        for _ in range(n_episodes):
            obs, _ = env.reset()
            pid.reset()
            done = False
            step_count = 0
            ep_reward = 0.0
            reason = None
            while not done and step_count < n_steps:
                action = pid.compute_action(env.drone_state, env.target_pos)
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
    Indices [start, end] (inclusive) of every run where value >= threshold
    for at least min_len consecutive checkpoints in a row -- not one lucky
    checkpoint, but the diagnostic batch clearing the bar min_len times
    back to back.
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
# Plots -- 5 figures total (down from 11), each covering one question:
#   1) training_error     -- is the optimization itself behaving?
#   2) policy_std          -- is exploration decaying as expected?
#   3) distance_distribution -- how close/consistent is the drone to target?
#   4) success_and_outcomes -- is it succeeding, and why does it fail when it does?
#   5) vs_pid_baseline      -- did RL actually beat the classical controller?
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
    hover_success_steps, n_diag_episodes, hit_threshold, streak_cap, max_steps
    should match whatever RewardConfig / env / diagnostic loop you actually
    trained with -- they're only used to draw target reference lines and
    (for the reward-related ones) to rebuild an equivalent RewardConfig for
    the PID baseline comparison, not to recompute anything from the CSV.

    pid_baseline_distances/pid_gains_by_dist_path: pass these (e.g.
    distances=(3,10,50,150), gains_by_dist_path="app/control/best_pid_gains_per_dist.json")
    to compare the RL checkpoint against the PID baseline across the FULL
    distance curriculum it was actually trained on, instead of the single
    fixed target pid_gains_path/env default gives you.
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
    plt.axhline(y=hit_threshold, color="g", linestyle="--", label=f"hit_threshold={hit_threshold}")
    plt.axhline(y=0.0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("timesteps"); plt.ylabel("distance (m)")
    plt.title("Final distance to target (success = converges to 0, band narrows)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "distance_distribution.png")); plt.close()

    # --- 4) success_and_outcomes: hit rate + outcome breakdown ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    hit_ratio = col("outcome_hit") / n_diag_episodes
    ax1.plot(timesteps, hit_ratio, marker=".")
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
    _print_convergence_summary(rows, hover_success_steps, n_diag_episodes, hit_threshold)

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


def _print_convergence_summary(rows, hover_success_steps, n_diag_episodes, hit_threshold):
    """
    Two-part check:
      1) per-checkpoint sanity table for the LAST row (quick snapshot --
         can look good by luck, doesn't prove stability).
      2) the real signal -- for each hover-success-rate tier (25/50/75/100%),
         has the diagnostic batch stayed at or above that rate for
         STREAK_LEN consecutive checkpoints? A single good checkpoint isn't
         convergence; oscillating good/moving_away_cap checkpoints will fail
         this even if the last row looks great.
    """
    last = rows[-1]

    def f(name, default=0.0):
        return float(last.get(name, default) or default)

    attitude_total = f("outcome_attitude-ROLL") + f("outcome_attitude-PITCH")

    checks = [
        ("avg_final_dist -> 0",
         f("avg_final_dist"),
         f("avg_final_dist") < hit_threshold * 0.5),
        ("outcome_attitude(ROLL+PITCH) -> 0",
         attitude_total, attitude_total == 0),
        ("outcome_oob -> 0",
         f("outcome_oob"), f("outcome_oob") == 0),
        ("outcome_moving_away_cap -> 0",
         f("outcome_moving_away_cap"), f("outcome_moving_away_cap") == 0),
        (f"outcome_hit -> {n_diag_episodes}",
         f("outcome_hit"),
         f("outcome_hit") >= n_diag_episodes * 0.9),
    ]

    print(f"\n=== Snapshot check @ timestep {last['timesteps']} (last checkpoint only) ===")
    passed = 0
    for label, value, ok in checks:
        mark = "PASS" if ok else "not yet"
        print(f"  [{mark:7s}] {label:45s} current={value}")
        passed += int(ok)
    print(f"  {passed}/{len(checks)} criteria met.")
    print("=" * 50)

    timesteps = [r["timesteps"] for r in rows]
    ratios = _hit_ratios(rows, n_diag_episodes)

    print(f"\n=== Streak check: {STREAK_LEN}+ consecutive checkpoints at each hit-rate tier ===")
    achieved_tier = None
    for tier in STREAK_TIERS:
        windows = _streak_windows(ratios, tier)
        pct = int(tier * 100)
        if windows:
            start_i, end_i = windows[-1]
            length = end_i - start_i + 1
            print(f"  [PASS   ] >= {pct:3d}% hit rate held for {length} checkpoints straight "
                  f"-> timesteps {timesteps[start_i]}..{timesteps[end_i]}")
            achieved_tier = tier
        else:
            print(f"  [not yet] >= {pct:3d}% hit rate for {STREAK_LEN}+ checkpoints straight")

    if achieved_tier == 1.00:
        print("  -> Fully converged: sustained 100% hit rate. Run the manual deterministic "
              "hold-position test to confirm.")
    elif achieved_tier is not None:
        print(f"  -> Partially converged: sustained {int(achieved_tier*100)}% hit rate, "
              f"but never a clean {STREAK_LEN}-checkpoint 100% run. Still oscillating -- "
              f"keep training or investigate what's causing the dips.")
    else:
        print("  -> Not converged: never held even the lowest tier for "
              f"{STREAK_LEN} checkpoints in a row.")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Eval matrix plots -- fixed (start,target) pairs, see app/training/eval_matrix.py
# ---------------------------------------------------------------------------
def plot_eval_matrix_distance(labeled_results, output_path="plots_final/eval_matrix_distance.png"):
    """
    Option A: x = task_dist ("correct" distance the episode had to close),
    y = mean_final_dist actually achieved (lower is better; 0 = perfect).
    labeled_results: {label -> list of run_eval_matrix() result dicts},
    e.g. {"RL": rl_results, "PID": pid_results} to compare on one plot.
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
    """
    Option B: one group of bars per (start,target) pair, one bar per labeled
    result set -- shows which SPECIFIC configs fail, not just the overall trend.
    """
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
    history: list of {"round": int, "raw_counts": {dist:int}, "balanced_counts": {dist:int},
    "hit_rate": {dist:float}} accumulated by dagger.py, one entry per round.

    Two plots:
      1) dagger_raw_counts.png -- pairs collected per distance per round BEFORE
         balancing (shows the imbalance: long-distance episodes produce far
         more transitions than short ones, so unweighted aggregation lets them
         dominate the BC loss).
      2) dagger_hit_rate.png -- per-distance hit rate per round (the POLICY
         flying against the PID's labels, not the PID itself) -- the actual
         signal for whether balancing is helping the short-range case recover.
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


if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "runs/1m_10epochs_v2/metrics.csv"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "plots_final"
    plot_training_run(csv_arg, out_arg)
