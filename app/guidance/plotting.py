import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def plot_training_run(csv_path, output_dir="plots",
                       hover_success_steps=480, n_diag_episodes=20,
                       hit_threshold=0.3, streak_cap=30, max_steps=5000):
    """
    hover_success_steps, n_diag_episodes, hit_threshold, streak_cap, max_steps
    should match whatever RewardConfig / env / diagnostic loop you actually
    trained with -- they're only used to draw target reference lines, not
    to recompute anything, so pass the real values in from your TrainConfig
    if they differ from these defaults.
    """
    rows = _read_csv(csv_path)
    if not rows:
        print(f"[plot_training_run] no rows found in {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    timesteps = np.array([float(r["timesteps"]) for r in rows])

    def col(name, default=0.0):
        return np.array([float(r.get(name, default) or default) for r in rows])

    # --- reward ---
    reward_mean, reward_std = col("reward_mean"), col("reward_std")
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, reward_mean, label="reward_mean")
    plt.fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("timesteps"); plt.ylabel("reward")
    plt.title("Episode reward (mean +/- std, last 10 eps)")
    plt.legend(); plt.savefig(os.path.join(output_dir, "reward.png")); plt.close()

    # --- final distance: target = 0 ---
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, col("avg_final_dist"), label="avg_final_dist")
    plt.plot(timesteps, col("min_final_dist"), label="min_final_dist")
    plt.axhline(y=hit_threshold, color="g", linestyle="--", label=f"hit_threshold={hit_threshold}")
    plt.axhline(y=0.0, color="k", linestyle=":", alpha=0.5, label="target = 0")
    plt.xlabel("timesteps"); plt.ylabel("distance (m)")
    plt.title("Final distance to target (success = converges to 0)")
    plt.legend(); plt.savefig(os.path.join(output_dir, "final_distance.png")); plt.close()

    # --- avg steps survived: target = max_steps ---
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, col("avg_steps_survived"))
    plt.axhline(y=max_steps, color="g", linestyle="--", label=f"max_steps={max_steps}")
    plt.xlabel("timesteps"); plt.ylabel("avg steps survived")
    plt.title("Average episode length (success = approaches max_steps)")
    plt.legend(); plt.savefig(os.path.join(output_dir, "avg_steps_survived.png")); plt.close()

    # --- max hover streak: target = hover_success_steps ---
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, col("max_hover_streak"))
    plt.axhline(y=hover_success_steps, color="g", linestyle="--",
                label=f"hover_success_steps={hover_success_steps}")
    plt.xlabel("timesteps"); plt.ylabel("max hover streak (steps)")
    plt.title("Max hover streak (success = reaches hover_success_steps)")
    plt.legend(); plt.savefig(os.path.join(output_dir, "max_hover_streak.png")); plt.close()

    # --- effective std: target -> low (policy confident, not exploring) ---
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, col("effective_std_mean"))
    plt.axhline(y=0.0, color="k", linestyle=":", alpha=0.5, label="target: trending -> low")
    plt.xlabel("timesteps"); plt.ylabel("effective_std_mean")
    plt.title("Policy action std (success = decaying, not climbing)")
    plt.legend(); plt.savefig(os.path.join(output_dir, "effective_std.png")); plt.close()

    # --- approx KL ---
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, col("approx_kl"), label="approx_kl")
    plt.axhline(y=0.02, color="r", linestyle="--", label="target_kl=0.02")
    plt.xlabel("timesteps"); plt.ylabel("approx_kl")
    plt.title("Approximate KL divergence per update")
    plt.legend(); plt.savefig(os.path.join(output_dir, "approx_kl.png")); plt.close()

    # --- grad_norm: log scale, this is what actually caught the additional=3 blowup ---
    plt.figure(figsize=(8, 4))
    plt.semilogy(timesteps, col("grad_norm"))
    plt.xlabel("timesteps"); plt.ylabel("grad_norm (log scale)")
    plt.title("Pre-clip gradient norm (success = stays low & flat, not spiking)")
    plt.savefig(os.path.join(output_dir, "grad_norm.png")); plt.close()

    # --- value_loss: log scale, same reasoning as grad_norm ---
    plt.figure(figsize=(8, 4))
    plt.semilogy(timesteps, col("value_loss"))
    plt.xlabel("timesteps"); plt.ylabel("value_loss (log scale)")
    plt.title("Value function loss (success = stays low & flat)")
    plt.savefig(os.path.join(output_dir, "value_loss.png")); plt.close()

    # --- policy_loss / entropy_loss: can go negative, so linear/symlog not log ---
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, col("policy_loss"), label="policy_loss")
    plt.plot(timesteps, col("entropy_loss"), label="entropy_loss")
    plt.axhline(y=0.0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("timesteps"); plt.ylabel("loss")
    plt.title("Policy & entropy loss")
    plt.legend(); plt.savefig(os.path.join(output_dir, "policy_entropy_loss.png")); plt.close()

    # --- outcome distribution: target = hover_success -> n_diag_episodes, rest -> 0 ---
    outcome_keys = [k for k in rows[0].keys() if k.startswith("outcome_")]
    outcome_series = [col(k) for k in outcome_keys]
    plt.figure(figsize=(9, 5))
    plt.stackplot(timesteps, *outcome_series, labels=[k.replace("outcome_", "") for k in outcome_keys])
    plt.axhline(y=n_diag_episodes, color="g", linestyle="--",
                label=f"all {n_diag_episodes} eps -> hover_success")
    plt.xlabel("timesteps"); plt.ylabel("count (per checkpoint diagnostic batch)")
    plt.title("Outcome distribution (success = stack fills with hover_success only)")
    plt.legend(loc="upper left")
    _shade_streak_windows(plt.gca(), timesteps, rows, n_diag_episodes)
    plt.savefig(os.path.join(output_dir, "outcome_distribution.png")); plt.close()

    # --- hover_success ratio: the single clearest convergence signal ---
    hover_ratio = col("outcome_hover_success") / n_diag_episodes
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, hover_ratio, marker=".")
    for tier, style in ((0.25, ":"), (0.50, "--"), (0.75, "-."), (1.00, "-")):
        plt.axhline(y=tier, color="g", linestyle=style, alpha=0.5, label=f"{int(tier*100)}%")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("timesteps"); plt.ylabel("hover_success / n_diag_episodes")
    plt.title("Hover success rate per checkpoint")
    _shade_streak_windows(plt.gca(), timesteps, rows, n_diag_episodes)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "hover_success_rate.png")); plt.close()

    print(f"[plot_training_run] wrote plots to {output_dir}/")

    _print_convergence_summary(rows, hover_success_steps, n_diag_episodes,
                                hit_threshold, max_steps)


STREAK_TIERS = (0.25, 0.50, 0.75, 1.00)
STREAK_LEN = 5  # consecutive checkpoints required at a given tier


def _hover_ratios(rows, n_diag_episodes):
    return [float(r.get("outcome_hover_success", 0) or 0) / n_diag_episodes for r in rows]


def _streak_windows(values, threshold, min_len=STREAK_LEN):
    """
    Indices [start, end] (inclusive) of every run where value >= threshold
    for at least min_len consecutive checkpoints in a row -- this is what
    "5 global consecutive 20-streaks" means: not one lucky checkpoint, but
    the diagnostic batch clearing the bar min_len times back to back.
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


def _best_streak_window(rows, n_diag_episodes, tier=1.00):
    """Latest streak window at the given tier, or None if it never happened."""
    ratios = _hover_ratios(rows, n_diag_episodes)
    windows = _streak_windows(ratios, tier)
    return windows[-1] if windows else None


def _shade_streak_windows(ax, timesteps, rows, n_diag_episodes):
    """Overlay the highest tier's streak window(s) as shaded spans on a plot."""
    ratios = _hover_ratios(rows, n_diag_episodes)
    for tier in reversed(STREAK_TIERS):
        windows = _streak_windows(ratios, tier)
        if windows:
            for start_i, end_i in windows:
                ax.axvspan(timesteps[start_i], timesteps[end_i], color="g", alpha=0.12,
                           label=f"{int(tier*100)}% streak (>={STREAK_LEN} ckpts)")
            break  # only shade the highest tier actually achieved


def _print_convergence_summary(rows, hover_success_steps, n_diag_episodes,
                                hit_threshold, max_steps):
    """
    Two-part check:
      1) per-checkpoint sanity table for the LAST row (quick snapshot --
         can look good by luck, doesn't prove stability).
      2) the real signal -- for each hover-success-rate tier (25/50/75/100%),
         has the diagnostic batch stayed at or above that rate for
         STREAK_LEN consecutive checkpoints? A single good checkpoint isn't
         convergence; oscillating good/moving_away_cap checkpoints (as seen
         in earlier runs) will fail this even if the last row looks great.
    """
    last = rows[-1]

    def f(name, default=0.0):
        return float(last.get(name, default) or default)

    attitude_total = f("outcome_attitude-ROLL") + f("outcome_attitude-PITCH")

    checks = [
        ("avg_final_dist -> 0",
         f("avg_final_dist"),
         f("avg_final_dist") < hit_threshold * 0.5),
        ("max_hover_streak -> hover_success_steps",
         f("max_hover_streak"),
         f("max_hover_streak") >= hover_success_steps),
        ("outcome_attitude(ROLL+PITCH) -> 0",
         attitude_total, attitude_total == 0),
        ("outcome_oob -> 0",
         f("outcome_oob"), f("outcome_oob") == 0),
        ("outcome_moving_away_cap -> 0",
         f("outcome_moving_away_cap"), f("outcome_moving_away_cap") == 0),
        (f"outcome_hover_success -> {n_diag_episodes}",
         f("outcome_hover_success"),
         f("outcome_hover_success") >= n_diag_episodes * 0.9),
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
    ratios = _hover_ratios(rows, n_diag_episodes)

    print(f"\n=== Streak check: {STREAK_LEN}+ consecutive checkpoints at each hover-rate tier ===")
    achieved_tier = None
    for tier in STREAK_TIERS:
        windows = _streak_windows(ratios, tier)
        pct = int(tier * 100)
        if windows:
            start_i, end_i = windows[-1]
            length = end_i - start_i + 1
            print(f"  [PASS   ] >= {pct:3d}% hover_success held for {length} checkpoints straight "
                  f"-> timesteps {timesteps[start_i]}..{timesteps[end_i]}")
            achieved_tier = tier
        else:
            print(f"  [not yet] >= {pct:3d}% hover_success for {STREAK_LEN}+ checkpoints straight")

    if achieved_tier == 1.00:
        print("  -> Fully converged: sustained 100% hover rate. Run the manual deterministic "
              "hold-position test to confirm.")
    elif achieved_tier is not None:
        print(f"  -> Partially converged: sustained {int(achieved_tier*100)}% hover rate, "
              f"but never a clean {STREAK_LEN}-checkpoint 100% run. Still oscillating -- "
              f"keep training or investigate what's causing the dips.")
    else:
        print("  -> Not converged: never held even the lowest tier for "
              f"{STREAK_LEN} checkpoints in a row.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "runs/1m_10epochs_v2/metrics.csv"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "plots"
    plot_training_run(csv_arg, out_arg)