"""Post-hoc analysis of an Optuna study written by optuna_search_isaac.py -- ranks trials by how well
they HOLD their grade after the actor unfreezes, instead of by the last chunk's grade (which the
objective returns, and which is a single 5-episode sample: one failed episode = -0.2 grade).

Stdlib only (sqlite3), no isaaclab/optuna import, so it runs in any python:
    python app/guidance/analyze_optuna_isaac.py
    python app/guidance/analyze_optuna_isaac.py --top 20 --frozen-chunks 16

Per trial, from the per-chunk grades Optuna stored as intermediate values:
  baseline = mean grade over the frozen-actor chunks (same policy every chunk, so this is the imitation
             policy's level; the first --skip-chunks no-update chunks are excluded because grad_norm=0
             there inflates grade by ~0.04)
  delta    = mean grade after unfreeze - baseline   (0 = held, negative = PPO degraded it)
  vol      = mean |chunk-to-chunk change| after unfreeze
The frozen chunks double as a noise measurement: the policy doesn't change there, so their spread is pure
eval noise, and (spread / sqrt(post chunks)) is the smallest delta difference worth believing.
"""
import argparse
import csv
import os
import sqlite3
import statistics as st


def load_trials(db_path, study_name):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if study_name is None:
        names = [r[0] for r in c.execute("select study_name from studies")]
        if len(names) != 1:
            raise SystemExit(f"pass --study-name, study db has: {names}")
        study_name = names[0]
    rows = c.execute(
        """select t.trial_id, t.number, tv.value from trials t
           join studies s on s.study_id = t.study_id
           join trial_values tv on tv.trial_id = t.trial_id
           where s.study_name = ? and t.state = 'COMPLETE' order by t.number""", (study_name,)).fetchall()
    trials = []
    for trial_id, number, final in rows:
        grades = [r[0] for r in c.execute(
            "select intermediate_value from trial_intermediate_values "
            "where trial_id = ? and intermediate_value is not null order by step", (trial_id,))]
        params = dict(c.execute("select param_name, param_value from trial_params where trial_id = ?", (trial_id,)))
        trials.append({"number": number, "final": final, "grades": grades, "params": params})
    conn.close()
    return study_name, trials


def analyze(trial, frozen_chunks, skip_chunks, good_grade):
    g = trial["grades"]
    warm, post = g[skip_chunks:frozen_chunks], g[frozen_chunks:]
    baseline = st.mean(warm)
    return {
        "number": trial["number"], "final": trial["final"], "baseline": baseline,
        "post_mean": st.mean(post), "delta": st.mean(post) - baseline,
        "vol": st.mean(abs(b - a) for a, b in zip(post, post[1:])),
        "last5": st.mean(g[-5:]), "post_min": min(post),
        "frac_good": sum(x >= good_grade for x in post) / len(post),
        **{f"p_{k}": v for k, v in trial["params"].items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="runs/optuna_isaac/study.db")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--frozen-chunks", type=int, default=16, help="chunks with the actor frozen (MIN_WARMUP_CHUNKS)")
    ap.add_argument("--skip-chunks", type=int, default=3, help="no-update chunks excluded from baseline (RUN_START_FROZEN_CHUNKS)")
    ap.add_argument("--good-grade", type=float, default=0.9)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="runs/optuna_isaac/trial_analysis.csv")
    args = ap.parse_args()

    study_name, trials = load_trials(args.db, args.study_name)
    trials = [t for t in trials if len(t["grades"]) > args.frozen_chunks + 1]
    res = [analyze(t, args.frozen_chunks, args.skip_chunks, args.good_grade) for t in trials]
    res.sort(key=lambda r: r["delta"], reverse=True)

    warm_spread = st.mean(st.pstdev(t["grades"][args.skip_chunks:args.frozen_chunks]) for t in trials)
    n_post = len(trials[0]["grades"]) - args.frozen_chunks
    resolution = warm_spread / n_post ** 0.5
    print(f"study={study_name}  trials analyzed={len(res)}  chunks/trial={len(trials[0]['grades'])} "
          f"(frozen={args.frozen_chunks}, post-unfreeze={n_post})")
    print(f"mean baseline (frozen actor) = {st.mean(r['baseline'] for r in res):.3f}   "
          f"mean post-unfreeze = {st.mean(r['post_mean'] for r in res):.3f}")
    print(f"eval noise (spread of a frozen-actor trial's grade) = {warm_spread:.3f} per chunk -> "
          f"deltas closer than ~{2 * resolution:.2f} are indistinguishable\n")

    print(f"top {args.top} by delta (least degradation after unfreeze):")
    print("rank  trial  baseline  post_mean  delta   vol   last5  post_min  frac>=%.1f  final" % args.good_grade)
    for i, r in enumerate(res[:args.top], 1):
        print(f"{i:>4}  {r['number']:>5}  {r['baseline']:.3f}     {r['post_mean']:.3f}     {r['delta']:+.3f}  "
              f"{r['vol']:.3f}  {r['last5']:.3f}  {r['post_min']:+.3f}    {r['frac_good']:.2f}      {r['final']:.3f}")

    print("\ndistribution of delta over ALL trials:")
    buckets = [(-0.05, "held (>= -0.05)"), (-0.15, "-0.05..-0.15"), (-0.30, "-0.15..-0.30"),
               (-0.50, "-0.30..-0.50"), (float("-inf"), "worse than -0.50")]
    hi = float("inf")
    for lo, label in buckets:
        n = sum(lo <= r["delta"] < hi for r in res)
        print(f"  {label:<18} {n:>3}  {'#' * n}")
        hi = lo
    within = sum(res[0]["delta"] - r["delta"] <= 2 * resolution for r in res)
    print(f"\n{within} trials are within noise of the best delta; "
          f"{sum(r['frac_good'] > 0 for r in res)} trials have any post-unfreeze chunk >= {args.good_grade}; "
          f"best frac of post chunks >= {args.good_grade}: {max(r['frac_good'] for r in res):.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = sorted({k for r in res for k in r}, key=lambda k: (k.startswith("p_"), k))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(res)
    print(f"\nfull per-trial table ({len(res)} rows, with hyperparameters) -> {args.out}")


if __name__ == "__main__":
    main()
