"""Copy a finished run's artifacts out of the Windows project folder into the CI workspace.

    uv run app/workflow/collect_results.py                    # profile: acro (its checkpoint dir)
    uv run app/workflow/collect_results.py --profile angle
    uv run app/workflow/collect_results.py --run-dir runs/my_run   # explicit run folder
    uv run app/workflow/collect_results.py --profile acro --dry-run



The run folder defaults to the profile's `checkpoint_dir` in app/workflow/profiles.py, so workflow.py and this script
always agree. Copies BEST/best, metrics, run_config, summary and the newest .pt/.onnx into ./<run dir>, which the
workflow then uploads as an artifact.




to run

cd /mnt/c/Users/Pc-Egor/PycharmProjects/WelcomeScreen/python-simulator

# see the command it will build, don't run it
uv run app/workflow/workflow.py --profile angle_ground_150_350 --dry-run

# run with overrides
uv run app/workflow/workflow.py --profile angle_ground_150_350 \
  --set num_envs=4096 \
  --set total_timesteps=10000000 \
  --set distance-high=500

# and collect the artifacts the same way CI does

uv run app/workflow/collect_results.py --profile angle_ground_150_350


--set rules (from workflow.py):
- flag may be written with or without -- (num_envs / --num_envs);
- value is parsed as JSON first, so 4096, 1e-4, true, 3.0 all work; anything else stays a string;
- --set handoff=false drops a flag entirely (that's how you remove one);
- a dict becomes a JSON flag, a tuple/list spreads over several tokens.
For the reward, don't inline JSON with spaces — it survives shell quoting here but not through CI. Point at a file instead; the trainer accepts either:
--set reward-json=app/workflow/rewards/stage1.json


"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.workflow.profiles import WIN_PROJECT, get_profile, profile_names

EXACT_FILES = [
    "best.json",
    "BEST.json",
    "latest_full.pt",
    "metrics.csv",
    "run_config.json",
    "summary.json",
    "summary.png",
]


def newest_files(
    folder: Path, ext: str, count: int, exclude: set[str] | None = None
) -> list[Path]:
    exclude = exclude or set()
    matches = [
        f for f in folder.glob(f"*{ext}") if f.is_file() and f.name not in exclude
    ]
    matches.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return matches[:count]


def resolve_src_dir(win_project: Path, run_dir: Path) -> tuple[Path, Path]:
    """(directory to copy from, its path relative to the Windows project). Picks the newest timestamped subfolder
    when the trainer wrote one instead of the files directly in `run_dir`."""
    src = run_dir if run_dir.is_absolute() else win_project / run_dir
    if not src.exists():
        raise FileNotFoundError(f"run directory not found: {src}")
    subdirs = [d for d in src.iterdir() if d.is_dir()]
    if subdirs and not any((src / name).exists() for name in EXACT_FILES):
        src = max(subdirs, key=lambda d: d.stat().st_mtime)
    try:
        rel = src.relative_to(win_project)
    except ValueError:  # absolute run dir outside the project: mirror its basename
        rel = Path(src.name)
    return src, rel


def collect(src_dir: Path, dst_dir: Path, dry_run: bool = False) -> list[Path]:
    selected: list[Path] = []
    for name in EXACT_FILES:
        f = src_dir / name
        if f.exists():
            selected.append(f)
    selected.extend(newest_files(src_dir, ".pt", 2, exclude={"latest_full.pt"}))
    selected.extend(newest_files(src_dir, ".onnx", 2))

    copied: list[Path] = []
    seen: set[str] = set()
    for f in selected:
        if f.name in seen:
            continue
        seen.add(f.name)
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_dir / f.name)
        copied.append(f)
    return copied


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--profile",
        default="acro",
        choices=profile_names(),
        help="profile whose checkpoint dir to collect",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="explicit run folder (overrides the profile's checkpoint dir)",
    )
    p.add_argument(
        "--win-project",
        default=WIN_PROJECT,
        help="Windows project folder on the runner (default: $WIN_PROJECT)",
    )
    p.add_argument(
        "--out", default=".", help="destination root in this workspace (default: cwd)"
    )
    p.add_argument("--dry-run", action="store_true", help="list what would be copied")
    args = p.parse_args(argv)

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(get_profile(args.profile)["checkpoint_dir"])
    )
    try:
        src_dir, rel_target = resolve_src_dir(Path(args.win_project), run_dir)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    dst_dir = Path(args.out).resolve() / rel_target
    copied = collect(src_dir, dst_dir, args.dry_run)
    if not copied:
        print(f"nothing to collect in {src_dir}")
        return 1
    for f in copied:
        print(
            f"{'would collect' if args.dry_run else 'Collected'}: {rel_target / f.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
