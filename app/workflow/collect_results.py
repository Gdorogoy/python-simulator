import shutil
import sys
from pathlib import Path

WIN_PROJECT = Path(
    "/mnt/c/Users/Pc-Egor/PycharmProjects/WelcomeScreen/python-simulator"
)

RUN_REL_DIR = Path("runs/interceptor_v1_bckts_runner30")

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


def main():
    src_dir = WIN_PROJECT / RUN_REL_DIR
    if not src_dir.exists():
        print(f"Run directory not found: {src_dir}")
        sys.exit(1)

    # If the trainer creates a timestamped subfolder inside RUN_REL_DIR, pick the newest one
    subdirs = [d for d in src_dir.iterdir() if d.is_dir()]
    if subdirs and not any((src_dir / name).exists() for name in EXACT_FILES):
        src_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
        rel_target = src_dir.relative_to(WIN_PROJECT)
    else:
        rel_target = RUN_REL_DIR

    dst_dir = Path.cwd() / rel_target
    dst_dir.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []
    for name in EXACT_FILES:
        f = src_dir / name
        if f.exists():
            selected.append(f)

    selected.extend(newest_files(src_dir, ".pt", 2, exclude={"latest_full.pt"}))
    selected.extend(newest_files(src_dir, ".onnx", 2))

    seen = set()
    for f in selected:
        if f.name in seen:
            continue
        seen.add(f.name)
        shutil.copy2(f, dst_dir / f.name)
        print(f"Collected: {rel_target / f.name}")


if __name__ == "__main__":
    main()
