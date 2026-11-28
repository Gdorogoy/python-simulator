"""Ground-truth labels -> YOLO per-frame text — plan.md §3.5/§5.1, docs.md A2.

`labels.npz` stores boxes in *native raster* coordinates: fractions of the full 1920x1080 or
1280x1024 image. Training images are a centred square crop, so every box must be rewritten
against that crop before Ultralytics can read it:

    ox   = (W - crop_px) / 2
    px'  = cx * W - ox
    cx'  = px' / crop_px

Nothing is resized, so mrad/px is untouched: a drone that is 4 px wide in the native frame is
still 4 px wide in the crop, because they are the same pixels. Only the ruler changes.

One function owns the crop origin (`crop_origin`). The pixel crop in `generate.py` must import
it rather than recompute it: a one-pixel disagreement moves every box silently, without an
error, and only shows up as an unexplained recall loss after training.

Demo:

    python -m app.datagen.export_yolo --demo
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:  # avoids importing the config module at runtime just for a type
    from app.datagen.config import DatagenConfig

# Columns of labels.npz, in the order the spec lists them (plan.md §5).
LABEL_FIELDS = ("frame", "class_id", "cx", "cy", "w", "h", "track_id", "occluded")

# Drop reasons. Counted separately so a silent 20 % loss rate cannot go unnoticed.
DROP_OUTSIDE = "outside"
DROP_TOO_SMALL = "too_small"
DROP_INVALID = "invalid"

MIN_BOX_PX = 2.0  # plan.md §5.1: a 1 px box is label noise, not a target


@dataclass(frozen=True)
class CropWindow:
    """The centred square cut out of a native raster, in native pixels."""

    width: int
    height: int
    crop_px: int
    ox: int
    oy: int

    @classmethod
    def of(cls, native_wh: tuple[int, int], crop_px: int) -> "CropWindow":
        ox, oy = crop_origin(native_wh, crop_px)
        return cls(width=native_wh[0], height=native_wh[1], crop_px=crop_px, ox=ox, oy=oy)


def crop_origin(native_wh: tuple[int, int], crop_px: int) -> tuple[int, int]:
    """Top-left corner of the centred crop, in native pixels.

    Raises instead of rounding: a half-pixel offset shifts every box by half a pixel forever,
    and no test downstream would ever name it.
    """
    width, height = native_wh
    if crop_px <= 0:
        raise ValueError(f"crop_px must be positive, got {crop_px}")
    if crop_px > min(width, height):
        raise ValueError(
            f"crop_px {crop_px} exceeds native raster {width}x{height}; "
            "that would be an upscale, which plan.md §3.5 bans"
        )
    dx, dy = width - crop_px, height - crop_px
    if dx % 2 or dy % 2:
        raise ValueError(
            f"crop of {crop_px} in {width}x{height} leaves an odd margin "
            f"({dx}, {dy}); the crop would sit on a half pixel"
        )
    return dx // 2, dy // 2


def native_pixels(
    cx: float, cy: float, w: float, h: float, window: CropWindow
) -> tuple[float, float, float, float]:
    """Native-coordinate box -> pixel edges (x1, y1, x2, y2). Debug and drawing aid."""
    return (
        (cx - w / 2.0) * window.width,
        (cy - h / 2.0) * window.height,
        (cx + w / 2.0) * window.width,
        (cy + h / 2.0) * window.height,
    )


def _remap_with_reason(
    cx: float, cy: float, w: float, h: float, window: CropWindow, min_box_px: float
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    if not np.all(np.isfinite((cx, cy, w, h))) or w <= 0.0 or h <= 0.0:
        return None, DROP_INVALID

    c, ox, oy = window.crop_px, window.ox, window.oy
    x1, y1, x2, y2 = native_pixels(cx, cy, w, h, window)

    # The centre decides membership, before clipping. A box whose middle lies outside the crop
    # describes an object the training image does not really contain.
    if not (ox <= (x1 + x2) / 2.0 < ox + c and oy <= (y1 + y2) / 2.0 < oy + c):
        return None, DROP_OUTSIDE

    # Clip to the crop, then take the centre of what survived: the visible extent is the label.
    x1, x2 = max(x1, ox), min(x2, ox + c)
    y1, y2 = max(y1, oy), min(y2, oy + c)
    if (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
        return None, DROP_TOO_SMALL

    box = (
        float(np.clip(((x1 + x2) / 2.0 - ox) / c, 0.0, 1.0)),
        float(np.clip(((y1 + y2) / 2.0 - oy) / c, 0.0, 1.0)),
        float(np.clip((x2 - x1) / c, 0.0, 1.0)),
        float(np.clip((y2 - y1) / c, 0.0, 1.0)),
    )
    return box, None


def remap_box(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    window: CropWindow,
    min_box_px: float = MIN_BOX_PX,
) -> tuple[float, float, float, float] | None:
    """One native box -> crop-normalised box, or ``None`` if it must be dropped."""
    box, _ = _remap_with_reason(cx, cy, w, h, window, min_box_px)
    return box


@dataclass
class ExportStats:
    frames: int = 0
    boxes_in: int = 0
    boxes_out: int = 0
    dropped_outside: int = 0
    dropped_too_small: int = 0
    dropped_invalid: int = 0

    @property
    def dropped(self) -> int:
        return self.dropped_outside + self.dropped_too_small + self.dropped_invalid

    def line(self, tag: str = "") -> str:
        head = f"{tag}  " if tag else ""
        return (
            f"{head}{self.frames} frames\n"
            f"  boxes in        {self.boxes_in}\n"
            f"  written         {self.boxes_out}\n"
            f"  dropped outside {self.dropped_outside}\n"
            f"  dropped <2 px   {self.dropped_too_small}\n"
            f"  dropped invalid {self.dropped_invalid}"
        )


def load_labels(source: str | Path | Mapping[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
    """Read labels.npz from a path, a mapping of columns, or one structured array."""
    if isinstance(source, (str, Path)):
        with np.load(source) as npz:
            return {name: np.asarray(npz[name]) for name in npz.files}
    if isinstance(source, np.ndarray):
        if source.dtype.names is None:
            raise TypeError("expected a structured array or a column mapping")
        return {name: source[name] for name in source.dtype.names}
    return {name: np.asarray(values) for name, values in source.items()}


def export_clip(
    labels: str | Path | Mapping[str, np.ndarray] | np.ndarray,
    *,
    cfg: "DatagenConfig",
    modality: str,
    out_dir: str | Path,
    min_box_px: float = MIN_BOX_PX,
    n_frames: int | None = None,
    precision: int = 6,
) -> ExportStats:
    """Write one ``frame_XXXX.txt`` per frame under ``out_dir/labels/``.

    ``n_frames`` pads the output with empty files for frames that have no rows: YOLO pairs an
    image with a same-named label file, and a missing file is treated differently from an
    empty one.
    """
    columns = load_labels(labels)
    missing = [name for name in ("frame", "class_id", "cx", "cy", "w", "h") if name not in columns]
    if missing:
        raise KeyError(f"labels are missing columns: {missing}")

    frames = np.asarray(columns["frame"]).astype(int)
    if frames.size and frames.min() < 0:
        raise ValueError("frame indices must be non-negative")

    if n_frames is None:
        n_frames = int(frames.max()) + 1 if frames.size else 0
    elif frames.size and frames.max() >= n_frames:
        raise ValueError(f"frame index {frames.max()} outside n_frames={n_frames}")

    window = CropWindow.of(cfg.native_res(modality), cfg.train_crop_px(modality))
    out_labels = Path(out_dir) / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[tuple[int, float, float, float, float]]] = {i: [] for i in range(n_frames)}
    stats = ExportStats(frames=n_frames, boxes_in=int(frames.size))

    for row in range(frames.size):
        frame = int(frames[row])
        cls = int(columns["class_id"][row])
        box, reason = _remap_with_reason(
            float(columns["cx"][row]),
            float(columns["cy"][row]),
            float(columns["w"][row]),
            float(columns["h"][row]),
            window,
            min_box_px,
        )
        if box is None:
            if reason == DROP_OUTSIDE:
                stats.dropped_outside += 1
            elif reason == DROP_TOO_SMALL:
                stats.dropped_too_small += 1
            else:
                stats.dropped_invalid += 1
            continue
        grouped[frame].append((cls, *box))
        stats.boxes_out += 1

    for frame in range(n_frames):
        # Deterministic order: class first, then position, so two runs on the same npz produce
        # byte-identical files and a diff means a real change.
        rows = sorted(grouped[frame], key=lambda item: (item[0], item[1], item[2]))
        text = "".join(
            f"{cls} {cx:.{precision}f} {cy:.{precision}f} {w:.{precision}f} {h:.{precision}f}\n"
            for cls, cx, cy, w, h in rows
        )
        (out_labels / f"frame_{frame:04d}.txt").write_text(text, encoding="ascii", newline="\n")

    return stats


# --- demo -----------------------------------------------------------------------------


def _demo_fixture() -> dict[str, np.ndarray]:
    """Four actors on a Tier 2 IR raster, one of each fate: kept, clipped, outside, too small."""
    rows = [
        # frame, class, cx, cy, w, h, track, occluded
        (0, 0, 0.2400, 0.510, 0.0400, 0.038, 7, 0),  # quad, inside
        (1, 0, 0.2450, 0.511, 0.0400, 0.038, 7, 0),
        (2, 0, 0.2500, 0.512, 0.0400, 0.039, 7, 1),
        (0, 3, 0.5000, 0.300, 0.0080, 0.007, 3, 0),  # bird flock
        (0, 3, 0.5160, 0.295, 0.0075, 0.007, 4, 0),
        (0, 3, 0.5330, 0.302, 0.0090, 0.008, 5, 0),
        (1, 2, 0.0550, 0.400, 0.0040, 0.004, 9, 0),  # decoy left of the crop -> outside
        (1, 2, 0.1015, 0.700, 0.0012, 0.0012, 10, 0),  # 1.5 px wide -> too small
    ]
    data = {name: [] for name in LABEL_FIELDS}
    for row in rows:
        for name, value in zip(LABEL_FIELDS, row):
            data[name].append(value)
    return {name: np.asarray(values) for name, values in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialise YOLO labels for one clip.")
    parser.add_argument("--demo", action="store_true", help="run the self-contained fixture")
    parser.add_argument("--out", default="/tmp/yolo_demo")
    args = parser.parse_args()

    if not args.demo:
        parser.error("nothing to do without --demo (real clips arrive with Milestone B)")

    from app.datagen.config import DatagenConfig

    cfg = DatagenConfig(sensor_tier="tier2")
    modality = "ir"
    columns = _demo_fixture()
    window = CropWindow.of(cfg.native_res(modality), cfg.train_crop_px(modality))

    print(f"tier2/{modality}  native {window.width}x{window.height}  crop {window.crop_px}")
    print(f"origin  ox={window.ox}  oy={window.oy}")
    print()
    print(f"{'f':>2} {'cls':>3} {'status':<9} {'native x1..x2   y1..y2':<34} {'crop x1..x2   y1..y2':<32} yolo")

    for row in range(columns["frame"].size):
        frame = int(columns["frame"][row])
        cls = int(columns["class_id"][row])
        cx, cy = float(columns["cx"][row]), float(columns["cy"][row])
        w, h = float(columns["w"][row]), float(columns["h"][row])
        box, reason = _remap_with_reason(cx, cy, w, h, window, MIN_BOX_PX)
        x1, y1, x2, y2 = native_pixels(cx, cy, w, h, window)
        native = f"{x1:7.1f}..{x2:7.1f} {y1:7.1f}..{y2:7.1f}"
        if box is None:
            print(f"{frame:>2} {cls:>3} {reason:<9} {native:<34} {'-':<32} -")
            continue
        bx1 = (box[0] - box[2] / 2.0) * window.crop_px
        bx2 = (box[0] + box[2] / 2.0) * window.crop_px
        by1 = (box[1] - box[3] / 2.0) * window.crop_px
        by2 = (box[1] + box[3] / 2.0) * window.crop_px
        crop = f"{bx1:7.1f}..{bx2:7.1f} {by1:7.1f}..{by2:7.1f}"
        line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        print(f"{frame:>2} {cls:>3} {'kept':<9} {native:<34} {crop:<32} {line}")

    stats = export_clip(columns, cfg=cfg, modality=modality, out_dir=args.out, n_frames=3)
    print()
    print(stats.line(f"wrote {args.out}/labels"))
    for path in sorted((Path(args.out) / "labels").glob("*.txt")):
        body = path.read_text(encoding="ascii")
        print(f"\n{path.name}  ({len(body.splitlines())} boxes)")
        print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
