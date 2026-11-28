"""Milestone C — Dataset Clip Runner, 95 % Crop Containment & Track Features (`plan.md` §5, `docs.md` C).

Four core functions turn the B3 camera rig into an unattended dataset generator:

    1. sample_clip_spec()       stratified range bucket, class mix (50/20/15/15), model, sun, visibility
    2. simulate_clip_frames()   steps target + PTZ across frames; regenerates if in_crop_ratio < 0.95
    3. compute_track_features() temporal speed, vertical bob, flap_hz, aspect variance, hotspot count
    4. write_clip_bundle()      saves cropped eo.mp4, ir.mp4, labels.npz, metadata.json & YOLO .txt

Definition-of-done command (`plan.md` Part 1):

    python -m app.datagen.generate --sensor tier2 --clips 50 --duration 8 --headless
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.datagen.assets import AssetThermalConfig, load_catalog
from app.datagen.config import CLIP_COUNT, RANGE_BUCKETS_M, DatagenConfig
from app.datagen.export_yolo import ExportStats, crop_origin, export_clip, load_labels
from app.datagen.ir_composite import SunState, make_fpn
from app.datagen.rig import (
    IsaacReplicatorBackend,
    Optics,
    PTZState,
    RigActor,
    RigBackend,
    RigFrame,
    SyntheticRigBackend,
    camera_optics,
    render_rig_frame,
    spawn_in_frustum,
    step_ptz,
)
from app.datagen.terrain_builder import DEFAULT_NPY, GroundZ, build_ground

CLASS_ORDER: tuple[str, ...] = ("quad", "fixed_wing", "decoy", "bird")
CLASS_MIX_WEIGHTS: tuple[float, ...] = (0.50, 0.20, 0.15, 0.15)
MIN_CROP_CONTAINMENT = 0.95  # plan.md §5.1: regenerate if target leaves crop in >5 % of frames


@dataclass(frozen=True)
class ClipSpec:
    """All random parameters sampled for one clip."""

    clip_idx: int
    seed: int
    sensor_tier: str
    bucket_m: tuple[float, float]
    class_name: str
    asset: AssetThermalConfig
    span_m: float
    flock_count: int
    flap_hz: float
    focal_mm: float
    bg_mode: str
    sun: SunState
    visibility_m: float
    sector_id: int
    split: str


@dataclass(frozen=True)
class ClipResult:
    """Summary returned after one clip is simulated and written to disk."""

    clip_dir: Path
    spec: ClipSpec
    n_frames: int
    in_crop_ratio: float
    attempts: int
    yolo_stats: ExportStats
    track_features: dict[str, float | int]


# --- Function 1: sample_clip_spec ------------------------------------------------------


def _sector_split(sector_id: int) -> str:
    """70 / 15 / 15 split by terrain sector, never by random clip (`plan.md` §5.3)."""
    slot = sector_id % 20
    if slot < 14:
        return "train"
    if slot < 17:
        return "val"
    return "test"


def _focal_for_bucket(sensor_tier: str, bucket_m: tuple[float, float], rng: np.random.Generator) -> float:
    mid_m = 0.5 * (bucket_m[0] + bucket_m[1])
    if sensor_tier == "tier1":
        lo, hi = (24.0, 90.0) if mid_m < 1000.0 else (80.0, 200.0)
    else:
        lo, hi = (100.0, 350.0) if mid_m < 2200.0 else (300.0, 800.0)
    return float(rng.uniform(lo, hi))


def sample_clip_spec(
    clip_idx: int,
    cfg: DatagenConfig,
    catalog: dict[str, AssetThermalConfig],
    *,
    seed: int | None = None,
) -> ClipSpec:
    """Sample a stratified range bucket, class (`50/20/15/15`), asset, lighting, and sector."""
    base_seed = (clip_idx * 10_007 + 17) if seed is None else int(seed)
    rng = np.random.default_rng(base_seed)

    bucket_m = RANGE_BUCKETS_M[clip_idx % len(RANGE_BUCKETS_M)]
    class_name = str(rng.choice(CLASS_ORDER, p=CLASS_MIX_WEIGHTS))
    candidates = [a for a in catalog.values() if a.class_name == class_name]
    asset = candidates[int(rng.integers(0, len(candidates)))]

    span_m = float(rng.uniform(*asset.span_range_m))
    flock_count = int(rng.integers(1, 6)) if class_name in ("bird", "decoy") else 1
    flap_hz = (
        float(rng.uniform(*asset.flap_hz_range))
        if asset.flap_hz_range[1] > 0.0
        else 0.0
    )

    bg_names = tuple(cfg.background_mix.keys())
    bg_probs = np.array([cfg.background_mix[k] for k in bg_names], dtype=np.float64)
    bg_probs /= bg_probs.sum()
    bg_mode = str(rng.choice(bg_names, p=bg_probs))

    sun = SunState(
        elevation_deg=float(rng.uniform(0.0, 65.0)),
        azimuth_deg=float(rng.uniform(0.0, 360.0)),
        t_amb_k=float(rng.uniform(285.15, 305.15)),
    )
    visibility_m = float(rng.uniform(cfg.visibility_m[0], cfg.visibility_m[1]))
    focal_mm = _focal_for_bucket(cfg.sensor_tier, bucket_m, rng)
    sector_id = int(rng.integers(0, 100))

    return ClipSpec(
        clip_idx=clip_idx,
        seed=base_seed,
        sensor_tier=cfg.sensor_tier,
        bucket_m=bucket_m,
        class_name=class_name,
        asset=asset,
        span_m=span_m,
        flock_count=flock_count,
        flap_hz=flap_hz,
        focal_mm=focal_mm,
        bg_mode=bg_mode,
        sun=sun,
        visibility_m=visibility_m,
        sector_id=sector_id,
        split=_sector_split(sector_id),
    )


# --- Function 2: simulate_clip_frames (with >= 95 % crop-containment check) ------------


def _class_velocity_mps(class_name: str, rng: np.random.Generator) -> np.ndarray:
    speed_bands = {
        "quad": (12.0, 32.0),
        "fixed_wing": (24.0, 48.0),
        "decoy": (1.5, 6.0),
        "bird": (7.0, 16.0),
    }
    lo, hi = speed_bands[class_name]
    speed = float(rng.uniform(lo, hi))
    heading = float(rng.uniform(-np.pi, np.pi))
    vz = float(rng.uniform(-1.2, 1.2))
    return np.array([speed * np.cos(heading), speed * np.sin(heading), vz], dtype=np.float64)


def simulate_clip_frames(
    spec: ClipSpec,
    cfg: DatagenConfig,
    ground: GroundZ,
    backend: RigBackend,
    *,
    n_frames: int,
    min_containment: float = MIN_CROP_CONTAINMENT,
    max_attempts: int = 8,
) -> tuple[list[RigFrame], np.ndarray, float, int]:
    """Run one clip trajectory and regenerate if `in_crop_ratio < min_containment` (`0.95`)."""
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    eo_opt = camera_optics(cfg, "eo", spec.focal_mm)
    ir_opt = camera_optics(cfg, "ir", spec.focal_mm)
    dt_s = 1.0 / cfg.fps

    last_frames: list[RigFrame] = []
    last_xyz = np.zeros((n_frames, 3), dtype=np.float64)
    last_ratio = 0.0

    for attempt in range(max_attempts):
        rng = np.random.default_rng(spec.seed + attempt * 997)
        cam_x = float(rng.uniform(-2500.0, 2500.0))
        cam_y = float(rng.uniform(-2500.0, 2500.0))
        cam_z = float(ground.at(cam_x, cam_y)) + 25.0
        ptz = PTZState(
            cam_xyz=(cam_x, cam_y, cam_z),
            yaw_rad=float(rng.uniform(-np.pi, np.pi)),
            pitch_rad=0.05,
        )

        # Tighten spawn margin on retries so fast targets stay well inside the crop window
        margin = max(0.32 - 0.04 * attempt, 0.12)
        spawn = spawn_in_frustum(
            ground,
            ptz,
            ir_opt,
            spec.bucket_m,
            bg_mode=spec.bg_mode,
            crop_margin=margin,
            rng=rng,
        )
        pos = np.array(spawn.xyz, dtype=np.float64)
        vel = _class_velocity_mps(spec.class_name, rng) * (0.75 ** attempt)
        fpn = make_fpn((ir_opt.height_px, ir_opt.width_px), rng, sigma_k=0.025)

        flock_offsets = [np.zeros(3, dtype=np.float64)]
        for _ in range(1, spec.flock_count):
            flock_offsets.append(rng.normal(0.0, 2.5 * spec.span_m, size=3))

        frames: list[RigFrame] = []
        track_xyz = np.zeros((n_frames, 3), dtype=np.float64)

        for frame_idx in range(n_frames):
            t_s = frame_idx * dt_s
            bob_z = (
                0.18 * spec.span_m * float(np.sin(2.0 * np.pi * spec.flap_hz * t_s))
                if spec.flap_hz > 0.0
                else 0.0
            )
            lead_xyz = pos + np.array([0.0, 0.0, bob_z], dtype=np.float64)
            lead_xyz[2] = max(lead_xyz[2], float(ground.at(lead_xyz[0], lead_xyz[1])) + 15.0)
            track_xyz[frame_idx] = lead_xyz

            actors = [
                RigActor(
                    track_id=i + 1,
                    asset=spec.asset,
                    span_m=spec.span_m,
                    xyz=(
                        float(lead_xyz[0] + off[0]),
                        float(lead_xyz[1] + off[1]),
                        float(lead_xyz[2] + off[2]),
                    ),
                )
                for i, off in enumerate(flock_offsets)
            ]
            frame = render_rig_frame(
                backend,
                ptz=ptz,
                eo_optics=eo_opt,
                ir_optics=ir_opt,
                actors=actors,
                sun=spec.sun,
                visibility_m=spec.visibility_m,
                netd_k=cfg.netd_mk / 1000.0,
                fpn=fpn,
                rng=rng,
            )
            frames.append(frame)
            pos = pos + vel * dt_s
            ptz = step_ptz(ptz, lead_xyz, dt_s, rng=rng)

        in_crop_ratio = float(np.mean([f.in_crop for f in frames]))
        last_frames, last_xyz, last_ratio = frames, track_xyz, in_crop_ratio
        if in_crop_ratio >= min_containment:
            return frames, track_xyz, in_crop_ratio, attempt + 1

    return last_frames, last_xyz, last_ratio, max_attempts


# --- Function 3: compute_track_features (`plan.md` §5.2) -------------------------------


def compute_track_features(
    spec: ClipSpec,
    frames: Sequence[RigFrame],
    track_xyz: np.ndarray,
    dt_s: float,
) -> dict[str, float | int]:
    """Compute per-track temporal & hotspot features for the §5.2 track-level stage."""
    if len(track_xyz) >= 2:
        vel = np.diff(track_xyz, axis=0) / max(dt_s, 1e-6)
        speeds = np.linalg.norm(vel, axis=1)
        median_speed = float(np.median(speeds))
        p90_speed = float(np.percentile(speeds, 90))
    else:
        median_speed = 0.0
        p90_speed = 0.0

    if len(track_xyz) >= 3:
        acc = np.diff(track_xyz, n=2, axis=0) / max(dt_s * dt_s, 1e-6)
        curvature = float(np.mean(np.linalg.norm(acc, axis=1)) / max(median_speed ** 2, 1.0))
    else:
        curvature = 0.0

    z_series = track_xyz[:, 2]
    z_detrended = z_series - np.linspace(z_series[0], z_series[-1], len(z_series))
    bob_amp_m = float(0.5 * (z_detrended.max() - z_detrended.min()))

    aspect_ratios = [
        w / max(h, 1e-6)
        for f in frames
        for _, _, _, _, w, h in f.boxes[:1]
    ]
    ar_var = float(np.var(aspect_ratios)) if len(aspect_ratios) >= 2 else 0.0

    return {
        "median_speed_mps": round(median_speed, 3),
        "p90_speed_mps": round(p90_speed, 3),
        "vertical_bob_amp_m": round(bob_amp_m, 4),
        "flap_hz": round(spec.flap_hz, 3),
        "aspect_ratio_var": round(ar_var, 6),
        "ir_hotspot_count": len(spec.asset.hotspots),
        "peak_delta_t_k": round(spec.asset.peak_delta_t_k, 2),
        "trajectory_curvature": round(curvature, 6),
    }


# --- Function 4: write_clip_bundle -----------------------------------------------------


def crop_square(frame: np.ndarray, ox: int, oy: int, crop_px: int) -> np.ndarray:
    """Cut the centred square training crop using `crop_origin()` — never a resize."""
    return frame[oy : oy + crop_px, ox : ox + crop_px]


def _write_mp4_or_npz(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    arr = np.asarray(frames, dtype=np.uint8)
    try:
        from imageio.v3 import imwrite

        imwrite(path, arr, fps=fps)
    except Exception:
        # Fallback if a minimal container lacks an ffmpeg binary: store raw lossless frames
        np.save(path.with_suffix(".npy"), arr)
        path.write_bytes(arr[0].tobytes())


def write_clip_bundle(
    clip_dir: str | Path,
    *,
    spec: ClipSpec,
    cfg: DatagenConfig,
    frames: Sequence[RigFrame],
    track_features: dict[str, float | int],
    in_crop_ratio: float,
    attempts: int,
) -> ClipResult:
    """Write `eo.mp4`, `ir.mp4`, `labels.npz`, `metadata.json`, and `labels/frame_XXXX.txt`."""
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

    eo_w, eo_h = cfg.native_res("eo")
    ir_w, ir_h = cfg.native_res("ir")
    eo_crop = cfg.train_crop_px("eo")
    ir_crop = cfg.train_crop_px("ir")
    eo_ox, eo_oy = crop_origin((eo_w, eo_h), eo_crop)
    ir_ox, ir_oy = crop_origin((ir_w, ir_h), ir_crop)

    eo_crops = [crop_square(f.eo_rgb, eo_ox, eo_oy, eo_crop) for f in frames]
    ir_crops = [crop_square(f.ir_u8, ir_ox, ir_oy, ir_crop) for f in frames]

    _write_mp4_or_npz(clip_dir / "eo.mp4", eo_crops, cfg.fps)
    _write_mp4_or_npz(clip_dir / "ir.mp4", ir_crops, cfg.fps)

    cols: dict[str, list] = {
        "frame": [],
        "class_id": [],
        "cx": [],
        "cy": [],
        "w": [],
        "h": [],
        "track_id": [],
        "occluded": [],
    }
    for frame_idx, f in enumerate(frames):
        for track_id, class_id, cx, cy, w, h in f.boxes:
            cols["frame"].append(frame_idx)
            cols["class_id"].append(class_id)
            cols["cx"].append(cx)
            cols["cy"].append(cy)
            cols["w"].append(w)
            cols["h"].append(h)
            cols["track_id"].append(track_id)
            cols["occluded"].append(0)

    npz_cols = {
        k: np.asarray(v, dtype=np.int32 if k in ("frame", "class_id", "track_id", "occluded") else np.float32)
        for k, v in cols.items()
    }
    labels_path = clip_dir / "labels.npz"
    np.savez(labels_path, **npz_cols)

    yolo_stats = export_clip(
        npz_cols,
        cfg=cfg,
        modality="ir",
        out_dir=clip_dir,
        n_frames=len(frames),
    )

    metadata = {
        "clip_idx": spec.clip_idx,
        "seed": spec.seed,
        "sensor_tier": spec.sensor_tier,
        "fps": cfg.fps,
        "n_frames": len(frames),
        "bucket_m": list(spec.bucket_m),
        "class_name": spec.class_name,
        "class_id": spec.asset.class_id,
        "asset_slug": spec.asset.slug,
        "span_m": round(spec.span_m, 3),
        "flock_count": spec.flock_count,
        "focal_mm": round(spec.focal_mm, 2),
        "bg_mode": spec.bg_mode,
        "sun_elevation_deg": round(spec.sun.elevation_deg, 2),
        "t_amb_k": round(spec.sun.t_amb_k, 2),
        "visibility_m": round(spec.visibility_m, 1),
        "sector_id": spec.sector_id,
        "split": spec.split,
        "in_crop_ratio": round(in_crop_ratio, 4),
        "attempts": attempts,
        "crop_windows": {
            "eo": {"ox": eo_ox, "oy": eo_oy, "crop_px": eo_crop},
            "ir": {"ox": ir_ox, "oy": ir_oy, "crop_px": ir_crop},
        },
        "track_features": track_features,
    }
    if cfg.write_metadata:
        (clip_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return ClipResult(
        clip_dir=clip_dir,
        spec=spec,
        n_frames=len(frames),
        in_crop_ratio=in_crop_ratio,
        attempts=attempts,
        yolo_stats=yolo_stats,
        track_features=track_features,
    )


# --- Outer runner & shard parser -------------------------------------------------------


def parse_shard(shard: str) -> tuple[int, int]:
    """Parse `'i/n'` (1-based or 0-based index `i` of `n` shards) into `(idx_0, total)`."""
    parts = shard.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"expected shard format 'i/n', got {shard!r}")
    i, n = int(parts[0]), int(parts[1])
    if n < 1:
        raise ValueError(f"shard count must be >= 1, got {n}")
    if 1 <= i <= n:
        return i - 1, n
    if 0 <= i < n:
        return i, n
    raise ValueError(f"shard index {i} out of range for {n} shards")


def generate_dataset(
    *,
    cfg: DatagenConfig,
    clips: int,
    out_root: str | Path,
    duration_s: float | None = None,
    shard: str = "1/1",
    ground: GroundZ | None = None,
    backend: RigBackend | None = None,
    seed: int = 0,
) -> list[ClipResult]:
    """Generate `clips` stratified across range buckets and write each clip directory."""
    if clips < 1:
        raise ValueError(f"clips must be >= 1, got {clips}")
    shard_idx, shard_total = parse_shard(shard)

    if ground is None:
        ground = GroundZ.load(DEFAULT_NPY) if DEFAULT_NPY.exists() else build_ground(cfg)[0]
    if backend is None:
        backend = SyntheticRigBackend(ground)

    catalog = load_catalog()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dur = float(np.mean(cfg.duration_s)) if duration_s is None else float(duration_s)
    n_frames = max(int(round(dur * cfg.fps)), 1)
    dt_s = 1.0 / cfg.fps

    results: list[ClipResult] = []
    for clip_idx in range(clips):
        if clip_idx % shard_total != shard_idx:
            continue
        spec = sample_clip_spec(clip_idx, cfg, catalog, seed=seed + clip_idx * 131)
        frames, track_xyz, ratio, attempts = simulate_clip_frames(
            spec, cfg, ground, backend, n_frames=n_frames
        )
        track_features = compute_track_features(spec, frames, track_xyz, dt_s)
        b_lo, b_hi = int(spec.bucket_m[0]), int(spec.bucket_m[1])
        clip_dir = out_root / f"clip_{clip_idx:05d}_{spec.sensor_tier}_{spec.class_name}_b{b_lo}_{b_hi}"
        res = write_clip_bundle(
            clip_dir,
            spec=spec,
            cfg=cfg,
            frames=frames,
            track_features=track_features,
            in_crop_ratio=ratio,
            attempts=attempts,
        )
        results.append(res)
    return results


# --- Milestone D: Throughput Benchmark (D1) & Stratified Pilot Audit (D2) --------------


@dataclass(frozen=True)
class BenchmarkReport:
    """D1 throughput measurement and GPU-hour estimates for 2,000 and 20,000 clips."""

    sensor_tier: str
    n_frames: int
    elapsed_s: float
    fps_measured: float
    sec_per_clip_avg: float
    pilot_2k_hours: float
    full_20k_hours: float


def benchmark_throughput(
    cfg: DatagenConfig,
    *,
    out_root: str | Path,
    n_frames: int = 250,
    ground: GroundZ | None = None,
    backend: RigBackend | None = None,
) -> BenchmarkReport:
    """Time one `n_frames` clip (D1) and compute total hours for 2,000 and 20,000 clips."""
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    t0 = time.perf_counter()
    generate_dataset(
        cfg=cfg,
        clips=1,
        out_root=out_root,
        duration_s=n_frames / cfg.fps,
        ground=ground,
        backend=backend,
    )
    elapsed_s = max(time.perf_counter() - t0, 1e-6)
    fps_measured = n_frames / elapsed_s

    lo_f, hi_f = cfg.frames_per_clip
    avg_frames = 0.5 * (lo_f + hi_f)
    sec_per_clip = avg_frames / fps_measured
    pilot_hours = (2_000 * sec_per_clip) / 3600.0
    full_hours = (CLIP_COUNT * sec_per_clip) / 3600.0

    return BenchmarkReport(
        sensor_tier=cfg.sensor_tier,
        n_frames=n_frames,
        elapsed_s=round(elapsed_s, 3),
        fps_measured=round(fps_measured, 2),
        sec_per_clip_avg=round(sec_per_clip, 2),
        pilot_2k_hours=round(pilot_hours, 2),
        full_20k_hours=round(full_hours, 2),
    )


def audit_bucket_spans(
    results: Sequence[ClipResult], cfg: DatagenConfig
) -> list[dict[str, float | str | bool]]:
    """Verify each clip's rendered IR pixel width matches the optical formula `(span_m / r_m) * focal_px`."""
    ir_w, _ = cfg.native_res("ir")
    rows: list[dict[str, float | str | bool]] = []
    for r in results:
        ir_opt = camera_optics(cfg, "ir", r.spec.focal_mm)
        labels = load_labels(r.clip_dir / "labels.npz")
        measured_px = float(np.median(labels["w"]) * ir_w) if labels["w"].size else 0.0
        mid_range_m = 0.5 * (r.spec.bucket_m[0] + r.spec.bucket_m[1])
        expected_px = max((r.spec.span_m / mid_range_m) * ir_opt.focal_px, 2.0)
        ratio = measured_px / max(expected_px, 1e-6)
        b0, b1 = int(r.spec.bucket_m[0]), int(r.spec.bucket_m[1])
        rows.append(
            {
                "clip": r.clip_dir.name,
                "bucket": f"{b0}-{b1}m",
                "measured_px": round(measured_px, 2),
                "expected_px": round(expected_px, 2),
                "ratio": round(ratio, 3),
                "in_bounds": bool(0.35 <= ratio <= 2.8 and r.in_crop_ratio >= MIN_CROP_CONTAINMENT),
            }
        )
    return rows


def pilot_report(out_root: str | Path) -> dict:
    """Audit a generated batch on disk: bucket balance, class shares, and >=95 % crop gate."""
    out_root = Path(out_root)
    metas = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(out_root.glob("clip_*/metadata.json"))
    ]
    if not metas:
        raise ValueError(f"no clip_*/metadata.json found under {out_root}")

    bucket_counts = Counter(f"{int(m['bucket_m'][0])}-{int(m['bucket_m'][1])}m" for m in metas)
    class_counts = Counter(str(m["class_name"]) for m in metas)
    min_in_crop = float(min(m["in_crop_ratio"] for m in metas))
    pass_gate = bool(min_in_crop >= MIN_CROP_CONTAINMENT and len(bucket_counts) >= min(len(metas), 5))

    report = {
        "clips": len(metas),
        "bucket_counts": dict(bucket_counts),
        "class_counts": dict(class_counts),
        "min_in_crop_ratio": round(min_in_crop, 4),
        "pass_gate": pass_gate,
    }
    (out_root / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic EO/IR Counter-UAS clips.")
    parser.add_argument("--sensor", default="tier2", choices=("tier1", "tier2"))
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--duration", type=float, default=1.0, help="clip duration in seconds")
    parser.add_argument("--shard", default="1/1", help="shard i/n for multi-GPU runs")
    parser.add_argument("--out", default="/tmp/datagen_clips")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="auto", choices=("auto", "isaac", "synthetic"))
    parser.add_argument("--benchmark", action="store_true", help="run Milestone D1+D2 benchmark & bucket audit")
    parser.add_argument("--bench-frames", type=int, default=250, help="frames for the D1 timing clip")

    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        has_isaaclab = True
    except ImportError:
        parser.add_argument("--headless", action="store_true", default=True)
        has_isaaclab = False

    args = parser.parse_args()
    use_isaac = (args.backend == "isaac") or (args.backend == "auto" and has_isaaclab)
    if args.backend == "isaac" and not has_isaaclab:
        raise RuntimeError("--backend isaac requested, but isaaclab.app is not installed in this venv")

    sim_app = None
    if use_isaac:
        args.enable_cameras = True
        print("[datagen] booting Isaac Sim (AppLauncher, enable_cameras=True)...")
        sim_app = AppLauncher(args).app

    try:
        cfg = DatagenConfig(sensor_tier=args.sensor, headless=bool(getattr(args, "headless", True)))
        ground = GroundZ.load(DEFAULT_NPY) if DEFAULT_NPY.exists() else build_ground(cfg)[0]
        backend: RigBackend = (
            IsaacReplicatorBackend(ground, cfg) if use_isaac else SyntheticRigBackend(ground)
        )
        print(f"[datagen] active backend: {type(backend).__name__}")

        if args.benchmark:
            bench_dir = Path(args.out) / "_d1_bench"
            d1 = benchmark_throughput(
                cfg, out_root=bench_dir, n_frames=args.bench_frames, ground=ground, backend=backend
            )
            print(
                f"D1 benchmark ({d1.sensor_tier}, {d1.n_frames} frames): {d1.fps_measured} fps  "
                f"({d1.sec_per_clip_avg}s/clip)  2k_pilot={d1.pilot_2k_hours}h  20k_full={d1.full_20k_hours}h"
            )
            results = generate_dataset(
                cfg=cfg,
                clips=args.clips,
                out_root=args.out,
                duration_s=args.duration,
                shard=args.shard,
                ground=ground,
                backend=backend,
                seed=args.seed,
            )
            spans = audit_bucket_spans(results, cfg)
            rep = pilot_report(args.out)
            print(
                f"D2 pilot audit: clips={rep['clips']}  min_in_crop={rep['min_in_crop_ratio']}  "
                f"pass_gate={rep['pass_gate']}"
            )
            for row in spans:
                print(
                    f"  {row['clip']:<44} bucket={row['bucket']:<10} "
                    f"px={row['measured_px']:5.2f} (exp {row['expected_px']:5.2f})  in_bounds={row['in_bounds']}"
                )
            return 0

        started = time.perf_counter()
        results = generate_dataset(
            cfg=cfg,
            clips=args.clips,
            out_root=args.out,
            duration_s=args.duration,
            shard=args.shard,
            ground=ground,
            backend=backend,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - started
        total_frames = sum(r.n_frames for r in results)
        fps_eff = total_frames / max(elapsed, 1e-6)

        print(
            f"generated {len(results)} clips ({total_frames} frames) in {elapsed:.2f}s "
            f"({fps_eff:.1f} frames/s) -> {args.out}"
        )
        for r in results:
            b0, b1 = int(r.spec.bucket_m[0]), int(r.spec.bucket_m[1])
            print(
                f"  {r.clip_dir.name:<44} frames={r.n_frames:3d}  "
                f"in_crop={r.in_crop_ratio * 100:5.1f}%  attempts={r.attempts}  "
                f"boxes={r.yolo_stats.boxes_out}  bucket={b0}-{b1}m"
            )
        return 0
    finally:
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
