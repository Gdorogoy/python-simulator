"""Acceptance tests for docs.md Milestone C — clip runner, >=95 % crop check, and track features."""

import json

import numpy as np
import pytest

from app.datagen.assets import load_catalog
from app.datagen.config import RANGE_BUCKETS_M, DatagenConfig
from app.datagen.generate import (
    MIN_CROP_CONTAINMENT,
    audit_bucket_spans,
    benchmark_throughput,
    compute_track_features,
    generate_dataset,
    parse_shard,
    pilot_report,
    sample_clip_spec,
    simulate_clip_frames,
    write_clip_bundle,
)
from app.datagen.rig import SyntheticRigBackend
from app.datagen.terrain_builder import GroundZ


@pytest.fixture(scope="module")
def flat_ground():
    return GroundZ(z=np.full((32, 32), 100.0, dtype=np.float32), half_extent_m=5000.0)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_sample_clip_spec_stratifies_range_buckets_and_assigns_sector_split(catalog):
    cfg = DatagenConfig(sensor_tier="tier2")
    specs = [sample_clip_spec(i, cfg, catalog) for i in range(10)]
    seen_buckets = [s.bucket_m for s in specs[:5]]
    assert tuple(seen_buckets) == RANGE_BUCKETS_M
    assert all(s.split in ("train", "val", "test") for s in specs)
    assert all(s.span_m >= s.asset.span_range_m[0] for s in specs)


def test_simulate_clip_frames_enforces_95_percent_crop_containment(flat_ground, catalog):
    cfg = DatagenConfig(sensor_tier="tier1")
    spec = sample_clip_spec(0, cfg, catalog, seed=19)
    backend = SyntheticRigBackend(flat_ground)

    frames, track_xyz, ratio, attempts = simulate_clip_frames(
        spec, cfg, flat_ground, backend, n_frames=12
    )
    assert len(frames) == 12
    assert track_xyz.shape == (12, 3)
    assert ratio >= MIN_CROP_CONTAINMENT
    assert attempts >= 1


def test_compute_track_features_captures_motion_and_hotspots(flat_ground, catalog):
    cfg = DatagenConfig(sensor_tier="tier1")
    spec = sample_clip_spec(3, cfg, catalog, seed=42)
    backend = SyntheticRigBackend(flat_ground)
    frames, track_xyz, _, _ = simulate_clip_frames(spec, cfg, flat_ground, backend, n_frames=10)

    feats = compute_track_features(spec, frames, track_xyz, dt_s=1.0 / cfg.fps)
    required = {
        "median_speed_mps",
        "p90_speed_mps",
        "vertical_bob_amp_m",
        "flap_hz",
        "aspect_ratio_var",
        "ir_hotspot_count",
        "peak_delta_t_k",
        "trajectory_curvature",
    }
    assert required.issubset(feats.keys())
    assert feats["p90_speed_mps"] >= feats["median_speed_mps"] > 0.0
    assert feats["ir_hotspot_count"] == len(spec.asset.hotspots)


def test_write_clip_bundle_creates_all_deliverables_on_disk(tmp_path, flat_ground, catalog):
    cfg = DatagenConfig(sensor_tier="tier1")
    spec = sample_clip_spec(0, cfg, catalog, seed=7)
    backend = SyntheticRigBackend(flat_ground)
    frames, track_xyz, ratio, attempts = simulate_clip_frames(
        spec, cfg, flat_ground, backend, n_frames=6
    )
    feats = compute_track_features(spec, frames, track_xyz, dt_s=1.0 / cfg.fps)

    clip_dir = tmp_path / "clip_00000"
    res = write_clip_bundle(
        clip_dir,
        spec=spec,
        cfg=cfg,
        frames=frames,
        track_features=feats,
        in_crop_ratio=ratio,
        attempts=attempts,
    )

    assert (clip_dir / "eo.mp4").exists()
    assert (clip_dir / "ir.mp4").exists()
    assert (clip_dir / "labels.npz").exists()
    assert (clip_dir / "metadata.json").exists()
    assert len(list((clip_dir / "labels").glob("frame_*.txt"))) == 6
    assert res.in_crop_ratio >= MIN_CROP_CONTAINMENT

    meta = json.loads((clip_dir / "metadata.json").read_text())
    assert meta["crop_windows"]["eo"]["crop_px"] == 1024
    assert meta["crop_windows"]["ir"]["crop_px"] == 512
    assert "track_features" in meta


def test_parse_shard_and_sharded_dataset_generation(tmp_path, flat_ground):
    assert parse_shard("1/2") == (0, 2)
    assert parse_shard("2/2") == (1, 2)
    with pytest.raises(ValueError):
        parse_shard("3/2")

    cfg = DatagenConfig(sensor_tier="tier1")
    backend = SyntheticRigBackend(flat_ground)
    s1 = generate_dataset(
        cfg=cfg, clips=4, out_root=tmp_path / "s1", duration_s=0.1, shard="1/2", ground=flat_ground, backend=backend
    )
    s2 = generate_dataset(
        cfg=cfg, clips=4, out_root=tmp_path / "s2", duration_s=0.1, shard="2/2", ground=flat_ground, backend=backend
    )
    assert [r.spec.clip_idx for r in s1] == [0, 2]
    assert [r.spec.clip_idx for r in s2] == [1, 3]


def test_milestone_d_benchmark_and_pilot_audit(tmp_path, flat_ground):
    cfg = DatagenConfig(sensor_tier="tier1")
    backend = SyntheticRigBackend(flat_ground)

    d1 = benchmark_throughput(
        cfg, out_root=tmp_path / "bench", n_frames=10, ground=flat_ground, backend=backend
    )
    assert d1.fps_measured > 0.0
    assert d1.full_20k_hours == pytest.approx(10.0 * d1.pilot_2k_hours, rel=0.05)

    results = generate_dataset(
        cfg=cfg,
        clips=5,
        out_root=tmp_path / "pilot",
        duration_s=0.1,
        ground=flat_ground,
        backend=backend,
    )
    spans = audit_bucket_spans(results, cfg)
    assert len(spans) == 5
    assert all(row["in_bounds"] for row in spans)

    rep = pilot_report(tmp_path / "pilot")
    assert rep["clips"] == 5
    assert len(rep["bucket_counts"]) == 5
    assert rep["min_in_crop_ratio"] >= MIN_CROP_CONTAINMENT
    assert rep["pass_gate"] is True
    assert (tmp_path / "pilot" / "pilot_report.json").exists()
