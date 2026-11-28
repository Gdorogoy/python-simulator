"""Guards the facts plan.md §0.1 pins: fps derivation, the shutter law, bucket gaps, class ids.

fix(#12) — the doc used to assert these in prose (25 vs 50 vs 30 Hz, "125–500 frames", buckets
with holes, an unmapped class_id). Prose cannot fail a test; this can.
"""

import pytest

from app.datagen.config import (
    BUCKET_WEIGHTS,
    CLASS_IDS,
    CLIP_COUNT,
    EO_AOVS,
    FPS_BY_TIER,
    IR_AOVS,
    NATIVE_RES,
    QUAD_SPAN_M,
    RANGE_BUCKETS_M,
    DatagenConfig,
)


def test_frames_per_clip_follow_tier_fps():
    assert DatagenConfig(sensor_tier="tier1").frames_per_clip == (300, 600)
    assert DatagenConfig(sensor_tier="tier2").frames_per_clip == (150, 300)


def test_buckets_contiguous_and_cover_the_band():
    assert RANGE_BUCKETS_M[0][0] == 300.0
    assert RANGE_BUCKETS_M[-1][1] == 5000.0
    for (_, hi), (lo, _) in zip(RANGE_BUCKETS_M, RANGE_BUCKETS_M[1:]):
        assert hi == lo, f"gap between {hi} m and {lo} m"


def test_bucket_weights_match_buckets_and_sum_to_one():
    assert len(BUCKET_WEIGHTS) == len(RANGE_BUCKETS_M)
    assert sum(BUCKET_WEIGHTS) == pytest.approx(1.0)


def test_clip_counts_per_bucket_are_whole_clips():
    counts = [w * CLIP_COUNT for w in BUCKET_WEIGHTS]
    assert all(c == int(c) for c in counts), f"fractional clip counts: {counts}"
    assert sum(counts) == CLIP_COUNT


def test_train_crop_fits_native_without_upscaling():
    """fix(#14): Tier 1 IR is 640×512, so a global 1024² crop silently meant upscaling."""
    for tier, modality in NATIVE_RES:
        cfg = DatagenConfig(sensor_tier=tier)
        w, h = cfg.native_res(modality)
        crop = cfg.train_crop_px(modality)
        assert crop <= min(w, h), f"{tier}/{modality}: crop {crop} > native {w}×{h}"


def test_heavy_aovs_only_on_the_ir_camera():
    """fix(#14): the expensive rasters must not be requested at EO resolution."""
    heavy = {"instance_segmentation", "normals", "distance_to_image_plane", "motion_vectors"}
    assert not heavy & set(EO_AOVS), f"heavy AOVs on EO camera: {heavy & set(EO_AOVS)}"
    assert heavy <= set(IR_AOVS)
    for aovs in (EO_AOVS, IR_AOVS):
        assert "bounding_box_2d_tight" in aovs  # labels come from both cameras


def test_class_ids_dense_and_complete():
    assert sorted(CLASS_IDS.values()) == [0, 1, 2, 3]
    assert set(CLASS_IDS) == {"quad", "fixed_wing", "decoy", "bird"}


def test_quad_span_matches_the_threat_band():
    lo, hi = QUAD_SPAN_M
    assert (lo, hi) == (0.45, 1.0)


def test_background_mix_not_shared_between_instances():
    a, b = DatagenConfig(), DatagenConfig()
    a.background_mix["sky"] = 0.9
    assert b.background_mix["sky"] == 0.4


def test_shutter_never_exceeds_frame_period():
    for tier in FPS_BY_TIER:
        cfg = DatagenConfig(sensor_tier=tier)
        assert cfg.shutter_max_s == pytest.approx(1.0 / cfg.fps)
        assert cfg.effective_shutter_s(1 / 25) == pytest.approx(cfg.shutter_max_s)


def test_shutter_inside_frame_period_is_kept():
    cfg = DatagenConfig(sensor_tier="tier1")
    assert cfg.effective_shutter_s(1 / 1000) == pytest.approx(1 / 1000)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError, match="unknown sensor_tier"):
        DatagenConfig(sensor_tier="tier3")


def test_inverted_shutter_range_rejected():
    with pytest.raises(ValueError, match="shutter_s"):
        DatagenConfig(shutter_s=(1 / 25, 1 / 1000))
