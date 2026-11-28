"""Acceptance tests for docs.md A2 — crop maths before Ultralytics ever sees a label."""

import numpy as np
import pytest

from app.datagen.config import DatagenConfig
from app.datagen.export_yolo import (
    DROP_OUTSIDE,
    DROP_TOO_SMALL,
    CropWindow,
    _remap_with_reason,
    crop_origin,
    export_clip,
    native_pixels,
    remap_box,
)

# (native W, H, crop) -> expected top-left corner. Every margin here is even by construction.
ORIGINS = [
    ((1920, 1080), 1024, (448, 28)),
    ((640, 512), 512, (64, 0)),
    ((2560, 1440), 1024, (768, 208)),
    ((1280, 1024), 1024, (128, 0)),
]

CENTRE_CASES = [
    ("tier1", "eo"),
    ("tier1", "ir"),
    ("tier2", "eo"),
    ("tier2", "ir"),
]


def _window(native_wh=(1920, 1080), crop_px=1024) -> CropWindow:
    return CropWindow.of(native_wh, crop_px)


def _cfg(tier: str = "tier2") -> DatagenConfig:
    return DatagenConfig(sensor_tier=tier)


def _labels(rows, fields=("frame", "class_id", "cx", "cy", "w", "h", "track_id", "occluded")):
    columns = {name: [] for name in fields}
    for row in rows:
        for name, value in zip(fields, row):
            columns[name].append(value)
    return {name: np.asarray(values) for name, values in columns.items()}


@pytest.mark.parametrize("native_wh,crop_px,expected", ORIGINS)
def test_crop_origin_matches_the_pinned_table(native_wh, crop_px, expected):
    assert crop_origin(native_wh, crop_px) == expected


def test_crop_origin_refuses_an_upscale():
    with pytest.raises(ValueError, match="upscale"):
        crop_origin((640, 512), 1024)


def test_crop_origin_refuses_a_half_pixel_margin():
    with pytest.raises(ValueError, match="half pixel"):
        crop_origin((1921, 1080), 1024)


def test_crop_origin_rejects_non_positive_crop():
    with pytest.raises(ValueError, match="positive"):
        crop_origin((1920, 1080), 0)


@pytest.mark.parametrize("tier,modality", CENTRE_CASES)
def test_centre_box_stays_at_the_centre(tier, modality):
    """(0.5, 0.5) is the fixed point of a centred crop, for every sensor."""
    cfg = _cfg(tier)
    window = CropWindow.of(cfg.native_res(modality), cfg.train_crop_px(modality))
    box = remap_box(0.5, 0.5, 0.1, 0.1, window=window)
    assert box is not None
    assert box[0] == pytest.approx(0.5, abs=1e-12)
    assert box[1] == pytest.approx(0.5, abs=1e-12)
    # size is not invariant: the ruler shrank from W to crop_px, so the fraction grows.
    assert box[2] == pytest.approx(0.1 * window.width / window.crop_px)
    assert box[3] == pytest.approx(0.1 * window.height / window.crop_px)


def test_round_trip_returns_the_original_box():
    window = _window()
    src = (0.42, 0.61, 0.03, 0.02)
    cx, cy, w, h = remap_box(*src, window=window)
    back = (
        (cx * window.crop_px + window.ox) / window.width,
        (cy * window.crop_px + window.oy) / window.height,
        w * window.crop_px / window.width,
        h * window.crop_px / window.height,
    )
    assert back == pytest.approx(src, abs=1e-12)


def test_box_cut_by_the_crop_edge_is_recentred_on_what_survives():
    """Tier 1 EO: ox = 448. The box starts 25.6 px left of the crop and loses that sliver."""
    window = _window((1920, 1080), 1024)
    box = remap_box(0.240, 0.510, 0.040, 0.038, window=window)
    assert box is not None
    assert box[0] == pytest.approx(0.025, abs=1e-12)  # (473.6 - 448) / 1024
    assert box[2] == pytest.approx(0.050, abs=1e-12)  # 51.2 / 1024


def test_centre_outside_the_crop_is_dropped():
    window = _window((1280, 1024), 1024)  # ox = 128 -> native cx 0.05 is outside
    box, reason = _remap_with_reason(0.050, 0.5, 0.01, 0.01, window, 2.0)
    assert box is None
    assert reason == DROP_OUTSIDE


def test_centre_exactly_on_the_far_edge_is_outside():
    window = _window((1280, 1024), 1024)
    edge = (window.ox + window.crop_px) / window.width
    box, reason = _remap_with_reason(edge, 0.5, 0.01, 0.01, window, 2.0)
    assert box is None
    assert reason == DROP_OUTSIDE


def test_box_under_two_pixels_is_dropped():
    window = _window((1280, 1024), 1024)
    box, reason = _remap_with_reason(0.4, 0.5, 0.0012, 0.0012, window, 2.0)
    assert box is None
    assert reason == DROP_TOO_SMALL


def test_zero_size_and_nan_are_invalid():
    window = _window()
    assert _remap_with_reason(0.5, 0.5, 0.0, 0.1, window, 2.0)[1] == "invalid"
    assert _remap_with_reason(float("nan"), 0.5, 0.1, 0.1, window, 2.0)[1] == "invalid"


def test_nothing_is_dropped_when_every_box_fits(tmp_path):
    rows = [(0, 0, 0.5, 0.5, 0.02, 0.02, 1, 0), (0, 3, 0.45, 0.55, 0.01, 0.01, 2, 0)]
    stats = export_clip(_labels(rows), cfg=_cfg(), modality="ir", out_dir=tmp_path)
    assert stats.boxes_in == 2
    assert stats.boxes_out == 2
    assert stats.dropped == 0


def test_boxes_are_written_in_yolo_text_format(tmp_path):
    rows = [(0, 2, 0.5, 0.5, 0.02, 0.02, 1, 0)]
    export_clip(_labels(rows), cfg=_cfg(), modality="ir", out_dir=tmp_path)
    text = (tmp_path / "labels" / "frame_0000.txt").read_text()
    fields = text.split()
    assert len(fields) == 5
    assert fields[0] == "2"
    assert all(len(value.split(".")[1]) == 6 for value in fields[1:])
    assert text.endswith("\n")


def test_frames_without_boxes_get_empty_files(tmp_path):
    """A missing label file and an empty one are different to Ultralytics; write the empty one."""
    rows = [(0, 0, 0.5, 0.5, 0.02, 0.02, 1, 0)]
    stats = export_clip(_labels(rows), cfg=_cfg(), modality="ir", out_dir=tmp_path, n_frames=2)
    assert stats.frames == 2
    assert (tmp_path / "labels" / "frame_0001.txt").exists()
    assert (tmp_path / "labels" / "frame_0001.txt").stat().st_size == 0


def test_dropped_boxes_are_counted_by_reason(tmp_path):
    rows = [
        (0, 0, 0.5, 0.5, 0.02, 0.02, 1, 0),  # kept
        (0, 2, 0.01, 0.5, 0.01, 0.01, 2, 0),  # outside
    ]
    stats = export_clip(_labels(rows), cfg=_cfg(), modality="ir", out_dir=tmp_path)
    assert stats.dropped_outside == 1
    assert stats.boxes_out == 1
    assert "outside" in stats.line()


def test_npz_file_and_column_mapping_agree(tmp_path):
    rows = [(0, 0, 0.5, 0.5, 0.02, 0.02, 1, 0), (1, 3, 0.52, 0.48, 0.01, 0.01, 2, 1)]
    columns = _labels(rows)

    npz_path = tmp_path / "labels.npz"
    np.savez(npz_path, **columns)

    first = export_clip(columns, cfg=_cfg(), modality="ir", out_dir=tmp_path / "a")
    second = export_clip(npz_path, cfg=_cfg(), modality="ir", out_dir=tmp_path / "b")

    assert first == second
    for name in ("frame_0000.txt", "frame_0001.txt"):
        assert (tmp_path / "a" / "labels" / name).read_text() == (
            tmp_path / "b" / "labels" / name
        ).read_text()


def test_output_is_deterministic(tmp_path):
    rows = [(0, 3, 0.52, 0.48, 0.01, 0.01, 2, 0), (0, 0, 0.50, 0.50, 0.02, 0.02, 1, 0)]
    export_clip(_labels(rows), cfg=_cfg(), modality="ir", out_dir=tmp_path / "a")
    export_clip(_labels(rows), cfg=_cfg(), modality="ir", out_dir=tmp_path / "b")
    assert (tmp_path / "a" / "labels" / "frame_0000.txt").read_text() == (
        tmp_path / "b" / "labels" / "frame_0000.txt"
    ).read_text()


def test_frame_beyond_n_frames_is_rejected(tmp_path):
    rows = [(5, 0, 0.5, 0.5, 0.02, 0.02, 1, 0)]
    with pytest.raises(ValueError, match="outside n_frames"):
        export_clip(_labels(rows), cfg=_cfg(), modality="ir", out_dir=tmp_path, n_frames=3)


def test_native_pixels_matches_the_normalised_box():
    window = _window()
    x1, y1, x2, y2 = native_pixels(0.5, 0.5, 0.1, 0.1, window)
    assert (x1, y1, x2, y2) == (864.0, 486.0, 1056.0, 594.0)  # 192 px wide, 108 px tall
