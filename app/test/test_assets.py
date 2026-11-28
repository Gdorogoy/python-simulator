"""Acceptance tests for docs.md B2 — per-model JSON +K hotspots, projection, and downloader."""

import numpy as np
import pytest

from app.datagen.assets import (
    MESH_ROOT,
    SPAN_RANGES_M,
    AssetThermalConfig,
    Hotspot3D,
    load_catalog,
    project_hotspots,
)
from app.datagen.config import CLASS_IDS
from app.datagen.ir_composite import (
    SEG_QUAD_FRAME,
    SEG_SKY,
    Hotspot2D,
    SunState,
    compose_ir,
)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(MESH_ROOT)


def test_all_15_custom_jsons_exist_and_cover_all_four_classes(catalog):
    assert len(catalog) == 15
    assert {cfg.class_name for cfg in catalog.values()} == set(CLASS_IDS.keys())
    for cfg in catalog.values():
        assert cfg.json_path(MESH_ROOT).exists()
        assert cfg.class_id == CLASS_IDS[cfg.class_name]
        lo, hi = SPAN_RANGES_M[cfg.class_name]
        assert cfg.span_range_m[0] >= lo - 1e-6
        assert cfg.span_range_m[1] <= hi + 1e-6


def test_larger_k_shines_brighter_in_ir():
    """Core invariant: a larger delta_t_k at a hotspot makes the processed IR pixel brighter."""
    seg = np.full((32, 32), SEG_SKY, dtype=np.int32)
    seg[10:22, 10:22] = SEG_QUAD_FRAME
    normals = np.zeros((32, 32, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    depth = np.full((32, 32), 450.0, dtype=np.float32)

    low_spot = Hotspot2D(16.0, 16.0, radius_px=3.0, delta_t_k=10.0, seg_id=SEG_QUAD_FRAME)
    high_spot = Hotspot2D(16.0, 16.0, radius_px=3.0, delta_t_k=45.0, seg_id=SEG_QUAD_FRAME)

    img_low, t_low = compose_ir(
        seg, normals, depth, sun=SunState(15.0), hotspots=(low_spot,), netd_k=0.0, debug=True
    )
    img_high, t_high = compose_ir(
        seg, normals, depth, sun=SunState(15.0), hotspots=(high_spot,), netd_k=0.0, debug=True
    )

    assert t_high.max() > t_low.max() + 25.0
    assert int(img_high.max()) > int(img_low.max())


def test_hotspots_never_bleed_outside_the_mesh_mask():
    """Even a wide hotspot radius only heats pixels where seg == spot.seg_id."""
    seg = np.full((64, 64), SEG_SKY, dtype=np.int32)
    seg[28:36, 28:36] = SEG_QUAD_FRAME
    normals = np.zeros((64, 64, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    depth = np.full((64, 64), 500.0, dtype=np.float32)

    spot = Hotspot2D(u_px=32.0, v_px=32.0, radius_px=25.0, delta_t_k=50.0, seg_id=SEG_QUAD_FRAME)
    _, t_no_spot = compose_ir(seg, normals, depth, sun=SunState(20.0), netd_k=0.0, debug=True)
    _, t_with_spot = compose_ir(
        seg, normals, depth, sun=SunState(20.0), hotspots=(spot,), netd_k=0.0, debug=True
    )

    sky_mask = seg == SEG_SKY
    mesh_mask = seg == SEG_QUAD_FRAME
    assert np.array_equal(t_no_spot[sky_mask], t_with_spot[sky_mask])
    assert t_with_spot[mesh_mask].max() > t_no_spot[mesh_mask].max() + 30.0


def test_class_thermal_hierarchy_matches_spec(catalog):
    """Fixed-wing exhaust (+60 K) > quad rotor (+35 K) > bird core (+12 K) > decoy (+8 K)."""
    fw = catalog["orlan_10"]
    quad = catalog["fpv_drone_5part"]
    bird = catalog["flying_seagull"]
    decoy = catalog["decoy_balloon_reflector"]

    assert fw.peak_delta_t_k > quad.peak_delta_t_k > bird.peak_delta_t_k > decoy.peak_delta_t_k
    assert len(decoy.hotspots) == 0
    assert len(catalog["vampire_babayaga"].hotspots) == 7  # 6 motors + 1 core


def test_project_hotspots_scales_with_span_and_distance(catalog):
    cfg = catalog["fpv_drone_5part"]
    near = project_hotspots(cfg, span_m=0.8, center_uv=(100.0, 100.0), depth_m=500.0, focal_px=2000.0)
    far = project_hotspots(cfg, span_m=0.8, center_uv=(100.0, 100.0), depth_m=2500.0, focal_px=2000.0)

    near_spread = max(abs(s.u_px - 100.0) for s in near)
    far_spread = max(abs(s.u_px - 100.0) for s in far)
    assert near_spread == pytest.approx(5.0 * far_spread)
    assert all(s.radius_px >= 0.75 for s in far)
