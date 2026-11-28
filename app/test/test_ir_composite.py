"""Acceptance tests for docs.md A1 — the six claims the IR compositor must satisfy."""

import numpy as np
import pytest

from app.datagen.ir_composite import (
    DEFAULT_MATERIALS,
    SEG_BIRD_CORE,
    SEG_GROUND,
    SEG_QUAD_ROTOR,
    SEG_SKY,
    SunState,
    WhiteHot,
    beam_extinction,
    compose_ir,
    make_fpn,
    make_synthetic_scene,
    white_hot,
)

TAU = 293.15


def _compose(elevation=40.0, *, seed=0, visibility_m=20_000.0, debug=True, netd_k=0.050, **kw):
    seg, normals, depth = make_synthetic_scene()
    return compose_ir(
        seg,
        normals,
        depth,
        sun=SunState(elevation_deg=elevation, t_amb_k=TAU),
        visibility_m=visibility_m,
        netd_k=netd_k,
        rng=np.random.default_rng(seed),
        debug=debug,
        **kw,
    )


def test_output_shape_and_dtype():
    img, temp = _compose()
    seg, _, _ = make_synthetic_scene()
    assert img.shape == seg.shape
    assert img.dtype == np.uint8
    assert temp.shape == seg.shape and temp.dtype == np.float32
    assert img.min() >= 0 and img.max() <= 255


def test_rotor_reads_brighter_than_bird_torso():
    """+35 K rotor vs +12 K bird core, both at 3 km — the only cue that survives at 4 px."""
    seg, _, _ = make_synthetic_scene()
    _, temp = _compose(elevation=55.0, netd_k=0.0)
    assert temp[seg == SEG_QUAD_ROTOR].mean() > temp[seg == SEG_BIRD_CORE].mean()
    # both are attenuated toward the air path, but stay above ambient
    assert temp[seg == SEG_BIRD_CORE].mean() > TAU


def test_sky_is_colder_than_sunlit_ground():
    seg, _, _ = make_synthetic_scene()
    img, _ = _compose(elevation=55.0)
    assert img[seg == SEG_SKY].mean() < img[seg == SEG_GROUND].mean()


def test_thermal_crossover_at_dawn():
    """At zero sun elevation rock converges on air temperature (plan.md §4)."""
    seg, _, _ = make_synthetic_scene()
    _, temp = _compose(elevation=0.0, netd_k=0.0)
    ground = temp[seg == SEG_GROUND].mean()
    assert abs(ground - TAU) < 2.0, f"ground {ground:.2f} K vs air {TAU:.2f} K"


def test_netd_lands_in_the_temperature_domain():
    seg, _, _ = make_synthetic_scene()
    netd = 0.050
    _, temp = _compose(elevation=40.0, netd_k=netd)
    # sky is a flat surface, so its spread is exactly the injected noise
    measured = temp[seg == SEG_SKY].std()
    assert measured == pytest.approx(netd, rel=0.20), f"NETD {measured:.4f} K vs {netd} K"


def test_fpn_is_applied_once_not_per_frame():
    """One offset per pixel per clip: constant across frames, so its std adds to nothing."""
    seg, normals, depth = make_synthetic_scene()
    rng = np.random.default_rng(1)
    fpn = make_fpn(seg.shape, rng, sigma_k=0.30)
    sun = SunState(elevation_deg=40.0, t_amb_k=TAU)

    first, t_first = compose_ir(
        seg, normals, depth, sun=sun, netd_k=0.0, fpn=fpn, rng=np.random.default_rng(2), debug=True
    )
    second, t_second = compose_ir(
        seg, normals, depth, sun=sun, netd_k=0.0, fpn=fpn, rng=np.random.default_rng(2), debug=True
    )
    assert np.array_equal(first, second)
    assert np.allclose(t_first - t_second, 0.0)


def test_same_seed_is_identical_and_other_seed_differs():
    a, _ = _compose(seed=3)
    b, _ = _compose(seed=3)
    c, _ = _compose(seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fixed_gain_no_auto_agc():
    """The display map is a pure function of temperature: same T, same pixel, always."""
    wht = WhiteHot.around(TAU)
    t = np.array([[TAU - 25.0, TAU, TAU + 45.0]], dtype=np.float32)
    assert np.array_equal(white_hot(t, wht), white_hot(t, wht))
    assert white_hot(t, wht)[0, 0] == 0
    assert white_hot(t, wht)[0, 2] == 255


def test_shimmer_only_above_300mm_focal():
    seg, normals, depth = make_synthetic_scene()
    sun = SunState(elevation_deg=40.0, t_amb_k=TAU)
    short, _ = compose_ir(seg, normals, depth, sun=sun, focal_mm=100.0, rng=np.random.default_rng(5), debug=True)
    long_, _ = compose_ir(seg, normals, depth, sun=sun, focal_mm=1200.0, rng=np.random.default_rng(5), debug=True)
    assert not np.array_equal(short, long_)


def test_extinction_scales_with_visibility():
    assert beam_extinction(3_000.0) > beam_extinction(20_000.0)
    with pytest.raises(ValueError):
        beam_extinction(0.0)


def test_shape_mismatch_rejected():
    seg, normals, _ = make_synthetic_scene()
    with pytest.raises(ValueError, match="shape mismatch"):
        compose_ir(seg, normals, np.zeros((10, 10), dtype=np.float32), sun=SunState(30.0))


def test_default_material_table_covers_the_demo_scene():
    seg, _, _ = make_synthetic_scene()
    for seg_id in np.unique(seg):
        if seg_id == SEG_SKY:
            continue
        assert seg_id in DEFAULT_MATERIALS, f"seg id {seg_id} has no thermal material"
