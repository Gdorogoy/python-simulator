"""Acceptance tests for docs.md A3 — the heightmap half of the terrain builder."""

import json

import numpy as np
import pytest

from app.datagen.terrain_builder import (
    BIOME_EARTH,
    BIOME_ROCK,
    BIOME_WADI,
    GroundZ,
    _carve_wadis,
    _lattice,
    _terrace,
    build_ground,
    build_heightmap,
    classify_biomes,
    export_terrain_usd,
    hillshade,
    triangulate_grid,
)

# A tiny exact grid: z[y, x] = 3*y + x, x and y in [-1, 1] with 1 m spacing.
PATCH = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32)


@pytest.fixture(scope="module")
def full_map():
    return build_heightmap(grid_vertices=256, seed=0)


def test_shape_and_dtype(full_map):
    z, info = full_map
    assert z.shape == (256, 256)
    assert z.dtype == np.float32
    assert info["grid_vertices"] == 256


def test_relief_is_pinned_to_zero_and_850(full_map):
    z, _ = full_map
    assert z.min() == pytest.approx(0.0, abs=1.0)
    assert z.max() == pytest.approx(850.0, abs=1.0)


def test_no_nan_or_inf(full_map):
    z, _ = full_map
    assert np.all(np.isfinite(z))


def test_same_seed_is_identical_and_other_seed_differs():
    a, _ = build_heightmap(grid_vertices=96, seed=3)
    b, _ = build_heightmap(grid_vertices=96, seed=3)
    c, _ = build_heightmap(grid_vertices=96, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_default_config_builds_the_spec_grid():
    ground, info = build_ground(seed=0)
    assert ground.z.shape == (1000, 1000)
    assert info["half_extent_m"] == 5000.0
    assert ground.spacing_m == pytest.approx(10.0, abs=0.05)


def test_spacing_and_extent_follow_the_grid():
    ground = GroundZ(z=PATCH, half_extent_m=1.0)
    assert ground.spacing_m == pytest.approx(1.0)
    assert ground.extent_m == (-1.0, 1.0)


def test_at_matches_hand_computed_bilinear():
    """(0.5, -0.5) sits halfway between the four corners of the (1..2, 0..1) cell."""
    ground = GroundZ(z=PATCH, half_extent_m=1.0)
    # corners z[0,1]=1, z[0,2]=2, z[1,1]=4, z[1,2]=5 -> 1.5, 4.5 -> 3.0
    assert ground.at(0.5, -0.5) == pytest.approx(3.0)


def test_at_is_exact_on_a_plane():
    """Bilinear interpolation reproduces a linear surface exactly, at any sample point."""
    ground = GroundZ(z=PATCH, half_extent_m=1.0)
    # PATCH is z = x + 3*y + 4 with x, y in [-1, 1] (row 0 is y = -1)
    for x, y in [(-0.75, 0.25), (0.1, 0.9), (0.33, -0.66)]:
        expected = x + 3.0 * y + 4.0
        assert ground.at(x, y) == pytest.approx(expected, abs=1e-6)


def test_at_clamps_outside_the_grid():
    ground = GroundZ(z=PATCH, half_extent_m=1.0)
    assert ground.at(99.0, 0.0) == pytest.approx(ground.at(1.0, 0.0))
    assert ground.at(-99.0, -99.0) == pytest.approx(0.0)


def test_at_accepts_arrays():
    ground = GroundZ(z=PATCH, half_extent_m=1.0)
    out = ground.at(np.array([0.0, 0.5]), np.array([0.0, -0.5]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(4.0)


def test_agl_is_positive_above_and_negative_below():
    ground = GroundZ(z=np.full((3, 3), 100.0, dtype=np.float32), half_extent_m=1.0)
    assert ground.agl(0.0, 0.0, 115.0) == pytest.approx(15.0)
    assert ground.agl(0.0, 0.0, 85.0) == pytest.approx(-15.0)


def test_save_and_load_round_trip(tmp_path):
    z, _ = build_heightmap(grid_vertices=32, seed=7)
    ground = GroundZ(z=z, half_extent_m=5000.0)
    path = ground.save(tmp_path / "ground_z.npy")

    assert path.exists()
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["shape"] == [32, 32]

    loaded = GroundZ.load(path)
    assert np.array_equal(loaded.z, z)
    assert loaded.half_extent_m == 5000.0


def test_terrace_quantises_mid_slopes_only():
    z = np.arange(0.0, 100.0, dtype=np.float32).reshape(1, -1)
    mid = np.full_like(z, 30.0)  # 30 deg is inside the 20-38 band
    terraced = _terrace(z, mid, step_m=10.0)
    assert np.allclose(terraced % 10.0, 0.0), "mid-slopes should snap to 10 m steps"

    flat = np.full_like(z, 0.0)  # floor and cliff stay untouched
    assert np.array_equal(_terrace(z, flat, step_m=10.0), z)


def test_wadi_is_the_lowest_ground():
    axis = np.linspace(-5000.0, 5000.0, 200, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    flat = np.full_like(x, 500.0)

    carved, meta = _carve_wadis(
        flat, x, y, rng=np.random.default_rng(0), table=_lattice(2), half_extent_m=5000.0, count=1
    )
    assert len(meta) == 1
    assert meta[0]["depth_m"] >= 20.0
    assert carved.min() <= 500.0 - 20.0, "a wadi must cut at least 20 m down"
    assert np.median(carved) > 495.0, "most of the map must stay untouched"


def test_lattice_is_reproducible_and_in_range():
    a = _lattice(11)
    b = _lattice(11)
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_too_small_grid_is_rejected():
    with pytest.raises(ValueError, match="at least 8"):
        build_heightmap(grid_vertices=4)


def test_hillshade_is_uint8_and_flat_ground_is_bright():
    flat = np.full((16, 16), 10.0, dtype=np.float32)
    shade = hillshade(flat, spacing_m=10.0)
    assert shade.dtype == np.uint8
    assert shade.shape == flat.shape
    assert shade.min() == shade.max(), "a flat surface has one shading value"


def test_triangulate_grid_counts_and_decimation():
    ground = GroundZ(z=np.zeros((17, 17), dtype=np.float32), half_extent_m=100.0)
    v_pts, v_faces, v_norms = triangulate_grid(ground, stride=1)
    c_pts, c_faces, _ = triangulate_grid(ground, stride=4)

    assert v_pts.shape == (17 * 17, 3)
    assert v_faces.shape == (16 * 16 * 2, 3)
    assert v_norms.shape == (17 * 17, 3)
    assert np.allclose(v_norms[:, 2], 1.0)

    # stride=4 takes every 4th vertex -> 5x5 vertices, 4x4*2 = 32 faces (16x fewer than 512)
    assert c_pts.shape == (5 * 5, 3)
    assert c_faces.shape == (4 * 4 * 2, 3)
    assert len(v_faces) == 16 * len(c_faces)


def test_classify_biomes_separates_wadi_earth_and_rock(full_map):
    z, _ = full_map
    ground = GroundZ(z=z, half_extent_m=5000.0)
    biome, rgb = classify_biomes(ground)

    assert biome.shape == z.shape
    assert set(np.unique(biome)) == {BIOME_WADI, BIOME_EARTH, BIOME_ROCK}
    assert rgb.shape == (*z.shape, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_export_terrain_usd_writes_visual_and_collider_meshes(tmp_path):
    z, _ = build_heightmap(grid_vertices=17, seed=0)
    ground = GroundZ(z=z, half_extent_m=500.0)
    out = export_terrain_usd(ground, tmp_path / "map.usda", visual_stride=1, collider_stride=4)

    text = out.read_text(encoding="ascii")
    assert text.startswith("#usda 1.0")
    assert 'def Mesh "VisualMesh"' in text
    assert 'def Mesh "ColliderMesh"' in text
    assert "PhysicsCollisionAPI" in text
    assert "primvars:displayColor" in text


def test_export_terrain_usd_rejects_finer_collider_than_visual(tmp_path):
    ground = GroundZ(z=np.zeros((9, 9), dtype=np.float32), half_extent_m=100.0)
    with pytest.raises(ValueError, match="collider_stride"):
        export_terrain_usd(ground, tmp_path / "bad.usda", visual_stride=4, collider_stride=2)
