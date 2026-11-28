"""Terrain heightmap — plan.md §1.2.1, docs.md A3.

One 10 km x 10 km limestone map, built once and saved to `assets/ground_z.npy`. The USD export
(Milestone B) needs `pxr`; this half is pure NumPy so it runs anywhere and is the thing every
AGL check relies on.

Recipe, in order:

    1. ridged multifractal    6 octaves, lambda_0 4096 m, lacunarity 2, gain 0.5
    2. N-S limestone ridge    one dominant crest, meandering, amplification only
    3. terraced plateau        eastern uplift
    4. wadi carve             domain-warped centreline, 20-60 m deep
    5. terracing              20-38 deg mid-slopes, 8-15 m steps
    6. rescale                floor 0 m, peaks 850 m

Why bake it: rebuilding per clip costs seconds and is the main throughput killer (plan.md §1.2.1).
Why bilinear: at 10 m spacing a nearest lookup quantises AGL by up to 5 m, and the drones fly at
15 m AGL.

Build and eyeball:

    python -m app.datagen.terrain_builder --build
    python -m app.datagen.terrain_builder --demo --out /tmp/terrain_demo.png
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_NPY = ASSET_DIR / "ground_z.npy"
DEFAULT_USD = ASSET_DIR / "map_10km.usda"

BIOME_WADI = 0
BIOME_EARTH = 1
BIOME_ROCK = 2

# RGB colors in [0, 1] for wadi floor (green-brown), terraced earth (olive-brown), rock (limestone grey-tan)
BIOME_PALETTE = np.array(
    [
        [0.34, 0.41, 0.27],  # 0: wadi floor & vegetation
        [0.55, 0.47, 0.34],  # 1: terraced earth & scrub
        [0.76, 0.73, 0.67],  # 2: bare limestone rock
    ],
    dtype=np.float32,
)

LATTICE = 256  # value-noise lattice; wraps every 256 cells, invisible at our frequencies


# --- noise -----------------------------------------------------------------------------


def _lattice(seed: int, size: int = LATTICE) -> np.ndarray:
    """One random value per lattice point, reproducible from a seed."""
    rng = np.random.default_rng(seed)
    return rng.random((size, size), dtype=np.float32)


def _value_noise(px: np.ndarray, py: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Smooth value noise in [0, 1] at fractional lattice coordinates (px, py).

    Four corner values, smoothstep weights. O(1) per point and fully vectorised.
    """
    ix = np.floor(px).astype(np.int64)
    iy = np.floor(py).astype(np.int64)
    tx = (px - ix).astype(np.float32)
    ty = (py - iy).astype(np.float32)

    # smoothstep: zero first derivative at both ends, so octaves do not show lattice seams
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)

    mask = table.shape[0] - 1
    x0, x1 = ix & mask, (ix + 1) & mask
    y0, y1 = iy & mask, (iy + 1) & mask

    v00 = table[x0, y0]
    v10 = table[x1, y0]
    v01 = table[x0, y1]
    v11 = table[x1, y1]

    top = v00 + sx * (v10 - v00)
    bottom = v01 + sx * (v11 - v01)
    return top + sy * (bottom - top)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ridged_multifractal(
    x: np.ndarray,
    y: np.ndarray,
    *,
    table: np.ndarray,
    octaves: int = 6,
    base_wavelength_m: float = 4096.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Ridged fBm: `1 - |noise|` turns smooth bumps into sharp limestone creases.

    Amplitude halves per octave and wavelength halves, so each octave adds finer detail. The
    `weight` feedback suppresses detail in valleys, which is what makes ridges read as ridges
    rather than as uniform rubble.
    """
    total = np.zeros_like(x, dtype=np.float32)
    amplitude = 1.0
    wavelength = base_wavelength_m
    weight = np.ones_like(x, dtype=np.float32)

    for octave in range(octaves):
        n = _value_noise(x / wavelength, y / wavelength, table)
        ridge = 1.0 - np.abs(2.0 * n - 1.0)  # 0 at the crease, 1 at the flats
        ridge = ridge * ridge  # sharpen the crest
        total += (amplitude * ridge * weight).astype(np.float32)
        weight = np.clip(ridge * 2.0, 0.0, 1.0).astype(np.float32)
        amplitude *= gain
        wavelength /= lacunarity

    return total


# --- shaping ---------------------------------------------------------------------------


def _normalise(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = float(z.max() - z.min())
    if span <= 0.0:
        return np.full_like(z, lo)
    return (lo + (z - z.min()) * (hi - lo) / span).astype(np.float32)


def _ridge_amplification(
    x: np.ndarray, y: np.ndarray, *, table: np.ndarray, half_extent_m: float
) -> np.ndarray:
    """A dominant north-south crest that meanders by roughly +/-1.5 km."""
    centre = 0.30 * half_extent_m * np.sin(2.0 * np.pi * y / (2.2 * half_extent_m))
    centre += 0.10 * half_extent_m * (2.0 * _value_noise(y / (1.4 * half_extent_m), np.zeros_like(y), table) - 1.0)
    width = 0.28 * half_extent_m
    return np.exp(-0.5 * ((x - centre) / width) ** 2).astype(np.float32)


def _carve_wadis(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    table: np.ndarray,
    half_extent_m: float,
    count: int = 2,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Cut meandering valleys. Domain warping bends the centreline so it is not a clean sine."""
    z = z.copy()
    meta: list[dict[str, float]] = []

    for index in range(count):
        depth_m = float(rng.uniform(20.0, 60.0))
        sigma_m = float(rng.uniform(80.0, 160.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        offset = 0.0 if index == 0 else float(rng.uniform(0.25, 0.45)) * half_extent_m

        # centreline: a slow N-S sine plus a low-frequency warp in both axes
        centre = offset + 0.18 * half_extent_m * np.sin(
            2.0 * np.pi * y / (2.6 * half_extent_m) + phase
        )
        warp = (
            2.0 * _value_noise(x / (0.9 * half_extent_m), y / (0.9 * half_extent_m), table) - 1.0
        ) * 0.06 * half_extent_m

        distance = x - centre + warp
        narrow = np.exp(-0.5 * (distance / sigma_m) ** 2)
        broad = np.exp(-0.5 * (distance / (4.0 * sigma_m)) ** 2)

        # the broad term guarantees the wadi band is the lowest ground, not just a groove
        z -= (depth_m * narrow + 0.35 * depth_m * broad).astype(np.float32)
        meta.append({"depth_m": round(depth_m, 2), "sigma_m": round(sigma_m, 2), "offset_m": round(offset, 1)})

    return z, meta


def _slope_deg(z: np.ndarray, spacing_m: float) -> np.ndarray:
    grad_y, grad_x = np.gradient(z, spacing_m)
    return np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32)


def _terrace(
    z: np.ndarray,
    slope: np.ndarray,
    *,
    step_m: float,
    lo_deg: float = 20.0,
    hi_deg: float = 38.0,
    ramp_deg: float = 4.0,
) -> np.ndarray:
    """Quantise height on mid-slopes only. Cliffs and floors keep their shape."""
    alpha = _smoothstep((slope - lo_deg) / ramp_deg) * (
        1.0 - _smoothstep((slope - (hi_deg - ramp_deg)) / ramp_deg)
    )
    quantised = np.floor(z / step_m) * step_m
    return (z + alpha * (quantised - z)).astype(np.float32)


# --- the map ---------------------------------------------------------------------------


def build_heightmap(
    *,
    grid_vertices: int = 1000,
    half_extent_m: float = 5000.0,
    seed: int = 0,
    relief_min_m: float = 0.0,
    relief_max_m: float = 850.0,
    octaves: int = 6,
    base_wavelength_m: float = 4096.0,
) -> tuple[np.ndarray, dict]:
    """Build the height grid. Returns `(z, info)` with `z` shape `(n, n)` float32 metres."""
    if grid_vertices < 8:
        raise ValueError(f"grid_vertices must be at least 8, got {grid_vertices}")

    axis = np.linspace(-half_extent_m, half_extent_m, grid_vertices, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    spacing_m = float(axis[1] - axis[0])

    table = _lattice(seed)
    z = ridged_multifractal(
        x,
        y,
        table=table,
        octaves=octaves,
        base_wavelength_m=base_wavelength_m,
    )
    z = _normalise(z, relief_min_m + 40.0, relief_max_m)

    crest = _ridge_amplification(x, y, table=table, half_extent_m=half_extent_m)
    plateau = _smoothstep((x - 0.05 * half_extent_m) / (0.9 * half_extent_m))
    z = z * (1.0 + 0.85 * crest) + 0.30 * relief_max_m * plateau
    z = _normalise(z, relief_min_m + 40.0, relief_max_m)

    rng = np.random.default_rng(seed + 1)
    z, wadis = _carve_wadis(
        z,
        x,
        y,
        rng=rng,
        table=_lattice(seed + 2),
        half_extent_m=half_extent_m,
    )

    step_m = float(rng.uniform(8.0, 15.0))
    slope = _slope_deg(z, spacing_m)
    z = _terrace(z, slope, step_m=step_m)
    z = _normalise(z, relief_min_m, relief_max_m)
    z = np.nan_to_num(z, nan=relief_min_m, posinf=relief_max_m, neginf=relief_min_m).astype(
        np.float32
    )

    info = {
        "grid_vertices": grid_vertices,
        "half_extent_m": half_extent_m,
        "spacing_m": round(spacing_m, 4),
        "seed": seed,
        "relief_m": [float(z.min()), float(z.max())],
        "terrace_step_m": round(step_m, 2),
        "wadis": wadis,
        "slope_p50_deg": round(float(np.percentile(slope, 50)), 2),
        "slope_p95_deg": round(float(np.percentile(slope, 95)), 2),
        "mid_slope_share": round(float(np.mean((slope >= 20.0) & (slope <= 38.0))), 4),
    }
    return z, info


@dataclass(frozen=True)
class GroundZ:
    """Baked heightmap with O(1) bilinear lookup. Everything downstream uses `at()` / `agl()`."""

    z: np.ndarray
    half_extent_m: float = 5000.0

    def __post_init__(self) -> None:
        if self.z.ndim != 2:
            raise ValueError(f"expected a 2-D heightmap, got shape {self.z.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def spacing_m(self) -> float:
        return 2.0 * self.half_extent_m / (self.z.shape[1] - 1)

    @property
    def extent_m(self) -> tuple[float, float]:
        return -self.half_extent_m, self.half_extent_m

    def at(self, x, y):
        """Bilinear height at world (x, y). Outside the grid, clamps to the nearest edge."""
        xs = np.asarray(x, dtype=np.float64)
        ys = np.asarray(y, dtype=np.float64)
        scalar = xs.ndim == 0 and ys.ndim == 0

        rows, cols = self.z.shape
        fx = np.clip((xs + self.half_extent_m) / self.spacing_m, 0.0, cols - 1.0)
        fy = np.clip((ys + self.half_extent_m) / self.spacing_m, 0.0, rows - 1.0)

        ix = np.clip(np.floor(fx).astype(np.int64), 0, cols - 2)
        iy = np.clip(np.floor(fy).astype(np.int64), 0, rows - 2)
        tx = fx - ix
        ty = fy - iy

        z00 = self.z[iy, ix]
        z10 = self.z[iy, ix + 1]
        z01 = self.z[iy + 1, ix]
        z11 = self.z[iy + 1, ix + 1]

        top = z00 + tx * (z10 - z00)
        bottom = z01 + tx * (z11 - z01)
        height = top + ty * (bottom - top)
        return float(height) if scalar else height

    def agl(self, x, y, z):
        """Height above ground level: positive above the surface, negative below it."""
        return np.asarray(z) - self.at(x, y)

    def save(self, path: str | Path = DEFAULT_NPY) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.z.astype(np.float32))
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {"half_extent_m": self.half_extent_m, "shape": list(self.z.shape), "dtype": "float32"},
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_NPY, half_extent_m: float | None = None) -> "GroundZ":
        path = Path(path)
        z = np.load(path)
        if half_extent_m is None:
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                half_extent_m = float(json.loads(sidecar.read_text())["half_extent_m"])
            else:
                half_extent_m = 5000.0
        return cls(z=z.astype(np.float32), half_extent_m=half_extent_m)


def build_ground(config=None, *, seed: int = 0) -> tuple[GroundZ, dict]:
    """Build from a `DatagenConfig`, so the grid and relief cannot drift from the spec."""
    if config is None:
        from app.datagen.config import DatagenConfig

        config = DatagenConfig()
    z, info = build_heightmap(
        grid_vertices=config.grid_vertices,
        half_extent_m=config.map_half_extent_m,
        seed=seed,
        relief_min_m=config.relief_min_m,
        relief_max_m=config.relief_max_m,
    )
    return GroundZ(z=z, half_extent_m=config.map_half_extent_m), info


# --- preview ---------------------------------------------------------------------------


def hillshade(
    z: np.ndarray,
    spacing_m: float,
    *,
    azimuth_deg: float = 315.0,
    elevation_deg: float = 45.0,
) -> np.ndarray:
    """Greyscale relief shading, 0-255. Cheap way to eyeball a heightmap without a plot library."""
    grad_y, grad_x = np.gradient(z, spacing_m)
    normal = np.stack([-grad_x, -grad_y, np.ones_like(z)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9

    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    sun = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    shade = np.clip(normal @ sun, 0.0, 1.0)
    return np.round(255.0 * shade).astype(np.uint8)


# --- Milestone B1: 3D mesh triangulation, biome colors & USD export --------------------


def triangulate_grid(
    ground: GroundZ, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the 2D height grid into 3D vertices `(N, 3)`, triangles `(F, 3)`, and normals `(N, 3)`."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    z_sub = ground.z[::stride, ::stride].astype(np.float32)
    rows, cols = z_sub.shape
    if rows < 2 or cols < 2:
        raise ValueError(f"subsampled grid {z_sub.shape} is too small to triangulate")

    xs = np.linspace(-ground.half_extent_m, ground.half_extent_m, cols, dtype=np.float32)
    ys = np.linspace(-ground.half_extent_m, ground.half_extent_m, rows, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack([x_grid, y_grid, z_sub], axis=-1).reshape(-1, 3)

    spacing = float(xs[1] - xs[0])
    grad_y, grad_x = np.gradient(z_sub, spacing)
    normals = np.stack([-grad_x, -grad_y, np.ones_like(z_sub)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    normals = normals.reshape(-1, 3).astype(np.float32)

    idx = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    v00, v10 = idx[:-1, :-1], idx[:-1, 1:]
    v01, v11 = idx[1:, :-1], idx[1:, 1:]
    faces = np.stack([v00, v10, v11, v00, v11, v01], axis=-1).reshape(-1, 3)
    return points, faces, normals


def classify_biomes(
    ground: GroundZ, *, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each grid cell into `0=wadi`, `1=earth`, `2=rock` and return shaded RGB `[0..1]`."""
    z_sub = ground.z[::stride, ::stride].astype(np.float32)
    spacing = 2.0 * ground.half_extent_m / max(z_sub.shape[1] - 1, 1)
    slope = _slope_deg(z_sub, spacing)

    is_rock = (slope > 28.0) | (z_sub > 500.0)
    is_wadi = (z_sub < 120.0) & (slope < 12.0) & (~is_rock)
    biome = np.where(is_rock, BIOME_ROCK, np.where(is_wadi, BIOME_WADI, BIOME_EARTH)).astype(
        np.int32
    )

    shade = hillshade(z_sub, spacing).astype(np.float32) / 255.0
    lit = 0.35 + 0.65 * shade[..., None]
    rgb = np.clip(BIOME_PALETTE[biome] * lit, 0.0, 1.0).astype(np.float32)
    return biome, rgb


def _fmt_vec3f(arr: np.ndarray) -> str:
    rows = [f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in arr]
    return "[" + ", ".join(rows) + "]"


def export_terrain_usd(
    ground: GroundZ,
    out_path: str | Path = DEFAULT_USD,
    *,
    visual_stride: int = 1,
    collider_stride: int = 4,
) -> Path:
    """Export `/World/Terrain/VisualMesh` and `/World/Terrain/ColliderMesh` to a USD stage."""
    if collider_stride < visual_stride:
        raise ValueError(
            f"collider_stride ({collider_stride}) must be >= visual_stride ({visual_stride})"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v_pts, v_faces, v_norms = triangulate_grid(ground, stride=visual_stride)
    c_pts, c_faces, _ = triangulate_grid(ground, stride=collider_stride)
    _, v_rgb = classify_biomes(ground, stride=visual_stride)
    v_colors = v_rgb.reshape(-1, 3)

    v_counts = ", ".join(["3"] * len(v_faces))
    c_counts = ", ".join(["3"] * len(c_faces))
    v_indices = ", ".join(str(int(i)) for i in v_faces.reshape(-1))
    c_indices = ", ".join(str(int(i)) for i in c_faces.reshape(-1))

    usda = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "Terrain" (
        customData = {{
            float epsilon = 0.92
            int seg_id = 1
        }}
    )
    {{
        def Mesh "VisualMesh"
        {{
            int[] faceVertexCounts = [{v_counts}]
            int[] faceVertexIndices = [{v_indices}]
            normal3f[] normals = {_fmt_vec3f(v_norms)} (
                interpolation = "vertex"
            )
            point3f[] points = {_fmt_vec3f(v_pts)}
            color3f[] primvars:displayColor = {_fmt_vec3f(v_colors)} (
                interpolation = "vertex"
            )
            uniform token subdivisionScheme = "none"
        }}

        def Mesh "ColliderMesh" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            int[] faceVertexCounts = [{c_counts}]
            int[] faceVertexIndices = [{c_indices}]
            point3f[] points = {_fmt_vec3f(c_pts)}
             token purpose = "guide"
            uniform token subdivisionScheme = "none"
        }}
    }}
}}
"""
    out_path.write_text(usda, encoding="ascii")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 10 km terrain heightmap and 3D USD mesh.")
    parser.add_argument("--build", action="store_true", help="build the full map and save it")
    parser.add_argument("--export-usd", action="store_true", help="export the 3D USD stage (Milestone B1)")
    parser.add_argument("--visual-stride", type=int, default=4, help="visual mesh stride for fast local QuickLook")
    parser.add_argument("--collider-stride", type=int, default=8, help="collider mesh stride")
    parser.add_argument("--demo", action="store_true", help="small map + biome-colored hillshade PNG")
    parser.add_argument("--out", default=None, help="output path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()

    if args.demo:
        z, info = build_heightmap(grid_vertices=384, seed=args.seed)
        ground = GroundZ(z=z, half_extent_m=5000.0)
        biome, rgb = classify_biomes(ground)
        from imageio.v3 import imwrite

        out = Path(args.out or "/tmp/terrain_biome_demo.png")
        imwrite(out, np.round(255.0 * rgb).astype(np.uint8))
        info["biome_shares"] = {
            "wadi": round(float(np.mean(biome == BIOME_WADI)), 4),
            "earth": round(float(np.mean(biome == BIOME_EARTH)), 4),
            "rock": round(float(np.mean(biome == BIOME_ROCK)), 4),
        }
        out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"wrote {out}  shape={rgb.shape}  {time.perf_counter() - started:.2f}s")
        print(json.dumps(info, indent=2))
        return 0

    if args.export_usd:
        ground = (
            GroundZ.load(DEFAULT_NPY)
            if DEFAULT_NPY.exists()
            else build_ground(seed=args.seed)[0]
        )
        usd_out = Path(args.out or DEFAULT_USD)
        export_terrain_usd(
            ground,
            usd_out,
            visual_stride=args.visual_stride,
            collider_stride=args.collider_stride,
        )
        print(
            f"wrote {usd_out}  size={usd_out.stat().st_size / 1e6:.2f} MB  "
            f"{time.perf_counter() - started:.2f}s"
        )
        return 0

    out = Path(args.out or DEFAULT_NPY)
    ground, info = build_ground(seed=args.seed)
    ground.save(out)
    print(f"wrote {out}  shape={ground.z.shape}  {time.perf_counter() - started:.2f}s")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
