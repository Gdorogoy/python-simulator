"""LWIR frame synthesis from Isaac geometry AOVs — plan.md §3.6, docs.md A1.

Isaac Sim renders visible light only, so an IR frame is *computed*: the renderer supplies
geometry buffers, and this module turns them into a thermal image in four steps.

    1. background temperature   normals + sun elevation + per-material inertia
    2. target temperature       instance_segmentation masks + per-part delta_t
    3. atmospheric attenuation  Beer-Lambert, uses the depth buffer
    4. sensor noise + display   NETD in kelvin, then a fixed-gain white-hot map

Noise is added in kelvin, never to the 8-bit image, and the display gain is fixed per run:
auto-gain would normalise target-vs-background contrast and destroy the class cues.

Run the demo to eyeball a frame:

    python -m app.datagen.ir_composite --demo --out /tmp/ir_demo.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import map_coordinates

# Segment ids used by the demo scene and the tests. Real runs get these from assets.py.
SEG_SKY = 0
SEG_GROUND = 1
SEG_QUAD_ROTOR = 110
SEG_QUAD_FRAME = 111
SEG_BIRD_CORE = 120
SEG_BIRD_WING = 121
SEG_FIXED_WING_EXHAUST = 130
SEG_DECOY = 140

# Physical constants of the model (not of the sensor).
SOLAR_GAIN_K = 25.0        # rock face perpendicular to a zenith sun
SKY_DEPRESSION_K = 20.0    # clear sky reads colder than the air
SHIMMER_MIN_FOCAL_MM = 300.0
SHIMMER_BASE_PX = 1.5


@dataclass(frozen=True)
class ThermalMaterial:
    """What one segment id contributes to the thermal image."""

    epsilon: float          # LWIR emissivity, 0..1
    delta_t_k: float        # temperature above ambient, kelvin
    inertia_k: float = 0.0  # heat retained from earlier sun (ground only)


@dataclass(frozen=True)
class Hotspot2D:
    """Projected 2D thermal hotspot stamped onto a mesh mask in kelvin (docs.md B2)."""

    u_px: float             # horizontal pixel centre (column)
    v_px: float             # vertical pixel centre (row)
    radius_px: float        # 1-sigma Gaussian radius in pixels
    delta_t_k: float        # kelvin boost above the mesh base temperature
    seg_id: int             # restricts the stamp to seg == seg_id (never bleeds to sky/ground)
    epsilon: float = 0.95   # hotspot emissivity (matte motor bell / exhaust)


@dataclass(frozen=True)
class SunState:
    elevation_deg: float
    azimuth_deg: float = 0.0
    t_amb_k: float = 293.15

    @property
    def direction(self) -> np.ndarray:
        """Unit vector toward the sun, z up."""
        el = np.radians(self.elevation_deg)
        az = np.radians(self.azimuth_deg)
        return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])

    @property
    def solar_weight(self) -> float:
        """Sun-driven heating factor: 0 at the horizon, 1 at zenith."""
        return float(np.clip(np.sin(np.radians(self.elevation_deg)), 0.0, 1.0))


@dataclass(frozen=True)
class WhiteHot:
    """Fixed-gain display mapping. Constant for a whole run, never per frame."""

    t_min_k: float
    t_max_k: float
    gamma: float = 0.8

    @classmethod
    def around(cls, t_amb_k: float) -> "WhiteHot":
        return cls(t_min_k=t_amb_k - 25.0, t_max_k=t_amb_k + 45.0)


DEFAULT_MATERIALS: dict[int, ThermalMaterial] = {
    SEG_GROUND: ThermalMaterial(epsilon=0.92, delta_t_k=0.0, inertia_k=6.0),
    SEG_QUAD_ROTOR: ThermalMaterial(epsilon=0.95, delta_t_k=35.0),
    SEG_QUAD_FRAME: ThermalMaterial(epsilon=0.85, delta_t_k=2.0),
    SEG_BIRD_CORE: ThermalMaterial(epsilon=0.98, delta_t_k=12.0),
    SEG_BIRD_WING: ThermalMaterial(epsilon=0.98, delta_t_k=1.0),
    SEG_FIXED_WING_EXHAUST: ThermalMaterial(epsilon=0.95, delta_t_k=60.0),
    SEG_DECOY: ThermalMaterial(epsilon=0.97, delta_t_k=8.0),
}


def beam_extinction(visibility_m: float) -> float:
    """Koschmieder extinction coefficient beta for a meteorological visibility."""
    if visibility_m <= 0.0:
        raise ValueError(f"visibility_m must be positive, got {visibility_m}")
    return 3.912 / visibility_m


def make_fpn(shape: tuple[int, int], rng: np.random.Generator, sigma_k: float) -> np.ndarray:
    """Fixed-pattern noise: one offset per pixel, constant for a whole clip."""
    return rng.normal(0.0, sigma_k, size=shape).astype(np.float32)


def _material_fields(
    seg: np.ndarray, materials: dict[int, ThermalMaterial]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel epsilon / delta_t / inertia, defaulting to the ground material."""
    eps = np.full(seg.shape, materials[SEG_GROUND].epsilon, dtype=np.float32)
    dt = np.zeros(seg.shape, dtype=np.float32)
    inertia = np.full(seg.shape, materials[SEG_GROUND].inertia_k, dtype=np.float32)
    for seg_id, mat in materials.items():
        mask = seg == seg_id
        if not mask.any():
            continue
        eps[mask] = mat.epsilon
        dt[mask] = mat.delta_t_k
        inertia[mask] = mat.inertia_k
    return eps, dt, inertia


def apply_hotspots(
    t_k: np.ndarray,
    eps: np.ndarray,
    seg: np.ndarray,
    hotspots: Sequence[Hotspot2D],
) -> tuple[np.ndarray, np.ndarray]:
    """Stamp position-based +K hotspots onto their target masks in kelvin before attenuation.

    Larger ``delta_t_k`` increases surface temperature at ``(u_px, v_px)``, which shines brighter
    after Beer-Lambert attenuation and White-Hot mapping. Clamped to ``seg == spot.seg_id`` so
    heat never bleeds outside the mesh silhouette.
    """
    if not hotspots:
        return t_k, eps
    t_out = t_k.copy()
    eps_out = eps.copy()
    for spot in hotspots:
        rows, cols = np.nonzero(seg == spot.seg_id)
        if rows.size == 0:
            continue
        sigma = max(float(spot.radius_px), 0.75)
        d2 = (cols.astype(np.float32) - np.float32(spot.u_px)) ** 2 + (
            rows.astype(np.float32) - np.float32(spot.v_px)
        ) ** 2
        weight = np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)
        t_out[rows, cols] += np.float32(spot.delta_t_k) * weight
        eps_out[rows, cols] = np.clip(
            eps_out[rows, cols] + weight * (np.float32(spot.epsilon) - eps_out[rows, cols]),
            0.0,
            1.0,
        )
    return t_out, eps_out


def compose_ir(
    seg: np.ndarray,
    normals: np.ndarray,
    depth_m: np.ndarray,
    *,
    sun: SunState,
    materials: dict[int, ThermalMaterial] | None = None,
    hotspots: Sequence[Hotspot2D] = (),
    visibility_m: float = 20_000.0,
    netd_k: float = 0.050,
    rng: np.random.Generator | None = None,
    fpn: np.ndarray | None = None,
    focal_mm: float = 100.0,
    wht: WhiteHot | None = None,
    debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compose one IR frame.

    Returns a ``(H, W)`` uint8 image, or ``(uint8, float32 kelvin)`` when ``debug=True``.
    """
    materials = DEFAULT_MATERIALS if materials is None else materials
    rng = np.random.default_rng() if rng is None else rng
    wht = WhiteHot.around(sun.t_amb_k) if wht is None else wht

    if seg.shape != depth_m.shape or normals.shape[:2] != seg.shape:
        raise ValueError(f"shape mismatch: seg {seg.shape}, depth {depth_m.shape}, normals {normals.shape}")

    # 1. background: sun-driven heating, scaled by how directly each face meets the sun.
    #    solar_weight -> 0 at the horizon is what produces the dawn/dusk thermal crossover.
    cos_inc = np.clip(normals @ sun.direction, 0.0, 1.0).astype(np.float32)
    eps, delta_t, inertia = _material_fields(seg, materials)
    solar = sun.solar_weight
    t = sun.t_amb_k + solar * (SOLAR_GAIN_K * cos_inc + inertia)

    # 2. targets: base mesh temperature, plus any position-based +K hotspots from the asset JSON.
    is_sky = seg == SEG_SKY
    t = np.where(seg >= 100, sun.t_amb_k + delta_t, t)
    if hotspots:
        t, eps = apply_hotspots(t, eps, seg, hotspots)
    t = np.where(is_sky, sun.t_amb_k - SKY_DEPRESSION_K, t)

    # 3. attenuation toward the air path. Sky has no path, so it keeps its own temperature.
    #    A surface emits eps*T and reflects the rest of its surroundings (approximately the air),
    #    so a matte surface at air temperature stays at air temperature — that is the crossover.
    tau = np.exp(-beam_extinction(visibility_m) * np.asarray(depth_m, dtype=np.float32))
    surface = eps * t + (1.0 - eps) * sun.t_amb_k
    attenuated = tau * surface + (1.0 - tau) * sun.t_amb_k
    t = np.where(is_sky, t, attenuated)

    # 4. sensor noise, still in kelvin; then the fixed-gain display map.
    t = t + rng.normal(0.0, netd_k, size=t.shape).astype(np.float32)
    if fpn is not None:
        t = t + fpn

    img = white_hot(t, wht)
    if focal_mm >= SHIMMER_MIN_FOCAL_MM:
        img = apply_heat_shimmer(img, focal_mm=focal_mm, rng=rng)

    if debug:
        return img, t.astype(np.float32)
    return img


def white_hot(t_k: np.ndarray, wht: WhiteHot) -> np.ndarray:
    """Fixed-gain temperature -> 8-bit grey. No auto-gain, no per-frame normalisation."""
    span = max(wht.t_max_k - wht.t_min_k, 1e-6)
    norm = np.clip((t_k - wht.t_min_k) / span, 0.0, 1.0)
    return np.round(255.0 * np.power(norm, wht.gamma)).astype(np.uint8)


def apply_heat_shimmer(
    img_u8: np.ndarray, *, focal_mm: float, rng: np.random.Generator, coarse_px: int = 32
) -> np.ndarray:
    """Whole-frame refractive warp. Never displaces the target alone: that would fake motion."""
    if focal_mm < SHIMMER_MIN_FOCAL_MM:
        return img_u8
    h, w = img_u8.shape
    low_h, low_w = h // coarse_px + 2, w // coarse_px + 2
    field = rng.normal(size=(2, low_h, low_w)).astype(np.float32)

    yy, xx = np.meshgrid(np.arange(h) / coarse_px, np.arange(w) / coarse_px, indexing="ij")
    dy = map_coordinates(field[0], [yy, xx], order=1, mode="nearest")
    dx = map_coordinates(field[1], [yy, xx], order=1, mode="nearest")

    amplitude = SHIMMER_BASE_PX * (focal_mm / SHIMMER_MIN_FOCAL_MM)
    warped = map_coordinates(
        img_u8.astype(np.float32),
        [np.arange(h)[:, None] + amplitude * dy, np.arange(w)[None, :] + amplitude * dx],
        order=1,
        mode="nearest",
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


# --- demo scene -----------------------------------------------------------------------

def make_synthetic_scene(
    h: int = 360, w: int = 640, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five actors on a tilted ground plane. Stands in for the renderer until Milestone B."""
    rng = np.random.default_rng(7) if rng is None else rng
    seg = np.full((h, w), SEG_SKY, dtype=np.int32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    depth = np.full((h, w), np.inf, dtype=np.float32)

    horizon = int(h * 0.45)
    rows = np.arange(horizon, h)
    seg[horizon:, :] = SEG_GROUND
    normals[horizon:, :, 0] = 0.0
    normals[horizon:, :, 2] = 0.93
    normals[horizon:, :, 1] = np.linspace(0.0, 0.37, rows.size)[:, None]
    depth[horizon:, :] = np.linspace(500.0, 5000.0, rows.size)[:, None]

    def blob(top, left, height, width, seg_id, distance):
        seg[top : top + height, left : left + width] = seg_id
        depth[top : top + height, left : left + width] = distance
        normals[top : top + height, left : left + width] = (0.0, 0.0, 1.0)

    blob(horizon - 14, 300, 7, 7, SEG_QUAD_ROTOR, 3000.0)
    blob(horizon - 16, 400, 8, 5, SEG_BIRD_CORE, 3000.0)
    blob(horizon - 22, 480, 3, 12, SEG_FIXED_WING_EXHAUST, 4200.0)
    blob(horizon - 9, 200, 4, 4, SEG_DECOY, 1500.0)
    return seg, normals, depth


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose a synthetic LWIR frame.")
    parser.add_argument("--demo", action="store_true", help="render the 5-actor demo frame")
    parser.add_argument("--out", default="/tmp/ir_demo.png")
    parser.add_argument("--elevation", type=float, default=12.0)
    args = parser.parse_args()

    seg, normals, depth = make_synthetic_scene()
    img, temp = compose_ir(
        seg,
        normals,
        depth,
        sun=SunState(elevation_deg=args.elevation),
        rng=np.random.default_rng(0),
        debug=True,
    )

    if args.demo:
        from imageio.v3 import imwrite

        imwrite(args.out, img)
        print(f"wrote {args.out}  shape={img.shape} dtype={img.dtype}")
        for name, seg_id in (
            ("sky", SEG_SKY),
            ("ground", SEG_GROUND),
            ("quad rotor", SEG_QUAD_ROTOR),
            ("bird core", SEG_BIRD_CORE),
            ("fixed-wing exhaust", SEG_FIXED_WING_EXHAUST),
            ("decoy", SEG_DECOY),
        ):
            if name == "sky":
                m = seg == SEG_SKY
            else:
                m = seg == seg_id
            print(f"  {name:<20} u8={img[m].mean():6.1f}  T={temp[m].mean():7.2f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
