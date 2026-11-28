"""B2 Asset Loader, Sketchfab Downloader & +K Hotspot Applier — plan.md §2, docs.md B2.

Does only three things:
  1. Loads per-model `.json` configs from `app/datagen/assets/meshes/<class>/<slug>.json`.
  2. Projects each model's 3D `+K` hotspots onto the 2D IR frame (`project_hotspots`).
  3. Downloads the model's `.glb` mesh from Sketchfab (`download_sketchfab_model`).

Usage:
    python -m app.datagen.assets --download --token <SKETCHFAB_API_TOKEN>
"""

from __future__ import annotations

import argparse
import io
import json
import os
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.datagen.config import CLASS_IDS, QUAD_SPAN_M
from app.datagen.ir_composite import Hotspot2D, ThermalMaterial

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
MESH_ROOT = ASSET_ROOT / "meshes"

SPAN_RANGES_M: dict[str, tuple[float, float]] = {
    "quad": QUAD_SPAN_M,        # 0.45 .. 1.00 m tip-to-tip incl. props
    "fixed_wing": (1.20, 2.50), # 1.20 .. 2.50 m wingspan
    "decoy": (0.20, 1.50),      # 0.20 .. 1.50 m
    "bird": (0.25, 1.00),       # 0.25 .. 1.00 m wingspan
}


@dataclass(frozen=True)
class Hotspot3D:
    """One 3D position on a unit-span mesh that adds +K in the IR pass."""

    name: str
    offset_norm: tuple[float, float, float]  # (dx, dy, dz) in unit-span [-0.5, +0.5]
    radius_norm: float                       # 1-sigma radius as a fraction of span_m
    delta_t_k: float                         # +K temperature boost above base_delta_t_k
    epsilon: float = 0.95

    def offset_m(self, span_m: float) -> np.ndarray:
        return np.asarray(self.offset_norm, dtype=np.float64) * float(span_m)

    def radius_m(self, span_m: float) -> float:
        return float(self.radius_norm) * float(span_m)

    @classmethod
    def from_dict(cls, d: dict) -> "Hotspot3D":
        ox, oy, oz = d["offset_norm"]
        return cls(
            name=str(d["name"]),
            offset_norm=(float(ox), float(oy), float(oz)),
            radius_norm=float(d["radius_norm"]),
            delta_t_k=float(d["delta_t_k"]),
            epsilon=float(d.get("epsilon", 0.95)),
        )


@dataclass(frozen=True)
class AssetThermalConfig:
    """Per-model JSON configuration pairing a Sketchfab 3D mesh with its +K hotspots."""

    slug: str
    name: str
    class_name: str
    class_id: int
    sketchfab_uid: str | None
    sketchfab_url: str | None
    mesh_file: str
    span_range_m: tuple[float, float]
    base_seg_id: int
    base_epsilon: float
    base_delta_t_k: float
    hotspots: tuple[Hotspot3D, ...]
    flap_hz_range: tuple[float, float] = (0.0, 0.0)
    notes: str = ""

    @property
    def base_material(self) -> ThermalMaterial:
        return ThermalMaterial(epsilon=self.base_epsilon, delta_t_k=self.base_delta_t_k)

    @property
    def peak_delta_t_k(self) -> float:
        return self.base_delta_t_k + max((h.delta_t_k for h in self.hotspots), default=0.0)

    def json_path(self, root: Path = MESH_ROOT) -> Path:
        return root / self.class_name / f"{self.slug}.json"

    def mesh_path(self, root: Path = MESH_ROOT) -> Path:
        return root / self.class_name / self.mesh_file

    @classmethod
    def load_json(cls, path: str | Path) -> "AssetThermalConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        bm = d["base_material"]
        sr = d["span_range_m"]
        fr = d.get("flap_hz_range", [0.0, 0.0])
        return cls(
            slug=str(d["slug"]),
            name=str(d["name"]),
            class_name=str(d["class_name"]),
            class_id=int(d["class_id"]),
            sketchfab_uid=d.get("sketchfab_uid"),
            sketchfab_url=d.get("sketchfab_url"),
            mesh_file=str(d["mesh_file"]),
            span_range_m=(float(sr[0]), float(sr[1])),
            base_seg_id=int(bm["seg_id"]),
            base_epsilon=float(bm["epsilon"]),
            base_delta_t_k=float(bm["delta_t_k"]),
            hotspots=tuple(Hotspot3D.from_dict(h) for h in d.get("hotspots", [])),
            flap_hz_range=(float(fr[0]), float(fr[1])),
            notes=str(d.get("notes", "")),
        )


def load_catalog(root: Path = MESH_ROOT) -> dict[str, AssetThermalConfig]:
    """Load all per-model `.json` configs from `app/datagen/assets/meshes/<class>/`."""
    return {
        cfg.slug: cfg
        for path in sorted(root.glob("*/*.json"))
        for cfg in (AssetThermalConfig.load_json(path),)
    }


def project_hotspots(
    config: AssetThermalConfig,
    *,
    span_m: float,
    center_uv: tuple[float, float],
    depth_m: float,
    focal_px: float,
    yaw_rad: float = 0.0,
    pitch_rad: float = 0.0,
    seg_id: int | None = None,
) -> list[Hotspot2D]:
    """Project 3D `+K` hotspots onto the 2D IR frame for `compose_ir(..., hotspots=...)`."""
    if depth_m <= 0.0 or focal_px <= 0.0:
        raise ValueError(f"depth_m ({depth_m}) and focal_px ({focal_px}) must be positive")

    target_seg = config.base_seg_id if seg_id is None else seg_id
    u0, v0 = float(center_uv[0]), float(center_uv[1])
    scale = float(focal_px) / float(depth_m)
    cy, sy = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    cp, sp = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))

    projected: list[Hotspot2D] = []
    for spot in config.hotspots:
        dx, dy, dz = spot.offset_m(span_m)
        x_yaw = cy * dx - sy * dy
        y_yaw = sy * dx + cy * dy
        z_pitch = cp * dz + sp * x_yaw
        projected.append(
            Hotspot2D(
                u_px=float(u0 + y_yaw * scale),
                v_px=float(v0 - z_pitch * scale),
                radius_px=float(max(spot.radius_m(span_m) * scale, 0.75)),
                delta_t_k=spot.delta_t_k,
                seg_id=target_seg,
                epsilon=spot.epsilon,
            )
        )
    return projected


def download_sketchfab_model(
    cfg: AssetThermalConfig,
    *,
    token: str,
    root: Path = MESH_ROOT,
    timeout_s: float = 30.0,
) -> Path | None:
    """Download `<slug>.glb` from Sketchfab's v3 download API using `token`."""
    if not cfg.sketchfab_uid:
        return None
    if not token:
        raise ValueError("Sketchfab download requires --token or SKETCHFAB_API_TOKEN")

    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        f"https://api.sketchfab.com/v3/models/{cfg.sketchfab_uid}/download",
        headers={"Authorization": f"Token {token.strip()}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
        links = json.loads(resp.read().decode("utf-8"))

    archive = links.get("glb") or links.get("usdz") or links.get("gltf") or links.get("source")
    if not archive or "url" not in archive:
        raise RuntimeError(f"no archive link for {cfg.slug} ({cfg.sketchfab_uid})")

    with urllib.request.urlopen(archive["url"], timeout=timeout_s * 2, context=ctx) as dl:
        payload = dl.read()

    out_path = cfg.mesh_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if payload[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(out_path.parent / cfg.slug)
    out_path.write_bytes(payload)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Sketchfab .glb meshes for the 15 JSON configs.")
    parser.add_argument("--download", action="store_true", help="download .glb meshes from Sketchfab")
    parser.add_argument("--token", default=os.environ.get("SKETCHFAB_API_TOKEN", ""))
    args = parser.parse_args()

    catalog = load_catalog()
    print(f"loaded {len(catalog)} model JSONs from {MESH_ROOT}")
    if args.download:
        for slug, cfg in catalog.items():
            try:
                path = download_sketchfab_model(cfg, token=args.token)
                print(f"  downloaded {slug} -> {path}")
            except urllib.error.HTTPError as exc:
                print(f"  failed {slug} ({cfg.sketchfab_uid}): HTTP {exc.code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
