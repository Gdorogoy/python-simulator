"""Milestone B3 — Camera Rig, Frustum Target Spawner & Rate-Limited PTZ (`plan.md` §3, `docs.md` B3).

Four core functions glue terrain (A3/B1), target assets (B2), the IR compositor (A1), and the
YOLO crop mapper (A2) into a working camera rig:

    1. camera_optics()      native FOV, pixel focal length (`focal_px`), `ifov_mrad`, and crop FOV
    2. spawn_in_frustum()   places a target at `300..5000 m` inside the central crop cone, >=15 m AGL
    3. step_ptz()           slews pan/tilt toward the target with gimbal speed limit, lag & jitter
    4. render_rig_frame()   reads split EO_AOVS / IR_AOVS, projects +K hotspots (B2), runs compose_ir (A1)

Demo:

    python -m app.datagen.rig --demo --out /tmp/rig_demo.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from app.datagen.assets import AssetThermalConfig, load_catalog, project_hotspots
from app.datagen.config import EO_AOVS, IR_AOVS, DatagenConfig
from app.datagen.export_yolo import CropWindow, crop_origin, remap_box
from app.datagen.ir_composite import (
    DEFAULT_MATERIALS,
    SEG_GROUND,
    SEG_SKY,
    SunState,
    ThermalMaterial,
    compose_ir,
)
from app.datagen.terrain_builder import GroundZ

DEFAULT_SENSOR_WIDTH_MM: dict[str, float] = {"eo": 12.8, "ir": 10.88}


@dataclass(frozen=True)
class Optics:
    """Lens and raster geometry for one camera modality."""

    modality: str
    width_px: int
    height_px: int
    crop_px: int
    ox: int
    oy: int
    focal_mm: float
    sensor_width_mm: float
    fov_x_rad: float
    fov_y_rad: float
    crop_fov_rad: float
    focal_px: float
    ifov_mrad: float

    @property
    def window(self) -> CropWindow:
        return CropWindow(
            width=self.width_px,
            height=self.height_px,
            crop_px=self.crop_px,
            ox=self.ox,
            oy=self.oy,
        )


@dataclass(frozen=True)
class PTZState:
    """Camera position and gimbal pan/tilt angles in world coordinates (Z up)."""

    cam_xyz: tuple[float, float, float]
    yaw_rad: float
    pitch_rad: float
    max_rate_rad_s: float = np.radians(45.0)  # 45 deg/s gimbal slew limit
    tau_lag_s: float = 0.12                   # first-order servo time constant
    jitter_rad: float = 0.00015               # 0.15 mrad wind / tripod micro-jitter

    def with_angles(self, yaw_rad: float, pitch_rad: float) -> "PTZState":
        max_pitch = np.radians(85.0)
        return PTZState(
            cam_xyz=self.cam_xyz,
            yaw_rad=float(_wrap_pi(yaw_rad)),
            pitch_rad=float(np.clip(pitch_rad, -max_pitch, max_pitch)),
            max_rate_rad_s=self.max_rate_rad_s,
            tau_lag_s=self.tau_lag_s,
            jitter_rad=self.jitter_rad,
        )


@dataclass(frozen=True)
class SpawnPoint:
    """World spawn position inside the camera's training-crop frustum cone."""

    xyz: tuple[float, float, float]
    distance_m: float
    agl_m: float
    d_yaw_rad: float
    d_pitch_rad: float
    bg_mode: str


@dataclass(frozen=True)
class RigActor:
    """One active target in the scene at the current frame."""

    track_id: int
    asset: AssetThermalConfig
    span_m: float
    xyz: tuple[float, float, float]
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    seg_id: int | None = None

    @property
    def effective_seg_id(self) -> int:
        return self.asset.base_seg_id if self.seg_id is None else self.seg_id


@dataclass(frozen=True)
class RigFrame:
    """Outputs of one dual-modality rig step."""

    eo_rgb: np.ndarray                                      # (H_eo, W_eo, 3) uint8
    ir_u8: np.ndarray                                       # (H_ir, W_ir) uint8
    ir_temp_k: np.ndarray                                   # (H_ir, W_ir) float32
    boxes: list[tuple[int, int, float, float, float, float]]  # (track_id, class_id, cx, cy, w, h)
    in_crop: bool                                           # primary target inside IR train crop


class RigBackend(Protocol):
    """Interface implemented by Isaac Replicator on the GPU box or SyntheticRigBackend locally."""

    def read_eo(
        self, ptz: PTZState, optics: Optics, actors: Sequence[RigActor], aovs: tuple[str, ...]
    ) -> tuple[np.ndarray, list[tuple[int, int, float, float, float, float]]]: ...

    def read_ir(
        self, ptz: PTZState, optics: Optics, actors: Sequence[RigActor], aovs: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float, float, float, float]]]: ...


# --- helpers ---------------------------------------------------------------------------


def _wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def look_angles(
    cam_xyz: Sequence[float], target_xyz: Sequence[float]
) -> tuple[float, float, float]:
    """Return `(yaw_rad, pitch_rad, slant_range_m)` from `cam_xyz` to `target_xyz`."""
    dx = float(target_xyz[0]) - float(cam_xyz[0])
    dy = float(target_xyz[1]) - float(cam_xyz[1])
    dz = float(target_xyz[2]) - float(cam_xyz[2])
    horiz = float(np.hypot(dx, dy))
    slant = float(np.hypot(horiz, dz))
    if slant < 1e-6:
        return 0.0, 0.0, 0.0
    yaw = float(np.arctan2(dy, dx))
    pitch = float(np.arctan2(dz, max(horiz, 1e-9)))
    return yaw, pitch, slant


def project_world_to_pixel(
    ptz: PTZState, optics: Optics, target_xyz: Sequence[float]
) -> tuple[float, float, float] | None:
    """Project a 3D world point to `(u_px, v_px, depth_m)` in `optics`, or `None` if behind camera."""
    yaw_t, pitch_t, slant_m = look_angles(ptz.cam_xyz, target_xyz)
    if slant_m <= 0.0:
        return None
    d_yaw = _wrap_pi(yaw_t - ptz.yaw_rad)
    d_pitch = pitch_t - ptz.pitch_rad
    if abs(d_yaw) >= 0.49 * np.pi or abs(d_pitch) >= 0.49 * np.pi:
        return None

    u_px = optics.width_px / 2.0 + optics.focal_px * float(np.tan(d_yaw))
    v_px = optics.height_px / 2.0 - optics.focal_px * float(np.tan(d_pitch))
    depth_m = slant_m * float(np.cos(d_yaw) * np.cos(d_pitch))
    return float(u_px), float(v_px), max(depth_m, 1.0)


# --- Function 1: camera_optics ---------------------------------------------------------


def camera_optics(
    cfg: DatagenConfig,
    modality: str,
    focal_mm: float,
    *,
    sensor_width_mm: float | None = None,
) -> Optics:
    """Compute lens FOV, pixel focal length (`focal_px`), `ifov_mrad`, and crop FOV."""
    if focal_mm <= 0.0:
        raise ValueError(f"focal_mm must be positive, got {focal_mm}")
    width_px, height_px = cfg.native_res(modality)
    crop_px = cfg.train_crop_px(modality)
    ox, oy = crop_origin((width_px, height_px), crop_px)

    sw_mm = DEFAULT_SENSOR_WIDTH_MM[modality] if sensor_width_mm is None else float(sensor_width_mm)
    if sw_mm <= 0.0:
        raise ValueError(f"sensor_width_mm must be positive, got {sw_mm}")

    fov_x_rad = float(2.0 * np.arctan(sw_mm / (2.0 * focal_mm)))
    focal_px = float((width_px / 2.0) / np.tan(fov_x_rad / 2.0))
    fov_y_rad = float(2.0 * np.arctan((height_px / 2.0) / focal_px))
    crop_fov_rad = float(2.0 * np.arctan((crop_px / 2.0) / focal_px))
    ifov_mrad = float(1000.0 * fov_x_rad / width_px)

    return Optics(
        modality=modality,
        width_px=width_px,
        height_px=height_px,
        crop_px=crop_px,
        ox=ox,
        oy=oy,
        focal_mm=float(focal_mm),
        sensor_width_mm=sw_mm,
        fov_x_rad=fov_x_rad,
        fov_y_rad=fov_y_rad,
        crop_fov_rad=crop_fov_rad,
        focal_px=focal_px,
        ifov_mrad=ifov_mrad,
    )


# --- Function 2: spawn_in_frustum ------------------------------------------------------


def spawn_in_frustum(
    ground: GroundZ,
    ptz: PTZState,
    optics: Optics,
    bucket_m: tuple[float, float],
    *,
    bg_mode: str = "sky",
    min_agl_m: float = 15.0,
    crop_margin: float = 0.35,
    rng: np.random.Generator | None = None,
) -> SpawnPoint:
    """Spawn a target at `bucket_m` inside the central training-crop cone and >= `min_agl_m` AGL."""
    lo_m, hi_m = float(bucket_m[0]), float(bucket_m[1])
    if lo_m <= 0.0 or hi_m < lo_m:
        raise ValueError(f"invalid bucket_m {bucket_m}")
    if bg_mode not in ("sky", "horizon", "ground"):
        raise ValueError(f"unknown bg_mode {bg_mode!r}; expected 'sky', 'horizon', or 'ground'")
    if not (0.0 < crop_margin <= 0.49):
        raise ValueError(f"crop_margin must be in (0, 0.49], got {crop_margin}")

    rng = np.random.default_rng() if rng is None else rng
    r_m = float(rng.uniform(lo_m, hi_m))
    half_cone = crop_margin * optics.crop_fov_rad
    d_yaw = float(rng.uniform(-half_cone, half_cone))

    if bg_mode == "sky":
        d_pitch = float(rng.uniform(0.05 * half_cone, half_cone))
    elif bg_mode == "ground":
        d_pitch = float(rng.uniform(-half_cone, -0.05 * half_cone))
    else:
        d_pitch = float(rng.uniform(-0.25 * half_cone, 0.25 * half_cone))

    ray_yaw = ptz.yaw_rad + d_yaw
    ray_pitch = ptz.pitch_rad + d_pitch
    horiz_m = r_m * float(np.cos(ray_pitch))

    x = float(ptz.cam_xyz[0]) + horiz_m * float(np.cos(ray_yaw))
    y = float(ptz.cam_xyz[1]) + horiz_m * float(np.sin(ray_yaw))
    z_ray = float(ptz.cam_xyz[2]) + r_m * float(np.sin(ray_pitch))
    z_gnd = float(ground.at(x, y))
    z = max(z_gnd + float(min_agl_m), z_ray)
    _, _, actual_dist_m = look_angles(ptz.cam_xyz, (x, y, z))

    return SpawnPoint(
        xyz=(x, y, z),
        distance_m=actual_dist_m,
        agl_m=z - z_gnd,
        d_yaw_rad=d_yaw,
        d_pitch_rad=d_pitch,
        bg_mode=bg_mode,
    )


# --- Function 3: step_ptz --------------------------------------------------------------


def step_ptz(
    ptz: PTZState,
    target_xyz: Sequence[float],
    dt_s: float,
    *,
    rng: np.random.Generator | None = None,
) -> PTZState:
    """Slew camera pan/tilt toward `target_xyz` with rate limit, first-order lag, and wind jitter."""
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    rng = np.random.default_rng() if rng is None else rng

    want_yaw, want_pitch, _ = look_angles(ptz.cam_xyz, target_xyz)
    err_yaw = _wrap_pi(want_yaw - ptz.yaw_rad)
    err_pitch = want_pitch - ptz.pitch_rad

    tau = max(ptz.tau_lag_s, dt_s)
    rate_yaw = float(np.clip(err_yaw / tau, -ptz.max_rate_rad_s, ptz.max_rate_rad_s))
    rate_pitch = float(np.clip(err_pitch / tau, -ptz.max_rate_rad_s, ptz.max_rate_rad_s))

    jy, jp = (
        rng.normal(0.0, ptz.jitter_rad, size=2) if ptz.jitter_rad > 0.0 else (0.0, 0.0)
    )
    return ptz.with_angles(
        ptz.yaw_rad + rate_yaw * dt_s + float(jy),
        ptz.pitch_rad + rate_pitch * dt_s + float(jp),
    )


# --- Function 4: render_rig_frame ------------------------------------------------------


def render_rig_frame(
    backend: RigBackend,
    *,
    ptz: PTZState,
    eo_optics: Optics,
    ir_optics: Optics,
    actors: Sequence[RigActor],
    sun: SunState,
    visibility_m: float = 20_000.0,
    netd_k: float = 0.050,
    fpn: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> RigFrame:
    """Read split EO/IR AOVs, project each actor's +K hotspots (B2), and run `compose_ir` (A1)."""
    rng = np.random.default_rng() if rng is None else rng

    eo_rgb, _ = backend.read_eo(ptz, eo_optics, actors, EO_AOVS)
    seg, normals, depth_m, ir_boxes = backend.read_ir(ptz, ir_optics, actors, IR_AOVS)

    materials: dict[int, ThermalMaterial] = dict(DEFAULT_MATERIALS)
    spots_2d = []
    for actor in actors:
        materials[actor.effective_seg_id] = actor.asset.base_material
        proj = project_world_to_pixel(ptz, ir_optics, actor.xyz)
        if proj is None:
            continue
        u_px, v_px, d_m = proj
        spots_2d.extend(
            project_hotspots(
                actor.asset,
                span_m=actor.span_m,
                center_uv=(u_px, v_px),
                depth_m=d_m,
                focal_px=ir_optics.focal_px,
                yaw_rad=actor.yaw_rad,
                pitch_rad=actor.pitch_rad,
                seg_id=actor.effective_seg_id,
            )
        )

    ir_u8, ir_temp_k = compose_ir(
        seg,
        normals,
        depth_m,
        sun=sun,
        materials=materials,
        hotspots=spots_2d,
        visibility_m=visibility_m,
        netd_k=netd_k,
        fpn=fpn,
        focal_mm=ir_optics.focal_mm,
        rng=rng,
        debug=True,
    )

    in_crop = False
    for _, _, cx, cy, w, h in ir_boxes:
        if remap_box(cx, cy, w, h, window=ir_optics.window) is not None:
            in_crop = True
            break

    return RigFrame(
        eo_rgb=eo_rgb,
        ir_u8=ir_u8,
        ir_temp_k=ir_temp_k,
        boxes=ir_boxes,
        in_crop=in_crop,
    )


# --- Local Synthetic Backend (runs without Isaac Sim for tests and smoke previews) -----


class SyntheticRigBackend:
    """Software AOV rasterizer over `GroundZ` that implements `RigBackend` without `pxr`."""

    def __init__(self, ground: GroundZ, *, horizon_frac: float = 0.46) -> None:
        self.ground = ground
        self.horizon_frac = horizon_frac

    def _rasterize_actors(
        self, ptz: PTZState, optics: Optics, actors: Sequence[RigActor]
    ) -> list[tuple[RigActor, int, int, int, int, float]]:
        placed = []
        for actor in actors:
            proj = project_world_to_pixel(ptz, optics, actor.xyz)
            if proj is None:
                continue
            u_px, v_px, depth_m = proj
            span_px = max(float(actor.span_m) * optics.focal_px / depth_m, 2.0)
            half_w = max(int(round(0.5 * span_px)), 1)
            half_h = max(int(round(0.35 * span_px)), 1)
            x1 = int(np.clip(round(u_px) - half_w, 0, optics.width_px))
            x2 = int(np.clip(round(u_px) + half_w + 1, 0, optics.width_px))
            y1 = int(np.clip(round(v_px) - half_h, 0, optics.height_px))
            y2 = int(np.clip(round(v_px) + half_h + 1, 0, optics.height_px))
            if x2 > x1 and y2 > y1:
                placed.append((actor, x1, y1, x2, y2, depth_m))
        return placed

    @staticmethod
    def _boxes_from_placed(
        placed: Sequence[tuple[RigActor, int, int, int, int, float]], optics: Optics
    ) -> list[tuple[int, int, float, float, float, float]]:
        boxes = []
        for actor, x1, y1, x2, y2, _ in placed:
            cx = ((x1 + x2) / 2.0) / optics.width_px
            cy = ((y1 + y2) / 2.0) / optics.height_px
            w = (x2 - x1) / optics.width_px
            h = (y2 - y1) / optics.height_px
            boxes.append((actor.track_id, actor.asset.class_id, cx, cy, w, h))
        return boxes

    def read_eo(
        self, ptz: PTZState, optics: Optics, actors: Sequence[RigActor], aovs: tuple[str, ...]
    ) -> tuple[np.ndarray, list[tuple[int, int, float, float, float, float]]]:
        assert aovs == EO_AOVS
        h, w = optics.height_px, optics.width_px
        horizon = int(np.clip(round(h * (self.horizon_frac + 0.25 * ptz.pitch_rad)), 1, h - 1))

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:horizon] = (175, 200, 228)   # daytime sky
        rgb[horizon:] = (142, 125, 96)    # Galilee limestone/earth slope

        placed = self._rasterize_actors(ptz, optics, actors)
        for _, x1, y1, x2, y2, _ in placed:
            rgb[y1:y2, x1:x2] = (38, 42, 46)
        return rgb, self._boxes_from_placed(placed, optics)

    def read_ir(
        self, ptz: PTZState, optics: Optics, actors: Sequence[RigActor], aovs: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float, float, float, float]]]:
        assert aovs == IR_AOVS
        h, w = optics.height_px, optics.width_px
        horizon = int(np.clip(round(h * (self.horizon_frac + 0.25 * ptz.pitch_rad)), 1, h - 1))

        seg = np.full((h, w), SEG_SKY, dtype=np.int32)
        seg[horizon:] = SEG_GROUND
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 2] = 0.94
        normals[horizon:, :, 1] = np.linspace(0.0, 0.34, h - horizon, dtype=np.float32)[:, None]
        depth = np.full((h, w), np.inf, dtype=np.float32)
        depth[horizon:] = np.linspace(600.0, 4800.0, h - horizon, dtype=np.float32)[:, None]

        placed = self._rasterize_actors(ptz, optics, actors)
        for actor, x1, y1, x2, y2, d_m in placed:
            seg[y1:y2, x1:x2] = actor.effective_seg_id
            depth[y1:y2, x1:x2] = d_m
            normals[y1:y2, x1:x2] = (0.0, 0.0, 1.0)
        return seg, normals, depth, self._boxes_from_placed(placed, optics)


# --- Isaac Sim / Omniverse Replicator GPU Backend --------------------------------------


class IsaacReplicatorBackend:
    """Hardware-raytraced Omniverse Replicator backend (requires a running `SimulationApp`)."""

    def __init__(self, ground: GroundZ, cfg: DatagenConfig) -> None:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        from app.datagen.terrain_builder import DEFAULT_USD, export_terrain_usd

        self._rep = rep
        self._Gf = Gf
        self._UsdGeom = UsdGeom
        self.ground = ground
        self.cfg = cfg

        if not DEFAULT_USD.exists():
            export_terrain_usd(ground, DEFAULT_USD, visual_stride=4, collider_stride=8)

        ctx = omni.usd.get_context()
        ctx.new_stage()
        self.stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        terrain_prim = self.stage.DefinePrim("/World/Terrain", "Xform")
        terrain_prim.GetReferences().AddReference(str(DEFAULT_USD))

        sun_light = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/World/SunLight"))
        sun_light.CreateIntensityAttr(3000.0)

        self.eo_cam = UsdGeom.Camera.Define(self.stage, Sdf.Path("/World/Rig/EOCamera"))
        self.ir_cam = UsdGeom.Camera.Define(self.stage, Sdf.Path("/World/Rig/IRCamera"))
        self.eo_cam.CreateClippingRangeAttr(Gf.Vec2f(1.0, float(cfg.far_clip_m)))
        self.ir_cam.CreateClippingRangeAttr(Gf.Vec2f(1.0, float(cfg.far_clip_m)))

        eo_res = cfg.native_res("eo")
        ir_res = cfg.native_res("ir")
        self.eo_rp = rep.create.render_product("/World/Rig/EOCamera", eo_res)
        self.ir_rp = rep.create.render_product("/World/Rig/IRCamera", ir_res)

        self.eo_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.eo_rgb_annot.attach([self.eo_rp])

        self.ir_depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.ir_norm_annot = rep.AnnotatorRegistry.get_annotator("normals")
        self.ir_depth_annot.attach([self.ir_rp])
        self.ir_norm_annot.attach([self.ir_rp])

        self._actor_prims: dict[int, object] = {}
        self._last_step_key: tuple | None = None

    def _sync_camera(self, cam, ptz: PTZState, optics: Optics) -> None:
        Gf = self._Gf
        UsdGeom = self._UsdGeom
        cam.GetHorizontalApertureAttr().Set(float(optics.sensor_width_mm))
        cam.GetFocalLengthAttr().Set(float(optics.focal_mm))

        # USD camera looks down -Z with +Y up. For world Z-up and look yaw/pitch:
        # rotate X by (90 deg + pitch), then Z by (yaw - 90 deg).
        rot_x = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0 + float(np.degrees(ptz.pitch_rad)))
        rot_z = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(np.degrees(ptz.yaw_rad)) - 90.0)
        rot = rot_x * rot_z
        mat = Gf.Matrix4d(1.0)
        mat.SetRotate(rot)
        mat.SetTranslate(Gf.Vec3d(*[float(v) for v in ptz.cam_xyz]))
        xform = UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(mat)

    def _sync_actors(self, actors: Sequence[RigActor]) -> None:
        Gf = self._Gf
        UsdGeom = self._UsdGeom
        for actor in actors:
            prim_path = f"/World/Targets/Actor_{actor.track_id}"
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                prim = self.stage.DefinePrim(prim_path, "Xform")
                mesh_file = actor.asset.mesh_path()
                if mesh_file.exists():
                    prim.GetReferences().AddReference(str(mesh_file))
                else:
                    cube = UsdGeom.Cube.Define(self.stage, f"{prim_path}/Geom")
                    cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in actor.xyz]))
            xform.AddScaleOp().Set(Gf.Vec3f(float(actor.span_m), float(actor.span_m), float(actor.span_m)))

    def _step_if_needed(
        self, ptz: PTZState, eo_optics: Optics, ir_optics: Optics, actors: Sequence[RigActor]
    ) -> None:
        key = (ptz.cam_xyz, ptz.yaw_rad, ptz.pitch_rad, tuple((a.track_id, a.xyz) for a in actors))
        if key == self._last_step_key:
            return
        self._sync_camera(self.eo_cam, ptz, eo_optics)
        self._sync_camera(self.ir_cam, ptz, ir_optics)
        self._sync_actors(actors)
        self._rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        self._last_step_key = key

    def read_eo(
        self, ptz: PTZState, optics: Optics, actors: Sequence[RigActor], aovs: tuple[str, ...]
    ) -> tuple[np.ndarray, list[tuple[int, int, float, float, float, float]]]:
        assert aovs == EO_AOVS
        self._step_if_needed(ptz, optics, optics, actors)
        raw = self.eo_rgb_annot.get_data()
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3].astype(np.uint8)
        else:
            rgb = np.zeros((optics.height_px, optics.width_px, 3), dtype=np.uint8)
        placed = SyntheticRigBackend(self.ground)._rasterize_actors(ptz, optics, actors)
        return rgb, SyntheticRigBackend._boxes_from_placed(placed, optics)

    def read_ir(
        self, ptz: PTZState, optics: Optics, actors: Sequence[RigActor], aovs: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float, float, float, float]]]:
        assert aovs == IR_AOVS
        self._step_if_needed(ptz, optics, optics, actors)
        h, w = optics.height_px, optics.width_px

        raw_depth = np.asarray(self.ir_depth_annot.get_data(), dtype=np.float32)
        if raw_depth.shape[:2] == (h, w):
            depth = raw_depth.reshape(h, w)
        else:
            depth = np.full((h, w), np.inf, dtype=np.float32)

        raw_norm = np.asarray(self.ir_norm_annot.get_data(), dtype=np.float32)
        if raw_norm.ndim == 3 and raw_norm.shape[:2] == (h, w) and raw_norm.shape[2] >= 3:
            normals = raw_norm[..., :3].copy()
        else:
            normals = np.zeros((h, w, 3), dtype=np.float32)
            normals[..., 2] = 1.0

        is_ground = np.isfinite(depth) & (depth < self.cfg.far_clip_m)
        seg = np.where(is_ground, SEG_GROUND, SEG_SKY).astype(np.int32)

        helper = SyntheticRigBackend(self.ground)
        placed = helper._rasterize_actors(ptz, optics, actors)
        for actor, x1, y1, x2, y2, d_m in placed:
            seg[y1:y2, x1:x2] = actor.effective_seg_id
            depth[y1:y2, x1:x2] = d_m
            normals[y1:y2, x1:x2] = (0.0, 0.0, 1.0)
        return seg, normals, depth, helper._boxes_from_placed(placed, optics)


# --- Demo CLI --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Milestone B3 dual-camera rig smoke demo.")
    parser.add_argument("--demo", action="store_true", help="render a 10-frame EO/IR rig demo")
    parser.add_argument("--out", default="/tmp/rig_demo.png")
    args = parser.parse_args()

    if not args.demo:
        parser.error("pass --demo to run the local smoke test")

    cfg = DatagenConfig(sensor_tier="tier1")
    eo_opt = camera_optics(cfg, "eo", focal_mm=80.0)
    ir_opt = camera_optics(cfg, "ir", focal_mm=80.0)

    z = np.full((64, 64), 120.0, dtype=np.float32)
    ground = GroundZ(z=z, half_extent_m=5000.0)
    ptz = PTZState(cam_xyz=(0.0, 0.0, 140.0), yaw_rad=0.0, pitch_rad=0.04)

    rng = np.random.default_rng(7)
    spawn = spawn_in_frustum(ground, ptz, ir_opt, bucket_m=(300.0, 800.0), bg_mode="sky", rng=rng)
    asset = load_catalog()["meshy_tail_fiber_fpv"]
    backend = SyntheticRigBackend(ground)

    pos = np.array(spawn.xyz, dtype=np.float64)
    vel = np.array([-8.0, 14.0, 1.5], dtype=np.float64)
    dt_s = 1.0 / cfg.fps
    in_crop_count = 0
    last_frame: RigFrame | None = None

    for step in range(10):
        actor = RigActor(track_id=1, asset=asset, span_m=0.75, xyz=(float(pos[0]), float(pos[1]), float(pos[2])))
        last_frame = render_rig_frame(
            backend,
            ptz=ptz,
            eo_optics=eo_opt,
            ir_optics=ir_opt,
            actors=(actor,),
            sun=SunState(elevation_deg=25.0),
            rng=rng,
        )
        in_crop_count += int(last_frame.in_crop)
        pos = pos + vel * dt_s
        ptz = step_ptz(ptz, pos, dt_s, rng=rng)

    assert last_frame is not None
    from imageio.v3 import imwrite

    imwrite(args.out, last_frame.ir_u8)
    print(
        f"wrote {args.out}  ir_shape={last_frame.ir_u8.shape}  "
        f"eo_shape={last_frame.eo_rgb.shape}  "
        f"spawn_dist={spawn.distance_m:.1f} m  agl={spawn.agl_m:.1f} m  "
        f"in_crop={in_crop_count}/10  peak_T={last_frame.ir_temp_k.max():.2f} K"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
