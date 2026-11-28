"""Acceptance tests for docs.md B3 — camera optics, frustum spawn, PTZ slew, and dual-pass frame."""

import numpy as np
import pytest

from app.datagen.assets import load_catalog
from app.datagen.config import RANGE_BUCKETS_M, DatagenConfig
from app.datagen.export_yolo import remap_box
from app.datagen.ir_composite import SunState
from app.datagen.rig import (
    PTZState,
    RigActor,
    SyntheticRigBackend,
    camera_optics,
    look_angles,
    project_world_to_pixel,
    render_rig_frame,
    spawn_in_frustum,
    step_ptz,
)
from app.datagen.terrain_builder import GroundZ


@pytest.fixture(scope="module")
def flat_ground():
    return GroundZ(z=np.full((32, 32), 100.0, dtype=np.float32), half_extent_m=5000.0)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_camera_optics_zoom_narrows_fov_and_ifov():
    cfg = DatagenConfig(sensor_tier="tier2")
    wide = camera_optics(cfg, "eo", focal_mm=35.0)
    tele = camera_optics(cfg, "eo", focal_mm=350.0)

    assert wide.width_px == 2560 and wide.height_px == 1440
    assert wide.crop_px == 1024
    assert tele.fov_x_rad < wide.fov_x_rad
    assert tele.ifov_mrad < wide.ifov_mrad
    assert tele.focal_px > wide.focal_px
    assert wide.crop_fov_rad <= wide.fov_x_rad


def test_camera_optics_rejects_non_positive_focal():
    cfg = DatagenConfig(sensor_tier="tier1")
    with pytest.raises(ValueError, match="focal_mm"):
        camera_optics(cfg, "ir", focal_mm=0.0)


@pytest.mark.parametrize("bucket_m", RANGE_BUCKETS_M)
def test_spawn_in_frustum_lands_inside_crop_window_and_above_min_agl(flat_ground, bucket_m):
    cfg = DatagenConfig(sensor_tier="tier2")
    ir_opt = camera_optics(cfg, "ir", focal_mm=100.0)
    ptz = PTZState(cam_xyz=(0.0, 0.0, 130.0), yaw_rad=0.25, pitch_rad=0.06)
    rng = np.random.default_rng(11)

    spawn = spawn_in_frustum(flat_ground, ptz, ir_opt, bucket_m, bg_mode="sky", min_agl_m=15.0, rng=rng)
    assert spawn.agl_m >= 15.0 - 1e-6
    assert bucket_m[0] <= spawn.distance_m <= bucket_m[1] * 1.05

    proj = project_world_to_pixel(ptz, ir_opt, spawn.xyz)
    assert proj is not None
    u_px, v_px, _ = proj
    assert ir_opt.ox <= u_px < ir_opt.ox + ir_opt.crop_px
    assert ir_opt.oy <= v_px < ir_opt.oy + ir_opt.crop_px


def test_step_ptz_slews_toward_target_within_rate_limit():
    ptz = PTZState(
        cam_xyz=(0.0, 0.0, 120.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        max_rate_rad_s=0.5,
        tau_lag_s=0.1,
        jitter_rad=0.0,
    )
    target = (500.0, 250.0, 160.0)
    want_yaw, want_pitch, _ = look_angles(ptz.cam_xyz, target)

    dt_s = 0.05
    moved = step_ptz(ptz, target, dt_s)
    assert 0.0 < moved.yaw_rad <= ptz.max_rate_rad_s * dt_s + 1e-9
    assert 0.0 < moved.pitch_rad <= ptz.max_rate_rad_s * dt_s + 1e-9
    assert abs(want_yaw - moved.yaw_rad) < abs(want_yaw - ptz.yaw_rad)
    assert abs(want_pitch - moved.pitch_rad) < abs(want_pitch - ptz.pitch_rad)


def test_render_rig_frame_applies_plus_k_hotspots_and_keeps_box_in_crop(flat_ground, catalog):
    cfg = DatagenConfig(sensor_tier="tier1")
    eo_opt = camera_optics(cfg, "eo", focal_mm=80.0)
    ir_opt = camera_optics(cfg, "ir", focal_mm=80.0)
    ptz = PTZState(cam_xyz=(0.0, 0.0, 130.0), yaw_rad=0.0, pitch_rad=0.05)

    spawn = spawn_in_frustum(
        flat_ground, ptz, ir_opt, bucket_m=(300.0, 800.0), bg_mode="sky", rng=np.random.default_rng(3)
    )
    quad = RigActor(track_id=7, asset=catalog["meshy_tail_fiber_fpv"], span_m=0.8, xyz=spawn.xyz)
    decoy = RigActor(track_id=8, asset=catalog["decoy_balloon_reflector"], span_m=0.8, xyz=spawn.xyz)
    backend = SyntheticRigBackend(flat_ground)

    quad_frame = render_rig_frame(
        backend,
        ptz=ptz,
        eo_optics=eo_opt,
        ir_optics=ir_opt,
        actors=(quad,),
        sun=SunState(elevation_deg=20.0),
        netd_k=0.0,
        rng=np.random.default_rng(0),
    )
    decoy_frame = render_rig_frame(
        backend,
        ptz=ptz,
        eo_optics=eo_opt,
        ir_optics=ir_opt,
        actors=(decoy,),
        sun=SunState(elevation_deg=20.0),
        netd_k=0.0,
        rng=np.random.default_rng(0),
    )

    assert quad_frame.eo_rgb.shape == (1080, 1920, 3)
    assert quad_frame.ir_u8.shape == (512, 640)
    assert quad_frame.in_crop is True
    assert len(quad_frame.boxes) == 1
    _, cls_id, cx, cy, w, h = quad_frame.boxes[0]
    assert cls_id == 0
    assert remap_box(cx, cy, w, h, window=ir_opt.window) is not None

    # The quad's +33 K motor hotspots make its peak IR temperature hotter than the +8 K decoy
    assert quad_frame.ir_temp_k.max() > decoy_frame.ir_temp_k.max() + 10.0
    assert int(quad_frame.ir_u8.max()) > int(decoy_frame.ir_u8.max())
