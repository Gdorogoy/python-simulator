"""DatagenConfig — single source of truth for the synthetic EO/IR dataset.

See `app/datagen/docs.md` §0.1 (reconciled values) and §3 (camera rig) / §5 (output).
Imported, never copied, so the doc and the code cannot silently drift apart — the same rule
`app/training/config.py` follows for the PPO hyperparameters.
"""

from dataclasses import dataclass, field

# Pinned by docs.md §0.1. Frame counts are derived from these, never configured separately.
FPS_BY_TIER: dict[str, int] = {"tier1": 60, "tier2": 30}

# Contiguous and non-overlapping. A gap here is a bug — see app/test/test_datagen_config.py.
# The 300–800 m bucket is the close range; it is what the Tier 1 minimum range was lowered for.
RANGE_BUCKETS_M: tuple[tuple[float, float], ...] = (
    (300.0, 800.0),
    (800.0, 1200.0),
    (1200.0, 2000.0),
    (2000.0, 3500.0),
    (3500.0, 5000.0),
)

# Share of CLIP_COUNT per bucket, same order as RANGE_BUCKETS_M. Equal shares by default;
# raise the long-range weights if hard cases need oversampling (docs.md §5.3).
BUCKET_WEIGHTS: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20, 0.20)

CLIP_COUNT = 20_000

CLASS_IDS: dict[str, int] = {"quad": 0, "fixed_wing": 1, "decoy": 2, "bird": 3}

# Native sensor rasters, keyed by (tier, modality). docs.md §3.2/§3.3.
NATIVE_RES: dict[tuple[str, str], tuple[int, int]] = {
    ("tier1", "eo"): (1920, 1080),
    ("tier1", "ir"): (640, 512),
    ("tier2", "eo"): (2560, 1440),
    ("tier2", "ir"): (1280, 1024),
}

# Square centre crop used for training, keyed by (tier, modality). Always <= min(NATIVE_RES):
# cropping keeps mrad/px, resizing does not, so Tier 1 IR stays at 512 rather than upscaling.
TRAIN_CROP_PX: dict[tuple[str, str], int] = {
    ("tier1", "eo"): 1024,
    ("tier1", "ir"): 512,
    ("tier2", "eo"): 1024,
    ("tier2", "ir"): 1024,
}

# AOV split (docs.md §3.5): the IR compositor needs full rasters; EO needs only the picture and
# its labels. Requesting masks/normals/depth at 2560x1440 would cost 5x the readback for nothing.
EO_AOVS: tuple[str, ...] = ("rgb", "bounding_box_2d_tight", "camera_params")
IR_AOVS: tuple[str, ...] = (
    "instance_segmentation",
    "normals",
    "distance_to_image_plane",
    "motion_vectors",
    "bounding_box_2d_tight",
    "camera_params",
)

# Actual threat band: fiber-optic FPV, tip-to-tip including propellers (docs.md §2).
QUAD_SPAN_M: tuple[float, float] = (0.45, 1.0)

# Optional close-range rotor-detail capture only (docs.md §3.1); never a dataset-wide rate.
HIGH_RATE_CAPTURE_HZ = 240


@dataclass
class DatagenConfig:
    # world
    map_half_extent_m: float = 5000.0  # X, Y in [-5000, +5000]
    ceiling_m: float = 3000.0  # Z in [0, 3000]
    grid_vertices: int = 1000  # 1000 x 1000, 10 m spacing
    relief_min_m: float = 0.0  # wadi floor
    relief_max_m: float = 850.0  # ridge peaks
    landmarks_per_clip: tuple[int, int] = (4, 8)
    scatter_per_sector: tuple[int, int] = (5_000, 20_000)

    # clip
    duration_s: tuple[float, float] = (5.0, 10.0)
    headless: bool = True
    high_rate_capture: bool = False  # 240 Hz, close-range detail only

    # sensor
    sensor_tier: str = "tier1"  # "tier1" | "tier2"
    far_clip_m: float = 12_000.0
    tier1_range_m: tuple[float, float] = (300.0, 2000.0)
    tier2_range_m: tuple[float, float] = (2000.0, 5000.0)

    # environment
    # Dict defaults need default_factory: a bare {...} is a mutable default and raises
    # ValueError at class creation. Field order is (lo, hi) everywhere, matching the rest.
    background_mix: dict[str, float] = field(
        default_factory=lambda: {"sky": 0.4, "horizon": 0.3, "ground": 0.3}
    )
    visibility_m: tuple[int, int] = (3_000, 20_000)
    netd_mk: float = 50.0
    shutter_s: tuple[float, float] = (
        1 / 1000,
        1 / 25,
    )  # (fastest, slowest) before clamping

    # output
    write_yolo: bool = True
    write_metadata: bool = True

    def __post_init__(self) -> None:
        if self.sensor_tier not in FPS_BY_TIER:
            raise ValueError(
                f"unknown sensor_tier {self.sensor_tier!r}; expected one of {sorted(FPS_BY_TIER)}"
            )
        lo, hi = self.shutter_s
        if lo <= 0.0 or hi < lo:
            raise ValueError(
                f"shutter_s must be (fastest, slowest) and positive, got {self.shutter_s!r}"
            )

    @property
    def fps(self) -> int:
        return FPS_BY_TIER[self.sensor_tier]

    @property
    def shutter_max_s(self) -> float:
        """Longest legal exposure: one frame period (docs.md §3.1 shutter law)."""
        return 1.0 / self.fps

    @property
    def frames_per_clip(self) -> tuple[int, int]:
        lo, hi = self.duration_s
        return round(lo * self.fps), round(hi * self.fps)

    @property
    def range_m(self) -> tuple[float, float]:
        return self.tier1_range_m if self.sensor_tier == "tier1" else self.tier2_range_m

    def native_res(self, modality: str) -> tuple[int, int]:
        """(width, height) of the rendered raster for `modality` in this tier."""
        return NATIVE_RES[(self.sensor_tier, modality)]

    def train_crop_px(self, modality: str) -> int:
        """Square centre-crop size used for training; never larger than the native raster."""
        return TRAIN_CROP_PX[(self.sensor_tier, modality)]

    def effective_shutter_s(self, sampled_s: float) -> float:
        """Clamp a sampled exposure to the frame period — 1/25 s cannot exist inside a 1/60 s frame."""
        return min(sampled_s, self.shutter_max_s)
