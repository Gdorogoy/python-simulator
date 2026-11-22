# Data Generation Specification (`DatagenConfig`)

Synthetic EO/IR dataset for border Counter-UAS detection. Goal: clips that match real
camera physics (optics, SNR, motion).

**Two-tier camera setup this is calibrated to:**

| Tier | Role | Slant range |
| --- | --- | --- |
| Tier 1 | Close / mid tracker | `800 m – 2 km` |
| Tier 2 | Heavy-duty PTZ searcher | `2 km – 5 km` |

**Where the original spec and the project notes disagree, the project notes win.** The
reconciled values are used throughout this document and are marked where relevant.

---

## 0. Non-negotiables

* Every clip is `5–10 s` long.
* Every clip is rendered on the **same map** — only spawn position, sector, lighting, and
  scatter change between clips.
* All geometry stays inside `X, Y ∈ [-5000, +5000] m`, `Z ∈ [0, 3000] m`. 
* Terrain is pre-built **once**, never rebuilt per clip (per-clip rebuild costs several
  seconds per clip and is the main throughput killer).
* Every clip ships with machine-readable ground truth — no `.mp4`-only runs.
* Temporal motion (`5–10 s`) is what makes `2–5 km` classification solvable; never emit
  single-frame samples for the long-range tier.

---

## 1. World & Terrain

### 1.1 Map bounds & resolution

| Parameter | Spec / Range | Why it matters |
| --- | --- | --- |
| **Map bounds** | `X, Y ∈ [-5000, +5000] m`, `Z ∈ [0, 3000] m` | Centered at origin to avoid `float32` jitter; `3000 m` ceiling covers high-altitude glide approaches. |
| **Grid resolution** | `1000 × 1000` vertices (`10 m` spacing) | Keeps PhysX/raycast collision fast while giving crisp ridgelines at up to `5 km` slant range. |
| **Elevation relief** | `0 m` (wadi floor) → `850 m` (ridge peaks) | Mimics Galilee / southern Lebanon limestone topography so targets appear against **both sky and cluttered ground**. |
| **Static landmarks** | `4–8` ridge-top compounds / towers / roads **per clip** | Anchors realistic background thermal clutter (concrete retains heat into dusk/night). |
| **Dynamic scatter** | `UsdGeom.PointInstancer` (trees, scrub, rocks) | Instanced per sector — e.g. `5,000–20,000` olive/pine bushes & boulders in a single draw call. **No collision** — airborne clips never touch the ground. |

`10 m` spacing is deliberate: IR is `640 px` at best and needs far less geometric detail
than EO, so `1000 × 1000` is already oversampled for the thermal tier while staying cheap
for PhysX. Do not raise it for IR-quality reasons.

### 1.2 Pre-generation strategy

Do **not** regenerate terrain per clip. Build the map once and reuse it:

* Generate **one** `10 km × 10 km` map with distinct regional zones — rocky ridge,
  terraced valley, wadi, open plateau.
* (Optional, if variety is needed later) pre-bake `3–5` terrain variants and pick one per
  run, not per clip.
* Randomize **per clip**: sector coordinates, sun/lighting state, dynamic scatter
  placement and density.

### 1.3 Heightmap cache

`terrain_builder.py` also writes `ground_z.npy` — a `1000 × 1000` NumPy heightmap — so
ground-clearance and AGL checks are `O(1)` lookups instead of raycasts during generation.

---

## 2. Entities

Four classes. Each needs distinct **3D geometry, kinematics, and thermal contrast** so
YOLO/tracking learns real discriminative features at both `800 m–2 km` and `2 km–5 km`.

| Class | Variants & scale | Speed & flight profile | EO / IR signature |
| --- | --- | --- | --- |
| **Quad (Threat)** | Micro `0.25–0.35 m`, Tactical `0.45–0.70 m`, Heavy `1.0–1.5 m` | `5–50 m/s`; hover, dash, max-speed run, sharp banked turns, deep/terminal dive, terrain-following at `15–300 m` AGL | **EO:** dark matte frame, spinning rotor blur disks, optional strobes.<br>**IR:** rotors and mid-body are **white-hot**, everything else dark. At `<1.5 km` the 4 motor/rotor hotspots resolve individually; at `2–5 km` they merge into a single hot blob. |
| **Fixed-wing** | `1.2–2.5 m` wingspan, delta / pusher | `20–60 m/s`; smooth shallow-bank sweeps, constant-descent dive | **EO:** high-aspect wing silhouette.<br>**IR:** dark wings, **bright exhaust** (single rear motor/exhaust hotspot), semi-bright nose, cold wingtips. |
| **Decoy** | `0.5–1.5 m` wingspan balloon/debris, `0.2–0.4 m` bird-sized units; `1–15` per scene | `0–15 m/s`; wind drift, free-floating, road-bound vehicles | **IR:** bright point at center, darkening outward. Prevents false positives on slow wind-blown objects and hot vehicle engines near the horizon. |
| **Bird** | `0.25–1.0 m` span; single, or flock of `3–15` | `5–10 m/s`; cyclic sinusoidal vertical bobbing, wing-flap animation `2–6 Hz`, thermal soaring circles | **EO:** articulated wing-flap cycles, organic silhouette.<br>**IR:** slightly bright overall — uniform warm torso core, cooler flapping wings (pulsing thermal area), **no localized motor spikes**. |

**Class-separation rule:** the discriminative cues are (a) hotspot count/location,
(b) wing articulation, (c) speed band, (d) vertical motion signature. If a render cannot
show at least two of these at the target range, the clip is not useful — regenerate it.

---

## 3. Camera Rig, Modalities & Clip Structure

A single static frame cannot separate a distant drone from a bird at `2–5 km`. **Temporal
motion over `5–10 s` plus calibrated optics** is what makes the task solvable.

### 3.1 Clip format

* **Length:** `5.0–10.0 s`.
* **Frame rate:** `25–50 Hz` → `125–500` frames per clip, matching native sensor rates.
* **Both modalities per clip:** EO (RGB) and IR (LWIR `8–14 µm`), temporally aligned.

### 3.2 Tier 1 — Close / Mid Tracker (`800 m – 2 km`)

Hardware reference: **Vision Datum iVaster640LWPro FL100mm** (shutterless).

| | Spec |
| --- | --- |
| **EO (RGB)** | Paired, co-aligned to IR. `1920×1080 @ 50 Hz`, `~4.4°–15°` FOV. |
| **IR (LWIR 8–14 µm)** | `640×512 @ 50 Hz`, `12 µm` pitch (`7.68 × 6.144 mm` sensor), **`100 mm` fixed lens**, `4.40° × 3.52°` FOV, `0.12 mrad/px`, `NETD ≤ 50 mK`, `1–4×` digital zoom. |
| **Slant range** | `800 m – 2,000 m`. |
| **Mount / gimbal** | Agile gimbal or airborne interceptor mount; `0.03°` wind / airframe jitter. |
| **Notes** | Zero NUC shutter freeze (shutterless) — do not simulate a calibration shutter. |

### 3.3 Tier 2 — Long-Range Searcher (`2 km – 5 km`)

Hardware reference: **HZSoar Heavy-Duty Long-Range Thermal PTZ** (60 kg, 10 km LRF).

| | Spec |
| --- | --- |
| **EO (RGB)** | `2560×1440 @ 25 Hz` (or `2688×1520 @ 30 Hz`), `1/1.8"` CMOS, **`10–1200 mm` 120× zoom** (`38.4°` wide scan → `0.34°` narrow ID), optical defog + EIS. |
| **IR (LWIR 8–14 µm)** | `1280×1024 @ 25–50 Hz`, `12 µm` pitch (`15.36 × 12.288 mm` sensor), **`25–225 mm` continuous zoom** (`34.2°` wide scan → `3.91° × 3.13°` narrow tele, `0.053 mrad/px` at `225 mm`), `NETD ≤ 50 mK`. |
| **Slant range** | `2,000 m – 5,000 m`. |
| **Mount** | Ridge-tower mounted, `5–20 m` AGL. |
| **PTZ dynamics** | Harmonic drive: `0.003°` pan / `0.001°` tilt precision, `0.05°–150°/s` slew. |
| **Behavior** | Simulates patrol scan → zoom-in handoff, plus a `1 m`-accurate LRF range readout. |

Both tiers: **far clipping plane `12,000 m`**.

### 3.4 Spawn cone (critical for narrow FOVs)

At `3.91°–4.40°` narrow HFOV, spawning targets uniformly across a `10 × 10 km` map leaves
**99 % of frames empty**. Instead:

1. Sample slant range `R ∈ [800, 2000] m` (Tier 1) or `R ∈ [2000, 5000] m` (Tier 2).
2. Spawn the target **inside or entering** the camera's forward cone, with a
   crossing / approaching velocity vector.
3. Drive the PTZ with a realistic **rate-limited PID tracker**, including occasional
   scan-sweep misses and re-acquisition (so the model sees partial/absent detections,
   not a perfectly centered target every frame).

---

## 4. Environment, Weather & Lighting Randomization

| Variable | Randomization range | Effect on EO & IR |
| --- | --- | --- |
| **Time of day / low light** | Dawn, harsh midday sun, golden hour, dusk, night (`0.0005 Lux` color / `0.0001 Lux` B&W IR-cut mode) | Sun elevation, long ridge shadows, EO day/night B&W switchover, and **thermal crossover** (ground rock and air equalize at dawn/dusk). |
| **Atmospheric visibility** | Clear (`20 km`), dusty/haze (`3–8 km`), light fog (with and without **optical defog** contrast curve) | Exponential distance fog / scattering degrades EO contrast past `1.5–2 km`, forcing the model onto LWIR (`8–14 µm`) cues. |
| **Background contrast** | Sky `40 %`, horizon line `30 %`, ground clutter `30 %` | Horizon-line and ground-clutter clips at `2–5 km` are the hardest cases — **explicitly oversample them**. |
| **Sensor imperfections** | Motion blur (`1/25 s – 1/1000 s` shutter), rolling shutter / compression, `≤ 50 mK` NETD thermal noise, FPN, digital zoom (`1–4×` / `1–8×`) crop artifacts | Prevents memorizing razor-sharp synthetic pixels, especially on sub-10-pixel distant targets. |

---

## 5. Ground-Truth Output Per Clip

Never ship `.mp4`-only. Every run writes a self-contained clip directory:

```
clips/<clip_id>/
  eo.mp4            # H.264 high bitrate, native EO res (1920×1080 or 2560×1440)
  ir.mp4            # H.264 high bitrate, native IR res (640×512 or 1280×1024)
  labels_yolo/
    frame_0000.txt  # class_id cx cy w h track_id
    ...
    frame_XXXX.txt
  metadata.json
```

* **`labels_yolo/`** — normalized `class_id cx cy w h` plus `track_id` for multi-object
  tracking (ByteTrack / BoT-SORT). One file per frame, empty file = no target.
* **`metadata.json`** — per clip and per frame:
  * camera preset: `ivaster_fl100` | `hzsoar_ptz`
  * active focal length (`mm`), digital zoom factor
  * simulated LRF slant range (`m`)
  * per-frame 3D world position, groundspeed, aspect angle, occlusion `%`
  * weather + lighting tags

Range buckets are part of the schema so validation metrics can be sliced exactly:

```
800–1200 m | 1200–2000 m | 2000–3500 m | 3500–5000 m
```

---

## 6. `DatagenConfig` Sketch

```python
@dataclass
class DatagenConfig:
    # world
    map_half_extent_m: float = 5000.0     # X, Y in [-5000, +5000]
    ceiling_m: float = 3000.0             # Z in [0, 3000]
    grid_vertices: int = 1000             # 1000 x 1000, 10 m spacing
    relief_min_m: float = 0.0             # wadi floor
    relief_max_m: float = 850.0           # ridge peaks
    landmarks_per_clip: tuple[int, int] = (4, 8)
    scatter_per_sector: tuple[int, int] = (5_000, 20_000)

    # clip
    duration_s: tuple[float, float] = (5.0, 10.0)
    fps: int = 25                         # 25 or 50
    headless: bool = True

    # sensor
    sensor_tier: Literal["tier1", "tier2"] = "tier1"
    far_clip_m: float = 12_000.0
    tier1_range_m: tuple[int, int] = (800, 2000)
    tier2_range_m: tuple[int, int] = (2000, 5000)

    # environment
    background_mix: dict[str, float] = {"sky": 0.4, "horizon": 0.3, "ground": 0.3}
    visibility_m: tuple[int, int] = (3_000, 20_000)
    netd_mk: int = 50
    shutter_s: tuple[float, float] = (1 / 1000, 1 / 25)

    # output
    write_yolo: bool = True
    write_metadata: bool = True
```

---

## 7. Build Order (`app/datagen/`)

1. **`terrain_builder.py`** — offline script. Generates the `10 × 10 × 3 km` ridge/wadi
   `.usd` mesh plus the `1000 × 1000` NumPy heightmap (`ground_z.npy`) for `O(1)`
   ground-clearance checks. Run once, not per clip.
2. **`assets.py`** — builds/loads drone USDs with thermal material attributes on motor
   bells and battery; procedural articulated bird USD with wings driven by a sine wave in
   the step loop; decoy + fixed-wing assets.
3. **`rig.py`** — spawns Tier 1 (`ivaster_fl100`) and Tier 2 (`hzsoar_ptz`) camera rigs via
   `omni.replicator.core` render products (RGB + thermal AOV + 2D tight bounding boxes),
   far clip `12,000 m`, harmonic-drive PTZ dynamics.
4. **`generate.py`** — CLI entrypoint:
   `--sensor tier1|tier2 --clips 50 --duration 6 --fps 25 --headless`
   Steps physics/kinematics and writes `eo.mp4`, `ir.mp4`, YOLO labels, `metadata.json`.

---

## 8. Open Items

* Decide whether the `3–5` pre-baked terrain variants are needed at all, or whether
  sector + lighting randomization on the single map is enough.
* Define the per-class `track_id` assignment policy across occlusions and re-acquisition.
* Pin the exact defog contrast curve used for Tier 2 EO so validation stays comparable
  across runs.
