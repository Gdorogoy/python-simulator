# Data Generation Specification (`DatagenConfig`)

Synthetic EO/IR dataset for border Counter-UAS detection. Goal: clips that match real
camera physics (optics, SNR, motion).

**Two-tier camera setup this is calibrated to:**

| Tier   | Role                    | Slant range   |
| ------ | ----------------------- | ------------- |
| Tier 1 | Close / mid tracker     | `300 m – 2 km`  |
| Tier 2 | Heavy-duty PTZ searcher | `2 km – 5 km` |

<!-- fix(#12): the "where the original spec and the project notes disagree, the project notes win"
     clause is gone — the project notes are not in the repo and nothing in this file was ever
     marked as reconciled. Every pinned figure now lives in §0.1. -->

---

## 0. Non-negotiables

- Every clip is `5–10 s` long.
- Every clip is rendered on the **same map** — only spawn position, sector, lighting, and
  scatter change between clips.
- All geometry stays inside `X, Y ∈ [-5000, +5000] m`, `Z ∈ [0, 3000] m`.
- Terrain is pre-built **once**, never rebuilt per clip (per-clip rebuild costs several
  seconds per clip and is the main throughput killer).
- Every clip ships with machine-readable ground truth — no `.mp4`-only runs.
- Temporal motion (`5–10 s`) is what makes `2–5 km` classification solvable; never emit
  single-frame samples for the long-range tier.
- <!-- fix(#11): carrying motion in the data is not enough — something has to consume it. -->
  Motion must be _consumed_ by the track-level stage in §5.2, not merely present in the pixels.

### 0.1 Reconciled values

Single source of truth. If a figure anywhere else in this document disagrees with this
table, the table wins.

<!-- fix(#12): new section, replaces the out-of-repo "project notes" rule. -->

| Item              | Value                                               | Source           |
| ----------------- | --------------------------------------------------- | ---------------- |
| Quad size band    | `0.45–1.0 m` tip-to-tip incl. propellers            | project decision |
| Tier 1 frame rate | `60 Hz` → `300–600` frames/clip                     | §3.2             |
| Tier 2 frame rate | `30 Hz` → `150–300` frames/clip                     | §3.3             |
| Shutter law       | `exposure ≤ 1 / fps` (clamped, never violated)      | §3.1             |
| Tier 1 slant range | `300 m – 2 km` (300–800 m feeds the close bucket)  | §3.2             |
| Range buckets     | `300–800 \| 800–1200 \| 1200–2000 \| 2000–3500 \| 3500–5000 m` | §5.3             |
| Class ids         | `0 quad, 1 fixed_wing, 2 decoy, 3 bird`             | §5.1             |
| Label sharing     | one label set serves EO and IR                      | §5.1             |
| Terrain           | one prebuilt `10×10 km` map, no per-clip rebuild    | §1.2             |
| Terrain variants  | none — sector + lighting randomisation only         | §8               |
| Training input    | square centre crop per sensor: EO `1024²`, Tier 1 IR `512²`, Tier 2 IR `1024²` — never resized | §3.5 |
| AOV split         | heavy AOVs on the IR camera only; EO gets `rgb` + boxes | §3.5 |
| IR source         | composited from AOVs, not a second render pass      | §3.6             |
| Dataset size      | `20,000` clips, `0.20` per range bucket             | §5.3             |

---

## 1. World & Terrain

### 1.1 Map bounds & resolution

| Parameter            | Spec / Range                                            | Why it matters                                                                                                                                           |
| -------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Map bounds**       | `X, Y ∈ [-5000, +5000] m`, `Z ∈ [0, 3000] m`            | Centered at origin to avoid `float32` jitter; `3000 m` ceiling covers high-altitude glide approaches.                                                    |
| **Grid resolution**  | `1000 × 1000` vertices (`10 m` spacing)                 | Keeps PhysX/raycast collision fast while giving crisp ridgelines at up to `5 km` slant range.                                                            |
| **Elevation relief** | `0 m` (wadi floor) → `850 m` (ridge peaks)              | Mimics Galilee / southern Lebanon limestone topography so targets appear against **both sky and cluttered ground**.                                      |
| **Static landmarks** | `4–8` ridge-top compounds / towers / roads **per clip** | Anchors realistic background thermal clutter (concrete retains heat into dusk/night).                                                                    |
| **Dynamic scatter**  | `UsdGeom.PointInstancer` (trees, scrub, rocks)          | Instanced per sector — e.g. `5,000–20,000` olive/pine bushes & boulders in a single draw call. **No collision** — airborne clips never touch the ground. |

`10 m` spacing is deliberate: IR is `640 px` at best and needs far less geometric detail
than EO, so `1000 × 1000` is already oversampled for the thermal tier while staying cheap
for PhysX. Do not raise it for IR-quality reasons.

### 1.2 Pre-generation strategy

Do **not** regenerate terrain per clip. Build the map once and reuse it:

- Generate **one** `10 km × 10 km` map with distinct regional zones — rocky ridge,
  terraced valley, wadi, open plateau.
- <!-- fix(#12): terrain variants are dropped, not "optional if needed later" — the open
       item in §8 is now closed. Randomisation per clip is sector + lighting + scatter only. -->
  No terrain variants. Variety comes from per-clip sector, lighting state and dynamic
  scatter, all of which reuse the same baked map.
- Randomize **per clip**: sector coordinates, sun/lighting state, dynamic scatter
  placement and density.

### 1.2.1 Terrain recipe

<!-- fix(#12): §1.1 gave parameters and a geography to imitate but no method; this is the
     actual recipe terrain_builder.py implements. -->

| Step            | Method                                                                                                      | Params                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Base relief     | multi-octave **ridged multifractal** noise                                                                  | `6` octaves, lacunarity `2.0`, gain `0.5`, base wavelength `4096 m` |
| Layout          | one dominant N–S limestone ridge, terraced plateau east, wadi floor along the centreline                    | peaks `600–850 m`, floor `0–40 m`                                   |
| Wadis           | **domain-warped spline channels** carved into the valley floor                                              | spline warp amplitude `±120 m`, channel depth `20–60 m`             |
| Terraces        | subtle quantization along mid-slopes (`20–38°`)                                                             | step `8–15 m`, applied only above the wadi floor                    |
| Erosion         | optional hydraulic pass, baked once (never per clip)                                                        | dropped if a full build exceeds `5 min`                             |
| Materials       | slope- and elevation-blended PBR: exposed limestone on steep faces, dry Mediterranean scrub/soil in valleys | `4` pairs (limestone, terra rossa, dry scrub, concrete)             |
| LWIR emissivity | per-material constant consumed by the IR compositor (§3.6)                                                  | limestone `0.92`, soil `0.95`, scrub `0.90`, concrete `0.88`        |
| Scatter         | instanced olive/pine (2 LODs) + boulders, no collision                                                      | `5,000–20,000` per sector                                           |
| Landmarks       | `4–8` per clip drawn from a prebuilt kit of `6` (compound, tower, road)                                     | —                                                                   |

Output is a single watertight USD triangle mesh (`map_10km.usd`) with vertex normals, UVs
and a **decimated** `UsdPhysics.CollisionAPI` collider — not the render mesh at ~2 M
triangles, which PhysX would chew on for no reason since airborne clips never land. The
same build writes `ground_z.npy` (§1.3).

### 1.3 Heightmap cache

`terrain_builder.py` also writes `ground_z.npy` — a `1000 × 1000` NumPy heightmap — so
ground-clearance and AGL checks are `O(1)` lookups instead of raycasts during generation.
Lookups are **bilinear**, not nearest-neighbour: at `10 m` grid spacing a nearest lookup
quantises clearance by up to `5 m`, which matters for a drone at `15 m` AGL.

<!-- fix(#12): interpolation rule added — nearest-neighbour would have silently quantised AGL. -->

---

## 2. Entities

Four classes. Each needs distinct **3D geometry, kinematics, and thermal contrast** so
YOLO/tracking learns real discriminative features at both `800 m–2 km` and `2 km–5 km`.

| Class             | Variants & scale                                                                                                                                                               | Speed & flight profile                                                                                              | EO / IR signature                                                                                                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Quad (Threat)** | <!-- fix(#10): narrowed from 0.25–1.5 m to the actual threat band. --> `0.45–0.70 m` (typical fiber-optic FPV), `0.70–1.0 m` (large), measured **tip-to-tip incl. propellers** | `5–50 m/s`; hover, dash, max-speed run, sharp banked turns, deep/terminal dive, terrain-following at `15–300 m` AGL | **EO:** dark matte frame, spinning rotor blur disks, optional strobes.<br>**IR:** rotors and mid-body are **white-hot**, everything else dark. At `<1.5 km` the 4 motor/rotor hotspots resolve individually; at `2–5 km` they merge into a single hot blob. |
| **Fixed-wing**    | `1.2–2.5 m` wingspan, delta / pusher                                                                                                                                           | `20–60 m/s`; smooth shallow-bank sweeps, constant-descent dive                                                      | **EO:** high-aspect wing silhouette.<br>**IR:** dark wings, **bright exhaust** (single rear motor/exhaust hotspot), semi-bright nose, cold wingtips.                                                                                                        |
| **Decoy**         | `0.5–1.5 m` wingspan balloon/debris, `0.2–0.4 m` bird-sized units; `1–15` per scene                                                                                            | `0–15 m/s`; wind drift, free-floating, road-bound vehicles                                                          | **IR:** bright point at center, darkening outward. Prevents false positives on slow wind-blown objects and hot vehicle engines near the horizon.                                                                                                            |
| **Bird**          | `0.25–1.0 m` span; single, or flock of `3–15`                                                                                                                                  | `5–10 m/s`; cyclic sinusoidal vertical bobbing, wing-flap animation `2–6 Hz`, thermal soaring circles               | **EO:** articulated wing-flap cycles, organic silhouette.<br>**IR:** slightly bright overall — uniform warm torso core, cooler flapping wings (pulsing thermal area), **no localized motor spikes**.                                                        |

Fiber-optic FPV specifics: the tether spool and line are physical, but the line is only
resolvable below ~`800 m`, so it belongs to the Tier 1 close-tracker tier only, if at all.
<!-- fix(#10): tether modelling is an open decision, recorded rather than assumed. -->

**Class-separation rule:** the discriminative cues are (a) hotspot count/location,
(b) wing articulation, (c) speed band, (d) vertical motion signature.
<!-- fix(#10): "two cues or regenerate" rejected nearly every long-range clip, because a
     0.45 m quad is ~1.7 px at 5 km in Tier 2 IR and shows one merged blob. The rule now
     degrades instead of rejecting, and the temporal cues are explicitly carried to §5.2. -->

If a render cannot show at least two of these at the target range, the clip is regenerated —
**except** in the long-range regime, where a single merged hot blob counts as cue (a) in
degraded form and cue (c)/(d) are recovered from the track (§5.2) instead of from one frame.

---

## 3. Camera Rig, Modalities & Clip Structure

A single static frame cannot separate a distant drone from a bird at `2–5 km`. **Temporal
motion over `5–10 s` plus calibrated optics** is what makes the task solvable.

### 3.1 Clip format

<!-- fix(#12): "25–50 Hz → 125–500 frames" was four different answers in one line. Rates are
     now pinned per tier, frame count is derived, and the shutter is clamped by the frame period. -->

- **Length:** `5.0–10.0 s`.
- **Frame rate, by tier:** Tier 1 `60 Hz` → `300–600` frames; Tier 2 `30 Hz` → `150–300` frames.
  `frames = round(duration × fps_tier)` is _derived_, never configured separately, so duration
  and rate cannot drift apart.
- **Shutter law:** exposure must satisfy `exposure ≤ 1 / fps`. The sampled shutter is clamped to
  the frame period (a `1/25 s` exposure is impossible inside a `1/60 s` frame).
- **Both modalities per clip:** EO (RGB) and IR (LWIR `8–14 µm`), temporally aligned — one frame
  index, so a single label set serves both (§5.1).
- **Optional high-rate capture** (`240 Hz`, off by default): close-range rotor-detail clips only,
  never a dataset-wide rate. It multiplies frames per clip by 8 and forces per-clip label storage
  (§5); enable it only after §7.1's measured throughput supports it.

### 3.2 Tier 1 — Close / Mid Tracker (`300 m – 2 km`)

Hardware reference: **Vision Datum iVaster640LWPro FL100mm** (shutterless).

|                       | Spec                                                                                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EO (RGB)**          | Paired, co-aligned to IR. `1920×1080 @ 60 Hz`, `~4.4°–15°` FOV.                                                                                                 |
| **IR (LWIR 8–14 µm)** | `640×512 @ 60 Hz`, `12 µm` pitch (`7.68 × 6.144 mm` sensor), **`100 mm` fixed lens**, `4.40° × 3.52°` FOV, `0.12 mrad/px`, `NETD ≤ 50 mK`, `1–4×` digital zoom. |
| **Slant range**       | `300 m – 2,000 m` (the `300–800 m` band feeds the close bucket; optics unchanged).                                                                              |
| **Mount / gimbal**    | Agile gimbal or airborne interceptor mount; `0.03°` wind / airframe jitter.                                                                                     |
| **Notes**             | Zero NUC shutter freeze (shutterless) — do not simulate a calibration shutter.                                                                                  |

<!-- fix(#12): 50 Hz → 60 Hz, matching the drone-mounted gimbal class. -->

### 3.3 Tier 2 — Long-Range Searcher (`2 km – 5 km`)

Hardware reference: **HZSoar Heavy-Duty Long-Range Thermal PTZ** (60 kg, 10 km LRF).

|                       | Spec                                                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **EO (RGB)**          | `2560×1440 @ 30 Hz`, `1/1.8"` CMOS, **`10–1200 mm` 120× zoom** (`38.4°` wide scan → `0.34°` narrow ID), optical defog + EIS.                                                                     |
| **IR (LWIR 8–14 µm)** | `1280×1024 @ 30 Hz`, `12 µm` pitch (`15.36 × 12.288 mm` sensor), **`25–225 mm` continuous zoom** (`34.2°` wide scan → `3.91° × 3.13°` narrow tele, `0.053 mrad/px` at `225 mm`), `NETD ≤ 50 mK`. |
| **Slant range**       | `2,000 m – 5,000 m`.                                                                                                                                                                             |
| **Mount**             | Ridge-tower mounted, `5–20 m` AGL.                                                                                                                                                               |
| **PTZ dynamics**      | Harmonic drive: `0.003°` pan / `0.001°` tilt precision, `0.05°–150°/s` slew.                                                                                                                     |
| **Behavior**          | Simulates patrol scan → zoom-in handoff, plus a `1 m`-accurate LRF range readout.                                                                                                                |

<!-- fix(#12): dropped the 2688×1520@30 alternative (one pinned raster per tier) and pinned IR
     at 30 Hz instead of "25–50". -->

**Search coverage.** The searcher covers ground by _scanning_, not by framing faster: the PTZ
walks a raster of sub-FOV cells inside its wide FOV with a per-cell dwell, e.g. a `6×4` grid at
`80 ms` dwell. Coverage per second comes from slew rate and cell size, not from frame rate —
raising fps on a static FOV buys nothing.
<!-- fix(#14): this is what "the searcher needs to be faster" actually means — a scan/dwell
     model, not a 240 Hz sensor. Hardware of this class does not run 1280×1024 LWIR at 240 Hz. -->

Both tiers: **far clipping plane `12,000 m`**.

### 3.4 Spawn cone (critical for narrow FOVs)

At `3.91°–4.40°` narrow HFOV, spawning targets uniformly across a `10 × 10 km` map leaves
**99 % of frames empty**. Instead:

1. Sample slant range `R ∈ [300, 2000] m` (Tier 1) or `R ∈ [2000, 5000] m` (Tier 2).
2. Spawn the target **inside or entering** the camera's forward cone, with a
   crossing / approaching velocity vector.
3. Drive the PTZ with a realistic **rate-limited PID tracker**, including occasional
   scan-sweep misses and re-acquisition (so the model sees partial/absent detections,
   not a perfectly centered target every frame).

### 3.5 Render path (pinned)

<!-- fix(#12): "resolution given per tier, render path not pinned" — pinned. Resolution is the
     physics, so it is never reduced; zoom is a crop; training input is a crop, never a resize. -->

Exactly one render pass per clip, at the native resolution of the active tier:

1. One `rep.create.render_product` per camera — `1920×1080` + `640×512` (Tier 1),
   `2560×1440` + `1280×1024` (Tier 2).
2. AOVs, split so the heavy buffers render only where they are needed:

   | Camera | AOVs |
   | --- | --- |
   | EO | `rgb`, `bounding_box_2d_tight`, `camera_params` |
   | IR | `instance_segmentation`, `normals`, `distance_to_image_plane`, `motion_vectors`, `bounding_box_2d_tight`, `camera_params` |

   <!-- fix(#14): the IR composite consumes masks/normals/depth at IR resolution. Requesting them
        at 2560×1440 costs 5× the pixels and 5× the GPU→CPU readback for zero extra information. -->
   Nothing in §3.6 needs EO-resolution masks, so the heavy AOVs are never requested on the EO
   camera. `bounding_box_2d_tight` and `camera_params` are per-object and per-camera metadata, not
   rasters — they cost next to nothing.
3. **EO** frames are `rgb` as rendered. **IR** frames are composited in NumPy from the AOVs
   (§3.6). There is no second render pass and no resampling between modalities.
4. **Zoom is a crop, not a resolution change.** Optical zoom changes the requested FOV at render
   time (it is a real lens); digital zoom crops the native raster. Both are recorded per frame in
   `metadata.json`.
5. **Training input** is a centre crop, square, sized per sensor — never a resize:

   | Sensor | Native | Train crop |
   | --- | --- | --- |
   | Tier 1 EO | `1920×1080` | `1024×1024` |
   | Tier 1 IR | `640×512` | `512×512` |
   | Tier 2 EO | `2560×1440` | `1024×1024` |
   | Tier 2 IR | `1280×1024` | `1024×1024` |

   <!-- fix(#14): one global 1024² crop was impossible for Tier 1 IR — 640×512 is smaller than
        1024², so "crop to 1024" silently meant upscaling, which is the resize the rule bans. -->
   Cropping preserves `mrad/px`, so the pixel-size arithmetic in §2/§5.1 still holds; resizing
   would not. The 16:9 rasters keep only their central region, which is why §5.1 requires the
   target to stay inside the crop.

### 3.6 IR (LWIR) compositing

<!-- fix(#9): Isaac Sim has no thermal/LWIR camera and no white-hot mode. IR is produced by
     compositing the AOVs against a per-part emissivity table, not by filtering the RGB image —
     filtering RGB would make a sunlit airframe or a white sky read as white-hot. -->

Isaac Sim renders visible light only, so IR is synthesised rather than rendered:

1. **Per-part emissivity.** Every mesh part carries an emissivity/elevated-temperature attribute
   (rotor bells and battery hot, airframe cold, exhaust hot on fixed-wings, wing membranes warm
   on birds). Values come from the same table as §1.2.1 for the ground and from `assets.py` for
   vehicles.
2. **Mask, don't filter.** The part list is resolved through `instance_segmentation` masks, so
   only genuinely hot parts emit. `normals` gives the ground/sky thermal background (sun-baked
   rock vs cooler wadi floor, with the dawn/dusk crossover term driven by sun elevation), and
   `distance_to_image_plane` applies atmospheric transmittance.
3. **Noise happens in temperature space.** NETD (`≤ 50 mK`), fixed-pattern noise and the
   drone-to-sensor gain are applied _before_ the display mapping, then the result is mapped to
   `8-bit` gray with a fixed-gain white-hot curve (stated gamma, clipped saturation). Fixed gain,
   no per-frame AGC: auto-gain would normalise target-vs-background contrast and destroy the
   absolute thermal cues the classes depend on.
4. **Motion blur from `motion_vectors`**, applied per-part so rotor disks blur and the merged
   blob at `2–5 km` integrates the right amount of energy.

This also means IR is free of RGB-specific artifacts (compression, EIS warping), which is correct
— the two sensors are physically different chains.

---

## 4. Environment, Weather & Lighting Randomization

| Variable                    | Randomization range                                                                                                                                                       | Effect on EO & IR                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Time of day / low light** | Dawn, harsh midday sun, golden hour, dusk, night (`0.0005 Lux` color / `0.0001 Lux` B&W IR-cut mode)                                                                      | Sun elevation, long ridge shadows, EO day/night B&W switchover, and **thermal crossover** (ground rock and air equalize at dawn/dusk).                  |
| **Atmospheric visibility**  | Clear (`20 km`), dusty/haze (`3–8 km`), light fog (with and without **optical defog** contrast curve)                                                                     | Exponential distance fog / scattering degrades EO contrast past `1.5–2 km`, forcing the model onto LWIR (`8–14 µm`) cues.                               |
| **Background contrast**     | Sky `40 %`, horizon line `30 %`, ground clutter `30 %`                                                                                                                    | Horizon-line and ground-clutter clips at `2–5 km` are the hardest cases — **explicitly oversample them**.                                               |
| **Sensor imperfections**    | Motion blur (`1/fps – 1/1000 s` shutter, clamp per §3.1), rolling shutter / compression, `≤ 50 mK` NETD thermal noise, FPN, digital zoom (`1–4×` / `1–8×`) crop artifacts | Prevents memorizing razor-sharp synthetic pixels, especially on sub-10-pixel distant targets.                                                           |
| **Heat shimmer**            | <!-- fix(#12): was fog-only. Whole-frame screen-space UV refraction from a 2D noise field; amplitude ∝ focal length and ground–air ΔT; off below `300 mm` focal. -->      | Distorts the whole frame, background included. Applied per object it would fake target motion and corrupt boxes, `track_id` and §5.2's motion features. |
| **Wind / platform jitter**  | Camera transform only: `0.03°` airframe jitter (Tier 1), harmonic-drive micro-jitter (Tier 2), amplitude scaled by wind                                                   | Camera-level, never object-level — jittering the target fakes kinematics the labels then contradict.                                                    |
| **Defog curve**             | Tier 2 EO only, `256`-point contrast LUT at `app/datagen/defog_lut.npz` (generated by the Tier 2 EO calibration, §8); file hash recorded in `metadata.json`               | Keeps across-run validation comparable instead of re-tuning defog per run.                                                                              |

**Transmittance.** One `β` per clip, derived from the sampled visibility, applied as `e^(-βd)`:
in Isaac for EO and, using the depth AOV, in the IR compositor for IR. The same `β` and the
scattering/absorption split are written to `metadata.json` so the two modalities stay physically
consistent.
<!-- fix(#12): "atmospherics limited to exponential fog, no shimmer/turbulence/scattering" —
     shimmer and platform jitter added, and the extinction term is now shared between EO and IR. -->

---

## 5. Ground-Truth Output Per Clip

Never ship `.mp4`-only. Every run writes a self-contained clip directory:

```
clips/<clip_id>/
  eo.mp4            # H.264 high bitrate, native EO res (1920×1080 or 2560×1440)
  ir.mp4            # H.264 high bitrate, native IR res (640×512 or 1280×1024)
  labels.npz        # arrays: frame, class_id, cx, cy, w, h, track_id, occluded
  metadata.json
```

<!-- fix(#14): one .txt per frame was 5M+ files at the pinned rates and 17M+ at 240 Hz. Labels
     live in one .npz per clip; `export_yolo.py` materialises per-frame .txt for Ultralytics
     when a training run actually needs it. -->

- **`labels.npz`** — columnar, one file per clip: `frame, class_id, cx, cy, w, h, track_id,
occluded`. Normalised `cx cy w h`. Empty arrays mean no labelled target in that frame.
  `export_yolo.py` (§7) converts to the conventional `labels/<frame>.txt` layout on demand.
- **`metadata.json`** — per clip and per frame:
  - camera preset: `ivaster_fl100` | `hzsoar_ptz`
  - active focal length (`mm`), digital zoom factor
  - simulated LRF slant range (`m`)
  - per-frame 3D world position, groundspeed, aspect angle, occlusion `%`
  - per-frame `track_state` (`active` | `lost` | `reacquired`) and `track_id`
  - weather + lighting tags, `β`, defog LUT hash
  - `unlabeled_targets`: targets too small to supervise (§5.1)

Range buckets are part of the schema so validation metrics can be sliced exactly:

```
300–800 m | 800–1200 m | 1200–2000 m | 2000–3500 m | 3500–5000 m
```

### 5.1 Class ids and label rules

<!-- fix(#12): the doc specified "class_id" but never mapped it — two implementers would have
     produced silently mislabelled datasets. -->

| id  | class        | size band                                |
| --- | ------------ | ---------------------------------------- |
| 0   | `quad`       | `0.45–1.0 m` tip-to-tip incl. propellers |
| 1   | `fixed_wing` | `1.2–2.5 m` span                         |
| 2   | `decoy`      | `0.2–1.5 m`                              |
| 3   | `bird`       | `0.25–1.0 m` span                        |

- **One label set, both modalities.** EO and IR share frame index, boxes and `track_id`; only
  appearance differs.
- **Boxes are tight to the rendered silhouette** (`bounding_box_2d_tight`), never padded.
- **Minimum box `2×2 px` in IR.** <!-- fix(#10) --> Below that the target is recorded under
  `metadata.json:unlabeled_targets` and excluded from `labels.npz`: a 1 px box is label noise, and
  its class comes from EO at zoom instead (a 0.45 m quad is ~39 px at `1200 mm`).
- **Occlusion shrinks the box, it does not erase it.** Write the visible extent plus
  `occlusion_pct` in `metadata.json`.
- **The train crop must contain the target.** <!-- fix(#14) --> Because the 16:9 rasters are
  cropped to a square (§3.5), a target near the frame edge would be cut out of the training
  sample while still labelled in the full frame. The crop window is centred on the frame and the
  clip is regenerated if the target leaves that window in more than `5 %` of frames; the window
  rectangle is recorded per clip in `metadata.json`.
- **An empty label set is only legal** during a PTZ scan-miss window or at `occlusion_pct ≥ 80`.
  Any other empty frame is a generation bug — regenerate the clip.

### 5.2 Track-level features (temporal)

<!-- fix(#11): the spec argued motion is the discriminator at 2–5 km, but YOLO is per-frame and
     ByteTrack/BoT-SORT only link boxes — nothing consumed the temporal signal. -->

A track-level stage consumes the whole track instead of one frame:

- **Per-track features**, aggregated from `metadata.json`: median and `p90` groundspeed, vertical
  bob amplitude and dominant frequency, flap-rate estimate (`2–6 Hz`, birds only), aspect-ratio
  variance, IR hotspot count and area (from `instance_segmentation`), trajectory curvature,
  range rate, and track lifetime.
- **Classifier**: a small model (MLP or gradient-boosted trees) over those aggregates → class +
  confidence per track. Cheap, debuggable, and it needs no new labels — the per-frame metadata is
  already a superset of what it eats.
- **`track_id` policy**: an id survives occlusion and re-acquisition for up to `2 s`; beyond that
  it is a new id. `track_state` is written per frame so the policy is auditable.
- **Evaluation is per track**, not per frame, for the `2–5 km` buckets — that is the metric the
  dataset exists to move.

### 5.3 Dataset composition, split and validation

<!-- fix(#13): no clip counts, no class balance, no split, and no way to tell whether synthetic
     clips look real. -->

| Item                 | Value                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| Total clips          | `20,000`                                                                                                 |
| Per range bucket     | `0.20` each = `4,000` per bucket (`300–800`, `800–1200`, `1200–2000`, `2000–3500`, `3500–5000 m`)         |
| Class mix per bucket | `quad 50 %`, `fixed_wing 20 %`, `decoy 15 %`, `bird 15 %` (bird/decoy clips carry flocks/`1–15` objects) |
| Background mix       | `sky 40 %`, `horizon 30 %`, `ground clutter 30 %` — hard cases oversampled inside the long buckets       |
| Split                | `70 / 15 / 15` **by terrain sector**, never by clip: adjacent clips from one sector would leak           |
| Real held-out set    | `200–500` real clips, same optics, never trained on                                                      |
| Reported metrics     | per-bucket recall and precision at a fixed false-alarm rate, plus per-track accuracy (§5.2)              |

Sim-to-real is **measured, not assumed**: report the per-bucket recall delta between the real
held-out set and the synthetic val split, and treat a large Tier 2 IR delta as a signal to revisit
§3.6's compositing rather than to add more domain randomisation.

---

## 6. `DatagenConfig`

<!-- fix(#12): the inline sketch is deleted. It duplicated the class, carried a mutable default
     that raised at import, and would drift from code the moment either changed. -->

The config lives in `app/datagen/config.py` — imported, never copied, the same rule as
`app/training/config.py`. Facts that are easy to get wrong are derived properties rather than
fields: `fps` (per tier), `frames_per_clip` (`duration × fps`), `shutter_max_s` (`1 / fps`),
`range_m` (per tier). `app/test/test_datagen_config.py` guards the fps derivation, the shutter law
and the bucket contiguity that `docs.md` previously only asserted in prose.

---

## 7. Build Order (`app/datagen/`)

1. **`config.py`** — the dataclass plus the pinned constants (fps per tier, range buckets, class
   ids). Imported by everything below; no Isaac dependency, so it is unit-testable in the uv env.
2. **`terrain_builder.py`** — offline script implementing §1.2.1. Writes `map_10km.usd` (render
   mesh + decimated collider) and `ground_z.npy`. Run once, not per clip.
3. **`assets.py`** — builds/loads drone USDs with thermal attributes on motor bells and battery;
   procedural articulated bird USD with wings driven by a sine wave in the step loop; decoy +
   fixed-wing assets.
4. **`rig.py`** — spawns Tier 1 (`ivaster_fl100`) and Tier 2 (`hzsoar_ptz`) camera rigs via
   `omni.replicator.core` render products, requests the §3.5 AOV set, implements the §3.3 scan
   model, far clip `12,000 m`.
5. **`ir_composite.py`** — the §3.6 compositor: part emissivity → temperature → NETD/FPN →
   white-hot mapping. Pure NumPy, testable against the spike frames.
6. **`generate.py`** — CLI entrypoint:
   `--sensor tier1|tier2 --clips 50 --duration 6 --headless`.
   Steps physics/kinematics and writes `eo.mp4`, `ir.mp4`, `labels.npz`, `metadata.json`.
7. **`export_yolo.py`** — materialises per-frame `.txt` labels from `labels.npz` for Ultralytics.

### 7.1 Compute budget

<!-- fix(#14): no render-time estimate existed. The rule is: measure one clip before quoting a
     total, and never render two resolutions per clip for one dataset. -->

- **Measure first.** Time one `250`-frame clip end to end on the Isaac box and record wall-clock
  fps. Every number below is derived from that measurement, not assumed.
- **Frames to render:** at the pinned `60 Hz` / `30 Hz`, `20,000` clips average ~`340` frames →
  roughly `6.8 M` frames. At measured `N` fps the wall clock is `6.8e6 / N` seconds.
- **Storage:** mp4 at ~`3 Mbps` is ~`4 MB` per modality per `10 s` clip → ~`150 GB` for the full
  set. Raw PNG frames would be `~9 TB`; don't.
- **Do not render two resolutions.** The lower resolution is derivable from the native frame, so a
  second render is pure cost. `1024×1024` is a centre crop applied offline (§3.5).
- **Reuse boundary:** the Isaac _environment_ is reused; the datagen code is new. Nothing in
  `app/` currently uses Replicator, and `app/datagen/` starts from `config.py`.

---

## 8. Open Items

Resolved:

- **Terrain variants** — dropped. One baked map plus per-clip sector, lighting and scatter
  randomisation (§1.2).
- **`track_id` policy** — pinned in §5.2: survives up to `2 s` of occlusion, new id after.

Still open:

- **Tier 2 defog curve values** — the mechanism is pinned (§4: a `256`-point LUT at
  `app/datagen/defog_lut.npz`, hash recorded per run), but the curve itself is not generated yet,
  and until it is, Tier 2 EO is rendered with defog **off** rather than with an improvised curve.
- **Fiber-optic tether** — model the spool/line for Tier 1 close-range clips, or explicitly
  exclude it? Currently unresolvable past ~`800 m`, so it is an asset-authoring decision (§2).
- **IR compositing validation** — §3.6 must be checked against real LWIR frames of a small
  multirotor before the full run; this is the spike that gates `assets.py` and `ir_composite.py`.

---

## 9. Fix map

Which change in this document answers which issue. Every inline change is also tagged
`<!-- fix(#n): ... -->`, so `grep -n 'fix(#' app/datagen/docs.md` lists them in file order.

| Change | Fixes | Section |
| --- | --- | --- |
| IR composited from AOVs + per-part emissivity, no thermal AOV assumed | #9 | §3.6 |
| Quad narrowed to `0.45–1.0 m` tip-to-tip; micro/heavy dropped | #10 | §2, §0.1 |
| `2×2 px` minimum IR box; sub-pixel targets go to `unlabeled_targets` | #10 | §5.1 |
| "Two cues or regenerate" relaxed to a degraded-cue rule at long range | #10 | §2 |
| Track-level feature list, small classifier, per-track evaluation | #11 | §5.2 |
| `track_id` survives `2 s` of occlusion, then a new id | #11, #12 | §5.2, §8 |
| Reconciled-values table replaces the unverifiable "project notes win" rule | #12 | §0.1 |
| `DatagenConfig` extracted to `config.py`; mutable default fixed | #12 | §6, `config.py` |
| Frame rate pinned per tier (`60 / 30 Hz`) + `shutter ≤ 1/fps` law | #12 | §3.1, §0.1 |
| Render path pinned: one native pass, zoom is a crop, per-sensor square train crop | #12, #14 | §3.5 |
| AOV split: heavy buffers on the IR camera only (≈5× less readback) | #14 | §3.5 |
| Train crop must contain the target (regenerate beyond 5 % of frames) | #14 | §5.1 |
| Terrain recipe: ridged multifractal, wadis, decimated collider, emissivity | #12 | §1.2.1 |
| Heat shimmer, camera-level wind jitter, shared `β`, defog LUT | #12 | §4 |
| Class-id table, label sharing, occlusion and empty-frame rules | #12 | §5.1 |
| Range buckets incl. the `300–800 m` close bucket | #13, #12 | §0.1, §5.3 |
| `20,000` clips, `0.20` per bucket, class mix | #13 | §5.3, `config.py` |
| `70 / 15 / 15` split by terrain sector, not by clip | #13 | §5.3 |
| Real held-out set + measured sim-to-real delta | #13 | §5.3 |
| Compute budget, one native render, per-clip `.npz` labels | #14 | §7.1, §5 |
