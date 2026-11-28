# Datagen Implementation Notes

What to write, in what order, and how each piece is proven. The specification these notes build
against lives in `plan.md` (Part 2); the milestone gates live in `plan.md` (Part 1); the physics
derivations live in `implementation_guide.md`.

Generated artefacts go to `app/datagen/assets/` (`ground_z.npy`, `map_10km.usda`, `defog_lut.npz`).

---

## Current Vision System Flow (updated per milestone)

```
[A3 + B1 terrain_builder.py]                 [B2 assets.py]
   ground_z.npy  +  map_10km.usda               15 .glb meshes + 15 .json (+K hotspots)
         |                |                                   |
         v                v                                   v
+-------------------------------------------------------------------------------+
| [B3 rig.py: Camera Rig, Frustum Spawner & Rate-Limited PTZ]                   |
|   camera_optics() -> spawn_in_frustum() -> step_ptz() -> render_rig_frame()   |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| [C1 + C2 generate.py: Dataset Clip Runner, 95 % Crop Check & Track Features]  |
|   1. sample_clip_spec()       -> stratified bucket (300..5000 m) & 50/20/15/15|
|   2. simulate_clip_frames()   -> retries if in_crop_ratio < 0.95              |
|   3. compute_track_features() -> speed, bob, flap_hz, hotspot count           |
|   4. write_clip_bundle()      -> eo.mp4, ir.mp4, labels.npz, metadata.json    |
|                                  + [A2 export_yolo.py] -> labels/frame_XXXX   |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| [D1 + D2 generate.py --benchmark: Throughput & Stratified Pilot Gate]         |
|   1. benchmark_throughput() -> times 1 clip -> fps, 2k pilot h, 20k full h    |
|   2. audit_bucket_spans()   -> checks rendered px width vs optical formula    |
|   3. pilot_report()         -> verifies 5-bucket balance & writes report.json |
+-------------------------------------------------------------------------------+
```

- **Milestone A1** — Thermal IR frame synthesis (`ir_composite.py`)
- **Milestone A2** — Native-to-crop YOLO label mapper (`export_yolo.py`)
- **Milestone A3** — Terrain heightmap generation (`terrain_builder.py`)
- **Milestone B1** — Terrain 3D mesh & biome USD export (`terrain_builder.py`)
- **Milestone B2** — Sketchfab downloader & `+K` hotspot applier (`assets.py`)
- **Milestone B3** — Camera rig, frustum spawn & PTZ tracker (`rig.py`)
- **Milestone C** — Dataset clip runner, `>= 95 %` crop check & track features (`generate.py`)
- **Milestone D** — Throughput benchmark & stratified pilot gate (`generate.py --benchmark`)

---

## Milestone A — local modules (no Isaac)

Three modules, none of which import Isaac, all runnable in the uv env. Gate for the milestone:
`uv run pytest` green **and** one synthetic IR frame you can look at.

---

## Milestone A1 — Thermal IR frame synthesis (`ir_composite.py`)

You are not rendering heat, you are computing it. The renderer supplies four geometry buffers and
heat is a function of them.

**Inputs**

| Argument | Shape / type | Source |
| --- | --- | --- |
| `aovs["instance_segmentation"]` | `(H, W)` int | IR render product |
| `aovs["normals"]` | `(H, W, 3)` float | IR render product |
| `aovs["distance_to_image_plane"]` | `(H, W)` float, metres | IR render product |
| `aovs["motion_vectors"]` | `(H, W, 2)` float | IR render product |
| `cfg` | `DatagenConfig` | `config.py` |
| `sun` | `{elevation_deg, azimuth_deg, t_amb_k}` | per-clip lighting sample |
| `materials` | `{instance_id: (epsilon, delta_t_k)}` | `assets.py` table |
| `rng` | `np.random.Generator` | seeded per clip |

**Output:** `ir_u8` `(H, W)` uint8, plus `ir_temp` `(H, W)` float32 when `debug=True`.

**Order of operations — do not reorder.**

```
1. background    cos_inc  = clip(dot(n_hat, sun_dir), 0, 1)
                 t_bg     = t_amb + dT_solar(elevation) * cos_inc + dT_inertia(material)

2. targets       for each instance id:  t[seg == id] = t_amb + delta_t_k

3. attenuation   beta = 3.912 / visibility_m
                 tau  = exp(-beta * depth)
t    = tau * (epsilon * t + (1 - epsilon) * t_amb) + (1 - tau) * t_amb
                 # emitted + reflected-ambient, then the air path

4. sensor noise  t += N(0, netd_k^2) + fpn      fpn fixed per clip, not per frame

5. display map   u8 = round(255 * clip((t - t_min) / (t_max - t_min), 0, 1)^gamma)

6. shimmer       only if focal_mm >= 300: whole-frame warp of u8
```

**Why this order**

- Noise in step 4, not step 5. Noise added to the 8-bit image scales with brightness, so the
  bright target gets noisier than the sky. In kelvin, 50 mK is 50 mK everywhere.
- Fixed gain in step 5. Real cameras auto-gain per scene, which normalises target-vs-background
  contrast; do that here and class cues change clip to clip.
- Step 1 is what produces the dawn/dusk crossover. At zero sun elevation the solar term vanishes
  and rock converges on air temperature, so no special case is needed.

**Invariants**

- Noise is added in kelvin, never after the 8-bit mapping.
- Masks are consumed at IR resolution only (`plan.md` §3.5).
- Internal dtype float32; only the final buffer is uint8.

**Acceptance tests** — synthetic 5-actor fixture: quad, fixed-wing, decoy, bird, empty sky.

1. Output shape `(H, W)` and dtype uint8.
2. A rotor pixel is brighter than a bird-torso pixel (`+35 K` vs `+12 K`).
3. Sky is darker than sunlit ground at `elevation_deg = 55`.
4. At `elevation_deg = 0`, `|t_rock - t_air| < 2 K` (thermal crossover).
5. Noise std in the temperature domain is within 20 % of `netd_k`.
6. Same seed gives bit-identical output; a different seed differs.

### Concepts this module covers (plain language)

- **Temperature vs. brightness.** A thermal camera does not measure "hot" or "cold", it measures
  incoming infrared energy (radiance). Radiance grows roughly with `T^4`, so small temperature
  differences at 300 K are still easily visible. We compute in kelvin because that is the physical
  quantity; the 0-255 image is produced only at the very end.
- **Emissivity, `epsilon`.** No real surface is a perfect radiator. A matte black motor bell emits
  nearly everything it has (`epsilon = 0.95`); polished metal emits little. Whatever a surface does
  not emit, it reflects from its surroundings.
- **The `(1 - epsilon)` term.** This is why the formula has two parts: `epsilon * T` is what the
  surface emits, `(1 - epsilon) * T_amb` is the room it reflects. Without that second part, a rock
  at air temperature would read 8 % colder than the air, which is wrong — and the dawn crossover
  test fails.
- **Solar heating and the crossover.** A sunlit face warms by `cos(angle to the sun)` times the sun
  elevation. At dawn and dusk the elevation term goes to zero, so rock, wadi floor and air all
  converge on the same temperature. That single line is the whole thermal-crossover feature.
- **Beer-Lambert extinction.** Air absorbs and scatters IR along the path: `tau = exp(-beta * d)`.
  `beta = 3.912 / visibility` means "if visibility is 20 km, a target at 20 km is 2 % visible".
  This is why a 4 km drone is dimmer than a 500 m drone even at the same temperature.
- **NETD (noise equivalent temperature difference).** The smallest temperature change the sensor can
  see, here 50 mK. Sensor noise is modelled as a Gaussian of that width — added in kelvin, so it is
  equally strong on the cold sky and on the white-hot rotor.
- **FPN (fixed-pattern noise).** Every pixel has a small permanent offset of its own, like a dirty
  lens. It is constant for the whole clip, not refreshed per frame, which is exactly how real
  microbolometers behave.
- **Fixed gain vs. AGC.** Real thermal cameras constantly rescale the image so the scene fills
  0-255 (auto-gain). That makes a dim target look the same as a bright one. We pin `t_min`, `t_max`
  and `gamma` for the whole run, so "brighter pixel" always means "hotter object" — the class cue
  survives.
- **White-hot mapping.** `u8 = 255 * clip((T - t_min) / (t_max - t_min), 0, 1) ^ gamma`. Hue is not
  information here: white = hot, black = cold, nothing else. `gamma < 1` lifts the dark end so cold
  background texture stays visible.
- **Why the sky is special-cased.** The depth buffer returns infinity for sky, so `tau = 0` and the
  formula would return air temperature — a bright grey sky. Real clear sky is much colder than the
  air, so sky pixels get `t_amb - 20 K` directly. This is a modelling choice, not physics.


---

## Milestone A2 — Native-to-crop YOLO label mapper (`export_yolo.py`)

Turns one ground-truth table per clip into one plain-text snapshot per frame. It touches no
pixels, detects nothing and tracks nothing: every row is converted independently, and the only
thing linking frames is `track_id`, which is dropped on the way out.

**Why it exists.** `labels.npz` stores boxes as fractions of the *native* raster (1920x1080,
1280x1024). Training images are a centred square crop, so Ultralytics has to be told where the
object is **on that crop**. Every box is rewritten against the crop's origin and size.

**Input:** `labels.npz` (`frame, class_id, cx, cy, w, h, track_id, occluded`), `cfg`, modality.
**Output:** `<clip>/labels/frame_XXXX.txt`, one line per box:

```
class_id cx cy w h
```

Five fields, space-separated, no header, no commas, six decimals. A frame with no boxes gets a
zero-byte file: Ultralytics distinguishes a missing label file from an empty one.

### The crop window

The crop is centred, so its top-left corner is half of what gets cut away:

```
ox = (W - C) / 2
oy = (H - C) / 2
```

| Sensor | Native `W x H` | Crop `C` | `ox` | `oy` |
| --- | --- | --- | --- | --- |
| Tier 1 EO | 1920 x 1080 | 1024 | 448 | 28 |
| Tier 1 IR | 640 x 512 | 512 | 64 | 0 |
| Tier 2 EO | 2560 x 1440 | 1024 | 768 | 208 |
| Tier 2 IR | 1280 x 1024 | 1024 | 128 | 0 |

Every margin is even, so the crop never lands on a half pixel. `crop_origin()` raises rather than
rounding if that ever stops being true — a half-pixel shift moves every box forever, and no
downstream test would ever name it.

### The transform

```
cx' = (cx * W - ox) / C
cy' = (cy * H - oy) / C
w'  = (w  * W)     / C
h'  = (h  * H)     / C
```

That is the whole mapping: subtract where the crop starts, divide by how big it is. No rotation,
no perspective, no resampling — one offset and one scale per axis, which makes it exactly
invertible.

**No resize.** Every pixel keeps its original angular size, so `mrad/px` is unchanged and the
pixel-size arithmetic in `plan.md` §5.1 still describes the training image. A 4 px drone stays
4 px because they are the same pixels, not because anything was scaled.

**The centre is the fixed point.** Put `cx = 0.5` in: `(0.5 * W - (W - C)/2) / C = 0.5` for every
sensor. That is the invariant the tests check, and it is what catches any sign or offset error.

**The fraction is not fixed.** `w' = w * W / C` and `W > C`, so the box's *share* of the image
grows even though its pixel width does not. `0.040` of 1280 is 51.2 px; 51.2 / 1024 = `0.050`.

### Worked example — Tier 2 IR

```
native 1280x1024, crop 1024, ox 128, oy 0
quad at cx 0.240, cy 0.510, w 0.040, h 0.038

cx: 0.240 * 1280 = 307.2 px -> (307.2 - 128) / 1024 = 0.175000
cy: 0.510 * 1024 = 522.2 px -> (522.2 -   0) / 1024 = 0.510000
w : 0.040 * 1280 =  51.2 px ->  51.2 / 1024        = 0.050000
h : 0.038 * 1024 =  38.9 px ->  38.9 / 1024        = 0.038000

written line:  0 0.175000 0.510000 0.050000 0.038000
```

`cy` and `h` pass through untouched because the IR raster is already 1024 tall, so `oy = 0` and
`H = C`.

### Drop rules, in this order

1. **invalid** — non-finite, zero or negative size.
2. **outside** — the box *centre* falls outside the crop rectangle.
3. **too_small** — after clipping, the visible box is under `2 px` on either axis.

Order matters. The centre decides membership *before* clipping, because a box whose middle is
outside the crop describes an object the training image does not really contain. The size test
runs *after* clipping, because clipping is what removes pixels: a box clipped to a 1.5 px sliver
is label noise, not a small object.

**The recentring rule.** Clipping changes which pixels exist, so the label must describe the box
that survived, not the box that was rendered:

```
Tier 1 EO, ox 448, box cx 0.240 w 0.040
  native   422.4 .. 499.2   (76.8 px wide, 25.6 px left of the crop)
  clipped  448.0 .. 499.2   (51.2 px wide)
  cx'      (473.6 - 448) / 1024 = 0.025
  w'       51.2 / 1024          = 0.050

unclipped it would have been cx' 0.0125, w' 0.075 — both describe pixels that were thrown away.
```

**Every drop is counted.** Three counters per clip, printed to stdout, so a 20 % loss rate cannot
go unnoticed:

```
000123  tier2/ir  180 frames
  boxes in        184
  written         181
  dropped outside   2
  dropped < 2 px    1
```

### One owner for the crop origin

`crop_origin()` is imported by the pixel crop in `generate.py`; neither stage recomputes
`(W - C) / 2`. Two copies of that formula can disagree by one pixel, and the result is valid
files, valid images, and every box silently misaligned — visible only as unexplained recall loss
after training.

### Acceptance tests

25 tests in `app/test/test_export_yolo.py`:

- origin table matches all four sensors; upscale, half-pixel and non-positive crops are rejected
- `(0.5, 0.5)` stays `(0.5, 0.5)` for all four sensors (centre invariance)
- round trip: remap then invert returns the original box to `1e-12`
- edge clipping recentres on the visible extent and uses only surviving pixels
- centre outside, and centre exactly on the far edge, are dropped and counted
- sub-2 px and non-finite boxes are dropped and counted
- text format: five fields, six decimals, trailing newline, integer class id
- frames with no boxes get zero-byte files; frames absent from the npz still get files
- npz path and column mapping produce byte-identical output; output is deterministic
- frame index beyond `n_frames` is rejected

### Concepts this module covers (plain language)

- **Normalised coordinates.** A YOLO box is stored as fractions of the image, `cx = 0.5` means
  "middle". That is why the same label works at 640 px and 2560 px.
- **The transform is affine.** Moving from the full frame to the crop is one scale plus one offset
  per axis. No rotation, no perspective — so the same four numbers describe it exactly, and it is
  invertible.
- **Why crop and not resize.** Cropping keeps each pixel's field of view, so "a 0.45 m drone is 4 px"
  stays true. Resizing averages pixels, changes that number, and makes the size rules in `plan.md`
  §5.1 meaningless.
- **Why the fraction grows while the pixel count does not.** The object still covers 51 pixels,
  but the ruler it is measured against shrank from 1280 to 1024, so its share went from 4 % to 5 %.
  Same object, different denominator.
- **Why boxes get dropped.** A box whose centre falls outside the crop describes pixels that no
  longer exist in the training image. Keeping it teaches the model to detect things it cannot see.
  The same rule kills boxes below 2 px, because their centre is noise.
- **Why clipping needs a second centre.** YOLO's `cx, cy` are the centre of the box *as drawn*.
  Once the crop deletes part of the box, the drawn box is smaller and its centre moves. Copying the
  original centre would place the box off to one side of the pixels it claims.

---

## Milestone A3 — Terrain heightmap generation (`terrain_builder.py`)

Builds the ground once, bakes it to disk, and answers "how high is the ground here" with an array
lookup. The USD export stays in Milestone B because it needs `pxr`; this half is pure NumPy and
runs anywhere.

**Why baked, not per clip:** regenerating terrain costs seconds per clip and is the main
throughput killer (`plan.md` §1.2.1). Per-clip variation is sector, lighting and scatter — not
geology.

**Output:** `app/datagen/assets/ground_z.npy` `(1000, 1000)` float32 metres, plus
`ground_z.json` (sidecar with the extent, so `load()` cannot guess wrong) and a stats printout.

### Pipeline, in order

```
1. ridged multifractal   6 octaves, lambda_0 4096 m, lacunarity 2, gain 0.5
2. N-S ridge             one dominant crest, meandering +/-1.5 km, amplification only
3. eastern plateau       smooth uplift on the +x half
4. wadi carve            domain-warped centreline, 20-60 m deep
5. terracing             20-38 deg mid-slopes, step 8-15 m, floors and cliffs untouched
6. rescale               floor 0 m, peaks 850 m
```

### 1. The noise

Everything is built from one primitive: smooth value noise on a 256 x 256 lattice.

```
value(x, y) = bilerp( lattice[floor(x), floor(y)] ... ,
                    smoothstep(frac(x)), smoothstep(frac(y)) )

smoothstep(t) = t*t*(3 - 2*t)
```

`floor` gives the cell, `frac` gives the position inside it, `smoothstep` makes the blend flat at
both ends so octaves do not show grid seams. The lattice wraps every 256 cells — invisible,
because the lowest frequency used is 4096 m and the map is 10 km.

### 2. Ridged multifractal

```
n_o     = value(x / lambda_o, y / lambda_o)      lambda_o = 4096 / 2^o
ridge_o = (1 - |2*n_o - 1|)^2
z       = sum over o of  gain^o * ridge_o * weight_o
weight  = clip(2*ridge_o, 0, 1)
```

`1 - |n|` turns smooth noise into sharp creases: wherever the noise crosses its midpoint, the
absolute value has a corner, and the corner becomes a ridgeline. Squaring sharpens the crest.
The `weight` feedback halves detail inside valleys, so ridges read as ridges instead of uniform
rubble.

Six octaves give detail from 4096 m down to 128 m:

```
o:        0     1     2     3     4     5
lambda: 4096  2048  1024   512   256   128   metres
gain:   1.00  0.50  0.25  0.13  0.06  0.03
```

### 3. Ridge, plateau

```
crest_x(y) = 1500*sin(2*pi*y / 11000) + 500*noise(y)
crest      = exp(-0.5 * ((x - crest_x) / 1400)^2)
plateau    = smoothstep((x - 250) / 4500)

z = z * (1 + 0.85*crest) + 0.30*850*plateau
```

This is what makes one north-south limestone ridge dominate instead of the map being uniform
noise, and lifts the east side into the terraced plateau described in §1.2.1.

### 4. Wadi carve

Two valleys. The centreline is a slow sine (period 2.6 x half-extent) plus a low-frequency warp
in both axes, so it meanders instead of being a clean curve:

```
centre = offset + 900*sin(2*pi*y / 13000 + phase)
warp   = 300 * (2*value(x/4500, y/4500) - 1)
d      = x - centre + warp

z -= depth * exp(-0.5*(d/sigma)^2) + 0.35*depth * exp(-0.5*(d/(4*sigma))^2)
```

`depth` is drawn in `20..60 m` and `sigma` in `80..160 m`. The narrow Gaussian is the channel; the
broad one (4x sigma, 35 % of the depth) guarantees the wadi band is the lowest ground on the map,
so the `0 m` end of the rescale lands on the valley floor rather than on a random dip.

### 5. Terracing

Only mid-slopes are quantised — cliffs stay cliffs and floors stay flat:

```
slope = degrees(arctan(hypot(dz/dx, dz/dy)))

alpha = smoothstep((slope - 20) / 4) * (1 - smoothstep((slope - 34) / 4))
z     = z + alpha * (floor(z / step) * step - z)
```

`alpha` is a raised-cosine window: `0` below 20 deg, `1` between 24 and 34 deg, `0` again above
38 deg. `step` is drawn once per map in `8..15 m`. On Mediterranean hillsides the terraces are
agricultural, and they matter here because they create flat facets that catch the sun at
different angles — realistic thermal clutter at 1-3 km.

### 6. Rescale

```
z = relief_min + (z - z.min()) * (relief_max - relief_min) / (z.max() - z.min())
```

Two passes: one before the wadi (so the carve has something in metres to cut), one at the end
(so the final relief is exactly `0..850 m` regardless of what the terraces did to the extremes).

### Lookup

```
fx = clip((x + half_extent) / spacing, 0, nx - 1)
fy = clip((y + half_extent) / spacing, 0, ny - 1)

h = (1-tx)(1-ty)*z[iy,   ix  ] + tx(1-ty)*z[iy,   ix+1]
  + (1-tx) ty  *z[iy+1, ix  ] + tx ty  *z[iy+1, ix+1]

agl(x, y, z) = z - h
```

**Why bilinear and not nearest:** the grid spacing is ~10 m. Nearest-neighbour snaps to one
vertex, so the returned height can be off by half a cell — about 5 m — and drones fly at 15 m AGL.
That is a 33 % error on the quantity every spawn and collision check uses.

Outside the grid, coordinates clamp to the nearest edge. The world is pinned to +/-5 km, so a
query outside it is a bug upstream, not a case to interpolate.

### Numbers from the actual build

```
shape            1000 x 1000   float32
spacing          10.0103 m     (1000 vertices over 10 km; the extent is pinned, so spacing follows)
relief           0.0 .. 850.0 m
terrace step     10.86 m
wadis            40.5 m deep / 156 m sigma, 58.0 m deep / 105 m sigma
slope p50 / p95  12.1 / 28.9 deg
mid-slope share  19.7 %         (the band the terraces act on)
build time       0.46 s
```

1000 vertices over 10 km is 10.01 m, not exactly 10. The extent is the pinned quantity and the
spacing is derived, so the grid stays inside the world bounds.

### Acceptance tests

17 tests in `app/test/test_terrain_builder.py`:

- shape, dtype, `min == 0`, `max == 850` within 1 m, no NaN or inf
- same seed identical, different seed different
- default `DatagenConfig` gives `(1000, 1000)` and ~10 m spacing
- `at()` matches a hand-computed bilinear value on a 3x3 patch
- `at()` reproduces a tilted plane exactly, and accepts arrays
- `at()` clamps outside the grid; `agl()` positive above and negative below
- save/load round trip keeps the array and the extent
- `_terrace` snaps mid-slopes to multiples of the step and leaves flat ground untouched
- `_carve_wadis` cuts at least 20 m and leaves most of the map untouched
- lattice is reproducible and inside `[0, 1]`; a 4-vertex grid is rejected
- hillshade is uint8 and flat ground has one shading value

### Concepts this module covers (plain language)

- **Noise as a building block.** `value(x, y)` returns a smooth pseudo-random number, the same
  number every time for the same point. Everything else is that function, stacked and reshaped.
- **Octaves / fBm.** Add several copies of the noise at doubling frequency and halving amplitude.
  Low octaves make continents, high octaves make rubble. That sum is "fractal Brownian motion".
- **The ridged trick.** Ordinary noise peaks are rounded. Taking `1 - |noise|` turns the
  zero-crossings into sharp creases, which is what limestone ridges look like. Squaring sharpens
  the crest further.
- **Domain warping.** Instead of drawing a straight valley, offset the sampling coordinates by
  another noise field. The valley then wanders. Same idea as bending a photo before sampling it.
- **Quantisation = terracing.** `floor(z / step) * step` snaps elevation to flat steps. Blend it
  in only on mid-slopes, or you get stairs on the cliffs too.
- **Bilinear interpolation.** A heightmap is samples at grid points, so asking "what is the ground
  height here" needs interpolation between four neighbours. Nearest-neighbour rounds to the
  closest vertex: at 10 m spacing that is up to 5 m of error, which matters for a drone at 15 m
  AGL. Bilinear weights the four corners by distance and is exact for a flat surface.
- **Baking beats recomputing.** Generate once, save the grid to disk, load it read-only
  afterwards. Every later stage gets a free `O(1)` lookup instead of a raycast or a regeneration.

---

## Milestone A gate

```bash
uv run pytest app/test/ -q
python -m app.datagen.ir_composite --demo --out /tmp/ir_demo.png
python -m app.datagen.export_yolo --demo
python -m app.datagen.terrain_builder --demo --out /tmp/terrain_demo.png
```

Pass condition: tests green, the quad reads white-hot with merged hotspots at range while the
bird reads as a dim uniform blob, the YOLO demo reports the expected drops, and the hillshade
shows one dominant N-S ridge with a wadi band and terraced mid-slopes.

---

## Milestone B1 — Terrain 3D mesh & biome USD export (`terrain_builder.py`)

Without B1, Isaac Sim only has a flat spreadsheet of numbers (`ground_z.npy`) and nothing 3D to
put under the camera. B1 turns that height table into the 3D Galilee mountain stage
(`app/datagen/assets/map_10km.usda`) that the EO camera renders and that gives the IR camera its
ground `normals` and `depth_m` buffers.

### What `(row, col)` means in `ground_z.npy`

`ground_z.npy` is a `1000 x 1000` spreadsheet of heights:

1. **`col` (`0..999`) = how many 10 m steps East (`x`):** `col = 0` is `-5000 m` (West),
   `col = 500` is `0 m` (centre), `col = 999` is `+5000 m` (East).
2. **`row` (`0..999`) = how many 10 m steps North (`y`):** `row = 0` is `-5000 m` (South),
   `row = 999` is `+5000 m` (North).
3. **`ground_z[row, col]` = the height `z` in metres at that spot:**

```
             col=0 (West)     col=1 (+10 m)    col=2 (+20 m)
row=0 (South)   120 m            125 m            140 m
row=1 (+10 m)   118 m            130 m            155 m
```

So cell `(row=1, col=2)` becomes the 3D point `(x = -4980, y = -4990, z = 155)`.

### How the height array becomes a 3D mesh (`map_10km.usda`)

1. **Turn every array cell `(row, col)` into a 3D point `(x, y, z)`:**
   ```
   x = -5000 + col * 10.01
   y = -5000 + row * 10.01
   z = ground_z[row, col]
   ```
2. **Connect every `2 x 2` square of points into 2 triangles:**
   ```
   v00 (row, col) ------ v10 (row, col+1)
         |             /        |
         |           /          |
   v01 (row+1,col) ----- v11 (row+1,col+1)

   triangle 1 = (v00, v10, v11)
   triangle 2 = (v00, v11, v01)
   ```
3. **Write those points + triangles into `map_10km.usda`:**
   - `VisualMesh` uses the fine grid with vertex normals and biome RGB colors.
   - `ColliderMesh` uses a decimated stride (`PhysicsCollisionAPI`) so PhysX collision is `16x`
     faster.

### How rocky, earth, and wadi zones get applied automatically

Height `z` and slope angle `slope_deg` classify every vertex without hand-painting:

| Zone | Rule from `ground_z.npy` | EO look (RGB vertex color) | IR behavior (`epsilon`, `inertia_k`) |
| --- | --- | --- | --- |
| **1. Bare limestone rock (`2`)** | `slope > 28 deg` or `z > 500 m` | Pale grey-tan cliff rock | Holds daytime heat (`inertia +8 K`, `eps 0.90`) |
| **2. Terraced earth & scrub (`1`)** | `10..28 deg` mid-slopes | Olive-brown soil & scrub | Medium warmth (`inertia +4 K`, `eps 0.94`) |
| **3. Wadi floor & vegetation (`0`)** | `z < 120 m` and `slope < 12 deg` | Dark green-brown valley floor | Coolest zone (`inertia +1 K`, `eps 0.96`) |

### The 3 functions added in B1 (with 4-5 line pseudocode)

#### 1. `triangulate_grid(ground, stride=1)`
Turns the 2D height array (`ground.z`) into 3D vertex coordinates, triangle indices, and surface
normals:

```python
z_sub = ground.z[::stride, ::stride]
x, y  = meshgrid(linspace(-5000, 5000, cols), linspace(-5000, 5000, rows))
points = stack([x, y, z_sub], axis=-1).reshape(-1, 3)
# split each (row, col) grid square into 2 triangles: (v00,v10,v11) and (v00,v11,v01)
faces  = stack([v00, v10, v11, v00, v11, v01], axis=-1).reshape(-1, 3)
```

#### 2. `classify_biomes(ground, stride=1)`
Assigns rock (`2`), terraced earth (`1`), or wadi floor (`0`) to every vertex and computes lit RGB
colors:

```python
slope = _slope_deg(ground.z, ground.spacing_m)
is_rock = (slope > 28.0) | (ground.z > 500.0)
is_wadi = (ground.z < 120.0) & (slope < 12.0) & (~is_rock)
biome   = where(is_rock, 2, where(is_wadi, 0, 1))  # 2=rock, 1=earth, 0=wadi
rgb     = BIOME_PALETTE[biome] * (0.35 + 0.65 * hillshade(ground.z, spacing)[..., None] / 255.0)
```

#### 3. `export_terrain_usd(ground, out_path, visual_stride=1, collider_stride=4)`
Writes both `/World/Terrain/VisualMesh` and `/World/Terrain/ColliderMesh` into
`app/datagen/assets/map_10km.usda`:

```python
v_pts, v_faces, v_norms = triangulate_grid(ground, stride=visual_stride)
c_pts, c_faces, _       = triangulate_grid(ground, stride=collider_stride)
_, v_rgb                = classify_biomes(ground, stride=visual_stride)
write_usd_stage(out_path, visual=(v_pts, v_faces, v_norms, v_rgb), collider=(c_pts, c_faces))
return out_path
```

### How to view the 3D mesh

```bash
uv run python -m app.datagen.terrain_builder --export-usd
open app/datagen/assets/map_10km.usda
uv run python -m app.datagen.terrain_builder --demo --out /tmp/terrain_biome_demo.png
```

- **macOS QuickLook:** `open app/datagen/assets/map_10km.usda` (or press `Spacebar` in Finder)
  spins and zooms the 3D mountain mesh directly.
- **Blender:** `File -> Import -> Universal Scene Description (.usd / .usda)`.
- **2D biome map:** `/tmp/terrain_biome_demo.png` shows the limestone rock ridges (`15.9 %`),
  terraced earth (`79.9 %`), and wadi floor (`4.2 %`).

### Acceptance tests

4 new tests in `app/test/test_terrain_builder.py` (21 total):

- `triangulate_grid()` produces exact vertex/triangle counts and `stride=4` cuts triangles by `16x`
- `classify_biomes()` separates all 3 zones (`BIOME_WADI`, `BIOME_EARTH`, `BIOME_ROCK`) with RGB in `[0, 1]`
- `export_terrain_usd()` writes valid `#usda 1.0` with `VisualMesh`, `ColliderMesh`, `PhysicsCollisionAPI`, and `primvars:displayColor`
- `export_terrain_usd()` rejects `collider_stride < visual_stride`

### Concepts this module covers (plain language)

- **Triangulation of a grid.** A 3D renderer draws triangles, not height tables. Every `2 x 2`
  square of adjacent `(row, col)` cells is cut diagonally into 2 triangles.
- **Visual mesh vs. collider mesh.** The camera needs fine triangles so distant ridgelines do not
  look blocky. Physics collision only checks whether a drone hit the mountain, so taking every 4th
  point cuts the triangle count by `4 x 4 = 16x` with negligible height error.
- **Procedural biome splatting.** Instead of painting textures by hand, height and slope angle
  decide whether a point is cliff rock, terraced soil, or valley floor.

---

## Milestone B2 — Sketchfab downloader & `+K` hotspot applier (`assets.py`)

Every 3D model in `app/datagen/assets/meshes/<class_name>/<slug>.json` defines **one dark base
temperature** (`base_material.delta_t_k`) for the whole mesh, plus a list of **3D local hotspot
positions** (`offset_norm = [dx, dy, dz]` in unit-span coordinates `[-0.5, +0.5]`) and their
temperature boost `delta_t_k` (`+K`).

Larger `delta_t_k` in the JSON raises the surface kelvin at that position before Beer-Lambert
attenuation and White-Hot mapping, so that spot shines brighter in the processed IR image — with
zero Blender mesh splitting required.

### JSON structure (`app/datagen/assets/meshes/<class_name>/<slug>.json`)

```json
{
  "slug": "meshy_tail_fiber_fpv",
  "name": "[Meshy] Tail FPV Optic Fibre Drone",
  "class_name": "quad",
  "class_id": 0,
  "sketchfab_uid": "0f037de4489b4c6eaae3edc4d9ae23b2",
  "mesh_file": "meshy_tail_fiber_fpv.glb",
  "span_range_m": [0.45, 1.0],
  "base_material": { "seg_id": 111, "epsilon": 0.86, "delta_t_k": 2.0 },
  "hotspots": [
    { "name": "motor_fl", "offset_norm": [ 0.32,  0.32, 0.02], "radius_norm": 0.08, "delta_t_k": 33.0, "epsilon": 0.95 },
    { "name": "motor_fr", "offset_norm": [ 0.32, -0.32, 0.02], "radius_norm": 0.08, "delta_t_k": 33.0, "epsilon": 0.95 },
    { "name": "motor_rl", "offset_norm": [-0.32,  0.32, 0.02], "radius_norm": 0.08, "delta_t_k": 33.0, "epsilon": 0.95 },
    { "name": "motor_rr", "offset_norm": [-0.32, -0.32, 0.02], "radius_norm": 0.08, "delta_t_k": 33.0, "epsilon": 0.95 },
    { "name": "battery_esc", "offset_norm": [0.0, 0.0, 0.03], "radius_norm": 0.10, "delta_t_k": 11.0, "epsilon": 0.92 }
  ]
}
```

### The projection + stamping math

```
1. scale to span_m     (dx, dy, dz) = offset_norm * span_m
                       r_m          = radius_norm * span_m

2. project to 2D IR    u_px = u_center + (dy_rot * focal_px) / depth_m
                       v_px = v_center - (dz_rot * focal_px) / depth_m
                       r_px = max((r_m * focal_px) / depth_m, 0.75)

3. stamp in kelvin     mask      = (seg == base_seg_id)
                       d2        = (u - u_px)^2 + (v - v_px)^2
                       t[mask]  += delta_t_k * exp(-0.5 * d2 / r_px^2)
```

Because step 3 only indexes `t[mask]`, a hotspot never leaks outside the mesh silhouette onto the
sky or ground.

### All 15 generated model configs on disk

| Class | Count | Files in `app/datagen/assets/meshes/<class>/` | Base `+K` | Peak `+K` |
| --- | --- | --- | --- | --- |
| `quad` | 8 | `fpv_drone_5part`, `fpv_kamikaze_lowpoly`, `combat_fpv_rpg7`, `meshy_tail_fiber_fpv`, `rodin_long_fiber_fpv`, `fpv_drone_lod_3k`, `vampire_babayaga` (6-motor hex), `fpv_dron_nonstop` | `+2.0..2.5 K` | `+35.0..37.0 K` |
| `fixed_wing` | 4 | `lancet_zala52_lowpoly`, `lancet_detailed`, `leleka_100` (tail pusher), `orlan_10` (nose engine) | `+2.0 K` | `+60.0 K` |
| `decoy` | 1 | `decoy_balloon_reflector` (zero hotspots) | `+8.0 K` | `+8.0 K` |
| `bird` | 2 | `bird_flying_lowpoly`, `flying_seagull` (`2-6 Hz` flap) | `+1.0 K` | `+12.0 K` |

### Sketchfab `.glb` downloader via API token

Sketchfab's `v3/models/<uid>/download` endpoint requires a free account API token (`HTTP 401`
without it). Copy your token from `https://sketchfab.com/settings/password` and run the download
command below; it pulls the Sketchfab `.glb` archives into `app/datagen/assets/meshes/<class>/`
next to each `<slug>.json`. Curated model table and URLs live in `app/datagen/sketchfab_models.md`.

### Commands

```bash
uv run python -m app.datagen.assets --download --token <YOUR_SKETCHFAB_API_TOKEN>
```

### Acceptance tests

5 tests in `app/test/test_assets.py`:

- all 15 custom JSONs exist on disk and match `CLASS_IDS` + `SPAN_RANGES_M`
- increasing `delta_t_k` (`+K`) in a model's JSON makes the processed IR pixel strictly brighter
- hotspots are clamped to `seg == base_seg_id` and never bleed onto sky or ground pixels
- thermal peak hierarchy holds: fixed-wing (`+60 K`) > quad (`+35 K`) > bird (`+12 K`) > decoy (`+8 K`)
- `project_hotspots()` scales pixel spread linearly with `1 / depth_m` and clamps `radius_px >= 0.75`

### Concepts this module covers (plain language)

- **Why stamp `+K` by position instead of splitting 3D meshes.** Many downloaded models (especially
  AI-generated ones like `[Meshy]` and `[Rodin]`) are a single fused mesh where the motors and
  frame share one surface. Defining hotspots as 3D points `(dx, dy, dz)` with a `+K` boost lets any
  single-piece mesh work out of the box without opening Blender.
- **Unit-span coordinates (`offset_norm`).** Hotspot coordinates are stored for a `1.0 m` object
  (`-0.5` to `+0.5`). Multiplying by `span_m` at spawn time moves the 4 motor hotspots to the exact
  arm tips whether the quad is `0.45 m` or `1.00 m`.
- **Why hotspots cannot bleed onto the sky.** The Gaussian heat bump is added only to pixels where
  `seg == base_seg_id`. Even if `radius_px` is wider than a thin carbon arm, sky and ground pixels
  around the arm are untouched.
- **Why `+K` is added before Beer-Lambert, not after.** Adding `+K` to the surface temperature means
  atmospheric fog (`tau = exp(-beta * d)`) dims the hotspot naturally with distance, and White-Hot
  maps the attenuated kelvin into `0..255` brightness.

---

## Milestone B3 — Camera rig, frustum spawn & PTZ tracker (`rig.py`)

Without B3, we have the mountain map (B1), the drone meshes + `+K` hotspots (B2), the IR compositor
(A1), and the YOLO label mapper (A2), but nothing that places the drone inside the camera's view,
slews the camera across frames, and connects the render buffers to A1 and A2. B3 is that glue.

### The 4 functions coded in B3 (with 4-5 line pseudocode)

#### 1. `camera_optics(cfg, modality, focal_mm)`
Computes the lens field of view (`fov_x_rad`, `fov_y_rad`), pixel focal length (`focal_px`),
angular pixel size (`ifov_mrad`), and crop-window FOV from the tier resolution and zoom lens:

```python
width_px, height_px = cfg.native_res(modality)
fov_x_rad = 2.0 * arctan(sensor_width_mm / (2.0 * focal_mm))
focal_px  = (width_px / 2.0) / tan(fov_x_rad / 2.0)
ifov_mrad = 1000.0 * fov_x_rad / width_px
return Optics(width_px, height_px, fov_x_rad, fov_y_rad, focal_px, ifov_mrad)
```

#### 2. `spawn_in_frustum(ground, ptz, optics, bucket_m, bg_mode="sky")`
Places the target inside the central `70 %` of the training-crop view cone at a distance `r_m` in
`bucket_m` (`300..5000 m`), and clamps altitude at least `15 m` above `ground.at(x, y)`:

```python
r_m = rng.uniform(bucket_m[0], bucket_m[1])
d_yaw, d_pitch = rng.uniform(-0.35 * optics.crop_fov_rad, +0.35 * optics.crop_fov_rad)
x, y  = ptz.cam_xyz[:2] + r_m * cos(ptz.pitch + d_pitch) * unit_vec(ptz.yaw + d_yaw)
z_ray = ptz.cam_xyz[2]  + r_m * sin(ptz.pitch + d_pitch)
z     = max(ground.at(x, y) + 15.0, z_ray)
```

#### 3. `step_ptz(ptz, target_xyz, dt_s, rng)`
Slews the camera pan/tilt toward the moving target with a real gimbal's max angular speed
(`max_rate_rad_s`), servo lag (`tau_lag_s`), and wind/tripod jitter:

```python
want_yaw, want_pitch, _ = look_angles(ptz.cam_xyz, target_xyz)
err_yaw  = wrap_pi(want_yaw - ptz.yaw_rad)
rate_yaw = clip(err_yaw / ptz.tau_lag_s, -ptz.max_rate_rad_s, +ptz.max_rate_rad_s)
jitter   = rng.normal(0.0, ptz.jitter_rad, size=2)
return ptz.with_angles(ptz.yaw_rad + rate_yaw * dt_s + jitter[0], ...)
```

#### 4. `render_rig_frame(backend, ptz=, eo_optics=, ir_optics=, actors=, sun=, rng=)`
Reads the split AOVs (`EO_AOVS` on EO, `IR_AOVS` on IR), projects each actor's 3D `+K` hotspots via
B2 (`project_hotspots`), runs A1 (`compose_ir`), and checks whether the target stayed inside the
training crop window (`in_crop`):

```python
eo_rgb, _                     = backend.read_eo(ptz, eo_optics, actors, EO_AOVS)
seg, normals, depth_m, boxes  = backend.read_ir(ptz, ir_optics, actors, IR_AOVS)
spots_2d = [s for a in actors for s in project_hotspots(a.asset, span_m=a.span_m, ...)]
ir_u8, ir_temp_k = compose_ir(seg, normals, depth_m, sun=sun, hotspots=spots_2d, debug=True)
return RigFrame(eo_rgb, ir_u8, ir_temp_k, boxes, in_crop)
```

### Demo command

```bash
uv run python -m app.datagen.rig --demo --out /tmp/rig_demo.png
```

Runs a 10-frame smoke clip over `GroundZ` with a moving `meshy_tail_fiber_fpv` quadcopter in the
`300-800 m` bucket, slewing the PTZ camera each step and verifying `in_crop = 10/10`.

### Acceptance tests

9 tests in `app/test/test_rig.py`:

- zooming (`focal_mm` increase) narrows `fov_x_rad` and `ifov_mrad` and increases `focal_px`
- non-positive `focal_mm` is rejected
- `spawn_in_frustum()` places the target inside the training crop window and `>= 15 m` AGL across all 5 range buckets (`300-5000 m`)
- `step_ptz()` slews toward the target while obeying `max_rate_rad_s * dt_s`
- `render_rig_frame()` requests `EO_AOVS` on EO and `IR_AOVS` on IR, keeps the target inside the crop window, and makes a quad (`+35 K` hotspots) hotter and brighter in IR than a passive decoy (`+8 K`)

### Concepts this module covers (plain language)

- **IFOV (`ifov_mrad`).** The angle covered by a single pixel, in milliradians (`1 mrad` = `1 m` at
  `1 km`). Zooming in makes `ifov_mrad` smaller, so the same `0.45 m` drone covers more pixels.
- **Frustum cone spawning.** If you spawned a drone at a random `(x, y, z)` on a `10 km x 10 km`
  map, a narrow zoom lens (`0.3 deg` FOV) would miss it `99.99 %` of the time. Instead, we shoot a
  ray out of the camera lens at distance `r_m` (`300..5000 m`) inside the central `70 %` of the crop
  window and place the drone there.
- **Why PTZ has servo lag and speed limits.** A real pan-tilt turret cannot teleport to face the
  drone instantaneously. Dividing the angle error by `tau_lag_s` and clamping to `max_rate_rad_s`
  makes fast close-range drones lead the crosshair slightly, while `jitter_rad` adds tripod/wind
  vibration.
- **Why AOVs are split (`EO_AOVS` vs `IR_AOVS`).** The EO camera renders at up to `2560x1440` and
  only needs `rgb` + boxes. The heavy geometry buffers (`instance_segmentation`, `normals`,
  `distance_to_image_plane`) are requested only on the smaller IR camera (`640x512` or `1280x1024`),
  cutting GPU-to-CPU memory readback by `~5x`.

---

## Milestone C — Dataset clip runner, `>= 95 %` crop check & track features (`generate.py`)

Turns the single-frame B3 rig into the unattended dataset generator (`C1` clip runner + sharding,
`C2` `>= 95 %` crop-containment gate + §5.2 temporal track features).

### The 4 functions coded in C (with 4-5 line pseudocode)

#### 1. `sample_clip_spec(clip_idx, cfg, catalog)`
Stratifies across the 5 range buckets (`300..5000 m`), samples the target class (`quad 50 %`,
`fixed_wing 20 %`, `decoy 15 %`, `bird 15 %`), model, wingspan `span_m`, sun elevation, visibility,
and assigns the `70 / 15 / 15` train/val/test split by terrain sector:

```python
bucket_m   = RANGE_BUCKETS_M[clip_idx % len(RANGE_BUCKETS_M)]
class_name = rng.choice(["quad", "fixed_wing", "decoy", "bird"], p=[0.50, 0.20, 0.15, 0.15])
asset      = rng.choice([a for a in catalog.values() if a.class_name == class_name])
span_m     = rng.uniform(*asset.span_range_m)
return ClipSpec(clip_idx, bucket_m, asset, span_m, bg_mode, sun, visibility_m, split=...)
```

#### 2. `simulate_clip_frames(spec, cfg, ground, backend, n_frames)`
Spawns the target(s) via `spawn_in_frustum()`, steps 3D motion + `step_ptz()` + `render_rig_frame()`
for `n_frames`, and regenerates with a tighter frustum margin if `in_crop_ratio < 0.95`:

```python
for attempt in range(max_attempts):
    frames, track_xyz = run_trajectory(spec, cfg, ground, backend, n_frames, attempt)
    in_crop_ratio = mean([f.in_crop for f in frames])
    if in_crop_ratio >= 0.95:
        return frames, track_xyz, in_crop_ratio, attempt + 1
```

#### 3. `compute_track_features(spec, frames, track_xyz, dt_s)`
Computes the §5.2 temporal and thermal-hotspot track features (`median_speed_mps`, `p90_speed_mps`,
`vertical_bob_amp_m`, `flap_hz`, `aspect_ratio_var`, `ir_hotspot_count`, `trajectory_curvature`):

```python
vel    = diff(track_xyz, axis=0) / dt_s
speeds = linalg.norm(vel, axis=1)
ar_var = var([w / max(h, 1e-6) for f in frames for _, _, _, _, w, h in f.boxes[:1]])
return {"median_speed_mps": median(speeds), "p90_speed_mps": percentile(speeds, 90),
        "flap_hz": spec.flap_hz, "ir_hotspot_count": len(spec.asset.hotspots), ...}
```

#### 4. `write_clip_bundle(clip_dir, spec=, cfg=, frames=, track_features=, in_crop_ratio=)`
Cuts the centred square training crop using `crop_origin()` (never resizing), writes `eo.mp4`,
`ir.mp4`, `labels.npz`, `metadata.json`, and calls `export_clip()` to write
`labels/frame_XXXX.txt`:

```python
eo_crops = [crop_square(f.eo_rgb, eo_ox, eo_oy, eo_crop_px) for f in frames]
ir_crops = [crop_square(f.ir_u8,  ir_ox, ir_oy, ir_crop_px) for f in frames]
write_video(clip_dir / "eo.mp4", eo_crops, cfg.fps); write_video(clip_dir / "ir.mp4", ir_crops, cfg.fps)
savez(clip_dir / "labels.npz", ...); write_json(clip_dir / "metadata.json", ...)
return export_clip(npz_cols, cfg=cfg, modality="ir", out_dir=clip_dir, n_frames=len(frames))
```

### Commands

```bash
uv run python -m app.datagen.generate --sensor tier2 --clips 5 --duration 0.5 --headless --out /tmp/datagen_smoke
uv run python -m app.datagen.generate --sensor tier2 --clips 50 --duration 8 --headless --shard 1/1
```

### Acceptance tests

5 tests in `app/test/test_generate.py`:

- `sample_clip_spec()` stratifies across all 5 `RANGE_BUCKETS_M` and assigns sector-based `train/val/test` splits
- `simulate_clip_frames()` enforces `in_crop_ratio >= 0.95`
- `compute_track_features()` outputs all 8 temporal & hotspot keys with `p90_speed_mps >= median_speed_mps > 0`
- `write_clip_bundle()` writes `eo.mp4`, `ir.mp4`, `labels.npz`, `metadata.json`, and `labels/frame_XXXX.txt`
- `parse_shard()` and `generate_dataset(..., shard="1/2")` partition clip indices across workers without overlap

### Concepts this module covers (plain language)

- **Why the `>= 95 %` crop-containment gate exists.** The camera renders at `16:9` (`1920x1080` or
  `2560x1440`), while training cuts out the central square (`1024x1024`). If a fast target drifts
  into the outer side strips for half the clip, half the training frames would have no visible
  target. Rejecting and retrying any trajectory below `95 %` guarantees every clip keeps the target
  inside the training square.
- **Why split `70 / 15 / 15` by terrain sector instead of by clip.** Two clips filmed over the same
  hilltop share the exact same background rocks and ridges. Splitting by `sector_id` guarantees the
  validation and test sets test on unseen terrain sectors rather than memorized rocks.
- **Why track features live in `metadata.json`.** At `2-5 km`, a drone and a bird are both `1-7 px`
  blobs in a single frame. Their motion across frames (`speed`, `2-6 Hz` wing-flap bob, trajectory
  curvature) and IR hotspot count separate them when a single frame cannot.

---

## Milestone D — Throughput benchmark & stratified pilot gate (`generate.py --benchmark`)

Measures real `fps` on one clip, computes the exact runtime hours for the `2,000`-clip pilot and
`20,000`-clip full dataset, and audits per-bucket pixel spans and `>= 95 %` crop containment.

### Where to run what (Mac vs. Isaac GPU PC)

1. **Mac (`uv run python -m app.datagen.generate --benchmark`):** runs the pipeline end-to-end with
   `SyntheticRigBackend` in seconds to verify bucket balance, crop math, and `pilot_report.json`.
2. **Isaac GPU PC (`/mnt/e/Isaac/env_isaaclab/Scripts/python.exe -m app.datagen.generate ...`):**
   runs the raytraced 3D `.glb` meshes + `map_10km.usda` on the NVIDIA GPU to measure the real GPU
   `N` fps and generate the `2,000` / `20,000` clip dataset.

### The 3 functions added in D (with 4-5 line pseudocode)

#### 1. `benchmark_throughput(cfg, out_root=, n_frames=250)` (D1)
Times one full clip and computes wall-clock GPU-hours for `2,000` and `20,000` clips:

```python
t0 = perf_counter()
generate_dataset(cfg=cfg, clips=1, duration_s=n_frames / cfg.fps, out_root=out_root)
fps_measured = n_frames / (perf_counter() - t0)
pilot_hours  = (2_000  * avg_frames) / (fps_measured * 3600.0)
full_hours   = (20_000 * avg_frames) / (fps_measured * 3600.0)
```

#### 2. `audit_bucket_spans(results, cfg)` (D1/D2 gate)
Checks each clip's rendered IR bounding-box pixel width against the optical formula
`expected_px = (span_m / mid_range_m) * focal_px`:

```python
for r in results:
    measured_px = median(load_labels(r.clip_dir / "labels.npz")["w"]) * ir_width_px
    expected_px = max((r.spec.span_m / mean(r.spec.bucket_m)) * ir_opt.focal_px, 2.0)
    ratio       = measured_px / expected_px
    in_bounds   = (0.35 <= ratio <= 2.8) and (r.in_crop_ratio >= 0.95)
```

#### 3. `pilot_report(out_root)` (D2)
Audits all generated clips on disk, checks 5-bucket balance and `min_in_crop_ratio >= 0.95`, and
writes `pilot_report.json`:

```python
metas = [json.loads(p.read_text()) for p in sorted(out_root.glob("clip_*/metadata.json"))]
bucket_counts = Counter(f"{int(m['bucket_m'][0])}-{int(m['bucket_m'][1])}m" for m in metas)
class_counts  = Counter(m["class_name"] for m in metas)
min_in_crop   = min(m["in_crop_ratio"] for m in metas)
return {"clips": len(metas), "bucket_counts": bucket_counts, "pass_gate": min_in_crop >= 0.95}
```

### Command

```bash
uv run python -m app.datagen.generate --benchmark --bench-frames 30 --clips 5 --duration 0.2 --out /tmp/datagen_d_bench
```
