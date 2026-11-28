# Datagen Implementation Guide: Foundations, Formulas & Commands

**State:** `app/datagen/config.py` and `app/datagen/plan.md` are committed (`05050f0`). No
implementation files exist yet. This guide covers the physics, math and 1–4 formulas/commands
needed to build each remaining module in `app/datagen/`.

**Format note:** all formulas are plain ASCII inside fenced blocks, so they render identically in
a terminal, VS Code, GitHub and Obsidian. Variable names match the Python code you will write
(`t_amb`, `sigma_netd`, `ifov`), so the formula is also a naming spec.

---

## 0. The three unconditional truths

Lock these in first. Every formula below is a consequence of one of them.

1. **Every synthetic image is a 2D array of numbers derived from 3D geometry.**
   EO is `(H, W, 3)` reflected light. IR is `(H, W)` emitted heat. Isaac Sim simulates visible
   light only, so **every IR frame here is computed from Isaac's geometry masks plus a heat
   equation in NumPy** — it is never rendered.

2. **Every zoom, resolution and crop rule reduces to one invariant: angle per pixel.**
   ```
   ifov = pixel_pitch / focal_length        [radians per pixel]
   metres_covered_by_one_pixel = R * ifov   [R = slant range]
   ```
   Resizing an image averages neighbouring pixels, which changes `ifov`. Cropping cuts the outer
   margins and leaves `ifov` untouched — which is why §3.5 crops and never resizes.

3. **Throughput is decided at the GPU→CPU boundary.**
   One frame of heavy AOVs at 2560×1440 is ~200 MB copied across PCIe. At 6.8 M frames that is
   the dominant cost, not the shading. So: never request an AOV larger than its consumer needs,
   and never run CPU math on the render thread.

### Dependency map

```
[Truth 1: IR = masks + heat math] ---> ir_composite.py
                                  |--> terrain_builder.py   (heightmap + ground_z.npy)
                                  '--> assets.py            (part temperature table)

[Truth 2: ifov = pitch / focal, invariant under crop]
                                  |--> rig.py               (optics, spawn cone, PTZ)
                                  '--> export_yolo.py       (native -> crop coordinate remap)

[Truth 3: GPU->CPU boundary is the bottleneck]
                                  |--> rig.py               (AOVs: heavy ones on IR camera only)
                                  '--> generate.py          (sharding, worker pool, encoding)
```

---

## Module 1 — `app/datagen/terrain_builder.py`

**Why.** Rebuilding a 10×10 km mesh per clip costs 3–5 s × 20,000 clips ≈ 25 h of wasted CPU.
Raycast AGL checks are also slow. So: build the map once, export `map_10km.usd`, and save a
1000×1000 elevation grid to `ground_z.npy` for O(1) altitude lookups.

**The four formulas.**

1. Ridged multifractal — sharp limestone crests instead of rounded hills:

```
z_ridge(x, y) = SUM over k = 0..5 of  w^k * (1 - |n(x * 2^k / lambda_0, y * 2^k / lambda_0)|)^2

    lambda_0 = 4096 m       base wavelength
    w        = 0.5          gain per octave
    n(.)     = simplex/perlin noise in [-1, 1]
    |.|      makes the zero crossings sharp ridges
```

2. Domain-warped wadi — a straight valley reads as fake, so the centreline follows noise:

```
x_c(y)   = A * n(y / lambda_w)                                    A = 120 m

z_wadi(x, y) = z_ridge(x, y) - D * exp( -(x - x_c(y))^2 / (2 * sigma_w^2) )

    D        = 20..60 m     channel depth
    sigma_w  = channel half-width
```

3. Terrace quantisation on mid-slopes (20°..38°) — reads as Mediterranean terracing:

```
z_terrace = (1 - alpha) * z_wadi + alpha * step_h * floor(z_wadi / step_h)

    step_h = 8..15 m
    alpha  = 0 below 20 deg slope, 1 above 38 deg, ramped between
```

4. Bilinear heightmap lookup — nearest-neighbour quantises AGL by up to 5 m, which matters for a
   drone at 15 m AGL:

```
u = (x + 5000) / 10        v = (y + 5000) / 10
i = floor(u)   s = u - i   j = floor(v)   t = v - j

z_ground(x, y) = (1-s)*(1-t) * Z[i,   j  ]
               +      s *(1-t) * Z[i+1, j  ]
               + (1-s)*     t  * Z[i,   j+1]
               +      s *    t  * Z[i+1, j+1]
```

**Build commands.**

```bash
python -m app.datagen.terrain_builder --octaves 6 --wavelength 4096 --terraces
# writes app/datagen/ground_z.npy + app/datagen/map_10km.usd (USD export needs Isaac's pxr)
```

---

## Module 2 — `app/datagen/assets.py`

**Why.** At 2–5 km a 0.45 m quad and a 0.5 m bird are both 1.7–4 px. Shape cannot separate them.
Two things survive at that scale: **where the heat is concentrated**, and **how the object moves
over 5–10 s**.

**The three formulas.**

1. Per-part thermal table — what the IR compositor reads:

```
epsilon_rotor      = 0.95    delta_t_rotor      = +35 K
epsilon_frame      = 0.85    delta_t_frame      = +2  K
epsilon_bird_core  = 0.98    delta_t_bird_core  = +12 K
epsilon_bird_wing  = 0.98    delta_t_bird_wing  = 0  K   (cooler, pulsing)

t_part = t_amb + delta_t_part
```

2. Bird kinematics — sinusoidal bobbing with the wing flap lagging by 90°:

```
z(t)   = z_0 + A_z * sin(2 * pi * f * t)          f = 2..6 Hz
phi(t) = phi_max * cos(2 * pi * f * t)            A_z = vertical bob amplitude
```

3. Apparent pixel span — this is what the 2×2 px rule measures:

```
N_px = S / (R * ifov)

    S = 0.45..1.0 m   tip-to-tip incl. propellers
    if N_px < 2 in IR -> metadata["unlabeled_targets"], not labels.npz
```

**Sanity check you can run by hand:** 0.45 m at 5 km on Tier 2 IR
(`ifov = 0.0533 mrad`) gives `0.45 / (5000 × 0.0000533) ≈ 1.7 px`. At 2 km it is 4.2 px. That is
the whole reason behind the degraded-cue rule in §2.

---

## Module 3 — `app/datagen/rig.py`

**Why.** A 3.9°–4.4° FOV aimed at a 10×10 km map sees empty sky ~99 % of the time if targets spawn
uniformly. And heavy AOVs on the EO camera waste ~5× the readback.

**The four formulas / rules.**

1. Optics — derive FOV from sensor geometry, never hardcode it:

```
sensor_w_mm = N_x * pitch_um / 1000
hfov        = 2 * atan( sensor_w_mm / (2 * f_mm) )
ifov        = pitch_um / f_mm                     # um/mm -> mrad

Tier 1 IR: 12/100 = 0.12   mrad/px
Tier 2 IR: 12/225 = 0.0533 mrad/px
```

2. Frustum-cone spawn — sample inside the cone, then clamp altitude:

```
p_target(0) = p_cam + R * R_rot(d_az, d_el) * u_hat

    R            ~ U(R_min, R_max)
    d_az, d_el   ~ U(-0.6 * fov, +0.6 * fov)
    AGL          = z_target - z_ground(x, y)   clamped to [15, 300] m
```

3. Rate-limited PTZ — the camera cannot teleport to the target:

```
psi_dot = clip(Kp * psi_err + Kd * psi_err_dot, -w_max, +w_max)
psi_cam = quantize(psi) + eta_wind(t)

    w_max   = 150 deg/s
    quantize= 0.003 deg pan / 0.001 deg tilt   (harmonic drive)
    eta_wind= camera-only jitter, never applied to the target
```

4. AOV split — the throughput rule from Truth 3:

```python
rp_eo = rep.create.render_product(cam_eo_path, cfg.native_res("eo"))
for name in EO_AOVS:          # rgb, bounding_box_2d_tight, camera_params
    rep.annotators.get(name).attach([rp_eo])

rp_ir = rep.create.render_product(cam_ir_path, cfg.native_res("ir"))
for name in IR_AOVS:          # seg, normals, depth, motion_vectors, boxes, camera_params
    rep.annotators.get(name).attach([rp_ir])
```

---

## Module 4 — `app/datagen/ir_composite.py`

**Why.** This is issue #9. You cannot make IR by inverting the RGB image: a white cold airframe is
bright in RGB and dark in IR, a matte-black hot motor is dark in RGB and white-hot in IR. So the
thermal signal is built in **temperature space** from geometry AOVs, noise is added there, and only
then is it mapped to 8-bit.

**The four formulas.**

1. Background temperature — solar heating plus the dawn/dusk crossover:

```
cos_inc  = max(0, dot(n_hat(u,v), s_hat))
t_bg(u,v) = t_amb
          + delta_t_solar(alpha_sun) * cos_inc
          + delta_t_inertia(material_id(u,v))

    alpha_sun = sun elevation
    at alpha_sun ~ 0 (dawn/dusk) delta_t_solar -> 0, so rock == air (thermal crossover)
```

2. Per-part radiance + Beer–Lambert transmittance:

```
beta          = 3.912 / V_vis                V_vis = 3000..20000 m
tau(u,v)      = exp(-beta * d(u,v))          d from distance_to_image_plane

t_scene(u,v)  = t_bg where no target, t_amb + delta_t_part inside mask M_k
t_sensor(u,v) = tau * (epsilon * t_scene + (1 - epsilon) * t_amb)
              + (1 - tau) * t_atm
```

3. Sensor noise in temperature space — before display mapping, never after:

```
t_noisy = t_sensor + N(0, sigma_netd^2) + fpn(u, v)

    sigma_netd = 0.050 K   (NETD <= 50 mK)
    fpn        = static per-pixel offset, fixed per clip
```

4. Fixed-gain White-Hot mapping — no per-frame AGC, or absolute contrast cues vanish:

```
I = round( 255 * clip( (t_noisy - t_min) / (t_max - t_min), 0, 1 )^gamma )

    t_min, t_max, gamma fixed for a run
```

**Heat shimmer** is a whole-frame 2D warp, applied to the composite so target and background move
together:

```
u' = u + du(u, v, t)      v' = v + dv(u, v, t)
amplitude ~ focal_length * (t_ground - t_air)   , disabled below 300 mm focal
```

---

## Module 5 — `app/datagen/generate.py`

**Why.** 6.8 M frames cannot be composited and encoded synchronously on the render thread, and a
2 px dot at 4 km is only classifiable from its motion.

**The four rules / formulas.**

1. Shutter law and frame count — derived, never configured:

```
t_shutter = min(t_sampled, 1 / fps)
n_frames  = round(duration_s * fps)

    fps 60 (Tier 1) -> 300..600 frames per clip
    fps 30 (Tier 2) -> 150..300 frames per clip
```

2. Square centre crop + containment check (§3.5, §5.1):

```
C  = cfg.train_crop_px(modality)        # 512 (Tier 1 IR) or 1024 (EO, Tier 2 IR)
x0 = (W - C) // 2                       y0 = (H - C) // 2

inside_ratio = n_inside / n_visible
if inside_ratio < 0.95: discard and regenerate the clip
```

3. Track features — the answer to issue #11, computed over one track:

```
speed_med = median(||v(t)||)        speed_p90 = p90(||v(t)||)
f_peak    = argmax over f in [1, 10] Hz of |FFT(z_tilde)(f)|
kappa     = median( ||v x a|| / ||v||^3 )
track_id survives an occlusion gap <= 2.0 s
```

4. Sharded CLI — one process per GPU, N cameras per process:

```bash
# pilot: 2,000 stratified clips (400 per bucket) to measure N fps
python -m app.datagen.generate --sensor tier1 --clips 1000 --shard 0/2 --headless
python -m app.datagen.generate --sensor tier2 --clips 1000 --shard 1/2 --headless

# anything that does not need Isaac
uv run pytest app/test/test_datagen_config.py -q
```

---

## Module 6 — `app/datagen/export_yolo.py`

**Why.** 6.8 M individual `.txt` files exhaust inodes and kill disk I/O. Store one `labels.npz` per
clip (20,000 files) and materialise Ultralytics `.txt` only when a training run needs it.

**The coordinate remap** — native raster `(W, H)` → square crop `(C, C)`:

```
c_x' = ( c_x * W - (W - C) / 2 ) / C
c_y' = ( c_y * H - (H - C) / 2 ) / C
w'   = ( w   * W ) / C
h'   = ( h   * H ) / C
```

```python
data = np.load(clip_dir / "labels.npz")   # frame, class_id, cx, cy, w, h, track_id, occluded
# remap as above, then emit labels/frame_XXXX.txt per frame
```

**Check:** a box at `c_x = 0.5, c_y = 0.5` must land at `0.5, 0.5` after the remap for any `W, H, C`.
That is a one-line unit test worth writing first.

---

## Build order and what blocks what

| # | Module | Needs Isaac? | Blocks |
| --- | --- | --- | --- |
| 1 | `ir_composite.py` | no — NumPy only | #9, `assets.py` tuning |
| 2 | `terrain_builder.py` (heightmap half) | no (USD export: yes) | `rig.py` |
| 3 | `export_yolo.py` | no | training loader |
| 4 | `assets.py` | yes | `rig.py` |
| 5 | `rig.py` | yes | `generate.py` |
| 6 | `generate.py` | yes | the 20k run |

Start at 1: it is the issue #9 blocker, it is testable with synthetic masks on any machine, and it
is the only piece whose quality cannot be verified by reading code.
