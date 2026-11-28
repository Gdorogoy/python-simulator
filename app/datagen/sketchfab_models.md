# Curated Sketchfab Models for `app/datagen/` (Milestone B2)

Source collection inspected:
`https://sketchfab.com/bearatcn/collections/04939948b70145d6bc3309e51f64bf5f-12f0d5568c1445adbbdda198f67aae66`
(68 models total; items 1–17 and 23–29 are drones, items 18–68 are mechs/robots/props and skipped).

---

## 1. Technical Rule for Thermal Compositing (`ir_composite.py`)

`ir_composite.py` requires distinct instance/part masks for:
- **Hotspots:** `SEG_QUAD_ROTOR` (`+35 K`, `epsilon = 0.95`) or `SEG_FIXED_WING_EXHAUST` (`+60 K`, `epsilon = 0.95`)
- **Cool body:** `SEG_QUAD_FRAME` (`+2 K`, `epsilon = 0.85`)
- **Bird parts:** `SEG_BIRD_CORE` (`+12 K`) vs `SEG_BIRD_WING` (`+1 K`)

When importing external Sketchfab meshes into `assets.py`:
1. **Multi-object CAD/Blender models** (e.g., `92152c31...` with 5 separate objects) map their 4 rotor sub-meshes directly to `SEG_QUAD_ROTOR` and the body to `SEG_QUAD_FRAME`.
2. **Single-mesh AI models** (`[Meshy]`, `[Rodin]`, `[Krea.AI]`) are fused into 1 mesh. For these, `assets.py` uses the mesh as `SEG_QUAD_FRAME` for EO visual fidelity and overlays 4 small procedural motor-bell cylinders at the arm tips tagged `SEG_QUAD_ROTOR` so the IR hotspots stay physically accurate.

---

## 2. Class 0 — `quad` (`0.45–1.0 m` tip-to-tip incl. propellers)

All picks below are from `bearatcn`'s collection and are **free to download** under Creative Commons licenses.

| Priority | Name | UID | Faces | License | Topology / Notes | URL |
| --- | --- | --- | --- | --- | --- | --- |
| **1 (Primary)** | **FPV Drone** | `92152c31ec9a4a87b2a0478bd114d293` | `27,946` | CC-BY | **5 separate objects** (frame + 4 rotors). Cleanest direct mapping to `SEG_QUAD_FRAME` + `SEG_QUAD_ROTOR`. | https://sketchfab.com/3d-models/fpv-drone-92152c31ec9a4a87b2a0478bd114d293 |
| **2 (Combat FPV)** | **FPV kamikaze drone Low-poly** | `87f5bbb5b08641b782bc084ddd4082a7` | `24,460` | CC-BY | Blender 4.3 model with removable under-slung anti-tank warhead and top battery. | https://sketchfab.com/3d-models/fpv-kamikaze-drone-low-poly-87f5bbb5b08641b782bc084ddd4082a7 |
| **3 (Close EO `300–800 m`)** | **Combat FPV drone** | `df640a20c2af49a6af7c371b2d49d51f` | `39,716` | CC-BY | Improvised combat FPV with RPG-7 warhead and nose wire trigger; strong close-range visual detail. | https://sketchfab.com/3d-models/combat-fpv-drone-df640a20c2af49a6af7c371b2d49d51f |
| **4 (Fiber-optic spool)** | **[Meshy] Tail FPV Optic Fibre Drone** | `0f037de4489b4c6eaae3edc4d9ae23b2` | `24,015` | CC-BY | Single AI mesh with prominent rear fiber-optic guidance spool (`plan.md` §2 threat band). Overlay 4 procedural motor hotspots for IR. | https://sketchfab.com/3d-models/meshy-tail-fpv-optic-fibre-drone-0f037de4489b4c6eaae3edc4d9ae23b2 |
| **5 (Fiber-optic spool alt)** | **[Rodin] Long FPV Optic Fibre Drone** | `b7ca3ae3f8294752bcab959fb3b756b4` | `38,354` | CC-BY | Single AI mesh, elongated fiber-optic spool + dropper frame. | https://sketchfab.com/3d-models/rodin-long-fpv-optic-fibre-drone-b7ca3ae3f8294752bcab959fb3b756b4 |
| **6 (Distant LOD `2–5 km`)** | **Fpv Drone (Low-poly)** | `30f38473936b4b0b96642b28304e98f5` | `3,234` | CC-BY | Ultra-lightweight (`3.2k` faces); ideal at `2000–5000 m` where the drone subtends `1–7 px`. | https://sketchfab.com/3d-models/fpv-drone-30f38473936b4b0b96642b28304e98f5 |
| **7 (Large `1.0 m` bound)** | **Vampire UAV (BabaYaga)** | `f49e65781b32465c86841617e82fcae7` | `16,254` | CC-BY-NC-SA | Heavy multirotor for the `1.0 m` upper end of the quad span band. | https://sketchfab.com/3d-models/vampire-uav-babayaga-f49e65781b32465c86841617e82fcae7 |
| **8 (Commercial quad)** | **FPV-dron_NonStop** | `c75dea6e3ae441ac87f292efb17f5bae` | `24,713` | CC-BY | Handcrafted 4K textures, standard 7–10 inch FPV silhouette. | https://sketchfab.com/3d-models/fpv-dron-nonstop-c75dea6e3ae441ac87f292efb17f5bae |

---

## 3. Class 1 — `fixed_wing` (`1.2–2.5 m` span, `+60 K` exhaust)

Found via Sketchfab downloadable search to match the `1.2–2.5 m` tactical/loitering wingspan band:

| Priority | Name | UID | Faces | License | Notes | URL |
| --- | --- | --- | --- | --- | --- | --- |
| **1 (Distant LOD)** | **Lancet Zala-52 (Low-poly)** | `e6bbd4ae1c3543339160317fb111b502` | `1,201` | CC-BY | Cruciform X-wing loitering munition (`~1.6 m` span), fast raytracing. | https://sketchfab.com/3d-models/none-e6bbd4ae1c3543339160317fb111b502 |
| **2 (Close EO)** | **Lancet** | `311f824410b840cf8da15f951b791e01` | `14,716` | CC-BY | Detailed X-wing silhouette for `300–1200 m`. | https://sketchfab.com/3d-models/none-311f824410b840cf8da15f951b791e01 |
| **3 (Recon wing `2.0 m`)** | **DeViRo Leleka-100 Fixed-wing UAV** | `ac9fafc59bbe4991820bf21a3f3b1d17` | `35,830` | CC-BY | Classic high-wing recon UAV (`~2.0 m` span). | https://sketchfab.com/3d-models/none-ac9fafc59bbe4991820bf21a3f3b1d17 |
| **4 (Recon wing `2.5 m`)** | **Orlan-10** | `e2efb223c1814b7291818a9a7b4e80c5` | `37,636` | CC-BY | Nose engine/exhaust (`+60 K` hotspot) + high-wing fuselage. | https://sketchfab.com/3d-models/none-e2efb223c1814b7291818a9a7b4e80c5 |

---

## 4. Class 3 — `bird` (`0.25–1.0 m` span, `2–6 Hz` flap, flocks of `1–15`)

| Priority | Name | UID | Faces | Animations | Notes | URL |
| --- | --- | --- | --- | --- | --- | --- |
| **1 (Flock LOD)** | **Bird Flying Animation** | `61e8ff53d4e845d5b0aeaaf39b5869ef` | `174` | `1` flap cycle | Only 174 faces — 15-bird flock costs `<3,000` faces total; flap speed scales to `2–6 Hz`. | https://sketchfab.com/3d-models/none-61e8ff53d4e845d5b0aeaaf39b5869ef |
| **2 (Close EO)** | **Flying Seagull** | `07dde3ea7d9048588d3c4edfd37ac20d` | `1,182` | `3` cycles | `~0.9 m` wingspan, good silhouette for `300–800 m`. | https://sketchfab.com/3d-models/none-07dde3ea7d9048588d3c4edfd37ac20d |

---

## 5. Class 2 — `decoy` (`0.2–1.5 m`, `+8 K`, clusters of `1–15`)

- Decoys (balloons, corner-reflector gliders, styrofoam dummy wings) are best spawned procedurally or by re-skinning the low-poly `Lancet Zala-52` (`1,201` faces) / sphere-cone primitives with a **single uniform material (`SEG_DECOY`, `+8 K`, no `+35 K`/`+60 K` motor hotspot)**.
