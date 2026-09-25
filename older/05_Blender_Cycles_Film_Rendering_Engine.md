# Blender Cycles Film-Grade Rendering Engine (Production Specification)

> **Document:** DOCS/05_Blender_Cycles_Film_Rendering_Engine.md  
> **Status:** ACTIVE PRODUCTION SPECIFICATION  
> **Target Environment:** Blender 3.6+ / 4.x Headless & Interactive Python API (`bpy`)

---

## 1. Overview

A recurring point of failure in Research 1 was evaluating 3D head meshes using primitive 2D polygon projection or headless OpenCV renders without physical shading.

This specification defines the **Blender Cycles Automated Studio Engine** (`src/stage5_export/blender_film_render.py`). When the pipeline completes an asset, it automatically constructs a production `.blend` scene containing:
1. The canonical 3D head mesh with Adaptive Micro-Polygon Subdivision enabled.
2. A complete PBR Skin Shader node tree utilizing **Random Walk (Skin) Subsurface Scattering**.
3. A cinematic 3-point studio lighting rig matching film look-dev turnarounds (as seen in the user's reference image).
4. An 85mm portrait camera with subtle depth-of-field.
5. An automated high-resolution Cycles render output.

---

## 2. Shader Node Architecture (`Principled BSDF`)

The Blender material tree is constructed programmatically via Python:

```
[Textures: 1024 / 4096]
  │
  ├── albedo.png ─────────────> Base Color (sRGB)
  │
  ├── roughness.png ──────────> Roughness (Non-Color, Linear)
  │                              ├── Base Roughness: [0.45 - 0.60]
  │                              └── Coat Roughness: [0.15 - 0.25] via T-zone mask
  │
  ├── sss_thickness.png ──────> Subsurface Weight & Radius Scale
  │                              ├── Subsurface Method: RANDOM_WALK_SKIN
  │                              ├── Subsurface Weight: 0.18
  │                              ├── Subsurface Radius: (1.0, 0.25, 0.12)  [Red blood scatter]
  │                              └── Subsurface IOR: 1.40
  │
  ├── normal.png ─────────────> [Normal Map Node (Tangent)] ──> Normal
  │
  └── displacement.png ───────> [Displacement Node] ─────────> Material Output: Displacement
                                 ├── Midlevel: 0.50
                                 ├── Scale: 0.005 (5.0 mm maximum displacement)
                                 └── Space: Object Space / Height
```

### Adaptive Micro-Polygon Subdivision Configuration:
To resolve 50-micron pores directly in the 3D silhouette without storing 100-million-polygon OBJ files on disk:
```python
# Cycles Adaptive Subdivision in Blender Python API
head_obj.modifiers.new(name="Subsurf", type='SUBSURF')
head_obj.modifiers["Subsurf"].subdivision_type = 'CATMULL_CLARK'
head_obj.modifiers["Subsurf"].use_adaptive_subdivision = True
head_obj.cycles.use_adaptive_subdivision = True
scene.cycles.dicing_rate = 1.0  # 1 polygon per screen pixel at render time
```

---

## 3. Cinematic Studio Lighting Rig

To match the dramatic depth, rim shadows, and specular pop seen in the reference MetaHuman viewport:

```
                      [Rim / Sun Lamp]
                       (High Energy, Cool White,
                        Sharp Angle from Behind/Side)
                               │
                               ▼
                        ┌─────────────┐
                        │   3D HEAD   │
                        │    ASSET    │
                        └─────────────┘
                         ▲           ▲
                         │           │
       [Key Light] ──────┘           └────── [Fill Light]
    (Warm 45° Front-Left,                  (Soft Ambient, Cool Blue,
     Medium Spread, Shadow)                 Low Intensity, Shadow Off)
                               ▲
                               │
                          [Camera]
                       (85mm Portrait Lens)
```

### Lighting Parameters:

| Lamp Name | Type | Power / Energy | Color (Kelvin / Hex) | Angle / Position | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Key Light** | Area Lamp | $180\,\text{W}$ | $4500\,\text{K}$ (Warm White) | $+45^\circ$ Yaw, $+30^\circ$ Pitch, $1.5\,\text{m}$ | Primary directional illumination & facial volume definition |
| **Fill Light**| Area Lamp | $45\,\text{W}$ | $6500\,\text{K}$ (Soft Daylight) | $-45^\circ$ Yaw, $+10^\circ$ Pitch, $2.0\,\text{m}$ | Softens dark side; prevents total shadow crushing |
| **Rim / Sun** | Sun / Spot | $350\,\text{W}$ | $5500\,\text{K}$ (Neutral White)| $+135^\circ$ Yaw, $+45^\circ$ Pitch, High Angle | Separates jawline/neck from dark background; creates razor rim |

---

## 4. Camera & Viewport Setup

* **Focal Length:** $85\,\text{mm}$ (Industry-standard prime portrait lens; avoids focal distortion or fish-eye flattening of facial proportions).
* **Sensor Size:** $36.0 \times 24.0\,\text{mm}$ (Full-frame 35mm sensor).
* **Focus Distance:** Constrained to the tip of the nose / pupils with shallow depth of field ($f/2.8$).
* **Color Management:**
  * **View Transform:** `AgX` or `Filmic` (High dynamic range specular rolloff; avoids blown-out clipping on oily skin highlights).
  * **Look:** `Medium High Contrast`.

---

## 5. Automated Python Execution Interface

The engine can be invoked automatically at the end of the production pipeline or standalone:

```bash
# Execute headless Blender render on any exported production asset
python scripts/render_blender_film.py \
    --mesh outputs/production_run/carell/head_mesh_neutral.obj \
    --textures_dir outputs/production_run/carell/textures/ \
    --output outputs/production_run/carell/film_render_cycles.png \
    --save_blend outputs/production_run/carell/studio_scene.blend \
    --samples 128 \
    --resolution 2048 2048
```

This guarantees that both the rendered turnaround image and the fully configured, editable `.blend` project file are delivered directly to the user.
