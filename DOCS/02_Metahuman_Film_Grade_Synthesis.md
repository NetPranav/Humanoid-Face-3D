# Film-Grade MetaHuman Detail & PBR Synthesis (Production Architecture)

> **Document:** DOCS/02_Metahuman_Film_Grade_Synthesis.md  
> **Status:** ACTIVE PRODUCTION SPECIFICATION  
> **Objective:** Bridge the fidelity gap to achieve photorealistic, film-grade digital human detail matching Epic Games MetaHuman and VFX studio standards (as seen in Blender Cycles / ZBrush reference).

---

## 1. The Multi-Tier Anatomy of a Film-Grade Digital Human

To achieve visual parity with top-tier cinematic renders, the reconstruction pipeline abandons monolithic generative hallucination and adopts the industry-standard **4-Tier Hierarchical Decomposition**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   4-TIER HIERARCHICAL DECOMPOSITION                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  TIER 1: MACRO PROPORTIONS (Cranial & Facial Bone Landmark Mesh)       │
│  ├── Model: FLAME 2020 + MICA 300-D Identity Regression                │
│  ├── Resolution: Canonical Base Mesh (5,023 vertices, 9,976 triangles) │
│  └── Output: Metric Skull, Jawline, Eye Sockets, Nose Bridge Proportions│
│                                                                        │
│  TIER 2: MESO GEOMETRY (Person-Specific Anatomical Wrinkles & Folds)   │
│  ├── Source: Direct Multi-View Photometric Shape-from-Shading (SfS)   │
│  ├── Resolution: 1024×1024 Displacement & Tangent Normal Delta Maps    │
│  └── Output: Crow's Feet, Nasolabial Sulcus, Forehead Furrows, Neck    │
│              Tendon Definition (Sternocleidomastoid & Larynx)          │
│                                                                        │
│  TIER 3: MICRO DETAIL (50-Micron Epidermal Pores & Cellular Follicles) │
│  ├── Method: Anatomical Zone Synthesis (Texturing.xyz / MetaHuman spec)│
│  ├── Resolution: 4096×4096 / 1024×1024 Multi-Channel Displacement      │
│  └── Output: Circular T-Zone Follicles, Elliptical Cheek Pores,        │
│              Directional Forehead Micro-Creases, Vertical Lip Ridges   │
│                                                                        │
│  TIER 4: FILM-GRADE PBR SHADING (Optical Light-Tissue Interaction)     │
│  ├── Engine: Blender Cycles Path Tracing / UE5 Subsurface Profile      │
│  ├── Components:                                                       │
│  │   ├── Random Walk Subsurface Scattering (Hemoglobin Translucency)   │
│  │   ├── Dual-Lobe Specular Roughness (Stratum Corneum + Sebum Coat)   │
│  │   ├── Micro-Cavity & Ambient Occlusion (Pore Light Trapping)        │
│  │   └── Delighted Diffuse Albedo (Lighting/Flash Stripped from Photos)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tier 2: Photo-Derived Meso Detail (Shape-from-Shading)

### The Principle:
The subject's actual portrait photographs contain sub-millimeter recordings of their unique wrinkles and skin folds. Instead of discarding this data, Tier 2 extracts meso-geometry directly from the multi-view photos:

1. **Multi-View Backprojection:**  
   Input portraits (frontal, left 45°, right 45°) are backprojected onto the FLAME UV space with angle-weighted cosine blending and z-buffer visibility testing (`src/stage6_texture/`).
2. **Photometric Gradient Decomposition:**  
   Using a multi-scale steerable filter / shape-from-shading estimator:
   $$I_{\text{high-pass}} = I_{\text{albedo}} - G_{\sigma_{\text{meso}}} * I_{\text{albedo}}$$
   Where $\sigma_{\text{meso}} \approx 12\,\text{px}$.
3. **Surface Normal Integration:**  
   High-pass intensity gradients are mapped to surface normal deviations $\Delta \vec{n}$ via the calibrated camera direction vectors:
   $$\vec{n}_{\text{meso}}(u, v) = \text{Normalize}\left(\vec{n}_{\text{base}}(u, v) + \alpha_{\text{meso}} \cdot \left(\frac{\partial I}{\partial u}\vec{t}_u + \frac{\partial I}{\partial v}\vec{t}_v\right)\right)$$
4. **Result:**  
   The subject's *exact* crow's feet, laugh lines, and brow furrows are reconstructed with 1:1 fidelity to their real-life photograph, without guessing or hallucinating.

---

## 3. Tier 3: Anatomical Micro-Pore Synthesis (Texturing.xyz Standard)

### The Principle:
Human facial pores are not random noise. Their scale, orientation, and depth vary strictly according to anatomical skin tension lines and sebaceous gland density.

Using the anatomical zone masks from `src/stage8_pbr/material_stack.py`:

```
               [Forehead: Stretched Horizontal Pores]
                                 │
     [Nose / T-Zone:             │             [Temple: Smooth,
      Large Circular Follicles] ─┼─            Fine Micro-Lines]
                                 │
                 [Cheeks: Fine Elliptical Pores]
                                 │
               [Lips: Vertical Dermal Striations]
                                 │
                 [Chin: Dense Circular Pores]
                                 │
               [Neck: Horizontal Tension Bands]
```

### Anatomical Zone Parameters:

| Facial Zone | Pore Geometry | Physical Depth Range | Orientation Constraint |
| :--- | :--- | :--- | :--- |
| **Nose / T-Zone** | Dilated circular sebaceous pores | $0.15\,\text{mm}$ to $0.35\,\text{mm}$ | Isotropic |
| **Cheeks** | Fine elliptical micro-pores | $0.05\,\text{mm}$ to $0.12\,\text{mm}$ | Follows Langer's lines (gravity/chewing tension) |
| **Forehead** | Fine porous background + directional furrow bands | $0.08\,\text{mm}$ to $0.22\,\text{mm}$ | Transverse horizontal |
| **Lips (Vermilion)**| Vertical micro-folds & papillary ridges | $0.10\,\text{mm}$ to $0.25\,\text{mm}$ | Strictly vertical (orthonormal to lip contour) |
| **Neck & Collar** | Dermal tension wrinkles ($\Delta v = 0$ at collar) | $0.10\,\text{mm}$ to $0.30\,\text{mm}$ | Concentric circular |

### Synthesis Engine:
Multi-octave cellular Voronoi-Worley basis functions combined with Gabor directional wavelets, modulated by the anatomical zone masks $M_{\text{zone}}(u, v)$:
$$D_{\text{micro}}(u, v) = \sum_{z \in \text{Zones}} M_z(u, v) \cdot \mathcal{F}_z\left(u, v; \text{scale}_z, \text{depth}_z, \theta_z\right)$$

---

## 4. Tier 4: Film-Grade PBR Shading (Why It Looks Real in Blender)

The visual chasm between a flat grey clay preview and the reference MetaHuman screenshot is primarily **shading and lighting physics**:

### 1. Random Walk Subsurface Scattering (SSS)
* **The Physics:** Skin is not opaque plastic. Photons penetrate the epidermis, undergo multiple anisotropic scattering events through the dermis (collagen and hemoglobin blood cells), and exit at a different point.
* **Blender Cycles Configuration:**
  * **Method:** `Random Walk (Skin)`
  * **Subsurface Weight:** $0.15 - 0.22$
  * **Subsurface Radius (Scatter Color):** `(1.0, 0.25, 0.12)` $\implies$ Deep red scatter for vascular tissue.
  * **Subsurface IOR:** $1.40$
  * **Thickness Mask:** SSS map derived via geometric opposing-normal ray-march in `src/stage8_pbr/` (high scatter through ears, nostril wings, and eyelids).

### 2. Dual-Lobe Specular Roughness
* **Base Roughness:** Matte stratum corneum lipid barrier ($0.45 - 0.60$).
* **Coat / Micro-Roughness:** Sharp, glossy specular sheen from sebum oil and perspiration in the T-zone ($0.15 - 0.28$).
* **Anatomical Modulation:** T-zone and eyelid margins receive lower roughness; dry cheeks and outer jaw receive higher roughness.

### 3. Micro-Cavity / Ambient Occlusion
* Derived from the negative Laplacian of the combined meso+micro displacement:
  $$\text{Cavity}(u, v) = \text{clip}\left(1.0 - \beta_{\text{cavity}} \cdot \max(0, -\nabla^2 D), 0.0, 1.0\right)$$
* Prevents ambient light from falsely illuminating the deepest depths of pores and wrinkle folds.

### 4. Delighted Diffuse Albedo
* Strips external directional light, shadows, and camera flash using AI delighting (`src/stage7_delight/`).
* Ensures that when the 3D head is placed in Blender with a single sun lamp, the shadows and highlights match the 3D scene lighting 100%, without double-shadowing.

---

## 5. Verification Gate: Film-Grade vs. Research 1

| Evaluation Criterion | Research 1 (Failed Baseline) | Production Architecture (This Spec) |
| :--- | :--- | :--- |
| **Displacement Origin** | Blurry 20k-vertex optical tracking meshes | Photo-derived meso wrinkles + 4K anatomical pore synthesis |
| **Pore Resolution** | Zero (smooth interpolated plastic) | 50-micron cellular follicles calibrated to facial zones |
| **Artifact Vulnerability**| Rubber bald caps, open mouth deltas | Clean facial bounds; bitwise collar seam pinning ($\Delta v = 0$) |
| **Shading Representation**| 2D OpenCV diffuse clay rendering | Native Blender Cycles Random Walk SSS + Dual-Lobe PBR |
| **Delivery Asset** | Loose OBJ + PNGs | Packaged `.blend` studio scene with lighting, camera & materials |
