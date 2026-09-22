# 3D Face Geometry Pipeline — Master Architectural Roadmap

> **Target:** A production-grade, identity-preserving image-to-3D **fully textured, film-grade MetaHuman humanoid head asset** reconstruction pipeline that synthesizes film/VFX-grade 3D facial assets from 3–5 multi-view portraits, complete with physically based rendering material maps (delighted albedo, dual-lobe roughness, SSS thickness, cavity/AO, displacement, normal), ARKit-52 blendshapes, 4-tier LOD decimation, 5-joint skeleton, and automated Blender Cycles studio rendering.
> **Quality Bar:** Epic Games MetaHuman / VFX look-dev cinematic realism with true 50-micron epidermal pores, photo-derived meso wrinkles, and Random Walk Subsurface Scattering.
> **Active Architecture Specification:** [DOCS/02_Metahuman_Film_Grade_Synthesis.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/02_Metahuman_Film_Grade_Synthesis.md) & [DOCS/05_Blender_Cycles_Film_Rendering_Engine.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/05_Blender_Cycles_Film_Rendering_Engine.md).
> **Failed Research Archive:** [DOCS/FAILED/RESEARCH_1/](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1).

---

## 1. System Architecture: The 4-Tier Hierarchical Decomposition

Following the disproven hypothesis of monolithic GAN scan hallucination in Research 1, the pipeline operates on the industry-standard 4-tier frequency decomposition:

```
[3–5 Portrait Photos]
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Preprocessing & Validation                                     │
│ • InsightFace detection & 5-point alignment (RGB, [-1, 1] normalized)  │
│ • Pre-flight checks: identity consistency, yaw diversity, frontal view │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: MACRO GEOMETRY (Craniofacial Proportions & Skull Shape)        │
│ • MICA ArcFace feature extraction across views with frontality weighting│
│ • Regress 300-D FLAME shape coefficients (beta)                        │
│ • Canonical neutral normalization: psi=0, theta=0                      │
│ • Base mesh: 5,023 vertices, 9,976 triangles                           │
└────────────────────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────┐
       ▼                                             ▼
┌────────────────────────────────────────┐  ┌────────────────────────────┐
│ TIER 2: MESO GEOMETRY                  │  │ TIER 3: MICRO DETAIL       │
│ • Multi-view Photometric SfS           │  │ • 4K Anatomical Pores      │
│ • Direct photo-derived wrinkle extract │  │ • Texturing.xyz / MetaHuman│
│ • Crow's feet, brow furrows, laugh lines│  │ • T-zone follicles, lips, │
│ • 1:1 real photo correspondence        │  │   cheek pores, neck bands  │
└──────────────────┬─────────────────────┘  └─────────────┬──────────────┘
                   │                                      │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PBR TEXTURE & MATERIAL ENGINE (Stages 6, 7, 8)                         │
│ • Stage 6: UV Texture Projection (backprojection, cosine blend, z-buff)│
│ • Stage 7: AI Delighting & Inpainting (clean diffuse albedo)           │
│ • Stage 8: PBR Material Stack Derivation                               │
│   ├── Dual-Lobe Roughness (stratum corneum base + sebum coat sheen)    │
│   ├── Micro-Cavity / AO (displacement negative Laplacian)              │
│   └── SSS Thickness (opposing-normal ray-march)                        │
└─────────────────────────────────────┬──────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 4: FILM-GRADE BLENDER CYCLES STUDIO ENGINE                        │
│ • Adaptive Micro-Polygon Subdivision (1 polygon per screen pixel)      │
│ • Principled BSDF with Random Walk (Skin) Subsurface Scattering        │
│ • Cinematic 3-point studio lighting rig (Key, Fill, sharp Rim/Sun)     │
│ • 85mm prime portrait camera, shallow depth-of-field, AgX / Filmic     │
│ • Output: 2048² turnaround render + production .blend studio scene     │
└─────────────────────────────────────┬──────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: GAME ENGINE RIGGING & PRODUCTION EXPORT                       │
│ • ICT-FaceKit retopology & 52 ARKit blendshapes (boundary pinned)      │
│ • 4-tier LOD chain (LOD0: 24.5k tris down to LOD3: 500 tris)           │
│ • 5-joint skeletal armature (head, neck, jaw, left eye, right eye)     │
│ • Unreal Engine 5 Live Link FBX export                                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Master Phase Status Matrix

| Phase | Description | Architecture / Method | Current Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Foundation, P0 Bug Fixes & Honest Failure Verification | FLAME loader, silent fallback elimination, test suite | ✅ **Completed** (160 tests passing) |
| **Phase 1** | Multi-View Inference Baseline | InsightFace + MICA + SMIRK canonical normalization | ✅ **Completed** |
| **Phase 2** | MICA Identity Regressor Verification | ArcFace ViT Backbone + MPI Shape MLP ($\beta \in \mathbb{R}^{300}$) | ✅ **Completed** (Verified $\|\beta_a - \beta_b\| \in [4.6, 7.2]$) |
| **Phase 2.5**| Geometry Preprocessing & Scan Extraction | Differentiable Ray-Casting & $C^2$ Smoothing ($\sigma=8.0$) | ⚠️ **Archived (Research 1)** (Laplacian $0.029$; scan lacks pores) |
| **Phase 3 (Old)**| Adversarial Detail GAN on Multiface Scans | U-Net Generator + PatchGAN Discriminator (40k steps) | 🛑 **FAILED & ARCHIVED** (Wireframe defect & flat scans; see `FAILED/RESEARCH_1`) |
| **Phase 4** | Commercial Licensing & Ingest Protocol | Talent release forms, multi-view capture protocol | ✅ **Completed** |
| **Phase 5** | Production Retopology, ARKit-52 Rigging & LODs | Sparse $W$ matrix, 52 ARKit blendshapes, 4 LODs, skeleton | ✅ **Completed** |
| **Phase 6** | Procedural Facial Hair & Beard Stubble | 3D hair cards, stubble displacement, collar pinning | ✅ **Completed** |
| **Phase 7** | Headless Production Packaging & UE5 Export | Headless Blender FBX packager, manifest generator | ✅ **Completed** |
| **Phase 8** | Benchmarking & Automated Quality Gates | Chamfer metrics, collar pinning assertion ($\Delta v \equiv 0$) | ✅ **Completed** |
| **Phase 9** | PBR Texture Engine (Stages 6, 7, 8) | Pure math UV projection, AI delighting, roughness, cavity, SSS | ✅ **Completed** |
| **Phase 10 (NEW)**| **Film-Grade MetaHuman Synthesis & Cycles Engine** | **Photo-derived meso wrinkles + 4K pores + Cycles SSS** | 🔄 **ACTIVE ROADMAP (IN PROGRESS)** |

---

## 3. Active Sequential Execution Plan: Phase 10 (Step-by-Step)

To bridge the gap from flat geometry to the film-grade MetaHuman quality demonstrated in the reference render, execute the following steps strictly in sequence:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.1: Tier 2 Meso Wrinkle Engine (Photo-Derived Shape-from-Shading)     │
│ ──> Extract real crow's feet, brow furrows, and laugh lines from photos      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.2: Tier 3 Micro Pore Engine (4K Anatomical Zone Synthesis)          │
│ ──> Synthesize 50-micron follicles, cheek pores, lip striations, neck bands  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.3: Multi-Tier Fusion & PBR Material Coupling                         │
│ ──> Fuse displacements; derive Cavity AO from Laplacian; dual-lobe roughness │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.4: Automated Film-Grade Blender Cycles Studio Engine                 │
│ ──> Headless .blend generator with Random Walk SSS & 3-point studio lighting │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.5: End-to-End Pipeline Integration & Benchmark Turnaround            │
│ ──> Full pipeline run on test portraits; verify collar pinning; deliver .blend│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 10.1: Tier 2 Meso Wrinkle Engine (`src/stage3_detail/photometric_detail.py`)
> **Goal:** Stop asking a neural network to guess wrinkles. The subject's high-resolution portrait photos *already* record their exact crow's feet, forehead furrows, and laugh lines. Extract them directly into UV displacement space.

- [ ] **Multi-Scale Steerable Filter:**
  - Decompose backprojected portrait textures into high-frequency luminance gradients:
    $$I_{\text{high}} = I_{\text{albedo}} - G_{\sigma} * I_{\text{albedo}} \quad (\sigma \approx 8 - 12\,\text{px})$$
- [ ] **Photometric Normal Deviation Estimation:**
  - Using calibrated view directions and base surface normals, convert high-pass luminance gradients into surface normal perturbations $\Delta \vec{n}_{\text{meso}}(u, v)$.
- [ ] **Poisson Gradient Integration to Displacement:**
  - Integrate surface normal slopes into height displacements $D_{\text{meso}}(u, v)$ via Poisson solver:
    $$\nabla^2 D_{\text{meso}} = \frac{\partial \Delta n_x}{\partial u} + \frac{\partial \Delta n_y}{\partial v}$$
- [ ] **Multi-View Confidence Blending:**
  - Fuse displacement estimates from frontal, left-45°, and right-45° views using angle-weighted cosine visibility masks.
- [ ] **Unit Tests:** Add `tests/test_photometric_detail.py` verifying gradient extraction, zero NaNs, and seamless UV blending.

---

### Step 10.2: Tier 3 Micro Pore Engine (`src/stage3_detail/anatomical_pores.py`)
> **Goal:** Synthesize true 50-micron epidermal pores and cellular micro-texture calibrated strictly to human facial anatomy (Texturing.xyz / MetaHuman standard).

- [ ] **Procedural Basis Functions:**
  - Implement multi-octave cellular Voronoi-Worley basis functions combined with Gabor directional wavelets.
- [ ] **Anatomical Zone Modulation:**
  - Wire to `src/stage8_pbr/material_stack.py` facial zone masks:
    * **T-Zone / Nose:** Large dilated circular follicles ($0.15 - 0.35\,\text{mm}$, isotropic).
    * **Cheeks:** Fine elliptical pores ($0.05 - 0.12\,\text{mm}$) oriented along Langer's skin tension lines.
    * **Lips (Vermilion):** Vertical micro-creases and dermal papillary ridges ($0.10 - 0.25\,\text{mm}$, strictly vertical).
    * **Forehead:** Directional transverse micro-furrows ($0.08 - 0.22\,\text{mm}$).
    * **Neck:** Concentric tension bands with strict collar pinning ($\Delta v \equiv 0$ on lowest 20%).
- [ ] **4096² & 1024² Multi-Resolution Generation:**
  - Support full 4K procedural rasterization for film rendering and 1024² downsampled maps for real-time engines.
- [ ] **Unit Tests:** Add `tests/test_anatomical_pores.py` asserting zone depth ranges, pore density variations, and boundary safety.

---

### Step 10.3: Multi-Tier Fusion & PBR Material Coupling
> **Goal:** Combine Macro, Meso, and Micro geometry into cohesive PBR maps where light physically interacts with pore crevices.

- [ ] **Displacement Fusion:**
  - Composite multi-tier displacement:
    $$D_{\text{total}}(u, v) = D_{\text{macro}}(u, v) + D_{\text{meso}}(u, v) + D_{\text{micro}}(u, v)$$
  - Export lossless 16-bit unsigned PNG ($[0, 65535]$, midlevel $0.50$, scale $5.0\,\text{mm}$).
- [ ] **Micro-Cavity & Ambient Occlusion Coupling:**
  - Update `src/stage8_pbr/material_stack.py` to derive micro-cavity directly from the negative Laplacian of the combined displacement:
    $$\text{Cavity}(u, v) = \text{clip}\left(1.0 - \beta_{\text{cavity}} \cdot \max(0, -\nabla^2 D_{\text{total}}), 0.0, 1.0\right)$$
  - Ensures pore bottoms and deep wrinkle crevasses receive zero ambient light bounce.
- [ ] **Dual-Lobe Specular Roughness Map:**
  - Modulate base skin roughness ($0.45 - 0.60$) with sebum coat sheen ($0.15 - 0.25$) concentrated in the T-zone and eyelid margins.
- [ ] **Tangent-Space Normal Map:**
  - Compute tangent normal map from the composite displacement heightfield.

---

### Step 10.4: Automated Film-Grade Blender Cycles Studio Engine (`scripts/render_blender_film.py`)
> **Goal:** Automate headless production `.blend` scene generation and 2048² turnaround rendering with cinema-grade look-dev lighting matching the user's reference image.

- [ ] **Headless Blender Python Script (`bpy`):**
  - Build `scripts/render_blender_film.py` executable headlessly via `blender -b -P scripts/render_blender_film.py -- [args]`.
- [ ] **Adaptive Micro-Polygon Subdivision:**
  - Enable Catmull-Clark adaptive subdivision on the neutral head mesh with dicing rate = 1.0 px/polygon at render time.
- [ ] **Principled BSDF Skin Shader Node Tree:**
  - Connect Delighted Albedo $\to$ Base Color.
  - Connect Roughness Map $\to$ Roughness (Non-Color).
  - Connect SSS Thickness $\to$ Subsurface Weight.
  - Set Subsurface Method to **`RANDOM_WALK_SKIN`** with red vascular scatter radius `(1.0, 0.25, 0.12)` and IOR $1.40$.
  - Connect Tangent Normal Map $\to$ Normal.
  - Connect 16-bit Displacement $\to$ Material Output Displacement (scale $0.005\,\text{m}$, midlevel $0.5$).
- [ ] **Cinematic 3-Point Lighting Rig:**
  - **Key Light:** Area lamp ($180\,\text{W}$, $4500\,\text{K}$ warm white) at $+45^\circ$ Yaw, $+30^\circ$ Pitch.
  - **Fill Light:** Area lamp ($45\,\text{W}$, $6500\,\text{K}$ daylight) at $-45^\circ$ Yaw, $+10^\circ$ Pitch.
  - **Rim/Sun Light:** Sharp directional lamp ($350\,\text{W}$, $5500\,\text{K}$) at $+135^\circ$ Yaw, $+45^\circ$ Pitch to carve the jawline silhouette.
- [ ] **Camera & Color Management:**
  - 85mm prime portrait lens with shallow depth of field ($f/2.8$).
  - Color management set to `AgX` or `Filmic` with `Medium High Contrast`.
- [ ] **Deliverables:**
  - Render high-resolution portrait turnaround (`film_render_cycles.png`).
  - Save production-ready, artist-editable `.blend` scene (`studio_scene.blend`).

---

### Step 10.5: End-to-End Pipeline Integration & Benchmark Turnaround
> **Goal:** Wire all components into the single-command orchestrator, verify system invariants, and validate against benchmark portraits.

- [ ] **Pipeline Orchestrator Wiring:**
  - Update `src/pipeline.py` and `scripts/production_inference.py` to invoke Tier 2, Tier 3, Stage 8 PBR, Stage 5 FBX export, and the Blender film rendering pass.
- [ ] **System Invariant Validation:**
  - Enforce bitwise collar pinning: verify $\Delta v \equiv 0.000000\,\text{mm}$ across the lowest 20% of vertices.
  - Enforce zero silent fallbacks: ensure explicit exceptions are raised if required inputs are invalid.
- [ ] **Benchmark Execution:**
  - Run full pipeline on test portraits (`carell`, `connelly`, `lawrence`).
  - Inspect output rendered images and `.blend` scenes to verify visual parity with the MetaHuman reference standard.
- [ ] **Full Test Suite Run:**
  - Ensure all unit tests pass with $>95\%$ coverage.

---

## 4. Operational Invariants for Phase 10

1. **Zero Silent Fallbacks:** Always raise explicit errors (`FileNotFoundError`, `RuntimeError`) with clear setup instructions.
2. **Collar Seam Contract:** Displacements on the lowest 20% of vertices (neck boundary collar) must remain strictly pinned to zero (`masks['neck_pinning']`).
3. **No Low-Frequency GAN Retraining:** Never spend GPU quota training GANs on low-frequency optical tracking datasets (Meta Multiface).
4. **Shading Completeness:** Never evaluate facial assets without Random Walk Subsurface Scattering and calibrated 3-point lighting.
