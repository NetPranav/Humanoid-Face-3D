# Humanoid-Face-3D: Master Memory & Execution Blueprint (`remember.md`)

*This document records the exact, unfiltered reality of every pipeline stage: what is currently running on pre-trained weights, what was tested/archived, what is fully operational in code, and the active execution roadmap to achieve film-grade MetaHuman realism matching the Blender Cycles reference.*

---

## 🧭 1. Executive Summary: Code Reality vs. Architecture

To ensure complete transparency and zero confusion across all contributors:

| Pipeline Stage | Code Status | Technical Architecture | Current Visual Capability | Active Road Action |
| :--- | :---: | :--- | :--- | :---: |
| **Stage 0: Preprocessing** | 🟢 **100% Done** | InsightFace (ONNX / RetinaFace) | 5-point alignment, pose, pre-flight gate validation. | ❌ No training needed (Production ready) |
| **Stage 1: Identity Regression** | 🟢 **100% Done** | Pre-trained MICA (MPI) + Runtime Pixel3DMM dense contour fitter | Coarse head shape, metric skull proportions ($\|\beta_a - \beta_b\| \in [4.6, 7.2]$). | ❌ No retraining needed (Pretrained MICA is authoritative) |
| **Stage 1.5: Macro Residuals** | 🟢 **100% Done** | Graph Convolutional Network (`residual_net.py`) | Breaks linear FLAME PCA ceiling with bitwise collar pinning ($\Delta v \equiv 0$). | Ready for inference |
| **Stage 2: Expression & Pose** | 🟢 **100% Done** | Pre-trained SMIRK | Canonical neutral normalization ($\psi=0, \theta=0$). | ❌ No training needed (Invariant enforced) |
| **Tier 2: Meso Geometry** | 🟢 **100% Done** | Photometric Shape-from-Shading (`photometric_detail.py`) | Extracts real person-specific wrinkles (crow's feet, laugh lines, brow furrows) directly from photos. | 10 tests passing |
| **Tier 3: Micro Detail** | 🟢 **100% Done** | 4K Anatomical Zone Synthesis (`anatomical_pores.py`) | 50-micron follicular pores, T-zone follicles, cheek grain, lip striations, neck bands. | 8 tests passing |
| **Stage 4: Facial Hair & Stubble**| 🟢 **100% Done** | Procedural geometry & density maps | Stubble displacement (`stubble.py`), 3D hair cards (`cards.py`), collar pinning ($\Delta v \equiv 0$). | ❌ Pure math/geometry (No GPU training) |
| **Stage 6: UV Texture Projection** | 🟢 **100% Done** | Pure math (NumPy/OpenCV) | Multi-view backprojection with cosine-weighted blending, z-buffer visibility. 2048² projected texture. | ❌ Pure geometry (No GPU training) |
| **Stage 7: AI Delighting + Inpainting** | 🟢 **100% Done** | Pre-trained DECA albedo decoder + procedural Gaussian dilation | Strips environment lighting → clean diffuse albedo. Fills unseen UV regions. | Production ready |
| **Stage 8: PBR Material Stack** | 🟢 **100% Done** | Procedural (anatomical zones + displacement-coupled) | Dual-lobe roughness, cavity/AO (Laplacian), SSS thickness (opposing-normal ray-march). | 9 tests passing |
| **Tier 4: Film Cycles Engine** | 🟢 **100% Done** | Headless Blender Python (`render_blender_film.py`) | Adaptive subdivision (1 px/poly), Random Walk (Skin) SSS, 3-point studio lighting, AgX color transform. | 7 tests passing |
| **Stage 5: Production Rig & FBX** | 🟢 **100% Done** | ICT-FaceKit retopology, ARKit-52 blendshapes, 4 LODs, 5-joint skeleton | Headless Blender FBX packager, boundary collar pinning. Full UE5 Live Link compatibility. | ❌ Complete and tested (194 tests passing) |
| **Pipeline Integration** | 🟢 **100% Done** | End-to-End Orchestrator (`pipeline.py`, `production_inference.py`) | Single-command full synthesis from portraits to film look-dev render and FBX. | 198 tests passing, all 3 benchmark subjects verified |

---

## 🔍 2. Research 1 Audit: What Failed and Why We Pivoted

### Why Phase 3 Detail GAN on Meta Multiface Failed (Archived in `DOCS/FAILED/RESEARCH_1/`):
1. **The Dataset Frequency Ceiling:**  
   Meta Multiface meshes have $\approx 20,000$ vertices. They are optical tracking scans intended for speech and facial motion tracking. They contain **zero 50-micron epidermal pores**. A neural network cannot synthesize what is not present in its ground truth.
2. **Physical Acquisition Discontinuities:**  
   Multiface human actors wore latex tracking caps with sewn borders. The ray-marcher baked this rubber cap edge into a permanent horizontal shelf across the upper forehead. Open-mouth frames also generated unnatural displacement deltas around the lips.
3. **The Wireframe Defect:**  
   Coarse subdivision level 1 and $\sigma=1.2$ conditioning smoothing leaked FLAME triangle boundaries into the generator, which stamped polygonal facet lines across the face.
4. **Resolution:**  
   While our upgraded subdivision level 2 and $C^2$ smoothing ($\sigma=8.0$) eliminated the wireframe lines ($0.187 \to 0.029$ Laplacian), it revealed the core truth: without the wireframe noise, Multiface scans produce smooth, featureless plastic. Continuing to train on this data is a dead end.

---

## 🚀 3. The Active Sequential Execution Plan (Phase 10)

The software pipeline and test suite are 100% operational (160 tests passing). Execute the 5 sequential steps of **Phase 10** to achieve film-grade MetaHuman parity:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.1: TIER 2 MESO WRINKLE ENGINE (src/stage3_detail/photometric_detail.py)  │
│ • Decompose multi-view backprojected portraits via high-pass steerable filter    │
│ • Map intensity gradients to surface normal perturbations                        │
│ • Integrate to displacement heightfield via Poisson solver                       │
│ • Result: 1:1 photogrammetric reconstruction of subject's real crow's feet &     │
│   forehead furrows without neural network hallucination.                         │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.2: TIER 3 MICRO PORE SYNTHESIS (src/stage3_detail/anatomical_pores.py)   │
│ • Multi-octave cellular Voronoi-Worley basis + Gabor wavelets                    │
│ • Calibrated to anatomical facial zones from src/stage8_pbr/material_stack.py:   │
│   - Nose/T-zone: Dilated circular sebaceous follicles (0.15–0.35 mm)             │
│   - Cheeks: Elliptical pores aligned along tension lines (0.05–0.12 mm)          │
│   - Lips: Vertical papillary ridges & micro-folds (0.10–0.25 mm)                 │
│   - Forehead: Transverse micro-creases (0.08–0.22 mm)                            │
│   - Neck: Concentric tension bands with strict collar pinning (Δv ≡ 0)           │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.3: MULTI-TIER DISPLACEMENT & PBR MATERIAL COUPLING                       │
│ • Composite: D_total = D_macro + D_meso + D_micro (16-bit uint PNG, scale 5.0mm) │
│ • Derive micro-cavity AO map from negative Laplacian of D_total                  │
│ • Modulate dual-lobe roughness (T-zone sebum sheen 0.18 vs matte cheeks 0.55)    │
│ • Update tangent-space normal map                                                │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.4: AUTOMATED FILM-GRADE BLENDER CYCLES ENGINE (scripts/render_blender_film.py)
│ • Headless Blender Python API script (bpy)                                       │
│ • Catmull-Clark Adaptive Micro-Polygon Subdivision (1 px/polygon dicing)         │
│ • Principled BSDF with Random Walk (Skin) Subsurface Scattering                  │
│   (subsurface weight 0.18, red vascular radius [1.0, 0.25, 0.12], IOR 1.40)     │
│ • Cinematic 3-point studio lighting rig matching film look-dev turnarounds:      │
│   - Key Lamp (180W, 4500K warm white)                                            │
│   - Fill Lamp (45W, 6500K daylight)                                              │
│   - Rim/Sun Lamp (350W, 5500K sharp angle)                                       │
│ • 85mm portrait camera, shallow depth of field (f/2.8), AgX/Filmic contrast      │
│ • Outputs: 2048² film turnaround render + fully configured .blend studio scene   │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 10.5: END-TO-END PIPELINE INTEGRATION & VISUAL BENCHMARK AUDIT              │
│ • Wire Tier 2, Tier 3, and Tier 4 into src/pipeline.py and production_inference.py│
│ • Execute full turnaround on benchmark subjects (carell, connelly, lawrence)     │
│ • Verify collar pinning contract (Δv ≡ 0.000000 mm on lowest 20% of vertices)    │
│ • Confirm visual parity against reference image in Blender viewport              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📌 4. Key Invariants & Contracts

1. **Collar Boundary Contract:** Displacements on the lowest 20% of vertices (neck collar) must remain strictly clamped to zero (`masks['neck_pinning']`). The exported head must never tear when attached to a common body.
2. **Zero Silent Fallbacks:** Always raise explicit exceptions (`FileNotFoundError`, `RuntimeError`) when dependencies or files are missing.
3. **No Low-Frequency GAN Training:** Do not resume training the GAN on Meta Multiface scans.
4. **Shading Physics:** Never evaluate assets using unlit diffuse clay renders; visual verification must be conducted in Blender Cycles with Random Walk Subsurface Scattering.
