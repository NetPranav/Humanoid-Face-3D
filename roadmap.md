# 3D Face Geometry Pipeline — Master Architectural Roadmap

> **Target:** A locally-orchestrated, Kaggle T4×2-trainable face reconstruction pipeline that accepts 3–5 multi-view portraits and synthesizes a game-ready, untextured 3D facial mesh (neutral base mesh + micro-displacement details + ARKit-52 blendshapes + LODs + UE5 Live Link rig).
> **Quality Bar:** Beat Meshy 7.1's self-reported 59.8% surface-detail score on facial anatomy.
> **Source Documents:** Derived from `Research.md`, `DOCS/01–03`, `DOCS/00_code_review.md`, and `DOCS/another_guide.md`.

---

## Architecture Overview & Mental Model

```
[3–5 Face Photos]
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 0: Preprocessing & Validation                                     │
│ • InsightFace detection & 5-point alignment (RGB, [-1, 1] normalized)  │
│ • Pre-flight checks: identity consistency, yaw diversity, frontal view │
└────────────────────────────────────────────────────────────────────────┘
       │
       ├─── Aligned crops (112×112)
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Multi-View Identity Shape Regression (MICA)                    │
│ • ArcFace feature extraction across views                              │
│ • Pose-weighted embedding-space fusion: w_i = det_score · cos(yaw)²    │
│ • Regress 300-D FLAME shape parameter (beta)                           │
└────────────────────────────────────────────────────────────────────────┘
       │
       ├─── Aligned frontal crop (224×224)
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Expression & Pose Regression (SMIRK)                          │
│ • Regress 100-D expression (psi) and 15-D joint pose (theta)           │
│ • CRITICAL: Base mesh normalized to CANONICAL NEUTRAL (psi=0, theta=0) │
│ • Expression parameters saved as metadata for Stage 5 blendshape deltas│
└────────────────────────────────────────────────────────────────────────┘
       │
       ├─── Canonical Neutral Mesh (5023 vertices, FLAME topology)
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: High-Frequency Detail Synthesis (Detail GAN)                  │
│ • Rasterize neutral position (3) + normal (3) maps in UV space (512²)  │
│ • U-Net Generator (InstanceNorm, AdaIN identity/expression conditioning│
│   + MultiView cross-attention with residual connection)                │
│ • PatchGAN Discriminator (SpectralNorm, FP32 lazy R1 regularization)   │
│ • Output: 16-bit signed UV displacement map (wrinkles, pores, stubble) │
└────────────────────────────────────────────────────────────────────────┘
       │
       ├─── Neutral Base Mesh + UV Displacement Map + Expression Metadata
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Production Retopology, Rigging & Export                       │
│ • FLAME → ICT-FaceKit correspondence matrix (W @ v_flame)              │
│ • 52 ARKit blendshapes via deformation transfer                        │
│ • 4 LOD levels (LOD0: ~24.5k tris, LOD1: 5k, LOD2: 2k, LOD3: 500)      │
│ • 5-joint skeletal armature (head, neck, jaw, left eye, right eye)     │
│ • Headless Blender packaging → Unreal Engine 5 Live Link FBX           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Kaggle T4×2 Resource Budget

* **Weekly GPU Quota:** ~30 GPU-hours per week.
* **T4×2 Consumption:** 1 wall-clock hour = **2 GPU-quota hours**.
* **Weekly Limit:** **~15 wall-clock hours** of 2×T4 per week (1 full 11.5h session + 1 short 3.5h run).
* **CPU Session Utilization:** CPU-only sessions cost **0 GPU quota**. All geometry preprocessing (Phase 2.5), retopology ICP (Phase 5), evaluation runs (Phase 8), and Blender FBX packaging (Phase 7) must run on CPU sessions.
* **Checkpoints:** Emergency checkpoint triggered at **11.5 hours** before Kaggle's 12-hour session kill.

---

## Master Phase Status Matrix

| Phase | Description | Estimated Wall-Clock | Accelerator | Status |
|---|---|---|---|---|
| **Phase 0** | Foundation, P0 Bug Fixes & Honest Failure Verification | 2–4 days | Local / CPU | 🔄 In Progress (Subphase 0.1 Done) |
| **Phase 1** | Multi-View Inference Baseline (Stages 0, 1, 2) | 1–2 weeks | Local + Kaggle T4×2 | ⏹ Queued |
| **Phase 2** | Identity Regressor Demographic Fine-Tuning | 1–2 weeks | Kaggle T4×2 (~15h quota) | ⏹ Queued |
| **Phase 2.5**| Geometry Preprocessing & UV Displacement Dataset Engine | 1 week | Kaggle CPU (0 quota) | ⏹ Queued |
| **Phase 3** | Adversarial High-Frequency Detail Synthesis (Detail GAN) | 3–4 weeks | Kaggle T4×2 (~30h quota) | ⏹ Queued |
| **Phase 4** | Commercial Licensing & Own-Capture Asset Track | Weeks 1–10 (Parallel) | Business / Legal | ⏹ Queued |
| **Phase 5** | Production Retopology, ARKit-52 Rigging & LODs | 4–6 weeks | Local / Kaggle CPU | ⏹ Queued |
| **Phase 6** | Detail Hybridization & Static Facial Hair | 2–3 weeks | Kaggle T4×2 | ⏹ Queued |
| **Phase 7** | Headless Production Packaging & UE5 Live Link Export | 1–2 weeks | Kaggle CPU (0 quota) | ⏹ Queued |
| **Phase 8** | Comprehensive Benchmarking & Quality Assurance | Ongoing | Local / Kaggle CPU | ⏹ Queued |

---

## Detailed Roadmap & Execution Checklists

---

### Phase 0: Foundation, P0 Bug Fixes & Honest Failure Verification
> **Objective:** Fix the critical defects identified in `DOCS/00_code_review.md`, eliminate every silent fallback that masks pipeline failures, install an automated test suite, and ensure the pipeline fails loudly and honestly when dependencies or weights are missing.

#### Subphase 0.1: Core FLAME Loader Replacement (P0 Fix)
- [x] Replace `src/utils/flame_model.py` with the robust implementation in `DOCS/flame_model.py`.
- [x] Verify `shapedirs` unpacking: `shapedirs[:, :, :300]` for shape, `shapedirs[:, :, 300:400]` for expression (fix `KeyError: 'exprdirs'`).
- [x] Implement Chumpy compatibility shim with NumPy aliases (`np.bool`, `np.int`, `np.float`) and `_ChumpyStub` fallback.
- [x] Enforce FLAME native coordinate units in **metres** (`[-0.13, 0.13]m`) with explicit `scale_to_mm` parameter.
- [x] Implement Rodrigues axis-angle rotation and Linear Blend Skinning (LBS) in pure NumPy for jaw/neck/eye joints (`theta`).
- [x] Implement vectorised vertex normal calculation (`vertex_normals`) replacing per-face Python loops.

#### Subphase 0.2: Eliminate Silent Fallbacks Across All Modules (P1 Fix)
- [ ] **`src/stage0_preprocess/detector.py`:** Remove fake detection fallback (`det_score=0.99`, `yaw=0.0`); raise `RuntimeError` if InsightFace is unavailable.
- [ ] **`src/utils/validation.py`:** Fix `insightface.model_zoo.get_model('buffalo_l')` crash; use `FaceAnalysis.get(img)` and extract `normed_embedding`. Eliminate bare `except Exception: pass`.
- [ ] **`src/stage1_identity/inference.py`:** Remove silent `np.zeros(300)` mean-face fallback; raise explicit error if MICA weights or architecture fail to load.
- [ ] **`src/stage2_expression/encoder.py`:** Remove `zeros((128,128))` fake detail fallback; load SMIRK encoder weights by stripping `smirk_encoder.` key prefix.
- [ ] **`src/pipeline.py`:** Remove 5023 coincident vertex / degenerate OBJ fallback; raise `FileNotFoundError` if FLAME model is missing.
- [ ] **`src/stage3_detail/data.py`:** Remove all-zero tensor fallback when displacement files are missing; restore `raise RuntimeError`.
- [ ] **`evaluation/identity_score.py`:** Delete fallback comparing photo to itself (`render = photo`); require real rendered mesh image or raise `FileNotFoundError`.

#### Subphase 0.3: Dependency & Environment Harmonization
- [ ] Update `requirements.txt`: Remove PyPI `nvdiffrast`, remove unused `open3d`/`mediapipe`, add `chumpy`, `ninja`, `pytest`.
- [ ] Document Kaggle install command for `nvdiffrast` (`pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation`).
- [ ] Document Kaggle NumPy 2.x downgrade pattern with mandatory kernel restart in Cell 1.
- [ ] Update `configs/default.yaml`: Document FLAME version selection, set default fusion temperature/exponent, eliminate dead configuration fields.

#### Subphase 0.4: Test Suite Installation & Gate Verification
- [ ] Create `tests/test_flame.py`: Assert 5023 vertices, no NaNs, template bounds within `[-0.25, 0.25]`m, zero-pose deviation `< 1e-6`, jaw rotation moves vertices.
- [ ] Create `tests/test_validation.py`: Assert rejection on no face, different individuals, and extreme angular deviation.
- [ ] Create `tests/test_identity_score.py`: Assert `compute_identity_score` raises when preview PNG is missing.
- [ ] Run `pytest -v` locally: **Confirm that tests fail honestly on missing assets** rather than passing trivially.

**Phase 0 Gate:**
1. `src/utils/flame_model.py` passes all unit tests with zero NaNs and real LBS articulation.
2. Every module raises explicit exceptions on missing weights/dependencies.
3. Automated test suite exists and is wired into CI/local verification.

---

### Phase 1: Multi-View Inference Baseline (Stages 0, 1, 2)
> **Objective:** Stand up an inference-only pipeline that takes 3–5 portraits of a subject and outputs a canonical neutral `.obj` base mesh with verified identity shape. No training.

#### Subphase 1.1: MICA Model Integration & Preprocessing Fixes
- [ ] Vendor MICA repository via pinned Git submodule (`vendor/MICA`).
- [ ] Fix input crop color space: Convert OpenCV BGR crop to **RGB** (`crop_112[:, :, ::-1]`).
- [ ] Fix input crop normalization: Scale from `[0, 1]` to **`[-1, 1]`** via `(img - 127.5) / 127.5`.
- [ ] Implement thin Python adapter matching MICA's actual constructor signature (`config`, `flame_model`).

#### Subphase 1.2: Multi-View Fusion in Embedding Space
- [ ] Refactor `encode_multiview`: Extract ArcFace embeddings per view, normalize, and fuse in **embedding space** prior to shape regression MLP.
- [ ] Implement frontality weighting: $w_i = \text{det\_score}_i \cdot \cos^2(\text{yaw}_i)$.
- [ ] Provide fallback $\beta$-space fusion with tunable temperature $T$ if backbone cannot be separated from regressor.

#### Subphase 1.3: Stage 2 Expression Separation & Canonical Neutral Normalization
- [ ] Wrap SMIRK encoder to extract 100-D expression $\psi$ and 15-D pose $\theta$.
- [ ] Remove `coarse_detail` references from pipeline, config, and manifest (Stage 2 is expression/pose only).
- [ ] Enforce base mesh normalization: Output geometry decoded strictly with $\psi = 0$ and $\theta = 0$.
- [ ] Save extracted expression $\psi$ and pose $\theta$ to `manifest.json` for downstream ARKit blendshape retargeting.

#### Subphase 1.4: Real Offscreen Preview Rendering
- [ ] Implement `render_neutral_preview(mesh_obj, out_png, size=512)`: Orthographic Lambertian grey shaded render of frontal neutral base mesh.
- [ ] Calibrate ArcFace cosine similarity distribution across 20 known-good rendered neutral meshes to establish a proven baseline threshold (replacing arbitrary 0.5 number).

#### Subphase 1.5: Kaggle Phase 1 Baseline Notebook
- [ ] Stand up `notebooks/kaggle/phase1_inference_baseline.ipynb`.
- [ ] Run inference across 5 distinct test subjects.
- [ ] Assert pairwise $\beta$ Euclidean distance between all pairs: $\|\beta_a - \beta_b\| > 10^{-3}$ (verifying no mean-face fallback).
- [ ] Generate rendered previews and compute real identity scores.

**Phase 1 Gate:**
1. 5 subjects produce 5 distinct `.obj` meshes with $\|\beta_a - \beta_b\| > 10^{-3}$.
2. Every output directory contains a valid rendered `head_mesh.png` and complete `manifest.json`.
3. Identity verification score exceeds calibrated baseline on real renders.

---

### Phase 2: Identity Regressor Demographic Fine-Tuning (Stage 1)
> **Objective:** Fine-tune MICA's regression head on target demographic data to beat pretrained MICA on the NoW benchmark ($< 0.90\text{mm}$ median error).

#### Subphase 2.1: Training Data Preparation
- [ ] Assemble registered FLAME scans from MICA unified dataset (LYHM, FaceWarehouse, Stirling).
- [ ] Package registered dataset into private Kaggle Dataset (`face-geo-training-data-stage1`).
- [ ] Create validation split on held-out subjects.

#### Subphase 2.2: DDP Training Script with PyTorch 2.4+ AMP
- [ ] Build `src/stage1_identity/trainer.py` using `torch.amp.autocast('cuda', dtype=torch.float16)` and `torch.amp.GradScaler('cuda')`.
- [ ] Implement multi-GPU DistributedDataParallel (DDP) across both T4 GPUs via `torchrun --nproc_per_node=2`.
- [ ] Implement 11.5-hour session timer with automatic checkpoint upload.
- [ ] Stand up `notebooks/kaggle/phase2_finetune_identity.ipynb` using `%%writefile` + `!torchrun`.

#### Subphase 2.3: Verification on NoW Benchmark
- [ ] Evaluate fine-tuned checkpoint on NoW validation set.
- [ ] Verify median scan-to-mesh error improves over pretrained MICA baseline.
- [ ] Push verified checkpoint to Kaggle Models registry with FLAME version metadata.

**Phase 2 Gate:**
1. Fine-tuned model achieves lower median error than pretrained baseline on NoW.
2. Identity similarity improves on 20 held-out photographic portraits.
3. Checkpoint registered on Kaggle Models.

---

### Phase 2.5: Geometry Preprocessing & UV Displacement Dataset Engine
> **Objective:** Transform raw high-resolution 3D scans (e.g. FaceScape) into paired 512×512 UV displacement maps, neutral position maps, normal maps, and facial validity masks. Run on CPU sessions (0 GPU quota).

#### Subphase 2.1: Ray-Mesh Geometry Correspondence
- [ ] Implement surface ray-casting using `trimesh.ray`: Cast rays from each coarse FLAME vertex along its normal vector to intersect scan surface.
- [ ] Compute signed displacement: $\delta_v = (p_{\text{intersect}} - v_{\text{flame}}) \cdot n_v$.
- [ ] Record `hit_mask`: Exclude non-intersecting or self-occluded vertices.

#### Subphase 2.2: Barycentric UV Triangle Rasterization
- [ ] Download and integrate FLAME UV coordinates (`head_template.obj` / `FLAME_texture.npz`).
- [ ] Implement barycentric triangle rasterizer over FLAME UV layout at 512×512 resolution.
- [ ] Interpolate per-vertex displacement across triangles; dilate mask by 2–4 pixels across UV seams to avoid border artifacts.
- [ ] Rasterize neutral position map (`pos.png` or `pos.npy`) and unit normal map (`norm.png` or `norm.npy`).

#### Subphase 2.3: Dataset Normalization & Storage Contract
- [ ] Measure corpus-wide absolute displacement 99th percentile ($p_{99}$).
- [ ] Save `normalization_stats.json` containing measured $p_{99}$ value, sample count, and date.
- [ ] Enforce lossless storage contract:
  - Write: $d_{\text{norm}} = \text{clip}(d_{\text{mm}} / p_{99}, -1, 1)$, saved as 16-bit uint PNG ($[0, 65535]$) or float16 `.npy`.
  - Read: $d = (u16 / 65535.0) \times 2.0 - 1.0 \in [-1, 1]$.
  - Inference: $d_{\text{mm}} = d_{\text{model}} \times p_{99}$.
- [ ] Unit test: Verify round-trip conversion error $< 1/65535$.

#### Subphase 2.4: Subject-Stratified Split & Packaging
- [ ] Split dataset strictly by unique **Subject ID** (not filename) to prevent smile vs. neutral data leakage.
- [ ] Package preprocessed dataset into private Kaggle Dataset (`face-geo-uv-displacement-512`).

**Phase 2.5 Gate:**
1. Script processes scans and generates real `*_disp`, `*_pos`, `*_norm`, and `*_mask` maps.
2. Valid facial mask coverage is measured and verified ($> 50\%$ UV canvas).
3. Round-trip normalization test passes.
4. Ran completely on CPU Kaggle session with 0 GPU quota consumed.

---

### Phase 3: Adversarial High-Frequency Detail Synthesis (Detail GAN)
> **Objective:** Train a generator/discriminator pair to synthesize 512×512 UV displacement maps containing subject-specific micro-wrinkles and pore texture, conditioned on multi-view features and neutral coarse geometry.

#### Subphase 3.1: Generator Architecture Upgrades (P1 Fixes)
- [ ] Replace `nn.BatchNorm2d` with `nn.InstanceNorm2d(affine=True)` or `nn.GroupNorm(8)` to eliminate batch-coupling artifacts and batch-size-4 noise.
- [ ] Fix cross-attention bottleneck: Add LayerNorm and residual connection ($q = q + \text{MultiViewAttention}(\text{LN}(q), \text{feats})$) to preserve spatial features and AdaIN conditioning.
- [ ] Verify bilinear upsampling in decoder path (no checkerboard artifacts).

#### Subphase 3.2: Discriminator & Loss Recipe Upgrades (P1 Fixes)
- [ ] PatchGAN architecture with Spectral Normalization evaluating 70×70 receptive field patches.
- [ ] Fix lazy R1 penalty accumulation: Remove `opt_d.zero_grad()` before R1 backward; accumulate R1 gradients with adversarial gradients every 16 steps.
- [ ] Force R1 gradient calculation strictly in **FP32** (`autocast(enabled=False)`).
- [ ] Masked L1 reconstruction loss: Restrict computation strictly to valid UV pixels ($\text{mask} = 1$).
- [ ] Anneal reconstruction weight $\lambda_{\text{recon}}$ from 100 to 10 over 50k steps.

#### Subphase 3.3: EMA & Model Serialization (P1 Fix)
- [ ] Fix `update_ema`: Copy model buffers (`running_mean`, `running_var`) in addition to parameter interpolation.
- [ ] Enforce upload gate in `scripts/upload_to_kaggle_models.py`: Require `ema_generator.pt` and verified `normalization_stats.json`.

#### Subphase 3.4: Pilot Run & Full Training on Kaggle T4×2
- [ ] Pilot run: Train on 50 subjects for 5k steps to verify loss convergence, non-zero recon loss, and batch output std $> 0.01$.
- [ ] Full training run: 50k steps across 2×T4 using DDP and PyTorch 2.4+ AMP (~15–20 wall-clock hours over 2 weekly quotas).
- [ ] Monitor health metrics every 100 steps (D real/fake accuracy, G loss, batch diversity).

**Phase 3 Gate:**
1. Pilot run demonstrates non-zero loss and distinct wrinkle maps.
2. Batch output standard deviation remains $> 0.01$ throughout training (no mode collapse).
3. Surface-detail Chamfer distance improves over coarse Stage 1 baseline on held-out test scans.
4. ArcFace identity score does not drop more than 0.05 from Phase 2 baseline.

---

### Phase 4: Commercial Licensing & Own-Capture Asset Track (Parallel)
> **Objective:** Resolve legal and commercial viability by migrating foundation components to open licenses, establishing clean data provenance, and building an in-house capture rig.

#### Subphase 4.1: FLAME 2023 Open Migration
- [ ] Evaluate MPI's FLAME 2023 Open release (released November 2025 under **CC-BY-4.0**).
- [ ] Migrate `configs/default.yaml` to `FLAME2023_Open` if commercial shipping is required.
- [ ] Integrate MPI parameter conversion utilities if using MICA/SMIRK heads trained on 2020 basis.

#### Subphase 4.2: Dependency License Audit
- [ ] Audit licenses for every component in pipeline:
  - FLAME (2020 Non-Commercial vs. 2023 Open CC-BY-4.0).
  - MICA (Non-Commercial Research).
  - SMIRK / EMOCA (Research Non-Commercial).
  - FaceScape (Explicit No Commercial Use).
  - nvdiffrast (NVIDIA Source Code License).
  - ICT-FaceKit (MIT - Fully Commercial).
- [ ] Establish architecture boundary: Treat research models as offline "teachers" to supervise custom clean models if needed.

#### Subphase 4.3: FaceScape Commercial Licensing Inquiry
- [ ] Submit commercial license inquiry to Nanjing University team (`nju3dv@gmail.com`).
- [ ] Document terms, pricing, and restrictions.

#### Subphase 4.4: In-House Photogrammetry Capture Protocol
- [ ] Define multi-camera portrait capture protocol (minimum 5 synchronized or static poses: frontal, $\pm 45^\circ$ quarter, $\pm 90^\circ$ profile).
- [ ] Draft commercial talent release and consent documentation.
- [ ] Execute pilot capture on 10 internal subjects to validate pipeline independence from academic datasets.

**Phase 4 Gate:**
1. Clear legal pathway established for all runtime components.
2. FLAME version decision permanently recorded.
3. In-house capture protocol verified end-to-end.

---

### Phase 5: Production Retopology, ARKit-52 Rigging & LODs (Stage 5)
> **Objective:** Bridge the research mesh to a production game asset by retopologizing FLAME to ICT-FaceKit, generating 52 ARKit blendshapes, creating 4 LODs, and adding a skeletal armature. Runs entirely on CPU sessions (0 GPU quota).

#### Subphase 5.1: FLAME → ICT-FaceKit Dense Correspondence Matrix ($W$)
- [ ] Load FLAME neutral template (5023 verts) and ICT-FaceKit neutral template (~24.5k tris).
- [ ] Align templates via anatomical landmarks and compute non-rigid iterative closest point (NICP).
- [ ] For each ICT vertex, compute barycentric coordinates $(f, u, v, w)$ relative to corresponding FLAME triangle.
- [ ] Construct sparse correspondence matrix $W \in \mathbb{R}^{N_{\text{ICT}} \times 5023}$.
- [ ] Validate round-trip error: $\|v_{\text{ICT}} - W v_{\text{FLAME}}\| < 1.0\text{mm}$ across facial surface.

#### Subphase 5.2: ARKit-52 Semantic Blendshape Generation
- [ ] Extract ICT-FaceKit's 52 ARKit-compatible blendshape target meshes (MIT licensed).
- [ ] Implement deformation transfer (Sumner & Popović) to transfer generic ARKit deltas onto the subject-specific identity mesh.
- [ ] Define standard `blendshapes.json` delta format:
  ```json
  {
    "jawOpen": [[dx0, dy0, dz0], [dx1, dy1, dz1], ...],
    "mouthSmileLeft": [[dx0, dy0, dz0], ...]
  }
  ```

#### Subphase 5.3: LOD Decimation Preserving Shape Keys
- [ ] Implement multi-resolution decimation for neutral mesh targeting LOD levels:
  - LOD0: ~24,500 triangles (100% detail)
  - LOD1: ~5,000 triangles (~20%)
  - LOD2: ~2,000 triangles (~8%)
  - LOD3: ~500 triangles (~2%)
- [ ] Re-project ARKit shape keys onto decimated LOD topologies using barycentric transfer.
- [ ] Validate that all 52 shape keys exist and animate without topology tears across all 4 LODs.

#### Subphase 5.4: Skeletal Armature & Rigging
- [ ] Construct 5-joint armature hierarchy (`head`, `neck`, `jaw`, `eye_L`, `eye_R`) with correct joint center locations.
- [ ] Transfer FLAME joint regression weights to ICT-FaceKit topology via correspondence matrix $W$.
- [ ] Bind mesh to armature with Linear Blend Skinning weights.

#### Subphase 5.5: Headless Blender Export Script Fixes (P2 Fix)
- [ ] Fix vector arithmetic in `scripts/blender_export.py`: Use `mathutils.Vector` when applying shape key offsets.
- [ ] Add explicit object selection before export: `mesh_obj.select_set(True)`.
- [ ] Export clean FBX configured for Unreal Engine 5 (`FBX_SCALE_ALL`, `mesh_smooth_type='FACE'`, shape keys included).

**Phase 5 Gate:**
1. Correspondence matrix $W$ transfers geometry with $< 1\text{mm}$ error.
2. FBX contains all 52 ARKit blendshapes and 4 LOD levels.
3. Mesh imports into UE5 without smoothing group or scale warnings.
4. Driving `jawOpen` in UE5 Live Link articulates the jaw cleanly.

---

### Phase 6: Detail Hybridization & Static Facial Hair
> **Objective:** Enhance micro-displacement with diffusion-based detail synthesis, sharpen adversarially, and validate static facial hair geometry (stubble and short beards).

#### Subphase 6.1: UV-Space Denoising Diffusion Backbone
- [ ] Build 1-channel U-Net diffusion backbone operating on 512×512 displacement space.
- [ ] Condition on neutral position map, normal map, multi-view image features, and identity codes.
- [ ] Train diffusion model on preprocessed UV displacement dataset for coverage and stability.

#### Subphase 6.2: Adversarial Sharpening Pass
- [ ] Freeze trained diffusion backbone.
- [ ] Utilize Stage 3 PatchGAN discriminator as a sharpness critic.
- [ ] Fine-tune diffusion model with discriminator gradient feedback to enforce sharp pore and wrinkle boundaries.

#### Subphase 6.3: Static Facial Hair Validation
- [ ] Evaluate displacement fidelity on 20 subjects with visible facial hair (stubble, mustache, short beard).
- [ ] Determine if displacement maps sufficiently capture short hair volume.
- [ ] If displacement is insufficient, implement Approach 2: Static card geometry generation anchored to scalp/jaw mask.

**Phase 6 Gate:**
1. Hybrid model produces higher high-frequency FFT power spectrum than GAN alone.
2. Facial hair stubble/short beard is cleanly represented without blurring into skin.

---

### Phase 7: Headless Production Packaging & UE5 Live Link Export
> **Objective:** Integrate all pipeline stages into an automated production script that produces validated, game-ready assets and comprehensive run manifests. Runs on CPU sessions (0 GPU quota).

#### Subphase 7.1: Production Inference Script
- [ ] Build `scripts/production_inference.py` orchestrating Stages 0 → 1 → 2 → 3 → 5.
- [ ] Enforce automated input validation and neutral base mesh normalization.
- [ ] Package output directory structure:
  ```
  outputs/{run_id}/
  ├── head_lod0.fbx
  ├── head_lod1.fbx
  ├── head_lod2.fbx
  ├── head_lod3.fbx
  ├── displacement_512.png (or .npy)
  ├── preview_render.png
  └── manifest.json
  ```

#### Subphase 7.2: Comprehensive Manifest Generation
- [ ] Generate detailed `manifest.json` recording:
  - `pipeline_version`, `run_id`, `timestamp`.
  - Input photo metadata (filename, detection score, estimated yaw).
  - Model versions and checkpoint handles.
  - Normalization parameters (`displacement_p99_mm`).
  - Topology specs (ICT-FaceKit, ARKit-52, LOD vertex/triangle counts).
  - Identity retention scores against input photos.
  - `degraded` flag (indicating whether any stage ran in fallback mode).

#### Subphase 7.3: Unreal Engine 5 Validation
- [ ] Import exported FBX into Unreal Engine 5.4+.
- [ ] Connect mesh to Live Link Face iOS application via ARKit pose blueprint.
- [ ] Verify real-time facial tracking, blendshape delta fidelity, and LOD transition distances.

**Phase 7 Gate:**
1. Single command executes end-to-end inference from photos to FBX.
2. Live Link Face drives facial animation in UE5 at $\ge 60\text{ fps}$.
3. Run manifest accurately reflects all runtime parameters.

---

### Phase 8: Comprehensive Benchmarking & Quality Assurance
> **Objective:** Rigorously evaluate the complete pipeline against ground-truth 3D scans and benchmark models (NoW, FaceScape, Meshy 7.1).

#### Subphase 8.1: Geometric Metric Suite
- [ ] Implement point-to-surface Chamfer distance calculation.
- [ ] Implement surface normal angular error measurement.
- [ ] Implement high-frequency FFT power spectrum curvature comparison (surface-detail proxy).

#### Subphase 8.2: Benchmark Comparison Matrix
- [ ] Run evaluation suite on 50 held-out test subjects:
  - Compare coarse mesh against NoW benchmark ($< 0.90\text{mm}$ median error).
  - Compare full detailed geometry against ground-truth FaceScape scans.
  - Compare reconstructed meshes against Meshy 7.1 outputs on the same photo sets.
- [ ] Document results in project `README.md` and benchmark report.

**Phase 8 Gate:**
1. Statistically significant geometric improvement over Meshy 7.1 on facial micro-displacement.
2. NoW median error strictly $< 0.90\text{mm}$.
3. Automated evaluation suite runs headlessly and outputs standardized tables.

---

## Immediate Next Actions (Phase 0 Step 1)

1. **Step 0.1:** Replace `src/utils/flame_model.py` with `DOCS/flame_model.py` to fix the `exprdirs` KeyError and Chumpy unpickler bug.
2. **Step 0.2:** Remove silent fallbacks in `src/stage0_preprocess/detector.py`, `src/utils/validation.py`, and `evaluation/identity_score.py`.
3. **Step 0.3:** Update `requirements.txt` and fix Kaggle installation recipes.
4. **Step 0.4:** Write the Phase 0 test suite in `tests/` and run `pytest`.
