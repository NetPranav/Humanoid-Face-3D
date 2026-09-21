# Humanoid-Face-3D: Master Execution Dashboard & Runtime Log

*Comprehensive automated pipeline execution tracking, technical artifact catalog, and live runtime diagnostics.*

---

## 🟢 1. System Health & Environment Verification

| Component | Status | Technical Details / Active Link |
| :--- | :---: | :--- |
| **Git Repository** | 🟢 Synced | [`NetPranav/Humanoid-Face-3D`](https://github.com/NetPranav/Humanoid-Face-3D) (`main`) |
| **Unit Test Suite** | 🟢 100% Passing | **101 / 101 tests passing** (`python3 -m unittest discover tests`) |
| **Kaggle Account** | 🟢 Authenticated | Username: `nightshowdown` (Personal Access Token active) |
| **Stage 3 Resolution**| 🟢 Verified | **1024×1024 Ultra-Resolution** (16-bit uint PNG, $p_{99} = 9.045\,\text{mm}$) |
| **Neck Seam Contract** | 🟢 Strictly Pinned | Collar vertices ($y_{\text{norm}} \le 0.20$) strictly pinned to $\Delta v \equiv 0$ |
| **Phase 1 Baseline** | 🟢 100% Complete | **157 assets (417.21 MB)** across 4 benchmark subjects verified on disk |
| **Stage 1.5 Residual** | 🟢 Verified | Graph convolutional network breaking FLAME linear ceiling with strict collar pinning |
| **Pixel3DMM Dense** | 🟢 Verified | Dense normal and UV prediction module for contour-anchored FLAME fitting |
| **Diff Rendering** | 🟢 Verified | Soft silhouette IoU + landmark reprojection losses wired into Stage 1 training |

---

## 📡 2. Execution History & Runtime Diagnostics

| Version | Status | Execution Time | Root-Cause / Diagnostic | Action Taken & Resolution |
| :---: | :---: | :---: | :--- | :--- |
| **v1** | ❌ Error | ~10s | Attached dataset `nightshowdown/mica-pretrained` was un-indexed on Kaggle backend. | Re-architected kernel to mount `flame-model` and stream `pretrained.tar` directly via high-speed CDN in 5s. |
| **v2** | ❌ Error | ~15s | Syntax error in Cell 3 due to unescaped quotes during bash string concatenation. | Built [`scripts/generate_phase1_nb.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/generate_phase1_nb.py) to syntax-check each cell with Python `compile(..., 'exec')`. |
| **v3** | ❌ Error | ~47s | PyTorch 2.6 defaulted `weights_only=True`, triggering unpickling failure on MICA NumPy arrays. | Patched [`src/stage1_identity/inference.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/inference.py) and [`src/stage2_expression/encoder.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage2_expression/encoder.py) with `weights_only=False` and `TypeError` fallback. |
| **v4** | 🟢 **COMPLETE** | **2m 14s** | None. All cells executed cleanly on Kaggle Tesla T4 GPU. | Verified Gate 1 ($\Delta \beta > 4.6$) and Gate 2 ($\Delta v = 0$ collar). Downloaded 157 assets locally. |

---

## 📋 3. Master Jobs Queue & Execution Pipeline

| Step | Job ID | Phase / Description | Generating Source / Notebook | Hardware | Status | Primary Output Files |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **0** | `job_01` | **Phase 2.5: Geometry Preprocessing 1024²** | [`scripts/build_uv_displacement_dataset.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/build_uv_displacement_dataset.py) | Kaggle CPU | 🟢 **COMPLETE** | `outputs/uv_displacement_dataset_1024/`<br>• `subject_001_neutral_disp.png`<br>• `subject_001_neutral_norm.png`<br>• `normalization_stats.json` |
| **1** | `job_02_inference` | **Phase 1: Multi-View Reconstruction Baseline** | [`notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb) | Kaggle GPU (T4×1) | 🟢 **COMPLETE** | `outputs/phase1_baseline/`<br>• `carell/`, `connelly/`, `justin/`, `lawrence/`<br>• `phase1_preview_grid.png`<br>• 157 files (417.21 MB) |
| **2** | `job_03_scale_data`| **Phase 2.5: Multi-Subject Scan Dataset** | [`scripts/build_uv_displacement_dataset.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/build_uv_displacement_dataset.py) | Kaggle CPU | ⏹️ **Queued** | 20–50 subjects with 1024² 16-bit displacement maps for GAN training. |
| **3** | `job_04_detail_gan` | **Phase 3: High-Frequency Detail GAN** | [`notebooks/kaggle/phase3_detail_gan_train.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase3_detail_gan_train.ipynb) | Kaggle GPU (2×T4) | ⏹️ **Queued** | Trained U-Net Generator (`ema_generator.pt`) synthesizing pore-level wrinkles. |
| **4** | `job_05_finetune` | **Phase 2: Demographic Identity Fine-Tuning** | [`notebooks/kaggle/phase2_identity_finetune.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase2_identity_finetune.ipynb) | Kaggle GPU (2×T4) | ⏸️ **On Hold** | Supervised MICA checkpoint (requires registered 3D scan FLAME betas). |

---

## 📦 4. Detailed Technical Output Catalog (Code-to-Artifact Mapping)

Every output artifact produced by the pipeline is cataloged below with its exact source code origin, execution parameters, format specifications, and downstream consumers.

### Phase 1: Multi-View Reconstruction Baseline (Completed & Verified)

* **Execution Driver:** [`notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb) (Kaggle GPU Tesla T4)
* **Generating Modules:**
  * Master Pipeline Orchestrator: [`src/pipeline.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/pipeline.py) (`FaceGeoPipeline.run()`)
  * Stage 0 (Preprocess): [`src/stage0_preprocess/detector.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage0_preprocess/detector.py) (`FaceDetector.detect_aligned()`)
  * Stage 1 (Identity): [`src/stage1_identity/mica_encoder.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/mica_encoder.py) (`MICAIdentityEncoder.encode_multiview()`)
  * Stage 2 (Expression/Pose): [`src/stage2_expression/encoder.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage2_expression/encoder.py) (`ExpressionEncoder.encode()`)
  * Stage 5 (Production Export): [`src/stage5_export/exporter.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/exporter.py) (`Stage5Exporter.export_production_asset()`)
  * Parametric Stylizer: [`src/stage5_export/stylize.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/stylize.py) (`FaceStylizer.apply_stylization()`)
* **Local Delivery:** `outputs/phase1_baseline/` (157 files, 417.21 MB)
* **Validation Gate Results:**
  * **Gate 1 (Identity Divergence):** $\|\beta_a - \beta_b\|_2 \gg 10^{-3}$ across all pairs.
    * `carell` vs `connelly`: **$4.8511$**
    * `carell` vs `justin`: **$7.1725$**
    * `carell` vs `lawrence`: **$7.0147$**
    * `connelly` vs `justin`: **$5.2351$**
    * `connelly` vs `lawrence`: **$5.4214$**
    * `justin` vs `lawrence`: **$4.6134$**
  * **Gate 2 (Neck Seam Pinning Contract):** Lowest 20% collar vertices ($y_{\text{norm}} \le 0.20$) strictly verified $\Delta v = 0.000000\,\text{mm}$ across all stylized variants.

#### Subject Asset Matrix (`outputs/phase1_baseline/<subject>/`):

| File Path / Pattern | Technical Specifications | Purpose & Downstream Consumer |
| :--- | :--- | :--- |
| `head_mesh_neutral.obj` | 3D Wavefront OBJ (5,023 vertices, 9,976 triangular faces) | Canonical neutral base mesh ($\psi=0, \theta=0$) decoded from frontality-weighted 300-D $\beta$. Unreal Engine 5 base asset. |
| `manifest.json` | JSON format (300-D $\beta$, vertex count, photo paths) | Full reconstruction audit trail and identity descriptor vector. |
| `head_mesh.png` | 512×512 8-bit RGB PNG (orthographic frontal render) | Rendered validation thumbnail generated without display server via OpenCV polygon rasterizer. |
| `blendshapes_arkit52.json` | JSON format (52 Apple ARKit blendshapes, $\Delta v \in \mathbb{R}^{5023 \times 3}$, ~14.8 MB) | Complete semantic deformation targets for real-time facial performance capture in UE5 Live Link. |
| `armature_rig.json` | JSON format (5 anatomical joints: Neck, Head, LeftEye, RightEye, Jaw) | Linear blend skinning (LBS) bone hierarchy and vertex weights for game skeleton attachment. |
| `face_lod[0-3].obj` | 4-tier quadric decimation chain (LOD0: 9,976 tris, LOD1: 1,120 tris, LOD2: 1,120 tris, LOD3: 268 tris) | Performance-tiered meshes with shape keys preserved for runtime game engine rendering. |
| `blendshapes_lod[0-3].json` | Decimated ARKit-52 shape key deltas corresponding to each LOD | Topology-matched blendshape targets for decimated LOD meshes. |
| `stylized_chiseled/` | Subdirectory containing stylized base mesh, blendshapes, LODs, and rig | Parametrically deformed chiseled jaw/chin variant (~17.2mm max displacement) with bitwise collar boundary pinning ($\Delta v = 0$). |
| `stylized_heroic/` | Subdirectory containing stylized base mesh, blendshapes, LODs, and rig | Comic/action superhero stylized variant (~12.2mm max displacement) with preserved neck seam boundary. |
| `stylized_gigachad/` | Subdirectory containing stylized base mesh, blendshapes, LODs, and rig | Exaggerated hyper-masculine jaw/chin variant (~23.3mm max displacement) with bitwise collar boundary pinning ($\Delta v = 0$). |
| `phase1_preview_grid.png` | 2048×512 composite comparison image across all 4 subjects | Master visual verification matrix confirming distinct identity morphology. |
| `stylization_comparison_boosted.png`| 1920×640 3/4 perspective comparison | Multi-preset comparison (Neutral vs Heroic vs Chiseled vs Gigachad) proving visual pop with zero neck seam tear. |

---

### Phase 2.5: Geometry Preprocessing 1024² (Completed)

* **Generating Source Code:**
  * Script: [`scripts/build_uv_displacement_dataset.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/build_uv_displacement_dataset.py) (`main()` function)
  * Geometry Module: [`src/stage3_detail/geometry.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage3_detail/geometry.py) (`bake_displacement_map_from_meshes`)
* **Execution Invocation:**
  `python3 scripts/build_uv_displacement_dataset.py --mesh_dir data/scans --flame_model data/flame_model/generic_model.pkl --resolution 1024 --output_dir outputs/uv_displacement_dataset_1024`
* **Artifact Files Generated:**

| File Path | Technical Specifications | Purpose & Downstream Consumer |
| :--- | :--- | :--- |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_disp.png` | 1024×1024, 1-channel, 16-bit uint PNG ($p_{99} = 9.045\,\text{mm}$) | Ground-truth high-frequency micro-displacement rasterized from raw 3D scan to FLAME UV coordinates. Input to Phase 3 Detail GAN. |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_norm.png` | 1024×1024, 3-channel, 8-bit RGB PNG (UV tangent space normals) | Surface normal tensor input providing geometric curvature priors to the U-Net Generator (`src/stage3_detail/model.py`). |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_pos.png` | 1024×1024, 3-channel, 8-bit RGB PNG (normalized XYZ spatial field) | 3D vertex coordinate tensor input concatenated with normal map to form the 6-channel input condition. |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_mask.png` | 1024×1024, 1-channel, 8-bit binary PNG | Valid facial UV unwrapping boundary mask; excludes non-face UV islands and prevents loss penalty on seams. |
| `outputs/uv_displacement_dataset_1024/normalization_stats.json` | JSON format (`scale`: `9.045434`, `mean`: `-0.003926`) | Millimeter de-normalization parameters required at inference time to map uint16 values back to true metric units. |

---

## 📜 5. Chronological Engineering Milestones

* **`[PHASE 1 COMPLETE]`** Successfully ran end-to-end Phase 1 Multi-View Reconstruction on Kaggle GPU (Version 4, 2m 14s). Reconstructed 4 benchmark subjects (`carell`, `connelly`, `justin`, `lawrence`), cleared Gate 1 & Gate 2, and downloaded all 157 assets (417.21 MB).
* **`[PYTORCH 2.6 FIX]`** Fixed PyTorch 2.6 `WeightsUnpickler` exception by adding `weights_only=False` with `TypeError` fallbacks across MICA and SMIRK checkpoint loaders ([`src/stage1_identity/inference.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/inference.py), [`src/stage2_expression/encoder.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage2_expression/encoder.py)).
* **`[NOTEBOOK COMPILER]`** Created [`scripts/generate_phase1_nb.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/generate_phase1_nb.py) to compile and verify all Jupyter cells before notebook generation, eliminating string-escaping and JSON parsing bugs.
* **`[STAGE 5 STYLIZATION]`** Built continuous deformation sliders and presets (`chiseled`, `heroic`) in [`src/stage5_export/stylize.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/stylize.py) with bitwise collar pinning.
* **`[UNIT TEST SUITE EXPANSION]`** **101 / 101 tests passing** locally (`python3 -m unittest discover tests`). Added test suites for Stage 1.5 residual network, Pixel3DMM fitting, and differentiable rendering.
* **`[STYLIZATION SCALE BOOST & GIGACHAD PRESET]`** Boosted deformation scaling factors in [`src/stage5_export/stylize.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/stylize.py) from $0.035$ up to $0.095 \times H$. Max jaw/chin displacement now reaches $17.24\,\text{mm}$ (Chiseled) and $23.30\,\text{mm}$ (Gigachad) while collar vertices retain bitwise strict zero displacement ($\Delta v \equiv 0.000000\,\text{mm}$). All 4 subjects re-exported across neutral, chiseled, heroic, and gigachad with preserved ARKit-52 blendshapes and 4-tier LODs.
* **`[BETA-NORM MATCHING & COLLAPSE SMOKE TEST]`** Implemented explicit β-norm matching loss term in [`src/stage1_identity/trainer.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/trainer.py) and created [`scripts/beta_collapse_smoke_test.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/beta_collapse_smoke_test.py). Validated that explicit norm penalization prevents magnitude collapse toward the population mean face ($\beta=0$) under optimizer weight decay.
* **`[ARCFACE THRESHOLD CALIBRATION]`** Created [`scripts/calibrate_arcface_threshold.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/calibrate_arcface_threshold.py) and generated [`configs/arcface_calibration.json`](file:///Users/pranav/Project%20Folder/3d%20Model%20/configs/arcface_calibration.json) with empirical ROC, FAR, FRR, and EER distributions.
* **`[PIXEL3DMM DENSE PER-PIXEL FITTING]`** Built [`src/stage1_identity/pixel3dmm_fitter.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/pixel3dmm_fitter.py) with ViT-based per-pixel surface normal and UV coordinate decoders plus `DenseFLAMEFitter` for contour-anchored FLAME fitting, integrated into [`src/pipeline.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/pipeline.py).
* **`[STAGE 1.5 MACRO-SHAPE RESIDUAL NETWORK]`** Implemented [`src/stage1_5_residual/residual_net.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_5_residual/residual_net.py) and [`src/stage1_5_residual/trainer.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_5_residual/trainer.py). Graph convolutional vertex displacement network with Laplacian smoothness and bitwise collar pinning ($\Delta v \equiv 0.0$ for $y_{\text{norm}} \le 0.20$), breaking the linear PCA ceiling to minimize manual artist sculpt passes.
* **`[DIFFERENTIABLE RENDERING IN STAGE 1 TRAINING]`** Built [`src/stage1_identity/diff_render.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/diff_render.py) with soft silhouette IoU loss and 2D landmark reprojection loss, wired into [`src/stage1_identity/trainer.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/trainer.py).

