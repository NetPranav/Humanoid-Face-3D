# Humanoid-Face-3D: Master Execution Dashboard & Runtime Log

*Comprehensive automated pipeline execution tracking, technical artifact catalog, and live runtime diagnostics.*

---

## 🟢 1. System Health & Environment Verification

| Component | Status | Technical Details / Active Link |
| :--- | :---: | :--- |
| **Git Repository** | 🟢 Synced | [`NetPranav/Humanoid-Face-3D`](https://github.com/NetPranav/Humanoid-Face-3D) (`main` @ [`2bbe37f`](https://github.com/NetPranav/Humanoid-Face-3D/commit/2bbe37f)) |
| **Unit Test Suite** | 🟢 100% Passing | **89 / 89 tests passing** (`python3 -m unittest discover tests`) |
| **Kaggle Account** | 🟢 Authenticated | Username: `nightshowdown` (Personal Access Token active) |
| **Stage 3 Resolution**| 🟢 Verified | **1024×1024 Ultra-Resolution** (16-bit uint PNG, $p_{99} = 9.045\,\text{mm}$) |
| **Neck Seam Contract** | 🟢 Bitwise Pinned | Collar vertices ($y_{\text{norm}} \le 0.20$) strictly pinned to $\Delta v \equiv 0$ |

---

## 📡 2. Active Runtime Monitor (Live GPU Processes)

### [ACTIVE KERNEL RUN] Phase 1 Multi-View Reconstruction Baseline
* **Kaggle Kernel:** [`nightshowdown/phase-1-inference-baseline`](https://www.kaggle.com/code/nightshowdown/phase-1-inference-baseline) (Version 3)
* **Status:** 🟢 **`KernelWorkerStatus.RUNNING`**
* **Hardware Allocation:** Kaggle GPU (1× NVIDIA Tesla T4 16GB VRAM, CUDA 12.x)
* **Driver Notebook:** [`notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb)
* **Active Steps in Container:**
  1. Setting up repository workspace and submodules (`vendor/MICA`).
  2. Installing GPU acceleration libraries (`insightface`, `onnxruntime-gpu`, `trimesh`).
  3. Mounting `nightshowdown/flame-model` (`generic_model.pkl`) + high-speed CDN stream for MICA weights (`pretrained.tar`, ~479 MB).
  4. Staging multi-subject portrait benchmarks (`connelly`, `carell`, `lawrence`, `justin`).
  5. End-to-end multi-view frontality-weighted 3D reconstruction $\to$ canonical neutral base mesh ($\psi=0, \theta=0$).
  6. Stage 5 ARKit-52 blendshape extraction, 4-tier LOD generation, 5-joint skeletal rig, and parametric stylization (`chiseled`, `heroic`).
  7. Automated validation gates (pairwise $\beta$ divergence & collar pinning).

---

## 📋 3. Master Jobs Queue & Execution Pipeline

| Step | Job ID | Phase / Description | Generating Source / Notebook | Hardware | Status | Primary Output Files |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **0** | `job_01` | **Phase 2.5: Geometry Preprocessing 1024²** | [`scripts/build_uv_displacement_dataset.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/build_uv_displacement_dataset.py) | Kaggle CPU | 🟢 **COMPLETE** | `outputs/uv_displacement_dataset_1024/`<br>• `subject_001_neutral_disp.png`<br>• `subject_001_neutral_norm.png`<br>• `normalization_stats.json` |
| **1** | `job_02_inference` | **Phase 1: Multi-View Reconstruction Baseline** | [`notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb) | Kaggle GPU (T4×1) | 🟢 **RUNNING (v3)** | `outputs/phase1_baseline/`<br>• `head_mesh.obj`<br>• `stylized_chiseled/`<br>• `stylized_heroic/`<br>• `blendshapes_arkit52.json`<br>• `armature_rig.json` |
| **2** | `job_03_scale_data`| **Phase 2.5: Multi-Subject Scan Dataset** | [`scripts/build_uv_displacement_dataset.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/build_uv_displacement_dataset.py) | Kaggle CPU | ⏹️ **Queued** | 20–50 subjects with 1024² 16-bit displacement maps for GAN training. |
| **3** | `job_04_detail_gan` | **Phase 3: High-Frequency Detail GAN** | [`notebooks/kaggle/phase3_detail_gan_train.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase3_detail_gan_train.ipynb) | Kaggle GPU (2×T4) | ⏹️ **Queued** | Trained U-Net Generator (`ema_generator.pt`) synthesizing pore-level wrinkles. |
| **4** | `job_05_finetune` | **Phase 2: Demographic Identity Fine-Tuning** | [`notebooks/kaggle/phase2_identity_finetune.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase2_identity_finetune.ipynb) | Kaggle GPU (2×T4) | ⏸️ **On Hold** | Supervised MICA checkpoint (requires registered 3D scan FLAME betas). |

---

## 📦 4. Detailed Technical Output Catalog (Code-to-Artifact Mapping)

Every output artifact produced by the system is documented below with its exact source code origin, execution parameters, format specifications, and downstream consumers.

### Phase 2.5: Geometry Preprocessing 1024² (Completed)

* **Generating Source Code:**
  * Script: [`scripts/build_uv_displacement_dataset.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/build_uv_displacement_dataset.py) (`main()` function)
  * Geometry Module: [`src/stage3_detail/geometry.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage3_detail/geometry.py) (`bake_displacement_map_from_meshes`)
  * Notebook Wrapper: [`notebooks/kaggle/phase2_5_geometry_preprocessing.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase2_5_geometry_preprocessing.ipynb)
* **Execution Invocation:**
  `python3 scripts/build_uv_displacement_dataset.py --mesh_dir data/scans --flame_model data/flame_model/generic_model.pkl --resolution 1024 --output_dir outputs/uv_displacement_dataset_1024`
* **Artifact Files Generated:**

| File Path | Technical Specifications | Purpose & Downstream Consumer |
| :--- | :--- | :--- |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_disp.png` | 1024×1024, 1-channel, 16-bit unsigned integer PNG ($p_{99} = 9.045\,\text{mm}$) | Ground-truth high-frequency micro-displacement rasterized from raw 3D scan to FLAME UV coordinates. Consumed by Phase 3 Detail GAN Generator loss. |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_norm.png` | 1024×1024, 3-channel, 8-bit RGB PNG (UV tangent space normals) | Surface normal tensor input providing geometric curvature priors to the U-Net Generator (`src/stage3_detail/model.py`). |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_pos.png` | 1024×1024, 3-channel, 8-bit RGB PNG (normalized XYZ spatial field) | 3D vertex coordinate tensor input concatenated with normal map to form the 6-channel input condition. |
| `outputs/uv_displacement_dataset_1024/subject_001_neutral_mask.png` | 1024×1024, 1-channel, 8-bit binary PNG | Valid facial UV unwrapping boundary mask; excludes non-face UV islands and prevents loss penalty on seams. |
| `outputs/uv_displacement_dataset_1024/normalization_stats.json` | JSON format (`scale`: `9.045434`, `mean`: `-0.003926`) | Millimeter de-normalization parameters required at inference time to map uint16 values back to true metric units. |

---

### Phase 1: Multi-View Reconstruction Baseline (Currently Executing)

* **Generating Source Code:**
  * Master Pipeline Orchestrator: [`src/pipeline.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/pipeline.py) (`FaceGeoPipeline.run()`)
  * Stage 0 (Preprocess): [`src/stage0_preprocess/detector.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage0_preprocess/detector.py) (`FaceDetector.detect_aligned()`)
  * Stage 1 (Identity): [`src/stage1_identity/mica_encoder.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage1_identity/mica_encoder.py) (`MICAIdentityEncoder.encode_multiview()`)
  * Stage 2 (Expression/Pose): [`src/stage2_expression/encoder.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage2_expression/encoder.py) (`ExpressionEncoder.encode()`)
  * Stage 5 (Production Export): [`src/stage5_export/exporter.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/exporter.py) (`Stage5Exporter.export_production_asset()`)
  * Parametric Stylizer: [`src/stage5_export/stylize.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/stylize.py) (`FaceStylizer.apply_stylization()`)
* **Execution Invocation:**
  `kaggle kernels push -p notebooks/kaggle/build_phase1/`
* **Artifact Files Produced Per Subject (`outputs/phase1_baseline/<subject_name>/`):**

| File Path | Technical Specifications | Purpose & Downstream Consumer |
| :--- | :--- | :--- |
| `head_mesh.obj` | 3D Wavefront OBJ (5023 vertices, 9976 triangular faces) | Canonical neutral base mesh ($\psi=0, \theta=0$) decoded from frontality-weighted 300-D $\beta$. UE5 base geometry. |
| `manifest.json` | JSON format (300-D $\beta$, expression parameters, vertex count) | Complete reconstruction audit log and identity vector descriptor for validation gates. |
| `head_mesh.png` | 512×512 8-bit RGB PNG (orthographic frontal render) | Autonomous preview thumbnail generated via headless OpenCV polygon rasterizer. |
| `blendshapes_arkit52.json` | JSON format (52 Apple ARKit blendshapes, $\Delta v \in \mathbb{R}^{5023 \times 3}$) | Standardized facial animation targets for Unreal Engine 5 Live Link tracking. |
| `armature_rig.json` | JSON format (5 anatomical joints: Neck, Head, LeftEye, RightEye, Jaw) | Linear blend skinning (LBS) bone hierarchy and vertex weights for game skeleton attachment. |
| `lods/head_mesh_lod[0-3].obj` | 4-tier decimated meshes (LOD0: 100%, LOD1: ~50%, LOD2: ~25%, LOD3: ~10%) | Performance-tiered meshes with shape keys preserved for real-time rendering. |
| `stylized_chiseled/head_mesh.obj` | 3D Wavefront OBJ with chiseled jaw/brow deformation | Game-ready stylized variant with strict neck collar boundary pinning ($\Delta v = 0$). |
| `stylized_heroic/head_mesh.obj` | 3D Wavefront OBJ with heroic comic proportions | Stylized superhero silhouette variant with preserved neck seam boundary. |
| `../phase1_preview_grid.png` | Composite comparison image of all reconstructed subjects | Master visual verification matrix across all benchmark subjects. |

---

## 📜 5. Chronological Engineering Milestones

* **`[STAGE 2 FIX]`** [`2bbe37f`](https://github.com/NetPranav/Humanoid-Face-3D/commit/2bbe37f): Added `allow_neutral=True` in [`src/pipeline.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/pipeline.py) so the pipeline gracefully falls back to canonical neutral pose ($\psi=0, \theta=0$) when running in baseline inference mode without SMIRK weights.
* **`[NOTEBOOK GENERATOR]`** Created [`scripts/generate_phase1_nb.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/generate_phase1_nb.py) to compile and syntax-verify every notebook cell with Python `compile()`, completely preventing unescaped strings or Papermill JSON parsing failures.
* **`[KAGGLE KERNEL V3]`** Pushed Version 3 of `nightshowdown/phase-1-inference-baseline` with automated submodule cloning, high-speed CDN weight streaming, and automated gate assertions.
* **`[STAGE 5 STYLIZATION]`** Built continuous deformation sliders and presets (`chiseled`, `heroic`) in [`src/stage5_export/stylize.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/stylize.py) with bitwise collar pinning.
* **`[UNIT TEST EXPANSION]`** Full test suite expanded to **89/89 tests passing** (`python3 -m unittest discover tests`).
