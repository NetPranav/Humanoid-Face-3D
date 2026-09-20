# Humanoid-Face-3D: Master Execution Dashboard & Runtime Log

*Automated execution tracking, artifact catalog, and live runtime diagnostics.*

---

## 🟢 1. System Environment & Verification Health

| Component | Status | Details / Active Reference |
| :--- | :---: | :--- |
| **Repository** | 🟢 Synced | [NetPranav/Humanoid-Face-3D](https://github.com/NetPranav/Humanoid-Face-3D) (`main` @ `0a92b31`) |
| **Unit Test Suite** | 🟢 100% Green | **89 / 89 tests passing** (`python3 -m unittest discover tests`) |
| **Kaggle Account** | 🟢 Authenticated | Account: `nightshowdown` (Active API Personal Access Token) |
| **Displacement Resolution**| 🟢 Verified | **1024×1024 Ultra-Resolution** (16-bit uint PNG, $p_{99} = 9.045\,\text{mm}$) |
| **Neck Seam Contract** | 🟢 Strictly Pinned | Displacements in lowest 20% collar bitwise pinned ($\Delta v \equiv 0$, $y_{\text{norm}} \le 0.20$) |

---

## 📡 2. Active Runtime Monitor (Live Processes)

### [IN-FLIGHT] Dataset Upload: `nightshowdown/mica-pretrained`
* **Process:** `kaggle datasets create -p models_cache/mica/` (PID: `24280`)
* **Payload:** MICA Pretrained Neural Weights (`pretrained.tar`, ~479 MB)
* **Status:** ⏳ **Uploading to Kaggle Google Cloud Storage** (TCP Socket Established to GCS endpoint)
* **Impact:** Once indexed by Kaggle backend, unlocks execution of Phase 1 Multi-View Inference.

### [RUNTIME DIAGNOSTIC] Phase 1 Kernel (Version 1 Run)
* **Kernel:** [`nightshowdown/phase-1-inference-baseline`](https://www.kaggle.com/code/nightshowdown/phase-1-inference-baseline) (Version 1)
* **Execution Status:** `KernelWorkerStatus.ERROR`
* **Root-Cause Analysis:**
  1. Kernel was pushed with dataset dependency `nightshowdown/mica-pretrained`.
  2. Because the dataset was still uploading and not yet indexed on Kaggle, Kaggle rejected adding `mica-pretrained` to the kernel session.
  3. Cell 3 executed safe guard: `FileNotFoundError: MICA checkpoint not found! Please attach dataset nightshowdown/mica-pretrained.`
* **Remediation:** As soon as upload completes, re-push/trigger version 2 with both datasets attached.

---

## 📋 3. Master Jobs Queue & Execution Pipeline

| Step | Job ID | Phase / Description | Source Code / Notebook | Target Hardware | Current Status | Key Output Artifacts |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **0** | `job_01` | **Phase 2.5: Geometry Preprocessing 1024²** | [`notebooks/kaggle/phase2_5_geometry_preprocessing.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase2_5_geometry_preprocessing.ipynb) | Kaggle CPU (0 GPU quota) | 🟢 **COMPLETE** | `outputs/uv_displacement_dataset_1024/`<br>• `subject_001_neutral_disp.png`<br>• `subject_001_neutral_norm.png`<br>• `normalization_stats.json` |
| **1** | `job_02_inference` | **Phase 1: Multi-View Reconstruction Baseline** | [`notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb) | Kaggle GPU (T4×1, ~3 min) | ⏳ **Awaiting Dataset Sync** | `outputs/phase1_baseline/`<br>• `head_mesh_neutral.obj`<br>• `stylized_chiseled/`<br>• `stylized_heroic/`<br>• `blendshapes_arkit52.json`<br>• `armature_rig.json` |
| **2** | `job_03_scale_data`| **Phase 2.5: Multi-Subject Scan Dataset** | `scripts/build_uv_displacement_dataset.py` | Kaggle CPU (0 GPU quota) | ⏹️ **Queued** | Batch of 20–50 subjects with 1024² 16-bit displacement maps for GAN training. |
| **3** | `job_04_detail_gan` | **Phase 3: High-Frequency Detail GAN** | [`notebooks/kaggle/phase3_detail_gan_train.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase3_detail_gan_train.ipynb) | Kaggle GPU (2×T4, 11.5h limit) | ⏹️ **Queued (after Step 2)** | Trained U-Net Generator (`ema_generator.pt`) synthesizing pore-level wrinkles. |
| **4** | `job_05_finetune` | **Phase 2: Demographic Identity Fine-Tuning** | [`notebooks/kaggle/phase2_identity_finetune.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/phase2_identity_finetune.ipynb) | Kaggle GPU (2×T4) | ⏸️ **On Hold** | Supervised MICA checkpoint (requires registered 3D scan FLAME betas). |

---

## 📦 4. Detailed Job Traceability & Output Catalog

### Job 01: Phase 2.5 Geometry Preprocessing 1024² (Completed)
* **Execution Environment:** Kaggle CPU Session (Kernel `nightshowdown/phase-2-5-geometry-preprocessing-1024`, Version 4)
* **Code Executed:** [`scripts/build_uv_displacement_dataset.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/scripts/build_uv_displacement_dataset.py)
* **Input Data:** FLAME 2020 Model (`data/flame_model/generic_model.pkl`) + Centroid-aligned 3D Head Mesh
* **Output Assets Verified Locally:**
  * `outputs/uv_displacement_dataset_1024/subject_001_neutral_disp.png` (1024×1024 16-bit uint PNG, $p_{99} = 9.045\,\text{mm}$)
  * `outputs/uv_displacement_dataset_1024/subject_001_neutral_norm.png` (1024×1024 8-bit RGB normal map)
  * `outputs/uv_displacement_dataset_1024/subject_001_neutral_mask.png` (1024×1024 facial validity mask)
  * `outputs/uv_displacement_dataset_1024/subject_001_neutral_pos.png` (1024×1024 3D spatial coordinate map)
  * `outputs/uv_displacement_dataset_1024/normalization_stats.json` (Scale: `9.045434`, Mean: `-0.003926`)

---

### Job 02: Phase 1 Multi-View Baseline (Packaged & Ready)
* **Execution Environment:** Kaggle GPU Session (Single T4, Internet: Enabled)
* **Code Executed:** [`notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb)
* **Input Data:**
  * Datasets: `nightshowdown/flame-model` + `nightshowdown/mica-pretrained`
  * Test Portraits: 4 demo subjects (`connelly`, `carell`, `lawrence`, `justin`) from `vendor/MICA/demo/input/`
* **Output Pipeline Stages:**
  * **Stage 0:** InsightFace `buffalo_l` face detection & 5-point landmark alignment.
  * **Stage 1:** MICA multi-view frontality-weighted embedding fusion ($w_i = \text{det\_score}_i \cdot \cos^2(\text{yaw}_i)$) $\to 300\text{-D } \beta$.
  * **Stage 2:** Canonical neutral base mesh normalization ($\psi = 0, \theta = 0$).
  * **Stage 5:** Production retopology, ARKit-52 blendshapes, 4-tier LOD chain, skeletal armature, and chiseled/heroic stylization.
* **Verification Gates:**
  * Gate 1: Pairwise identity divergence $\|\beta_a - \beta_b\| > 10^{-3}$ across all subject pairs (eliminates mean-face collapse).
  * Gate 2: Neck seam contract ($\Delta v \equiv 0$ on lowest 20% collar vertices).

---

## 📜 5. Chronological Engineering Milestones

* **`[STAGE 5 STYLIZATION]`** Implemented [`src/stage5_export/stylize.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/stage5_export/stylize.py): 8 continuous sliders (`jaw_width`, `chin_depth`, `chin_cleft`, `gonial_flare`, `cheekbone_prominence`, etc.) and presets (`chiseled`, `gigachad`, `heroic`).
* **`[UE5 RIGGING CONTRACT]`** Bitwise zero-pinned lowest 20% collar vertices ($y_{\text{norm}} \le 0.20$) using smooth $C^1$ Hermite blending ($y_{\text{norm}} \in [0.20, 0.32]$) to eliminate neck seam tearing in Unreal Engine 5.
* **`[TEST SUITE EXPANSION]`** Built [`tests/test_stylize.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/tests/test_stylize.py) with 13 tests verifying symmetry, scale-invariance (m vs mm), clamping, and collar pinning. Total test suite expanded to **89/89 tests passing**.
* **`[PIPELINE REPAIR]`** Fixed latent `MICAIdentityEncoder` import in [`src/pipeline.py`](file:///Users/pranav/Project%20Folder/3d%20Model%20/src/pipeline.py). Updated [`configs/default.yaml`](file:///Users/pranav/Project%20Folder/3d%20Model%20/configs/default.yaml) to support 1 to 5 input photos.
* **`[KAGGLE KERNEL PACKAGING]`** Built self-contained Phase 1 inference notebook in [`notebooks/kaggle/build_phase1/`](file:///Users/pranav/Project%20Folder/3d%20Model%20/notebooks/kaggle/build_phase1/). Pushed to GitHub as commit `0a92b31`.
* **`[KAGGLE CLOUD UPLOAD]`** Authenticated updated Kaggle Personal Access Token (`KGAT_7d44...`) and initiated background streaming upload of `nightshowdown/mica-pretrained` (~479 MB).
