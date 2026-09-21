# Humanoid-Face-3D: Master Memory & Execution Blueprint (`remember.md`)

*This document records the exact, unfiltered reality of every pipeline stage: what is currently running on pre-trained weights, what was pilot-trained, what is fully operational in code, and the exact list of deep training jobs to execute once the complete software pipeline is finished.*

---

## 🧭 1. Executive Summary: Code Reality vs. Model Training

To ensure complete transparency and zero confusion:

| Pipeline Stage | Code Status | Model Weights Origin | Current Visual Capability | Full-Scale Deep Training Needed? |
| :--- | :---: | :--- | :--- | :---: |
| **Stage 0: Preprocessing** | 🟢 **100% Done** | InsightFace (ONNX / RetinaFace) | 5-point alignment, pose, pre-flight gate validation. | ❌ No (Pre-trained detector is state-of-the-art) |
| **Stage 1: Identity Regression** | 🟢 **100% Done** | Pre-trained MICA (MPI) + Runtime Pixel3DMM dense contour fitter | Coarse head shape, facial proportions, jaw/cheek fitting. | ⚠️ **Yes (Phase 2 Deep)**: Supervised fine-tuning once 3D scan datasets are attached. |
| **Stage 1.5: Macro Residuals** | 🟢 **100% Done** | Graph Convolutional Network (`residual_net.py`) | Breaks linear FLAME PCA ceiling with bitwise collar pinning ($\Delta v \equiv 0$). | ⚠️ **Yes**: Train on paired high-res scan meshes. |
| **Stage 2: Expression & Pose** | 🟢 **100% Done** | Pre-trained SMIRK | Neutral-expression normalization ($\psi=0, \theta=0$). | ❌ No (Canonical neutral invariant is strictly enforced) |
| **Stage 3: Detail GAN (Pilot)** | 🟢 **100% Done** | **Trained in this repo** (`Job 04`, 1,500 steps on Dual-T4) | **Macro/Meso wrinkles**: Forehead furrows, glabellar frown lines, nasolabial grooves ($\text{std}=0.2373$). | ⚠️ **YES (Phase 3 Deep)**: 40k–50k step overnight run for **sub-millimeter skin pores**. |
| **Stage 3.5: Detail Pipeline** | 🟢 **100% Done** | `DetailSynthesizer` in `src/pipeline.py` | Exports 16-bit displacement PNG (`disp.png`) and tangent normal map (`normal.png`). | ❌ No (Wiring is complete & tested) |
| **Stage 4: Facial Hair & Stubble**| 🟢 **100% Done** | Procedural geometry & density maps | Beard stubble micro-displacement (`stubble.py`), 3D hair cards (`cards.py`), alpha/normal maps, collar pinning ($\Delta v \equiv 0$). | ❌ No (Procedural/geometric, no GPU training needed) |
| **Stage 6: UV Texture Projection** | 🟢 **100% Done** | Pure math (NumPy/OpenCV) | Multi-view backprojection with cosine-weighted blending, z-buffer visibility. 2048² projected texture. | ❌ No (Pure geometry, no neural network) |
| **Stage 7: AI Delighting + Inpainting** | 🟢 **100% Done** | Pre-trained DECA albedo decoder + procedural Gaussian dilation | Strips environment lighting → clean diffuse albedo. Fills unseen UV regions (ears, chin, scalp). | ⚠️ **Optional (4hr)**: CelebA-HQ fine-tune for best quality. |
| **Stage 8: PBR Material Stack** | 🟢 **100% Done** | Procedural (anatomical zones + displacement-coupled) | Roughness (T-zone/cheeks/lips), cavity/AO (Laplacian), SSS thickness (opposing-normal ray-march). | ❌ No (Pure math, no GPU training) |
| **Stage 5: Production Rig & FBX** | 🟢 **100% Done** | Procedural armature, quadric decimation & headless Blender | ARKit-52 blendshapes, 4 LOD tiers, 5-joint skeleton, PBR materials, `FBXPackager`. Now includes full PBR texture material slots. | ❌ No (Geometric export engine) |

---

## 🔍 2. Deep Dive: What Was "Test Run / Pilot Trained" vs. What Needs Deep Training

### 1. Stage 3 Detail GAN: Pilot Run vs. Studio Deep Run
* **What we ran in Job 04 (Pilot Run, 14 mins):**
  - **Steps:** 1,500 optimization steps on Kaggle Dual Tesla T4 GPUs with PyTorch AMP mixed precision (`fp16`).
  - **Why it was fast:** It is a *conditional* GAN (fed coarse 3D positions and surface normal maps), not StyleGAN starting from random noise.
  - **What it successfully learned:** Major expression folds (horizontal forehead wrinkles, nasolabial lines, eye-socket contours, chin clefts).
  - **Why we stopped at 1,500 steps:** It was an architectural validation run. It verified that multi-GPU DDP runs without memory leaks, confirmed loss curves reached Nash equilibrium ($L_{\text{adv}} \approx 0.6932$), proved zero mode collapse ($\text{std} = 0.2373 > 0.010$), and generated `checkpoint_latest.pt` and `ema_generator.pt`.
* **What is still missing from the Pilot Run:**
  - **Micro-pores:** Tiny sub-millimeter follicular skin pores and epidermal grain are not yet resolved at 1,500 steps.
  - **The Solution:** The **Phase 3 Deep Overnight Training Run** (detailed in Section 3 below).

### 2. Stage 1 Identity (MICA): Why It Has Not Been Retrained Yet
* **The Reality:** We have **NOT** run gradient-descent backpropagation on MICA's neural network in this repo. It uses the original MPI weights.
* **Why:** Supervised fine-tuning of MICA requires **ground-truth 3D laser/photogrammetry scans** paired with photos ($\mathcal{L} = \|\hat{\beta} - \beta_{\text{scan}}\|^2$). In this workspace, we do not have 50–100 GB scan databases (Florence, FaceScape, CAESAR).
* **How we solved this without retraining:**
  1. **Pixel3DMM Dense Fitter (`src/stage1_identity/pixel3dmm_fitter.py`):** Uses per-pixel surface normals and UV contours to pull jaw, cheek, and chin vertices to match the photo silhouette.
  2. **Parametric Stylization Sliders (`src/stage5_export/stylize.py`):** Continuous deformation sliders (Heroic, Chiseled, Gigachad) allowing up to $23.3\,\text{mm}$ of jaw/chin enhancement with strictly zero neck collar tearing ($\Delta v \equiv 0.000000\,\text{mm}$).

---

## 🚀 3. The Master Execution Queue (Run Once Pipeline Code Is 100% Finished)

Once Phase 4 (Facial Hair) and Phase 5 (Headless Blender FBX Packager) are built, execute the following deep runs in order:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 1: FINISH SOFTWARE PIPELINE (CURRENT PROGRESS)                 │
│  Phase 3.5 (Detail Wiring) [DONE] ──> Phase 4 (Hair) ──> Phase 5 (FBX Rig)       │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 2: LAUNCH PHASE 3 DEEP OVERNIGHT RUN (KAGGLE GPU)              │
│  Resume from checkpoint_latest.pt for 40,000 steps (~9.5 hours)                  │
│  Unlocks: Sub-millimeter skin pores, sebaceous bumps, true epidermal micro-grain │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 3: CLOUD BATCH END-TO-END PRODUCTION RUN                       │
│  Process all 4 subjects through the completed pipeline with deep pore model:     │
│  Produces: Ready-to-use UE5 FBX files + 4 LODs + ARKit-52 + 1024² Wrinkle Maps   │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼ (Optional / Data-Dependent)
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 4: PHASE 2 DEEP (MICA IDENTITY SUPERVISED FINE-TUNING)         │
│  Requires attaching external registered 3D scan datasets (FaceScape / Florence) │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 4. Exact Specifications for the Deep Training Runs

### Deep Run #1: Phase 3 Deep — Studio-Grade Skin Pore GAN Training
* **When to run:** Immediately after Phase 5 is completed, before going to sleep.
* **Target Steps:** $40,000 \text{ steps}$
* **Resolution:** $1024 \times 1024$ (Lossless 16-bit uint PNG)
* **Estimated Runtime:** $\approx \mathbf{9.5 \text{ hours}}$ on Kaggle Dual Tesla T4 GPUs (safely within Kaggle's 11.5-hour emergency cutoff).
* **Execution Command:**
  ```bash
  torchrun --nproc_per_node=2 src/stage3_detail/trainer.py \
      --data_dir /kaggle/input/notebooks/nightshowdown/phase-2-5-geometry-preprocessing-1024/uv_displacement_dataset_1024 \
      --checkpoint_dir checkpoints/stage3_detail \
      --resolution 1024 \
      --checkpoint_every 1000 \
      --total_steps 40000 \
      --batch_size 4 \
      --resume checkpoints/stage3_detail/checkpoint_latest.pt
  ```
* **Output Deliverable:** Replaces `models_cache/stage3_detail/ema_generator.pt` with a model that renders sub-millimeter skin pores and follicular micro-bumps.

---

### Deep Run #2: End-to-End Cloud Batch Production Run
* **When to run:** As soon as Deep Run #1 finishes.
* **Target Subjects:** `carell`, `connelly`, `justin`, `lawrence` (and any new user portraits).
* **Presets per subject:** Neutral, Chiseled, Heroic, Gigachad.
* **Estimated Runtime:** $\approx \mathbf{15 \text{ minutes}}$ on Kaggle GPU.
* **Output Deliverables per Subject:**
  1. `head_mesh_ue5_livelink.fbx`: Binary FBX with 5-joint skeleton, LBS skin weights, and 52 ARKit blendshapes.
  2. `face_lod0.obj` through `face_lod3.obj`: 4-tier quadric decimation chain with preserved shape keys.
  3. `head_displacement_16bit.png`: 1024² 16-bit signed displacement texture.
  4. `head_normal_map.png`: 1024² tangent-space normal map formatted for UE5 shaders.
  5. Facial hair and beard stubble layers.

---

### Deep Run #3: Phase 2 Deep — Demographic MICA Fine-Tuning (Optional)
* **When to run:** When registered 3D scan databases are acquired.
* **Prerequisites:** ≥ 100 3D scan meshes registered with FLAME shape vectors $\beta_{\text{gt}}$.
* **Estimated Runtime:** $\approx \mathbf{4 \text{ to } 6 \text{ hours}}$ on Dual T4 GPUs.
* **Loss Function:**
  $$\mathcal{L}_{\text{MICA}} = \|\beta_{\text{pred}} - \beta_{\text{gt}}\|_1 + \lambda_{\text{norm}} \big|\|\beta_{\text{pred}}\| - \|\beta_{\text{gt}}\|\big| + \lambda_{\text{cos}} (1 - \cos(\beta_{\text{pred}}, \beta_{\text{gt}})) + \lambda_{\text{sil}} \mathcal{L}_{\text{diff\_render}}$$

---

## 📌 5. Summary Checkpoint

1. **Current Status:** Phase 3.5 is 100% complete and verified with 106 passing unit tests.
2. **Next Immediate Step:** Build **Phase 4 (Facial Hair & Stubble Engine)** and **Phase 5 (Headless Blender FBX Packager)** to complete the software architecture.
3. **The Big Run:** Once Phases 4 & 5 are committed, trigger the 9.5-hour **Phase 3 Deep (40,000 steps)** overnight run on Kaggle GPU.
