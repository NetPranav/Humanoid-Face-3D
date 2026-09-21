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

The complete software pipeline (Stages 0–8 + Stage 5 Export) is 100% finished and verified with 153 passing tests. Execute the following training runs and production steps in order:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 1: FINISH SOFTWARE PIPELINE (COMPLETE ✅)                      │
│  Stages 0–4 [DONE] ──> Stage 5 Rig/FBX [DONE] ──> Stages 6–8 PBR Texture [DONE]  │
│  153 unit tests passing. Commit: 240f04a on main                                 │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 2: LAUNCH PHASE 3 DEEP RUN (KAGGLE GPU — RUNNING 🔄)          │
│  Resume from checkpoint_latest.pt for 40,000 steps (~9.5 hours on Dual T4)       │
│  Unlocks: Sub-millimeter skin pores, sebaceous bumps, true epidermal micro-grain │
│  Output: ema_generator.pt → models_cache/stage3_detail/                          │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 3: DELIGHTING FINE-TUNE — Albedo Training (TEXTURE ⏹)         │
│  Train Stage 7 DelightUNet on 30,000 CelebA-HQ 1024² images projected onto UV   │
│  • Estimated: ~4 hours on Kaggle Dual T4                                         │
│  • Improves: Albedo quality from "good (DECA)" to "film-grade (custom-tuned)"    │
│  • Output: delight_unet.pt → models_cache/stage7_delight/                        │
│  • Why before production batch: Train all models FIRST so the final assets       │
│    benefit from the highest-quality delighting and pore geometry.                │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼ (Data-Dependent)
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 4: PHASE 2 DEEP — MICA IDENTITY SUPERVISED FINE-TUNING (⏹)     │
│  Supervised fine-tuning of MICA on registered 3D scan datasets (FaceScape/LYHM)  │
│  • Requires: ≥100 registered 3D scans with ground-truth β                        │
│  • Estimated: 4–6 hours on Kaggle Dual T4                                        │
│  • Improves: Identity accuracy across diverse demographics                       │
│  • Output: mica_finetuned.pt → models_cache/mica/                                │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              STEP 5: FINAL PRODUCTION BATCH RUN (BEST MODELS → BEST ASSETS ⏹)    │
│  Process all benchmark subjects (carell, connelly, justin, lawrence + user)     │
│  using all peak models (best Phase 3 GAN + best Delighting + best MICA).         │
│  Produces: Ready-to-use UE5 FBX (ARKit-52 + LOD0-3) + 7 PBR Texture Maps (2048²) │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 4. Exact Specifications for Training Runs & Production Batch

### Deep Run #1: Phase 3 Deep — Studio-Grade Skin Pore GAN Training
* **When to run:** Currently running on Kaggle GPU (`nightshowdown/phase-3-deep-detail-gan-train`).
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

### Deep Run #2: Delighting U-Net Fine-Tune (CelebA-HQ 1024²)
* **When to run:** Once Phase 3 Deep finishes, train the texture delighting network to peak quality.
* **Target Dataset:** 30,000 CelebA-HQ images projected onto FLAME UV space.
* **Estimated Runtime:** $\approx \mathbf{4 \text{ hours}}$ on Dual T4 GPUs.
* **Architecture:** 6-level Encoder-Decoder U-Net with skip connections (`src/stage7_delight/model.py`).
* **Loss Function:** $\mathcal{L}_{\text{delight}} = \|\hat{A} - A_{\text{gt}}\|_1 + \lambda_{\text{perc}} \mathcal{L}_{\text{VGG}}(\hat{A}, A_{\text{gt}}) + \lambda_{\text{chroma}} \mathcal{L}_{\text{chroma}}(\hat{A})$
* **Output Deliverable:** `models_cache/stage7_delight/delight_unet.pt` providing studio-grade diffuse albedo with zero baked-in shadows or specular hot-spots.

---

### Deep Run #3: Phase 2 Deep — Demographic MICA Fine-Tuning (Optional)
* **When to run:** When registered 3D scan databases are acquired.
* **Prerequisites:** ≥ 100 3D scan meshes registered with FLAME shape vectors $\beta_{\text{gt}}$.
* **Estimated Runtime:** $\approx \mathbf{4 \text{ to } 6 \text{ hours}}$ on Dual T4 GPUs.
* **Loss Function:**
  $$\mathcal{L}_{\text{MICA}} = \|\beta_{\text{pred}} - \beta_{\text{gt}}\|_1 + \lambda_{\text{norm}} \big|\|\beta_{\text{pred}}\| - \|\beta_{\text{gt}}\|\big| + \lambda_{\text{cos}} (1 - \cos(\beta_{\text{pred}}, \beta_{\text{gt}})) + \lambda_{\text{sil}} \mathcal{L}_{\text{diff\_render}}$$
* **Output Deliverable:** `models_cache/mica/mica_finetuned.pt` fine-tuned for high identity accuracy across diverse demographics.

---

### Production Run: Final End-to-End Batch Generation
* **When to run:** Once models are trained to peak quality.
* **Target Subjects:** `carell`, `connelly`, `justin`, `lawrence` (and user portraits).
* **Presets per subject:** Neutral, Chiseled, Heroic, Gigachad.
* **Estimated Runtime:** $\approx \mathbf{15 \text{ minutes}}$ on Kaggle GPU.
* **Output Deliverables per Subject:**
  1. `head_mesh_ue5_livelink.fbx`: Binary FBX with 5-joint skeleton, LBS skin weights, 52 ARKit blendshapes, and wired PBR material slots.
  2. `face_lod0.obj` through `face_lod3.obj`: 4-tier quadric decimation chain with preserved shape keys.
  3. `textures/albedo_diffuse.png`: 2048² film-grade delighted diffuse albedo texture.
  4. `textures/roughness_map.png`: 2048² micro-roughness map (anatomical zones + pore roughness).
  5. `textures/cavity_ao_map.png`: 2048² pore-coupled ambient occlusion / cavity map.
  6. `textures/sss_thickness_map.png`: 2048² subsurface scattering thickness map (ears, nose, lips).
  7. `head_displacement_16bit.png`: 1024² 16-bit signed displacement texture (sub-mm skin pores).
  8. `head_normal_map.png`: 1024² tangent-space normal map formatted for UE5 shaders.
  9. Procedural facial hair & beard stubble geometry cards.

---

## 📌 5. Summary Checkpoint

1. **Software Pipeline:** 100% complete across all 9 stages (Stages 0–8 + Stage 5 Export). Verified with **153 passing unit tests**.
2. **Current Active Job:** **Phase 3 Deep (40,000 steps)** skin pore GAN is running on Kaggle Dual Tesla T4 GPUs.
3. **Execution Strategy:** Complete all model training runs (Pore GAN → Delighting U-Net → MICA if scans available) **before** generating the final production batch assets, ensuring the deliverables utilize the absolute best weights across geometry, identity, and PBR textures.
