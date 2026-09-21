# Antigravity Agent Guidelines: Humanoid-Face-3D

This document guides any AI agent resuming or collaborating on the **Humanoid-Face-3D** repository.

---

## 1. Project Objective & Architecture

A production-grade, identity-preserving image-to-3D **fully textured, PBR-ready humanoid head mesh** reconstruction pipeline that synthesizes a film-grade 3D facial asset from 3–5 multi-view portraits, complete with physically based rendering material maps (albedo, roughness, SSS, cavity/AO, displacement, normal).

### Pipeline Stages:
* **Stage 0 (Preprocessing):** InsightFace detection, 5-point alignment, pose estimation, and pre-flight identity/angular validation (`src/stage0_preprocess/`).
* **Stage 1 (Identity Regression):** ArcFace feature extraction across views, frontality-weighted embedding fusion ($w_i = \text{det\_score}_i \cdot \cos^2(\text{yaw}_i)$), regressing 300-D FLAME shape coefficients $\beta$ (`src/stage1_identity/`).
* **Stage 2 (Expression & Pose):** SMIRK regression of 100-D expression ($\psi$) and 15-D pose ($\theta$). **Invariant:** Base mesh is kept in canonical neutral pose ($\psi=0, \theta=0$); expressions are saved as metadata for blendshape targets (`src/stage2_expression/`).
* **Stage 3 (Micro-Detail GAN):** U-Net Generator with InstanceNorm and MultiView cross-attention + PatchGAN Discriminator with SpectralNorm. Synthesizes 1024×1024 signed 16-bit displacement maps (`src/stage3_detail/`).
* **Stage 4 (Facial Hair):** Static facial hair geometry and micro-displacement (`src/stage4_facial_hair/`).
* **Stage 6 (UV Texture Projection):** Multi-view backprojection of input photos onto FLAME UV space with angle-weighted cosine blending and z-buffer visibility testing. Pure math, no GPU (`src/stage6_texture/`).
* **Stage 7 (AI Delighting + Inpainting):** Encoder-decoder U-Net strips environment lighting from projected textures → clean diffuse albedo. Procedural Gaussian dilation fills unseen UV regions. Supports pre-trained DECA albedo decoder weights (`src/stage7_delight/`).
* **Stage 8 (PBR Material Stack):** Procedural generation of roughness (anatomical zone-based), cavity/AO (displacement Laplacian), and SSS thickness (opposing-normal ray-march). Pure math, no GPU (`src/stage8_pbr/`).
* **Stage 5 (Production Retopology & UE5 Rig):** Sparse barycentric correspondence matrix $W$, ARKit-52 blendshapes with neck boundary pinning ($\Delta v = 0$), 4-tier LOD decimation (LOD0 to LOD3), 5-joint skeletal armature, PBR material slot wiring, and headless Blender FBX packaging (`src/stage5_export/`).

---

## 2. Critical System Invariants & Rules

1. **Zero Silent Fallbacks:**  
   NEVER revert silently to mean shapes, fake random arrays, or synthetic defaults when weights or dependencies are missing. Always raise explicit errors (`FileNotFoundError`, `RuntimeError`) with actionable download/setup instructions.
2. **Kaggle 19.5 GB Disk Quota & File Inode Limit:**  
   `/kaggle/working` has a hard 19.5GB / 500-file cap. Never accumulate unbounded step checkpoints. Maintain only `checkpoint_latest.pt`, `ema_generator.pt`, and a sliding window of the last 2 step checkpoints. Route Blender extraction and scratch data to `/tmp/` (~50GB uncounted headroom).
3. **Emergency Checkpoint & Resumability:**  
   Kaggle hard-kills sessions at 12 hours. Always trigger an emergency checkpoint at 11.5 hours. All training loops must support resuming via `--resume <path/to/checkpoint_latest.pt>`.
4. **Neck Seam Contract:**  
   In Stage 5, the lowest 20% of vertices (neck boundary collar) must have their delta displacements pinned strictly to zero (`masks['neck_pinning']`) so the exported head never tears when attached to a common torso in Unreal Engine 5.
5. **Dynamic GPU Topology:**  
   Do not hardcode `--nproc_per_node=2`. Always detect `torch.cuda.device_count()` to gracefully handle single-GPU (P100) or multi-GPU (T4×2) allocations.
6. **Texture Pipeline Graceful Degradation:**  
   Stages 6, 7, 8 are config-gated (`stage6.enabled`, etc.). If any texture stage fails or lacks weights, the pipeline must still produce valid untextured geometry. The texture engine NEVER blocks geometry export.

---

## 3. Key Repositories & Credentials

* **GitHub Repository:** [https://github.com/NetPranav/Humanoid-Face-3D](https://github.com/NetPranav/Humanoid-Face-3D)
* **Branch:** `main`
* **Kaggle CLI Path:** `/Users/pranav/.local/bin/kaggle`
* **Local Test Suite:** `python3 -m unittest discover tests` (102 tests passing).
