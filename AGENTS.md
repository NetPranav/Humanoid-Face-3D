# Antigravity Agent Guidelines: Humanoid-Face-3D

This document guides any AI agent resuming or collaborating on the **Humanoid-Face-3D** repository.

---

## 1. Project Objective & Architecture

A production-grade, identity-preserving image-to-3D **fully textured, PBR-ready humanoid head mesh** reconstruction pipeline that synthesizes a film-grade 3D facial asset from 3–5 multi-view portraits, complete with physically based rendering material maps (albedo, roughness, SSS, cavity/AO, displacement, normal).

### Pipeline Stages:
> **Architecture v3 is the plan of record: read `DOCS/README.md` first.** Phase 0 of `DOCS/04_roadmap.md` is implemented; old design docs live in `older/`.

* **Stage 0 (Preprocessing):** InsightFace detection + 68 landmarks, EXIF intrinsics (`camera.py`), MediaPipe skin parsing (`parsing.py`), validation (`src/stage0_preprocess/`).
* **Stage 1 (Identity Regression):** MICA with **MICA's own ArcFace** (never InsightFace embeddings), frontality-weighted feature fusion, 300-D FLAME β (`src/stage1_identity/inference.py`). FLAME is rebuilt from the MICA checkpoint by `scripts/extract_flame_from_mica.py`.
* **Stage 2 (Camera, Pose & Expression):** per-view landmark fit of camera, head pose, ψ and jaw with EXIF focal (`src/stage2_expression/landmark_fit.py`). **Invariant:** the exported base mesh stays neutral (ψ=0, θ=0); the fitted posed mesh is used only for texture projection.
* **Stage 3 (Sculpt detail, Phase 4A):** `stage3.detail_level` 0–100 drives photo-derived wrinkle grooves (Hessian crease detection on the delit texture) + synthesized pores/micro-grooves/lip striations, 4K maps and a subdivided `head_mesh_detail.obj` (`src/stage3_detail/sculpt_detail.py`). The Multiface GAN and v1 luminance→height stay **off** (DOCS/01 R5).
* **Stage 4 (Facial Hair):** Static facial hair geometry and micro-displacement (`src/stage4_facial_hair/`).
* **Stage 6 (UV Texture Projection):** per-texel backprojection onto the fitted posed mesh with a real z-buffer (`src/render/soft_raster.py`) and skin-only parsing masks (`src/stage6_texture/`).
* **Stage 6/7 (Texture, Phase 3A):** fitted-SH delighting (`stage7.delight_strength`), mouth-interior exclusion, seam-free mesh-harmonic colour fill, lip-only lip fill, and the subject's own skin grain quilted into unseen areas; provenance map classes observed / mirrored / synthesized (`src/stage7_delight/`).
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

7. **Commercial use required (2026-09-25):**
   Never train or ship on non-commercial data or models: FaceScape, FFHQ(-UV), NPHM, NeRSemble, Multiface, MetaHuman renders, InsightFace model weights, MICA, FLAME 2020. `DOCS/07_commercial_licensing.md` lists the allowed sources and the switch plan (FLAME 2023 Open, MediaPipe).

## 3. Key Repositories & Credentials

* **GitHub Repository:** [https://github.com/NetPranav/Humanoid-Face-3D](https://github.com/NetPranav/Humanoid-Face-3D)
* **Branch:** `main`
* **Kaggle CLI Path:** `/Users/pranav/.local/bin/kaggle`. Auth: `KAGGLE_API_TOKEN` in the gitignored `.env` (`set -a; . ./.env; set +a`). Never print or commit it.
* **Local Test Suite:** `python3 -m unittest discover tests` (129 tests, 2 skipped when weights are absent).
* **Quality gate:** `python3 scripts/run_golden_set.py --baseline outputs/golden_v1_baseline`. A change that lowers the golden-set identity score is a regression even if unit tests pass.
