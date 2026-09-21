# Humanoid-Face-3D — Execution Guide

> **Current Status:** All pipeline code is 100% complete (Stages 0–8 + Stage 5 export). The next steps are GPU training runs and production batch processing.

---

## What's Already Built & Operational

| Stage | Status | Description |
|-------|--------|-------------|
| Stage 0 — Preprocessing | ✅ Done | InsightFace detection, alignment, pose estimation |
| Stage 1 — Identity (MICA) | ✅ Done | Multi-view shape regression → 300-D FLAME β |
| Stage 1.5 — Macro Residuals | ✅ Done | Graph CNN vertex correction |
| Stage 2 — Expression (SMIRK) | ✅ Done | Canonical neutral normalisation (ψ=0, θ=0) |
| Stage 3 — Detail GAN | ✅ Done | 1024² displacement + normal maps |
| Stage 4 — Facial Hair | ✅ Done | Procedural stubble + 3D hair cards |
| Stage 6 — UV Texture Projection | ✅ Done | Multi-view photo → UV backprojection (pure math) |
| Stage 7 — Delighting + Inpainting | ✅ Done | AI lighting removal + UV hole filling |
| Stage 8 — PBR Material Stack | ✅ Done | Roughness, Cavity/AO, SSS thickness (pure math) |
| Stage 5 — Production Export | ✅ Done | ARKit-52, LODs, skeleton, FBX, PBR material slots |

**Total tests:** 153 passing.

---

## Execution Sequence — What To Run Next

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ✅ STEP 1: SOFTWARE PIPELINE (COMPLETE)                                     │
│  All stages 0–8 + Stage 5 export are built, tested, and pushed to GitHub.   │
│  Commit: 170b480 on main                                                     │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  🔄 STEP 2: PHASE 3 DEEP — Skin Pore GAN Training (GEOMETRY)                │
│  Currently RUNNING on Kaggle (nightshowdown/phase-3-deep-detail-gan-train)  │
│  • 40,000 steps, 1024×1024, Dual Tesla T4, ~9.5 hours                       │
│  • Resumes from checkpoint_latest.pt                                         │
│  • Unlocks: Sub-millimeter skin pores, sebaceous bumps, epidermal grain     │
│  • Output: ema_generator.pt → models_cache/stage3_detail/                    │
│  • Command:                                                                  │
│    torchrun --nproc_per_node=2 src/stage3_detail/trainer.py \                │
│      --data_dir <uv_dataset_1024> --total_steps 40000 --batch_size 4 \      │
│      --resolution 1024 --resume checkpoints/stage3_detail/checkpoint_latest  │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⏹ STEP 3: DELIGHTING FINE-TUNE — Albedo Training (TEXTURE)                 │
│  Train the Stage 7 DelightUNet so the texture quality is film-grade.        │
│  • Fine-tune on 30,000 CelebA-HQ 1024² images projected onto FLAME UV      │
│  • Estimated: ~4 hours on Kaggle Dual T4                                     │
│  • Improves: Albedo quality from "good (DECA)" → "great (custom-tuned)"    │
│  • Output: delight_unet.pt → models_cache/stage7_delight/                    │
│  • Why before production batch: So the final textured output uses the        │
│    best possible lighting-removed skin texture, not the generic fallback.   │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼ (Data-Dependent)
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⏹ STEP 4: PHASE 2 DEEP — MICA Identity Fine-Tuning (SHAPE)                │
│  Supervised fine-tune of MICA on registered 3D scan datasets so the          │
│  base head shape is as accurate as possible before final production run.    │
│  • Requires: ≥100 registered 3D scans with ground-truth β (FaceScape/LYHM) │
│  • Estimated: 4–6 hours on Kaggle Dual T4                                    │
│  • Improves: Identity accuracy for diverse demographics (jaw, cheeks, nose) │
│  • Output: mica_finetuned.pt → models_cache/mica/                            │
│  • Skip if: External scan datasets are not yet acquired.                     │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⏹ STEP 5: FINAL PRODUCTION BATCH RUN (Best Models → Best Output)           │
│  Run ALL subjects through the fully trained pipeline with every model       │
│  at peak quality — pore-level geometry + film-grade textures + accurate     │
│  identity shape. This is the final deliverable generation step.             │
│  • Subjects: carell, connelly, justin, lawrence + user portraits            │
│  • Uses: Best Phase 3 GAN + Best Delighting + Best MICA (all trained)       │
│  • Produces per subject:                                                     │
│    - head_mesh_ue5_livelink.fbx (skeleton + ARKit-52 + LODs)                │
│    - textures/albedo_diffuse.png (2048² film-grade delighted skin)           │
│    - textures/roughness_map.png (2048² anatomical zones)                     │
│    - textures/cavity_ao_map.png (2048² pore-coupled AO)                     │
│    - textures/sss_thickness_map.png (2048² SSS)                              │
│    - head_displacement_16bit.png (1024² sub-mm skin pores)                   │
│    - head_normal_map.png (1024² tangent normals)                             │
│  • Estimated: ~15 minutes on Kaggle GPU                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Running Inference Locally

```bash
# From the project root
cd "/Users/pranav/Project Folder/3d Model"

# Run on a single portrait (minimum)
python3 -c "
from src.pipeline import FaceGeoPipeline
pipe = FaceGeoPipeline(config_path='configs/default.yaml')
result = pipe.run(['path/to/photo.jpg'], 'outputs/my_head')
print(result)
"
```

**Output:** `outputs/my_head/` containing the complete textured head asset bundle.

**VRAM requirement:** ~3.5 GB peak (runs on Apple Silicon 8 GB Macs).

---

## Key Files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | All pipeline configuration (stages 0–8) |
| `src/pipeline.py` | Main pipeline orchestrator |
| `AGENTS.md` | Agent guidelines & system invariants |
| `remember.md` | Master memory & training blueprint |
| `roadmap.md` | Full architectural roadmap |
| `DOCS/texture_engine.md` | PBR texture engine technical reference |
| `DATASETS.md` | Data procurement registry |