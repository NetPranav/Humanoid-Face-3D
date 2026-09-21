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
│  Commit: 1d1fd7a on main                                                     │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  🔄 STEP 2: PHASE 3 DEEP — Overnight Skin Pore GAN Training                 │
│  Currently RUNNING on Kaggle (nightshowdown/phase-3-deep-detail-gan-train)  │
│  • 40,000 steps, 1024×1024, Dual Tesla T4, ~9.5 hours                       │
│  • Resumes from checkpoint_latest.pt                                         │
│  • Unlocks: Sub-millimeter skin pores, sebaceous bumps, epidermal grain     │
│  • Command:                                                                  │
│    torchrun --nproc_per_node=2 src/stage3_detail/trainer.py \                │
│      --data_dir <uv_dataset_1024> --total_steps 40000 --batch_size 4 \      │
│      --resolution 1024 --resume checkpoints/stage3_detail/checkpoint_latest  │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⏹ STEP 3: CLOUD BATCH PRODUCTION RUN (Full Textured Pipeline)              │
│  Once Phase 3 Deep finishes, run all subjects through the FULL pipeline:    │
│  • Subjects: carell, connelly, justin, lawrence + user portraits            │
│  • Produces per subject:                                                     │
│    - head_mesh_ue5_livelink.fbx (skeleton + ARKit-52 + LODs)                │
│    - textures/albedo_diffuse.png (2048² delighted skin)                      │
│    - textures/roughness_map.png (2048² anatomical zones)                     │
│    - textures/cavity_ao_map.png (2048² pore-coupled AO)                     │
│    - textures/sss_thickness_map.png (2048² SSS)                              │
│    - head_displacement_16bit.png (1024² skin pores)                          │
│    - head_normal_map.png (1024² tangent normals)                             │
│  • Estimated: ~15 minutes on Kaggle GPU                                      │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⏹ STEP 4: DELIGHTING FINE-TUNE (Optional — 4 hours Kaggle)                 │
│  Fine-tune the Stage 7 DelightUNet on CelebA-HQ 1024² images to improve    │
│  albedo quality from "good (DECA pre-trained)" to "great (custom-tuned)".   │
│  • Dataset: 30,000 CelebA-HQ images projected onto FLAME UV                 │
│  • Estimated: ~4 hours on Kaggle Dual T4                                     │
│  • Skip this if DECA albedo quality is already sufficient.                   │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼ (Data-Dependent)
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⏹ STEP 5: PHASE 2 DEEP — MICA Identity Supervised Fine-Tuning             │
│  Requires attaching external registered 3D scan datasets (FaceScape/LYHM).  │
│  • Prerequisites: ≥100 registered 3D scan meshes with ground-truth β        │
│  • Estimated: 4–6 hours on Kaggle Dual T4                                    │
│  • Improves: Identity accuracy for diverse demographics                      │
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