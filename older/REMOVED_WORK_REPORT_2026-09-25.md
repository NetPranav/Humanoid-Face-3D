# Removed Work Report: v1 Outputs, Datasets and Dead Code

> **Date:** 2026-09-25 · **Repository state before removal:** commit `23e6fe6` (plus the uncommitted Phase 0 / 3A / 4A work)
> **Why this cleanup happened:**
> 1. Disk space (the laptop was at 1.6–7 GB free).
> 2. The switch to a **commercially usable** stack (see `DOCS/07_commercial_licensing.md`).
> 3. Removing code paths the current pipeline can no longer reach.
>
> This report is the permanent record. Small visual traces of the removed results are in [`removed_traces/`](removed_traces/).

---

## 1. How to get anything back

| Kind | Recovery |
|---|---|
| Items that were **tracked in git** (marked **G** below) | `git checkout 23e6fe6 -- <path>` restores them exactly. |
| Kaggle run outputs (marked **K**) | The kernels still exist on Kaggle under `nightshowdown/…`, so `kaggle kernels output nightshowdown/<kernel> -p <dir>` re-downloads them (needs `KAGGLE_API_TOKEN` from `.env`). |
| Local, untracked outputs (marked **L**) | Not recoverable byte-for-byte. The regeneration command is listed; most need the v1 code (restore it with git first). |
| This session's golden-set baselines (marked **S**) | Regenerate with `scripts/run_golden_set.py`; the v1 baseline needs the v1 code snapshot (`git archive 23e6fe6 src scripts evaluation configs`). |

## 2. Findings worth remembering (from the removed material)

These are the lessons the deleted files proved. Keep them even though the files are gone.

1. **The v1 GAN never learned adversarially.**
   - In the pilot run (`kaggle_phase3_gan`, 1,500 steps), discriminator loss sat at exactly **1.3863 = ln 4** and adversarial loss at **0.6931 = ln 2** from start to finish. The discriminator was at chance the whole time; only the L1 reconstruction term did anything.
   - The deep run (`phase-3-deep-detail-gan-train`, 2×T4) reached **step 32,291 after ~11.7 h**, then ended in a `CalledProcessError` at the Kaggle session limit.
   - Reported L1: pilot 0.1110 → deep 0.0826 (−26 %), "sharpness" 0.0000. Trace: `removed_traces/gan_pilot1500_vs_deep32291.jpg`.
2. **The ground truth the GAN learned from was registration error, not skin detail.** The Multiface-derived "displacement" targets contain saturated rims, a mouth hole and bald-cap edges. Traces: `multiface_disp_dataset_grid.jpg`, `multiface_reextract_verify_320k.jpg`. See also `older/FAILED/RESEARCH_1/`.
3. **The v1 production batch (4 subjects) was broken.**
   - Two of four heads (carell, justin) render with exploded geometry: large stray triangles.
   - All four subjects got an **identical** normal map, so the detail network ignored its input.
   - Trace: `v1_production_batch_showcase.jpg`.
4. **v1 identity collapse was visible early.** The Phase 1 baseline neutral meshes for four different people are nearly indistinguishable (`v1_phase1_baseline_grid.jpg`). The cause, diagnosed on 2026-09-25, was that MICA was fed InsightFace embeddings instead of its own ArcFace features (DOCS/01 R1).
5. **The Elon v1 → v2 → v3 progression** (`elon_v1_film_render.jpg`, `elon_v2_film_render.jpg`, `elon_v3_film_render.jpg`, `elon_v3_hero45.jpg`, `elon_v3_displacement.jpg`, `elon_v3_projected_texture.jpg`) is the evidence behind `DOCS/01_diagnosis_elon_v3.md`: ghost second eye, embossed displacement, and background and hair projected onto the head.
6. **Golden-set v1 baseline (this session):** unmodified v1 code on 9 public-domain portraits. Mean identity cosine (buffalo_l judge): frontal **0.383**, +30° **0.323**, −30° **0.361**. Phase 0 raised these to 0.782 / 0.681 / 0.712. Trace: `golden_v1_baseline_contact_sheet.jpg`. The full numbers are in `outputs/golden_phase0/_summary/summary.md`, which is kept.

## 3. Outputs removed

| Path | Size | Kind | What it was | Why removed |
|---|---|---|---|---|
| `outputs/phase1_baseline/` | 532 MB | L | v1 Phase 1 inference on 4 celebrity photos (carell, connelly, justin, lawrence): meshes, previews, stylization tests | Superseded; showed the identity collapse (finding 4); non-public-domain inputs |
| `outputs/upgraded_inference/` | 532 MB | L/K | v1 "upgraded inference" (Stage 1.5 + Pixel3DMM stub) on the same 4 subjects | Superseded; Stage 1.5 deformer removed |
| `outputs/kaggle_upgraded_run/` | 603 MB | K | Kaggle run of the above, incl. `upgraded_inference_assets.tar.gz` and repo snapshot | Same |
| `outputs/kaggle_phase2_5_scaled/` | 33 MB | K | Phase 2.5 displacement dataset build (p99 = 1.146 mm, N = 20) | GAN data pipeline removed |
| `outputs/kaggle_phase3_error/` | 50 MB | K | Failed GAN training attempt (`CalledProcessError` at start) | GAN removed |
| `outputs/kaggle_phase3_gan/` | 51 MB | K (3 files G) | Pilot GAN, 1,500 steps (finding 1) | GAN removed |
| `outputs/phase-3-deep-detail-gan-train/` | 51 MB | K | Deep GAN, 32,291 steps (finding 1) | GAN removed |
| `outputs/phase-production-cloud-batch-ue5/` | 51 MB | K | Cloud production batch repo snapshot (contained a 0-byte `pretrained.tar`) | Broken, superseded |
| `outputs/production_batch/` + `outputs_production_batch.tar.gz` + `production_batch_showcase.png` | 201 MB | G / L | v1 "game-ready production batch" (finding 3) | Broken output of v1 |
| `outputs/online_test/elon`, `elon_v2`, `elon_v3` | 330 MB | G | v1 Elon runs (finding 5) | Superseded by `outputs/golden_phase3a4a/elon` |
| `outputs/comparisons/` | 65 MB | G | GAN pilot-vs-deep comparisons, 320k-subdivision progressions | GAN removed |
| `outputs/uv_displacement_dataset_1024/`, `real_scan_displacement_dataset_1024/`, `dataset-re-extract-subdiv2-c2-smoothing/`, `subdivided_320k/` | 47 MB | L (2 files G) | Multiface-derived displacement datasets and verification renders (finding 2) | Non-commercial source data, dead approach |
| `outputs/scratch_dense_test/` | 1.5 MB | G | Dense ray-cast / micro-displacement experiments | Superseded by Phase 4A |
| `outputs/focal_mask_overlay.png`, `model_comparison_1500_vs_32291.png` | 4 MB | G | Stage 1 focal mask viz; GAN comparison (kept as traces) | Superseded |
| `outputs/golden_v1_baseline/` | 1.0 GB | S | This session's run of unmodified v1 code on the 9-subject golden set (finding 6) | Numbers recorded; regenerable |
| `outputs/golden_phase0/*/` production files | ~0.5 GB | S | FBX, 15 MB blendshape JSONs, LODs, armature and 1K PBR maps per subject | Heuristic v1 rig assets (replaced in Phase 2). **Kept:** meshes, textures, reports, eval images, `_summary/` |
| `outputs/golden_phase3a4a/*/` production files | ~0.6 GB | S | Same kind of files for the current run | Same. **Kept:** `head_mesh.obj`, `head_mesh_detail.obj`, textures, 4K displacement/normal maps, eval, reports |

## 4. Data removed

| Path | Size | Kind | What it was | Why removed |
|---|---|---|---|---|
| `data/external/3d_scans/multiface/` | 423 MB | L | Meta Multiface tracked meshes, 5 subjects (E001 neutral) | **CC-BY-NC** (not commercial); dead approach (finding 2) |
| `data/external/multiface_test/` | 26 MB | L | One Multiface test tarball | Same |
| `data/real_scan_displacement_dataset_1024/`, `real_scan_test_single/`, `real_scan_test_multi/` | 128 MB | L | GAN training/validation maps built from Multiface (585 files; `num_samples: 80`, p99 2.24 mm) | Same |
| `data/external/3d_scans/hsrd/` | 36 MB | G | One low-res full-body scan (HSR0015-Body-009, LOD2) used by the GAN dataset builder | Unused; unclear licence |
| `models_cache/stage3_detail/`, empty `models_cache/{emoca,smirk,flame}/`, empty `checkpoints/` | <1 MB | L | GAN normalization stats and empty placeholders | GAN removed |
| `scratch/` | 3 MB | L | Projector experiments on Elon (old vs per-texel, inpainting) | Superseded by Phase 0 projector |

The non-commercial dataset request templates (`data/licenses/*`) and the v1 root documents were **moved** to `older/`, not deleted.

## 5. Code removed (all tracked: `git checkout 23e6fe6 -- <path>`)

| Code | Introduced | What it did | Why removed |
|---|---|---|---|
| `src/stage3_detail/{generator,discriminator,trainer,data,losses,inference}.py` | `76cdfe7`, `90db9a9` (2026-09-18) | U-Net + PatchGAN detail GAN, DDP trainer, dataset, inference | Off since Phase 0; replaced by Phase 4A sculpt detail; its data was non-commercial |
| `src/stage3_detail/photometric_detail.py`, `src/utils/spectral_check.py` | `6292816` (2026-09-22) | "Tier 2" luminance → height meso wrinkles | Embossed features (DOCS/01 R5); replaced by Hessian crease detection |
| `src/stage3_detail/anatomical_pores.py` + pipeline `stage3.mode: legacy` | `7539643` (2026-09-22) | "Tier 3" bbox-zone procedural pores | Replaced by `sculpt_detail.py` (FLAME-region masks, mm-scaled) |
| `src/stage1_5_residual/` (contour deformer, residual net, trainer, legacy 5-pt camera) | `e93af2d` (2026-09-23) | ±18 mm landmark Laplacian deformation, neural residual | Disabled in Phase 0 (DOCS/01 R3); returns as Δ in the Phase 1 fitter. `load_flame_landmark_matrix` moved to `src/utils/flame_landmarks.py` |
| `src/stage1_identity/{trainer,data,diff_render,pixel3dmm_fitter}.py` | `90db9a9`, `d28030f` | MICA fine-tuning trainer, loader, differentiable render stub, Pixel3DMM stub | Never ran (blocked on licensed data); MICA itself is non-commercial |
| `src/stage2_expression/encoder.py` | early Phase 0 | SMIRK wrapper that silently returned ψ = θ = 0 | Replaced by `landmark_fit.py` |
| `src/stage7_delight/delight_net.py` | `1d1fd7a` (2026-09-21) | DelightUNet (no weights ever existed) + procedural "dichromatic" delight and hardcoded-colour inpainting | Replaced by fitted-SH delighting + `uv_fill.py` / `skin_synthesis.py` |
| `evaluation/identity_score.py` | `76cdfe7` | v1 preview-image identity score | Replaced by `evaluation/phase0_eval.py` |
| `src_v2_generative/` | `ea672fa` (2026-09-24) | Loaders for FaceScape / ICT / THuman (Hunyuan3D/TRELLIS direction) | Non-commercial datasets; direction parked (DOCS/03 §4) |
| `vendor/multiface/` (gitlink) | `5c0cbfe` | Meta Multiface code | CC-BY-NC; dataset removed |
| `notebooks/kaggle/*` (16 files) | `06851aa` (2026-09-18) onward | v1 Kaggle notebooks (phase 1, 2, 2.5, 3, production batch, re-extract) | Drive the removed v1 stages; new training notebooks come with Phase 3B/4B |
| `scripts/` (24 files): GAN/Multiface dataset builders, `compare_gan_models`, `render_real_*`, `render_3d_*`, `generate_*_nb`, `download_community_data`, `beta_collapse_smoke_test`, `extract_arcface_features`, `calibrate_arcface_threshold`, `production_inference`, `run_production_pipeline`, `visualize_focal_mask`, `upload_to_kaggle_models`, `live_logger`, `fetch_models`, `configs/arcface_calibration.json` | 2026-09-18 → 22 | Tooling for the stages above | Their targets are gone; the current entry points are `scripts/run_golden_set.py` and `src/pipeline.py` |
| Tests for all of the above (14 files) | — | Unit tests of removed modules | Removed with their modules; the rest of the suite still passes |
| `Humanoid-Face-3D/` | untracked | A stray `.git` directory without a working tree (same HEAD as this repo) | Accidental clone |

## 6. What was deliberately kept

- `vendor/MICA`, `models_cache/mica/mica.tar` and the FLAME 2020 files rebuilt from it. The current pipeline still needs them for research use. The commercial path replaces them (see `DOCS/07`).
- `src/stage3_detail/{rasterizer,fusion}.py`: UV layout loading, and the roughness/cavity maps used by Stage 8 and Blender.
- `older/`: every previous design document, postmortems, the capture protocol and the talent release form.
- `outputs/golden_phase0/_summary` and `outputs/golden_phase3a4a/_summary`: all metrics and contact sheets.
