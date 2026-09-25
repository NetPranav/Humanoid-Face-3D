# Face Reconstruction Models — Research Reference

> This document is a standalone reference for every model considered or used in the face geometry pipeline. Read this to understand what each model actually does, how it works internally, where it fits in the pipeline, and what its real limitations are.
>
> ⚠️ **Historical & Architectural Updates:**
> - For the postmortem on the failed Meta Multiface Detail GAN experiment, see [DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md).
> - For the active film-grade MetaHuman synthesis architecture (Photo-derived meso wrinkles + 4K anatomical pore synthesis + Cycles Random Walk SSS), see [DOCS/02_Metahuman_Film_Grade_Synthesis.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/02_Metahuman_Film_Grade_Synthesis.md).

---

## Table of Contents

1. [FLAME — The Parametric Foundation](#1-flame--the-parametric-foundation)
2. [MICA — Metrical Identity from a Single Image](#2-mica--metrical-identity-from-a-single-image)
3. [DECA / EMOCA — Expression + Coarse Detail](#3-deca--emoca--expression--coarse-detail)
4. [SMIRK — Analysis-by-Synthesis Expression Regression](#4-smirk--analysis-by-synthesis-expression-regression)
5. [Pixel3DMM — Per-Pixel Geometric Cues (ICLR 2026)](#5-pixel3dmm--per-pixel-geometric-cues-iclr-2026)
6. [NPHM / MonoNPHM / Pix2NPHM — Beyond Linear PCA](#6-nphm--mononphm--pix2nphm--beyond-linear-pca)
7. [Supporting Tools — InsightFace, MediaPipe, ArcFace](#7-supporting-tools--insightface-mediapipe-arcface)
8. [General-Purpose 3D Generators — Why They Don't Apply Here](#8-general-purpose-3d-generators--why-they-dont-apply-here)

---

## 1. FLAME — The Parametric Foundation

**Full name:** Faces Learned with an Articulated Model and Expressions  
**Paper:** Li et al., SIGGRAPH Asia 2017  
**Repository:** [FLAME-Universe](https://github.com/TimoBolkart/FLAME-Universe)  
**License:** Research license, free for most uses — check terms for commercial products

### What it is

FLAME is not a neural network — it's a **linear parametric face model** (a 3D Morphable Model / 3DMM), analogous to SMPL for bodies. It defines a fixed-topology mesh (5,023 vertices, 9,976 faces) whose shape is controlled by three sets of low-dimensional parameters:

| Parameter | Dimension | Controls |
|---|---|---|
| **Shape (β)** | 300 | Identity — bone structure, face width, nose size, jaw shape. PCA coefficients learned from ~3,800 registered 3D scans. |
| **Expression (ψ)** | 100 | Expression — smile, frown, raised eyebrows. PCA coefficients, orthogonal to identity. |
| **Pose (θ)** | 15 (5 joints × 3 DoF) | Jaw, neck, two eyeballs — articulated rotation via linear blend skinning (LBS). |

### How the decode works

Given `(β, ψ, θ)`, FLAME computes the output mesh as:

```
M(β, ψ, θ) = LBS(T_mean + B_S · β + B_E · ψ + B_P(θ), J(β), θ, W)
```

Where:
- `T_mean` = template mesh (average face), 5023 × 3
- `B_S` = shape blend shapes (PCA basis), 5023×3 × 300
- `B_E` = expression blend shapes, 5023×3 × 100
- `B_P(θ)` = pose-corrective blend shapes (correct LBS artifacts)
- `J(β)` = joint locations (depend on identity shape)
- `W` = skinning weights (fixed)

This is a **linear model** — it can only represent faces that lie within the span of its PCA basis. That's its fundamental ceiling: if a face shape requires a deformation that's not in the training scan distribution, FLAME literally cannot represent it, regardless of what β values you feed it.

### What it's good at

- **Interoperability.** FLAME is the de facto interchange format for face research — MICA, DECA, EMOCA, SMIRK, Pixel3DMM all output FLAME parameters.
- **Fixed topology.** The 5,023-vertex mesh always has the same connectivity, vertex ordering, and UV layout. This makes retopology to ICT-FaceKit / MetaHuman topology a one-time, deterministic mapping.
- **Compact.** The entire model file is ~300MB. Decoding β+ψ+θ into a mesh is a single matrix multiply — microseconds, no GPU needed.

### What it cannot do

- **Fine detail.** 300 shape components ≈ "overall face proportions." Cannot represent wrinkles, pores, or any high-frequency geometry. This is why Stage 3 (detail GAN) adds displacement on top.
- **Ears, back of head.** The 5,023-vertex mesh covers the face and a small amount of neck/ear. Not a full head model.
- **Hair, facial hair.** FLAME has no representation of hair or beards — these must be handled as separate geometry.

### How it's loaded in code — DAY 0 TASK

> [!CAUTION]
> **This is the single highest-priority task on Day 0 — resolve it before writing any downstream code.** Every stage in the pipeline depends on FLAME loading correctly. If this breaks, nothing else can even start.

FLAME ships as a `.pkl` file containing numpy arrays for the PCA bases, template mesh, skinning weights, etc. The legacy loader uses `chumpy` (a dead autodiff library), which breaks on modern numpy (`np.bool`/`np.int` removed). **Use a pure-numpy loader** — MICA's codebase has one, and there are standalone forks. The `.pkl` is just numpy arrays wrapped in chumpy objects; you can unpickle them with a custom unpickler that substitutes numpy.

**Version-specific verification (Day 0 checklist):**
- The **generic model** (`generic_model.pkl`) and the **FLAME2020** variant have slightly different pickle structures. Verify your chosen loader handles the specific `.pkl` version you're using — load it, decode a random β, check that the output mesh has 5,023 verts and no NaN values.
- If using MICA's loader, check that it's compatible with the FLAME version MICA itself was trained on (FLAME2020, not the older 2017 release).
- Run this test on Kaggle (not just locally) to confirm the numpy version in Kaggle's image matches.

---

## 2. MICA — Metrical Identity from a Single Image

**Full name:** Towards Metrical Reconstruction of Human Faces  
**Paper:** Zielonka et al., ECCV 2022  
**Repository:** [github.com/Zielon/MICA](https://github.com/Zielon/MICA)  
**License:** MIT-style research code  
**Parameters:** ~65M (encoder)

### What it does

MICA takes a single face photo and outputs a **metrically accurate FLAME identity shape code (β)** — meaning the output mesh not only looks like the person but is the correct physical size. It is expression-invariant: regardless of the subject's expression in the photo, MICA outputs the *neutral* identity shape.

### How it works internally

1. **Input:** 112×112 aligned face crop (same preprocessing as ArcFace face recognition).
2. **Backbone:** A frozen or fine-tuned **ArcFace face recognition network** (ResNet-100 or ViT). The key insight: face *recognition* networks already learn features that are identity-specific and expression-invariant, because that's exactly what recognition needs. MICA repurposes these features for 3D shape regression.
3. **Regression head:** A small MLP: ArcFace feature vector → 300-dim FLAME β.
4. **Output:** FLAME β → decoded via FLAME's PCA basis → neutral identity mesh (5,023 verts).

### Training data

A **unified multi-dataset** combining:
- **LYHM** (Liverpool-York Head Model) — 1,212 subjects with high-precision 3D scans
- **FaceWarehouse** — 150 subjects × 20 expressions
- **Stirling** — 136 subjects with 3D scans

All registered to FLAME topology, giving ~2,315 identities with paired (photo, ground-truth FLAME β) supervision.

### Published accuracy (NoW Challenge)

| Model | Validation median (mm) | Test median (mm) |
|---|---|---|
| **MICA** | **≈0.913** | **≈0.90** |
| DECA | ≈1.18 | — |
| Deep3D | — | ≈1.27 |

The ~1.27mm number is frequently misattributed to MICA — it's actually Deep3D's.

### Limitations

- Outputs only **neutral identity** — no expression, no detail
- ArcFace backbone was trained for 2D recognition, so 3D shape is derived from features learned for a 2D task — there's an alignment gap that fine-tuning on more scan data can narrow

### Multi-view fusion gotcha

> [!WARNING]
> **MICA was trained on single-image input.** Feeding it a profile photo produces a worse β than a frontal photo — the ArcFace backbone expects near-frontal faces and degrades on extreme poses. **Flat-averaging** β codes across N views (including bad profiles) **degrades** accuracy vs. using the best single frontal view alone.
>
> The correct approach is **confidence-weighted averaging**: InsightFace's `FaceAnalysis` returns a detection confidence score per face. Use this as the weight:

```python
# Correct multi-view fusion (in stage1_identity/inference.py)
betas = [mica(crop_i) for crop_i in aligned_crops]  # one β per view
confidences = [det_i.det_score for det_i in detections]  # InsightFace confidence

# Normalize weights to sum to 1
weights = softmax(torch.tensor(confidences), dim=0)
beta_fused = sum(w * b for w, b in zip(weights, betas))
```

> This naturally downweights profile and poorly-lit photos (low detection confidence) and upweights well-lit frontal photos (high confidence). Implement this from Day 1, not as a future upgrade.

---

## 3. DECA / EMOCA — Expression + Coarse Detail

### DECA

**Full name:** Detailed Expression Capture and Animation  
**Paper:** Feng et al., SIGGRAPH 2021  
**Repository:** [github.com/yfeng95/DECA](https://github.com/yfeng95/DECA)  
**License:** Non-commercial research — **check before shipping**

DECA takes a single face photo and outputs FLAME shape/expression/pose **plus** a UV-space displacement map encoding wrinkles and medium-frequency surface detail.

**Two-stage architecture:**
1. **Coarse stage:** ResNet-50 → FLAME parameters (β, ψ, θ) + camera + lighting. Differentiable renderer produces a synthetic image. Loss = photometric + landmark + regularization.
2. **Detail stage:** Separate encoder → **UV displacement map** (128×128 or 256×256) applied to FLAME mesh as per-vertex normal displacement. Trained with differentiable rendering loss.

### EMOCA

**Full name:** Emotion Driven Monocular Face Capture and Animation  
**Paper:** Danecek et al., CVPR 2022  
**Repository:** [github.com/radekd91/emoca](https://github.com/radekd91/emoca)

EMOCA improves DECA with emotion recognition supervision and perceptual loss (LPIPS), producing better detail under strong expressions.

### Licensing concern

Both released under **non-commercial research licenses**. For a commercial game product, either reimplement from the paper, obtain a commercial license, or use SMIRK (more permissive).

---

## 4. SMIRK — Analysis-by-Synthesis Expression Regression

**Paper:** Retsi et al., CVPR 2024  
**Repository:** [github.com/georgeretsi/smirk](https://github.com/georgeretsi/smirk)  
**License:** Research/MIT-ish (verify)  
**Parameters:** Small (ResNet-class)

SMIRK replaces DECA/EMOCA's "encode directly to parameters" approach with an **analysis-by-synthesis loop**: encode → render → compare → update → repeat. The render-and-compare loop is the training objective itself, producing more accurate expressions (especially subtle ones) than direct regression. Small model, fast inference, more permissive license than DECA/EMOCA.

---

## 5. Pixel3DMM — Per-Pixel Geometric Cues (ICLR 2026)

**Paper:** ICLR 2026  
**OpenReview:** [openreview.net/forum?id=UmOdd5KQ8K](https://openreview.net/forum?id=UmOdd5KQ8K)

Uses per-pixel depth/normal/curvature ViTs to constrain 3DMM optimization. **+15% geometric accuracy over prior SOTA** on both posed and neutral geometry. The strongest starting point available — track for v2 swap once code/weights are stable.

---

## 6. NPHM / MonoNPHM / Pix2NPHM — Beyond Linear PCA

**Website:** [simongiebenhain.github.io/NPHM/](https://simongiebenhain.github.io/NPHM/)

Replaces FLAME's linear PCA with **per-region neural SDF MLPs** — much higher capacity for ears, hairline, fine local deformations. Pix2NPHM (2025) provides single-image regression into NPHM space.

**Why it's a v2 upgrade, not v1:** Smaller tooling ecosystem, variable-topology output (harder retopology), and Pix2NPHM is very recent. Build v1 on FLAME, swap to NPHM if the 300-PCA ceiling becomes the fidelity bottleneck.

---

## 7. Supporting Tools — InsightFace, MediaPipe, ArcFace

### InsightFace
[github.com/deepinsight/insightface](https://github.com/deepinsight/insightface) — MIT license

Used for: (1) face detection, (2) 5-point landmarks for alignment, (3) ArcFace embedding for identity-preservation loss and evaluation.

### MediaPipe FaceMesh
Google MediaPipe — Apache 2.0

Used for: 478-point dense landmark detection (input to DECA/EMOCA). CPU-only, very fast.

### ArcFace as a loss function

In this pipeline, ArcFace is used as a **differentiable identity metric**, not for recognition:

```python
input_emb = frozen_arcface(input_photo)        # 512-dim
render = diff_render(output_mesh, camera)
output_emb = frozen_arcface(render)             # 512-dim
loss = 1 - cosine_similarity(input_emb, output_emb)
```

The ArcFace network is frozen — it's a fixed perceptual metric.

---

## 8. General-Purpose 3D Generators — Why They Don't Apply Here

| Model | VRAM (inference) | Why not |
|---|---|---|
| TRELLIS 2 | 16–24GB+ | Not face-specialized, can't match MICA's 0.9mm identity accuracy |
| Hunyuan3D 2.1 | 10–29GB | Same issue, plus restrictive Tencent license |
| TripoSR | ~6GB | Lightest but least detailed, not face-specialized |

These models generate arbitrary 3D objects from scratch. A face-specialized pipeline starts with 25 years of face-specific research. On "this specific person's face," the specialized pipeline always wins.

---

## Quick Reference

| Pipeline Stage | Model | Output | Params | VRAM |
|---|---|---|---|---|
| Stage 0 — Preprocess | InsightFace + MediaPipe | Detection, landmarks, alignment, camera pose | Small | CPU |
| Stage 1 — Identity | **MICA** (confidence-weighted multi-view fusion) | FLAME β (300-dim) → neutral mesh | ~65M | <2GB |
| Stage 1.5 — Neutral normalization | (post-processing) | Zero expression ψ and pose θ → canonical neutral mesh | N/A | CPU |
| Stage 2 — Expression | **EMOCA/SMIRK** | Expression ψ, pose θ (saved as metadata, NOT baked into base mesh) | ~50–100M | <4GB |
| Stage 3 — Detail | **Custom GAN** (conditioned via cross-attention over multi-view features) | High-freq displacement map (512×512) | ~20–50M | <8GB |
| Stage 5 — Export | ICT-FaceKit topology | Retopologized, rigged, MetaHuman-compatible FBX | N/A | CPU |
| Rendering | **nvdiffrast** (decided) | UV-space rasterization, differentiable rendering | N/A | <1GB |
| Evaluation | ArcFace (frozen) | Identity similarity score | ~65M | <2GB |
| Future (v2) | Pixel3DMM / NPHM | Higher-accuracy identity/detail | ViT / MLPs | TBD |

---

## Implementation Decisions Log

Update this section as you make real implementation choices during coding.

| Date | Decision | Value | Reason |
|---|---|---|---|
| Pre-impl | Differentiable renderer | **nvdiffrast** | Lighter install, actively maintained by NVIDIA, handles UV-space rasterization natively, has working pip wheel for Linux/Kaggle. pytorch3d would also work but has a heavier build and less native UV-space support. |
| Pre-impl | FLAME loader | Pure-numpy (MICA's loader) | Avoids chumpy dependency. Must verify against FLAME2020 `.pkl` on Day 0. |
| Pre-impl | Multi-view identity fusion | Confidence-weighted average (InsightFace det_score) | Flat averaging degrades accuracy because MICA produces worse β on profile photos. |
| Pre-impl | Base mesh expression state | Canonical neutral (ψ=0, θ=neutral) | UE5/MetaHuman blendshapes require a neutral base mesh. Expression stored as metadata for blendshape mapping. |
| | | | |
