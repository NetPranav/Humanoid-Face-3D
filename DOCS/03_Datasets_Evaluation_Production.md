# Datasets, Evaluation & Production Pipeline — Research Reference

> This document covers everything outside the core ML models: the datasets used for training and evaluation, the evaluation metrics and benchmarks, the production export pipeline (UE5/MetaHuman targeting), facial hair as static geometry, and the Kaggle-only environment strategy.
>
> ⚠️ **CRITICAL ARCHITECTURE UPDATE (September 2026):**
> - **Meta Multiface Scan Limitations:** The Meta Multiface scan dataset was evaluated in Research 1 and found to be an optical tracking dataset with a frequency ceiling insufficient for 50-micron pores, alongside physical acquisition artifacts (latex bald caps). See [DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md).
> - **Active Production Specification:** The pipeline has adopted a 4-Tier Hybrid Engine combining photo-derived meso wrinkles, 4K anatomical pore synthesis, and Blender Cycles Random Walk SSS. See [DOCS/02_Metahuman_Film_Grade_Synthesis.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/02_Metahuman_Film_Grade_Synthesis.md) and [DOCS/05_Blender_Cycles_Film_Rendering_Engine.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/05_Blender_Cycles_Film_Rendering_Engine.md).

---

## Table of Contents

1. [Datasets — What Exists, What's Licensed, What to Use](#1-datasets)
2. [Evaluation Metrics and Benchmarks](#2-evaluation-metrics-and-benchmarks)
3. [Meshy 7 / 7.1 — The Benchmark Target](#3-meshy-7--71--the-benchmark-target)
4. [Production Export — UE5, MetaHuman, ICT-FaceKit](#4-production-export--ue5-metahuman-ict-facekit)
5. [Neutral-Expression Normalization](#5-neutral-expression-normalization)
6. [Facial Hair as Static Geometry](#6-facial-hair-as-static-geometry)
7. [Multi-Image Input — Why and How](#7-multi-image-input--why-and-how)
8. [Input Validation](#8-input-validation)
9. [Production Output Format — What the Artist Gets](#9-production-output-format)
10. [Kaggle Environment Strategy — Details](#10-kaggle-environment-strategy)

---

## 1. Datasets

### MICA Unified Dataset (Training — Stage 1/2)

**Contents:** RGB photos paired with registered FLAME identity shapes for ~2,315 subjects

**Component datasets:**

| Dataset | Subjects | What it provides | License |
|---|---|---|---|
| **LYHM** (Liverpool-York Head Model) | 1,212 | High-precision 3D head scans + photos | Academic — verify for commercial |
| **FaceWarehouse** | 150 × 20 expressions | RGB-D scans with multi-expression coverage | Academic |
| **Stirling** | 136 | 3D face scans + photos | Academic |

All registered to FLAME topology by the MICA authors, giving paired (photo → FLAME β ground truth) supervision.

**Use in this pipeline:** Fine-tuning the Stage 1 identity regressor (Phase 2). Attach as a private Kaggle Dataset.

---

### FaceScape (Training — Stage 3 detail)

**Contents:** 18,760 pore-level textured 3D face scans  
**Scale:** 938 subjects × 20 expressions  
**Topology:** Topologically uniform (all scans share the same connectivity)  
**Includes:** Per-vertex displacement maps, registered FLAME parameters  
**Source:** [nju-3dv.github.io/projects/FaceScape/](https://nju-3dv.github.io/projects/FaceScape/)  
**License:** ⚠️ **Non-commercial research only** — explicit "NO COMMERCIAL USE" clause

### Why FaceScape matters

It is the **single best public dataset** for the exact problem Stage 3 solves: pore-level, wrinkle-accurate facial detail geometry with ground-truth displacement maps. The GAN discriminator's "real" examples come from FaceScape scans. Multiple published papers (Section 2 of the GAN doc) train their facial detail networks specifically on FaceScape.

### The licensing problem

FaceScape is explicitly non-commercial. For a shipped commercial game product, three options:

1. **License FaceScape commercially.** Contact the Nanjing University authors. Academic groups often negotiate commercial terms — worth asking early (Phase 4).
2. **Capture your own scans.** Even a modest multi-camera or structured-light rig on 20–50 consenting subjects with release forms gives you commercially clean ground truth. Higher effort, highest quality ceiling.
3. **Synthetic bootstrapping.** Use procedural wrinkle/pore generators (Substance Designer, Houdini) on permissively-licensed base head meshes to create synthetic training data. Lower quality ceiling than real scans, but zero licensing risk.

**For prototyping (Phases 1–3):** FaceScape is fine under research terms.  
**For shipping:** Must resolve the licensing path before training production weights.

Since there's no deadline on this project, the recommended approach is to prototype on FaceScape, validate the pipeline works, then pursue Option 1 (licensing) as the primary path with Option 3 (synthetic) as a fallback.

---

### NoW Challenge (Evaluation)

**Full name:** "Now" Challenge — Not only When  
**Contents:** Standardized single-image face reconstruction accuracy benchmark  
**What it measures:** Point-to-surface distance (mm) between reconstructed face mesh and ground-truth 3D scan, on a set of held-out subjects  
**Published baselines:**

| Model | Validation median (mm) | Test median (mm) |
|---|---|---|
| **MICA** | **0.913** | **≈0.90** |
| DECA | 1.18 | — |
| Deep3D | — | 1.27 |

**Use in this pipeline:** The primary automated accuracy benchmark. Run after every phase to track improvement.

---

### ICT-FaceKit (Export Target — not a dataset)

**Contents:** A **topology + rig + blendshape standard**, not training data  
**License:** **MIT** — safe for commercial use  
**Source:** [github.com/USC-ICT/ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit)

ICT-FaceKit defines the target mesh topology, UV layout, skeleton hierarchy, and blendshape set for the exported game asset. It's MetaHuman-compatible by design — meshes that follow this standard import cleanly into Unreal Engine 5.

---

## 2. Evaluation Metrics and Benchmarks

### Point-to-Surface Distance (Chamfer Distance)

The standard geometric accuracy metric across all face reconstruction papers.

**What it measures:** For each vertex on the reconstructed mesh, find the closest point on the ground-truth scan surface. Report the median distance in millimeters.

**Variants:**
- **One-directional:** predicted → scan (how close is the prediction to the truth?)
- **Bidirectional (Chamfer):** average of predicted→scan and scan→predicted (also penalizes missing regions)

**Typical values:**
- MICA on NoW: ~0.9mm median
- "Good enough for games": <1.5mm median (below this, differences are invisible in-engine at normal camera distances)
- "Pore-level accurate": <0.3mm (requires scan-quality ground truth to even evaluate)

### Identity Verification Score (ArcFace Cosine Similarity)

**What it measures:** "Does the reconstructed face still look like the input person?"

```python
input_embedding = arcface(input_photo)              # 512-dim
render_embedding = arcface(rendered_mesh_image)      # 512-dim
score = cosine_similarity(input_embedding, render_embedding)
```

**Typical values:**
- Same person, different photo: 0.6–0.8
- Same person, reconstructed mesh render: >0.5 is a sanity check, >0.7 is good
- Different person: <0.3

**Use:** Both as a training loss (Stage 2/3) and as an evaluation metric. The same frozen ArcFace model is used for both.

### Surface-Detail Frequency Analysis

**What it measures:** How much high-frequency geometric detail the pipeline preserves compared to ground truth.

**Method:** Compute the mean curvature of the reconstructed mesh and the ground-truth scan, convert to frequency domain (FFT), compare power spectra. A pipeline that smooths away wrinkles will show a drop in high-frequency power relative to ground truth.

**Use:** This is the closest reproducible proxy for Meshy's self-reported "surface detail" axis.

---

## 3. Meshy 7 / 7.1 — The Benchmark Target

### What Meshy actually is

Meshy is a **general-purpose, any-object image/text-to-3D** commercial service, not a face specialist. Meshy 7 (August 2026) and Meshy 7.1 (September 2026) are their current flagship models.

### Their published numbers

Meshy's own benchmark (on their own evaluation methodology):

| Axis | Meshy 7 Score |
|---|---|
| Overall proportion | 81.0% |
| Spatial distribution | 79.7% |
| **Surface details** | **59.8%** |

Meshy 7.1 was released specifically to address the surface-detail gap, scaling internal voxel resolution from 2048³ to 4096³ (raw meshes up to 80M triangles before simplification).

### Why these numbers are and aren't useful

**Are useful:** They confirm that even the current best general-purpose model scores only ~60% on surface detail. This is the axis where a face-specialized pipeline has the clearest advantage.

**Aren't directly reproducible:** The "proportion / spatial distribution / surface details" scoring methodology is Meshy's internal system. You cannot run the same evaluation on your own output to get comparable numbers. Don't try.

### How to actually compare against Meshy

Instead of reproducing their private benchmark:
1. Run the same input photos through both Meshy and your pipeline
2. Measure **Chamfer distance to ground-truth scan** for both outputs (if scan data is available)
3. Measure **surface-detail frequency analysis** for both outputs
4. Measure **identity score** (ArcFace cosine similarity) for both outputs

This gives a fair, reproducible, head-to-head comparison on the axis that matters.

---

## 4. Production Export — UE5, MetaHuman, ICT-FaceKit

### Target: Unreal Engine 5 (MetaHuman-compatible)

The pipeline targets **UE5 with MetaHuman integration**. This means the exported asset must conform to MetaHuman's expectations:

### Topology: ICT-FaceKit

- **MIT-licensed**, MetaHuman-compatible lineage
- Fixed vertex count and ordering
- Standard UV layout that UE5's face material system expects
- The retopology step (Stage 5) maps FLAME's 5,023-vertex output onto ICT-FaceKit topology via closest-point projection + barycentric interpolation

### Blendshapes: ARKit-52

MetaHuman and UE5's Live Link system use **ARKit-52 blendshape targets** (Apple's facial expression standard, also called "blend shapes" in UE5). ICT-FaceKit already provides a mapping from its topology to ARKit-52 targets — reuse it.

The 52 targets include:
- `eyeBlinkLeft/Right`, `eyeWideLeft/Right`
- `jawOpen`, `jawForward`, `jawLeft/Right`
- `mouthSmileLeft/Right`, `mouthFrownLeft/Right`
- `browDownLeft/Right`, `browInnerUp`, `browOuterUpLeft/Right`
- `cheekPuff`, `cheekSquintLeft/Right`
- `noseSneerLeft/Right`
- ... and more

### Skeleton

Standard UE5 face skeleton: jaw bone, neck bone, two eye bones. ICT-FaceKit provides this. The FLAME model's pose parameters (jaw, neck, eyes) map directly onto these bones.

### Export format: FBX

UE5 imports FBX natively. The export (via headless Blender / `bpy`) must include:
- Mesh with ICT-FaceKit topology
- Skeleton (armature) with jaw/neck/eye bones
- Blendshapes (shape keys) for all 52 ARKit targets
- Clean UV layout (even without texture — artists paint later in UE5)
- LOD meshes (LOD0 ~10–30K tris, LOD1 ~5K, LOD2 ~2K, LOD3 ~500)
- Displacement/normal map as a separate `.exr` or `.png` file

### What UE5 expects that this pipeline does NOT produce

- **Texture / albedo** — this is a geometry-only pipeline. Artists apply materials in UE5.
- **Head hair** — UE5's Groom system or MetaHuman's hair assets handle this. The pipeline outputs a bald head.
- **Body** — face only. MetaHuman has its own body system.

---

## 5. Neutral-Expression Normalization

> [!IMPORTANT]
> **This step is missing from many published pipelines and causes broken blendshapes if omitted.**

### The problem

MICA outputs a neutral identity shape (β only, no expression). EMOCA/SMIRK adds expression (ψ) and pose (θ) back. But in production, the exported mesh must be in a **canonical neutral pose** (mouth closed, eyes open, no brow raise) so that UE5's ARKit-52 blendshapes work correctly — each blendshape is defined as a *delta from neutral*, so if the base mesh already has the mouth slightly open, `jawOpen` will open it further than intended.

### The fix

Add a **post-Stage-2, pre-Stage-3** normalization step:

```python
# In pipeline.py, between Stage 2 and Stage 3:

# Stage 2 produces: beta, expression_psi, pose_theta, coarse_detail
# We need to SEPARATE these:

# 1. The BASE MESH for export uses ONLY beta (identity), with expression/pose zeroed:
base_mesh = flame_model.decode(beta=beta, expression=torch.zeros_like(expression_psi), 
                                pose=neutral_pose)  # jaw closed, neck straight

# 2. Expression and pose are saved as METADATA for blendshape mapping in Stage 5:
expression_metadata = {
    'expression_psi': expression_psi,  # used to compute blendshape deltas
    'pose_theta': pose_theta,           # used to set skeleton rest pose
    'coarse_detail': coarse_detail,     # the UV displacement from EMOCA/SMIRK
}

# 3. Stage 3 (detail GAN) receives the NEUTRAL base mesh + the expression metadata
#    as conditioning — the GAN should produce detail appropriate for this person,
#    but on the neutral geometry, not on a mid-expression mesh.
```

### Why this matters for the entire pipeline

- **Blendshapes:** ARKit-52 targets are defined as vertex deltas from neutral. If the base mesh isn't neutral, every blendshape is offset.
- **Detail consistency:** Detail displacement maps should be authored on neutral geometry. Expression-specific detail (nasolabial folds when smiling) should be controlled by blendshapes in-engine, not baked into the base displacement.
- **Retopology:** The FLAME-to-ICT-FaceKit retopology mapping is computed once on a neutral mesh. Applying it to an expressive mesh produces distorted vertex correspondence.

---

## 6. Facial Hair as Static Geometry

### The requirement

The pipeline needs **facial hair** — beards, short mustaches, stubble — as **static geometry** (not simulated, not moving with player movement). This is distinct from head hair (which UE5/MetaHuman provides separately).

### Why facial hair is simpler than head hair

Head hair is an actively-researched unsolved problem (strand-level reconstruction from a single image). Facial hair in a game context is much more tractable because:

1. **It's short and close to the skin surface** — short beards and mustaches are essentially a geometric texture on the face, not free-flowing strands. They can be represented as displacement/normal map detail or very low-profile geometry cards, not as simulated strands.
2. **It doesn't need to move** — static geometry, no simulation, no physics.
3. **The variation space is smaller** — beard styles are more constrained than hairstyles.

### Implementation approaches (in order of complexity)

#### Approach 1: Displacement/Normal Map (simplest — potentially free from Stage 3)

Short stubble and very short beards can be represented entirely as **additional displacement on the chin/jaw/upper-lip region** of the Stage 3 detail map. If the training data (FaceScape or own captures) includes subjects with facial hair, the GAN will learn to reproduce facial hair texture as part of the displacement map.

- **Pros:** No separate stage needed. Falls out of Stage 3 training naturally.
- **Cons:** Only works for very short facial hair (stubble, light mustache). Cannot represent longer beards that extend beyond the face surface.
- **Data requirement:** Training data must include subjects with facial hair.

#### Approach 2: Geometry Cards on the Face Mesh (moderate)

For short-to-medium beards, generate a small number of **hair cards** (flat polygonal strips with alpha-masked hair texture) placed on the chin, jawline, and upper lip regions of the reconstructed head mesh. This is the standard game-industry approach for facial hair.

- **Pros:** Works for any beard length up to medium. Standard game technique, well-understood by artists. Static geometry, no simulation.
- **Cons:** Needs an artist-quality alpha texture for the hair cards. Placement needs to be automated (projected onto the face mesh at known facial regions).
- **Implementation:** Use the landmarks from Stage 0 to identify chin/jaw/upper-lip regions. Place card geometry at those locations, oriented along the face normal. Card density and length parameterized by "beard style" input.

#### Approach 3: Region-specific detail GAN conditioned on facial hair labels (most accurate)

Train a variant of the Stage 3 GAN that takes an additional conditioning input: a **facial hair mask** (binary or multi-class: clean-shaven / stubble / short beard / mustache) derived from the input photo. The GAN then produces displacement appropriate for each facial hair region.

- **Pros:** Photo-realistic, matches the input photo's actual facial hair.
- **Cons:** Needs training data with facial hair labels. More complex conditioning.
- **Data requirement:** Labeled examples (FaceScape subjects with/without facial hair, or own captures).

### Recommendation

Start with **Approach 1** (free from Stage 3 if data has facial hair examples) and validate whether the displacement map captures enough facial hair detail. If not, add **Approach 2** (geometry cards) for medium-length beards. **Approach 3** is a v2 refinement if Approaches 1–2 aren't sufficient.

---

## 7. Multi-Image Input — Why and How

### Why multi-image is required (not optional)

A single photograph fundamentally does not contain the information for:
- The back and sides of the head
- Precise metric depth (only relative depth from shading cues)
- Facial features occluded by head pose (if the photo is angled)

Multi-image input (3–5 photos: front + 2 quarter views + profile) provides:
- **Multiple viewpoints** → disambiguates depth
- **Occluded regions** → side/profile photos show ears, jaw shape, hairline from angles the front photo can't
- **Consistency checking** → multiple views of the same feature cross-validate each other

### How multi-image works in this pipeline

#### Stage 0 — Preprocessing (extended for multi-image)

For each input photo:
1. Detect face, extract landmarks, estimate camera pose
2. Produce aligned crop
3. Estimate relative camera position (which direction the subject is facing in each photo)

The camera pose estimation can use:
- **Landmark-based PnP** (Perspective-n-Point) — use the 2D landmarks and a generic 3D face model to estimate camera extrinsics for each view. Fast, works with MediaPipe/InsightFace landmarks.
- **COLMAP-style SfM** (if using a phone orbit video) — full structure-from-motion, more accurate but heavier.

#### Stage 1 — Identity (extended for multi-image)

Run MICA on each photo independently → N FLAME β codes. Then **fuse with confidence weighting**:

> [!WARNING]
> **Do NOT flat-average β codes across views.** MICA was trained on single frontal images. Feeding it a profile photo produces a worse β than a frontal photo. Flat-averaging a good frontal β with a bad profile β actually *degrades* accuracy vs. using the best single frontal view alone.

```python
# Correct: confidence-weighted fusion (in stage1_identity/inference.py)
betas = [mica(crop_i) for crop_i in aligned_crops]  # one β per view
confidences = [det_i.det_score for det_i in detections]  # InsightFace detection confidence

# Normalize weights to sum to 1 — softmax gives a smooth distribution
weights = softmax(torch.tensor(confidences), dim=0)
beta_fused = sum(w * b for w, b in zip(weights, betas))

# This naturally downweights profile/poorly-lit photos (low confidence)
# and upweights well-lit frontal photos (high confidence)
```

For the **best** results (v2 upgrade), use **joint optimization** instead of any averaging:

```python
# Fit a single β that, when rendered from each estimated camera pose,
# best matches all N photos simultaneously
beta_fused = optimize(beta, sum([photo_loss(render(beta, camera_i), photo_i) for i in range(N)]))
```

#### Stage 2 — Expression (per-view, not fused)

Expression is view-dependent (the subject might have slightly different expressions across photos). Run EMOCA/SMIRK per-view, pick the front-facing photo's expression as the canonical expression, or average.

The GAN is conditioned on features from **all** input views via **cross-attention** (not channel concatenation — see [Doc 02, Section 3](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/02_Adversarial_Detail_Synthesis.md)):

1. Run a frozen feature extractor (ResNet/ViT) on each input view independently → N feature maps
2. In the generator's bottleneck, use `nn.MultiheadAttention` to cross-attend from UV-space query positions to all N view feature maps
3. The attended features are fed into the generator's decoder alongside the coarse conditioning maps

This handles variable N views with zero architectural changes and gives the generator detail information visible from each angle (wrinkles on the side of the face visible only in the profile photo, etc.).

### What this costs

- **Inference time:** ~N× Stage 0–2 time (one forward pass per photo), negligible for N=3–5
- **Stage 3 training:** Same GPU cost — the generator still outputs a single displacement map, it just has more conditioning information
- **User experience:** Ask the user to upload 3–5 photos instead of 1. Validation is handled by the input validation cell (see Section 8).

---

## 8. Input Validation

> [!IMPORTANT]
> **Without input validation, `production_inference.ipynb` will silently fail or produce garbage.** This is the difference between a tool that artists trust and one they don't.

Add a validation cell at the **top** of `production_inference.ipynb`, immediately after the photos are uploaded and before any pipeline stage runs:

```python
# Input validation cell — runs before any pipeline stage
from src.stage0_preprocess import detector
from src.utils.validation import validate_inputs

photos = load_uploaded_photos()  # from notebook file upload widget

validation_result = validate_inputs(photos)
if not validation_result.is_valid:
    print("\n❌ INPUT VALIDATION FAILED:")
    for error in validation_result.errors:
        print(f"  • {error}")
    raise ValueError("Fix the input photos and re-run this cell.")

# Show preview grid of what was detected
validation_result.show_preview_grid()  # matplotlib grid: each photo with bbox overlay
```

### Validation checks (implemented in `src/utils/validation.py`)

| Check | Condition | Error message |
|---|---|---|
| **Face detected** | InsightFace returns at least one detection per photo | "No face detected in photo {i}. Check that the face is visible and well-lit." |
| **Same person** | Pairwise ArcFace cosine similarity > 0.4 for all pairs | "Photos {i} and {j} appear to be different people (similarity: {score:.2f}). All photos must be the same subject." |
| **At least one near-frontal** | At least one photo has estimated yaw < 30° | "No near-frontal photo detected. Include at least one front-facing photo (within ±30° of center)." |
| **Minimum photo count** | N ≥ 3 | "Only {N} photo(s) provided. This pipeline requires 3–5 photos for accurate reconstruction." |
| **Sufficient angular coverage** | Max pairwise yaw difference > 45° | "All photos are from nearly the same angle. Include at least one profile or quarter-view photo for depth accuracy." |

### Preview grid

After validation passes, display a preview: matplotlib grid showing each input photo with the detected bounding box overlaid, the estimated head yaw angle, and the detection confidence score. This lets the artist see what the pipeline "sees" before committing to a full run.

---

## 9. Production Output Format — What the Artist Gets

> [!IMPORTANT]
> **Define this now, not after Phase 7.** An artist opening the output for the first time needs to know exactly what's in the folder.

### Output directory structure

```
/kaggle/working/outputs/{run_id}/
├── manifest.json                    # What was generated, which model versions, reproducibility
├── head_mesh.fbx                    # Primary deliverable: bald head + skeleton + 52 blendshapes
├── head_mesh.obj                    # Validation/debugging copy (no rig)
├── lod/
│   ├── lod0.fbx                     # Full resolution (~10–30K tris)
│   ├── lod1.fbx                     # ~5K tris
│   ├── lod2.fbx                     # ~2K tris
│   └── lod3.fbx                     # ~500 tris
├── maps/
│   ├── displacement.exr             # High-freq displacement map (512×512, 32-bit float)
│   └── normal.png                   # Derived normal map (for engines that prefer normal over displacement)
├── facial_hair/                     # (only if facial hair geometry cards were generated)
│   └── beard_cards.fbx              # Separate sub-mesh, same skeleton
├── preview_render.png               # Quick Blender render under neutral lighting — sanity check
└── expression_metadata.json         # Detected expression codes for blendshape reference
```

### `manifest.json` format

```json
{
  "pipeline_version": "0.1.0",
  "generated_at": "2026-10-15T14:32:00Z",
  "input_photos": [
    {"filename": "front.jpg", "yaw_deg": 2.1, "detection_confidence": 0.98},
    {"filename": "quarter_left.jpg", "yaw_deg": -38.5, "detection_confidence": 0.91},
    {"filename": "profile_right.jpg", "yaw_deg": 72.3, "detection_confidence": 0.74}
  ],
  "model_versions": {
    "stage1_identity": "you/face-geo-stage1-identity/pytorch/v3",
    "stage2_expression": "you/face-geo-stage2-expression/pytorch/v2",
    "stage3_detail_gan": "you/face-geo-stage3-detail-gan/pytorch/v5",
    "stage3_detail_diffusion": "you/face-geo-stage3-detail-diffusion/pytorch/v2"
  },
  "metrics": {
    "identity_cosine_similarity": 0.78,
    "beta_fusion_weights": [0.52, 0.31, 0.17]
  },
  "topology": "ICT-FaceKit",
  "blendshape_standard": "ARKit-52",
  "displacement_p99_mm": 0.43,
  "lod_triangle_counts": [24500, 5000, 2000, 500]
}
```

**Why `manifest.json` matters:**
- **Reproducibility:** When you push a new checkpoint version and output quality changes, the manifest tells you *which* model versions produced a given output. Without it, debugging "this face looked better last week" is a guessing game.
- **Artist trust:** The identity cosine similarity and fusion weights give the artist a quick confidence signal before opening the FBX.
- **Automation:** Downstream tools (UE5 import scripts, batch processing) can parse the manifest rather than guessing file structure.

### Preview render

Generated by headless Blender (`bpy`) at the end of the export step: a single 1024×1024 render of the mesh under neutral 3-point lighting, front view. Takes <5 seconds on CPU. Lets the artist sanity-check the output before downloading the FBX and importing to UE5.

---

## 10. Kaggle Environment Strategy

### What Kaggle gives you

- **2× NVIDIA Tesla T4**, 16GB VRAM each (32GB combined)
- ~32GB system RAM
- 12-hour session cap (notebook is killed after 12 hours)
- ~30 GPU-hours/week quota (resets weekly)
- Linux-based environment (sidesteps Windows build issues)

### The model registry workflow

```
Training notebook                    Kaggle Models registry
┌─────────────────┐                 ┌──────────────────┐
│  Train Stage N  │ ──model_upload──▶│  face-geo-stageN │
│  on T4×2        │                 │  /pytorch/v1     │
└─────────────────┘                 └────────┬─────────┘
                                             │
Production notebook                          │ model_download
┌─────────────────┐                          │
│  Run inference  │ ◀───────────────────────┘
│  production_    │
│  inference.ipynb│
└─────────────────┘
```

Each stage's checkpoint is versioned independently. Rolling back a bad fine-tune is a matter of pointing `production_inference.ipynb` at the previous Model version.

### GPU quota budgeting

The 30-hour weekly quota is shared between training and inference:

| Activity | GPU-hours per week (estimate) |
|---|---|
| Phase 1–2 training | 5–15 (varies by phase) |
| Phase 3 GAN training | 10–20 |
| Production inference runs | 0.1–0.5 per face (5 stages × a few seconds each) |

At 0.1–0.5 GPU-hours per face, a 30-hour quota supports **60–300 faces per week** of pure inference use. If training is also happening in the same week, subtract training hours first.

### What doesn't need GPU

Run on **CPU-only Kaggle sessions** (no quota cost):
- Stage 5 export (retopology, LODs, blendshapes, FBX)
- Most evaluation (Chamfer distance, ArcFace on a few images)
- UV displacement dataset preprocessing
- Code testing and validation

### Code delivery

**Recommended: private GitHub repo + deploy key in Kaggle Secrets.**

```python
# Top of every Kaggle notebook:
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
deploy_key = secrets.get_secret("github_deploy_key")

# Write deploy key, clone repo
os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
with open(os.path.expanduser("~/.ssh/id_ed25519"), "w") as f:
    f.write(deploy_key)
os.chmod(os.path.expanduser("~/.ssh/id_ed25519"), 0o600)

!ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
!git clone git@github.com:youruser/face-geo-pipeline.git /kaggle/working/face-geo-pipeline
!pip install -e /kaggle/working/face-geo-pipeline
```

Every `git pull` picks up your latest local commits without re-uploading anything.

### Session management and checkpointing

```python
# Checkpoint pattern for 12-hour session cap
import time

start_time = time.time()
MAX_SESSION_HOURS = 11.5  # leave 30 min buffer before the hard 12hr kill

for step in range(total_steps):
    train_step(...)
    
    # Checkpoint every N steps
    if step % checkpoint_every == 0:
        save_checkpoint(f"/kaggle/working/checkpoint_step_{step}.pt")
    
    # Emergency checkpoint near session end
    elapsed_hours = (time.time() - start_time) / 3600
    if elapsed_hours > MAX_SESSION_HOURS:
        save_checkpoint("/kaggle/working/checkpoint_final.pt")
        print(f"Session approaching 12hr limit. Saved checkpoint at step {step}.")
        break

# At end of session: push to Kaggle Dataset for next session to resume from
# Or, if checkpoint is good enough: push to Kaggle Models for production use
```

### Limitations to be honest about

- **No uptime guarantee.** Sessions can be pre-empted. Always checkpoint.
- **30 GPU-hours/week is shared.** Heavy training weeks leave less room for inference.
- **Not suitable for multi-user production serving.** Fine for a solo developer generating assets on demand. For a team, pull checkpoints from Kaggle Models onto a persistent server.
- **No persistent filesystem between sessions.** Everything in `/kaggle/working/` is lost when the session ends unless saved as a Dataset or Model output. This is why the Model registry is the source of truth, not the filesystem.

---

## Implementation Decisions Log

Update this section as you make real implementation choices during coding.

| Date | Decision | Value | Reason |
|---|---|---|---|
| Pre-impl | Multi-view identity fusion | Confidence-weighted average (InsightFace det_score) | Flat averaging degrades accuracy because MICA produces worse β on profile photos. |
| Pre-impl | Multi-view detail conditioning | Cross-attention (not channel concat) | Variable N views, no architectural changes per view count. |
| Pre-impl | Base mesh state | Canonical neutral (ψ=0, θ=neutral) | UE5 blendshapes are deltas from neutral; baked expression breaks them. |
| Pre-impl | Output format | manifest.json + preview render + structured folder | Reproducibility, artist trust, downstream automation. |
| Pre-impl | Input validation | 5 checks before pipeline runs | Prevents silent failures on bad input (wrong person, no face, same angle). |
| Pre-impl | Displacement normalization | 99th percentile clip saved in manifest | Prevents D collapse on magnitude; enables denormalization at inference. |
| | | | |
