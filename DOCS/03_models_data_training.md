# 03 — Models, Data, Training & the TRELLIS Question

> Which pretrained models to adopt, what (little) you should train yourself, what data exists,
> what the licenses allow, and what your hardware can actually do.
>
> ⚠️ Model and dataset facts below reflect the state of the field as I know it. Anything marked
> *(verify)* should be checked on the project page before you depend on it.

---

## 1. Reframe: **fit, don't fine-tune**

The instinct after v1 was "the models are bad, so train better ones" (Detail GAN, then MICA fine-tuning, then TRELLIS).
But `01_diagnosis` shows that the dominant errors are **wiring and estimation bugs**, not model capacity:

- a wrong embedding into MICA
- an 8× wrong focal length
- no occlusion test
- a dead expression model
- luminance used as height

The fix for most of these is **test-time optimization** (S3): fit the head to *this* photo. That needs **zero training**.

Training is only worth it where the photo physically lacks information and a learned prior must supply it:

1. **Unseen or badly lit texture** → T1, UV completion and delighting
2. **Wrinkle geometry** → T2, meso detail
3. **Unseen views** → T3, multi-view generation (optional; try pretrained first)

## 2. Pretrained models to adopt (no training)

| Stage | Model | Role | License | Confidence |
|---|---|---|---|---|
| S0 | **MediaPipe Face Landmarker** | 478 landmarks + 52 ARKit blendshape scores | Apache-2.0 | high |
| S0 | InsightFace RetinaFace / ArcFace | detection, loss-side identity | **non-commercial models** | high |
| S0 | AdaFace (or FaceNet) | *held-out* identity judge for evaluation | varies *(verify)* | high |
| S0 | **Pixel3DMM** | per-pixel FLAME-UV correspondence + normals, plus a FLAME fitting recipe | research *(verify)* | medium-high |
| S0 | SegFace / BiSeNet (CelebAMask-HQ) / Sapiens-seg | face parsing incl. hair, ears, cloth | mostly non-commercial *(verify)* | high |
| S0 | BiRefNet / ViTMatte | hair alpha matte | MIT / research | high |
| S0 | GeoCalib | focal estimate when EXIF is missing | *(verify)* | medium |
| S1 | **MICA** (with its own ArcFace!) | β₀ | non-commercial | high |
| S1 | **SMIRK** or TEASER | ψ₀, jaw, pose | code permissive; needs FLAME *(verify)* | high / medium |
| S3 | metrical-tracker / **VHAP** | reference FLAME photometric fitting code | research *(verify)* | high |
| S3 | nvdiffrast / PyTorch3D | differentiable rasterization | NVIDIA NC / BSD | high |
| S5 | DECA / EMOCA detail decoder | *interim* meso displacement until T2 exists | non-commercial | high |
| S2 | FaceLift, CAP4D, Morphable Diffusion, Arc2Face(+pose ControlNet), PanoHead inversion | single-image → multi-view head | research *(verify each)* | medium |
| S6 | DiffLocks (single-image strands), HairStep, NeuralHaircut / GaussianHaircut (video) | hair strands | research *(verify)* | medium |
| S7 | **ICT-FaceKit** | production topology + ARKit-compatible expressions + eyes/teeth | MIT | high |

## 3. What to train yourself

### T1 — UV texture completion + delighting (highest value)

- **Task:** (partial lit albedo in FLAME UV, mask, mirrored partial, identity embedding) → (complete even-lit albedo, roughness/specular).
- **Key trick: self-supervised pairs from clean UV textures.** No paired captures are needed:
  1. Take a clean, evenly lit UV texture T (FFHQ-UV or FaceScape textures, mapped once into FLAME UV).
  2. Render it on a FLAME mesh with a random pose, random SH or HDRI lighting and a random expression.
  3. Backproject with z-buffer visibility, exactly as S4 does, to get (partial lit texture, mask).
  4. Train to recover T. The degradation model *is* your inference pipeline, so the train/test gap is small.
- **Architecture:** start with a **LaMa-style FFC U-Net** + PatchGAN at 512 → 1024. It is stable, deterministic and T4-friendly. Move to a latent-diffusion inpainting LoRA only if large generated regions (back of head) look flat.
- **Losses:** L1 (+ higher weight on observed texels), LPIPS, adversarial, and a render-space identity loss.
- **Compute:** roughly 20–40 T4-hours (about 1–2 Kaggle weeks at 30 GPU-h/week), fp16 + GradScaler.
- **Success metric:** on held-out FaceScape subjects, completion LPIPS in unseen regions, and a delit-albedo error vs the ground-truth texture under 5 different relightings.

### T2 — Meso detail (wrinkles and folds as real geometry)

- **Task:** (observed UV texture, UV normal/position maps of the fitted mesh) → displacement in mm.
- **Data:** FaceScape provides multi-view photos + registered meshes + **displacement maps derived from high-res scans** for hundreds of subjects × 20 expressions. This is exactly the missing supervision that Multiface could never provide. You need a one-time FaceScape-topology → FLAME-UV transfer.
- **Architecture:** a pix2pixHD-style U-Net at 1024 (reuse the v1 generator skeleton; the *data* was the problem, not the network).
- **Losses:** L1 + gradient-domain L1 + a shading loss (render the displacement under random lights, compare the high-pass band with the photo's).
- **Compute:** roughly 10–30 T4-hours.
- **Interim:** the DECA/EMOCA detail decoder, which is better than luminance-as-height today.

### T3 — Head multi-view generation (optional; last)

Only train this if the pretrained S2 candidates drift on identity or ignore the FLAME conditioning.
The recipe would be a LoRA on a multi-view diffusion backbone, conditioned on FLAME normal renders, with data from NeRSemble / RenderMe-360 / Ava-256 / FaceScape multi-view photos. This is the first task where renting an A100/H100 is justified.

### Not worth training now

| Candidate | Why not |
|---|---|
| MICA fine-tune (old "Phase 2") | Needs licensed paired scan data. S3 fitting gives more per-subject accuracy than a better regressor. |
| The v1 Detail GAN | The data problem is fundamental (see `older/FAILED/RESEARCH_1`) |
| Stage 7 DelightUNet from scratch | Replaced by un-lighting via S3 SH + T1 |
| TRELLIS (see §4) | Wrong tool for a rigged head, and infeasible on your compute |

## 4. The TRELLIS question, in detail

**What TRELLIS is:** Microsoft's image-to-3D model. It uses a sparse *structured latent* on a voxel grid, generated by two rectified-flow transformers (sparse structure, then latent features) conditioned on DINOv2 image features. It decodes to Gaussians, radiance fields or meshes, and was trained on roughly 500K general 3D assets (Objaverse-XL etc.) on a multi-GPU A100 cluster. A larger TRELLIS.2 and Tencent's Hunyuan3D-2.x are in the same family *(verify current versions)*.

**Why it is the wrong core for this product**

| Requirement | TRELLIS-class output |
|---|---|
| Fixed topology, UVs, rig, ARKit morphs, LODs | Arbitrary topology. You would **still** need S7 registration, and then you are back to "fit a template to a surface". |
| Identity fidelity | Conditioned on generic DINOv2 features and trained mostly on objects. Faces come out as "generic sculpt of a similar person". |
| Facial resolution | Eyelid and lip-line features span only a few voxels at the sparse-structure resolution |
| Neutral expression, separable hair | Bakes the expression, hair and head into one surface |
| Film-grade texture | Baked lighting, low-resolution texture |
| Compute | **From scratch: impossible on Kaggle.** Fine-tune: thousands of licensed head assets rendered at ~150 views each, plus A100-class memory. T4s have no bf16 and no FlashAttention-2 (Turing), so a 1B+-parameter flow transformer fine-tune on 2×T4 is impractically slow even with LoRA. |
| License of data you would fine-tune on | FaceScape / NPHM / RenderMe-360 are non-commercial, so the fine-tuned model would be too |

**When TRELLIS *would* be right:** a static bust, a 3D print or a collectible, where no rig is needed. Or as an **auxiliary prior** in v3: run a *pretrained* TRELLIS/Hunyuan3D on the photo, take only the **hair and back-of-head shell**, and use it as a weak silhouette/depth target in S3 and as a seed for S6 hair. No fine-tuning.

**What to do with the effort already spent:** Park TRELLIS fine-tuning. Keep `src_v2_generative/datasets/facescape_loader.py`, because FaceScape is exactly the data T1 and T2 need.

> **Update 2026-09-25:** the project must be commercially usable. The dataset table below is the *research* landscape. Every entry except your own captures is non-commercial. See `07_commercial_licensing.md` for the commercial-safe replacements and training approach.

## 5. Datasets

| Dataset | Contents | v3 use | License *(verify)* |
|---|---|---|---|
| **FaceScape** | ~850 subjects × 20 expressions, multi-view photos, registered meshes, textures, displacement maps | T1, T2, evaluation | non-commercial research (request form in `data/licenses/`) |
| **FFHQ-UV** | ~50k normalized, evenly lit UV textures (HiFi3D topology) | T1 | derived from FFHQ; non-commercial |
| NeRSemble | multi-view video, hundreds of subjects, FLAME tracking | S3 evaluation, T3 | non-commercial |
| NPHM | ~255 subjects × ~20 expressions, head scans **with hair** | evaluation, T3 | non-commercial |
| RenderMe-360 | ~500 subjects, multi-view, FLAME fits, hair | T3, evaluation | non-commercial |
| Ava-256 | 256 subjects, multi-view studio captures | T3 | CC-BY-NC |
| NoW, REALY | single-image → scan benchmarks | S1/S3 evaluation | research |
| Multiface | tracked meshes, **no micro detail** | ❌ detail. OK for expression research. | CC-BY-NC |
| **Your own captures** (`older/data_collection_protocol.md`) | cross-polarized multi-view | film-grade path, commercial data | yours (with talent releases) |

## 6. Licensing reality check

If you ever intend to **sell** this or ship it in a commercial product, most of the current stack is research-only:

| Research-only today | Commercial-safe substitute |
|---|---|
| InsightFace model zoo (buffalo_l, ArcFace weights) | MediaPipe detection/landmarks; your own-trained or licensed recognizer |
| FLAME 2020, MICA, DECA/EMOCA | FLAME 2023 "open" variant *(verify its terms)*, ICT-FaceKit (MIT); estimators retrained on licensed data |
| nvdiffrast | PyTorch3D (BSD) or Mitsuba 3 (BSD) |
| FaceScape / NPHM / FFHQ-UV training data | Own captures; purchased scan libraries with ML-training rights |

**Recommendation:** Build v3 as research now, but keep every model behind an interface (`perception/`, `priors/`, `fitting/renderer.py`) so you can swap components per license later.

## 7. Compute plan

| Resource | Use it for | Don't use it for |
|---|---|---|
| Local laptop (8 GB, Apple silicon) | Orchestration, Blender look-dev, authoring `assets/template/`, CPU unit tests, reviewing reports | Any PyTorch training; S3 fitting at subdivision-2 |
| Kaggle 2×T4 or P100 (16 GB, fp16, ~30 GPU-h/week) | S0–S3 inference and fitting (~minutes/subject), T1 and T2 training (multi-session with the existing resume and emergency-checkpoint utilities) | Large transformer/diffusion pre-training, TRELLIS |
| Rented A100/H100 (hourly) | T3 only, if you get there | Anything a T4 can do |

T4 notes: use fp16 + `GradScaler` (no bf16). Use PyTorch SDPA or xformers attention (FlashAttention-2 does not support Turing). Keep AGENTS.md rules 2, 3 and 5 (disk quota, 11.5 h emergency checkpoint, dynamic GPU count).
