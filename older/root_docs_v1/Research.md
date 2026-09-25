# Image → 3D Geometry-Only Face Reconstruction: Research & Build Plan

**Target:** a locally-trainable/fine-tunable pipeline that takes one face photo and produces an identical, game-ready 3D **shape** (no texture) — down to eyebrow/eye depth ratios, wrinkles, and eventually hair strands — deployable as production tooling for a game studio.

**Target inference hardware:** RTX 4060 8GB VRAM, 16GB RAM, Windows.
**Target training environment:** Kaggle Notebooks, GPU T4×2 accelerator, 12-hour session cap, ~30 GPU-hours/week quota.
**Benchmark to beat (faces specifically):** Meshy 7 / Meshy 7.1 (Meshy AI's current foundation models, Aug–Sep 2026).

> **Updates in this revision:** (1) added the real Kaggle T4×2 training-environment specs and strategy (Section 8), since that — not the 4060 — is where actual training happens; (2) added current research on Meshy 7 / 7.1 as the concrete quality bar, including their own published benchmark numbers (Section 4); (3) added a full adversarial (GAN) detail-synthesis design — generator + discriminator, losses, architecture, training cost — as the primary design for the fine-detail stage (Section 5), since this turns out to be closer to the actual published state of the art for facial detail than the diffusion-only version of Stage 3 in the original plan.

---

## 0. Read this first — correcting the starting premise

Before the research, two assumptions in the original plan need to be corrected, because they change the whole approach:

1. **"Take a 3B model and bump it to 7B params" is not a real operation on a pretrained model.** With LLMs, "3B" vs "7B" describes a from-scratch training run at that size. You cannot open a checkpoint and widen/deepen its layers and keep its learned behavior — that's architecture surgery, and it invalidates all the pretrained weights unless you retrain, which needs the same order of compute as training a 7B model from scratch (tens of thousands of GPU-hours, not something a Kaggle T4×2 quota or a 4060 does). **What you actually want** is not "more parameters," it's "more capacity where it matters" — which for this task means higher-resolution detail maps and a properly-designed detail stage (Sections 4–5), not a bigger base model.
2. **3D generative model VRAM budgets don't behave like LLM VRAM budgets.** LLM fine-tuning tricks (QLoRA, 4-bit) exist because attention over tokens is the dominant cost and is well understood. 3D diffusion backbones (Hunyuan3D, TRELLIS) spend most of memory on dense voxel/point/latent volumes and multi-view renders during the forward pass — quantizing weights barely touches that. This is why every general-purpose open-source image-to-3D backbone below lists **16–29GB VRAM even for inference**, let alone training. Neither your 4060 nor a single Kaggle T4 (16GB) fine-tunes one of these directly, regardless of technique.

The good news: **face geometry doesn't need a general 3D object generator.** Faces are the single most over-fit, most data-rich, most parametrized object category in 3D computer vision — there's 25 years of dedicated face-only research that gets you closer to "identical, pore/wrinkle-level" results than a generic image→mesh model ever will, and it runs on far less hardware because the output isn't an arbitrary voxel grid, it's a small set of numbers (shape/expression coefficients or a compact latent) driving a fixed or semi-fixed topology.

**Revised framing:** don't fine-tune a general 3D generator. Build a **staged, face-specialized pipeline** where each stage is small enough to train on Kaggle's free T4×2 tier, and only rent a cloud GPU (a few dollars, a few hours) for the one stage that genuinely needs it (likely the hair pass).

---

## 1. Problem decomposition

"Perfect shape, identical to the photo" is actually three different sub-problems with different data, different model classes, and different levels of feasibility from a *single* 2D image:

| Layer | What it is | Feasible from 1 image? | Best current approach |
|---|---|---|---|
| **Macro identity** | Overall head/face proportions, bone structure, ratios between eyes/nose/jaw | Yes, well-solved | Metrical identity regression (MICA-style) into FLAME or NPHM space |
| **Meso detail** | Wrinkles, eyebrow volume/depth, eyelid folds, lip shape, expression | Yes, well-solved | Detail displacement regression (DECA/EMOCA/HRN-style), now recommended as an **adversarial (GAN)** stage — see Section 5 |
| **Micro detail** | Pores, fine skin texture-driven geometry, individual hair strands | Partially — pores need multi-view/light-stage data to be *metrically* correct from one photo; strand-level hair from one photo is a genuinely unsolved, actively-researched problem (2025–2026 papers, see §6) | UV-space GAN/diffusion for pores; dedicated single-image strand-diffusion models for hair, run as a separate pass |

Also important: **a single photograph fundamentally does not contain the information for the back and sides of the head**, or precise metric depth. "Looks identical" from the photographed angle is achievable; "identical" from every angle is not derivable from one image by any model — that's a hallucination problem, not a training problem. If true fidelity matters (production game asset), plan for **3–5 photos (front + 2 quarter + profile) or a short phone video** as the real input, with single-image as a degraded-input fallback. This alone will improve your results more than any model change.

---

## 2. Recommended architecture: a staged pipeline, not one model

```
Photo(s)
   │
   ▼
[Stage 0]  Face detection, landmarking, alignment, crop, camera-pose estimate
   │        (InsightFace / MediaPipe FaceMesh / 3DDFA-v2 — all trivial on a 4060)
   ▼
[Stage 1]  Identity regression → metrical FLAME or NPHM identity code
   │        (MICA-style ViT/ResNet encoder, ~30-100M params)
   ▼
[Stage 2]  Expression + coarse-detail regression → jaw/eyelid/brow pose,
   │        wrinkle displacement map (DECA/EMOCA/HRN/SMIRK-style)
   ▼
[Stage 3]  High-frequency detail synthesis → **adversarial (GAN) generator/
   │        discriminator pair**, optionally hybridized with UV-space diffusion,
   │        refining the displacement/normal map (pore-scale), conditioned on
   │        the aligned face crop. See Section 5 for the full design — this is
   │        where LoRA/GAN fine-tuning on a small card or Kaggle quota works well.
   ▼
[Stage 4]  Hair — separate specialist model (strand diffusion / hair cards),
   │        composited, not baked into the face mesh's topology.
   ▼
[Stage 5]  Retopology + rig transfer → fixed game-ready topology
            (ICT-FaceKit / MetaHuman-compatible), LOD generation, export
            (FBX/glTF), blendshape/ARKit-52 mapping for animation.
```

Why this beats "one big fine-tuned model":
- Every stage individually fits comfortably on 8GB (and more comfortably still on Kaggle's 16GB×2), several fit on far less.
- You can swap/upgrade any stage later (e.g., replace Stage 3 with a better detail model in 2027) without retraining everything.
- Stage 5 (retopology) is a solved, mostly non-ML geometry-processing problem — doing it right is what actually makes this "production grade," not the ML.
- It maps naturally onto what game engines already consume: parametric identity/expression codes are exactly how MetaHuman, ICT-FaceKit and most modern facial rigs work internally.
- It is, structurally, the same decomposition Meshy itself ended up shipping — a base-shape stage, then a dedicated later detail stage (Section 4) — just built for one object category instead of everything.

---

## 3. Model survey (identity + detail — the face-specific core)

| Model | What it does | Params (approx) | License | Why it matters here |
|---|---|---|---|---|
| **FLAME** (Li et al.) | Linear 3DMM: identity + expression + jaw/neck/eye pose → fixed-topology mesh (5023 verts) | Not a network — PCA bases | Research license, free for most uses; check terms for commercial | The de facto interchange format for face geometry research; huge tooling ecosystem |
| **MICA** (Zielon et al., ECCV 2022) | Single image → *metrically accurate*, expression-invariant FLAME identity shape, using a face-recognition backbone as feature extractor | ~65M (encoder) | MIT-style research code | Directly solves your "macro identity, looks like them" requirement; trained on unified LYHM/FaceWarehouse/Stirling data (~2,315 identities) |
| **DECA / EMOCA** | Adds detailed, animatable wrinkle geometry on top of FLAME via a displacement UV map, from a single image; EMOCA improves under strong expressions | ~50-100M | Research license (non-commercial by default — check before shipping) | Reference architecture for the "meso detail" stage |
| **SMIRK** (CVPR 2024) | Analysis-by-neural-synthesis FLAME regression — closes the loop by rendering and comparing, more accurate expressions than DECA | Small (ResNet-class) | Research/MIT-ish, verify | Good base to fine-tune your own expression regressor |
| **Pixel3DMM** (ICLR 2026) | Per-pixel geometric-cue ViTs constrain 3DMM optimization; **+15% geometric accuracy over prior SOTA**, first benchmark covering both posed and neutral geometry | ViT-based | Research (check on release) | Newest (Jan 2026) SOTA for single-image accuracy — the strongest starting point available today |
| **NPHM / MonoNPHM** (Giebenhain et al.) | *Neural* parametric head model: replaces FLAME's linear PCA with per-region SDF MLPs in a canonical space — captures ears, hairline, fine local detail that FLAME's linear basis structurally cannot | Multiple small local MLPs | Research | If FLAME's ceiling on "identical" bothers you, this is the upgrade path — higher representational capacity, still small enough to run/train modestly |
| **Pix2NPHM** (2025) | Robust single-image regression *into* NPHM space (surface-normal supervised, not just photometric) | Moderate | Research | Closes the gap between NPHM's better representation and single-image usability |
| **HRN / FaceCraft4D / Any3DAvatar** | Full-head avatar reconstruction from a single portrait, geometry + some hair/neck handling | Varies | Research | Useful references for whole-head (not just face) completion |
| **FaceScape displacement-map network** | Learns *expression-specific dynamic pore/wrinkle detail* as UV displacement maps from FaceScape's pore-level scans | CNN-class | **Data is non-commercial only** (see §7) | The direct blueprint for your Stage 3 detail data — but you cannot train on the FaceScape data itself for a commercial game asset without a license from the authors |

**Recommendation:** start from **MICA (identity) + SMIRK or EMOCA (expression/coarse detail)**, both permissively-enough licensed and small enough to fine-tune end-to-end on Kaggle's free tier in hours, not days. Track **Pixel3DMM** (ICLR 2026, just published) as the accuracy upgrade once code/weights are public — it's explicitly benchmarked on the metric you care about (posed + neutral geometric accuracy). Treat **NPHM/Pix2NPHM** as your v2 target once the FLAME-based v1 pipeline works, since it removes FLAME's low-frequency ceiling.

### General-purpose 3D generators (for context / non-face use only)
Since you mentioned "living things" broadly, not just faces, these are the current best general open-source image→3D backbones — but note the hardware reality:

| Model | VRAM (inference) | License | Notes |
|---|---|---|---|
| TRELLIS 2 (Microsoft) | 16–24GB+ | MIT | Best general quality, PBR materials, won't fit your 4060 for anything but maybe distilled/quantized inference |
| Hunyuan3D 2.1 (Tencent) | 10GB (shape only) – 29GB (full pipeline) | Tencent Community License — has geographic/scale commercial restrictions, check before shipping | Shape-only model at 10GB *inference* is roughly your local ceiling; training is well above 8GB and tight even on a single Kaggle T4 |
| Hunyuan3D-2mini | ~6GB | Same license family | 0.6B, lightest in the family, image-to-shape only |
| TripoSR / Stable Fast 3D | ~6GB | Permissive (check per-model) | Fast, general, but not face-specialized — not what you want for identity fidelity |

**Bottom line:** don't route your face pipeline through these. They're built for arbitrary objects, need more VRAM than a single 4060 or T4 has for training, and none of them will out-perform a face-specialized FLAME/NPHM pipeline on "identical to this specific person's face."

---

## 4. Benchmark target: Meshy 7 / Meshy 7.1 — and why a face-specialized pipeline can still beat it

Meshy 7 went live on August 10, 2026 (GA August 12), as the successor to Meshy 6; Meshy 7.1 rolled out September 10, 2026 as a follow-up release. As of this writing (mid-September 2026) it's Meshy AI's current flagship image-to-3D foundation model, so it's a reasonable, concrete quality bar — but it's worth understanding *what kind* of model it is before treating it as the thing to beat.

**What it actually is:** a general-purpose, any-object image/text-to-3D model, not a face specialist. Meshy's own description of the Meshy 7 upgrade is explicitly about alignment, not about faces or biological detail specifically: it uses a multi-scale image encoder that reads the reference image at several resolutions in one pass so fine shape cues aren't averaged away, and a rebuilt training set where every sample is tied tightly to its target geometry with style, lighting and background stripped out, so the model is pushed to learn shape rather than appearance.

**Its own published numbers are the most useful data point here.** On Meshy's own single-view geometry-alignment benchmark, Meshy 7 scores **81.0% on overall proportion, 79.7% on spatial distribution, and only 59.8% on surface details** — ahead of the competing models it was benchmarked against (Tripo 3.1, Hunyuan 3.1, Rodin 2.5), but by a much smaller margin on surface detail than on proportion. That gap is exactly the pattern this document's Section 1 predicted independently: macro proportion is the easy, largely-solved axis; fine surface detail is the hard one, even for the current best general-purpose model.

**Meshy 7.1 confirms this directly.** It's described as a pure detail upgrade on top of 7 — scaling the internal geometry-generation resolution from 2048³ to 4096³ voxels (raw meshes up to 80M triangles before simplification), specifically because, in Meshy's own framing, alignment and structural correctness are approaching production quality across the field and "the surface itself becomes the frontier." Structurally this is the same two-tier decomposition as this document's Section 2 design: get the base shape right first, then add a dedicated, separately-trained stage purely for fine surface detail. Meshy validated theirs with a "Detail Richness" benchmark and reports leading scores at 1K/2K/4K render resolutions.

**Why a narrow, face-specialized pipeline has a real (not just aspirational) shot at beating Meshy on faces specifically:** Meshy has to spend its capacity being decent across *every* object category — people, animals, buildings, hard-surface props, organic shapes — from one shared representation, with training supervision that's mostly image/geometry pairs rather than metrically-precise scan data for any one category. This project's pipeline instead:
1. spends all of its capacity on one object category,
2. can use ground-truth, pore-level metric scan supervision for that one category (FaceScape-class data — Section 7), which Meshy's general training set almost certainly doesn't have at the same density for faces specifically, and
3. outputs into a small, fixed, well-studied parametric space (FLAME/NPHM) instead of a raw dense voxel grid, so the model doesn't have to "discover" facial structure from scratch the way a general voxel/latent generator does.

Those three levers — category specialization, better ground truth, and a constrained output space — are precisely the standard reasons a small specialized model beats a much larger general one on a narrow task. Beating Meshy 7.1's *surface-detail* number specifically on faces is a realistic target for this pipeline; beating Meshy on arbitrary objects is not the goal and isn't what this pipeline is built for.

One caveat to carry forward honestly: the 81.0% / 79.7% / 59.8% figures are Meshy's own self-reported benchmark, not an independently audited third-party number — treat it as a vendor-stated ballpark to aim past, not a certified target.

---

## 5. The adversarial (GAN) detail stage — Generator vs. Discriminator, applied to Stage 3

This is the piece you specifically asked to have added: a setup where one network produces the detail geometry and a second network scores how convincing it is, with both improving through competition. The correct terminology for the two networks is **generator (G)** and **discriminator (D)** — there's no separate "constructor" in the standard formulation, so that's the vocabulary used below and in any code/papers you read next.

### 5.1 This is not a novel idea for this exact problem — it's closer to the mainstream published approach than the diffusion-only version originally proposed

Applying a generator/discriminator pair specifically to *facial displacement/detail maps* (not photos) is an established sub-field, with several directly relevant papers:

- **Chen et al., "Photo-Realistic Facial Details Synthesis from Single Image" (ICCV 2019)** — trains a *Deep Facial Detail Net* with exactly this setup: a generator takes an image patch (plus a noise vector) and predicts a wrinkle/pore displacement map; a discriminator judges real scan-derived maps versus generated ones. It's trained on 706 high-precision 3D face scans plus 163,000 in-the-wild images, using a conditional-GAN loss combined with an L1 reconstruction term (weighted roughly 100:1 in favor of L1 early in training). This is close to a direct blueprint for what you're describing.
- A dedicated paper ("Detail 3D Face Reconstruction Based on 3DMM and Displacement Map") trains this setup directly on **FaceScape**: a conditional GAN synthesizes the displacement map that gets applied back onto the coarse 3DMM reconstruction to reproduce real facial detail, evaluated specifically against FaceScape data.
- **Abrevaya et al., "A Decoupled 3D Facial Shape Model by Adversarial Training" (ICCV 2019)** puts the discriminator on the *whole mesh* (converted to a geometry image) rather than only a 2D displacement patch, and — usefully for you — adds auxiliary **identity** and **expression** classifier heads to the discriminator, AC-GAN-style. The discriminator doesn't just decide real/fake; it also has to correctly say *whose* face and *what expression* it's looking at. That pushes the generator toward detail that's actually tied to the specific input person and expression, rather than generically plausible-looking wrinkles — which is exactly your "must match this person's eyebrow depth/wrinkles" requirement.
- **"Structure-aware Editable Morphable Model" (arXiv 2207.09019)** is the most modern reference: it reuses a StyleGAN2-style generator/discriminator backbone, adds an R1 gradient-penalty term on the discriminator for training stability, models the displacement map at 256×256 resolution (found sufficient to encode wrinkle-level detail), and normalizes displacement/distance-field value ranges so the discriminator doesn't collapse onto a single channel. This is a good, modern architecture to copy nearly as-is.

Given that, this document now recommends the adversarial setup as the **primary** design for Stage 3, with UV-space diffusion (from the earlier draft of this plan) kept as a complementary technique — see 5.5.

### 5.2 What G and D concretely are, mapped onto this pipeline

- **Generator (G):** a small encoder–decoder (U-Net-style, or a StyleGAN2-style synthesis network). Inputs: (a) the coarse geometry already produced by Stage 1+2, rendered as a low-frequency UV position/normal map; (b) aligned face-crop image features from a frozen or lightly fine-tuned CNN/ViT; (c) the identity + expression codes already computed. Output: a **high-frequency residual** — a displacement or normal map in UV space encoding wrinkles, eyebrow volume, eyelid folds (the "meso detail" row from Section 1's table). This residual is *added on top of* the coarse FLAME/NPHM mesh rather than generated from scratch, so G never has to reinvent macro identity — only the fine layer on top of it, which is a much lower-dimensional, easier, and cheaper problem, which is exactly why it fits modest hardware.
- **Discriminator (D):** a patch-based classifier (PatchGAN- or StyleGAN2-discriminator-style) that looks at crops of the generated displacement/normal map and tries to tell it apart from a *real* displacement map taken from an actual high-precision scan (FaceScape-class data for prototyping, or your own captured scans for a shippable version — see licensing notes in Section 7). Optionally give D the two auxiliary heads from Abrevaya et al.: one predicting identity, one predicting expression, both of which must be correct even on real samples and are used to penalize generated samples that get misclassified.
- **The competition, concretely:** G is updated to make its detail maps fool D (and get correctly attributed by D's auxiliary heads); D is updated to get better at telling real scans from generated maps. Alternating gradient steps, standard adversarial minimax objective — this is the "one network tells the other how wrong it is, so it corrects itself" loop you described, applied specifically to facial detail geometry.

### 5.3 Full loss recipe — never train on the adversarial term alone

Every paper above adds the adversarial loss *on top of*, never instead of, several supervised terms; pure adversarial training drifts away from the real face on its own:

1. **Adversarial loss** — non-saturating GAN loss or hinge loss on D's real/fake decision, plus an R1 gradient penalty on D (StyleGAN2-style) for training stability.
2. **L1/L2 reconstruction loss** against the ground-truth displacement map wherever you have paired scan data — weight this heavily early in training (λ ≈ 100 relative to the adversarial term, following the Chen et al. cGAN+L1 recipe), then anneal it down as training progresses.
3. **Identity-preservation loss** — render the detailed mesh, pass both it and the original photo through a frozen face-recognition embedding network (ArcFace/InsightFace, the same tool used for evaluation in Section 11), and penalize embedding distance. Directly optimizes for "still looks like this specific person."
4. **Photometric/relighting-consistency loss** — render the detail-displaced mesh under a few synthetic light directions and compare shading gradients against the input photo (or the real scan's shading). Stops G from producing detail that looks plausible flat but shades wrong under light — standard practice in this literature.
5. **Auxiliary identity/expression classification losses** on D's heads, if included (Section 5.2), applied to both real and generated samples.

### 5.4 Hardware and training-cost fit

This is comfortably cheaper than the diffusion alternative. A StyleGAN2-class G/D pair operating on 256×256 (up to 512×512) single- or few-channel UV maps is a small fraction of the compute of a photoreal StyleGAN2 run on RGB images, let alone an SDXL-class diffusion model. Expect low tens of GPU-hours for a working checkpoint, not hundreds — comfortably inside a few weeks of Kaggle's T4×2 quota (Section 8), and light enough to fine-tune further afterward on the local 4060 for fast iteration once a base checkpoint exists.

### 5.5 GAN vs. diffusion for Stage 3 — not actually an either/or choice

- **GAN-only Stage 3** (this section): fast to train and to run at inference (single forward pass, no denoising steps), sharper high-frequency detail — GANs are known to produce crisper high-frequency texture than diffusion models, which is exactly what pores/wrinkles need — but can be less stable to train and, without the auxiliary losses in 5.3, can drift toward generic-looking detail.
- **Diffusion-only Stage 3** (the earlier version of this plan): more stable training, better sample diversity, easier to condition cleanly, but slower at inference (many denoising steps) and prone to slightly softer high-frequency output — arguably part of why even Meshy needed a dedicated 7.1 release just to push surface-detail sharpness further (Section 4).
- **Hybrid (the actual recommended target):** train the UV-space diffusion model first for stability and diversity, then fine-tune it with an added adversarial loss from a discriminator on its denoised output (the same "diffusion-GAN hybrid" trick used elsewhere to sharpen diffusion outputs, e.g. in fast-diffusion/distillation work). You get the diffusion model's stability through the bulk of training and the adversarial loss's sharpening effect in a final pass. This combination is plausibly your best realistic shot at beating Meshy 7.1's surface-detail number specifically on faces, since Meshy is a single general-purpose model with no equivalent face-specific adversarial fine-tuning pass.

---

## 6. Hair (separate problem, separate model)

Strand-level hair from a single photo is an active 2025–2026 research area, not a solved problem — be honest with your own expectations here before promising it in a production spec.

| Model | Approach | Maturity |
|---|---|---|
| **Neural Haircut** (ICCV 2023) | Prior-guided strand reconstruction, originally multi-view | Foundational, strong prior model reused everywhere since |
| **DiffLocks** (CVPR 2025) | Diffusion model generating 3D hair strands from a **single image** | Most directly matches your ask |
| **Im2Haircut** (ICCV 2025) | Single-view strand-based reconstruction combining a global hair prior + local optimization | Newest, strongest single-image strand result |
| **MonoHair** | High-fidelity hair from monocular **video** (not single image) | Fallback if you allow short video input |
| **hair-gs / CGHair** | Gaussian-splatting-based strand/card reconstruction | Better for render quality than for exporting clean game-ready strand data |

**Practical recommendation for a game pipeline:** don't try to get strand geometry directly into your face mesh's topology. Run hair as its own pass (DiffLocks/Im2Haircut-style), output strands or hair cards independently, and composite them onto the head mesh at the scalp boundary — this is also how every production game/film pipeline already treats hair, for good reason (simulation, LOD, and rendering all need hair to be separate from skin topology). This is also the stage most likely to need cloud GPU rental beyond Kaggle's free tier (Section 9).

---

## 7. Datasets

| Dataset | Content | Size | License | Use |
|---|---|---|---|---|
| **MICA unified dataset** (LYHM + FaceWarehouse + Stirling + others, unified to FLAME topology) | RGB + registered neutral 3D identity shape | ~2,315 identities | Mix of licenses from component datasets — verify each for commercial use | Training/fine-tuning the identity stage |
| **FaceScape** | 18,760 pore-level textured 3D face scans, 938 subjects × 20 expressions, topologically uniform, includes displacement maps | 938 subjects | **Non-commercial research only** — explicit "NO COMMERCIAL USE" clause; would need a direct license from the authors (Nanjing University) for a shipped game product | The best public reference for pore/wrinkle-level detail, and the dataset most directly used in the GAN-based facial detail papers cited in Section 5 — usable to *prototype and validate* your detail-stage architecture, but **do not train your shipped model's weights on it without clearing the license first** |
| **NoW Challenge benchmark** | Standardized single-image face reconstruction accuracy benchmark | Benchmark, small | Research/eval use | Use this to *measure* your pipeline's geometric accuracy against published numbers (Pixel3DMM, MICA, etc.) |
| **NPHM dataset** | Multi-view/scan-based data used to train NPHM's SDF identity/expression space | Research-scale | Research | If you go the NPHM route |
| **ICT-FaceKit** | Not a training dataset — a **topology + rig + blendshape standard** used across the games/VFX industry | — | **MIT license** | Your Stage 5 target topology; safe for commercial use |
| **USC-HairSalon** and similar synthetic hair strand datasets | Synthetic 3D hairstyles for strand-model training | Research-scale | Check per-release | Used by Neural Haircut/Im2Haircut-style pipelines |

**The licensing reality you need to plan around:** the single best public dataset for exactly the "pore-level, wrinkle-perfect" detail you're describing (FaceScape) is explicitly non-commercial. For a shipped, production, commercial game pipeline you have three real options, and you should decide this early because it changes your whole data plan:

1. **License FaceScape (or similar) commercially** — email the authors, most academic groups will negotiate a commercial license or at least clarify terms.
2. **Capture your own high-fidelity data** — even a modest DIY multi-camera or structured-light rig (or renting time on one) captured on consenting subjects with a proper release, gives you a small but fully-owned, commercially-clean detail dataset. This is what most serious game/VFX studios who care about IP actually do.
3. **Synthetic bootstrapping** — use permissively-licensed 3D head assets (or your own scans of a handful of people) plus procedural wrinkle/pore displacement (Substance/Houdini-style procedural skin detail generators) to synthesize a large, license-clean training set, then fine-tune on a small amount of real data for realism. Weaker fidelity ceiling than real scan data, but zero licensing risk. This is also the safest data source for the discriminator's "real" examples in Section 5 if you want to stay fully license-clean from day one.

---

## 8. Training & inference environments: Kaggle T4×2 (primary training) and RTX 4060 8GB (dev/inference)

### 8.1 What Kaggle's T4×2 accelerator actually gives you

Kaggle's free "GPU T4×2" notebook accelerator provisions **two NVIDIA Tesla T4 GPUs, 16GB VRAM each (32GB combined), alongside roughly 32GB of system RAM**. The two hard limits to plan around: a **~12-hour cap per individual session** (the notebook is killed at that point regardless of progress) and a **weekly quota of about 30 GPU-hours**, which resets weekly and can occasionally float higher depending on Kaggle's current demand/capacity — never assume more than the stated 30 as your planning baseline.

The practical implication for this project: **on Kaggle, VRAM is no longer your binding constraint for most of this pipeline** — 32GB combined is roughly 4× your local 4060's 8GB. The binding constraint becomes *wall-clock time inside the session/weekly caps*, which changes how you should structure training runs:

- **Use both T4s, don't waste one.** For Stage 1/2 regressor fine-tuning and the Stage 3 GAN/diffusion training (Section 5), use data-parallel training across both GPUs (PyTorch `DistributedDataParallel` launched via `torchrun` or Hugging Face `accelerate launch --multi_gpu`, both work fine in a Kaggle notebook) rather than leaving the second T4 idle. This roughly halves wall-clock per epoch, which is the difference between fitting a full training run in one 12-hour session versus needing three.
- **For anything that doesn't fit in 16GB alone**, shard across both T4s with DeepSpeed ZeRO-2/3 or PyTorch FSDP instead of plain data parallelism — this gives you an effective ~28–30GB of usable combined capacity (after overhead), which is enough headroom to, if you ever wanted to, attempt LoRA fine-tuning of something like Hunyuan3D-2mini that wouldn't fit on the 4060 alone. This is not required for the recommended face-specific pipeline (every stage in Sections 3–5 fits in 16GB alone), but it's there if a future stage needs it.
- **T4 is Turing architecture — no bf16 tensor-core support.** Use fp16 mixed precision with loss scaling (not bf16) for training, and keep numerically sensitive parts — discriminator logits, the R1 gradient penalty in Section 5.3 — in fp32 to avoid NaNs, which is a common failure mode for GAN training under fp16.
- **No run survives past 12 hours, so checkpoint-and-resume is mandatory, not optional.** Write a checkpoint to `/kaggle/working` every N steps (or every few minutes), and at the end of each session push it out as a Kaggle Dataset (or "Save Version" with data persisted); start the next session by attaching that dataset as an input and resuming from it. Avoid the "commit/batch run" execution mode for actual training runs — it re-runs the whole notebook top-to-bottom from scratch each time, which is the opposite of what you want; keep training in an interactive session you checkpoint out of manually.
- **Turn Internet on** in the notebook settings whenever you need to pull pretrained checkpoints (MICA/EMOCA/SMIRK weights, Hugging Face diffusion backbones) — this requires a phone-verified Kaggle account, which is worth doing once up front rather than discovering the restriction mid-session.
- **Budget math for this pipeline specifically:** the Stage 1/2 regressor fine-tuning (small CNN/ViT encoders, tens of millions of params) is a matter of a few hours per experiment — easily iterated on within a single week's 30-hour quota. The Stage 3 GAN (Section 5.4) trains in the tens-of-GPU-hours range at 256–512px UV-map resolution (far cheaper than a photoreal StyleGAN2 run), which comfortably fits inside 1–3 weeks of Kaggle quota rather than months. The hair pass (Section 6) is the one stage genuinely likely to need a rented cloud GPU beyond Kaggle's free tier — plan for that separately.

### 8.2 What fits on the local RTX 4060 8GB / 16GB RAM (dev iteration + inference)

- **Identity + expression regressors (MICA/DECA/EMOCA/SMIRK-class, tens of millions of params, CNN/ViT encoder → small parametric output):** comfortably trainable/fine-tunable on 8GB, including full fine-tuning of the encoder, not just LoRA. These are closer in size to an SDXL LoRA training job than to an LLM. Good for quick local iteration between Kaggle sessions.
- **The Stage 3 GAN (Section 5), once you have a base checkpoint from Kaggle:** cheap enough at 256–512px UV-map resolution to keep fine-tuning locally for fast iteration without needing another Kaggle session every time.
- **UV-space detail diffusion, if used (Section 5.5):** this is the same VRAM class as training a Stable Diffusion 1.5/SDXL LoRA — SD1.5 LoRAs run comfortably in 8GB; SDXL LoRAs are tight but workable at rank ≤32 with gradient checkpointing and fused backward pass.
- **Full 3D generative backbones (TRELLIS 2, Hunyuan3D full pipeline):** not trainable on 8GB by any current technique — even the *inference*-only VRAM floor (10–29GB) exceeds your card. Don't fine-tune these locally, and don't rely on a single Kaggle T4 (16GB) for them either.
- **Final inference (running the finished pipeline on new photos):** every stage in this pipeline is designed to run comfortably inside 8GB at inference time — that's the whole point of the parametric/staged design over a raw dense-voxel generator.

**Practical rule of thumb for this whole project:** if a stage's *output* is "a few hundred to a few thousand numbers" (shape/expression coefficients, a small latent) or "a modest UV-space image" (displacement/normal map), it trains on Kaggle's T4×2 comfortably and even on your 4060 workably. If a stage's output is "a dense 3D volume or full mesh generated from scratch," rent a cloud GPU for that stage's training run, then bring the *trained, ideally distilled* model back down to run inference on your 4060.

---

## 9. Making it faster / more feasible

1. **Don't train from scratch — fine-tune the published checkpoints.** MICA, EMOCA, SMIRK all ship pretrained weights. Fine-tuning them on your own curated identity set (or your own captured subjects) to sharpen accuracy is a matter of hours on Kaggle's T4×2, not weeks.
2. **Prioritize the GAN track (Section 5) over full diffusion-from-scratch for Stage 3.** It's the cheaper of the two to get to a working checkpoint given the Kaggle quota, and — per Section 5.1 — it's also closer to the actual published state of the art for facial detail specifically, so you're not trading quality for speed here, you're getting both.
3. **Use both Kaggle T4s via DDP, always.** Leaving the second GPU idle roughly doubles your effective wall-clock cost against the weekly quota for no reason.
4. **Treat the detail stage as image-space (UV-map) synthesis, not 3D volumetric synthesis.** Displacement/normal maps in UV space are just 2D images — this sidesteps the entire "3D diffusion needs 24GB+" problem and lets you use the mature, 8GB/16GB-friendly GAN and diffusion tooling ecosystems (StyleGAN2 training code, Kohya SS, diffusers) directly.
5. **Burst to a rented cloud GPU only for the hair stage.** An A100/H100 by the hour (RunPod, Lambda, Vast.ai) for a day or two to fine-tune the hair-strand model, then bring the resulting checkpoint home to run inference on the 4060. This is standard practice and far cheaper than buying bigger hardware or trying to force hair training into the Kaggle quota.
6. **Multi-image input beats any model upgrade.** If you can get 3–5 photos or a short orbit video per subject instead of one image, your macro-identity accuracy jumps more than any architecture change would buy you — worth building into the product from day one even if single-image is the advertised "minimum."
7. **Distill/quantize only at the end, for deployment — not during training.** Train the detail-stage GAN/diffusion model at reasonable precision on Kaggle, then export a distilled/step-reduced (LCM-style, for the diffusion half of the hybrid) or quantized version for fast local inference once it's already trained and validated.

---

## 10. Output pipeline for game production

Geometry alone isn't "production grade" until it's usable by an engine and an animator:

1. **Fixed target topology:** map your FLAME/NPHM output onto **ICT-FaceKit** topology (MIT-licensed, industry-standard, MetaHuman-compatible lineage) via a one-time trained or optimization-based retopology step.
2. **LODs:** generate 2–4 levels of detail via standard mesh decimation (this is off-the-shelf geometry processing, not ML).
3. **Blendshapes/rig:** ICT-FaceKit and FLAME both ship blendshape/pose bases — map your expression stage's output onto ARKit-52-style blendshapes for animator/engine compatibility (Unreal, Unity, MetaHuman pipelines all expect this).
4. **Export:** FBX or glTF with blendshapes, skeleton (jaw/neck/eyes), and clean UVs (even though you're not generating texture, game engines need valid UVs for artists to paint later).
5. **Hair composited separately** (see Section 6) as cards or guide strands, not baked into the face topology.

---

## 11. Evaluation

- **NoW Challenge** — the standard public benchmark for single-image face geometry accuracy; benchmark your pipeline against published MICA/EMOCA/Pixel3DMM numbers.
- **Point-to-surface / Chamfer distance** against held-out scans (if you have any captured data) — the standard metric used across MICA, FaceScape, Pixel3DMM papers, typically reported in mm.
- **Identity verification score** — run a face-recognition embedding (e.g., ArcFace via InsightFace) on renders of your output mesh vs. the input photo, as an automated "does it actually look like them" proxy metric during development, and reused directly as the identity-preservation loss in Section 5.3.
- **Surface-detail comparison against Meshy 7.1** — since Meshy publishes its own "Detail Richness" and geometry-alignment benchmark numbers (Section 4), it's worth building an equivalent surface-detail comparison against Meshy's output on the same input photos, specifically on the "surface details" axis where even Meshy 7 self-reports only 59.8%, since that's the number this pipeline is actually positioned to beat.

---

## 12. Suggested phased roadmap

| Phase | Goal | Hardware | Rough effort |
|---|---|---|---|
| 1 | Stand up inference-only: MICA + EMOCA/SMIRK pretrained, get a working image→FLAME mesh pipeline | 4060, no training yet | Days |
| 2 | Fine-tune identity + expression regressors on your own licensed/captured subject set | Kaggle T4×2 (DDP across both GPUs), checkpoint/resume across sessions | 1–2 weeks (well inside weekly quota) |
| 3 | Build the Stage 3 detail stage: start with the adversarial GAN design (Section 5.2–5.4), prototype/validate on FaceScape (non-commercial, validation only) or synthetic procedural detail | Kaggle T4×2 for GAN training; 4060 for fast local fine-tuning iteration once a base checkpoint exists | 2–4 weeks |
| 4 | Resolve data licensing for shipping (license FaceScape-equivalent data, or finalize your own capture pipeline / synthetic set) | — | Parallel, legal/business track |
| 5 | Retrain the detail stage on cleared data; add the diffusion half of the hybrid (Section 5.5) and fine-tune the adversarial loss on top for final sharpening; add the hair pass (DiffLocks/Im2Haircut-style, likely cloud-rented) | Kaggle T4×2 for the hybrid detail stage; rented cloud GPU for hair; 4060 for inference | 2–4 weeks |
| 6 | Retopology/export pipeline to ICT-FaceKit topology, LODs, blendshapes, engine integration | 4060 | 2–3 weeks |
| 7 | Evaluate on NoW + identity-verification metric + a direct surface-detail comparison against Meshy 7.1 on the same inputs (Section 11), iterate | 4060 | Ongoing |
| 8 | Track and swap in NPHM/Pix2NPHM and Pixel3DMM once tooling matures, for the accuracy ceiling above FLAME | Kaggle T4×2 / 4060 | Future upgrade |

---

## 13. Key risks / honest limitations

- **Single-image "identical from every angle" is not achievable** — it's an information-theoretic limit, not a modeling weakness. Plan for multi-image input if true production fidelity is the goal.
- **Pore-level and hair-strand fidelity are the current research frontier**, not solved problems — budget schedule risk for these two specifically, and consider shipping v1 without strand-accurate hair (use good hair cards) while the strand research matures.
- **Data licensing is a legal blocker, not a technical one, for the highest-quality public dataset (FaceScape).** Resolve this before building your production data pipeline (or your GAN discriminator's "real" examples) around it.
- **General 3D generators (Hunyuan3D/TRELLIS) are a dead end for this specific goal** on this hardware and are not face-specialized — don't spend early effort trying to fine-tune them for faces.
- **GANs are notoriously less stable to train than regressors or diffusion models.** Budget for mode-collapse and training-instability debugging time in Phase 3 — the R1 penalty and the multiple supervised loss terms in Section 5.3 are there specifically to control this, don't skip them to save time.
- **Meshy's 81.0%/79.7%/59.8% benchmark numbers are vendor-published, not third-party audited** — a reasonable target to aim past, not a certified ceiling.

---

## References (starting points)

- MICA — https://github.com/Zielon/MICA
- FLAME-Universe (hub for FLAME code/data/papers) — https://github.com/TimoBolkart/FLAME-Universe
- SMIRK — https://github.com/georgeretsi/smirk
- Pixel3DMM (ICLR 2026) — https://openreview.net/forum?id=UmOdd5KQ8K
- NPHM — https://simongiebenhain.github.io/NPHM/
- FaceScape (license terms) — https://nju-3dv.github.io/projects/FaceScape/
- ICT-FaceKit (MIT) — https://github.com/USC-ICT/ICT-FaceKit
- Im2Haircut — https://github.com/Vanessik/Im2Haircut
- DiffLocks — search "DiffLocks CVPR 2025 3D hair diffusion"
- NoW Challenge benchmark — search "NoW Challenge RingNet benchmark"
- InsightFace (detection/landmarking/recognition-embedding tooling) — https://github.com/deepinsight/insightface
- Chen et al., "Photo-Realistic Facial Details Synthesis from Single Image" (ICCV 2019) — https://arxiv.org/abs/1903.10873
- Abrevaya et al., "A Decoupled 3D Facial Shape Model by Adversarial Training" (ICCV 2019) — https://openaccess.thecvf.com/content_ICCV_2019/papers/Abrevaya_A_Decoupled_3D_Facial_Shape_Model_by_Adversarial_Training_ICCV_2019_paper.pdf
- "Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation" — https://arxiv.org/pdf/2207.09019
- "Detail 3D Face Reconstruction Based on 3DMM and Displacement Map" (CGAN + FaceScape) — ResearchGate, search title
- Meshy 7 — image-to-3D alignment engine — https://www.meshy.ai/blog/meshy-7-image-to-3d-geometry-alignment
- Meshy 7.1 — detail richness benchmark — https://www.meshy.ai/blog/meshy-7-1-launch
- Kaggle — Efficient GPU Usage docs — https://www.kaggle.com/docs/efficient-gpu-usage