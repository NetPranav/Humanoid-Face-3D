# Adversarial Detail Synthesis — Research Reference

> ⚠️ **ARCHITECTURAL STATUS NOTICE:**
> The original monolithic conditional GAN implementation described in this document was evaluated in Research 1 and failed to produce film-grade pore fidelity due to dataset frequency limits and wireframe leakage.
> - **Failure Postmortem:** See [DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md).
> - **Wireframe Artifact Breakdown:** See [DOCS/FAILED/RESEARCH_1/02_wireframe_leakage_and_smoothing_failure.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/02_wireframe_leakage_and_smoothing_failure.md).
> - **Current Production Architecture:** The pipeline has transitioned to the 4-Tier Hybrid Engine (Photo-Derived Meso Wrinkles + 4K Anatomical Pore Synthesis + Cycles Random Walk SSS) detailed in [DOCS/02_Metahuman_Film_Grade_Synthesis.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/02_Metahuman_Film_Grade_Synthesis.md).
>
> This document remains preserved for architectural reference regarding the generator/discriminator loss formulation.

---

## Table of Contents

1. [Why GANs for Facial Detail (Not Just Regression)](#1-why-gans-for-facial-detail)
2. [The Papers This Design Is Based On](#2-the-papers-this-design-is-based-on)
3. [Generator Architecture](#3-generator-architecture)
4. [Discriminator Architecture](#4-discriminator-architecture)
5. [Full Loss Recipe — Six Terms, Carefully Weighted](#5-full-loss-recipe)
6. [Training Stability — What Goes Wrong and How to Fix It](#6-training-stability)
7. [The GAN-Diffusion Hybrid](#7-the-gan-diffusion-hybrid)
8. [UV Displacement Dataset Preprocessing](#8-uv-displacement-dataset-preprocessing)

---

## 1. Why GANs for Facial Detail

Stage 1 (MICA) produces a face mesh with correct overall proportions. Stage 2 (EMOCA/SMIRK) adds expression and coarse detail. But both stages are **regressors** — they predict a single "best guess" output for a given input. Regressors trained with L1/L2 loss tend to average over uncertainty, producing smooth, blurry output. This is fine for macro proportions but terrible for fine detail: wrinkles, pore texture, and eyebrow depth are high-frequency features that get averaged away.

**The adversarial approach fixes this.** Instead of predicting the "average plausible detail," a GAN generator learns to produce detail that is **indistinguishable from real scan data** as judged by a discriminator. The discriminator forces the generator to commit to specific, sharp, high-frequency features rather than hedging with smooth averages.

This isn't a novel idea for this exact problem — it's the established approach in the published literature for facial displacement map synthesis. The design in this pipeline is a direct adaptation of several published papers (Section 2).

---

## 2. The Papers This Design Is Based On

### Chen et al., "Photo-Realistic Facial Details Synthesis from Single Image" (ICCV 2019)
**[arxiv.org/abs/1903.10873](https://arxiv.org/abs/1903.10873)**

The closest direct blueprint. Trains a **conditional GAN** where:
- G takes an image patch + noise vector → predicts a wrinkle/pore displacement map
- D judges real scan-derived displacement maps vs. generated ones
- Combined loss: cGAN adversarial + L1 reconstruction, weighted ~100:1 in favor of L1 early in training
- Training data: 706 high-precision 3D face scans + 163,000 in-the-wild images

**Key takeaway:** The 100:1 L1-to-adversarial weighting early in training is critical. Without it, the GAN drifts — the generator learns to produce plausible-looking wrinkles that don't correspond to the actual input face. The L1 term anchors it to the ground truth; the adversarial term sharpens the high-frequency detail.

### Abrevaya et al., "A Decoupled 3D Facial Shape Model by Adversarial Training" (ICCV 2019)
**[ICCV 2019 proceedings](https://openaccess.thecvf.com/content_ICCV_2019/papers/Abrevaya_A_Decoupled_3D_Facial_Shape_Model_by_Adversarial_Training_ICCV_2019_paper.pdf)**

Key innovation: **auxiliary identity and expression classifier heads on the discriminator** (AC-GAN style). The discriminator doesn't just decide real/fake — it also has to correctly predict *whose face* and *what expression*. This pushes G toward detail that is:
- **Person-specific** — wrinkle patterns that match the specific input identity, not generic
- **Expression-specific** — detail that corresponds to the actual expression (crow's feet for squinting, nasolabial folds for smiling)

This is directly relevant to the requirement "must match this specific person's eyebrow depth/wrinkles."

### "Structure-aware Editable Morphable Model" (arXiv 2207.09019)
The most modern reference architecture:
- **StyleGAN2-style generator/discriminator backbone** — proven stable architecture
- **R1 gradient penalty** on D for training stability
- **256×256 displacement map resolution** (found sufficient for wrinkle-level detail; this pipeline defaults to 512×512 given the additional VRAM headroom on Kaggle T4s)
- Normalizes displacement/distance-field value ranges so D doesn't collapse onto a single channel

### "Detail 3D Face Reconstruction Based on 3DMM and Displacement Map"
Trains this exact setup on **FaceScape data**: conditional GAN synthesizes the displacement map applied back onto a coarse 3DMM reconstruction. Directly validates that FaceScape's scan quality is sufficient for this approach.

---

## 3. Generator Architecture

Two viable architectures; either works, with different tradeoffs:

### Option A: U-Net Encoder-Decoder (recommended for v1)

```
Input (concatenated in channel dimension):
├── Coarse UV position map (3 channels)
├── Coarse UV normal map (3 channels)
├── Face crop features (N channels, from frozen ResNet/ViT, projected to UV space)
└── Identity/expression codes (tiled to spatial dims, or injected via AdaIN)

       ┌──────────────────────────────────┐
       │     U-Net Encoder (downsampling)  │
       │  512→256→128→64→32 spatial res    │
       │  Conv-BN-LeakyReLU blocks         │
       └──────────┬───────────────────────┘
                  │ bottleneck
       ┌──────────▼───────────────────────┐
       │     U-Net Decoder (upsampling)    │
       │  32→64→128→256→512 spatial res    │
       │  ConvTranspose/Upsample+Conv      │
       │  Skip connections from encoder    │
       └──────────┬───────────────────────┘
                  │
       Output: residual displacement map (1 or 3 channels, 512×512)
```

**Why U-Net:** Skip connections preserve spatial correspondence between the input conditioning (coarse position/normal maps) and the output detail, which is critical — the wrinkle at coordinate (u, v) in the output must correspond to the same facial location as coordinate (u, v) in the coarse map. Without skip connections, the generator has to re-learn this spatial alignment from scratch.

### Option B: StyleGAN2-style Synthesis Network

Uses AdaIN (Adaptive Instance Normalization) to inject identity/expression codes at each resolution level. Better for unconditional diversity (different wrinkle patterns for the same identity), but less tightly conditioned on the coarse geometry input. More appropriate if you want to *sample* different plausible detail maps rather than *reconstruct* a specific one.

### Key design choices (both options)

- **Output is a residual, not absolute geometry.** The displacement map is *added to* the coarse FLAME mesh — the generator never has to reinvent macro identity, only the fine layer on top of it.
- **AdaIN or style injection** from identity/expression codes ensures detail is person-specific and expression-specific, not generic wrinkles.
- **Resolution: 512×512** — one 16GB T4 comfortably handles this for a model of this size (~20–30M params for G).

### Multi-view conditioning via cross-attention

> [!IMPORTANT]
> **Do NOT concatenate per-view features along the channel dimension.** This blows up the generator's input size proportionally to N views and creates a fixed-N architecture. Instead, use **cross-attention**:

```python
# In generator.py — multi-view feature aggregation
class MultiViewAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
    
    def forward(self, query_features, per_view_features):
        """
        query_features: (B, HW, C) — UV-space query positions
        per_view_features: (B, N, HW, C) — features from each input view
        """
        # Reshape per_view_features to (B, N*HW, C) as key/value
        B, N, HW, C = per_view_features.shape
        kv = per_view_features.reshape(B, N * HW, C)
        
        # Cross-attend: UV-space queries attend to all views
        out, _ = self.cross_attn(query_features, kv, kv)
        return out
```

This is one function in PyTorch and handles variable N views naturally. The generator's bottleneck receives the cross-attended features, so it gets information from all input views without architectural changes per view count. Specify this now — it affects `generator.py`'s architecture.

---

## 4. Discriminator Architecture

### PatchGAN (recommended)

Instead of one real/fake logit for the entire image, PatchGAN produces a **grid of real/fake logits**, each judging a local patch of the displacement map. This:
- Forces the generator to produce convincing detail at every location, not just globally
- Is computationally cheaper than a full-image discriminator
- Works better for high-resolution output (512×512) where global coherence is less important than local detail quality

```
Input: UV displacement map (1 or 3 channels, 512×512)
  │
  ▼
Conv-LeakyReLU blocks (70×70 effective receptive field is standard)
  │
  ▼
Output: grid of real/fake logits (e.g., 30×30 for a 512×512 input)
```

### Auxiliary heads (from Abrevaya et al.)

Added on top of the PatchGAN's penultimate feature map:

1. **Identity head:** Linear layer → N_identities classes. Cross-entropy loss. Must correctly predict whose face this displacement map belongs to.
2. **Expression head:** Linear layer → N_expressions classes. Cross-entropy loss. Must correctly predict what expression.

These heads are trained on both real and generated samples. The generator is penalized not just for being detected as fake, but for producing detail that gets misclassified (wrong person or wrong expression).

---

## 5. Full Loss Recipe

**Critical rule: never train on the adversarial term alone.** Pure adversarial training drifts — the generator produces plausible-looking but identity-incorrect detail. Every paper cited above uses the adversarial loss *on top of* several supervised terms.

### Term 1: Adversarial loss

```python
# Non-saturating GAN loss (standard)
D_loss = -E[log(D(real))] - E[log(1 - D(G(z)))]
G_loss = -E[log(D(G(z)))]

# Alternative: Hinge loss (sometimes more stable)
D_loss = E[max(0, 1 - D(real))] + E[max(0, 1 + D(G(z)))]
G_loss = -E[D(G(z))]
```

### Term 2: R1 gradient penalty (on D only)

```python
# StyleGAN2-style R1 regularization
r1_penalty = grad(D(real), real).pow(2).sum()
D_loss += (gamma / 2) * r1_penalty  # gamma ≈ 10
```

**Why this matters:** Without R1, the discriminator's gradients can explode, causing training collapse. R1 penalizes the discriminator for having large gradients on real data, keeping it smooth. This is the single most important stability technique for GAN training — do not skip it.

**Implementation note for fp16 training:** The R1 gradient computation must be done in **fp32**, not fp16. The gradient of D w.r.t. its input involves second-order derivatives that underflow in fp16 and produce NaN. This is a known failure mode on T4 GPUs with mixed-precision training.

### Term 3: Reconstruction loss (L1/L2) — MASKED

```python
# reconstruction loss MUST be masked to exclude invalid UV regions
recon_loss = L1(G(condition) * validity_mask, ground_truth_displacement * validity_mask)
recon_loss = recon_loss.sum() / validity_mask.sum()  # normalize by valid pixel count
```

> [!WARNING]
> **Do NOT compute reconstruction loss on invalid UV pixels.** When displacement maps are rasterized from 3D scans, regions of the UV layout corresponding to the back of the head, inside the nostrils, or behind the ears have no valid scan data. Those pixels are zeroed or interpolated, NOT ground truth. Without masking:
> - The GAN learns to smooth over invalid regions in a way that **leaks into valid regions** at UV boundaries
> - The discriminator collapses onto detecting the fake/smooth invalid regions rather than evaluating actual detail quality
>
> Use the validity mask from `build_uv_displacement_dataset.py` (see Section 8).

**Weighting schedule:** Start with λ_recon ≈ 100 (reconstruction dominates), anneal down to λ_recon ≈ 10 over training. Early high weighting prevents the generator from hallucinating random detail; later reduction lets the adversarial term sharpen high-frequency output.

### Term 4: Identity preservation loss

```python
render = diff_render(coarse_mesh + G_output, camera)
input_emb = frozen_arcface(input_photo)      # 512-dim
output_emb = frozen_arcface(render)           # 512-dim
id_loss = 1 - cosine_similarity(input_emb, output_emb)
```

Directly optimizes for "still looks like this specific person after detail is applied." Uses the same frozen ArcFace network used for evaluation.

### Term 5: Photometric consistency loss

```python
for light_dir in [front, left, right, top, bottom]:
    shaded_render = shade(coarse_mesh + G_output, light_dir)
    shaded_input = approximate_shading(input_photo, light_dir)
    photo_loss += L1(shaded_render, shaded_input)
```

Prevents the generator from producing detail that looks right under flat lighting but shades wrong under directional light. This catches artifacts like inverted displacement (bumps where there should be dents) that L1 alone doesn't penalize.

### Term 6: Auxiliary classification losses (on D's heads)

```python
id_class_loss = CrossEntropy(D_id_head(real), real_id_label)
                + CrossEntropy(D_id_head(fake), fake_id_label)
expr_class_loss = CrossEntropy(D_expr_head(real), real_expr_label)
                  + CrossEntropy(D_expr_head(fake), fake_expr_label)
```

Applied to both real and generated samples. The generator receives gradient from D's classification heads — it's penalized for producing detail that the discriminator classifies as the wrong person or wrong expression.

### Combined loss

```python
G_total = λ_adv * G_adv + λ_recon * recon + λ_id * id_loss + λ_photo * photo_loss + λ_aux * aux_class_loss
D_total = D_adv + (γ/2) * r1 + λ_aux * aux_class_loss_on_D
```

Typical starting weights: `λ_adv=1, λ_recon=100 (annealed), λ_id=5, λ_photo=1, λ_aux=1, γ=10`

---

## 6. Training Stability

### Common failure modes and fixes

| Failure | Symptom | Fix |
|---|---|---|
| **Mode collapse** | All outputs look the same regardless of input | Increase λ_recon, check that conditioning input is actually reaching G's bottleneck (not dropped by skip connections), reduce G learning rate |
| **Discriminator saturation** | D accuracy → 100%, D loss → 0, G stops learning | Reduce D learning rate, increase R1 penalty γ, add noise to D's input (instance noise), train G more steps per D step |
| **NaN gradients** | Loss becomes NaN mid-training | Almost always fp16 underflow in R1 penalty or D logits — cast these to fp32. Also check displacement value range normalization. |
| **Checkerboard artifacts** | Regular grid pattern in output displacement map | Use bilinear upsampling + conv instead of ConvTranspose2d in the generator's decoder |
| **Discriminator oscillation** | D loss oscillates wildly | Lower D learning rate, increase R1 penalty, use spectral normalization on D |

### Training hyperparameters (starting point)

```yaml
generator:
  lr: 2e-4
  betas: [0.0, 0.99]  # Adam with β1=0 (standard for StyleGAN2-class)
  
discriminator:
  lr: 2e-4
  betas: [0.0, 0.99]
  r1_gamma: 10.0

training:
  batch_size: 16-32  # per GPU, at 512×512 on a 16GB T4
  g_steps_per_d_step: 1  # increase to 2 if D saturates
  ema_decay: 0.999  # exponential moving average of G weights for inference
  recon_lambda_start: 100.0
  recon_lambda_end: 10.0
  recon_anneal_steps: 50000
  fp16: true
  fp32_ops: [r1_penalty, d_logits]  # critical for T4
```

### EMA (Exponential Moving Average) of generator weights

Standard practice: maintain a running average of G's weights during training, and use the averaged weights (not the raw training weights) for inference. This smooths out training oscillations and consistently produces better output than the last training checkpoint.

```python
ema_g = copy.deepcopy(generator)
for p_ema, p_train in zip(ema_g.parameters(), generator.parameters()):
    p_ema.data.mul_(0.999).add_(p_train.data, alpha=0.001)
# Use ema_g for inference, generator for training
```

> [!CAUTION]
> **EMA weights are the production weights.** The `upload_to_kaggle_models.py` script MUST push the EMA checkpoint (`ema_generator.pt`), not the raw training checkpoint (`generator.pt`). If the upload directory does not contain `ema_generator.pt`, the upload should **fail with an error** rather than silently pushing inferior training weights. Add an explicit check:

```python
# In upload_to_kaggle_models.py
ema_path = os.path.join(checkpoint_dir, "ema_generator.pt")
if not os.path.exists(ema_path):
    raise FileNotFoundError(
        f"EMA checkpoint not found at {ema_path}. "
        "Do NOT upload raw training weights — they produce noticeably worse output. "
        "Run EMA averaging first."
    )
```

---

## 7. The GAN-Diffusion Hybrid

### Why hybrid?

- **GAN alone:** Fast inference (single forward pass), sharp high-frequency output, but less stable training and can mode-collapse
- **Diffusion alone:** Very stable training, good diversity, but slower inference (many denoising steps) and slightly softer high-frequency output (the "diffusion blur" problem)
- **Hybrid:** Train the diffusion model first (stable), then fine-tune it with an adversarial loss from the GAN discriminator (sharpen)

### How it works

1. **Phase A — Train UV-space diffusion model standalone.** Standard denoising loss (predict the noise added at each timestep). Architecture: similar to Stable Diffusion's U-Net but operating on 1–3 channel displacement maps (not 4-channel latent RGB). Conditioning: coarse UV normal/position map + face features + identity/expression codes. This trains stably and produces reasonable but slightly soft displacement maps.

2. **Phase B — Add adversarial loss.** Take the trained discriminator from the GAN stage (or train a new one). Run the diffusion model to denoise a sample, feed the denoised output to D, and backprop D's gradient through the denoising step into the diffusion model's parameters. This is essentially using D as a "sharpness critic" on the diffusion model's output.

3. **Phase C (optional) — Distill to fewer steps.** The adversarial fine-tuning often allows the diffusion model to produce sharp output in fewer denoising steps (similar to LCM/consistency distillation), because the adversarial loss penalizes the soft output that early-stopped denoising would otherwise produce.

### Training cost estimate

- Phase A (diffusion backbone): ~20–30 GPU-hours on T4×2
- Phase B (adversarial fine-tuning): ~10 GPU-hours on T4×2
- Phase C (step distillation): ~5 GPU-hours, optional

---

## 8. UV Displacement Dataset Preprocessing

### The problem

FaceScape (and similar scan datasets) provide raw 3D meshes — `.obj` files with millions of vertices in arbitrary topology. The GAN operates on **UV-space displacement maps** — 2D images where each pixel encodes how far the real surface is from the coarse FLAME fit at that UV coordinate. Getting from raw scans to these paired images is nontrivial geometry processing.

### The pipeline (implemented in `scripts/build_uv_displacement_dataset.py`)

```
Raw FaceScape .obj scan
    │
    ▼
1. Load scan + its FLAME registration (FaceScape provides coefficient files)
    │
    ▼
2. Compute coarse base mesh: run FLAME model at the registered (β, ψ, θ)
    │
    ▼
3. Per-vertex displacement: displacement = scan_vertex - coarse_vertex
   (can be computed in world space or tangent space; tangent space is more
    useful for displacement mapping but requires computing the tangent frame)
    │
    ▼
4. Rasterize displacement into UV space at 512×512:
   For each pixel in the UV map, find which triangle it falls in
   (barycentric lookup using FLAME's UV layout), interpolate the
   per-vertex displacement → per-pixel displacement value
    │
    ▼
5. Also rasterize coarse position map and normal map into same UV space
   (these become the conditioning inputs for the generator)
    │
    ▼
6. Save as paired files:
   {subject}_{expression}_disp.png   (displacement map — G's target)
   {subject}_{expression}_pos.png    (position map — G's input)
   {subject}_{expression}_norm.png   (normal map — G's input)
```

### Verification (before any GAN training)

1. **Round-trip test:** Load a displacement map, add it back to the coarse mesh, compare against the original scan. Chamfer distance should be <0.1mm — if it's larger, the UV rasterization is losing information.
2. **Visual inspection:** Display 10–20 displacement maps. They should show visible wrinkle/pore patterns, with no:
   - UV seam artifacts (discontinuities along UV boundary edges)
   - NaN or infinite values
   - Unexpectedly large values (displacement should be sub-millimeter)
3. **Value range check:** Mean displacement should be near 0 (the coarse mesh is already a good fit), standard deviation should be small (0.1–0.5mm typically).

### Displacement map normalization strategy

> [!IMPORTANT]
> **Raw displacement values have a long tail — a few large outlier vertices throw off the range.** This causes the discriminator to collapse onto the magnitude channel rather than evaluating spatial patterns.

**The normalization pipeline (applied in `build_uv_displacement_dataset.py`):**

```python
# 1. Compute 99th percentile of absolute displacement across the ENTIRE dataset
all_displacements = load_all_displacement_maps()  # list of arrays
abs_values = np.concatenate([np.abs(d).ravel() for d in all_displacements])
p99 = np.percentile(abs_values, 99)

# 2. Clip to that range
for d in all_displacements:
    d_clipped = np.clip(d, -p99, p99)
    
    # 3. Normalize to [-1, 1] for the GAN
    d_normalized = d_clipped / p99
    
    # Save both: normalized (for GAN training) and p99 value (for denormalization at inference)
```

**Critical:** Save the `p99` value alongside the dataset. At inference time, the GAN's output is in [-1, 1] and must be denormalized back to real displacement units by multiplying by `p99`. If you lose this value, the output scale is wrong.

### Validity mask for occluded UV regions

> [!IMPORTANT]
> **Parts of the UV layout have no valid scan data.** When rasterizing a displacement map from a 3D scan, UV pixels corresponding to the back of the head, inside the nostrils, or behind the ears have no ground-truth displacement — they're either zeroed, NaN, or filled with interpolated garbage.

**Output an explicit validity mask** as part of the preprocessing pipeline:

```python
# In build_uv_displacement_dataset.py, step 4
validity_mask = np.zeros((512, 512), dtype=np.float32)

for u, v in uv_pixels:
    triangle_idx = find_containing_triangle(u, v, flame_uv_layout)
    if triangle_idx is not None:
        # Check that ALL three vertices of this triangle have valid scan data
        # (i.e., are on the front face of the scan, not self-occluded)
        if all_vertices_valid(triangle_idx, scan_mesh):
            validity_mask[v, u] = 1.0

# Save as: {subject}_{expression}_mask.png
```

**This mask is used in:**
- `reconstruction_loss()` — only computed on valid pixels (see Section 5, Term 3)
- Visual inspection — overlay on displacement maps to verify coverage
- The discriminator should only see valid-region crops (or the valid/invalid boundary becomes the easiest real/fake signal)

### Why this is its own script, not inside the dataset class

This preprocessing is **run once** on the entire scan dataset (takes hours, CPU-heavy), then the output is uploaded as a Kaggle Dataset and reused for every training run. The dataset class (`data.py`) loads only the preprocessed paired images — it should never touch raw `.obj` files. Separating them means:
- You can verify the preprocessing output independently before committing GPU hours to GAN training
- Multiple training experiments reuse the same preprocessed data without re-running geometry processing
- The preprocessing can run on a CPU-only Kaggle session (no GPU quota cost)

---

## Implementation Decisions Log

Update this section as you make real implementation choices during coding.

| Date | Decision | Value | Reason |
|---|---|---|---|
| Pre-impl | Generator architecture | U-Net encoder-decoder (Option A) | Skip connections preserve UV-space spatial correspondence. StyleGAN2-style is better for sampling diversity but worse for reconstruction fidelity. |
| Pre-impl | Multi-view conditioning | Cross-attention over per-view features | Channel-concatenation creates fixed-N architecture and blows up input size. Cross-attention handles variable N views naturally. |
| Pre-impl | Displacement normalization | 99th percentile clip → [-1, 1] | Raw values have long-tail outliers that cause D to collapse on magnitude rather than spatial patterns. |
| Pre-impl | Reconstruction loss masking | Validity mask applied to L1 loss | Unmasked loss on invalid UV regions causes smoothing artifacts that leak into valid regions. |
| Pre-impl | EMA enforcement | Upload script requires `ema_generator.pt` | Raw training weights produce noticeably worse output than EMA-averaged weights. |
| Pre-impl | Renderer for UV rasterization | **nvdiffrast** | Native UV-space rasterization, lighter install than pytorch3d, working pip wheel for Kaggle/Linux. |
| | | | |
