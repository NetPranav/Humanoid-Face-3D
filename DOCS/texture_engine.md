# PBR Texture Engine — Technical Documentation

> **Stages 6, 7, 8** of the Humanoid-Face-3D pipeline.  
> Transforms raw input photographs into a complete set of physically based rendering (PBR) texture maps for film-grade 3D head assets.

---

## 1. Architecture Overview

```
Input Photos (3–5)          Neutral FLAME Mesh + Displacement/Normal Maps
       │                                    │
       ▼                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 6: Multi-View UV Texture Projection                       │
│  • Backproject photos → UV space via per-view camera matrices   │
│  • Angle-weighted cosine blending: w = (n·v)^γ · det_score      │
│  • Z-buffer occlusion testing                                    │
│  • Output: projected_raw.png (2048²) + projection_mask.png       │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 7A: UV Inpainting                                          │
│  • Procedural Gaussian dilation (50 iterations, 5×5 kernel)      │
│  • Fills: back of ears, under chin, scalp, occluded areas        │
│  • Preserves original pixel values exactly                        │
│                                                                   │
│ Stage 7B: AI Delighting                                          │
│  • DelightUNet (6-level encoder-decoder, InstanceNorm)            │
│  • Input: 6ch (RGB + surface normals)                             │
│  • Output: 3ch clean diffuse albedo (no lighting artifacts)       │
│  • Supports: Pre-trained DECA / custom fine-tuned weights         │
│  • Fallback: Inverse gamma + luminance normalisation              │
│  • Output: albedo_diffuse.png (2048² sRGB)                        │
│            albedo_linear.exr (2048² linear for UE5)               │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 8: PBR Material Stack Derivation                           │
│                                                                   │
│ 8A Roughness:                                                     │
│  • Anatomical zone classification (T-zone, cheeks, lips, eyes)    │
│  • Displacement-coupled micro-variation: R += 0.08 · ∇²D          │
│  • Output: roughness_map.png (2048² grayscale)                    │
│                                                                   │
│ 8B Cavity / Ambient Occlusion:                                    │
│  • Laplacian of displacement field                                │
│  • Cavity(u,v) = clamp(0.5 + k·∇²D(u,v), 0, 1)                  │
│  • Output: cavity_ao_map.png (2048² grayscale)                    │
│                                                                   │
│ 8C SSS Thickness:                                                 │
│  • Opposing-normal vertex distance heuristic                      │
│  • Thin areas (ears, nostrils) → high SSS value                   │
│  • Output: sss_thickness_map.png (2048² grayscale)                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage 6: Multi-View UV Texture Projection

### Camera Model

Each input photo's camera is estimated using `cv2.solvePnP` (EPnP algorithm) from 5 canonical FLAME 3D landmarks and their corresponding 2D detections from Stage 0.

The projection matrix $P_i \in \mathbb{R}^{3 \times 4}$ maps 3D world coordinates to 2D pixel coordinates:

$$\begin{bmatrix} u \\ v \\ w \end{bmatrix} = P_i \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}, \quad x_{px} = u/w, \quad y_{px} = v/w$$

### View Blending

For UV texels visible from multiple cameras, colours are blended using:

$$\text{Color}(u,v) = \frac{\sum_i w_i \cdot C_i(u,v)}{\sum_i w_i}$$

where the per-vertex per-view weight is:

$$w_i = \max(0, \vec{n}_{\text{surface}} \cdot \hat{v}_{\text{camera}_i})^{\gamma} \cdot \text{det\_score}_i \cdot \text{visible}_i$$

- $\gamma = 2.0$ (configurable) sharpens blending to prefer head-on views.
- $\text{visible}_i$ is a binary z-buffer occlusion flag.
- $\text{det\_score}_i$ is the face detection confidence from Stage 0.

### Visibility Testing

A per-vertex z-buffer is used: for each vertex, project to 2D pixel space, record the minimum depth at each pixel, and mark a vertex visible if its depth is within $\epsilon = 0.005$ of the minimum.

### Source Files

| File | Purpose |
|------|---------|
| `src/stage6_texture/__init__.py` | Package exports |
| `src/stage6_texture/projector.py` | `MultiViewTextureProjector`, `estimate_camera_projection_matrix` |
| `tests/test_stage6_texture.py` | 8 unit tests |

---

## 3. Stage 7: AI Delighting + UV Inpainting

### 7A: Delighting U-Net Architecture

```
Input: (B, 6, H, W) — [RGB, Normal]
    │
    ├── Encoder: 6 levels (64 → 128 → 256 → 512 → 512 → 512)
    │   AvgPool2d(2) between levels, InstanceNorm + LeakyReLU(0.2)
    │
    ├── Bottleneck: 512ch
    │
    ├── Decoder: 6 levels with skip connections
    │   Bilinear upsampling + concat + conv
    │
    └── Output: Sigmoid → (B, 3, H, W) albedo in [0, 1]
```

**Weight Sources:**
- **DECA (default):** `models_cache/deca/deca_model.tar` — automatic key remapping strips `E_albedo.`, `module.`, `albedo_decoder.` prefixes.
- **Custom:** Any checkpoint at `stage7.delight_checkpoint` config path.
- **Fallback:** Inverse sRGB gamma + luminance normalisation (no weights needed).

### 7B: Procedural UV Inpainting

Iterative Gaussian dilation:
1. Compute weighted blur of valid pixels (mask × texture).
2. Normalise by blurred mask to get proper colour values.
3. Fill newly-reachable pixels (where blur reached but original mask was 0).
4. Repeat for 50 iterations.

**Guarantees:**
- Original pixel values preserved exactly where mask was valid.
- Fills holes up to ~100 pixels deep.

### Source Files

| File | Purpose |
|------|---------|
| `src/stage7_delight/__init__.py` | Package exports |
| `src/stage7_delight/delight_net.py` | `DelightUNet`, `UVInpainter`, `DelightingPipeline` |
| `tests/test_stage7_delight.py` | 10 unit tests |

---

## 4. Stage 8: PBR Material Stack

### 8A: Roughness Map

Anatomical zone roughness values (GGX metallic roughness):

| Zone | Roughness | Rationale |
|------|-----------|-----------|
| T-zone (forehead, nose, chin) | 0.28–0.38 | Oily sebaceous glands |
| Cheeks | 0.50–0.65 | Drier skin |
| Lips | 0.18–0.25 | Glossy mucosa |
| Periorbital (eye area) | 0.42–0.52 | Thin, delicate skin |
| Default | 0.45 | Baseline |

Displacement-coupled micro-variation:
$$R(u,v) = R_{\text{zone}}(u,v) + 0.08 \cdot \text{normalize}(\nabla^2 D(u,v))$$

### 8B: Cavity / Ambient Occlusion

$$\text{Cavity}(u,v) = \text{clamp}\big(0.5 + 0.35 \cdot \nabla^2 D(u,v),\ 0,\ 1\big)$$

- 0.5 = flat surface (neutral)
- < 0.5 = concavity (pore pit, wrinkle valley → darkened)
- > 0.5 = convexity (ridge → brightened)

### 8C: SSS Thickness

For each vertex $v_i$:
1. Find all vertices with opposing normals ($\vec{n}_i \cdot \vec{n}_j < -0.3$).
2. Compute minimum distance to any opposing vertex.
3. Normalise to [0, 1] and invert (thin → high SSS value).

Results are rasterised into UV space and Gaussian-smoothed.

### Source Files

| File | Purpose |
|------|---------|
| `src/stage8_pbr/__init__.py` | Package exports |
| `src/stage8_pbr/material_stack.py` | `RoughnessMapGenerator`, `CavityMapGenerator`, `SSSThicknessGenerator`, `PBRMaterialStack` |
| `tests/test_stage8_pbr.py` | 9 unit tests |

---

## 5. VRAM Budget (Inference)

| Component | VRAM (FP16) | Notes |
|-----------|-------------|-------|
| Stage 1: MICA | ~1.2 GB | Released after β computed |
| Stage 3: Detail GAN | ~1.5 GB | Released after displacement maps |
| Stage 6: UV Projection | ~200 MB (CPU) | No GPU |
| Stage 7: Delight U-Net | ~1.8 GB | FP16 at 2048² |
| Stage 8: PBR Stack | ~300 MB (CPU) | No GPU |
| **Peak Concurrent** | **~3.5 GB** | Sequential, memory released between stages |

---

## 6. Output File Specification

After a complete pipeline run, the `textures/` subdirectory contains:

```
textures/
├── head_projected_raw.png        # 2048² RGB — raw multi-view projection
├── head_projection_mask.png      # 2048² binary — data coverage mask
├── head_albedo_diffuse.png       # 2048² sRGB — clean delighted albedo
├── head_albedo_linear.exr        # 2048² linear — for UE5 material input
├── head_roughness_map.png        # 2048² grayscale — GGX roughness
├── head_cavity_ao_map.png        # 2048² grayscale — cavity/AO
└── head_sss_thickness_map.png    # 2048² grayscale — SSS thickness
```

All maps share the FLAME UV parameterisation and can be directly imported into Unreal Engine 5 Principled BSDF material.

---

## 7. Configuration Reference

```yaml
# configs/default.yaml

stage6:
  enabled: true
  texture_resolution: 2048
  blend_gamma: 2.0
  visibility_epsilon: 0.005

stage7:
  enabled: true
  delight_model: deca
  delight_checkpoint: null
  inpaint_method: procedural
  inpaint_iterations: 50
  use_fp16: true

stage8:
  enabled: true
  roughness_t_zone: 0.33
  roughness_cheeks: 0.57
  roughness_lips: 0.22
  cavity_strength: 0.35
  sss_ray_count: 32
  sss_normalize: true
```
