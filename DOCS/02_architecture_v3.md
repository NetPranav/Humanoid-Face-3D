# 02 — Architecture v3: Measure → Fit → Complete → Retarget

> Replaces the v1 stage chain (`older/`) and the unfinished v2 generative direction (`src_v2_generative/`).
> Read `01_diagnosis_elon_v3.md` first for the reasons behind each decision.

---

## 1. Output contract (what one run must deliver)

| Deliverable | Format | Source in v3 |
|---|---|---|
| Neutral head mesh on a **fixed production topology** (face, head, neck) | FBX / glTF / OBJ | S7 |
| Eyes, teeth, gums, tongue, lashes, eye-occlusion meshes | same file | S7 (template assets, placed per subject) |
| ARKit-52 morph targets that are **subject-specific** but semantically correct | FBX morphs + JSON | S7 (deformation transfer) |
| Skeleton (root, neck, head, jaw, eye_L, eye_R) + skin weights | FBX | S7 (template-authored) |
| LOD0–LOD3 | FBX | S7 (template-authored) |
| Albedo (linear EXR + sRGB PNG), roughness/specular, normal (micro), displacement (meso, mm), cavity, SSS thickness | 4K/2K UV maps | S4, S5 |
| **Provenance map**: observed / inferred / generated, per texel | 8-bit PNG | S4 |
| Hair: strands (UE Groom) + card fallback; brows; facial hair | Alembic / FBX | S6 |
| Quality report: identity score, reprojection error, coverage, overlays | JSON + PNGs | S8 |

## 2. Principles

1. **Every per-subject number is measured, fitted, or generated, and labelled as such.**
   - *Measured:* a pretrained perception model's output for this photo.
   - *Fitted:* optimized so that a render matches the photo.
   - *Generated:* hallucinated for unseen regions and flagged in the provenance map.
2. **Every subject-independent quantity is an authored asset, not code.** This covers region masks, blendshape library, LODs, skeleton, skin weights and seam vertices. These are made once on the template and versioned under `assets/`. This replaces the "hardcoded" logic.
3. **Closed loop.** The core stage renders the head and compares it with the photos (analysis-by-synthesis). No stage makes a guess that is never checked against the pixels.
4. **Two topologies.**
   - FLAME is used for *estimation*, because the whole ecosystem of pretrained estimators speaks FLAME.
   - A production template (ICT-FaceKit) is used for *delivery*, because it carries a real rig, UVs and LODs.
5. **Observed beats generated.** Generated content fills gaps and never overwrites observed pixels or geometry.
6. **Fail loudly.** A missing model or a failed gate is an error. Degraded modes exist only as explicit config flags and are written into the manifest.
7. **One unit system.** Metres for geometry and millimetres for displacement, everywhere, including the renderer.

## 3. Representation decision: keep FLAME?

| Option | Strengths | Weaknesses | Decision |
|---|---|---|---|
| **FLAME (2020 / 2023)** | Largest estimator ecosystem (MICA, SMIRK, EMOCA/DECA, TEASER, Pixel3DMM, metrical-tracker, VHAP). Has jaw/eye joints and a UV layout. | 5,023 vertices. 300-D linear PCA, so low-frequency only. Bald. FLAME2020 is non-commercial. | **Keep, as the estimation space** |
| FLAME + per-vertex residual on subdivision-2 (~80k verts) | Breaks the PCA ceiling while keeping FLAME correspondences | Needs **dense** supervision and strong regularization (v1's sparse 68-point version failed) | **Use, inside S3** |
| **ICT-FaceKit** (USC ICT, MIT) | Permissive license. Game-ready topology with face, head, neck, eyes, teeth, gums, tongue and lashes. Its expression blendshapes are designed to be ARKit-compatible. | Identity PCA is less diverse than FLAME's; a one-time FLAME→ICT registration is needed | **Production topology** |
| MetaHuman (UE5) | Best-in-class UE5 rig, grooms and shaders | Identity solve happens inside UE ("Mesh to MetaHuman"); check the current MetaHuman license terms for your use | **Second export target** (feed it our neutral mesh + albedo) |
| NPHM / MonoNPHM (neural SDF heads) | Hair and detail, strong shape | Slow, not rig-ready, non-commercial data | Not now |
| 3D Gaussians (GaussianAvatars, LAM, Avat3r…) | Photoreal renders | Not a UE5 mesh, no PBR material | Not the deliverable |
| TRELLIS / Hunyuan3D mesh | Full head including a hair shell | Arbitrary topology, identity drift, coarse facial features | Optional prior for the hair shell and back of head only (see `03 §4`) |

**Answer:** Do not replace FLAME as the thing you *fit*. Do stop *shipping* FLAME. You ship the ICT template, fitted to the FLAME result. You can swap the production template later (for example to a custom game topology) without touching estimation.

## 4. Pipeline overview

```
                ┌──────────────────────────────────────────────────────────────┐
 photos (1..N) ─▶ S0 PERCEPTION   EXIF focal · detection · 478 lmk + ARKit scores ·
 (+ optional     │                dense FLAME-UV correspondence + normals · parsing ·
  selfie video)  │                hair matte · quality gates · identity embeddings
                └──────────────┬───────────────────────────────────────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ S1 PRIORS (feed-forward)     │ MICA β₀ (correct ArcFace) · SMIRK/TEASER ψ₀,θ₀ ·
                │                              │ camera₀ = EXIF f + PnP on dense landmarks
                └──────────────┬───────────────┘
                               ▼
              coverage < target?──yes──▶ S2 VIEW COMPLETION (generative, validated, flagged)
                               │                         │
                               ▼                         ▼
                ┌───────────────────────────────────────────────────────┐
                │ S3 ANALYSIS-BY-SYNTHESIS FIT (core, differentiable)    │
                │ β, ψᵢ, θᵢ, f, Rᵢ tᵢ, SHᵢ, albedo, Δ(80k)  ⇐ render vs   │
                │ photo: lmk · dense-UV · normals · silhouette · photo · │
                │ identity · priors · ARAP/Laplacian · seam pinning      │
                └──────────────┬────────────────────────────────────────┘
                               ▼
      ┌────────────────────────┼─────────────────────────┬──────────────────────┐
      ▼                        ▼                         ▼                      │
 S4 TEXTURE &            S5 DETAIL                  S6 HAIR & GROOM             │
 REFLECTANCE             meso: learned photo→disp    strands (UE Groom),        │
 backproject on posed    micro: pore tile library    cards fallback, brows,     │
 mesh w/ z-buffer +      (authored masks)            facial hair from parsing   │
 parsing · un-light ·                                                           │
 UV completion model ·                                                          │
 provenance map                                                                 │
      └────────────────────────┬─────────────────────────┘                      │
                               ▼                                                │
                ┌───────────────────────────────────────────────┐               │
                │ S7 RETARGET & RIG                             │◀── assets/template/ict/*
                │ FLAME+Δ → ICT (NRICP) · deformation-transfer   │    (authored once)
                │ ARKit-52 · template skeleton/LODs/eyes/teeth · │
                │ FBX/glTF/Alembic · MetaHuman export            │
                └──────────────┬────────────────────────────────┘
                               ▼
                ┌───────────────────────────────────────────────┐
                │ S8 VALIDATE & LOOK-DEV                         │ identity (held-out judge) ·
                │ overlays · metrics · gates · Cycles/UE renders │ reprojection · LPIPS · coverage
                └───────────────────────────────────────────────┘
```

## 5. Stage specifications

### S0 — Perception (measure everything the photos can tell us)

| Signal | Model (research → commercial-safe alternative) | Replaces v1 |
|---|---|---|
| Focal length | EXIF `FocalLength` × sensor px/mm, else `GeoCalib`-style estimate. Always refined in S3. | `f = image_width` |
| Face box, 5-pt | InsightFace RetinaFace → MediaPipe Face Detector | same |
| 478 landmarks + 52 ARKit scores | MediaPipe Face Landmarker (Apache-2.0) | 5-pt only |
| Dense FLAME correspondence + normals | Pixel3DMM-style ViT (per-pixel UV-coord + normal) | nothing |
| Face parsing (skin, brows, eyes, lips, teeth, ears, neck, hair, cloth, glasses, hat, bg) | SegFace / BiSeNet-CelebAMask / Sapiens-seg (check licenses) | `y_norm > 0.16` |
| Hair alpha matte | BiRefNet / ViTMatte | nothing |
| Identity embeddings | ArcFace (loss) **and** a *different* recognizer such as AdaFace (evaluation only) | buffalo_l only |
| Quality gates | blur (Laplacian var), face px size, occlusion %, extreme expression, sunglasses | nothing |

**Output:** one `Observation` per image. Pixels are never modified; cosmetic enhancement is not allowed at this stage.

### S1 — Priors (fast feed-forward initialization)

- **β₀:** MICA with **MICA's own ArcFace** on a `norm_crop` 112 crop. Multi-view: average the normalized MICA features, weighted by quality.
- **ψ₀, jaw, θ₀:** SMIRK or TEASER per image. MediaPipe ARKit scores act as a sanity cross-check.
- **Camera₀:** EXIF focal plus `solvePnP` on the 478→FLAME-mapped landmarks, excluding contour points.
- These are **initial values only.** S3 is free to move all of them.

### S2 — View completion (only if needed; generative and validated)

- **Trigger:** predicted UV coverage from S1 falls below `coverage.target`, per region (sides, ears, back, under the chin).
- **Method:** a head-specific multi-view diffusion model, conditioned on (a) the input photo and (b) FLAME normal/depth renders at the target poses. The FLAME conditioning keeps the generated views geometrically consistent with the fit. Candidates: FaceLift-, CAP4D- or Morphable-Diffusion-style models; see `03`.
- **Validation:** reject a generated view if ArcFace cos(view, input) < τ_id, or if its landmarks disagree with the FLAME render by more than τ_px.
- **Provenance:** every pixel is labelled `generated`, with loss weight × w_gen (≈0.2–0.3) in S3. Generated pixels never override observed texels in S4.

### S3 — Analysis-by-synthesis fitting (the core of v3)

**Renderer:** nvdiffrast (fast; NVIDIA source licence, non-commercial) or PyTorch3D (BSD; slower). Put it behind a `Renderer` interface.

**Unknowns**

| Symbol | Meaning | Shared/per-view |
|---|---|---|
| β ∈ ℝ³⁰⁰ | FLAME identity | shared |
| Δ ∈ ℝ^(80k×3) | residual on subdivision-2 FLAME (mostly along the normal) | shared |
| ψᵢ ∈ ℝ¹⁰⁰, jawᵢ, eyesᵢ | expression / jaw / gaze | per view |
| f (and cx, cy) | intrinsics, EXIF prior | shared per camera body |
| Rᵢ, tᵢ | head pose | per view |
| SHᵢ ∈ ℝ²⁷ | lighting (2nd-order SH) | per view |
| A (UV) | diffuse albedo texture (coarse, used for the photometric loss) | shared |

**Losses** (all masked by S0 parsing; hair, glasses and background excluded)

| Loss | Purpose |
|---|---|
| `L_lmk` | 478 landmarks, with dynamic jaw contour (silhouette-aware, **not** a static embedding) |
| `L_uv` | dense per-pixel correspondence to predicted FLAME-UV (breaks the "5-point" weakness) |
| `L_normal` | rendered vs predicted normals on skin |
| `L_sil` | IoU of the rendered face/ear/neck silhouette vs the parsing mask |
| `L_photo` | robust photometric error (Charbonnier + LPIPS) under SH shading |
| `L_id` | 1 − cos(ArcFace(render), ArcFace(photo)). Low weight, final stage only. |
| `R_β, R_ψ` | Mahalanobis priors |
| `R_Δ` | Laplacian + ARAP + magnitude + soft bilateral symmetry |
| `R_seam` | Δ = 0 on the **authored** neck-seam vertex set (replaces "lowest 20 % Y") |

**Schedule** (coarse → fine, about 1–3 min per subject on a T4)

1. f, R, t (landmarks + silhouette)
2. \+ β, ψ, jaw (+ `L_uv`)
3. \+ SH, A (+ `L_photo`, `L_normal`)
4. \+ Δ (all losses)
5. short `L_id` polish, capped so that geometry moves less than 1 mm

**Output:** `SubjectFit`, containing:
- neutral identity mesh (ψ = 0 + Δ)
- per-view expressed and posed meshes
- cameras and SH
- per-vertex confidence (visibility count × residual)
- fit residuals

**Starting points:** metrical-tracker (MICA authors), VHAP (GaussianAvatars authors) and the Pixel3DMM fitting code all implement most of this for FLAME. Adapt one of them rather than writing from zero.

### S4 — Texture and reflectance

1. **Backprojection** per view onto the **posed, expressed** mesh from S3, so eyelids and lips match the photo:
   - z-buffer visibility via the rasterizer
   - only parsing classes {skin, lips, brows, ears} are used
   - weight = cos(θ)^γ · (projected texel footprint) · sharpness · confidence
   - Laplacian-pyramid blending across views, seam-aware on the UV layout
2. **Un-lighting:** A_obs = I / shading(SHᵢ, n). This is physically grounded and uses S3's lighting, not constants.
3. **UV completion and delighting model (T1, see `03`).** Takes A_obs, the mask, the mirrored-UV A_obs (faces are near-symmetric) and the identity embedding. Outputs a complete, evenly lit albedo plus roughness/specular. Observed texels are kept; the model only fills gaps and removes residual shading.
4. **Eyes:** iris region from the photo → polar unwrap → template iris texture; sclera, cornea and wetness come from template materials.
5. **Provenance map:** 255 = observed, 128 = inferred (symmetry), 0 = generated.

### S5 — Geometric detail

| Band | Scale | Method |
|---|---|---|
| Macro | > 5 mm | S3 (β + Δ) |
| Meso (wrinkles, folds) | 0.1–2 mm | **Learned** photo-texture + normal → displacement (T2, trained on real paired data). Interim: pretrained DECA/EMOCA detail decoder (non-commercial). Never luminance. |
| Micro (pores) | < 0.1 mm | Tiled micro-normal / pore library under **painted** template region masks (the same approach MetaHuman uses) |

Meso goes into displacement (mm) and micro into the normal map; they are never double-counted. Blender and UE read the same physical scale from the manifest.

### S6 — Hair, brows, lashes, facial hair

- **Scalp hair:** parsing + matte give the hair region. Then:
  - single-image strand generation (DiffLocks-style), or
  - multi-view/video strand fitting when available (NeuralHaircut / GaussianHaircut-style).
  - Export as Alembic → **UE5 Groom**, with an automatic strand-to-card conversion for LODs and non-UE targets.
- **Scalp albedo** under the hair: completed by T1 as skin.
- **Brows / facial hair:** density and direction from parsing and the texture, as short strands or cards. No "clean-shaven" presets; densities are measured.
- **Lashes:** template asset, colour from the photo.

### S7 — Retarget and rig (where the old hardcoding goes to die)

All of the following live in `assets/template/ict/`, are **authored once** and are versioned:
mesh, UVs, region masks, seam vertex set, 52 ARKit shapes (+ correctives), skeleton, skin weights, LOD meshes + barycentric maps, eye/teeth/tongue/lash meshes, and the FLAME↔ICT dense correspondence.

Per subject:
1. **Registration:** FLAME(+Δ) surface → ICT neutral, using the precomputed correspondence followed by landmark-guided NRICP refinement (< 1 mm target).
2. **Blendshapes:** deformation transfer (Sumner & Popović 2004) of each ICT ARKit shape onto the subject's neutral. Real semantics, subject-specific proportions.
   - *Optional (S7b):* if a 10–20 s selfie video is supplied, fit **personalized correctives** using the MediaPipe ARKit scores as weak labels (example-based facial rigging).
3. **Skeleton:** template joints re-positioned from fitted landmarks (jaw pivot, eye centres).
4. **LODs:** `V_lod = W_lod · V_lod0`, which is just a matrix multiply.
5. **Export:** FBX (skeletal mesh + morph targets + materials), glTF, Alembic hair.
6. **MetaHuman path:** export the neutral mesh + albedo in a form that UE's *Mesh to MetaHuman* accepts. Drive it with real UE Python API calls (verify against your UE version) instead of the current print-only stub.

### S8 — Validation and look-dev

| Metric | How | Gate (initial, tune on a test set) |
|---|---|---|
| Identity | cos(AdaFace(render_i), AdaFace(photo_i)), same camera; the judge ≠ the loss network | ≥ τ_id (set from a 20-subject test set) |
| Reprojection | 478-landmark error / inter-ocular distance | ≤ 3 % |
| Photometric | LPIPS on skin mask | report |
| Coverage | % of face UV observed vs generated | report + warn |
| Rig sanity | each ARKit shape's displacement lies inside its authored region mask | 100 % |
| Seam | Δ and every morph = 0 on seam vertices | exact |

The **overlay** (render composited over the photo with the fitted camera) is the most important debug image. Generate it for every view on every run.

## 6. Data contracts (sketch)

```python
@dataclass
class Observation:            # S0, one per image
    image: np.ndarray; exif_focal_px: float | None
    bbox: np.ndarray; lmk478: np.ndarray; arkit_scores: np.ndarray
    uv_corr: np.ndarray; normals: np.ndarray        # dense priors (H,W,2)/(H,W,3)
    parsing: np.ndarray; hair_alpha: np.ndarray
    id_arcface: np.ndarray; id_judge: np.ndarray; quality: dict
    provenance: Literal["photo", "generated"]

@dataclass
class SubjectFit:             # S3
    beta: np.ndarray; delta: np.ndarray              # (300,), (V_sub,3)
    views: list[ViewFit]                             # psi, jaw, eyes, R, t, SH per view
    intrinsics: Intrinsics; albedo_coarse: np.ndarray
    vertex_confidence: np.ndarray; residuals: dict

@dataclass
class AssetBundle:            # S7 output + S4/S5/S6 maps
    template_version: str; neutral: Mesh; morphs: dict[str, np.ndarray]
    skeleton: Skeleton; lods: list[Mesh]; textures: dict[str, Path]
    provenance_map: Path; hair: HairAsset | None; report: dict
```

## 7. Failure policy

| Situation | Behaviour |
|---|---|
| Required model or asset missing | `FileNotFoundError` with the download command. **No** zero-vectors, `strict=False` loads or mean shapes. |
| Quality gate fails (blur, tiny face, heavy occlusion) | Error, unless `--allow-low-quality`, which is recorded in the manifest |
| Optional module unavailable (S2, S6 strands) | Only if `enabled: false` in config; the manifest records `skipped_by_config` |
| Metric gate fails in S8 | Exit code ≠ 0; assets are still written for inspection, with `"status": "below_gate"` |

Texture stages still never block geometry export (v1 Rule 6 is kept), but they must report their status explicitly.

## 8. Proposed code layout

```
src/
  core/        types.py · config.py · units.py · provenance.py · errors.py
  perception/  exif.py · detect.py · landmarks.py · dense_corr.py · parsing.py · matting.py · quality.py
  priors/      mica.py · expression.py · camera_init.py
  completion/  multiview.py · validate_views.py
  fitting/     renderer.py · flame_layer.py · losses.py · fitter.py · schedule.py
  texture/     backproject.py · unlight.py · complete.py · eyes.py
  detail/      meso.py · micro.py
  hair/        strands.py · cards.py · facial_hair.py
  production/  template.py · register.py · deformation_transfer.py · rig.py · lod.py · export_fbx.py · metahuman.py
  evaluation/  identity.py · photometric.py · overlays.py · report.py
assets/template/ict/   authored, versioned template data (see S7)
tools/authoring/       one-off scripts that build assets/ (correspondence, masks, LOD maps)
```

## 9. What survives from v1

| Keep (move into the v3 layout) | Delete |
|---|---|
| `utils/flame_model.py` (FLAME loader, chumpy shim) | `stage3_detail/` GAN, trainer, Multiface datasets |
| `stage3_detail/rasterizer.py` (UV rasterization; use for UV-space ops) | `photometric_detail.py` (luminance → height) |
| `stage8_pbr` SSS-thickness bake (geometry-based, OK) | `stage1_5_residual/contour_deformer.py` |
| `scripts/render_blender_film.py` (fix units, drop the double-counted normal) | `stage7` procedural delight + hardcoded inpainting |
| Kaggle checkpoint/resume/emergency-save utilities | `blendshapes.py` bbox bands, `lod.py` clustering fallback |
| `older/data_collection_protocol.md`, `older/talent_release_form.md` (for the capture path) | `metahuman_bridge.py` fake format |
| Invariant tests (seam, units, no silent fallback) | `legacy_flame_pipeline/` duplicate, `utils/*.js` |
