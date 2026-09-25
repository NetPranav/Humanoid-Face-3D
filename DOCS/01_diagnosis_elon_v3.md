# 01 — Forensic Diagnosis of `outputs/online_test/elon_v3`

> Date: 2026-09-25 · Input: `data/online_test/elon.jpg` (1 photo) · Pipeline: `src/pipeline.py` (v0.1.0)
>
> This document explains **why** the best output so far looks the way it does, with evidence for each cause.
> It is the justification for the v3 architecture in `02_architecture_v3.md`.

---

## 1. What was examined

| Artifact | What it shows |
|---|---|
| `film_render_cycles.png`, `_hero45.png`, `_macro_eye.png` | Final look-dev renders |
| `head_mesh.png` | Untextured geometry (5,023-vertex FLAME) |
| `textures/head_projected_raw.png` | Raw photo → UV projection (Stage 6) |
| `textures/head_albedo_diffuse.png` | "Delighted" albedo (Stage 7) |
| `film_displacement_16bit.png`, `film_normal.png` | Fused Tier 1/2/3 detail (Stage 3) |
| `manifest.json`, `stage5_export_manifest.json` | β, ψ, θ, LOD stats |
| Input EXIF | **Canon EOS R5, 104 mm lens**, cropped to 2924×3843 |

## 2. What is visibly wrong

1. **Likeness is poor.** The head is wide and puffy with a rounded jaw. The subject has a long face and a defined jaw.
2. **The texture is not registered to the geometry.** The eyes are painted onto flat skin above the eye sockets, the nostril texture sits beside the nose tip, and the 45° render shows a **ghost second eye** on the brow.
3. **The mouth reads as open.** A dark hole is projected into the lip region of the UV map.
4. **There is a pink/purple colour cast**, with purple blotches around the eyes and nose.
5. **The sides and back of the head are smeared**: stretched streaks on the ears, neck and cranium, and blurred background crowd pixels on the ears.
6. **There is no hair.** A photo of hair is painted onto a bald FLAME skull.
7. **The displacement and normal maps contain whole facial features** (eye sockets, nose, ears, a mouth ring) instead of pores and wrinkles. The renders therefore show embossed "double" features.
8. **The geometry is lumpy** around the eyes and lips (`head_mesh.png`).

## 3. Root causes, ranked by impact

### R1 — Stage 1 feeds MICA the wrong embedding (identity bug) 🔴

- **Evidence:** In `src/stage1_identity/inference.py:148`, `extract_embedding()` prefers `det.embedding`, which is the InsightFace **buffalo_l** (w600k_r50) embedding from Stage 0. MICA's mapping network was trained on features from **MICA's own fine-tuned ArcFace** (`checkpoint['arcface']`; see `vendor/MICA/micalib/models/mica.py`, `encode()`). The two 512-D spaces are unrelated networks, not rotations of each other.
- **Effect:** β is a plausible FLAME face, but it is not this person's face. This is the main reason for the identity loss.
- **Also:** The regressor loads with `strict=False` (`inference.py:115`), so a key mismatch would leave random weights **without any error**.
- **Fix:** Use MICA's ArcFace on a `norm_crop` 112×112 crop (RGB, `(x-127.5)/127.5`), and load with `strict=True`. This takes about 20 lines and needs no training.

### R2 — There is no real camera, pose or expression estimation 🔴

- **Evidence:** In `src/stage6_texture/projector.py:146`, the intrinsics are hardcoded as `f = image_width`, which gives 2,924 px (53° FOV). The EXIF says 104 mm on a full-frame sensor at ~227 px/mm, so **f ≈ 23,600 px** (7° FOV). **The focal length is off by about 8×.**
- The pose comes from `solvePnP` on only **5 points** (eyes, nose tip, mouth corners). Nothing constrains the jaw, forehead, ears or silhouette.
- Stage 2 is dead:
  - `vendor/smirk` does not exist, so `src/stage2_expression/encoder.py:75` returns a raw `state_dict` and `:109` sets `output = {}`.
  - As a result, ψ and θ are **all zeros** (confirmed in `manifest.json`).
  - `allow_neutral=True` (`src/pipeline.py:94`) hides this.
- **Effect:** A wrong focal length means wrong perspective. The mesh centre (nose) and periphery (ears, jaw) project to the wrong pixels, so every downstream stage samples the photo in the wrong place.

### R3 — The Stage 1.5 contour deformer distorts the shape 🔴

- **Evidence:** `src/stage1_5_residual/contour_deformer.py`:
  - Weights the jaw landmarks ×2.5 and the chin ×4 (line 202), so jaw errors dominate the fit.
  - *Correction (Phase 0, 2026-09-25):* this doc first blamed FLAME's static 68-landmark embedding. InsightFace's `1k3d68` predicts **3D-projected** landmarks, not 2D silhouette points, so the static `full_lmk` embedding is the right correspondence. Measured jaw residuals are no worse than inner-face residuals with the static embedding. The damage comes from the points below.
  - Uses the wrong camera from R2.
  - Allows ±18 mm free-form per-vertex motion (line 101).
  - Fits a **neutral** mesh to landmarks from a face that has an expression.
- **Effect:** A bloated jaw and cheeks, and bumps near the eye and mouth landmarks. The deformer absorbs camera error as fake shape.

### R4 — Texture projection has no occlusion, no segmentation and the wrong mesh state 🔴

- **Evidence:**
  - `_compute_visibility_mask()` (`projector.py:209`) is **never called anywhere**. The only visibility test is backface culling (`cos_angles > 0.05`, line 453). The nose therefore paints onto the cheek behind it, the eyeball and mouth-interior meshes receive eyelid and lip pixels, and you get the ghost eye and the open mouth.
  - There is no face parsing, so hair, background, shirt collar and ears all project as "skin". The only exclusion is a hardcoded `y_norm > 0.16` (line 456).
  - The projection runs on `neutral_vertices` (`src/pipeline.py:375–377`) while the photo has pose and expression. The photo's eye opening does not match FLAME's neutral eyelids, which puts the **iris into eyelid skin**.
  - A single frontal view stretched over the sides of the head causes the grazing-angle smearing.
  - Eye regions get a CLAHE/unsharp "enhancement" (lines 400–420), a cosmetic hack applied before sampling.

### R5 — The displacement stack is fake detail 🔴

| Tier | What it actually does | Why it is wrong |
|---|---|---|
| Tier 1 GAN (`weight_macro=1.0`, `pipeline.py:457`) | Trained on `data/real_scan_displacement_dataset_1024`: 5 Multiface subjects, `num_samples: 80` | The targets are **FLAME↔tracked-mesh registration error** (see any `*_disp.png`: saturated edges, holes, mouth ring). Your own postmortem (`older/FAILED/RESEARCH_1`) deprecated it, but the pipeline still runs it at full weight. |
| Tier 2 "shape-from-shading" (`photometric_detail.py:94`) | High-pass of **albedo luminance** → integrate → height | Dark means deep: eyebrows, iris, nostrils, moles, lip colour and the misregistered texture all become geometry. That produces the ghost-feature embossing. The code also applies RGB weights to a BGR image. |
| Tier 3 pores | Procedural pores under region masks | Fine in principle. But the masks come from bbox heuristics, not an anatomical UV mask. |
| Normal map | Derived from the same displacement **and** rendered together with it (`DISPLACEMENT_AND_BUMP`) | The detail is counted twice. |
| Scale | Encoded as ±5 mm (`fusion.py:32`) but rendered at 0.8 mm (`render_blender_film.py:370`) | Physical units are inconsistent. |

### R6 — Procedural "delighting" invents colour 🟠

- No delighting weights exist. DECA has no U-Net albedo decoder, so the "DECA auto-discovery" (`delight_net.py:386`) could never load meaningful weights. The code falls back to `_dichromatic_delight`, which has hardcoded constants:
  - `target_lum = 0.38` (line 531)
  - a hardcoded sclera colour (136)
  - an ear "capillary boost" of +12 red in fixed UV rectangles (150)
  - Gaussian noise with `RandomState(42)` (158)
  - `median_skin` taken from a fixed UV crop
- **Effect:** The pink/purple cast and the painted-looking inpainted regions.

### R7 — Hair, brows, eyes and teeth are not modelled 🟠

- FLAME is a bald head. Hair is the largest single cue for likeness at a distance, and this subject's hairline and volume are distinctive.
- Stage 4 decided the subject is "clean shaven" with every density at 0.0. That is a preset, not a measurement.

### R8 — The rig and export are placeholders 🟠

- **ARKit-52 blendshapes** (`src/stage5_export/blendshapes.py:47`, `:95`) are made by moving **horizontal bands of the bounding box** (`y_norm` windows, `amp = 8.0`). They have no relation to real ARKit semantics, so "eyeBlink" moves a band of forehead.
- **LOD chain:** The fallback clustering decimator misses its targets. LOD1 and LOD2 are identical (587 verts / 1,180 tris each, against targets of 5,000 and 2,000).
- **MetaHuman bridge:** `EpicGames_MeshToMetaHuman_v1` (`metahuman_bridge.py:127`) is not a real Epic format, and `import_to_ue5_metahuman.py` only prints messages.

### R9 — Silent fallbacks break the project's own Rule 1 🟠

`AGENTS.md` says "Zero Silent Fallbacks". The pipeline instead has about 20 `try/except Exception: print(...)` blocks, `strict=False` loads, `allow_neutral` zeros and `except Exception: pass` (for example `extract_flame_5_landmarks` falls back to `CANONICAL_5_LANDMARKS`). As a result, every failure above still produced a "success" manifest. The 102 unit tests check array shapes and invariants, not whether the result looks like the person.

## 4. Inventory of hardcoded logic

"Remove the hardcoded parts" means different things for different items. Each hardcoded item falls into one of four buckets:

- **LEARN:** replace with a trained model's prediction.
- **FIT:** replace with a per-subject optimization against the photos.
- **AUTHOR:** turn into a subject-independent asset that is made once (painted mask, template rig).
- **KEEP:** procedural is the industry norm here.

| Hardcoded item | Location | Bucket | v3 replacement |
|---|---|---|---|
| Focal = image width | `projector.py:146` | FIT | EXIF prior, then joint optimization of f |
| 5-point PnP pose | `projector.py` | FIT | Dense-correspondence + silhouette + photometric fitting |
| Sparse 68-landmark correspondence (static embedding is correct for 1k3d68, but sparse) | `contour_deformer.py` | FIT | Dense UV correspondences (Pixel3DMM-style) in Phase 1 |
| ±18 mm free-form Laplacian deform | `contour_deformer.py` | FIT | Regularized residual on subdivided FLAME, supervised densely |
| ψ = 0, θ = 0 (dead SMIRK) | `encoder.py` | LEARN + FIT | SMIRK/TEASER init, then per-view ψ, θ in the fitting |
| `y_norm > 0.16` collar exclusion | `projector.py:456` | LEARN | Face/hair/cloth parsing masks |
| Eye CLAHE sharpen | `projector.py:400` | delete | nothing |
| No occlusion test | `projector.py` | FIT | Rasterized z-buffer (nvdiffrast / PyTorch3D) on the **posed** mesh |
| Luminance → height | `photometric_detail.py` | LEARN | Photo→displacement network trained on real paired data |
| Multiface GAN | `stage3_detail/*` | delete | Same as above |
| `target_lum`, sclera paint, ear boost, noise | `delight_net.py` | LEARN | Inverse-rendering albedo + UV completion model |
| BBox "anatomical zones" (T-zone, cheeks…) | `material_stack.py`, `anatomical_pores.py` | AUTHOR | Region masks painted once on the template UV |
| Roughness per zone (0.33/0.57/0.22) | `default.yaml` | AUTHOR + KEEP | Painted prior map, optionally modulated by a learned specular estimate |
| Pore synthesis | `anatomical_pores.py` | KEEP | Tiled pore/micro-normal library under the painted masks (this is what MetaHuman does) |
| ARKit-52 bbox bands | `blendshapes.py` | AUTHOR | Artist-quality ARKit shapes on the template (ICT-FaceKit), deformation-transferred per subject |
| Armature joint placement | `armature.py` | AUTHOR + FIT | Template joints, re-positioned by fitted landmarks |
| Decimation LODs | `lod.py` | AUTHOR | Precomputed LOD meshes on the fixed template topology |
| Facial-hair presets | `stage4_facial_hair/` | LEARN | Parsing density and a strand/card generator |
| Neck collar = lowest 20 % Y | many places | AUTHOR | A painted seam vertex set on the template |

## 5. The structural flaw: an open-loop chain

The current pipeline is a **feed-forward chain of independent guesses**:

```
detect → guess β → guess camera → deform → project → guess albedo → guess detail → guess rig
```

No step ever renders the result and compares it with the photo, so errors compound silently.
Every serious photo-to-head system (DECA/EMOCA training, metrical-tracker, VHAP, Pixel3DMM
fitting, commercial "headshot" tools) is **analysis-by-synthesis**: render, compare with the
image, and update shape, camera, expression and lighting together. This is the key change in v3.

## 6. How far can the current architecture go?

| Level | What you get | What it takes |
|---|---|---|
| L0 (now) | Unrecognizable, texture misregistered, embossed artifacts | — |
| L1 | A clean, smooth, **recognizable-ish** bald head (DECA/MICA-level), no ghost features | Only the Phase 0 fixes in `04_roadmap.md` (days, no training) |
| **Ceiling of v1 design** | ≈ L1+. FLAME at 5k vertices with 300 PCA coefficients, a single projected view, no hair, and fake blendshapes put a hard cap on it | — |
| L2 | Strong frontal likeness, correct camera, clean texture, real ARKit rig | The v3 fitting loop and template retarget |
| L3 | Commercial "photo-to-avatar" quality (hair, eyes, teeth, completed texture, wrinkles) | v3 learned texture, detail and hair modules |
| L4 | Film-grade, relightable, pore-accurate | **Controlled capture** (multi-view, cross-polarized). A single press photo does not contain this information, so anything beyond L3 from one photo is plausible hallucination, not reconstruction. |

## 7. Hygiene issues found along the way

- `legacy_flame_pipeline/` is a **byte-identical copy** of `src/`, `scripts/` and `tests/` (about 20k duplicated lines). `diff -rq` returns nothing.
- Recent commits tagged `[gitbrain]` and the ones after them (for example `23e6fe6`, "Add unit tests for steerable filter…") only touch `utils/dateCalculators.js` or `activity_log.md`. The commit messages do not describe the diffs, and the log's "Architecture verification: healthy" is not backed by any check. Disable that automation, or history becomes unreadable.
- `models_cache/` no longer contains FLAME, MICA or Stage 3 weights, and `data/flame_model/` has no `generic_model.pkl`. The pipeline cannot currently be rerun locally without re-downloading them.
- `src_v2_generative/` (Hunyuan3D/TRELLIS direction) contains only dataset loaders. See `03_models_data_training.md §4` for the recommendation.
