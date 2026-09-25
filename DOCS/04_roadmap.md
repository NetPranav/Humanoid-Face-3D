# 04 — Roadmap: from `elon_v3` to a production head

> Ordered by **impact ÷ effort**. Each phase has a verification gate measured by the evaluation
> protocol in §9, not by unit tests passing. Time estimates assume one developer working part-time on Kaggle.

---

## Phase 0 — Stop the bleeding (2–4 days, no training) → target L1

The goal is to take the existing code, remove the causes of the worst artifacts, and get a *clean, registered, recognizable* bald head. Do this **before** building v3, because it establishes the evaluation harness and a baseline.

### ✅ Status (2026-09-25): implemented and evaluated on a 9-subject golden set

`python3 scripts/run_golden_set.py --baseline outputs/golden_v1_baseline`. Results are in `outputs/golden_phase0/_summary/` (contact sheet, `summary.md`). The baseline is the **unmodified v1 code at `23e6fe6`**, run on the same photos; only the Multiface GAN weights were no longer available for it.

| Metric (held-out judge: buffalo_l recognition, used by no pipeline stage) | v1 | Phase 0 |
|---|---|---|
| Mean identity cosine, frontal render | 0.383 | **0.782** |
| Mean identity cosine, +30° render | 0.323 | **0.682** |
| Mean identity cosine, −30° render | 0.361 | **0.712** |
| Subjects improved | — | **9 / 9** at every angle |
| 68-landmark reprojection error | not measured | 4.5 – 6.9 % IOD (mean 5.8 %) |

**Gate check:**
- ✅ Overlays aligned, no ghost eyes, no embossing.
- ✅ Identity improved on every subject.
- ❌ **Landmark error ≤ 5 % IOD on only 2 / 9.** The fit keeps MICA's β fixed, so shape errors cannot be absorbed. That moves to Phase 1 (β + Δ fitting, dense correspondences).

**Deviations from the plan below**
- *0.4:* uses InsightFace's 68 **3D-projected** landmarks with FLAME's static embedding, not MediaPipe 478. The 478 → FLAME embedding is deferred to Phase 1.
- *0.5:* eyeballs and the mouth interior are handled by the z-buffer rather than FLAME_masks exclusions. The bbox zones in `material_stack.py` / `anatomical_pores.py` / `blendshapes.py` and the "lowest 20 % Y" seam rule are **not replaced yet**; that moves to Phase 2 with the template assets.
- *0.6:* SMIRK was not vendored (code and weights unavailable). It is **replaced** by a per-view camera/pose/expression/jaw landmark fit (`src/stage2_expression/landmark_fit.py`), which is what texture registration needed.
- *Weights:* FLAME is rebuilt from the MICA checkpoint (`scripts/extract_flame_from_mica.py`), because the Kaggle datasets are not reachable with the current API token.

**Known limitations visible in the results** (all planned in later phases):
- Bald heads (Phase 5).
- Photo lighting baked into the albedo (Phase 3).
- Expression baked into the texture: a smiling photo paints teeth onto the neutral closed lips (Phase 3; interim option: exclude the inner-lip texels when ψ/jaw open the mouth).
- Geometry stays near MICA's shape prior (Phase 1).
- About 55 % of the UV (scalp, back, far side) is interpolated from the observed skin, not observed.

### 0.1 Fix the MICA input (R1)

In `src/stage1_identity/inference.py`, stop using `det.embedding` (buffalo_l). Compute MICA's own feature the way `vendor/MICA/datasets/creation/util.py:get_arcface_input` does:

```python
from insightface.utils import face_align
crop = face_align.norm_crop(image_bgr, landmark=det.landmarks_5pt)            # 112×112
blob = cv2.dnn.blobFromImages([crop], 1.0 / 127.5, (112, 112),
                              (127.5, 127.5, 127.5), swapRB=True)
with torch.no_grad():
    feat = F.normalize(self.arcface(torch.from_numpy(blob).to(self.device)))  # MICA's ArcFace
    beta = self.regressor(feat)
```

Also:
- Make `checkpoint['arcface']` **required**.
- Load the regressor with `strict=True`.
- Add a golden test: β for one sample image must match `vendor/MICA/demo.py` output within 1e-4.

### 0.2 Turn off the fake detail (R5)

- `stage3.weight_macro: 0` (Multiface GAN off) and `stage3.enable_meso: false` (no luminance → height).
- Keep micro pores at a low weight. Derive the normal map from micro detail only.
- Use one displacement scale (mm) written into the manifest and read by `render_blender_film.py`. Remove the hardcoded `0.0008` there and the `max_scale_mm=5.0` in `fusion.py`.

### 0.3 Turn off the Stage 1.5 contour deformer (R3)

Set `stage1_5.enabled: false`. It returns in a correct form as Δ inside the S3 fitter.

### 0.4 Real camera (R2)

Take the focal length from EXIF:

```python
f_px = FocalLength_mm * FocalPlaneXResolution / 25.4
```

This assumes `FocalPlaneResolutionUnit == 2` (inches). If the image was resized after capture, scale by the width ratio. The fallback is `FocalLengthIn35mmFilm × diag_px / 43.27`. For `elon.jpg` this gives 104 × 5773/25.4 ≈ **23,600 px**, not 2,924.

- Replace 5-point PnP with PnP on MediaPipe's 478 landmarks, mapped to FLAME. The MICA/metrical-tracker ecosystem ships a FLAME↔MediaPipe landmark embedding *(verify filename)*. Exclude contour points.
- Pass the same intrinsics to projection, the deformer (if it is ever re-enabled) and the renderer.

### 0.5 Honest projection (R4)

- **Visibility:** replace the unused vertex-splat `_compute_visibility_mask` with a triangle-rasterized depth test (nvdiffrast on Kaggle; a CPU z-buffer per triangle is fine locally).
- **Exclusions:** don't project onto `left_eyeball`, `right_eyeball` or the mouth interior. Use the **official FLAME region masks** that already exist in `vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl`. They include face, lips, nose, ears, scalp, eye regions, eyeballs, neck and **boundary** (114 seam vertices). These masks also replace every bbox "zone" in `material_stack.py`, `anatomical_pores.py` and `blendshapes.py`, and the "lowest 20 % Y" seam rule.
- Add a face-parsing mask (skin/brows/lips/ears only). Hair and background pixels must never land in the skin UV.
- Delete the eye CLAHE/unsharp block.

### 0.6 Expression, or at least honesty about it

- Vendor SMIRK properly (`vendor/smirk` is missing), and apply the resulting jaw and ψ to the mesh **used for projection** (the export stays neutral).
- If SMIRK isn't available, **raise**. Remove `allow_neutral=True` from `pipeline.py:94`.

### 0.7 Remove silent fallbacks (R9)

- Remove every `except Exception: print/pass` in `pipeline.py`, `projector.py` and the Stage 3/7 loaders. Each becomes either a raised error or an explicit `status: skipped_by_config` in the manifest.
- Switch `strict=False` loads to `strict=True`.

### 0.8 Evaluation harness (see §9) — the most important deliverable of Phase 0

**Gate for Phase 0:**
- Overlay renders are aligned: 478-landmark error ≤ 5 % of the inter-ocular distance.
- No ghost eyes or embossed features.
- The held-out identity score on the golden set is **higher than the v1 `elon_v3` settings** for every subject.

### 0.9 Repository hygiene

- Delete `legacy_flame_pipeline/` (an identical duplicate; git history keeps the old code).
- Stop the `[gitbrain]` auto-commit automation and delete `utils/dateCalculators.js` and `activity_log.md`.
- Update `AGENTS.md`, `roadmap.md`, `GUIDE.md` and `remember.md` so that agents follow v3, not v1.

---

## Phase 1 — Perception + analysis-by-synthesis fitter (1–2 weeks) → L2 geometry

1. `src/perception/`: EXIF, MediaPipe 478 + ARKit scores, Pixel3DMM dense UV + normals, parsing, matte, quality gates.
2. `src/fitting/`: renderer interface, FLAME layer (PyTorch, including subdivision-2 with a fixed subdivision matrix), losses, and the 5-step schedule from `02 §S3`. Adapt from metrical-tracker or VHAP rather than writing from scratch.
3. Δ residual on subdivision-2 with Laplacian + ARAP + symmetry, and seam pinning using the FLAME `boundary` mask.
4. Multi-view support: shared β, f and Δ; per-view ψ, R, t and SH.

**Gate:**
- 478-landmark error ≤ 3 % IOD on the golden set.
- Identity score (held-out judge) improves over Phase 0 on ≥ 90 % of subjects.
- On NoW (or REALY), fitted geometry is no worse than MICA alone.

---

## Phase 2 — Production template and real rig (1–2 weeks) → L2 complete

1. `tools/authoring/`: import ICT-FaceKit and build `assets/template/ict/`:
   - FLAME-mean ↔ ICT-mean dense correspondence (one-time NRICP, reviewed by eye)
   - painted region masks and seam set
   - LOD meshes + barycentric maps
   - skeleton + skin weights
   - eye / teeth / tongue / lash meshes
2. `src/production/`: per-subject registration (NRICP refinement), deformation transfer of the 52 ARKit shapes, joint re-positioning, and FBX export with morph targets.
3. Delete `blendshapes.py` bands, the `lod.py` clustering and the `metahuman_bridge.py` fake format.
4. UE5 test: import the FBX, drive it with Live Link Face (iPhone ARKit) and record a clip.

**Gate:**
- Every ARKit shape's displacement lies ≥ 95 % inside its authored region mask.
- Seam vertices are exactly 0 on all morphs.
- UE5 imports with no warnings about the skeleton or morphs.

---

## Phase 3 — Texture: complete, delit, still recognizably the photo

Requested 2026-09-25: unseen areas (back of head, far side, scalp) must carry **the subject's own skin texture** instead of a flat interpolated colour, and the photo's lighting must be removed without losing the person's look.

### 3A — No training ✅ implemented 2026-09-25

**Results (golden set, `outputs/golden_phase3a4a/_summary/`):**
- The unseen ~55 % of the UV now carries the subject's own skin grain; 0–1 % is flat-interpolated.
- The back-of-head UV seam is gone: colour is interpolated on the mesh, not in UV.
- Identity vs Phase 0: frontal 0.782 → 0.763, +30° 0.681 → 0.664, −30° 0.712 → 0.706.
  - The two largest drops (−0.05) are smiling photos, where teeth are no longer painted onto neutral lips.
  - An ablation attributed the rest to delighting. At `delight_strength` 1.0 it costs ~0.05; the default 0.5 costs ~0.01.

**What changed vs the plan:**
- Delighting is luminance-only first-order SH with a clamped correction, applied at half strength in eye and lip areas. The planned 2nd-order per-channel fit bleached eye sockets and jaws.
- Gray-edge white balance was built but defaults **off**: clothing dominated the estimate (a blue flight suit turned skin green).
- Added: lip-only fill for lip texels, a mouth-interior rejection mask, and an upper-face-only skin exemplar (beards made the grain blotchy).

**Still open (→ 3B):**
- Colour cast from coloured light (Elon's magenta stage light).
- Smile-stretched lips on the neutral mesh (a pale band on wide smiles).
- Hair (Phase 5).
1. **Lighting from geometry.** Fit 2nd-order spherical-harmonic lighting per view from the fitted mesh normals and the observed skin (a least-squares solve). Divide the shading out, normalized so the average skin tone is kept. This removes the key-light gradient and shadow sides. It does not change the overall colour cast (a single photo cannot separate skin colour from light colour).
2. **Mouth and teeth exclusion.** Photo pixels inside the inner-lip contour (landmarks 60–67) are never sampled, so a smiling photo stops painting teeth onto the neutral closed lips.
3. **Own-skin texture synthesis.** Build a skin exemplar from the observed, delit high-frequency layer (forehead and cheeks via the FLAME region masks). Synthesize the unseen UV area with patch-based image quilting (Efros & Freeman 2001). Result = smooth low-frequency colour field + the subject's own skin grain. Mirrored real detail keeps priority where it exists.
4. The provenance map gains a fourth class: `synthesized`.

**Gate:** no flat-colour regions in the UV (local high-frequency energy in synthesized areas is within 50–150 % of the observed skin's). Golden-set identity scores do not regress.

### 3B — Learned completion + delighting (needs Kaggle + a dataset)
**T1** (`03 §3`) is a LaMa-style UV inpainting and delighting network trained on self-supervised degradations of clean UV textures.
- It fixes what 3A cannot: semantic completion (ears, eyelids, a hairline consistent with the person), baked expression wrinkles, and residual colour cast.
- **Needs (commercial-safe, see `07`):** Kaggle (token verified 2026-09-25), plus a corpus of portraits you hold commercial rights to (public-domain US-government portraits, your own captures, optionally Unsplash Lite after legal review). Training pairs are made self-supervised from our own Phase 0/3A outputs. FFHQ-UV and FaceScape are non-commercial and are not used.
- Budget: about 20–40 T4-hours.

## Phase 4 — Geometric detail: a sculpt-quality mesh with a 0–100 detail knob

Requested 2026-09-25: the mesh itself (not only the painted texture) should carry skin unevenness, wrinkles and pores, controlled by `detail_level` (0–100, where 100 looks like a studio sculpt).

**What a photo can and cannot give.** A 150 mm-wide face spanning ~1000 px is ~0.15 mm/px.
- Wrinkles, folds, crow's feet, nasolabial and forehead lines (0.5–3 mm) **are resolvable** and come from the photo.
- Pores (0.05–0.2 mm) and lip striations are **at or below the pixel limit**, so they are synthesized with anatomically placed statistics. That is also how studio sculpts get them (scan-derived alpha libraries).

### 4A — No training ✅ implemented 2026-09-25

**Results:**
- `stage3.detail_level` (default 75) produces 4K displacement and normal maps plus `head_mesh_detail.obj`: 20k / 80k / 320k / 1.28M vertices across the knob range.
- Blender Workbench clay renders are written to `eval/clay_*.png`.
- Photo-derived crow's feet, under-eye folds, forehead and nasolabial lines appear in the geometry. Pores, micro-grooves and lip striations are synthesized.

**What changed vs the plan:**
- The planned band-pass "dark is deep" produced crumpled-paper relief from skin-tone blotches. It was replaced by multi-scale Hessian crease detection (Frangi line measure) plus an elongation filter, which removes the rings that moles and freckles leave.
- Unit test: a synthetic crease becomes a groove, a round spot does not.

**Still open (→ 4B):**
- A few pigment lumps survive the filter.
- Expression wrinkles from a smiling photo are baked into the neutral sculpt (see `05` bet #3).
- Pores are statistical, not the person's own.
1. **Sculpt mesh.** Loop-subdivided FLAME: `detail_level` 0–24 → level 1 (20k verts), 25–59 → level 2 (80k), 60–89 → level 3 (320k), 90–100 → level 4 (1.3M). Exported as `head_mesh_detail.obj` with UVs. The low-poly `head_mesh.obj` stays the rig/export base.
2. **Photo-derived meso detail.** Band-passed (≈0.4–3 mm) luminance of the *delit* texture, "dark is deep" (Beeler et al. 2010 mesoscopic augmentation), restricted to skin. Brows, eyes, lash lines, nostrils and the lip colour border are masked by FLAME regions; amplitude is capped per region. Unlike v1 Tier 2 it runs on delit, registered texture and never integrates low frequencies, so it cannot emboss whole features.
3. **Synthesized micro detail (4K UV).**
   - pores: Poisson-disk distributed pits, density and size per region (nose > cheeks > forehead > chin, none on lips)
   - skin cross-hatch micro-grooves
   - lip vertical striations
   - all driven by UV masks built from the official FLAME region masks
4. **The knob** scales meso amplitude, micro amplitude and subdivision level together. Displacement and normal maps are written at 4K in one unit system (mm), and the value is recorded in the manifest.
5. **Clay renders** (Blender Workbench, studio matcap) of the sculpt mesh for review, like a ZBrush viewport.

**Gate:**
- Detail is visible on the geometry at `detail_level` ≥ 60.
- The anti-embossing test holds: |corr(displacement, albedo luminance)| < 0.2 inside brow, eye and lip-border masks.
- Identity scores do not regress.

### 4B — Learned wrinkle geometry (needs Kaggle + FaceScape)
**T2** (`03 §3`) is a photo-texture + normals → displacement network trained on FaceScape's scan-derived displacement maps.
- It replaces the "dark is deep" heuristic with learned shape, which separates pigment from relief properly.
- **Needs (commercial-safe, see `07`):** Kaggle, plus the same licensed portrait corpus with DECA-style self-supervised shading loss. Validation on the CC-BY 3.0 Lee Perry-Smith scan; optional Triplegangers scans with an ML licence for supervised relief. FaceScape is non-commercial and is not used.
- Budget: about 10–30 T4-hours.
- (The pretrained DECA detail decoder is non-commercial; not an option for the commercial build.)

## Phase C — Commercial switch (parallel track, see `07_commercial_licensing.md §6`)

C1 FLAME 2023 Open (CC-BY-4.0) · C2 MediaPipe replaces InsightFace detection/landmarks · C3 MICA removed, β from the Phase 1 fitter · C4 licensed evaluation judge · C5 commercial training corpus. Gates in `07 §6`.

## Phase 5 — Hair and grooming (2–4 weeks)

1. Single-image strands (DiffLocks-style) or, when multiple views or a video exist, strand fitting.
2. Alembic export → UE5 Groom. Automatic cards for LODs.
3. Brows, lashes and facial hair from measured densities (replaces the Stage 4 presets).

**Gate:**
- Hair-region IoU between the render and the photo's hair mask ≥ 0.8 from the input view.
- Plausible at ±60° (manual check).

---

## Phase 6 — Single-image view completion (research, 2–4 weeks)

1. S2 with a pretrained head multi-view model conditioned on FLAME renders.
2. View validation (identity + landmark agreement) and provenance weighting.
3. Ablation: fit with and without generated views.

**Gate:** On multi-view test subjects, fitting from **1 photo + generated views** gets closer to the **all-real-views** fit than 1 photo alone does (vertex error and held-out-view LPIPS).

---

## Phase 7 — Capture path for film grade (parallel, optional)

A single uncontrolled photo cannot produce film-grade (L4) skin; the information is not in the pixels.
If film grade is the goal, build the capture path from `older/data_collection_protocol.md`:
- 8–16 synchronized or turntable views, cross-polarized flash (separates diffuse from specular)
- a photometric-stereo pass for pore normals
- the same v3 fitter with every view real

This also produces **commercially owned data** (with talent releases) that removes the license problems in `03 §6`.

---

## Phase 8 — Personalized animation (innovation track)

From a 10–20 s selfie video, fit per-subject corrective blendshapes on top of the transferred ARKit set, using MediaPipe's per-frame ARKit scores as weak labels and the fitter as supervision. See `05_innovation_bets.md`.

---

## 9. Evaluation protocol (build in Phase 0, run on every change)

**Golden set:** 20–30 subjects, each with 1 hero photo and, where possible, 2–4 extra views.
- Vary lens (phone 26 mm to 105 mm+), pose, age, skin tone, facial hair, glasses on and off.
- Include `elon.jpg` as the regression anchor.
- Store it outside git, with a manifest in git.

**Per-subject metrics** (written to `report.json`):

| Metric | Definition |
|---|---|
| `id_judge` | cos(AdaFace(render), AdaFace(photo)) at the fitted camera. The judge network is **never** used in any loss. |
| `id_judge_novel` | same, on a real held-out view if one exists (catches frontal-only overfitting) |
| `lmk_err_iod` | mean 478-landmark error / inter-ocular distance |
| `lpips_skin` | LPIPS between the render and the photo, on the skin mask |
| `coverage_observed` | % of face UV that is observed (vs inferred or generated) |
| `emboss_corr` | the Phase 4 anti-embossing statistic |
| `rig_region_ok` | fraction of ARKit shapes that pass the region test |

**Artifacts per run:** overlay per view, a turntable (0/±45/±90), a UV provenance map, and a contact sheet across the golden set.

**Rule:** a change that lowers the median `id_judge` or raises the median `lmk_err_iod` on the golden set is a regression, no matter how many unit tests pass.
