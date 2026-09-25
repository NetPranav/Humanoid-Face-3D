# 05 — Innovation Bets: where this project can go beyond reproducing papers

> `02`–`04` get you to "a solid photo-to-head pipeline". This document lists ideas that would make the output
> **better than typical tools**, ranked by (impact × feasibility on your compute).
> Each bet names its v3 stage and a way to measure whether it worked.

| # | Bet | Stage | Impact | Effort | Novelty |
|---|---|---|---|---|---|
| 1 | Lens-aware reconstruction + lens-invariance test | S0/S3 | ★★★ | low | medium |
| 2 | Held-out-judge identity loop | S3/S8 | ★★★ | low | medium |
| 3 | Harvest expression wrinkles instead of discarding them | S4/S5/S7 | ★★★ | medium | high |
| 4 | Degradation-matched self-supervision for UV completion | T1 | ★★★ | medium | medium |
| 5 | Provenance-aware assets | S4/S7 | ★★ | low | high (as a product feature) |
| 6 | FLAME-conditioned, validated generated views as soft constraints | S2/S3 | ★★★ | high | medium |
| 7 | Measured specular/roughness from highlights | S3/S4 | ★★ | medium | medium |
| 8 | Active capture guidance ("take a photo turned 45° left") | S0/S2 | ★★★ | medium | high (as UX) |
| 9 | Personalized ARKit correctives from a selfie video | S7b | ★★★ | high | high |

---

## 1. Lens-aware reconstruction

**Observation:** The same face photographed with a 26 mm phone selfie and a 104 mm telephoto (your `elon.jpg`) looks like two different face shapes: a bigger nose and smaller ears at close range. Most photo-to-avatar tools ignore focal length, which is a hidden cause of "it doesn't look like them".

**Do:**
- Use the EXIF prior and jointly optimize `f` in S3. FLAME's metric scale makes `f` observable.
- When EXIF is missing, run a small focal sweep and pick the one with the lowest fit residual.

**Measure:** a *lens-invariance score*. The same person captured at 2 distances should give fitted neutral meshes within ≤ 1.5 mm (median vertex error). Few pipelines publish this; it is a strong differentiator.

## 2. Held-out-judge identity loop

**Problem:** If you optimize geometry and texture against ArcFace and also evaluate with ArcFace, you can "fool" the metric: identity-adversarial textures score well but look no better.

**Do:** Use network A (ArcFace) as a **low-weight loss** only in the final polish step. Use network B (AdaFace or FaceNet, different training data and architecture) **only for evaluation and gating**. Also track the score on a *held-out real view*.

**Measure:** `id_judge` and `id_judge_novel` rise together. If only the loss-side score rises, you are overfitting.

## 3. Harvest expression wrinkles (turn a nuisance into a feature)

**Observation:** Input photos usually have *some* expression (a smile, raised brows). v1 treated that as noise. It is actually the only evidence of this person's **expression-dependent wrinkles**: smile lines, crow's feet, forehead furrows.

**Do:**
- S3 fits the per-view expression ψ.
- S4/S5 compute the detail on the expressed mesh.
- The detail that is *not* explained by the neutral wrinkles becomes an **expression wrinkle map** tied to the ARKit shapes that were active (for example `mouthSmile*` and `cheekSquint*`).
- Export it as animated wrinkle maps (normal/displacement masks driven by morph weights). MetaHuman uses the same mechanism, and it is supported in UE materials.

**Measure:** with a neutral + smiling photo pair of the same person, render the rig at the smile weights and compare the wrinkle-band LPIPS against the real smiling photo.

## 4. Degradation-matched self-supervision (for T1)

**Do:** Generate training pairs for UV completion by running *your own* S4 backprojection (visibility, grazing-angle weighting, SH lighting, parsing masks) on clean UV textures. The network then learns to invert exactly the corruption it will see at inference.

**Measure:** the completion error on real held-out FaceScape views vs a model trained with random rectangular masks. You should see a clear gap in favour of matched degradations.

## 5. Provenance-aware assets

**Do:** Ship per-texel and per-vertex maps saying *observed / inferred (symmetry) / generated*, plus a confidence value.
- Artists know what they can trust.
- Legal and ethics reviews can verify what was hallucinated.
- Downstream tools (and your own later passes) can re-fit only the generated regions when a new photo arrives.

**Measure:** qualitative. It is a product feature, and it costs almost nothing once S4 tracks weights.

## 6. Generated views as validated soft constraints

**Do:** In S2, generate the side and back views **conditioned on FLAME normal renders at those poses**, so they agree with the current geometry. Then:
- reject any view whose identity (judge B) or landmarks drift, and
- feed the survivors into S3 at 20–30 % weight.

Alternate "fit → regenerate → refit" 2–3 times (an iterative dataset-update loop).

**Measure:** the Phase 6 gate. Does 1 photo + generated views approach the all-real-views fit?

## 7. Measured specular and roughness

**Do:** Add a simple specular lobe (Blinn-Phong or GGX with a per-region roughness texture at low resolution) to the S3 photometric model. Highlights on the forehead, nose and cheeks then *measure* this person's roughness instead of using the constants 0.33/0.57/0.22.

**Measure:** the render-vs-photo error in the highlight regions drops, and the roughness maps differ between oily and dry-skinned subjects in the golden set.

## 8. Active capture guidance

**Do:** After S1, compute the UV coverage and per-vertex uncertainty, then tell the user which *next* photo would reduce uncertainty the most ("turn your head ~45° to your left", "tilt your chin up"). In a mobile or web capture flow, that turns a single-photo problem into a 3–4-photo problem with nearly no extra friction. That is the largest achievable quality jump for real users.

**Measure:** quality gain per additional photo (golden-set metrics) with guided vs random extra photos.

## 9. Personalized ARKit correctives from a selfie video

**Do:** Transferred ARKit shapes have correct semantics but a *generic* shape: everyone's smile deforms the same way. Given a 10–20 s video (smile, blink, jaw open, pucker…):
1. Fit S3 per frame.
2. Use MediaPipe's per-frame ARKit scores as initial weights.
3. Solve for per-subject **corrective** deltas on top of the transferred shapes (example-based facial rigging), with the seam pinned and region masks enforced.

**Measure:** hold out some frames. Drive the rig with the tracked weights and compare the render with the real frames (landmark error + LPIPS) vs the non-personalized rig.
