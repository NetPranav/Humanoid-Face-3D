# Humanoid-Face-3D: Architecture v3 Docs

> Written 2026-09-25 after a full audit of the repo and the `elon_v3` output.
> All previous design docs, guides and postmortems were moved unchanged to [`../older/`](../older/).

## Verdict in one paragraph

The poor `elon_v3` result has two main causes: a handful of **concrete bugs**, and the **lack of any closed loop that checks the result against the photo**. Missing model capacity is not the main cause.
- MICA is fed embeddings from the wrong ArcFace network.
- The camera focal length is hardcoded about 8× too wide for this 104 mm photo.
- The expression model silently outputs zeros.
- Texture projection has no occlusion test and no hair/background masking.
- The "detail" maps are FLAME registration error plus photo brightness used as height. That is where the ghost eyes and embossed features come from.

Most of the "hardcoded" parts you suspected are real problems, but they don't all need to become neural networks:
- some must become **per-subject optimization** (camera, pose, shape residual),
- some must become **authored template assets** (region masks, blendshapes, LODs, rig),
- only a few need **learned models** (texture completion/delighting, wrinkle geometry, hair).

**Keep FLAME as the fitting model, but ship an ICT-FaceKit (or MetaHuman) production topology.**
**Stop fine-tuning TRELLIS**: it is the wrong tool for a rigged, animatable head, and it is infeasible on Kaggle T4s.

## Reading order

| Doc | What it answers |
|---|---|
| [01_diagnosis_elon_v3.md](01_diagnosis_elon_v3.md) | Why the output looks the way it does, with evidence for each root cause (file:line), an inventory of every hardcoded part, and the ceiling of the current design |
| [02_architecture_v3.md](02_architecture_v3.md) | The new architecture: output contract, principles, the FLAME-vs-alternatives decision, stages S0–S8, data contracts, failure policy, code layout |
| [03_models_data_training.md](03_models_data_training.md) | Which pretrained models to use, the only 2–3 models worth training (and how, on Kaggle), datasets, licensing, compute, and the full TRELLIS analysis |
| [04_roadmap.md](04_roadmap.md) | Phase 0 fixes you can do this week (with code), then phases 1–8 with measurable gates and the evaluation protocol |
| [05_innovation_bets.md](05_innovation_bets.md) | Ideas that could make this better than typical photo-to-avatar tools |
| [07_commercial_licensing.md](07_commercial_licensing.md) | What is and isn't commercially usable today, commercial-safe datasets, and the switch plan |

## Key decisions

| Question | Decision | Why |
|---|---|---|
| Replace FLAME? | **No, for estimation. Yes, for delivery.** | Every good pretrained face estimator speaks FLAME. Production needs a real rig, UVs and LODs, which ICT-FaceKit (MIT) provides. |
| Fine-tune TRELLIS? | **No** (use a pretrained one only as an optional hair-shell prior) | Arbitrary topology, weak identity, coarse facial resolution, no rig, and 1B+ parameters on T4s |
| Fine-tune MICA? | **No**. Fit per subject instead (analysis-by-synthesis). | No licensed paired data needed, and it is more accurate per subject |
| What to train? | **T1** UV completion/delighting, **T2** wrinkle displacement, optional **T3** multi-view generation | These are the only places where the photo lacks information |
| Micro pores | Keep procedural (tiled library + painted masks) | This is the industry norm, including MetaHuman |
| Film grade from 1 photo? | Not physically possible | Needs a controlled multi-view cross-polarized capture (Phase 7) |

## Status of the code

`src/` is still the v1 pipeline. It is being migrated per `04_roadmap.md`: Phase 0 patches v1 in place, and Phase 1 onward builds the v3 layout from `02 §8`.
`AGENTS.md` points here. The v1-era root documents (GUIDE, roadmap, remember, Research, MEMORY, DATASETS) are in `older/root_docs_v1/`. What was deleted on 2026-09-25, and why, is recorded in `older/REMOVED_WORK_REPORT_2026-09-25.md`.
