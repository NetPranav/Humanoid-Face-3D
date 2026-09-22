# Research 1 Archive: Meta Multiface & Conditional Detail GAN Failure Analysis

> **Status:** ARCHIVED & DEPRECATED  
> **Investigation Period:** September 2026  
> **Key Finding:** Monolithic generative GAN hallucination on Meta Multiface scans cannot produce film-grade 50-micron pores and suffers from physical acquisition artifacts (latex bald caps, open mouth deltas) and wireframe leakage.

---

## Navigation & Postmortem Directory

This folder documents the exact technical, mathematical, and data-domain failure points of the initial **Research 1** approach (Conditional U-Net Detail GAN + Meta Multiface Photogrammetry Scans). 

Each document breaks down a specific facet of what happened, why it failed, what was attempted to salvage it, and the definitive architectural rules derived from the failure:

| Document | Topic | Key Takeaway |
| :--- | :--- | :--- |
| **[01_postmortem_multiface_detail_gan.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/01_postmortem_multiface_detail_gan.md)** | **Primary Failure Postmortem** | Comprehensive breakdown of the dataset frequency ceiling (20k-vertex optical tracking vs 50-micron micro-pores), latex tracking cap border ridges, and open mouth displacement deltas. |
| **[02_wireframe_leakage_and_smoothing_failure.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/02_wireframe_leakage_and_smoothing_failure.md)** | **Wireframe Defect & Smoothing Breakdown** | Mathematical analysis of the triangular facet stamping defect, $\sigma=1.2$ vs $\sigma=8.0$ normalized convolution, Laplacian metric spikes ($0.187 \to 0.029$), and normalization domain bugs. |
| **[03_training_trajectory_and_cloud_runs.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/03_training_trajectory_and_cloud_runs.md)** | **Cloud Execution & Remote Run Audit** | Audit of all Kaggle GPU runs: Phase 2 (bypassed with pre-trained MICA weights), Phase 3 (40,000 steps, ~18h on dual T4s), and Phase 2.5 dataset re-extraction (48.3 minutes, Level 2 subdivision). |
| **[04_lessons_learned_and_dead_ends.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/FAILED/RESEARCH_1/04_lessons_learned_and_dead_ends.md)** | **Architectural Rules & Dead Ends** | The 5 false assumptions made during Research 1, and explicit operational rules on what never to repeat (e.g. never train on low-frequency scans, never evaluate without Subsurface Scattering). |

---

## Summary of the Pivot to Active Architecture

The failure of Research 1 necessitated a complete architectural pivot away from unconstrained generative scan-displacement synthesis to a **4-Tier Hierarchical Engine**:

1. **Tier 1 (Macro):** FLAME 2020 base geometry + MICA 300-D shape $\beta$ (retains skull and landmark proportions).
2. **Tier 2 (Meso):** Real person-specific wrinkles (crow's feet, nasolabial folds, brow furrows) extracted directly from high-resolution portrait photos via shape-from-shading / photometric gradient integration.
3. **Tier 3 (Micro):** 4K anatomical zone-based pore synthesis (Texturing.xyz / MetaHuman standard: T-zone follicles, cheek pores, lip striations).
4. **Tier 4 (Film Shading):** Blender Cycles scene with Random Walk (Skin) Subsurface Scattering, dual-lobe specular roughness, micro-cavity AO, delighted albedo, and 3-point studio lighting.

For the active production documentation, refer to:
* **[DOCS/02_Metahuman_Film_Grade_Synthesis.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/02_Metahuman_Film_Grade_Synthesis.md)**
* **[DOCS/05_Blender_Cycles_Film_Rendering_Engine.md](file:///Users/pranav/Project%20Folder/3d%20Model%20/DOCS/05_Blender_Cycles_Film_Rendering_Engine.md)**
