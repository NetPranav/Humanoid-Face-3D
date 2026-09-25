# Kaggle Cloud Execution & Training Trajectory Audit (Research 1)

> **Document:** DOCS/FAILED/RESEARCH_1/03_training_trajectory_and_cloud_runs.md  
> **Topic:** Audit of remote Kaggle execution, Phase 2 vs. Phase 3 status, compute time, and artifacts generated.

---

## 1. Cloud Execution Overview

The project roadmap originally defined two distinct remote GPU training workflows on Kaggle:

| Phase | Target Model | Architecture | Planned Training Time | Actual Execution Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 2** | **MICA Identity Regressor** | ArcFace ViT Backbone + 3D Shape Mapping Network ($\beta \in \mathbb{R}^{300}$) | ~6 hours | **Bypassed / Used Pretrained Weights** |
| **Phase 3** | **Detail GAN** | U-Net Generator + PatchGAN Discriminator + Cross-Attention | ~18 hours | **Trained 40,000 steps; Suffered Wireframe Defect** |
| **Phase 2.5/Re-Extract** | **Dataset Re-Extraction** | Differentiable PyTorch Registration + BVH Ray-Casting | ~48 minutes | **Completed Version 2; Clean but Flat** |

---

## 2. Phase 2 (MICA Identity) Audit: Why It Was Not Re-Trained

* **Planned Role:** Fine-tune MICA’s shape mapping network using paired multi-view portrait datasets (FaceScape, Stirling, Florence) to optimize cross-angle identity constancy ($w_i = \text{det\_score}_i \cdot \cos^2(\text{yaw}_i)$).
* **Actual Resolution:**  
  The official pre-trained weights from Zielonka et al. (*Towards Metrical Reconstruction of Human Faces*, ECCV 2022) were located and mounted (`models_cache/mica/pretrained.tar`, 147 MB).
* **Verification:**  
  Across our multi-view validation suite (`carell`, `connelly`, `justin`, `lawrence`), pre-trained MICA demonstrated:
  * Identity divergence $\|\beta_a - \beta_b\|_2 \in [4.61, 7.17]$ (strong inter-subject separation).
  * Yaw robustness across $-45^\circ$, $0^\circ$, $+45^\circ$ angles.
* **Decision:**  
  Re-training MICA from scratch on Kaggle would have consumed 15+ GPU hours while yielding marginal gain over MPI’s authoritative weights. The baseline proceeded with pre-trained MICA.

---

## 3. Phase 3 (Detail GAN Baseline) Audit

* **Kernel Slug:** `nightshowdown/phase-3-deep-detail-gan-train`
* **Hardware:** Dual Tesla T4 GPUs (PyTorch 2.1.0, DDP distributed, batch size 4 per GPU = 8 effective).
* **Duration:** Started 2026-09-21 14:18:15 UTC $\to$ Completed 2026-09-22 08:31:22 UTC (~18 hours across 2 Kaggle sessions).
* **Step Count:** 40,000 generator / discriminator iterations.
* **Metrics & Gates:**
  * Spatial Standard Deviation: $\sigma_{\text{batch}} = 0.0342$ (Passed anti-mode collapse gate $> 0.010$).
  * L1 Reconstruction Loss: Converged to $0.0184$.
  * Checkpoint Artifact: `checkpoints/stage3_detail/ema_generator.pt` (68 MB).

### Post-Training Finding:
When loaded into production inference, the generator reliably produced displacements, but **the output was dominated by the triangular wireframe cage**. The model had successfully learned to minimize L1 loss by matching the low-frequency facet boundaries of the subdivision level 1 training dataset.

---

## 4. Phase 2.5 / Re-Extraction Audit (Subdivision Level 2)

To test whether the wireframe artifact could be resolved simply by improving the dataset, a re-extraction kernel was deployed to Kaggle Cloud:

* **Kernel Slug:** `nightshowdown/dataset-re-extract-subdiv2-c2-smoothing` (Version 2)
* **Hardware:** Dual Tesla T4 GPUs.
* **Runtime:** 2,898.7 seconds (**48.3 minutes**).
* **Processed Assets:** 117 real photogrammetry scans from Meta Multiface across 5 identities (`2183941`, `6795937`, `5372021`, `8870559`, `7889059`).
* **Technical Upgrades Implemented:**
  1. FLAME mesh subdivided to Level 2 ($79,936$ vertices) prior to ray-casting.
  2. $C^2$ Gaussian smoothing with normalized convolution ($\sigma=8.0$).
  3. Output compressed to `real_scan_displacement_dataset_1024.zip` (118.5 MB) to respect Kaggle's 500-file cap.
* **Result:**
  * **Interior Max Laplacian:** Dropped from $0.124$ to **$0.029347$** (Passes wireframe check).
  * **Empirical Metric Scale:** $p_{99} = 2.2639\,\text{mm}$.
  * **Visual Inspection (`verify_320k_subdiv2.png`):** Facets were completely gone, but the surface was clay-smooth, marred by latex tracking cap boundaries and open mouth deltas, with zero 50-micron pores.

---

## 5. Summary of Cloud Compute Investment

| Kernel Run | GPU Time | Outcome |
| :--- | :--- | :--- |
| `phase1-inference-baseline` | ~25 min | Validated 5-point alignment, MICA shape, and ARKit blendshapes. |
| `phase-3-deep-detail-gan-train` (40k steps) | ~18 hours | Completed training; revealed severe wireframe leakage defect. |
| `dataset-re-extract-subdiv2-c2-smoothing` (v1) | ~1 min | Crashed due to missing gitignored FLAME weights (fixed). |
| `dataset-re-extract-subdiv2-c2-smoothing` (v2) | ~48 min | Successfully extracted 117 clean maps; proved Multiface lack of micro-detail. |

**Total GPU Time Expended:** $\approx 19.5$ hours.  
**Critical Conclusion:** Cloud compute cannot compensate for low-frequency source data. Generative training on Meta Multiface will not achieve MetaHuman visual parity.
