# Postmortem: Meta Multiface & Conditional Detail GAN (Research 1)

> **Status:** ARCHIVED / FAILED APPROACH  
> **Date:** September 2026  
> **Authors:** Humanoid-Face-3D Architecture Team  
> **Reason for Deprecation:** Fundamental dataset frequency ceiling, acquisition artifacts (latex tracking caps, open mouths), and inability to produce 50-micron film-grade micro-detail.

---

## 1. Executive Summary

In Research 1 (Stages 3 & Phase 3), the architectural hypothesis was:
> *"A conditional U-Net Generator + PatchGAN Discriminator trained on ray-cast displacement maps extracted from Meta's public Multiface 3D scan dataset will learn to synthesize photo-conditioned high-frequency micro-pores and wrinkles onto a canonical FLAME base mesh."*

**This hypothesis was disproven in production.** 

While the pipeline succeeded mathematically (loss converged, DDP distributed training ran 40,000 steps on dual Tesla T4s, and test suites passed), the visual output was unacceptable for film-grade or production game standards. The resulting geometry exhibited unnatural facial contours, rubber bald-cap shelf ridges, and a complete absence of true 50-micron epidermal pores.

This document analyzes the exact mechanical, mathematical, and data-domain reasons why this approach failed and why continuing to train on this data is a dead end.

---

## 2. Root Cause 1: The Dataset Frequency Ceiling

The foundational premise of generative adversarial modeling is that **a generator can never output higher spatial frequencies than what is present in its training ground truth**.

Meta Multiface was designed for **optical multi-view performance capture and animation tracking**, not microscopic photogrammetry:
* **Capture Modality:** 40 synchronized machine vision cameras at HD resolution capturing dynamic speech sequences.
* **Mesh Resolution:** Tracked meshes have $\approx 20,000$ vertices ($40,000$ triangles).
* **True Surface Frequency:** The surface contains smooth, low-to-medium frequency macro shape (cheeks, jaw, nose bridge). It contains **zero 50-micron pore-level skin follicles, lip striations, or epidermal micro-creases**.
* **Contrast with Film Standards (Texturing.xyz / MetaHuman):** Film-grade digital humans use 16K cross-polarized macro scans with polarized lighting to eliminate all specular bounce and resolve micro-cavities down to 25–50 microns.

**Outcome:** Training a GAN on Multiface was training it on smooth, interpolated polygon bumps. The GAN could never synthesize pores because pores did not exist anywhere in the training corpus.

---

## 3. Root Cause 2: Physical Acquisition Discontinuities (The Bald Cap & Mouth Deltas)

During the dataset re-extraction verification, inspection of sample `verify_320k_subject_2183941_frame_000120.png` revealed prominent, severe disfigurements:
1. **The Upper Forehead Shelf:**  
   A thick horizontal ridge cuts across the entire upper forehead and temple region. This was not a coding bug—the human actors in the Meta Multiface dataset wore **physical latex bald tracking caps** with sewn-in perimeter borders. The ray-marcher faithfully cast rays from the FLAME forehead to the scan surface, permanently baking the edge of the rubber cap into the displacement map.
2. **Open Mouth Deltas:**  
   Because FLAME base alignment was constrained to a neutral closed mouth ($\psi=0$), frames where the actor was speaking or had their jaw dropped produced massive outward displacement deltas around the lips and chin.
3. **Ear & Scalp Distortion:**  
   Boundary ray misses near the ears and scalp required procedural adjacency filling, creating stepped clay-like plateaus.

When conditioned on portrait photos of individuals who are not wearing rubber bald caps, the generator was forced to reconcile clean portrait pixels with training displacement maps that contained rubber cap borders and speech-deformed lips.

---

## 4. Root Cause 3: The Wireframe Leakage Catastrophe

During Phase 3 baseline training (40k steps), the generator output exhibited a severe artifact: **the 3D FLAME triangular wireframe was stamped all over the subject's face**.

### The Mechanism of Failure:
1. **Subdivision Level 1 Defect:**  
   The initial extraction used Loop Subdivision Level 1 ($\approx 20,000$ vertices). The triangle facets were large enough that the barycentric rasterizer created subtle linear gradient discontinuities along triangle edges.
2. **Insufficient Conditioning Smoothing ($\sigma=1.2$):**  
   The position and normal conditioning maps (`pos_map`, `norm_map`) were only blurred with $\sigma=1.2$. This left visible 2nd-derivative Laplacian edges at every FLAME triangle boundary.
3. **Discriminator Shortcut:**  
   Because the input conditioning maps had faint wireframe facets that aligned with the rasterized displacement training targets, the PatchGAN Discriminator learned to use wireframe facet alignment as a primary signal for "realism."
4. **Result:**  
   The Generator learned to aggressively amplify the triangular wireframe, carving visible polygonal cages into the subject's skin.

*Note:* While our subsequent update to Subdivision Level 2 ($\approx 80,000$ vertices) and $\sigma=8.0$ normalized convolution eradicated the wireframe leakage (reducing the Laplacian metric from $> 0.12$ to $0.029$), it confirmed that without the wireframe artifact, the underlying Multiface scan was simply smooth, clay-like geometry devoid of micro-detail.

---

## 5. Architectural Verdict

| Component | Assumption in Research 1 | Reality in Practice |
| :--- | :--- | :--- |
| **Training Scans** | Meta Multiface neutral frames provide high-resolution skin detail. | Scans are medium-frequency optical tracking meshes containing rubber bald-cap borders and no pores. |
| **Model Role** | GAN can hallucinate 50-micron pores from 1024² scan maps. | GAN only reproduces the smooth blur of the source meshes or amplifies wireframe noise. |
| **Shading / Output** | 2D orthographic displacement thumbnails prove 3D quality. | Flat grey diffuse rendering obscures lack of true SSS, roughness, and micro-cavity interaction. |

Continuing to train or fine-tune on this dataset will consume hundreds of GPU hours without ever bridging the gap to film-grade MetaHuman realism. Research 1 is officially declared **DEPRECATED**.
