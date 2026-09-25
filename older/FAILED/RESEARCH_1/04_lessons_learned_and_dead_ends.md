# Lessons Learned & Architectural Dead Ends (Research 1)

> **Document:** DOCS/FAILED/RESEARCH_1/04_lessons_learned_and_dead_ends.md  
> **Topic:** Strategic takeaways, false assumptions, and explicit guidelines on what NOT to repeat.

---

## 1. Top Five False Assumptions in Research 1

### Assumption 1: *"A 3D scan dataset automatically implies micro-detail."*
* **The Reality:** 3D scans come in radically different frequency tiers. Meta Multiface scans are optical multi-view stereo tracking meshes. They are accurate for facial performance, jaw motion, and macro skull shapes, but they have **zero micro-pores**.
* **Rule:** Never train a high-frequency generative model on meshes whose vertex spacing is larger than the features you wish to synthesize.

### Assumption 2: *"The neural network will invent the pores if we increase resolution to 1024²."*
* **The Reality:** Rasterizing a smooth 3D scan at 1024×1024 simply yields an interpolated, blurry 1024×1024 image. A GAN trained on smooth images with an L1/L2 reconstruction term will only learn to output smooth images. If pushed aggressively with adversarial loss, it invents random high-frequency noise or amplifies conditioning discretization artifacts (the wireframe).
* **Rule:** High resolution without high-frequency input data is purely wasted storage.

### Assumption 3: *"A single displacement map can carry the entire visual burden of a photorealistic human."*
* **The Reality:** In computer graphics and VFX, geometry is only half the illusion. When a human head is rendered with flat grey diffuse shading (as in `verify_320k_subdiv2.png`), even a 10-million-polygon ZBrush master sculpt looks like a lifeless clay statue or chalk dummy.
* **Rule:** Photorealism requires the complete PBR material interaction: Random Walk Subsurface Scattering (translucency), Dual-Lobe Specular Roughness, Micro-Cavity AO, and Delighted Albedo. Geometry alone cannot achieve the visual depth of human skin.

### Assumption 4: *"2D OpenCV polygon previews are adequate for pipeline validation."*
* **The Reality:** OpenCV polygon projection and orthographic matplot renders ignore camera perspective distortion, lighting bounce, specular Fresnel falloff, and subsurface transmission. Evaluating 3D assets via 2D flat renders creates a massive disconnect between developer metrics and artist reality.
* **Rule:** All visual gates must be evaluated under a calibrated 3-point cinematic lighting rig in a native path tracer (Blender Cycles or Unreal Engine 5 Lumen).

### Assumption 5: *"We should discard the input photos when generating 3D micro-geometry."*
* **The Reality:** In Research 1, the pipeline threw away the high-resolution input photographs after extracting coarse ArcFace identity vectors. Yet those very portrait photos contain the person's **real, actual micro-wrinkles, crow's feet, eye bags, and pore gradients** at sub-millimeter pixel resolution.
* **Rule:** Do not ask a neural network to hallucinate wrinkles from scratch when the subject's actual wrinkles are already recorded in the input photos. Use photometric shape-from-shading to transfer real features directly.

---

## 2. Explicit Dead Ends — DO NOT REPEAT

1. **Do not re-train the GAN on Meta Multiface scans.**  
   No adjustment of hyperparameters, loss weights, or discriminator depth can extract 50-micron pores from a dataset that does not contain them.
2. **Do not attempt to 'sharpen' raw scan displacement maps with unsharp masking.**  
   High-pass filtering optical tracking meshes amplifies tracking noise and triangulation seams, not anatomical pores.
3. **Do not evaluate head assets without Subsurface Scattering (SSS).**  
   Skin without SSS is optically dead. Testing skin shaders in standard diffuse mode will always result in chalky, plastic appearance.
4. **Do not rely on single-lobe roughness.**  
   Human skin has two distinct specular layers: a broad base lobe (stratum corneum) and a sharp micro-lobe (sebum oil). A single roughness value will look either greasy or rubbery, never organic.
