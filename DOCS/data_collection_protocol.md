# In-House Multi-View Facial Photogrammetry Capture Protocol

> **Objective:** Establish an independent, high-fidelity portrait acquisition protocol to collect commercially clean, proprietary 3D training and evaluation data, eliminating reliance on non-commercial academic scan datasets (e.g. FaceScape).

---

## 1. Multi-Camera Rig Geometry

To reliably reconstruct accurate 3D facial geometry, the pipeline requires 3–5 multi-view portraits captured under consistent illumination and neutral articulation.

```
                  [Cam 0: Frontal 0°]
                          ▲
                          │ 1.5m
                          │
     [Cam 1: Left 45°]    │    [Cam 2: Right 45°]
            ╲             │             ╱
             ╲            │            ╱
              ╲           │           ╱
               ▼          ▼          ▼
[Cam 3: Left 90°] ──►  [SUBJECT]  ◄── [Cam 4: Right 90°]
                      (Neutral)
```

| Camera ID | Yaw Angle | Pitch Angle | Working Distance | Critical Anatomy Captured |
|---|---|---|---|---|
| **Cam 0** | $0^\circ$ (Frontal) | $0^\circ$ (Eye level) | 1.5 metres | Facial symmetry, inter-pupillary distance, nasolabial folds, philtrum |
| **Cam 1** | $-45^\circ$ (Left Quarter) | $0^\circ$ | 1.5 metres | Left cheek curvature, jawline definition, nose bridge depth |
| **Cam 2** | $+45^\circ$ (Right Quarter) | $0^\circ$ | 1.5 metres | Right cheek curvature, jawline definition, eyebrow depth |
| **Cam 3** | $-90^\circ$ (Left Profile) | $0^\circ$ | 1.5 metres | Chin projection, ear placement, nose protrusion, submental neck angle |
| **Cam 4** | $+90^\circ$ (Right Profile) | $0^\circ$ | 1.5 metres | Profile cross-verification, ear-to-eye depth distance |

*Note: For single-camera static capture rigs, the subject remains seated in a fixed swivel chair with detents at $-90^\circ, -45^\circ, 0^\circ, +45^\circ, +90^\circ$, or a single camera moves along a marked semi-circular floor track.*

---

## 2. Optical Specifications & Camera Settings

To ensure zero perspective distortion and deep focal depth across the entire head volume:

- **Focal Length:** **85mm** prime equivalent (full-frame sensor equivalent). Lenses below 50mm exhibit perspective barrel distortion (enlarged nose, pushed-back ears) which destroys 3D geometric accuracy.
- **Aperture:** **$f/8.0$ – $f/11.0$**. Deep depth of field ensures the nose tip, pupils, ears, and neck remain tack-sharp simultaneously.
- **Shutter Speed:** **$\ge 1/160\text{s}$** (or $1/200\text{s}$ with studio strobe synchronization) to eliminate micro-motion blur from involuntary breathing or eye tremors.
- **ISO:** **100** (base sensor ISO) for maximum signal-to-noise ratio and dynamic range.
- **Color Format:** Uncompressed 14-bit RAW ($+ 8\text{-bit}$ lossless RGB PNG export).

---

## 3. Illumination & Cross-Polarization Setup

Skin exhibits both **diffuse reflection** (subsurface scattering and skin pigmentation) and **specular reflection** (surface oil and moisture glare). Uncontrolled specular highlights cause false depressions or artifacts in 3D reconstruction.

### Cross-Polarization Architecture:
1. **Light Sources:** 3-point continuous daylight LEDs ($5600\text{K}$, $\text{CRI} \ge 96$):
   - **Key Light:** 45° camera-left, elevated 30°.
   - **Fill Light:** 45° camera-right, lower intensity (1:2 ratio).
   - **Rim/Hair Light:** Directly overhead/behind subject to separate subject from backdrop.
2. **Polarizer Filters:**
   - Linear polarizing film mounted over all softboxes / LED panels in vertical polarization orientation.
   - Circular polarizer (CPL) mounted on each camera lens rotated to **$90^\circ$ cross-polarization (horizontal)** extinction.
   - **Result:** Pure specular glare cancellation; skin pores, wrinkles, and micro-structure are recorded with zero glare.

---

## 4. Subject Posing Protocol

To guarantee canonical neutral base mesh convergence:

1. **Head Alignment:** Aligned along the **Frankfort Horizontal Plane** (horizontal line from the upper margin of the external auditory canal to the lower margin of the orbit).
2. **Expression:**
   - Neutral resting pose (`psi = 0`).
   - Lips gently touching without pressing or pursing.
   - Jaw relaxed with teeth slightly out of occlusion (~2mm space between molars).
   - Brow unfurrowed; forehead smooth.
3. **Gaze:** Fixed straight ahead on the primary frontal camera fixation marker.
4. **Hair:** Hair pulled back with a neutral elastic headband exposing the hairline, forehead, temples, and ears.
5. **Facial Hair Protocol:**
   - For clean-skin subjects: clean shaven within 6 hours of capture.
   - For facial hair subjects (Phase 6): short beard or stubble ($1\text{mm} - 5\text{mm}$) neatly groomed; long beards (>10mm) prohibited as they obscure underlying mandible geometry.

---

## 5. Metric Scale & Color Calibration

- **Metric Scale Ground Truth:** A certified 100.0mm stainless steel gauge bar or precision Vernier caliper is photographed in the frontal view at chin level before each subject capture session.
- **Color Calibration:** An X-Rite ColorChecker Classic target is photographed under session lighting to calibrate camera colour matrix and white balance.

---

## 6. Session File Organization

Each capture session is structured in the following format:
```
data/raw_captures/{session_id}/
├── metadata.json              # Date, operator, camera settings, calibration values
├── release_form.pdf           # Signed talent release document
├── scale_calibration.png      # 100mm gauge bar reference photo
├── color_checker.png          # X-Rite chart reference photo
├── view_0_frontal.png         # Cam 0 (0° yaw)
├── view_1_left45.png          # Cam 1 (-45° yaw)
├── view_2_right45.png         # Cam 2 (+45° yaw)
├── view_3_left90.png          # Cam 3 (-90° yaw)
└── view_4_right90.png         # Cam 4 (+90° yaw)
```
