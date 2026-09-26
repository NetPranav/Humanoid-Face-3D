<p align="center">
  <h1 align="center">Humanoid-Face-3D</h1>
  <p align="center">
    <strong>Photo → Production-Ready 3D Head</strong><br>
    Identity-preserving facial reconstruction with PBR textures, ARKit blendshapes, and UE5-ready FBX export.
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#output">Output</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#license">License</a>
</p>

---

## What Is This?

A multi-stage pipeline that reconstructs a **fully textured, PBR-ready, rigged 3D humanoid head mesh** from 1–5 portrait photographs. The output is a game-ready asset package with physically based material maps, facial animation blendshapes, level-of-detail meshes, and a skeletal rig — packaged as FBX for direct import into Unreal Engine 5.

**This is not a research demo.** The goal is a production-grade asset that a character artist would accept as a starting point, not a paper reproduction with toy outputs.

---

## Features

### Core Reconstruction
- **Identity-preserving 3D head mesh** from a single photo (or multiple for higher accuracy)
- **FLAME parametric head model** (5,023 vertices, 9,976 triangles) as the geometric foundation
- **Analysis-by-synthesis fitting** — jointly optimizes shape, expression, jaw pose, camera focal length, and per-view lighting against 478 MediaPipe landmarks and photometric loss
- **Multi-view photo-consistency refinement** — when given 2+ same-session photos, refines per-vertex geometry so every surface point agrees across views

### Textures & Materials
- **Multi-view UV texture projection** with a real z-buffer for correct occlusion and skin-only parsing masks (no hair, clothing, or background bleeds into the texture)
- **Provenance-aware UV fill** — unseen areas (back of head, far side) get smooth colour interpolation + mirrored skin detail + quilted grain from the subject's own observed skin
- **Fitted-SH delighting** — removes baked-in lighting from the photo to produce a clean albedo
- **PBR material stack** (pure math, no GPU):
  - **Roughness** — anatomical zone-based (T-zone oilier, cheeks rougher)
  - **Cavity / AO** — Laplacian of displacement for pore-level ambient occlusion
  - **SSS Thickness** — geometric ray-march through the mesh for subsurface scattering

### Detail & Hair
- **Photo-derived wrinkle extraction** — Hessian crease detection on the delit texture captures meso-scale creases
- **Procedural micro-detail synthesis** — pores, micro-grooves, and lip striations at 4K resolution, unique per subject
- **Facial hair system** — stubble micro-displacement + 3D polygonal hair cards with UV, alpha, and strand normals

### Production Export
- **ARKit-52 blendshapes** for real-time facial animation (compatible with UE5 Live Link)
- **4-tier LOD chain** (LOD0 full → LOD3 impostor) with blendshapes re-projected onto every LOD
- **5-joint skeletal armature** (neck → head → jaw, eye_L, eye_R) with LBS skinning weights
- **Headless Blender FBX packaging** — mesh, blendshapes, armature, hair cards, PBR material slots in one file
- **MetaHuman bridge** — exports in a format compatible with UE5's "Mesh to MetaHuman"
- **Parametric stylization** — continuous sliders for jaw width, chin depth, cheekbone prominence

### Quality Assurance
- **208 unit tests** covering every stage
- **9-subject golden set** benchmark with automated regression detection
- **Ground-truth scan evaluation** (Lee Perry-Smith CC-BY head) with ICP alignment and mm-accurate surface error
- **Identity scoring** at frontal and ±30° yaw via ArcFace cosine similarity

---

## Architecture

```
  Photo(s)
     │
     ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    FaceGeoPipeline                       │
  │                                                         │
  │  Preprocessing ──► Face detection (MediaPipe 478 lmk)   │
  │                    EXIF camera intrinsics                │
  │                    Semantic skin parsing                 │
  │                                                         │
  │  Fitting ────────► Identity shape (β)                   │
  │                    Per-view camera, pose, expression     │
  │                    Multi-view photometric refinement     │
  │                                                         │
  │  Texturing ──────► UV backprojection (z-buffer + skin)  │
  │                    Provenance-aware hole fill            │
  │                    SH delighting + skin grain synthesis  │
  │                                                         │
  │  Detailing ──────► Photo-derived wrinkles (Hessian)     │
  │                    Procedural pores (4K)                 │
  │                    PBR maps (roughness, cavity, SSS)     │
  │                                                         │
  │  Hair ───────────► Stubble displacement                 │
  │                    Polygonal hair cards                  │
  │                                                         │
  │  Export ─────────► OBJ + MTL (neutral base mesh)        │
  │                    ARKit-52 blendshapes                  │
  │                    4-tier LODs                           │
  │                    Skeletal armature                     │
  │                    FBX (UE5 Live Link ready)             │
  │                    MetaHuman bridge asset                │
  └─────────────────────────────────────────────────────────┘
     │
     ▼
  Output folder: 20+ production files
```

### Code Layout

```
src/
├── pipeline.py                    # Master orchestrator
├── fitting/
│   ├── fitter.py                  # Analysis-by-synthesis joint optimizer (478 lmk + photometric)
│   └── mv_refine.py               # Multi-view photo-consistency refinement (NCC matching)
├── stage0_preprocess/
│   ├── camera.py                  # EXIF focal length → pinhole intrinsics
│   ├── detector.py                # InsightFace face detector (research path)
│   ├── landmarks_mp.py            # MediaPipe 478-point landmarks + ARKit scores
│   └── parsing.py                 # Semantic skin/hair/background segmentation
├── stage1_identity/
│   └── inference.py               # MICA identity encoder (research path)
├── stage2_expression/
│   └── landmark_fit.py            # 68-landmark camera + expression fitter (research path)
├── stage3_detail/
│   ├── sculpt_detail.py           # Photo-derived wrinkles + procedural pore synthesis
│   ├── fusion.py                  # Multi-tier displacement fusion
│   └── rasterizer.py              # UV-space rasterization utilities
├── stage4_facial_hair/
│   ├── generator.py               # Facial hair orchestrator
│   ├── regions.py                 # Anatomical hair zone segmentation
│   ├── cards.py                   # 3D polygonal hair card generator
│   └── stubble.py                 # Procedural stubble micro-displacement
├── stage5_export/
│   ├── exporter.py                # Stage 5 master orchestrator
│   ├── retopology.py              # Barycentric correspondence engine
│   ├── blendshapes.py             # ARKit-52 blendshape generation
│   ├── lod.py                     # Multi-resolution LOD decimation
│   ├── armature.py                # 5-joint skeletal rig builder
│   ├── fbx_packager.py            # Headless Blender FBX export
│   ├── metahuman_bridge.py        # UE5 MetaHuman bridge exporter
│   └── stylize.py                 # Parametric facial stylization
├── stage6_texture/
│   └── projector.py               # Multi-view UV texture backprojection
├── stage7_delight/
│   ├── uv_fill.py                 # Provenance-aware UV completion
│   └── skin_synthesis.py          # Skin grain quilting (Efros & Freeman 2001)
├── stage8_pbr/
│   └── material_stack.py          # Procedural PBR map generator
├── render/
│   ├── soft_raster.py             # CPU triangle rasterizer with z-buffer
│   └── soft_renderer.py           # Evaluation renderer (Lambert + ambient)
└── utils/
    ├── flame_model.py             # FLAME head model decoder (pure NumPy)
    ├── flame_torch.py             # Differentiable FLAME layer (PyTorch)
    ├── flame_landmarks.py         # 68-landmark sparse embedding
    ├── flame_regions.py           # UV region masks (face, lips, eyes, etc.)
    ├── subdivision.py             # Loop subdivision + displacement mapping
    └── validation.py              # Input photo validation
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [FLAME 2023 Open](https://flame.is.tue.mpg.de/) head model (CC-BY-4.0)
- [Blender 4.x](https://www.blender.org/) (for FBX export and film-grade renders; optional)

### Installation

```bash
git clone https://github.com/NetPranav/Humanoid-Face-3D.git
cd Humanoid-Face-3D
pip install -r requirements.txt
```

### Setup FLAME

Download the FLAME 2023 Open model and extract `generic_model.pkl` to `data/flame_model/`:

```bash
# Or extract from MICA checkpoint (research path):
python3 scripts/extract_flame_from_mica.py --mica models_cache/mica/mica.tar
```

### Run the Pipeline

```bash
# Single photo
python3 scripts/run_production_pipeline.py \
    --photos path/to/portrait.jpg \
    --output_dir outputs/my_head

# Multiple photos (same session, same person)
python3 scripts/run_production_pipeline.py \
    --photos front.jpg left.jpg right.jpg \
    --output_dir outputs/my_head
```

### Run Tests

```bash
python3 -m pytest tests/ -v
```

---

## Output

A single run produces the following in the output directory:

| File | Description |
|---|---|
| `head_mesh.obj` | Neutral base mesh (5,023 vertices) with UV coordinates |
| `head_mesh_detail.obj` | Subdivided mesh with displacement applied (~80k–320k vertices) |
| `albedo.png` | Diffuse colour texture (delit, seam-free) |
| `film_displacement_16bit.png` | 16-bit displacement map (±1mm encoded as [0, 65535]) |
| `film_normal.png` | Tangent-space normal map |
| `film_roughness_base.png` | Anatomical roughness map |
| `film_cavity_ao.png` | Cavity / ambient occlusion map |
| `sss_thickness.png` | Subsurface scattering thickness map |
| `provenance.png` | Per-texel source tracking (observed / mirrored / synthesized / filled) |
| `blendshapes_arkit52.json` | 52 ARKit blendshape deltas |
| `armature.json` | 5-joint skeletal rig definition |
| `face_lod{0-3}.obj` | 4-tier LOD chain |
| `head_mesh_ue5_livelink.fbx` | Complete UE5-ready FBX asset |
| `metahuman_identity.obj` | MetaHuman bridge asset |
| `manifest.json` | Full pipeline metadata, parameters, and stage status |

---

## Evaluation

### Golden Set Benchmark

```bash
python3 scripts/run_golden_set.py \
    --golden data/golden_set \
    --out outputs/golden_eval
```

Runs the pipeline on 9 public-domain portrait subjects and produces a contact sheet with identity cosine similarity scores at frontal and ±30° yaw.

### Ground-Truth Scan Evaluation

```bash
# Render a test photo from a known 3D scan
python3 scripts/render_gt_scan.py --out outputs/gt_scan/lps_front --yaw 0

# Run the pipeline on it
python3 scripts/run_production_pipeline.py --photos outputs/gt_scan/lps_front/photo.jpg --output_dir outputs/gt_scan/pipeline_result

# Score the reconstruction against the true scan (mm-accurate surface error)
python3 evaluation/gt_scan_eval.py --run outputs/gt_scan/pipeline_result --scene lps_front
```

### Compare Against External Services

```bash
# Score a mesh from Tripo, Meshy, etc. against the same ground truth
python3 scripts/compare_external.py --scene lps_front --mesh tripo_result.glb --name Tripo
```

---

## Current Metrics

From the 9-subject golden set evaluation:

| Metric | Value |
|---|---|
| Mean identity cosine similarity (frontal) | **0.782** |
| Mean identity cosine similarity (±30°) | **0.654** |
| Mean landmark error | **4.9% IOD** |
| Test suite | **208 tests passing** |
| Runtime per subject (Apple Silicon, CPU) | **~30–50s** |
| Peak VRAM | **~3.5 GB** |

---

## Tech Stack

| Component | Technology | License |
|---|---|---|
| Head model | FLAME 2023 Open | CC-BY-4.0 |
| Face landmarks | MediaPipe Face Landmarker | Apache-2.0 |
| Skin parsing | MediaPipe Selfie Segmenter | Apache-2.0 |
| Fitting | PyTorch (CPU) | BSD |
| Rasterization | Custom NumPy z-buffer | — |
| Mesh processing | trimesh, SciPy | BSD |
| FBX export | Blender (headless) | GPL-2.0 |
| PBR generation | Pure NumPy (no GPU) | — |

---

## License

This project's source code is provided as-is for research and educational purposes. 

**Important:** The pipeline currently depends on the FLAME parametric head model. You must obtain FLAME separately and agree to its license terms. See the [FLAME project page](https://flame.is.tue.mpg.de/) for details.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete list of third-party components and their licenses.

---

## Contributing

Contributions are welcome. If you're working on digital humans, real-time avatars, or VFX pipelines, feel free to open an issue or pull request.

---

<p align="center">
  Built by <a href="https://github.com/NetPranav">@NetPranav</a>
</p>
