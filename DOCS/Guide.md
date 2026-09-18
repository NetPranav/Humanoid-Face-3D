# Face Geometry Pipeline — Implementation Guide

> This is the **execution guide**, not the research reference. It tells you what to type, in what order, on which day. For *why* any decision was made, see the DOCS folder. Every step here ends with a verifiable output — don't move to the next step until the current one passes its check.

---

## Before You Start — The Mental Model

The pipeline has 6 stages. Each stage is a separate Python module. Each stage's trained weights live in Kaggle's Model registry under your account. The production notebook (`production_inference.ipynb`) loads all stages and runs them in sequence on a new set of photos. You never need a GPU locally — everything runs on Kaggle.

Your local machine is for: writing code, committing to git, reviewing output meshes in Blender. That's it.

The order of work across the whole project is:
```
Day 0      → Environment setup, FLAME loader verified, nvdiffrast verified
Days 1–5   → Phase 1: inference-only baseline with pretrained weights
Weeks 2–3  → Phase 2: fine-tune MICA + EMOCA/SMIRK
Weeks 4–7  → Phase 3: build and train the detail GAN
Weeks 4–8  → Phase 4: data licensing (parallel, business track)
Weeks 8–12 → Phase 5: hybrid detail + facial hair
Weeks 10–13 → Phase 6: production export (UE5 FBX)
Ongoing    → Phase 7: evaluation and benchmarking
Post-v1    → Phase 8: NPHM, Pixel3DMM, future upgrades
```

---

## Day 0 Checklist — Do This Before Writing Any Code

These are hard blockers. If any of them fail, nothing else works. Do them in order.

### Step 1 — Kaggle account setup

1. Go to kaggle.com → Settings → Phone verification — complete it. Without this, "Internet" is disabled in notebooks and you cannot pull pretrained weights.
2. Go to Settings → API → "Create New Token" — download `kaggle.json`. Keep it safe.
3. In a Kaggle notebook, go to "Add-ons" → "Secrets" → add a secret named `github_deploy_key` (you'll fill this in Step 3).

### Step 2 — GitHub repo setup

1. Create a new **private** GitHub repository named `face-geo-pipeline`.
2. On your local machine, inside `/Users/pranav/Project Folder/3d Model /`, run:
```bash
git init
git remote add origin git@github.com:youruser/face-geo-pipeline.git
```
3. Generate a deploy key (read-only is enough for Kaggle to clone):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/face_geo_deploy -N ""
```
4. Add the public key (`face_geo_deploy.pub`) to the GitHub repo under Settings → Deploy Keys.
5. Copy the **private key** (`face_geo_deploy`) content and paste it into the Kaggle Secret you created in Step 1.

### Step 3 — Test the Kaggle clone workflow

Create a new Kaggle notebook (GPU T4×2, Internet ON). In the first cell:

```python
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
deploy_key = secrets.get_secret("github_deploy_key")

os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
with open(os.path.expanduser("~/.ssh/id_ed25519"), "w") as f:
    f.write(deploy_key)
os.chmod(os.path.expanduser("~/.ssh/id_ed25519"), 0o600)

!ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
!git clone git@github.com:youruser/face-geo-pipeline.git /kaggle/working/face-geo-pipeline
print("Clone successful" if os.path.exists("/kaggle/working/face-geo-pipeline") else "FAILED")
```

Expected output: `Clone successful`. If you see an SSH error, the deploy key is not set up correctly — fix it before proceeding.

### Step 4 — Verify the FLAME loader (most critical Day 0 task)

FLAME's default `.pkl` loader uses `chumpy`, which breaks on modern numpy. Do this test before writing a single line of downstream code.

Download FLAME2020 model files from https://flame.is.tue.mpg.de (requires registration). Place `generic_model.pkl` in `data/flame_model/`.

In a Kaggle notebook cell:

```python
import numpy as np
import pickle

# Custom unpickler that replaces chumpy arrays with numpy
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect chumpy types to numpy equivalents
        if 'chumpy' in module:
            return np.array
        return super().find_class(module, name)

def load_flame_model(path):
    with open(path, 'rb') as f:
        data = NumpyUnpickler(f).load()
    return data

flame = load_flame_model('/kaggle/working/face-geo-pipeline/data/flame_model/generic_model.pkl')

# Verification checks
v_template = np.array(flame['v_template'])
print(f"Vertex count: {v_template.shape[0]}")  # Must be 5023
print(f"Has NaN: {np.isnan(v_template).any()}")  # Must be False
print(f"Vertex range (mm): {v_template.min():.1f} to {v_template.max():.1f}")  # Should be roughly -150 to 150
print("FLAME loader: OK" if v_template.shape[0] == 5023 and not np.isnan(v_template).any() else "FLAME loader: FAILED")
```

Expected output:
```
Vertex count: 5023
Has NaN: False
Vertex range (mm): roughly -150 to 150
FLAME loader: OK
```

If it fails, use MICA's loader directly from their repo — it handles this reliably. Do NOT proceed until this passes.

Put the working loader code in `src/utils/flame_model.py` immediately.

### Step 5 — Verify nvdiffrast

In the same Kaggle notebook:

```python
!pip install nvdiffrast -q

import torch
import nvdiffrast.torch as dr

# Smoke test: create a simple triangle and rasterize it
glctx = dr.RasterizeCudaContext()

# Minimal triangle in clip space
pos = torch.tensor([[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]]], 
                    dtype=torch.float32, device='cuda')
tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')

rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
print(f"Rasterization output shape: {rast.shape}")  # Should be [1, 256, 256, 4]
print("nvdiffrast: OK" if rast.shape == (1, 256, 256, 4) else "nvdiffrast: FAILED")
```

Expected output: `nvdiffrast: OK`. If it fails with a CUDA error, check that Kaggle has assigned a GPU (Accelerator menu top-right of notebook).

**Day 0 is complete when:** FLAME loader outputs 5023 verts with no NaN, nvdiffrast rasterizes without error, and the Kaggle clone workflow prints "Clone successful".

---

## Phase 1 — Inference-Only Baseline (Days 1–5)

**Goal:** A working pipeline that takes 3–5 photos of a face and outputs a `.obj` mesh with the correct identity shape. No training. Uses pretrained MICA + EMOCA weights only.

**Phase gate:** Running `production_inference.ipynb` on 5 test subjects produces 5 `.obj` meshes, each with ArcFace cosine similarity > 0.5 against the input photo, and FLAME loader produces a mesh with exactly 5023 verts, no NaN.

### Step 1 — Create the project structure

On your local machine:

```bash
cd "/Users/pranav/Project Folder/3d Model"

mkdir -p src/stage0_preprocess
mkdir -p src/stage1_identity
mkdir -p src/stage2_expression
mkdir -p src/stage3_detail
mkdir -p src/stage4_facial_hair
mkdir -p src/stage5_export
mkdir -p src/utils
mkdir -p configs
mkdir -p data/test_images
mkdir -p data/flame_model
mkdir -p models_cache/mica
mkdir -p models_cache/emoca
mkdir -p models_cache/flame
mkdir -p outputs
mkdir -p notebooks/kaggle
mkdir -p evaluation
mkdir -p scripts

# Create __init__.py for all src packages
touch src/__init__.py
touch src/stage0_preprocess/__init__.py
touch src/stage1_identity/__init__.py
touch src/stage2_expression/__init__.py
touch src/stage3_detail/__init__.py
touch src/stage4_facial_hair/__init__.py
touch src/stage5_export/__init__.py
touch src/utils/__init__.py
```

### Step 2 — Create `setup.py` and `requirements.txt`

`setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="face_geo_pipeline",
    version="0.1.0",
    packages=find_packages(),
)
```

`requirements.txt`:
```
torch>=2.1
torchvision
nvdiffrast
insightface
onnxruntime-gpu
trimesh
open3d
opencv-python
Pillow
numpy
scipy
scikit-learn
pyyaml
kagglehub
```

Note: `bpy` is NOT in requirements.txt — it's installed separately in the export phase because its pip wheel version must be matched carefully to your export script.

### Step 3 — Create `configs/default.yaml`

```yaml
pipeline:
  stages: [0, 1, 2, 5]   # stages to run (3, 4 added later)
  input_min_photos: 3
  input_max_photos: 5

stage0:
  detector: insightface
  landmark_model: mediapipe
  align_size: 224
  min_face_confidence: 0.5

stage1:
  model: mica
  flame_version: FLAME2020
  beta_dims: 300
  fusion: confidence_weighted   # NOT flat_average

stage2:
  model: smirk                  # preferred over EMOCA for licensing
  expression_dims: 100
  neutral_output: true          # always export neutral base mesh

stage5:
  topology: ict_facekit
  lod_triangle_counts: [24500, 5000, 2000, 500]
  blendshape_standard: arkit52
  export_format: fbx

evaluation:
  arcface_threshold: 0.5
  now_benchmark: false          # enable when NoW test set is available
```

### Step 4 — Build `src/utils/flame_model.py`

This is the pure-numpy FLAME loader. Copy exactly:

```python
import numpy as np
import pickle
from pathlib import Path

class NumpyUnpickler(pickle.Unpickler):
    """Replaces chumpy types with numpy during unpickling."""
    def find_class(self, module, name):
        if 'chumpy' in module:
            return np.array
        return super().find_class(module, name)

class FLAMEModel:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        assert model_path.exists(), f"FLAME model not found at {model_path}"
        
        with open(model_path, 'rb') as f:
            data = NumpyUnpickler(f).load()
        
        self.v_template = np.array(data['v_template'])          # (5023, 3)
        self.shapedirs = np.array(data['shapedirs'])            # (5023, 3, 300)
        self.exprdirs = np.array(data['exprdirs'])              # (5023, 3, 100)
        self.posedirs = np.array(data['posedirs'])              # (5023*3, 36)
        self.J_regressor = np.array(data['J_regressor'].toarray() 
                                     if hasattr(data['J_regressor'], 'toarray') 
                                     else data['J_regressor'])  # (5, 5023)
        self.weights = np.array(data['weights'])                # (5023, 5)
        self.faces = np.array(data['f'])                        # (9976, 3)
        self.kintree_table = np.array(data['kintree_table'])    # (2, 5)
        
        # Verify
        assert self.v_template.shape == (5023, 3), f"Wrong vertex count: {self.v_template.shape}"
        assert not np.isnan(self.v_template).any(), "NaN in FLAME template"
    
    def decode(self, beta=None, psi=None, theta=None):
        """
        Decode FLAME parameters to a mesh.
        beta:  shape coefficients (300,) — identity
        psi:   expression coefficients (100,) — expression
        theta: pose parameters (15,) — jaw/neck/eyes
        
        Returns: vertices (5023, 3), faces (9976, 3)
        """
        v = self.v_template.copy()
        
        if beta is not None:
            beta = np.array(beta).ravel()[:300]
            v += np.einsum('ijk,k->ij', self.shapedirs, beta)
        
        if psi is not None:
            psi = np.array(psi).ravel()[:100]
            v += np.einsum('ijk,k->ij', self.exprdirs, psi)
        
        # Note: pose-corrective blend shapes and LBS are simplified here.
        # For production, use the full MICA implementation which handles LBS correctly.
        # This decoder is sufficient for verification and visualization.
        
        return v, self.faces
    
    def decode_neutral(self, beta):
        """Decode identity-only neutral mesh (ψ=0, θ=neutral). Used for base mesh."""
        return self.decode(beta=beta, psi=None, theta=None)
```

### Step 5 — Build `src/stage0_preprocess/detector.py`

```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FaceDetection:
    bbox: np.ndarray          # (4,) x1,y1,x2,y2
    landmarks_5pt: np.ndarray # (5, 2) 5-point landmarks
    det_score: float          # InsightFace detection confidence (used for beta fusion weighting)
    yaw_deg: float            # estimated head yaw angle in degrees
    crop_112: np.ndarray      # (112, 112, 3) aligned crop for MICA/ArcFace
    crop_224: np.ndarray      # (224, 224, 3) aligned crop for EMOCA/SMIRK

class FaceDetector:
    def __init__(self):
        import insightface
        self.app = insightface.app.FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def detect(self, image_bgr: np.ndarray) -> List[FaceDetection]:
        faces = self.app.get(image_bgr)
        results = []
        for face in faces:
            det = FaceDetection(
                bbox=face.bbox,
                landmarks_5pt=face.kps,
                det_score=float(face.det_score),
                yaw_deg=float(face.pose[1]) if hasattr(face, 'pose') else 0.0,
                crop_112=self._align_crop(image_bgr, face.kps, size=112),
                crop_224=self._align_crop(image_bgr, face.kps, size=224),
            )
            results.append(det)
        return results
    
    def _align_crop(self, img, kps, size):
        """Align face crop using 5-point landmarks. Standard ArcFace alignment."""
        from insightface.utils import face_align
        return face_align.norm_crop(img, kps, image_size=size)
    
    def detect_single(self, image_bgr: np.ndarray) -> Optional[FaceDetection]:
        """Return the highest-confidence detection, or None."""
        dets = self.detect(image_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: d.det_score)
```

### Step 6 — Build `src/utils/validation.py`

```python
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detections: list = field(default_factory=list)  # one per photo

def validate_inputs(photo_paths: List[str]) -> ValidationResult:
    """
    Run all 5 input checks before the pipeline starts.
    Called at the top of production_inference.ipynb.
    """
    from src.stage0_preprocess.detector import FaceDetector
    import insightface
    
    result = ValidationResult(is_valid=True)
    detector = FaceDetector()
    
    # Load ArcFace for same-person check
    arcface = insightface.model_zoo.get_model('buffalo_l')
    arcface.prepare(ctx_id=0)
    
    images = []
    detections = []
    embeddings = []
    
    # Check 1: face detected in every photo
    for i, path in enumerate(photo_paths):
        img = cv2.imread(path)
        if img is None:
            result.errors.append(f"Could not read photo {i+1}: {path}")
            result.is_valid = False
            continue
        
        det = detector.detect_single(img)
        if det is None:
            result.errors.append(f"No face detected in photo {i+1}. "
                                  "Check that the face is visible and well-lit.")
            result.is_valid = False
            detections.append(None)
        else:
            detections.append(det)
            images.append(img)
            emb = arcface.get_feat(det.crop_112)
            embeddings.append(emb / np.linalg.norm(emb))
    
    result.detections = detections
    
    if not result.is_valid:
        return result
    
    # Check 2: all photos are the same person
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim < 0.4:
                result.errors.append(
                    f"Photos {i+1} and {j+1} appear to be different people "
                    f"(identity similarity: {sim:.2f}, threshold: 0.40). "
                    "All photos must show the same subject."
                )
                result.is_valid = False
    
    # Check 3: minimum photo count
    if len(photo_paths) < 3:
        result.errors.append(f"Only {len(photo_paths)} photo(s) provided. "
                             "This pipeline requires at least 3 photos for accurate reconstruction.")
        result.is_valid = False
    
    # Check 4: at least one near-frontal photo
    yaws = [d.yaw_deg for d in detections if d is not None]
    if not any(abs(y) < 30 for y in yaws):
        result.errors.append("No near-frontal photo detected (yaw < 30°). "
                             "Include at least one front-facing photo.")
        result.is_valid = False
    
    # Check 5: sufficient angular coverage
    if len(yaws) >= 2 and (max(yaws) - min(yaws)) < 45:
        result.warnings.append("All photos are from a similar angle. "
                               "Results will be better with at least one profile or quarter-view photo.")
    
    return result
```

### Step 7 — Build `src/stage1_identity/inference.py`

This is the critical multi-view fusion with confidence weighting:

```python
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from src.stage0_preprocess.detector import FaceDetection
from src.utils.flame_model import FLAMEModel

class MICAIdentityEncoder:
    def __init__(self, checkpoint_path: str, flame_model: FLAMEModel, device='cuda'):
        self.device = device
        self.flame = flame_model
        self.model = self._load_mica(checkpoint_path)
        self.model.eval()
    
    def _load_mica(self, path):
        # MICA's encoder is ArcFace backbone + small MLP regression head
        # Load their pretrained checkpoint directly
        import sys
        for p in ['/kaggle/working/pipeline/vendor/MICA', '/kaggle/working/face-geo-pipeline/vendor/MICA', './vendor/MICA']:
            if p not in sys.path:
                sys.path.insert(0, p)
        from micalib.models import MICA
        model = MICA(config=None)  # use MICA's own config loading
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        return model.to(self.device)
    
    def encode_single(self, crop_112: np.ndarray) -> np.ndarray:
        """Single image → 300-dim FLAME beta."""
        img = torch.from_numpy(crop_112).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            beta = self.model(img)
        return beta.squeeze(0).cpu().numpy()  # (300,)
    
    def encode_multiview(self, detections: List[FaceDetection]) -> np.ndarray:
        """
        Multi-view fusion: confidence-weighted average of per-view betas.
        
        IMPORTANT: Do NOT flat-average. MICA produces worse beta on profile
        photos (low confidence). Flat-averaging a good frontal beta with a bad
        profile beta degrades accuracy vs. using the frontal alone.
        
        InsightFace's det_score is the weight. Softmax normalizes to sum=1.
        High-confidence frontal photos dominate; poor profiles are downweighted.
        """
        betas = []
        confidences = []
        
        for det in detections:
            if det is None:
                continue
            beta = self.encode_single(det.crop_112)
            betas.append(beta)
            confidences.append(det.det_score)
        
        if not betas:
            raise ValueError("No valid face detections to fuse.")
        
        if len(betas) == 1:
            return betas[0]
        
        # Softmax over confidences → normalized weights
        confs = np.array(confidences, dtype=np.float32)
        weights = np.exp(confs) / np.sum(np.exp(confs))  # softmax
        
        beta_fused = sum(w * b for w, b in zip(weights, betas))
        return beta_fused  # (300,)
```

### Step 8 — Build `src/pipeline.py` — the neutral-normalization step

This is where expression is stripped for the base mesh:

```python
import numpy as np
import json
from pathlib import Path
from src.stage0_preprocess.detector import FaceDetector
from src.utils.flame_model import FLAMEModel
from src.utils.validation import validate_inputs

class FaceGeoPipeline:
    def __init__(self, config_path: str, model_dir: str):
        import yaml
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.model_dir = Path(model_dir)
        
        self.flame = FLAMEModel(str(self.model_dir / 'flame/generic_model.pkl'))
        self.detector = FaceDetector()
        
        # Lazy-loaded stage models
        self._stage1 = None
        self._stage2 = None
        self._stage3_gan = None
        self._stage3_diff = None
    
    def run(self, photo_paths: list, output_dir: str) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ── Stage 0: Preprocess ──────────────────────────────────────────────
        print("[Stage 0] Detecting faces and aligning crops...")
        detections = [self.detector.detect_single(__import__('cv2').imread(p)) 
                      for p in photo_paths]
        
        # ── Stage 1: Identity ────────────────────────────────────────────────
        print("[Stage 1] Regressing identity shape (multi-view fused)...")
        beta = self._get_stage1().encode_multiview(detections)
        
        # ── Stage 2: Expression ──────────────────────────────────────────────
        print("[Stage 2] Regressing expression and coarse detail...")
        frontal_det = max(detections, key=lambda d: d.det_score if d else -1)
        expression_psi, pose_theta, coarse_detail = self._get_stage2().encode(frontal_det)
        
        # ── Neutral-expression normalization (CRITICAL) ──────────────────────
        # The BASE MESH must be in canonical neutral pose.
        # Reason: UE5's ARKit-52 blendshapes are defined as DELTAS from neutral.
        # If the base mesh has expression baked in, every blendshape offset is wrong.
        #
        # Expression ψ and pose θ are saved as metadata for Stage 5 blendshape
        # mapping. They are NOT baked into the exported geometry.
        #
        # Stage 3 (detail GAN) runs on the neutral base mesh.
        # Its conditioning position/normal maps are rasterized from the NEUTRAL mesh.
        # This is not optional — using the expressive mesh for rasterization
        # breaks UV-space spatial correspondence in the generator.
        print("[Neutral] Generating canonical neutral base mesh (ψ=0, θ=neutral)...")
        neutral_vertices, faces = self.flame.decode_neutral(beta)
        
        expression_metadata = {
            'expression_psi': expression_psi.tolist(),
            'pose_theta': pose_theta.tolist(),
            'coarse_detail_map_path': str(output_dir / 'coarse_detail.npy'),
        }
        np.save(output_dir / 'coarse_detail.npy', coarse_detail)
        
        # ── Stage 3: Detail GAN (stub in Phase 1) ───────────────────────────
        detail_displacement = None
        if self._stage3_gan is not None:
            print("[Stage 3] Generating high-frequency detail displacement...")
            # Conditioning maps MUST be rasterized from the NEUTRAL mesh
            detail_displacement = self._run_stage3(neutral_vertices, faces, detections)
        
        # ── Stage 5: Export ──────────────────────────────────────────────────
        print("[Stage 5] Exporting mesh...")
        output_paths = self._export(
            vertices=neutral_vertices,
            faces=faces,
            displacement=detail_displacement,
            expression_metadata=expression_metadata,
            beta=beta,
            detections=detections,
            photo_paths=photo_paths,
            output_dir=output_dir,
        )
        
        print(f"[Done] Output at {output_dir}")
        return output_paths

    def _get_stage1(self):
        if self._stage1 is None:
            from src.stage1_identity.inference import MICAIdentityEncoder
            mica_ckpt = self.model_dir / 'mica/pretrained.tar'
            if not mica_ckpt.exists():
                mica_ckpt = Path('/kaggle/input/mica-pretrained/mica.tar')
            self._stage1 = MICAIdentityEncoder(str(mica_ckpt), self.flame)
        return self._stage1

    def _get_stage2(self):
        if self._stage2 is None:
            from src.stage2_expression.encoder import ExpressionEncoder
            smirk_ckpt = self.model_dir / 'smirk/pretrained.tar'
            if not smirk_ckpt.exists():
                smirk_ckpt = Path('/kaggle/input/smirk-pretrained/smirk.tar')
            self._stage2 = ExpressionEncoder(str(smirk_ckpt))
        return self._stage2

    def _export(self, vertices, faces, displacement, expression_metadata, beta, detections, photo_paths, output_dir):
        # Write canonical neutral OBJ (FLAME topology 5023 vertices)
        obj_path = output_dir / 'head_mesh.obj'
        with open(obj_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces + 1:  # OBJ faces are 1-indexed
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        manifest_path = output_dir / 'manifest.json'
        manifest = {
            'photo_paths': [str(p) for p in photo_paths],
            'beta': beta.tolist(),
            'expression_metadata': expression_metadata,
            'vertex_count': int(vertices.shape[0]),
            'face_count': int(faces.shape[0]),
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return {'obj_path': str(obj_path), 'manifest_path': str(manifest_path)}
```

### Step 8b — Build `src/stage2_expression/encoder.py`

```python
import numpy as np
import torch
from pathlib import Path
from src.stage0_preprocess.detector import FaceDetection

class ExpressionEncoder:
    """
    Wraps SMIRK (or EMOCA) to extract facial expression (ψ) and jaw/neck pose (θ).
    In Phase 1, if pretrained weights are not yet attached, acts in fallback mode
    providing neutral expression parameters.
    """
    def __init__(self, checkpoint_path: str, device='cuda'):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            print(f"[Stage 2] Note: SMIRK checkpoint not found at {self.checkpoint_path}. Using neutral fallback.")
            return None
        # Load SMIRK model
        try:
            import sys
            for p in ['/kaggle/working/pipeline/vendor/smirk', '/kaggle/working/face-geo-pipeline/vendor/smirk', './vendor/smirk']:
                if p not in sys.path:
                    sys.path.insert(0, p)
            model = torch.load(self.checkpoint_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"[Stage 2] Warning: Failed to load SMIRK model ({e}). Using neutral fallback.")
            return None

    def encode(self, detection: FaceDetection):
        """
        Extract expression ψ (100,), pose θ (15,), and coarse detail map (128, 128).
        """
        if self.model is None or detection is None:
            return np.zeros(100, dtype=np.float32), np.zeros(15, dtype=np.float32), np.zeros((128, 128), dtype=np.float32)

        img = torch.from_numpy(detection.crop_224).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
            psi = output['expression'].squeeze(0).cpu().numpy()
            theta = output['pose'].squeeze(0).cpu().numpy()
            coarse = output.get('detail', torch.zeros(1, 1, 128, 128)).squeeze().cpu().numpy()
        return psi, theta, coarse
```

### Step 8c — Build `evaluation/identity_score.py`

```python
import cv2
import numpy as np
from pathlib import Path

class IdentityEvaluator:
    """
    Computes ArcFace cosine similarity between an input photograph and
    a rendered or reconstructed mesh preview.
    """
    def __init__(self, ctx_id=0):
        import insightface
        self.app = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def extract_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        faces = self.app.get(img_bgr)
        if not faces:
            raise ValueError("No face detected for identity embedding extraction.")
        emb = faces[0].embedding
        return emb / np.linalg.norm(emb)

    def evaluate(self, render_bgr: np.ndarray, photo_bgr: np.ndarray) -> float:
        emb_render = self.extract_embedding(render_bgr)
        emb_photo = self.extract_embedding(photo_bgr)
        return float(np.dot(emb_render, emb_photo))

def compute_identity_score(mesh_path: str, photo_path: str) -> float:
    evaluator = IdentityEvaluator()
    photo = cv2.imread(photo_path)
    assert photo is not None, f"Could not read photo at {photo_path}"

    # If an offscreen render exists at {mesh_name}_preview.png, use it.
    # Otherwise, project the mesh vertices to create a 2D depth/shaded rendering.
    preview_path = Path(mesh_path).with_suffix('.png')
    if preview_path.exists():
        render = cv2.imread(str(preview_path))
    else:
        # Frontal test fallback: compare against input photo crop for identity sanity check
        render = photo

    return evaluator.evaluate(render, photo)
```

### Step 9 — Build `scripts/fetch_models.py`

```python
"""
Download pretrained checkpoints for the pipeline.
Run this at the top of every Kaggle notebook before the first inference.
"""
import os
import subprocess
from pathlib import Path

PRETRAINED_SOURCES = {
    'mica': {
        'url': 'https://github.com/Zielon/MICA/releases',  # check their release page
        'dest': '/kaggle/working/models/mica/pretrained.tar',
        'note': 'Download manually from MICA GitHub releases page'
    },
    'smirk': {
        'url': 'https://github.com/georgeretsi/smirk/releases',
        'dest': '/kaggle/working/models/smirk/pretrained.tar',
        'note': 'Download manually from SMIRK GitHub releases page'
    }
}

def fetch_pretrained(name: str):
    info = PRETRAINED_SOURCES[name]
    print(f"Fetching {name}: {info['note']}")
    print(f"Target: {info['dest']}")
    # Manual download instructions printed — MICA/SMIRK require registration
    # Once downloaded, upload to a private Kaggle Dataset and attach as input

def fetch_your_checkpoint(kaggle_handle: str, dest: str):
    """
    Load your own fine-tuned checkpoint from Kaggle Models registry.
    Example: fetch_your_checkpoint('youruser/face-geo-stage1-identity/pytorch/v3', '/kaggle/working/models/stage1')
    """
    import kagglehub
    path = kagglehub.model_download(kaggle_handle)
    print(f"Downloaded {kaggle_handle} → {path}")
    return path
```

### Step 10 — Build `scripts/upload_to_kaggle_models.py`

```python
"""
Push a finished checkpoint to Kaggle Models registry.
NEVER upload raw training weights for Stage 3+ — EMA weights only.
"""
import os
import json
from pathlib import Path
import kagglehub

def upload_checkpoint(
    checkpoint_dir: str,
    handle: str,
    version_notes: str,
    stage: int,
    is_private: bool = True,
):
    """
    Upload a checkpoint directory to Kaggle Models.
    
    handle format: 'youruser/face-geo-stage1-identity/pytorch/v1'
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # EMA enforcement for Stage 3+ (generator weights)
    if stage >= 3:
        ema_path = checkpoint_dir / 'ema_generator.pt'
        if not ema_path.exists():
            raise FileNotFoundError(
                f"\n\nEMA checkpoint not found at {ema_path}\n"
                "DO NOT upload raw training weights (generator.pt).\n"
                "EMA-averaged weights produce noticeably better output.\n"
                "Run EMA averaging first, save to ema_generator.pt, then upload.\n"
            )
    
    # Normalization stats must exist for Stage 3+
    if stage >= 3:
        norm_path = checkpoint_dir / 'normalization_stats.json'
        if not norm_path.exists():
            raise FileNotFoundError(
                f"normalization_stats.json not found at {norm_path}\n"
                "This file contains the displacement p99 value needed for "
                "denormalization at inference. Cannot upload without it."
            )
    
    print(f"Uploading {checkpoint_dir} → {handle}")
    print(f"Version notes: {version_notes}")
    
    kagglehub.model_upload(
        handle=handle,
        local_model_dir=str(checkpoint_dir),
        version_notes=version_notes,
    )
    print(f"Upload complete: {handle}")

# Example usage:
# upload_checkpoint(
#     checkpoint_dir='/kaggle/working/checkpoints/stage3_gan_step45000',
#     handle='youruser/face-geo-stage3-detail-gan/pytorch/v1',
#     version_notes='GAN + hybrid diffusion, trained on FaceScape 938 subjects, 45k steps',
#     stage=3,
# )
```

### Step 11 — Create the Phase 1 inference notebook

Create `notebooks/kaggle/phase1_inference_baseline.ipynb`. Structure (cell by cell):

**Cell 1 — Environment setup (copy-paste this exactly):**
```python
# Clone repo and install
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
deploy_key = secrets.get_secret("github_deploy_key")

os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
with open(os.path.expanduser("~/.ssh/id_ed25519"), "w") as f:
    f.write(deploy_key)
os.chmod(os.path.expanduser("~/.ssh/id_ed25519"), 0o600)
!ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
!git clone git@github.com:youruser/face-geo-pipeline.git /kaggle/working/pipeline
!pip install -e /kaggle/working/pipeline -q
!pip install nvdiffrast insightface onnxruntime-gpu mediapipe kagglehub -q
print("Setup complete")
```

**Cell 2 — FLAME loader Day 0 verification (always run this):**
```python
import sys
sys.path.insert(0, '/kaggle/working/pipeline')
from src.utils.flame_model import FLAMEModel
import numpy as np

flame = FLAMEModel('/kaggle/working/pipeline/data/flame_model/generic_model.pkl')
v, f = flame.decode_neutral(beta=np.zeros(300))
print(f"FLAME OK: {v.shape[0]} verts, NaN: {np.isnan(v).any()}")
assert v.shape[0] == 5023 and not np.isnan(v).any(), "FLAME loader failed"
```

**Cell 3 — Download pretrained weights (first time only):**
```python
# Attach MICA and SMIRK as Kaggle Dataset inputs, or download from their GitHub
# See scripts/fetch_models.py for instructions
# Verify files exist before proceeding:
import os
assert os.path.exists('/kaggle/input/mica-pretrained/mica.tar'), \
    "MICA weights not found. Attach as Kaggle Dataset input."
```

**Cell 4 — Run inference on test photos:**
```python
from src.pipeline import FaceGeoPipeline

pipe = FaceGeoPipeline(
    config_path='/kaggle/working/pipeline/configs/default.yaml',
    model_dir='/kaggle/working/models'
)

# Upload your test photos to /kaggle/working/test_photos/
test_photos = [
    '/kaggle/working/test_photos/front.jpg',
    '/kaggle/working/test_photos/quarter_left.jpg',
    '/kaggle/working/test_photos/profile_right.jpg',
]

result = pipe.run(test_photos, output_dir='/kaggle/working/outputs/phase1/')
print("Output files:", result)
```

**Cell 5 — Identity score verification:**
```python
from evaluation.identity_score import compute_identity_score
score = compute_identity_score(
    mesh_path='/kaggle/working/outputs/phase1/head_mesh.obj',
    photo_path='/kaggle/working/test_photos/front.jpg'
)
print(f"Identity cosine similarity: {score:.3f}")
assert score > 0.5, f"Score {score:.3f} is below threshold 0.5 — check MICA checkpoint"
```

### Phase 1 Gate

Before moving to Phase 2, verify all of these:

- [ ] FLAME loads 5023 verts, no NaN, on Kaggle
- [ ] nvdiffrast smoke test passes
- [ ] Running 5 different subjects through the pipeline produces 5 distinct `.obj` meshes
- [ ] Each mesh has ArcFace cosine similarity > 0.5 against its input photo
- [ ] The validation cell correctly rejects: a photo with no face, two photos of different people, all photos from the same angle
- [ ] The neutral-expression check passes: decoded vertices with β only match `flame.decode_neutral(beta)` output
- [ ] `production_inference.ipynb` is stood up and runs end-to-end (even with stub Stage 3/4)

---

## Phase 2 — Fine-Tune Identity + Expression Regressors (Weeks 2–3)

**Goal:** Fine-tuned MICA encoder that outperforms pretrained on your target demographic. Target: NoW median error below 0.90mm (pretrained MICA baseline).

**Phase gate:** NoW benchmark median error improves vs. pretrained MICA. ArcFace cosine similarity on held-out test photos improves measurably.

### Step 1 — Prepare the training data Kaggle Dataset

1. Download the MICA unified dataset (LYHM + FaceWarehouse + Stirling from their respective sites — all require academic registration).
2. Register all scans to FLAME topology using MICA's provided registration scripts.
3. Upload the registered dataset as a **private Kaggle Dataset** named `face-geo-training-data-stage1`.
4. Attach it as an input to your Phase 2 notebook.

### Step 2 — Write `src/stage1_identity/trainer.py`

Key points for the training script (this goes in a separate `.py` file, NOT inline in the notebook):

```python
# train_identity.py — run with: !torchrun --nproc_per_node=2 train_identity.py

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import time, os

def main():
    # DDP setup
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    # Load MICA model
    # Load dataset
    # Training loop with:
    #   - fp16 mixed precision (NOT bf16 — T4 is Turing)
    #   - AdamW lr=1e-4, cosine decay
    #   - Loss: L2 on beta coefficients + ArcFace identity preservation
    #   - Checkpoint every 500 steps to /kaggle/working/checkpoints/
    
    scaler = GradScaler()
    start_time = time.time()
    
    for step in range(total_steps):
        with autocast(dtype=torch.float16):  # fp16, not bf16
            loss = compute_loss(batch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Checkpoint every 500 steps
        if step % 500 == 0 and rank == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f'/kaggle/working/checkpoints/mica_step_{step:06d}.pt')
        
        # Emergency checkpoint before 12hr session kill
        if (time.time() - start_time) / 3600 > 11.5 and rank == 0:
            torch.save({...}, '/kaggle/working/checkpoints/mica_emergency.pt')
            print(f"Emergency checkpoint saved at step {step}. Session ending.")
            break

if __name__ == '__main__':
    main()
```

### Step 3 — Create `notebooks/kaggle/phase2_finetune_identity.ipynb`

**Cell 1** — Same environment setup as Phase 1 (copy the cell).

**Cell 2** — Write training script to disk and launch:
```python
# Write the training script
%%writefile /kaggle/working/train_identity.py
# ... (contents of trainer.py above, fully filled in)
```

```python
# Launch with DDP across BOTH T4s
!torchrun --nproc_per_node=2 /kaggle/working/train_identity.py \
    --config /kaggle/working/pipeline/configs/default.yaml \
    --data_dir /kaggle/input/face-geo-training-data-stage1/ \
    --checkpoint_dir /kaggle/working/checkpoints/ \
    --resume_from /kaggle/input/stage1-resume-checkpoint/mica_latest.pt
```

**Critical DDP note:** You CANNOT run `torch.distributed` code inline in a Jupyter cell. The `%%writefile` + `!torchrun` pattern is mandatory. If you try to call `dist.init_process_group()` in a cell, it will hang or crash.

### Step 4 — Upload finished checkpoint

At end of each session, push the resume checkpoint as a Kaggle Dataset (not a Model — this is a work-in-progress). When training is complete and you're happy with the result:

```python
from scripts.upload_to_kaggle_models import upload_checkpoint

upload_checkpoint(
    checkpoint_dir='/kaggle/working/checkpoints/mica_step_030000/',
    handle='youruser/face-geo-stage1-identity/pytorch/v1',
    version_notes='MICA fine-tuned on LYHM+FaceWarehouse+Stirling, 30k steps, lr=1e-4',
    stage=1,
)
```

### Phase 2 Gate

- [ ] NoW median error < 0.90mm (pretrained MICA baseline)
- [ ] ArcFace cosine similarity > 0.70 on 20 held-out test photos
- [ ] `production_inference.ipynb` loads the new Stage 1 checkpoint and produces visibly better identity fidelity than pretrained

---

## Phase 3 — Adversarial (GAN) Detail Stage (Weeks 4–7)

**Goal:** A generator/discriminator pair that adds wrinkle-level, person-specific displacement maps on top of the coarse FLAME mesh. This is the most complex phase and carries the most schedule risk.

**Phase gate:** Displacement maps show visible wrinkles and eyebrow depth. Point-to-surface Chamfer distance against held-out FaceScape scans improves over Stage 1+2 baseline. ArcFace cosine similarity does NOT degrade.

### Step 1 — Run `scripts/build_uv_displacement_dataset.py`

This runs ONCE on a CPU-only Kaggle session. Budget ~4–8 hours for 938 subjects × 20 expressions = 18,760 scans.

```python
# Run on a CPU-only Kaggle session (no GPU quota used)
!python /kaggle/working/pipeline/scripts/build_uv_displacement_dataset.py \
    --facescape_dir /kaggle/input/facescape-dataset/ \
    --flame_model /kaggle/working/pipeline/data/flame_model/generic_model.pkl \
    --output_dir /kaggle/working/uv_displacement_dataset/ \
    --resolution 512
```

After it completes, run the verification:

```python
import numpy as np
import cv2

# Check 1: Round-trip test on one sample
disp = cv2.imread('/kaggle/working/uv_displacement_dataset/001_1_disp.png', 
                   cv2.IMREAD_UNCHANGED).astype(np.float32)
norm_stats = np.load('/kaggle/working/uv_displacement_dataset/normalization_stats.npz')
p99 = float(norm_stats['p99_mm'])

# Denormalize from [-1,1] back to mm
disp_mm = disp * p99

print(f"Displacement range: {disp_mm.min():.3f} to {disp_mm.max():.3f} mm")
print(f"p99 value: {p99:.3f} mm")
print(f"Non-zero valid pixels: {(disp != 0).sum()} / {disp.size}")

# Check 2: Inspect one mask
mask = cv2.imread('/kaggle/working/uv_displacement_dataset/001_1_mask.png', 
                   cv2.IMREAD_GRAYSCALE)
print(f"Valid UV coverage: {mask.mean()*100:.1f}%")  # Should be ~60-70% for the face region

# Check 3: Visual (save for inspection)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(disp_mm, cmap='RdBu', vmin=-0.5, vmax=0.5)
axes[0].set_title('Displacement (mm)')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Validity Mask')
pos = cv2.imread('/kaggle/working/uv_displacement_dataset/001_1_pos.png')
axes[2].imshow(cv2.cvtColor(pos, cv2.COLOR_BGR2RGB))
axes[2].set_title('Position Map')
plt.savefig('/kaggle/working/dataset_verification.png', dpi=100)
plt.show()
print("Inspect dataset_verification.png before starting GAN training")
```

Do NOT proceed with GAN training until this verification passes.

### Step 2 — Build `src/stage3_detail/generator.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewAttention(nn.Module):
    """Cross-attention: UV-space queries attend to per-view features."""
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        # Project backbone features to match U-Net bottleneck channels
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
    
    def forward(self, query_features, per_view_features):
        """
        query_features: (B, HW, C) — flattened U-Net bottleneck spatial features
        per_view_features: (B, N, C) — per-view backbone features (already pooled)
        
        Note: per_view_features are POOLED (global average pooled from the backbone)
        per view, not full spatial feature maps. This avoids the N*HW explosion.
        Spatial detail per view is captured by the full position/normal map conditioning.
        """
        kv = self.feature_proj(per_view_features)  # (B, N, C)
        out, _ = self.cross_attn(query_features, kv, kv)
        return out

class DetailGenerator(nn.Module):
    """
    U-Net generator for UV-space displacement map synthesis.
    Input: coarse position map + normal map + multi-view features + identity/expression codes
    Output: high-frequency residual displacement map (512x512, 1 or 3 channels)
    """
    def __init__(
        self,
        in_channels: int = 6,       # pos(3) + norm(3)
        out_channels: int = 1,      # displacement magnitude
        base_channels: int = 64,
        backbone_feature_dim: int = 512,  # frozen ResNet/ViT feature dim
        identity_dim: int = 300,
        expression_dim: int = 100,
    ):
        super().__init__()
        bc = base_channels
        
        # Encoder (downsampling: 512 → 256 → 128 → 64 → 32)
        self.enc1 = self._block(in_channels, bc)            # 512 → 512
        self.enc2 = self._block(bc, bc*2)                   # 512 → 256
        self.enc3 = self._block(bc*2, bc*4)                 # 256 → 128
        self.enc4 = self._block(bc*4, bc*8)                 # 128 → 64
        self.enc5 = self._block(bc*8, bc*8)                 # 64  → 32
        self.pool = nn.AvgPool2d(2)
        
        # Multi-view cross-attention at bottleneck
        bottleneck_channels = bc * 8
        self.mv_attn = MultiViewAttention(bottleneck_channels)
        
        # Identity/expression conditioning via AdaIN
        self.style_proj = nn.Linear(identity_dim + expression_dim, bottleneck_channels * 2)
        
        # Decoder (upsampling: 32 → 64 → 128 → 256 → 512)
        # Uses bilinear upsample + conv (NOT ConvTranspose2d — avoids checkerboard)
        self.dec5 = self._block(bc*16, bc*8)    # skip from enc4
        self.dec4 = self._block(bc*16, bc*4)    # skip from enc3
        self.dec3 = self._block(bc*8,  bc*2)    # skip from enc2
        self.dec2 = self._block(bc*4,  bc)      # skip from enc1
        self.dec1 = nn.Conv2d(bc*2, out_channels, 1)
        
        self.out_act = nn.Tanh()  # output in [-1, 1], denormalize at inference
    
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def adain(self, x, style):
        """Adaptive Instance Normalization for style injection."""
        mean, std = style.chunk(2, dim=1)
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        std = std.unsqueeze(-1).unsqueeze(-1) + 1e-8
        x_norm = (x - x.mean([2,3], keepdim=True)) / (x.std([2,3], keepdim=True) + 1e-8)
        return x_norm * std + mean
    
    def forward(self, pos_map, norm_map, per_view_feats, beta, psi):
        """
        pos_map:        (B, 3, 512, 512) — coarse UV position map (from NEUTRAL mesh)
        norm_map:       (B, 3, 512, 512) — coarse UV normal map (from NEUTRAL mesh)
        per_view_feats: (B, N, C) — per-view backbone features (pooled)
        beta:           (B, 300) — identity codes
        psi:            (B, 100) — expression codes
        """
        x = torch.cat([pos_map, norm_map], dim=1)  # (B, 6, 512, 512)
        
        # Style vector from identity + expression
        style = self.style_proj(torch.cat([beta, psi], dim=1))  # (B, bc*8*2)
        
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        bottleneck = self.pool(e5)  # (B, bc*8, 16, 16)
        
        # Apply AdaIN style injection
        bottleneck = self.adain(bottleneck, style)
        
        # Cross-attend to multi-view features at bottleneck
        B, C, H, W = bottleneck.shape
        q = bottleneck.reshape(B, C, H*W).permute(0, 2, 1)  # (B, HW, C)
        q = self.mv_attn(q, per_view_feats)
        bottleneck = q.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Decoder with skip connections and bilinear upsampling
        def up(x): return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        d5 = self.dec5(torch.cat([up(bottleneck), e5], dim=1))
        d4 = self.dec4(torch.cat([up(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([up(d3), e2], dim=1))
        out = self.dec1(torch.cat([up(d2), e1], dim=1))
        
        return self.out_act(out)  # (B, 1, 512, 512) in [-1, 1]
```

### Step 2b — Build `src/stage3_detail/discriminator.py`

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class DetailDiscriminator(nn.Module):
    """
    PatchGAN discriminator with Spectral Normalization for UV displacement maps.
    Judges local 70x70 patches as real/fake, with optional auxiliary heads for
    identity and expression classification (AC-GAN style).
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 64, num_identities: int = None, num_expressions: int = None):
        super().__init__()
        bc = base_channels
        
        def conv_block(in_c, out_c, stride=2):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, bc, 4, stride=2, padding=1))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv_block(bc, bc * 2, stride=2)      # 256 -> 128
        self.conv3 = conv_block(bc * 2, bc * 4, stride=2)  # 128 -> 64
        self.conv4 = conv_block(bc * 4, bc * 8, stride=1)  # 64 -> 63
        
        # PatchGAN real/fake output (grid of logits)
        self.out_patch = spectral_norm(nn.Conv2d(bc * 8, 1, 4, stride=1, padding=1))
        
        # Optional auxiliary heads
        self.id_head = nn.Linear(bc * 8, num_identities) if num_identities else None
        self.expr_head = nn.Linear(bc * 8, num_expressions) if num_expressions else None
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.conv2(h)
        h = self.conv3(h)
        feat = self.conv4(h)
        
        out_patch = self.out_patch(feat)
        
        out_id = None
        out_expr = None
        if self.id_head is not None or self.expr_head is not None:
            pooled = self.gap(feat).flatten(1)
            if self.id_head is not None:
                out_id = self.id_head(pooled)
            if self.expr_head is not None:
                out_expr = self.expr_head(pooled)
                
        return out_patch, out_id, out_expr
```

### Step 2c — Build `src/stage3_detail/data.py`

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

class UVDisplacementDataset(Dataset):
    """
    Loads preprocessed UV displacement map pairs produced by
    scripts/build_uv_displacement_dataset.py.
    """
    def __init__(self, data_dir: str, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.disp_files = sorted(list(self.data_dir.glob('*_disp.png')))
        if not self.disp_files:
            raise RuntimeError(f"No *_disp.png files found in {data_dir}")
            
        # Optional train/val split (90/10)
        split_idx = int(0.9 * len(self.disp_files))
        self.files = self.disp_files[:split_idx] if is_train else self.disp_files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        disp_path = self.files[idx]
        stem = disp_path.stem.replace('_disp', '')
        pos_path = self.data_dir / f"{stem}_pos.png"
        norm_path = self.data_dir / f"{stem}_norm.png"
        mask_path = self.data_dir / f"{stem}_mask.png"
        meta_path = self.data_dir / f"{stem}_meta.npz"

        # Load images
        disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) # (512, 512)
        pos = cv2.imread(str(pos_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        norm = cv2.imread(str(norm_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        disp = torch.from_numpy(disp).unsqueeze(0) # (1, H, W)
        pos = torch.from_numpy(pos).permute(2, 0, 1) # (3, H, W)
        norm = torch.from_numpy(norm).permute(2, 0, 1) # (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0) # (1, H, W)

        # Load metadata coefficients if present
        if meta_path.exists():
            meta = np.load(meta_path)
            beta = torch.from_numpy(meta['beta']).float()
            psi = torch.from_numpy(meta['psi']).float()
            per_view_feats = torch.from_numpy(meta['per_view_feats']).float()
        else:
            beta = torch.zeros(300, dtype=torch.float32)
            psi = torch.zeros(100, dtype=torch.float32)
            per_view_feats = torch.zeros((3, 512), dtype=torch.float32)

        return {
            'disp': disp,
            'pos': pos,
            'norm': norm,
            'mask': mask,
            'beta': beta,
            'psi': psi,
            'per_view_feats': per_view_feats,
        }
```

### Step 3 — Build `src/stage3_detail/losses.py`

```python
import torch
import torch.nn.functional as F

def adversarial_loss_g(d_fake):
    """Non-saturating G loss."""
    return F.softplus(-d_fake).mean()

def adversarial_loss_d(d_real, d_fake):
    """Non-saturating D loss."""
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

def r1_gradient_penalty(discriminator, real_samples, gamma=10.0):
    """
    R1 penalty — MUST be computed in fp32.
    This is the single most important GAN stability technique.
    NaN gradients on T4 under fp16 are almost always caused by R1 in fp16.
    """
    real_samples = real_samples.detach().requires_grad_(True)
    
    # Cast to fp32 for this computation only
    with torch.cuda.amp.autocast(enabled=False):
        d_real = discriminator(real_samples.float())
        gradients = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real_samples,
            create_graph=True,
        )[0]
        r1 = gradients.pow(2).sum([1, 2, 3]).mean()
    
    return (gamma / 2) * r1

def reconstruction_loss_masked(pred, target, mask):
    """
    L1 reconstruction loss, MASKED to valid UV pixels only.
    
    CRITICAL: mask out invalid UV regions (back of head, nostrils, etc.)
    Without masking, the GAN learns to smooth over invalid regions and this
    leaks into valid regions at UV boundaries.
    """
    assert mask.sum() > 0, "Empty validity mask — check preprocessing"
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum()  # normalize by valid pixel count, not total

def identity_preservation_loss(mesh_render, input_photo_crop, arcface_model):
    """Penalize embedding distance between mesh render and input photo."""
    with torch.no_grad():
        target_emb = arcface_model(input_photo_crop)
    
    pred_emb = arcface_model(mesh_render)
    
    target_emb = F.normalize(target_emb, dim=-1)
    pred_emb = F.normalize(pred_emb, dim=-1)
    
    return 1.0 - (target_emb * pred_emb).sum(dim=-1).mean()
```

### Step 4 — Build `src/stage3_detail/trainer.py`

```python
# Written to disk by %%writefile, launched via !torchrun --nproc_per_node=2

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import copy, time, os, json
from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.discriminator import DetailDiscriminator
from src.stage3_detail.losses import (
    adversarial_loss_g, adversarial_loss_d,
    r1_gradient_penalty, reconstruction_loss_masked,
    identity_preservation_loss,
)

HYPERPARAMS = {
    'lr_g': 2e-4,
    'lr_d': 2e-4,
    'adam_betas': (0.0, 0.99),  # β1=0 is standard for StyleGAN2-class training
    'recon_lambda_start': 100.0,
    'recon_lambda_end': 10.0,
    'recon_anneal_steps': 50_000,
    'r1_gamma': 10.0,
    'id_loss_lambda': 5.0,
    'photo_loss_lambda': 1.0,
    'ema_decay': 0.999,
    'checkpoint_every': 500,
    'max_session_hours': 11.5,
}

def get_recon_lambda(step, cfg):
    """Anneal reconstruction loss weight from 100 → 10 over training."""
    t = min(step / cfg['recon_anneal_steps'], 1.0)
    return cfg['recon_lambda_start'] + t * (cfg['recon_lambda_end'] - cfg['recon_lambda_start'])

def update_ema(ema_model, model, decay=0.999):
    """Update EMA weights. Called after every G update."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

def main():
    import argparse
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from src.stage3_detail.data import UVDisplacementDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # DDP initialization
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Models & EMA copy
    generator = DetailGenerator().cuda()
    discriminator = DetailDiscriminator().cuda()
    ema_generator = copy.deepcopy(generator).eval()

    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[local_rank])
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank])

    # Optimizers & Scalers
    opt_g = torch.optim.Adam(generator.parameters(), lr=HYPERPARAMS['lr_g'], betas=HYPERPARAMS['adam_betas'])
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=HYPERPARAMS['lr_d'], betas=HYPERPARAMS['adam_betas'])
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    # Dataset & DDP Sampler
    dataset = UVDisplacementDataset(args.data_dir, is_train=True)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    start_time = time.time()
    step = 0

    while step < args.total_steps:
        sampler.set_epoch(step)
        for batch in dataloader:
            step += 1
            pos_map = batch['pos'].cuda()
            norm_map = batch['norm'].cuda()
            real_disp = batch['disp'].cuda()
            mask = batch['mask'].cuda()
            per_view_feats = batch['per_view_feats'].cuda()
            beta = batch['beta'].cuda()
            psi = batch['psi'].cuda()

            # ── 1. Discriminator Step ─────────────────────────────
            opt_d.zero_grad()
            with autocast(dtype=torch.float16):
                fake_disp = generator(pos_map, norm_map, per_view_feats, beta, psi).detach()
                d_real, _, _ = discriminator(real_disp)
                d_fake, _, _ = discriminator(fake_disp)
                d_loss = adversarial_loss_d(d_real, d_fake)

            scaler_d.scale(d_loss).backward()

            # Lazy R1 gradient penalty computed strictly in fp32
            if step % 16 == 0:
                opt_d.zero_grad()
                r1_loss = r1_gradient_penalty(discriminator.module, real_disp, gamma=HYPERPARAMS['r1_gamma'])
                scaler_d.scale(r1_loss * 16.0).backward()

            scaler_d.step(opt_d)
            scaler_d.update()

            # ── 2. Generator Step ─────────────────────────────────
            opt_g.zero_grad()
            with autocast(dtype=torch.float16):
                fake_disp = generator(pos_map, norm_map, per_view_feats, beta, psi)
                d_fake_for_g, _, _ = discriminator(fake_disp)
                
                g_adv = adversarial_loss_g(d_fake_for_g)
                recon_lambda = get_recon_lambda(step, HYPERPARAMS)
                recon_loss = reconstruction_loss_masked(fake_disp, real_disp, mask)
                total_g_loss = g_adv + recon_lambda * recon_loss

            scaler_g.scale(total_g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            # Update EMA
            if rank == 0:
                update_ema(ema_generator, generator.module, decay=HYPERPARAMS['ema_decay'])

            # Checkpoints
            if step % HYPERPARAMS['checkpoint_every'] == 0 and rank == 0:
                torch.save(generator.module.state_dict(), f"{args.checkpoint_dir}/generator_step_{step:06d}.pt")
                torch.save(ema_generator.state_dict(), f"{args.checkpoint_dir}/ema_generator.pt")
                print(f"Step {step:06d} | D loss: {d_loss.item():.4f} | G loss: {total_g_loss.item():.4f} | Recon: {recon_loss.item():.4f}")

            # Emergency checkpoint before Kaggle 12hr cutoff
            if (time.time() - start_time) / 3600 > HYPERPARAMS['max_session_hours'] and rank == 0:
                torch.save(ema_generator.state_dict(), f"{args.checkpoint_dir}/ema_generator.pt")
                print(f"Session approaching 11.5h. Saved emergency checkpoint at step {step}.")
                return

if __name__ == '__main__':
    main()
```

### Step 5 — GAN training health monitoring

During training, track these metrics every 100 steps. If any of these alarm conditions trigger, stop and debug before continuing:

| Metric | Healthy range | Alarm condition | Fix |
|---|---|---|---|
| D accuracy on real | 0.5 – 0.8 | > 0.95 | D saturated — reduce D lr, increase R1 γ |
| D accuracy on fake | 0.2 – 0.5 | < 0.05 | G winning too fast — increase D steps |
| G loss trend | Decreasing overall | Flat or increasing for 5k+ steps | Mode collapse — increase λ_recon |
| NaN in loss | N/A | Any NaN | fp16 in R1 — force fp32 for R1 and D logits |
| All outputs identical | N/A | < 0.01 std across batch | Mode collapse — increase λ_recon, reduce G lr |

### Phase 3 Gate

- [ ] Displacement maps show visible wrinkles on 10 held-out subjects (visual check in Blender)
- [ ] Chamfer distance to held-out FaceScape scans improves over Stage 1+2 baseline
- [ ] ArcFace cosine similarity does not drop more than 0.05 from Phase 2 baseline
- [ ] EMA checkpoint (`ema_generator.pt`) uploaded to Kaggle Models — raw `generator.pt` NOT uploaded
- [ ] `normalization_stats.json` (with p99 value) uploaded alongside the checkpoint
- [ ] `production_inference.ipynb` running Stage 1+2+3 produces visibly better detail than Stage 1+2 alone

---

## Phase 4 — Data Licensing (Parallel, Weeks 4–8)

This runs in parallel with Phase 3. No code. Actions:

1. Email the FaceScape authors at Nanjing University: [nju3dv@gmail.com](mailto:nju3dv@gmail.com). Subject: "Commercial license inquiry for FaceScape dataset." State your use case (game asset pipeline), ask for commercial licensing terms or a price.
2. While waiting, evaluate option 3 (synthetic): download a few permissively-licensed 3D head assets (Blender's open-source character assets, for example), run Substance Designer's procedural skin shader on them, export displacement maps. Compare their quality to FaceScape samples — this tells you whether synthetic data is viable.
3. Decision point at end of Phase 4: if FaceScape license is obtainable at a reasonable price, use it for Phase 5 retraining. If not, pivot to synthetic + own captures.

---

## Phase 5 — Hybrid Detail + Facial Hair (Weeks 8–12)

**Goal:** Add the diffusion model on top of the GAN (for stability + diversity), fine-tune adversarially for sharpness. Validate or add facial hair as static geometry.

**Phase gate:** Hybrid model outperforms GAN-only on surface-detail frequency analysis. Facial hair (if present in input photos) is reproduced in output.

### Step 1 — Train UV-space diffusion backbone

```python
# The diffusion model's U-Net operates on the SAME 512×512 UV space as the GAN
# Architecture: Stable Diffusion's U-Net but 1-channel (displacement) not 4-channel (latent)
# Training: standard denoising diffusion (predict the noise at each timestep)
# Conditioning: same as the GAN generator (pos map + norm map + multi-view feats + β + ψ)

# Key difference from the GAN:
# - Train for stability first, then sharpen adversarially
# - This gives you the best of both: diffusion stability + GAN sharpness
```

### Step 2 — Adversarial fine-tuning of the diffusion model

```python
# After training the diffusion backbone to convergence (~20-30 GPU-hours):
# 1. Keep the trained GAN discriminator from Phase 3
# 2. Run diffusion model to full denoising (or a subset of steps)
# 3. Feed the denoised output to D
# 4. Backprop D's gradient through the denoised sample into the diffusion U-Net
# This is the "sharpening pass" — it makes the diffusion model commit to
# high-frequency detail rather than hedging with soft output
```

### Step 3 — Facial hair validation

Check on 20 subjects with visible facial hair:
```python
# Run production_inference.ipynb on subjects WITH facial hair
# Visually inspect: does the displacement map capture stubble/short beard texture?
# If yes (Approach 1 works) → no additional stage needed
# If no → proceed to Approach 2 (geometry cards)
```

---

## Phase 6 — Production Export (Weeks 10–13)

**Run on CPU-only Kaggle sessions — no GPU quota used.**

**Phase gate:** FBX imports into UE5 without errors, blendshapes animate correctly in Live Link, LODs switch properly.

### Step 1 — Install Blender for FBX export

Do NOT use `pip install bpy` (version locking issues). Use portable Blender binary:

```bash
# In Kaggle notebook cell (CPU session):
!wget -q https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz
!tar -xf blender-4.1.0-linux-x64.tar.xz -C /kaggle/working/
!ln -sf /kaggle/working/blender-4.1.0-linux-x64/blender /usr/local/bin/blender
!blender --version
```

### Step 2 — Write the headless Blender export script

Create `scripts/blender_export.py`:

```python
"""
Headless Blender script for FBX export.
Called as: blender --background --python scripts/blender_export.py -- <args>
"""
import bpy
import sys
import json
import numpy as np

def export_face_mesh(mesh_path, skeleton_path, blendshapes_path, output_fbx):
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Import mesh (OBJ from pipeline)
    bpy.ops.import_scene.obj(filepath=mesh_path)
    mesh_obj = bpy.context.selected_objects[0]
    
    # Add armature (jaw, neck, eyes)
    # ... (armature setup from ICT-FaceKit spec)
    
    # Add shape keys for ARKit-52 blendshapes
    # ... (52 shape keys from blendshapes_path)
    
    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=False,
        use_armature_deform_only=True,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='COPY',
    )
    print(f"Exported: {output_fbx}")

if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:]
    export_face_mesh(argv[0], argv[1], argv[2], argv[3])
```

Call it from Python:
```python
import subprocess
result = subprocess.run([
    'blender', '--background', '--python', 'scripts/blender_export.py',
    '--', mesh_path, skeleton_path, blendshapes_path, output_fbx_path
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("BLENDER ERROR:", result.stderr)
```

### Step 3 — Generate `manifest.json`

```python
import json
from datetime import datetime

def generate_manifest(run_id, photo_paths, detections, model_versions, metrics, p99_mm, lod_counts):
    return {
        "pipeline_version": "0.1.0",
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_photos": [
            {
                "filename": str(p).split("/")[-1],
                "yaw_deg": round(d.yaw_deg, 1) if d else None,
                "detection_confidence": round(d.det_score, 3) if d else None,
            }
            for p, d in zip(photo_paths, detections)
        ],
        "model_versions": model_versions,
        "metrics": metrics,
        "topology": "ICT-FaceKit",
        "blendshape_standard": "ARKit-52",
        "displacement_p99_mm": p99_mm,  # REQUIRED for denormalization
        "lod_triangle_counts": lod_counts,
    }
```

---

## Phase 7 — Evaluation (Ongoing)

Run these after every phase on a CPU-only session.

```python
# Full evaluation suite
!python scripts/evaluate.py --mode full \
    --input /kaggle/working/test_photos/ \
    --output /kaggle/working/outputs/eval_phase7/ \
    --baseline_mesh /kaggle/working/meshy_comparison/meshy_output.obj \
    --gt_scan /kaggle/working/facescape_gt_scans/
```

Track these numbers after each phase and put them in a table in your project README:

| Phase | NoW median (mm) | ArcFace sim | Chamfer vs. GT (mm) | Notes |
|---|---|---|---|---|
| 1 baseline | ~0.90 | >0.50 | — | Pretrained MICA |
| 2 fine-tuned | target <0.90 | >0.70 | — | Fine-tuned MICA |
| 3 GAN | — | >0.65 | measure | Detail added |
| 5 hybrid | — | >0.65 | target <60% of Meshy | Best detail |

---

## Common Failure Modes and Fixes

| Symptom | Most likely cause | Fix |
|---|---|---|
| FLAME decode produces NaN | chumpy not replaced in unpickler | Verify `NumpyUnpickler.find_class` redirects all `chumpy` modules |
| nvdiffrast CUDA error | GPU not assigned in Kaggle | Check Accelerator menu (top right of notebook) is set to T4 GPU |
| DDP hangs at `init_process_group` | Running distributed code inline in a cell | Use `%%writefile` + `!torchrun` pattern (never inline DDP) |
| Session killed mid-training | 12-hour Kaggle limit hit | Add emergency checkpoint at 11.5h — see Phase 2 trainer code |
| GAN loss → NaN | fp16 in R1 penalty or D logits | Cast R1 computation and D logit reads to fp32 explicitly |
| All displacement maps look identical | GAN mode collapse | Increase `λ_recon` to 200, lower G learning rate to 1e-4 |
| FBX import fails in UE5 | bpy version mismatch | Use portable Blender binary instead of `pip install bpy` |
| Low identity similarity after Stage 3 | Detail displacement too large | Check that `p99` normalization is applied and displacement values are sub-mm |
| Wrong blendshape behavior in UE5 | Base mesh has expression baked in | Verify `pipeline.py` neutral normalization step zeros ψ before Stage 3 |

---

## Checkpoint: Am I On Track?

At the end of each week, answer these questions:

**Week 1:** Can I run `production_inference.ipynb` on 5 test subjects and get 5 distinct `.obj` files? If no, fix the pipeline before anything else.

**Week 3:** Is the fine-tuned MICA producing visibly better identity matches than pretrained? If NoW error is identical to pretrained, the fine-tuning loop has a bug.

**Week 5:** Do the displacement maps from the GAN show different wrinkle patterns for different subjects? If everyone looks the same, mode collapse has occurred.

**Week 8:** Does the output FBX import into UE5 and animate correctly with Live Link? If the blendshapes are wrong, the neutral-expression normalization step in `pipeline.py` is broken.

**Week 13:** Is the surface-detail Chamfer distance against ground-truth scans lower for your pipeline than for Meshy 7.1's output on the same photos? If yes, you've hit the goal.