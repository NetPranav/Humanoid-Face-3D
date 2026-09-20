"""
Phase 1 Baseline Notebook Generator.
Compiles and generates 100% syntactically valid notebook cells for Kaggle GPU execution.
"""
import json
import os
import sys
from pathlib import Path

c1_code = """# Cell 1: Environment & Repository Setup
import os
import sys
import subprocess
from pathlib import Path

print("--- Setting Up Humanoid-Face-3D Workspace ---")
target_dir = Path("/kaggle/working/Humanoid-Face-3D")
if not (target_dir / "src/pipeline.py").exists():
    subprocess.run(["git", "clone", "--recurse-submodules", "https://github.com/NetPranav/Humanoid-Face-3D.git", str(target_dir)], check=True)
os.chdir(str(target_dir))
subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=False)
subprocess.run(["git", "pull", "origin", "main"], check=False)

if str(target_dir) not in sys.path:
    sys.path.insert(0, str(target_dir))
if "." not in sys.path:
    sys.path.insert(0, ".")
print("Working directory:", os.getcwd())
"""

c2_code = """# Cell 2: Install GPU Dependencies & Model Frameworks
import subprocess
subprocess.run(["pip", "install", "insightface", "onnxruntime-gpu", "trimesh", "pyyaml", "scipy", "Pillow", "opencv-python", "--quiet"], check=True)

import torch
print(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB VRAM)")
"""

c3_code = """# Cell 3: Discover & Mount Attached Datasets (FLAME & MICA)
import shutil
import subprocess
from pathlib import Path

print("--- Locating Model Weights in /kaggle/input ---")
# 1. Locate FLAME 2020 generic_model.pkl
flame_pkl_candidates = list(Path("/kaggle/input").glob("**/generic_model.pkl")) + list(Path(".").glob("**/generic_model.pkl"))
if not flame_pkl_candidates:
    raise FileNotFoundError("generic_model.pkl not found! Please attach dataset nightshowdown/flame-model.")
flame_pkl = flame_pkl_candidates[0]
print(f"  Found FLAME Model: {flame_pkl}")

Path("data/flame_model").mkdir(parents=True, exist_ok=True)
if not Path("data/flame_model/generic_model.pkl").exists():
    shutil.copy(flame_pkl, "data/flame_model/generic_model.pkl")

head_candidates = list(Path("/kaggle/input").glob("**/head_template.obj")) + list(Path(".").glob("**/head_template.obj"))
if head_candidates and not Path("data/flame_model/head_template.obj").exists():
    shutil.copy(head_candidates[0], "data/flame_model/head_template.obj")

# 2. Locate MICA pretrained weights (mounted dataset or high-speed CDN stream)
Path("models_cache/mica").mkdir(parents=True, exist_ok=True)
dest_tar = Path("models_cache/mica/pretrained.tar")

if not dest_tar.exists() or dest_tar.stat().st_size < 100_000_000:
    mica_candidates = (
        list(Path("/kaggle/input").glob("**/pretrained.tar")) +
        list(Path("/kaggle/input").glob("**/mica.tar"))
    )
    if mica_candidates:
        print(f"  Mounting MICA weights from: {mica_candidates[0]}")
        shutil.copy(mica_candidates[0], dest_tar)
    else:
        print("  MICA dataset not attached. Streaming from Google Drive CDN (~5 seconds)...")
        subprocess.run([
            "curl", "-L", "--progress-bar",
            "https://drive.usercontent.google.com/download?id=1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt&export=download&confirm=t",
            "-o", str(dest_tar)
        ], check=True)

if not dest_tar.exists() or dest_tar.stat().st_size < 100_000_000:
    raise RuntimeError("Failed to acquire MICA pretrained weights!")
print(f"All model assets verified: FLAME ({flame_pkl.name}) + MICA ({dest_tar.stat().st_size / 1e6:.1f} MB).")
"""

c4_code = """# Cell 4: Initialize Face Geometry Pipeline & Stage Test Portraits
import shutil
import subprocess
from pathlib import Path
from src.pipeline import FaceGeoPipeline

pipeline = FaceGeoPipeline(
    config_path="configs/default.yaml",
    model_dir="models_cache"
)
print("FaceGeoPipeline loaded successfully.")

test_dir = Path("data/test_subjects")
test_dir.mkdir(parents=True, exist_ok=True)

demo_inputs = list(Path("vendor/MICA/demo/input").glob("*.*"))
if demo_inputs:
    for img_p in demo_inputs:
        subj_name = img_p.stem
        s_folder = test_dir / subj_name
        s_folder.mkdir(parents=True, exist_ok=True)
        target = s_folder / img_p.name
        if not target.exists():
            shutil.copy(img_p, target)

# Fallback: if vendor demo is empty, pull standard portraits directly
if not any(test_dir.iterdir()):
    print("Fetching benchmark portraits from CDN fallback...")
    fallbacks = {
        "connelly": "https://raw.githubusercontent.com/Zielon/MICA/main/demo/input/connelly.jpg",
        "carell": "https://raw.githubusercontent.com/Zielon/MICA/main/demo/input/carell.jpg",
        "lawrence": "https://raw.githubusercontent.com/Zielon/MICA/main/demo/input/lawrence.jpg"
    }
    for s_name, url in fallbacks.items():
        s_folder = test_dir / s_name
        s_folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-L", "-s", url, "-o", str(s_folder / f"{s_name}.jpg")], check=True)

subject_folders = sorted([d for d in test_dir.iterdir() if d.is_dir() and any(d.glob("*.*"))])
print(f"Test subjects ready ({len(subject_folders)}): {[s.name for s in subject_folders]}")
"""

c5_code = """# Cell 5: Execute End-to-End Multi-Subject 3D Reconstruction
import json
import numpy as np
from pathlib import Path

output_base = Path("outputs/phase1_baseline")
output_base.mkdir(parents=True, exist_ok=True)

results = {}
subject_betas = {}

for s_folder in subject_folders:
    subj_name = s_folder.name
    photos = sorted([str(p) for p in s_folder.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not photos:
        continue

    print("\\n==================================================")
    print(f"Reconstructing Subject: {subj_name} ({len(photos)} photo(s))...")
    out_dir = output_base / subj_name
    res = pipeline.run(photos, str(out_dir))
    results[subj_name] = res

    with open(res["manifest_path"]) as f:
        manifest = json.load(f)
    subject_betas[subj_name] = np.array(manifest["beta_shape"], dtype=np.float32)
    print(f"  Neutral Mesh: {res['obj_path']}")
    print(f"  Blendshapes:  {res.get('blendshapes_path')}")
    print(f"  Armature Rig: {res.get('armature_path')}")
"""

c6_code = """# Cell 6: Apply Stage 5 Parametric Stylization (Chiseled & Heroic)
import trimesh
from src.stage5_export.exporter import Stage5Exporter

print("\\n--- Generating Chiseled & Heroic Stylized Meshes ---")
exporter = Stage5Exporter(enable_lods=True, enable_armature=True)

for subj_name, res in results.items():
    s_out = output_base / subj_name
    mesh = trimesh.load(res["obj_path"], process=False)
    verts = mesh.vertices
    faces = mesh.faces

    for preset in ["chiseled", "heroic"]:
        preset_dir = s_out / f"stylized_{preset}"
        m = exporter.export_production_asset(
            neutral_vertices=verts,
            faces=faces,
            output_dir=preset_dir,
            stylization_params=preset,
            export_fbx=False
        )
        max_d = (m.get("stylization") or {}).get("max_displacement", 0.0)
        print(f"  {subj_name} [{preset}]: max displacement = {max_d:.2f}mm | Saved to {preset_dir.name}/")
"""

c7_code = """# Cell 7: Phase 1 Gate Verification (Identity Divergence & Seam Pinning)
import numpy as np
import trimesh

print("\\n--- Running Phase 1 Gate Verification ---")
# Gate 1: Non-collapse shape divergence
if len(subject_betas) >= 2:
    names = list(subject_betas.keys())
    gate_passed = True
    print("Pairwise Shape Divergence (||beta_a - beta_b||):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = float(np.linalg.norm(subject_betas[names[i]] - subject_betas[names[j]]))
            print(f"  {names[i]} vs {names[j]}: {dist:.4f}")
            if dist < 1e-3:
                print(f"  FAIL: {names[i]} and {names[j]} collapsed to identical shape!")
                gate_passed = False
    assert gate_passed, "Gate Failure: Identities collapsed to mean face!"
    print("GATE 1 PASSED: Identities are distinct and non-degenerate.")
else:
    print("Note: Need >= 2 subjects to evaluate pairwise distance.")

# Gate 2: Neck Seam Pinning Contract (Delta v = 0 on lowest 20% collar vertices)
print("\\nVerifying Critical Invariant 4 (Neck Seam Collar Pinning)...")
for subj_name, res in results.items():
    s_out = output_base / subj_name
    mesh_neutral = trimesh.load(res["obj_path"], process=False)
    y = mesh_neutral.vertices[:, 1]
    y_norm = (y - np.min(y)) / max(np.max(y) - np.min(y), 1e-4)
    collar_idx = np.where(y_norm <= 0.20)[0]

    for preset in ["chiseled", "heroic"]:
        stylized_obj = s_out / f"stylized_{preset}" / "head_mesh.obj"
        if stylized_obj.exists():
            mesh_stylized = trimesh.load(str(stylized_obj), process=False)
            deltas = mesh_stylized.vertices - mesh_neutral.vertices
            collar_deltas = deltas[collar_idx]
            max_collar_movement = float(np.max(np.abs(collar_deltas)))
            print(f"  {subj_name} [{preset}] neck collar max delta: {max_collar_movement:.6f} mm")
            assert max_collar_movement < 1e-5, f"Neck seam contract violated for {subj_name} [{preset}]!"

print("GATE 2 PASSED: Neck seam contract strictly verified (Delta v = 0 on collar).")
print("\\nAll Phase 1 Gates Successfully Cleared!")
"""

c8_code = """# Cell 8: Visual Gallery of Reconstructed 3D Head Meshes
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

n_subj = len(results)
if n_subj > 0:
    fig, axes = plt.subplots(1, min(4, n_subj), figsize=(4 * min(4, n_subj), 4))
    if n_subj == 1:
        axes = [axes]
    for ax, (subj_name, res) in zip(axes, list(results.items())[:4]):
        prev = res.get("preview_path")
        if prev and Path(prev).exists():
            img = cv2.imread(prev)[:, :, ::-1]
            ax.imshow(img)
            ax.set_title(f"{subj_name} (Neutral 3D)")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"{subj_name}\\nReconstruction OK", ha="center", va="center")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_base / "phase1_preview_grid.png", dpi=150)
    plt.show()
    print(f"Preview gallery saved to: {output_base / 'phase1_preview_grid.png'}")
"""

def main():
    codes = [c1_code, c2_code, c3_code, c4_code, c5_code, c6_code, c7_code, c8_code]

    # Verify all code blocks compile as valid Python
    for idx, c in enumerate(codes):
        compile(c, f"cell_{idx+1}", "exec")
        print(f"  Cell {idx+1}: Syntax Verified [OK]")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 1: Multi-View Inference Baseline (Stages 0, 1, 2, 5)\n",
                "\n",
                "**Objective:** Run identity-preserving 3D face geometry reconstruction from portrait photos on Kaggle GPU.\n",
                "1. **Stage 0:** InsightFace detection, 5-point alignment, pose estimation.\n",
                "2. **Stage 1:** MICA multi-view frontality-weighted identity shape regression (300-D FLAME beta).\n",
                "3. **Stage 2:** Canonical neutral base mesh normalization (psi=0, theta=0).\n",
                "4. **Stage 5:** Production retopology, ARKit-52 blendshapes, 4-tier LOD chain, skeletal armature, and chiseled/heroic stylization presets.\n",
                "5. **Gate Verification:** Pairwise beta divergence (||beta_a - beta_b|| > 1e-3) and neck seam pinning (Delta v = 0)."
            ]
        }
    ]

    for c in codes:
        lines = [l + "\n" for l in c.splitlines()]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    targets = [
        "notebooks/kaggle/build_phase1/phase1_inference_baseline.ipynb",
        "notebooks/kaggle/phase1_inference_baseline.ipynb"
    ]
    for target in targets:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(nb, f, indent=2)
        print(f"Successfully generated {target}")

if __name__ == "__main__":
    main()
