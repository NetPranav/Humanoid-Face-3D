#!/usr/bin/env python3
"""
Generates notebooks/kaggle/build_dataset_reextract/dataset_reextract_subdiv2.ipynb
and kernel-metadata.json for re-extracting the real scan displacement dataset
with subdivision level 2 (~80k vertices) and C² Gaussian smoothing (σ=8.0).
"""
import json
import shutil
from pathlib import Path


def create_dataset_reextract_notebook():
    nb_dir = Path("notebooks/kaggle/build_dataset_reextract")
    nb_dir.mkdir(parents=True, exist_ok=True)

    cells = []

    # Cell 0: Header Markdown
    c0_md = """# Dataset Re-Extraction: Subdivision Level 2 + C² Smoothing

**Objective:** Re-extract all real photogrammetry displacement maps from Meta Multiface scans using:
1. **Subdivision level 2** (~80k vertices) for facet-free dense ray sampling
2. **C² Gaussian smoothing (σ=8.0)** with normalized convolution on conditioning maps
3. **1024×1024 resolution** displacement, position, normal, and mask maps

This replaces the old subdivision-level-1 dataset with a clean, high-fidelity version
that eliminates all polygonal wireframe artifacts from the training data.
"""
    cells.append({"cell_type": "markdown", "metadata": {}, "source": c0_md})

    # Cell 1: Environment & Repository Setup
    c1_code = """# Cell 1: Environment & Repository Setup
import os
import sys
import subprocess
from pathlib import Path

print("--- Setting Up Humanoid-Face-3D Workspace ---")
# Clone to /tmp to avoid eating into Kaggle's 19.5GB /kaggle/working limit and 500-file cap
target_dir = Path("/tmp/Humanoid-Face-3D")
if not (target_dir / "src/pipeline.py").exists():
    subprocess.run(["git", "clone", "--recurse-submodules", "https://github.com/NetPranav/Humanoid-Face-3D.git", str(target_dir)], check=True)
os.chdir(str(target_dir))
subprocess.run(["git", "fetch", "origin", "main"], check=False)
subprocess.run(["git", "reset", "--hard", "origin/main"], check=False)
subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=False)

if str(target_dir) not in sys.path:
    sys.path.insert(0, str(target_dir))
if "." not in sys.path:
    sys.path.insert(0, ".")
print("Active working directory:", os.getcwd())
"""
    compile(c1_code, "<cell_1>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c1_code})

    # Cell 2: Install Dependencies
    c2_code = """# Cell 2: Install Runtime Dependencies
import subprocess
subprocess.run(["pip", "install", "trimesh", "opencv-python", "pyyaml", "scipy", "Pillow", "matplotlib", "tqdm", "--quiet"], check=True)

import torch
print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [GPU {i}] {torch.cuda.get_device_name(i)} | Memory: {props.total_memory / 1e9:.2f} GB")
    device = "cuda"
else:
    device = "cpu"
    print("No GPU detected — running on CPU (slower but functional).")
"""
    compile(c2_code, "<cell_2>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c2_code})

    # Cell 3: Locate & Mount FLAME Model & Head Template
    c3_flame_code = """# Cell 3: Locate & Mount FLAME Model & Head Template
import shutil
from pathlib import Path

print("--- Locating & Mounting FLAME Assets ---")
flame_dest = Path("data/flame_model")
flame_dest.mkdir(parents=True, exist_ok=True)

# 1. generic_model.pkl
flame_candidates = list(Path("/kaggle/input").glob("**/generic_model.pkl")) + list(Path(".").glob("**/generic_model.pkl"))
if not flame_candidates:
    raise FileNotFoundError("generic_model.pkl not found! Please attach dataset nightshowdown/flame-model.")
flame_pkl = flame_candidates[0]
dest_pkl = flame_dest / "generic_model.pkl"
if not dest_pkl.exists() or dest_pkl.stat().st_size < 1000:
    shutil.copy2(flame_pkl, dest_pkl)
print(f"  Mounted FLAME generic_model.pkl from {flame_pkl} -> {dest_pkl} ({dest_pkl.stat().st_size / 1e6:.1f} MB)")

# 2. head_template.obj
head_candidates = list(Path("/kaggle/input").glob("**/head_template.obj")) + list(Path(".").glob("**/head_template.obj"))
if not head_candidates:
    raise FileNotFoundError("head_template.obj not found! Please attach dataset nightshowdown/flame-model.")
head_obj = head_candidates[0]
dest_obj = flame_dest / "head_template.obj"
if not dest_obj.exists() or dest_obj.stat().st_size < 1000:
    shutil.copy2(head_obj, dest_obj)
print(f"  Mounted FLAME head_template.obj from {head_obj} -> {dest_obj} ({dest_obj.stat().st_size / 1e3:.1f} KB)")
"""
    compile(c3_flame_code, "<cell_3_flame>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c3_flame_code})

    # Cell 4: Download & Discover Real 3D Scans
    c4_scan_code = """# Cell 4: Download & Discover Real Photogrammetry 3D Scans
import json
import subprocess
from pathlib import Path

print("--- Locating / Downloading Meta Multiface Real 3D Scans ---")

# Check if scans are already available (from mounted Kaggle datasets or prior runs)
scans_dir = Path("data/external/3d_scans/multiface/extracted")
scan_files = []

if scans_dir.exists():
    scan_files = sorted(list(scans_dir.glob("**/*.obj")) + list(scans_dir.glob("**/*.ply")))
    scan_files = [p for p in scan_files if "template" not in p.name.lower()]

# Also check /kaggle/input for mounted scan datasets
if len(scan_files) == 0:
    for cand in Path("/kaggle/input").glob("**/extracted"):
        found = sorted(list(cand.glob("**/*.obj")) + list(cand.glob("**/*.ply")))
        found = [p for p in found if "template" not in p.name.lower()]
        if found:
            scans_dir = cand
            scan_files = found
            print(f"Found mounted scans at: {scans_dir}")
            break

# Download if not found
if len(scan_files) == 0:
    print("No pre-mounted scans found. Downloading Meta Multiface neutral scans...")
    from scripts.download_community_data import download_multiface_scans
    download_multiface_scans(identities=["2183941", "6795937", "5372021", "8870559", "7889059"])
    scan_files = sorted(list(scans_dir.glob("**/*.obj")) + list(scans_dir.glob("**/*.ply")))
    scan_files = [p for p in scan_files if "template" not in p.name.lower()]

print(f"\\n📦 Found {len(scan_files)} real 3D scan meshes ready for processing.")
assert len(scan_files) >= 1, "No scan meshes found! Check data/external/3d_scans/ or /kaggle/input mounts."
"""
    compile(c4_scan_code, "<cell_4_scan>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c4_scan_code})

    # Cell 5: Run Re-Extraction with Subdivision Level 2
    c5_extract_code = """# Cell 5: Run Dataset Re-Extraction (Subdivision Level 2, σ=8.0, 1024×1024)
import subprocess
import os
import sys
import time
from pathlib import Path

# Use /tmp for scratch to avoid Kaggle's 19.5GB /kaggle/working quota and inode limit
OUT_DIR = Path("/tmp/real_scan_displacement_dataset_1024")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  DATASET RE-EXTRACTION: Subdivision Level 2 + C² Smoothing")
print("=" * 72)
print(f"  Scans directory:  {scans_dir}")
print(f"  Output directory: {OUT_DIR}")
print(f"  Resolution:       1024×1024")
print(f"  Subdivision:      Level 2 (~80k vertices)")
print(f"  Smoothing σ:      8.0 (normalized convolution)")
print("=" * 72)

start = time.time()

env = os.environ.copy()
env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

extract_cmd = [
    sys.executable, "scripts/build_real_scan_displacement_dataset.py",
    "--scans_dir", str(scans_dir),
    "--out_dir", str(OUT_DIR),
    "--resolution", "1024",
    "--device", device,
    "--verify_render",
]

print("\\nCommand:", " ".join(extract_cmd))
result = subprocess.run(extract_cmd, env=env)

elapsed = time.time() - start
print(f"\\n⏱️  Total extraction time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

if result.returncode != 0:
    raise RuntimeError(f"❌ Extraction failed with exit code {result.returncode}! See log above.")
else:
    print("✅ Extraction completed successfully!")
"""
    compile(c5_extract_code, "<cell_5_extract>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c5_extract_code})

    # Cell 6: Validate Extracted Dataset Quality
    c6_val_code = """# Cell 6: Validate Extracted Dataset Quality
import json
import numpy as np
import cv2
from pathlib import Path

OUT_DIR = Path("/tmp/real_scan_displacement_dataset_1024")

# Load stats
stats_file = OUT_DIR / "normalization_stats.json"
assert stats_file.exists(), f"Missing normalization_stats.json at {OUT_DIR}!"

with open(stats_file) as f:
    stats = json.load(f)

print("=" * 60)
print("  DATASET VALIDATION REPORT")
print("=" * 60)
print(f"  Total samples:     {stats.get('num_samples', 'N/A')}")
print(f"  Resolution:        {stats.get('resolution', 'N/A')}×{stats.get('resolution', 'N/A')}")
print(f"  Empirical p99:     {stats.get('p99_mm', 'N/A'):.4f} mm")
print(f"  Dataset type:      {stats.get('dataset_type', 'N/A')}")

# Validate sample files
disp_files = sorted(OUT_DIR.glob("*_disp.png"))
pos_files = sorted(OUT_DIR.glob("*_pos.png"))
norm_files = sorted(OUT_DIR.glob("*_norm.png"))
mask_files = sorted(OUT_DIR.glob("*_mask.png"))
maps_files = sorted(OUT_DIR.glob("*_maps.npz"))

print(f"  Displacement maps: {len(disp_files)}")
print(f"  Position maps:     {len(pos_files)}")
print(f"  Normal maps:       {len(norm_files)}")
print(f"  Mask maps:         {len(mask_files)}")
print(f"  Metadata NPZ:     {len(maps_files)}")

assert len(disp_files) > 0, "Zero displacement maps extracted!"
assert len(disp_files) == len(pos_files) == len(norm_files), "Mismatched file counts!"

# Spot-check first sample for correctness
sample_disp = cv2.imread(str(disp_files[0]), cv2.IMREAD_UNCHANGED)
sample_pos = cv2.imread(str(pos_files[0]), cv2.IMREAD_COLOR)
sample_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)

print(f"\\n  Sample check: {disp_files[0].name}")
print(f"    Displacement dtype: {sample_disp.dtype}, shape: {sample_disp.shape}")
print(f"    Position dtype: {sample_pos.dtype}, shape: {sample_pos.shape}")
print(f"    Mask coverage: {np.sum(sample_mask > 0) / sample_mask.size * 100:.1f}% facial pixels")

# Verify 16-bit depth
assert sample_disp.dtype == np.uint16, f"Expected uint16 displacement, got {sample_disp.dtype}"
assert sample_disp.shape == (1024, 1024), f"Expected 1024×1024, got {sample_disp.shape}"

# Verify no wireframe artifacts via Laplacian check on position maps
pos_f = sample_pos.astype(np.float32) / 255.0
laplacian_mag = np.zeros(pos_f.shape[:2], dtype=np.float32)
for ch in range(3):
    lap = cv2.Laplacian(pos_f[:, :, ch], cv2.CV_32F, ksize=3)
    laplacian_mag += lap ** 2
laplacian_mag = np.sqrt(laplacian_mag)

kernel = np.ones((5, 5), np.uint8)
interior = cv2.erode(sample_mask, kernel, iterations=2)
if np.sum(interior > 0) > 0:
    max_lap = float(laplacian_mag[interior > 0].max())
    print(f"    Max Laplacian (interior): {max_lap:.6f} {'✅ CLEAN' if max_lap < 0.1 else '⚠️ POSSIBLE ARTIFACTS'}")

print("=" * 60)
print("✅ Dataset validation complete!")
"""
    compile(c6_val_code, "<cell_6_val>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c6_val_code})

    # Cell 7: Preview Grid
    c7_vis_code = """# Cell 7: Generate Visual Preview Grid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/tmp/real_scan_displacement_dataset_1024")
disp_files = sorted(OUT_DIR.glob("*_disp.png"))

n_preview = min(8, len(disp_files))
fig, axes = plt.subplots(2, n_preview, figsize=(4 * n_preview, 8))
if n_preview == 1:
    axes = axes.reshape(2, 1)

for i in range(n_preview):
    stem = disp_files[i].stem.replace("_disp", "")

    # Displacement
    disp = cv2.imread(str(disp_files[i]), cv2.IMREAD_UNCHANGED)
    disp_vis = ((disp.astype(np.float32) / 65535.0) * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA)
    axes[0, i].imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Disp: {stem[:25]}", fontsize=7)
    axes[0, i].axis("off")

    # Normal
    norm_path = OUT_DIR / f"{stem}_norm.png"
    if norm_path.exists():
        norm = cv2.imread(str(norm_path), cv2.IMREAD_COLOR)
        axes[1, i].imshow(cv2.cvtColor(norm, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Norm: {stem[:25]}", fontsize=7)
    axes[1, i].axis("off")

plt.suptitle(f"Real Scan Dataset Preview (Subdiv Level 2, σ=8.0) — {len(disp_files)} samples", fontsize=12)
plt.tight_layout()
preview_save_path = Path("/kaggle/working/dataset_preview_grid.png")
preview_save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(preview_save_path), dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved preview grid to {preview_save_path}")
"""
    compile(c7_vis_code, "<cell_7_vis>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c7_vis_code})

    # Cell 8: Package Dataset into Zip Archive for Kaggle Output Download
    c8_pkg_code = """# Cell 8: Package Dataset for Kaggle Output Download
import shutil
import subprocess
from pathlib import Path

OUT_DIR = Path("/tmp/real_scan_displacement_dataset_1024")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)

# 1. Package complete dataset into a zip archive (strictly respects Kaggle's 500-file cap & enables fast 1-file download)
zip_path = KAGGLE_WORKING / "real_scan_displacement_dataset_1024.zip"
print(f"📦 Packaging complete dataset into {zip_path}...")
subprocess.run(["zip", "-q", "-r", str(zip_path), "."], cwd=str(OUT_DIR), check=True)
print(f"✅ Created archive: {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")

# 2. Copy lightweight metadata files directly to /kaggle/working/ for instant inspection
for fname in ["normalization_stats.json", "manifest.json"]:
    src = OUT_DIR / fname
    if src.exists():
        shutil.copy2(src, KAGGLE_WORKING / fname)
        print(f"  Copied {fname} to /kaggle/working/")

for render in OUT_DIR.glob("verify_320k_*.png"):
    shutil.copy2(render, KAGGLE_WORKING / render.name)
    print(f"  Copied {render.name} to /kaggle/working/")

# 3. Audit final /kaggle/working inode & disk count
staged_files = list(KAGGLE_WORKING.glob("*"))
print(f"\\n📋 Final /kaggle/working contains {len(staged_files)} files (cap is 500 files):")
for f in sorted(staged_files):
    if f.is_file():
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.2f} MB)")

print("\\n🎉 Dataset re-extraction complete! Outputs ready for download.")
"""
    compile(c8_pkg_code, "<cell_8_pkg>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c8_pkg_code})

    # Build notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "cells": cells
    }

    nb_path = nb_dir / "dataset_reextract_subdiv2.ipynb"
    with open(nb_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"✅ Notebook written to: {nb_path}")

    # Also sync to notebooks/kaggle/dataset_reextract_subdiv2.ipynb
    root_nb = Path("notebooks/kaggle/dataset_reextract_subdiv2.ipynb")
    shutil.copy2(nb_path, root_nb)
    print(f"✅ Synchronized notebook to: {root_nb}")

    # Kernel metadata for Kaggle push
    metadata = {
        "id": "nightshowdown/dataset-re-extract-subdiv2-c2-smoothing",
        "title": "Dataset Re-Extract: Subdiv2 + C2 Smoothing",
        "code_file": "dataset_reextract_subdiv2.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["nightshowdown/flame-model"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    meta_path = nb_dir / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Kernel metadata written to: {meta_path}")

    return nb_path, meta_path


if __name__ == "__main__":
    create_dataset_reextract_notebook()
