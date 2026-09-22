#!/usr/bin/env python3
"""
Compiles and generates notebooks/kaggle/build_phase3/phase3_detail_gan_train.ipynb
and kernel-metadata.json with syntax validation on every cell.
"""
import json
from pathlib import Path

def create_phase3_notebook():
    nb_dir = Path("notebooks/kaggle/build_phase3")
    nb_dir.mkdir(parents=True, exist_ok=True)

    cells = []

    # Cell 0: Header Markdown
    c0_md = """# Phase 3: High-Frequency Detail GAN Training (Production Architecture)

**Objective:** Train an adversarial synthesis network to generate high-frequency UV displacement maps containing subject-specific micro-wrinkles and pore texture, conditioned on coarse neutral geometry and multi-view features.

### Core Architectural Features:
1. **U-Net Generator:** InstanceNorm2d (affine=True), multi-view cross-attention bottleneck with LayerNorm & residual connection, AdaIN style modulation from identity/expression codes.
2. **PatchGAN Discriminator:** Spectral Normalization with 70×70 receptive field patches.
3. **Lazy R1 Gradient Penalty:** Computed every 16 steps strictly in **FP32** accumulating into discriminator gradients.
4. **Masked Reconstruction Loss:** L1 loss restricted strictly to valid facial UV pixels (`mask == 1.0`), annealed linearly.
5. **Hardware & Distributed Strategy:** Dynamic GPU topology detection (`torch.cuda.device_count()`), PyTorch DDP (`torchrun`), PyTorch AMP mixed-precision float16.
6. **Quota Guard & Checkpoints:** Sliding window of 2 step checkpoints, `checkpoint_latest.pt` (full optimizer/scaler state), and continuous EMA weights in `ema_generator.pt`.
"""
    cells.append({"cell_type": "markdown", "metadata": {}, "source": c0_md})

    # Cell 1: Environment & Repository Setup
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

    # Cell 2: Install Runtime Dependencies & GPU Topology Detection
    c2_code = """# Cell 2: Install Runtime Dependencies & Detect GPU Topology
import subprocess
subprocess.run(["pip", "install", "trimesh", "opencv-python", "pyyaml", "scipy", "Pillow", "matplotlib", "--quiet"], check=True)

import torch
print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "Critical Error: CUDA GPU accelerator required for GAN training!"
n_gpus = torch.cuda.device_count()
print(f"Detected {n_gpus} GPU(s) for training:")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  [GPU {i}] {torch.cuda.get_device_name(i)} | Memory: {props.total_memory / 1e9:.2f} GB")
"""
    compile(c2_code, "<cell_2>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c2_code})

    # Cell 3: Dataset Discovery & Real Photogrammetry Scan Ingestion
    c3_code = """# Cell 3: Real Photogrammetry Scan Dataset Discovery & Ingestion
import json
import shutil
import subprocess
from pathlib import Path

print("--- Locating / Preparing Real 3D Scan Displacement Dataset ---")
DATA_DIR = None

# 1. Check for real photogrammetry scan displacement dataset in /kaggle/input
for cand in list(Path("/kaggle/input").glob("**/real_scan_displacement_dataset_1024")) + list(Path("/kaggle/input").glob("**/normalization_stats.json")):
    p = cand if cand.is_dir() else cand.parent
    if list(p.glob("*_disp.png")):
        DATA_DIR = p
        print(f"Found mounted real scan displacement dataset at: {DATA_DIR}")
        break

# 2. Check local repository data
if DATA_DIR is None and Path("data/real_scan_displacement_dataset_1024").exists():
    if list(Path("data/real_scan_displacement_dataset_1024").glob("*_disp.png")):
        DATA_DIR = Path("data/real_scan_displacement_dataset_1024")
        print(f"Found local real scan displacement dataset at: {DATA_DIR}")

# 3. If not mounted, automatically fetch Meta Multiface neutral scans and ray-cast
if DATA_DIR is None:
    print("Preprocessed real scan dataset not found. Downloading Meta Multiface real 3D scans...")
    from scripts.download_community_data import download_multiface_scans
    download_multiface_scans(identities=["6795937", "5372021", "8870559", "7889059"])

    DATA_DIR = Path("/tmp/real_scan_displacement_dataset_1024")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    scan_cmd = [
        "python", "scripts/build_real_scan_displacement_dataset.py",
        "--scans_dir", "data/external/3d_scans/multiface/extracted",
        "--out_dir", str(DATA_DIR),
        "--resolution", "512"
    ]
    print("Executing real scan ray-casting pipeline:", " ".join(scan_cmd))
    subprocess.run(scan_cmd, check=True)

disp_files = list(DATA_DIR.glob("*_disp.png"))
print(f"Verified dataset at {DATA_DIR} containing {len(disp_files)} real photogrammetry displacement maps.")
assert len(disp_files) >= 2, f"Expected >= 2 displacement samples, found {len(disp_files)}"

stats_file = DATA_DIR / "normalization_stats.json"
if stats_file.exists():
    with open(stats_file) as f:
        stats = json.load(f)
    print(f"Loaded normalization stats: p99 = {stats.get('p99_mm', 'N/A')} mm, resolution = {stats.get('resolution', 'N/A')}")
"""
    compile(c3_code, "<cell_3>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c3_code})

    # Cell 4: Multi-GPU / DDP Training Launch
    c4_code = """# Cell 4: Launch Detail GAN Training with Quota Guards & Resumability
import os
import subprocess
import torch
from pathlib import Path

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
checkpoint_dir = Path("checkpoints/stage3_detail")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_cmd = [
    "torchrun", f"--nproc_per_node={n_gpus}",
    "src/stage3_detail/trainer.py",
    "--data_dir", str(DATA_DIR),
    "--checkpoint_dir", str(checkpoint_dir),
    "--resolution", "1024",
    "--checkpoint_every", "1000",
    "--total_steps", "40000",
    "--batch_size", "4"
]

# Check for existing checkpoint to resume from (pilot run or prior step)
resume_candidates = (
    list(Path("/kaggle/input").glob("**/checkpoint_latest.pt")) +
    list(Path("checkpoints/stage3_detail").glob("checkpoint_latest.pt"))
)
if resume_candidates:
    print(f"Found checkpoint to resume from: {resume_candidates[0]}")
    train_cmd.extend(["--resume", str(resume_candidates[0])])

env = os.environ.copy()
env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

print("--- Launching Detail GAN Deep Studio Training (40k steps, 1024²) ---")
print("Command:", " ".join(train_cmd))
subprocess.run(train_cmd, env=env, check=True)
print("Training execution completed successfully.")
"""
    compile(c4_code, "<cell_4>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c4_code})

    # Cell 5: Validation Evaluation & Diversity Gate Verification
    c5_code = """# Cell 5: Validation Evaluation & Batch Output Diversity Gate
import torch
import cv2
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.data import UVDisplacementDataset

print("\\n--- Phase 3 Detail GAN Evaluation & Diversity Gate ---")
ckpt_ema = Path("checkpoints/stage3_detail/ema_generator.pt")
assert ckpt_ema.exists(), f"Critical Invariant Failure: Expected {ckpt_ema} to exist after training!"

device = "cuda" if torch.cuda.is_available() else "cpu"
gen = DetailGenerator().to(device)
try:
    state_dict = torch.load(ckpt_ema, map_location=device, weights_only=False)
except TypeError:
    state_dict = torch.load(ckpt_ema, map_location=device)
gen.load_state_dict(state_dict)
gen.eval()

val_ds = UVDisplacementDataset(str(DATA_DIR), is_train=False, target_resolution=1024)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=min(4, len(val_ds)), shuffle=False)

with torch.no_grad():
    batch = next(iter(val_loader))
    pos = batch["pos"].to(device)
    norm = batch["norm"].to(device)
    feats = batch["per_view_feats"].to(device)
    beta = batch["beta"].to(device)
    psi = batch["psi"].to(device)
    real_disp = batch["disp"].to(device)
    mask = batch["mask"].to(device)

    pred_disp = gen(pos, norm, feats, beta, psi)
    batch_std = float(pred_disp.std().item())
    pred_min = float(pred_disp.min().item())
    pred_max = float(pred_disp.max().item())

    print(f"Output Shape:             {pred_disp.shape}")
    print(f"Displacement Range:       [{pred_min:.4f}, {pred_max:.4f}]")
    print(f"Batch Standard Deviation: {batch_std:.4f} (Required Gate: > 0.010)")

    # Gate 1: Spatial Diversity (Anti-Mode Collapse)
    assert batch_std > 0.010, f"GATE FAILURE: Mode collapse detected (std={batch_std:.4f} <= 0.010)"
    print("GATE 1 PASSED: Detail generator produces diverse spatial micro-displacements with zero mode collapse.")

    # Copy normalization stats to checkpoint dir
    stats_src = DATA_DIR / "normalization_stats.json"
    if stats_src.exists():
        shutil.copy(stats_src, "checkpoints/stage3_detail/normalization_stats.json")

    # Generate visual validation preview
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    disp_pred_np = pred_disp[0, 0].cpu().numpy()
    disp_real_np = real_disp[0, 0].cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy()

    im0 = axes[0].imshow(disp_pred_np, cmap="inferno", vmin=-1.0, vmax=1.0)
    axes[0].set_title(f"Synthesized Displacement (std={batch_std:.4f})")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(disp_real_np, cmap="inferno", vmin=-1.0, vmax=1.0)
    axes[1].set_title("Target Ground Truth Disp")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask_np, cmap="gray")
    axes[2].set_title("UV Boundary Mask")
    axes[2].axis("off")

    plt.tight_layout()
    preview_path = Path("checkpoints/stage3_detail/validation_preview.png")
    plt.savefig(preview_path, dpi=150)
    plt.close()
    print(f"Saved validation comparison preview to: {preview_path}")
"""
    compile(c5_code, "<cell_5>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c5_code})

    # Cell 6: Package & Archive Assets
    c6_code = """# Cell 6: Package & Archive Output Assets
import tarfile
from pathlib import Path

print("\\n--- Packaging Trained Detail GAN Model & Checkpoints ---")
output_archive = Path("/kaggle/working/detail_gan_assets.tar.gz")
ckpt_dir = Path("checkpoints/stage3_detail")

with tarfile.open(output_archive, "w:gz") as tar:
    for f in ckpt_dir.glob("*"):
        if f.is_file():
            tar.add(f, arcname=f.name)
            print(f"  Archived: {f.name} ({f.stat().st_size / 1e6:.2f} MB)")

print(f"\\nAll assets successfully packaged to: {output_archive} ({output_archive.stat().st_size / 1e6:.2f} MB).")
"""
    compile(c6_code, "<cell_6>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c6_code})

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = nb_dir / "phase3_detail_gan_train.ipynb"
    with open(nb_path, "w") as f:
        json.dump(nb_data, f, indent=1)
    print(f"Created notebook at: {nb_path.resolve()}")

    root_nb_path = Path("notebooks/kaggle/phase3_detail_gan_train.ipynb")
    with open(root_nb_path, "w") as f:
        json.dump(nb_data, f, indent=1)
    print(f"Created notebook at: {root_nb_path.resolve()}")

    meta_data = {
        "id": "nightshowdown/phase-3-deep-detail-gan-train",
        "title": "Phase 3: Deep Detail GAN Train",
        "code_file": "phase3_detail_gan_train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "nightshowdown/flame-model"
        ],
        "competition_sources": [],
        "kernel_sources": [
            "nightshowdown/phase-2-5-geometry-preprocessing-1024",
            "nightshowdown/phase-3-detail-gan-train"
        ],
        "model_sources": []
    }
    meta_path = nb_dir / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    print(f"Created metadata at: {meta_path.resolve()}")

if __name__ == "__main__":
    create_phase3_notebook()
