"""
Kaggle Cloud Remote Execution & Live Monitoring Runner.

Automates:
1. Kernel metadata packaging for Phase 1, Phase 2, Phase 2.5, and Phase 3.
2. Direct cloud launch via `kaggle kernels push`.
3. Status polling and real-time streaming into view/execution_log.md.
4. Output downloading into outputs/.
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIEW_DIR = PROJECT_ROOT / "view"
EXECUTION_LOG = VIEW_DIR / "execution_log.md"
DEFAULT_USER = "nightshowdown"


KERNEL_CONFIGS = {
    "phase1": {
        "slug": "phase1-inference-baseline",
        "title": "Phase 1: Multi-View Inference Baseline",
        "notebook": "phase1_inference_baseline.ipynb",
        "enable_gpu": True,
        "datasets": [],
    },
    "phase2_5": {
        "slug": "phase-2-5-geometry-preprocessing-1024",
        "title": "Phase 2.5: Geometry Preprocessing 1024",
        "notebook": "phase2_5_geometry_preprocessing.ipynb",
        "enable_gpu": False,  # 0 GPU quota consumed
        "datasets": ["nightshowdown/flame-model"],
    },
    "phase2": {
        "slug": "phase2-identity-finetune",
        "title": "Phase 2: MICA Identity Regressor Fine-Tune",
        "notebook": "phase2_identity_finetune.ipynb",
        "enable_gpu": True,  # 2xT4
        "datasets": ["nightshowdown/flame-model", "nightshowdown/mica-pretrained"],
    },
    "phase3": {
        "slug": "phase3-detail-gan-train",
        "title": "Phase 3: 1024 Ultra-Detail GAN Training",
        "notebook": "phase3_detail_gan_train.ipynb",
        "enable_gpu": True,  # 2xT4
        "datasets": [],
    },
}


def log_event(message: str):
    """Appends timestamped event to view/execution_log.md and stdout."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"`[{timestamp}]` {message}\n"
    print(formatted.strip())

    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXECUTION_LOG, "a") as f:
        f.write(formatted)


def launch_kernel(phase_key: str, user: str = DEFAULT_USER) -> str:
    """Packages and pushes a notebook to Kaggle Cloud."""
    assert phase_key in KERNEL_CONFIGS, f"Unknown phase: {phase_key}. Available: {list(KERNEL_CONFIGS.keys())}"
    cfg = KERNEL_CONFIGS[phase_key]

    staging_dir = PROJECT_ROOT / "notebooks" / "kaggle" / f"build_{phase_key}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    src_nb = PROJECT_ROOT / "notebooks" / "kaggle" / cfg["notebook"]
    assert src_nb.exists(), f"Notebook not found: {src_nb}"

    dst_nb = staging_dir / cfg["notebook"]
    shutil.copy2(src_nb, dst_nb)

    kernel_id = f"{user}/{cfg['slug']}"
    metadata = {
        "id": kernel_id,
        "title": cfg["title"],
        "code_file": cfg["notebook"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if cfg["enable_gpu"] else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": cfg["datasets"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    meta_file = staging_dir / "kernel-metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    log_event(f"🚀 Packaging kernel `{kernel_id}` (GPU: {cfg['enable_gpu']}, Internet: True)")

    cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = "KGAT_2ef9ab9c57c5ca109e7af862a81c6b21"

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode == 0:
        log_event(f"✅ Successfully pushed `{kernel_id}` to Kaggle cloud!\n{res.stdout.strip()}")
    else:
        log_event(f"❌ Error pushing `{kernel_id}`:\n{res.stderr.strip()}")
        raise RuntimeError(f"Kaggle push failed: {res.stderr}")

    return kernel_id


def check_kernel_status(kernel_id: str) -> str:
    """Checks the cloud status of a running kernel."""
    cmd = ["kaggle", "kernels", "status", kernel_id]
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = "KGAT_2ef9ab9c57c5ca109e7af862a81c6b21"

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = res.stdout.strip()
    log_event(f"📊 Status `{kernel_id}`: {output}")
    return output


def pull_kernel_output(kernel_id: str, dest_dir: str = "outputs") -> Path:
    """Pulls generated output files from Kaggle cloud."""
    target_path = PROJECT_ROOT / dest_dir / kernel_id.split("/")[-1]
    target_path.mkdir(parents=True, exist_ok=True)

    log_event(f"📥 Pulling output for `{kernel_id}` to {target_path}...")
    cmd = ["kaggle", "kernels", "output", kernel_id, "-p", str(target_path)]
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = "KGAT_2ef9ab9c57c5ca109e7af862a81c6b21"

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode == 0:
        log_event(f"✅ Outputs downloaded to {target_path}")
    else:
        log_event(f"⚠️ Pull warning for `{kernel_id}`: {res.stderr.strip()}")

    return target_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kaggle Runner for Humanoid-Face-3D")
    parser.add_argument("action", choices=["launch", "status", "pull"], help="Action to execute")
    parser.add_argument("--phase", choices=list(KERNEL_CONFIGS.keys()), default="phase2_5")
    parser.add_argument("--kernel_id", type=str, default=None)
    parser.add_argument("--user", type=str, default=DEFAULT_USER)
    args = parser.parse_args()

    if args.action == "launch":
        kid = launch_kernel(args.phase, user=args.user)
    elif args.action == "status":
        target = args.kernel_id or f"{args.user}/{KERNEL_CONFIGS[args.phase]['slug']}"
        check_kernel_status(target)
    elif args.action == "pull":
        target = args.kernel_id or f"{args.user}/{KERNEL_CONFIGS[args.phase]['slug']}"
        pull_kernel_output(target)
