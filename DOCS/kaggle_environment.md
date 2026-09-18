# Kaggle T4×2 Environment & Dependency Guide

This guide details the exact environment setup, quota management, and dependency compilation instructions for running the 3D Face Geometry Reconstruction Pipeline on Kaggle Notebooks.

---

## 1. Hardware & Resource Budget

| Metric | Specification | Operational Constraint |
|---|---|---|
| **Accelerator** | 2 × NVIDIA Tesla T4 (16GB GDDR6 each) | Total 32GB VRAM across 2 devices |
| **Weekly GPU Quota** | ~30 GPU-hours per week | **1 wall-clock hour on T4×2 consumes 2 GPU-hours** |
| **Effective Weekly Runtime** | **~15 wall-clock hours** maximum per week | Schedule sessions deliberately (1×11.5h + 1×3.5h) |
| **Max Session Duration** | 12 hours hard cap | Auto-checkpoint at **11.5 hours** before kernel termination |
| **CPU Session Cost** | **0 GPU quota** | All geometry preprocessing (Phase 2.5), ICP retopology (Phase 5), and Blender exports (Phase 7) must run on CPU |

---

## 2. Kernel Setup & The NumPy 2.x Restart Pattern

### Why Kernel Restart is Required for NumPy < 2
Kaggle's modern container images (Ubuntu 22.04, Python 3.10+) ship with **NumPy 2.x pre-imported**. While our codebase includes shims (`_install_numpy_aliases()` in `src/utils/flame_model.py`) to run on NumPy 2.x, some pre-compiled C/C++ wheels (such as legacy PyTorch3D or custom InsightFace binary bindings) can fail with binary ABI mismatches (`numpy.ndarray size changed, may indicate binary incompatibility`) if downgraded in an active session without a restart.

If you need a strict `numpy<2.0.0` environment, **Cell 1 of your Kaggle notebook must execute the downgrade and force an immediate kernel restart**:

```python
# ── CELL 1: Environment Harmonization & Safe Kernel Restart ─────────────────
import sys
import subprocess
import os

print("[Kaggle Setup] Checking NumPy version...")
import numpy as np

if np.__version__.startswith("2."):
    print(f"Detected NumPy {np.__version__}. Downgrading to NumPy 1.26.4 for legacy C-extension ABI compatibility...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--quiet"])
    print("[Kaggle Setup] Restarting kernel to load new C-extension ABI...")
    # Force clean kernel restart without losing local storage
    os.kill(os.getpid(), 9)
else:
    print(f"NumPy {np.__version__} is active and compatible. Proceeding.")
```

When this cell triggers `os.kill(os.getpid(), 9)`, Kaggle automatically re-spawns the kernel within 2 seconds with NumPy 1.26.4 loaded into memory. Cells 2 onwards will run seamlessly.

---

## 3. Dependency Installation in Kaggle Notebooks

Once the kernel has restarted, install repository requirements and non-PyPI GPU extensions:

```python
# ── CELL 2: Repository & GPU Dependency Installation ─────────────────────────
!pip install -r requirements.txt --quiet

# Install NVIDIA Differentiable Rasterizer (nvdiffrast) from source
# Requires 'ninja' (included in requirements.txt) and matching CUDA toolkit
!pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation --quiet

# Verify GPU & CUDA availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
```

---

## 4. Git Authentication: Deploy Keys vs Fine-Grained PAT

When cloning a private repository inside Kaggle Notebooks:

### Recommended: Fine-Grained Personal Access Token (PAT)
Fine-grained GitHub tokens avoid SSH key formatting pitfalls:
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
github_token = secrets.get_secret("GITHUB_TOKEN")

!git clone https://x-access-token:{github_token}@github.com/your-org/face-geo-pipeline.git
```

### If using SSH Deploy Keys (Note on Trailing Newlines)
Kaggle Secrets strip trailing newlines from multiline secrets. OpenSSH fails to parse private keys that lack a trailing newline (`error in libcrypto`). Ensure you re-append the newline:
```python
from kaggle_secrets import UserSecretsClient
import os
from pathlib import Path

secrets = UserSecretsClient()
deploy_key = secrets.get_secret("DEPLOY_KEY")

ssh_dir = Path.home() / ".ssh"
ssh_dir.mkdir(parents=True, exist_ok=True)
os.chmod(ssh_dir, 0o700)

key_path = ssh_dir / "id_ed25519"
with open(key_path, "w") as f:
    f.write(deploy_key.rstrip() + "\n")  # Re-add required trailing newline
os.chmod(key_path, 0o600)

!ssh-keyscan -t rsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
!git clone git@github.com:your-org/face-geo-pipeline.git
```

---

## 5. Checkpoint Preservation & Kaggle Models Registry

To prevent losing weights when Kaggle's 12-hour session expires:
1. Run training loops with an elapsed timer.
2. At **11.5 hours** (`elapsed_seconds >= 11.5 * 3600`), break the training loop cleanly.
3. Save EMA weights (`ema_generator.pt`) and normalization stats (`normalization_stats.json`).
4. Push the model to the Kaggle Models registry using `scripts/upload_to_kaggle_models.py`:
   ```bash
   python scripts/upload_to_kaggle_models.py \
       --checkpoint_dir checkpoints/stage3_detail \
       --handle username/face-geo-stage3-detail/pytorch/v1 \
       --version_notes "Completed 35,000 steps on T4x2 session 1" \
       --stage 3
   ```
5. In the next session, resume training from the latest Kaggle model checkpoint using `scripts/fetch_models.py`.
