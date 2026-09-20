"""
Standardized Kaggle Environment & Repository Setup Utility.

Provides standard functions to:
1. Clone or pull latest code from GitHub into active Kaggle session.
2. Verify NumPy 1.x / 2.x compatibility with automated safe kernel restart.
3. Install repository requirements and JIT-compile nvdiffrast.
4. Verify CUDA GPU accelerator and device properties.

Usage inside Kaggle Notebook Cell 1:
```python
import sys
from pathlib import Path
if not Path('setup.py').exists() and Path('notebooks/kaggle/setup.py').exists():
    sys.path.insert(0, 'notebooks/kaggle')
import setup
setup.setup_environment(repo_url="https://github.com/yourusername/yourrepo.git")
```
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_and_fix_numpy(target_version: str = "1.26.4"):
    """
    Checks active NumPy version. If 2.x, downgrades and restarts kernel safely.
    """
    try:
        import numpy as np
        current_version = np.__version__
        print(f"[Setup] Active NumPy version: {current_version}")
        if current_version.startswith("2."):
            print(f"[Setup] Detected NumPy 2.x. Downgrading to numpy=={target_version} for C-extension ABI stability...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"numpy=={target_version}", "--quiet"])
            print("[Setup] Restarting kernel immediately (Cell will auto-resume on fresh process)...")
            os.kill(os.getpid(), 9)
        else:
            print("[Setup] NumPy version is compatible. Proceeding.")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"numpy=={target_version}", "--quiet"])


def install_dependencies(repo_dir: Path):
    """
    Installs requirements.txt and compiles nvdiffrast if not present.
    """
    req_file = repo_dir / "requirements.txt"
    if req_file.exists():
        print(f"[Setup] Installing dependencies from {req_file}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"])

    # Check nvdiffrast
    try:
        import nvdiffrast
        print("[Setup] nvdiffrast is already installed.")
    except ImportError:
        print("[Setup] Installing and JIT-compiling nvdiffrast from NVlabs...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/NVlabs/nvdiffrast.git",
            "--no-build-isolation",
            "--quiet"
        ])


def check_gpu():
    """
    Verifies PyTorch CUDA accelerator and prints GPU properties.
    """
    try:
        import torch
        print(f"[Setup] PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"[Setup] Detected {n_gpus} GPU device(s):")
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024 ** 3)
                print(f"  [GPU {i}] {props.name} | VRAM: {total_mem_gb:.2f} GB")
        else:
            print("[Setup] Warning: No CUDA accelerator detected. Training scripts require GPU runtime.")
    except ImportError:
        print("[Setup] Warning: PyTorch is not yet imported.")


def sync_repository(
    repo_url: str = "https://github.com/NetPranav/Humanoid-Face-3D.git",
    repo_name: str = "Humanoid-Face-3D"
) -> Path:
    """
    Clones or pulls latest code into /kaggle/working/.
    """
    working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    repo_path = working_dir / repo_name

    if repo_path.exists() and (repo_path / ".git").exists():
        print(f"[Setup] Repository exists at {repo_path}. Pulling latest changes from git...")
        try:
            subprocess.check_call(["git", "-C", str(repo_path), "pull", "--quiet"])
            print("[Setup] Successfully updated repository.")
        except Exception as e:
            print(f"[Setup] Git pull warning: {e}")
    elif repo_url:
        print(f"[Setup] Cloning repository from {repo_url} to {repo_path}...")
        subprocess.check_call(["git", "clone", repo_url, str(repo_path), "--quiet"])
        print("[Setup] Clone complete.")
    else:
        repo_path = Path.cwd()

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    return repo_path


def setup_environment(
    repo_url: str = "https://github.com/NetPranav/Humanoid-Face-3D.git",
    repo_name: str = "Humanoid-Face-3D"
) -> Path:
    """
    Unified entrypoint for Kaggle notebook Cell 1.
    """
    print("=" * 65)
    print(" 3D Face Geometry Pipeline — Kaggle Environment Setup")
    print("=" * 65)

    check_and_fix_numpy()
    repo_path = sync_repository(repo_url, repo_name)
    install_dependencies(repo_path)
    check_gpu()

    print("=" * 65)
    print(f" Ready to run pipeline from root: {repo_path.resolve()}")
    print("=" * 65)
    return repo_path


if __name__ == '__main__':
    setup_environment()
