# Kaggle Notebooks for Face Geometry Pipeline

This directory contains executable Jupyter Notebook templates specifically tailored for Kaggle GPU (T4×2) and CPU environments.

## Directory Structure
- `phase1_inference_baseline.ipynb`: End-to-end multi-view inference on T4 with identity threshold gate.
- `phase2_identity_finetune.ipynb`: MICA fine-tuning notebook on 2×T4 with DDP and vertex L1 loss.
- `phase2_5_geometry_preprocessing.ipynb`: FaceScape scan UV rasterization engine (runs on CPU - 0 GPU quota).
- `phase3_detail_gan_train.ipynb`: 50k-step Adversarial Detail synthesis on 2×T4 with DDP, InstanceNorm, and EMA.

## Kaggle Environment Rules
Before running any notebook, consult [DOCS/kaggle_environment.md](../../DOCS/kaggle_environment.md):
1. **GPU Quota:** You have ~30 GPU-hours/week = ~15 wall-clock hours of T4×2.
2. **Session Limit:** 12-hour session hard cap. Auto-checkpoint at 11.5 hours.
3. **NumPy 2.x Handling:** If downgrading to NumPy 1.x, execute the restart in Cell 1.
4. **nvdiffrast:** Install via `pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation`.
