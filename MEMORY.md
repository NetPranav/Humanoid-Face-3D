# Project Memory: Humanoid-Face-3D

*Last Updated: September 20, 2026*

---

## 1. Project Identity & Remotes
* **Repository:** `NetPranav/Humanoid-Face-3D`
* **GitHub URL:** https://github.com/NetPranav/Humanoid-Face-3D.git
* **Local Workspace:** `/Users/pranav/Project Folder/3d Model /`
* **Kaggle CLI:** `/Users/pranav/.local/bin/kaggle` (symlinked in PATH)
* **Configuration:** [configs/default.yaml](configs/default.yaml) (Stage 3 resolution: **1024×1024 Ultra-Resolution**)

---

## 2. Master Phase Status
* **Phase 0:** ✅ Foundation, P0 Bug Fixes & Honest Failure Verification (Completed)
* **Phase 1:** ✅ Multi-View Inference Baseline (Completed)
* **Phase 2:** 🔄 Identity Regressor Demographic Fine-Tuning (Code Ready & Queued for Kaggle)
* **Phase 2.5:** 🔄 Geometry Preprocessing & 1024² UV Displacement Engine (Code Ready for Kaggle CPU)
* **Phase 3:** 🔄 Adversarial High-Frequency Detail Synthesis (1024² Detail GAN Code Ready)
* **Phase 4:** ✅ Legal Protocol, Release Form, & Capture Ingestion (Completed)
* **Phase 5:** ✅ Production Retopology, ARKit-52 Rigging, 4 LODs, & UE5 Live Link FBX (Completed)
* **Test Suite:** 73 unit tests discovered and passing (`python3 -m unittest discover tests`).

---

## 3. Crash Recovery & Checkpoint Architecture
* **Checkpoint Frequency:** Checkpoint saved every 500 steps and at each epoch end.
* **Checkpoint Location:** `/kaggle/working/checkpoints/`
* **Sliding Window Pruning:** Retains only `checkpoint_latest.pt`, `ema_generator.pt`, and the last 2 step checkpoints to guarantee `/kaggle/working` never exceeds the 19.5GB limit.
* **Kaggle 11.5-Hour Safeguard:** Automatic emergency checkpoint triggered at 11.5 hours before Kaggle's 12-hour session kill.
* **Resuming Interrupted Training:**
  ```bash
  !torchrun --nproc_per_node=2 src/stage3_detail/trainer.py \
      --resume /kaggle/working/checkpoints/checkpoint_latest.pt \
      --batch_size 12 --resolution 1024
  ```

---

## 4. Immediate Next Steps
1. Configure Kaggle credentials in `~/.kaggle/kaggle.json`.
2. Push and monitor Phase 2.5 (FaceScape UV rasterization on Kaggle CPU).
3. Push and monitor Phase 2 (MICA fine-tuning on Kaggle 2×T4).
4. Push and monitor Phase 3 (1024² Detail GAN training on Kaggle 2×T4).
5. Download trained weights and run local inference with game-ready FBX generation.
