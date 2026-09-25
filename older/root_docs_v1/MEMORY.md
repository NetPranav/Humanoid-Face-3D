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
* **Phase 2:** ⏹ Identity Fine-Tuning (Code Ready, Blocked on Ground-Truth FLAME $\beta$ Supervision Dataset)
* **Phase 2.5:** ✅ Geometry Preprocessing & 1024² UV Displacement Engine (Job 01 Completed & Verified on Kaggle)
* **Phase 3:** 🔄 Adversarial High-Frequency Detail Synthesis (Detail GAN Code Ready; Consumes Phase 2.5 UV Maps)
* **Phase 4:** 🔄 Commercial Licensing & Own-Capture (Code/Protocol Built; Physical Captures & External Licenses Pending)
* **Phase 5:** ✅ Production Retopology, ARKit-52 Rigging, 4 LODs, & UE5 Live Link FBX (Completed)
* **Test Suite:** 76 unit tests discovered and passing (`python3 -m unittest discover tests`).
* **Pre-Flight Blockers Resolved (Commit `254899b`):**
  - Added `scripts/extract_arcface_features.py` for pre-extracting 512-D ArcFace embeddings and 112×112 crops for Stage 1 fine-tuning.
  - Added safe `per_view_feats` dictionary lookup and `(1, 512)` zero-fallback in `src/stage3_detail/data.py` to prevent `KeyError`.
  - Set default `id_lambda = 0.0` with CLI argument in `src/stage3_detail/trainer.py` to prevent passing 2D scalar displacement maps directly to ArcFace.
  - Added multi-path resolution in `src/pipeline.py` checking both `pretrained.tar` and `mica.tar`.

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

## 4. Current Execution Plan
1. **Automated Kaggle Datasets (CLI):**
   - ✅ `nightshowdown/flame-model`: Created and ready on Kaggle (`generic_model.pkl` + `head_template.obj`).
   - 🔄 `nightshowdown/mica-pretrained`: Uploaded (`mica.tar` + `pretrained.tar`), indexing on Kaggle backend.
2. **Job 01 (Phase 2.5 Geometry Preprocessing 1024²):**
   - ✅ **COMPLETE** on Kaggle CPU (`nightshowdown/phase-2-5-geometry-preprocessing-1024`).
   - Downloaded and verified outputs in `outputs/uv_displacement_dataset_1024/`:
     * `subject_001_neutral_disp.png` (1024×1024 16-bit uint PNG, $p_{99} = 9.045\text{ mm}$)
     * `subject_001_neutral_norm.png` (1024×1024 normal map)
     * `subject_001_neutral_mask.png` (1024×1024 facial validity mask)
     * `subject_001_neutral_pos.png` (1024×1024 position map)
     * `normalization_stats.json` & `sample_preview.png`
3. **Job 02 (Phase 2 MICA Fine-Tuning):**
   - Run feature extraction via `scripts/extract_arcface_features.py` to produce paired `.npz` files.
   - Run `notebooks/kaggle/phase2_identity_finetune.ipynb` on Kaggle 2×T4 once `mica-pretrained` is ready.
4. **Job 03 (Phase 3 1024² Detail GAN Training):**
   - Run `notebooks/kaggle/phase3_detail_gan_train.ipynb` on Kaggle 2×T4.

