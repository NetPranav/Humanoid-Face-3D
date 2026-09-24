"""
ArcFace Feature Extraction & Caching Engine.

Extracts normalized 512-D ArcFace feature embeddings and 112x112 canonical aligned crops
from training portrait images, pairing them with ground-truth FLAME 300-D shape parameters (beta)
for MICA identity regressor fine-tuning (Stage 1 / Job 02).
"""
import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage0_preprocess.detector import FaceDetector, FaceDetection


def find_ground_truth_beta(beta_dir: Optional[Path], subj_id: str, stem: str) -> np.ndarray:
    """
    Attempts to locate corresponding 300-D FLAME shape coefficients (beta)
    matching the subject or sample name.
    """
    if beta_dir is None or not beta_dir.exists():
        return np.zeros(300, dtype=np.float32)

    candidates = [
        beta_dir / f"{stem}.npz",
        beta_dir / f"{stem}.npy",
        beta_dir / f"{subj_id}.npz",
        beta_dir / f"{subj_id}.npy",
        beta_dir / f"{subj_id}_beta.npy",
        beta_dir / f"{subj_id}_neutral.npz",
    ]

    for c in candidates:
        if c.exists():
            try:
                data = np.load(c, allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    for k in ['beta', 'shape', 'shape_params']:
                        if k in data:
                            b = data[k].astype(np.float32).flatten()
                            out = np.zeros(300, dtype=np.float32)
                            out[:min(300, len(b))] = b[:300]
                            return out
                else:
                    arr = data.item() if data.ndim == 0 else data
                    if isinstance(arr, dict):
                        for k in ['beta', 'shape', 'shape_params']:
                            if k in arr:
                                b = np.array(arr[k], dtype=np.float32).flatten()
                                out = np.zeros(300, dtype=np.float32)
                                out[:min(300, len(b))] = b[:300]
                                return out
                    elif isinstance(arr, np.ndarray):
                        b = arr.astype(np.float32).flatten()
                        out = np.zeros(300, dtype=np.float32)
                        out[:min(300, len(b))] = b[:300]
                        return out
            except Exception:
                continue

    return np.zeros(300, dtype=np.float32)


def extract_features_for_dataset(
    image_dir: str,
    output_dir: str,
    beta_dir: Optional[str] = None,
    allow_degraded: bool = False,
    device: str = 'cuda'
) -> int:
    """
    Scans image_dir, runs InsightFace ArcFace extraction, pairs with GT beta,
    and writes out {subj_id}_{view_id}.npz archives.
    """
    in_path = Path(image_dir)
    out_path = Path(output_dir)
    b_path = Path(beta_dir) if beta_dir else None
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {in_path}")

    ctx_id = 0 if device == 'cuda' else -1
    detector = FaceDetector(ctx_id=ctx_id, allow_degraded=allow_degraded)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([p for p in in_path.glob("**/*") if p.suffix.lower() in valid_exts])

    if not image_files:
        print(f"[Feature Extractor] No images found in {in_path}")
        return 0

    print(f"[Feature Extractor] Found {len(image_files)} images to process. Output directory: {out_path}")

    processed_count = 0
    for img_p in image_files:
        # Determine subject ID and view identifier
        rel = img_p.relative_to(in_path)
        if len(rel.parts) > 1:
            subj_id = rel.parts[0]
            view_id = "_".join(rel.parts[1:]).rsplit('.', 1)[0]
        else:
            stem_parts = img_p.stem.split('_')
            subj_id = stem_parts[0]
            view_id = "_".join(stem_parts[1:]) if len(stem_parts) > 1 else "view0"

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            print(f"  ⚠️ Warning: Could not read image: {img_p}")
            continue

        detections = detector.detect(img_bgr)
        if not detections:
            print(f"  ⚠️ Warning: No face detected in: {img_p}")
            continue

        # Select detection with highest score
        best_det = max(detections, key=lambda d: d.det_score)

        if best_det.embedding is None:
            # Fallback zero embedding if degraded
            emb = np.zeros(512, dtype=np.float32)
        else:
            emb = best_det.embedding.astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)

        beta_gt = find_ground_truth_beta(b_path, subj_id, img_p.stem)

        out_name = f"{subj_id}_{view_id}.npz"
        out_file = out_path / out_name

        np.savez_compressed(
            str(out_file),
            embedding=emb,
            crop_112=best_det.crop_112,
            beta=beta_gt,
            det_score=float(best_det.det_score),
            yaw_deg=float(best_det.yaw_deg),
            subj_id=subj_id
        )
        processed_count += 1

    print(f"[Feature Extractor Complete] Successfully extracted and saved {processed_count} paired samples.")
    return processed_count


def main():
    parser = argparse.ArgumentParser(description="Pre-extract ArcFace features for MICA fine-tuning")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to input images directory")
    parser.add_argument('--output_dir', type=str, default='data/mica_preprocessed', help="Path to save preprocessed .npz files")
    parser.add_argument('--beta_dir', type=str, default=None, help="Optional directory with ground-truth FLAME beta .npy/.npz files")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device for InsightFace inference")
    parser.add_argument('--allow_degraded', action='store_true', help="Allow degraded mode if InsightFace is unavailable")
    args = parser.parse_args()

    extract_features_for_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        beta_dir=args.beta_dir,
        allow_degraded=args.allow_degraded,
        device=args.device
    )


if __name__ == '__main__':
    main()
