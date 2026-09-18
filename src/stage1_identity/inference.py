import numpy as np
from pathlib import Path
from typing import List

try:
    import torch
except ImportError:
    torch = None

from src.stage0_preprocess.detector import FaceDetection
from src.utils.flame_model import FLAMEModel

class MICAIdentityEncoder:
    """
    Identity shape encoder based on MICA (ArcFace backbone + MLP regressor).
    Predicts 300-dimensional FLAME shape parameters (beta).
    Fuses multiple camera angles via detection-confidence softmax weighting.
    """
    def __init__(self, checkpoint_path: str, flame_model: FLAMEModel, device: str = 'cuda'):
        if torch is None:
            raise RuntimeError("PyTorch is required for MICAIdentityEncoder. Install torch>=2.1.0.")
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.flame = flame_model
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_mica()

    def _load_mica(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MICA weights not found at {self.checkpoint_path}. "
                "Attach pretrained MICA dataset input or run scripts/fetch_models.py."
            )

        # Add vendor paths dynamically
        import sys
        vendor_paths = [
            '/kaggle/working/pipeline/vendor/MICA',
            '/kaggle/working/face-geo-pipeline/vendor/MICA',
            './vendor/MICA'
        ]
        for p in vendor_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from micalib.models import MICA
            # MICA constructor expects FLAME model or config dict
            model = MICA(config=None)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MICA model from {self.checkpoint_path}: {e}. "
                "Verify vendor/MICA submodule is present and compatible."
            ) from e

    def encode_single(self, crop_112: np.ndarray) -> np.ndarray:
        """
        Encodes single 112x112 face crop into 300-dim beta vector.
        Requires RGB image scaled to [-1, 1].
        """
        if self.model is None:
            raise RuntimeError("MICA encoder model is not loaded.")

        # Correct color space (BGR -> RGB) and normalization ([-1, 1])
        img = crop_112[:, :, ::-1].copy()
        img = (img.astype(np.float32) - 127.5) / 127.5
        t_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            beta = self.model(t_img)
        return beta.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_multiview(self, detections: List[FaceDetection]) -> np.ndarray:
        """
        Confidence-weighted multi-view identity fusion.
        Applies softmax over InsightFace det_score to weight frontal high-confidence shots
        higher than noisy profile shots, preventing profile angle degradation.
        """
        valid_dets = [d for d in detections if d is not None]
        if not valid_dets:
            raise ValueError("No valid face detections provided for multi-view identity fusion.")

        if len(valid_dets) == 1:
            return self.encode_single(valid_dets[0].crop_112)

        betas = []
        confidences = []
        for det in valid_dets:
            b = self.encode_single(det.crop_112)
            betas.append(b)
            confidences.append(det.det_score)

        # Softmax over confidence scores
        scores = np.array(confidences, dtype=np.float32)
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / np.sum(exp_scores)

        fused_beta = np.zeros(300, dtype=np.float32)
        for w, b in zip(weights, betas):
            fused_beta += w * b

        return fused_beta
