"""
Stage 1: MICA identity regression (β ∈ R^300).

Phase 0 fix (DOCS/01 R1): MICA's mapping network was trained on features from MICA's own
fine-tuned ArcFace (checkpoint['arcface'], iresnet100). v1 fed it InsightFace buffalo_l
embeddings, a different network whose 512-D space is unrelated, so β carried almost no
identity. Features are now computed exactly as vendor/MICA/demo.py does:

    crop  = face_align.norm_crop(img, kps)                          # 112x112 BGR
    blob  = blobFromImages(crop, 1/127.5, mean 127.5, swapRB=True)  # RGB in [-1, 1]
    feat  = normalize(MICA_ArcFace(blob));  β = regressor(feat)

Both networks load with strict=True; a missing or mismatched checkpoint is an error.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is a hard requirement at runtime
    torch = None
    nn = None
    F = None

from src.stage0_preprocess.detector import FaceDetection

logger = logging.getLogger("Stage1Identity")

_MICA_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "MICA"

if torch is not None:
    class MappingNetwork(nn.Module):
        """MICA's MLP regressor: normalized 512-D ArcFace feature -> 300-D FLAME β."""

        def __init__(self, z_dim: int = 512, map_hidden_dim: int = 300, map_output_dim: int = 300, hidden: int = 3):
            super().__init__()
            self.skips = [int(hidden / 2)] if hidden > 5 else []
            self.network = nn.ModuleList(
                [nn.Linear(z_dim, map_hidden_dim)] +
                [nn.Linear(map_hidden_dim, map_hidden_dim) if i not in self.skips else
                 nn.Linear(map_hidden_dim + z_dim, map_hidden_dim) for i in range(hidden)]
            )
            self.output = nn.Linear(map_hidden_dim, map_output_dim)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = z
            for i, layer in enumerate(self.network):
                h = F.leaky_relu(layer(h), negative_slope=0.2)
                if i in self.skips:
                    h = torch.cat([z, h], 1)
            return self.output(h)
else:
    MappingNetwork = None


def _load_mica_arcface():
    """Imports MICA's ArcFace (iresnet100) class from the vendored MICA repository."""
    if not _MICA_VENDOR.exists():
        raise FileNotFoundError(
            f"vendor/MICA not found at {_MICA_VENDOR}. Run: git submodule update --init vendor/MICA"
        )
    if str(_MICA_VENDOR) not in sys.path:
        sys.path.insert(0, str(_MICA_VENDOR))
    from models.arcface import Arcface  # noqa: E402  (vendor import)
    return Arcface


def mica_arcface_blob(crop_112_bgr: np.ndarray) -> np.ndarray:
    """MICA's ArcFace preprocessing (vendor/MICA/datasets/creation/util.py:get_arcface_input)."""
    if crop_112_bgr is None or crop_112_bgr.shape[:2] != (112, 112):
        raise ValueError("MICA expects a 112x112 insightface norm_crop (FaceDetection.crop_112).")
    return cv2.dnn.blobFromImages(
        [np.ascontiguousarray(crop_112_bgr, dtype=np.uint8)], 1.0 / 127.5, (112, 112),
        (127.5, 127.5, 127.5), swapRB=True,
    )[0]


class MICAIdentityEncoder:
    """
    Identity shape encoder: MICA ArcFace + MICA regressor, both from the MICA checkpoint.

    Multi-view fusion averages the normalized MICA features, weighted by
    det_score · max(0, cos yaw)^yaw_weight_exponent through a temperature softmax,
    then runs the regressor once.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        fusion_temperature: float = 0.05,
        yaw_weight_exponent: int = 2,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for MICAIdentityEncoder. Install torch>=2.1.0.")
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.checkpoint_path = Path(checkpoint_path)
        self.fusion_temperature = fusion_temperature
        self.yaw_weight_exponent = yaw_weight_exponent
        self.arcface, self.regressor = self._load()

    def _load(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MICA weights not found at: {self.checkpoint_path}. Download with:\n"
                "  python3 -m gdown 1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt -O models_cache/mica/mica.tar"
            )
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "arcface" not in ckpt or "flameModel" not in ckpt:
            raise RuntimeError(
                f"{self.checkpoint_path} is not a MICA checkpoint (needs 'arcface' and 'flameModel' keys)."
            )

        Arcface = _load_mica_arcface()
        arcface = Arcface()
        arcface.load_state_dict(ckpt["arcface"], strict=True)
        arcface.to(self.device).eval()

        reg_state = {k[len("regressor."):]: v for k, v in ckpt["flameModel"].items() if k.startswith("regressor.")}
        regressor = MappingNetwork(z_dim=512, map_hidden_dim=300, map_output_dim=300, hidden=3)
        regressor.load_state_dict(reg_state, strict=True)
        regressor.to(self.device).eval()
        return arcface, regressor

    # -- features -------------------------------------------------------------

    def extract_embedding(self, det: Union[FaceDetection, np.ndarray]) -> np.ndarray:
        """Normalized MICA ArcFace feature (512,) from a FaceDetection or a 112x112 BGR crop."""
        crop = det if isinstance(det, np.ndarray) else det.crop_112
        blob = torch.from_numpy(mica_arcface_blob(crop))[None].to(self.device)
        with torch.no_grad():
            feat = F.normalize(self.arcface(blob).float(), dim=1)
        return feat[0].cpu().numpy().astype(np.float32)

    def regress_beta(self, embedding_512: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(embedding_512, np.float32))[None].to(self.device)
        with torch.no_grad():
            beta = self.regressor(t)
        return beta[0].cpu().numpy().astype(np.float32)

    def encode_single(self, det: Union[FaceDetection, np.ndarray]) -> np.ndarray:
        return self.regress_beta(self.extract_embedding(det))

    def view_weights(self, detections: List[FaceDetection]) -> np.ndarray:
        scores = np.array([
            float(d.det_score) * max(0.0, float(np.cos(np.radians(d.yaw_deg)))) ** self.yaw_weight_exponent
            for d in detections
        ], dtype=np.float64)
        z = scores / max(self.fusion_temperature, 1e-4)
        w = np.exp(z - z.max())
        return (w / w.sum()).astype(np.float32)

    def encode_multiview(self, detections: List[FaceDetection]) -> np.ndarray:
        dets = [d for d in detections if d is not None]
        if not dets:
            raise ValueError("No valid face detections provided for identity regression.")
        if len(dets) == 1:
            return self.encode_single(dets[0])
        weights = self.view_weights(dets)
        fused = sum(w * self.extract_embedding(d) for w, d in zip(weights, dets))
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return self.regress_beta(fused)
