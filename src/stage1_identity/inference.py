import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from src.stage0_preprocess.detector import FaceDetection
from src.utils.flame_model import FLAMEModel

# Standalone PyTorch definition of MICA's shape mapping network
# Allows running the MLP regressor directly from checkpoint weights without full MICA package overhead
if torch is not None:
    class MappingNetwork(nn.Module):
        """
        MLP Mapping Network matching MICA architecture.
        Maps 512-D ArcFace normalized feature embedding to 300-D FLAME beta coefficients.
        """
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
            for i, l in enumerate(self.network):
                h = self.network[i](h)
                h = F.leaky_relu(h, negative_slope=0.2)
                if i in self.skips:
                    h = torch.cat([z, h], 1)
            return self.output(h)
else:
    MappingNetwork = None


class MICAIdentityEncoder:
    """
    Identity shape encoder based on MICA (ArcFace backbone + MLP regressor).
    Predicts 300-dimensional FLAME shape parameters (beta).
    Fuses multiple camera angles in ArcFace embedding space weighted by det_score * cos^2(yaw).
    """
    def __init__(
        self,
        checkpoint_path: str,
        flame_model: Optional[FLAMEModel] = None,
        device: str = 'cuda',
        fusion_space: str = 'embedding',
        fusion_temperature: float = 0.05,
        yaw_weight_exponent: int = 2
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for MICAIdentityEncoder. Install torch>=2.1.0.")

        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.flame = flame_model
        self.checkpoint_path = Path(checkpoint_path)
        self.fusion_space = fusion_space
        self.fusion_temperature = fusion_temperature
        self.yaw_weight_exponent = yaw_weight_exponent

        self.arcface = None
        self.regressor = None
        self.model = self._load_mica()

    def _load_mica(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MICA weights not found at: {self.checkpoint_path}. "
                "Download weights or attach Kaggle dataset input."
            )

        # Register vendor paths dynamically
        vendor_paths = [
            str(Path(__file__).resolve().parents[2] / 'vendor' / 'MICA'),
            '/kaggle/working/face-geo-pipeline/vendor/MICA',
            '/kaggle/working/pipeline/vendor/MICA',
            './vendor/MICA'
        ]
        for p in vendor_paths:
            if Path(p).exists() and p not in sys.path:
                sys.path.insert(0, p)

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            # 1. Initialize mapping regressor (512 -> 300)
            self.regressor = MappingNetwork(z_dim=512, map_hidden_dim=300, map_output_dim=300, hidden=3)

            if isinstance(ckpt, dict) and 'flameModel' in ckpt:
                flame_state = ckpt['flameModel']
                # Strip 'regressor.' prefix if present
                reg_state = {}
                for k, v in flame_state.items():
                    if k.startswith('regressor.'):
                        reg_state[k.replace('regressor.', '')] = v
                if reg_state:
                    self.regressor.load_state_dict(reg_state, strict=False)
                else:
                    self.regressor.load_state_dict(flame_state, strict=False)
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                self.regressor.load_state_dict(ckpt['state_dict'], strict=False)
            elif isinstance(ckpt, dict):
                self.regressor.load_state_dict(ckpt, strict=False)

            self.regressor.to(self.device).eval()

            # 2. Try loading ArcFace feature extractor if available in checkpoint
            if isinstance(ckpt, dict) and 'arcface' in ckpt:
                try:
                    from models.arcface import Arcface
                    self.arcface = Arcface()
                    self.arcface.load_state_dict(ckpt['arcface'])
                    self.arcface.to(self.device).eval()
                except Exception:
                    self.arcface = None

            return self.regressor

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MICA model from {self.checkpoint_path}: {e}. "
                "Verify checkpoint integrity and vendor/MICA submodule."
            ) from e

    def extract_embedding(self, det: FaceDetection) -> np.ndarray:
        """
        Extracts a normalized 512-dim ArcFace embedding vector.
        Prioritizes pre-extracted embedding from FaceDetection; falls back to ArcFace model on crop_112.
        """
        if det.embedding is not None:
            emb = det.embedding.astype(np.float32).copy()
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-8)

        if self.arcface is not None:
            # Correct color space (OpenCV BGR -> RGB) and normalization ([-1, 1])
            img_rgb = det.crop_112[:, :, ::-1].copy()
            img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
            t_img = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.arcface(t_img)
                feat = F.normalize(feat, dim=1)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)

        raise RuntimeError(
            "Cannot extract identity embedding: detection has no embedding and MICA ArcFace backbone is not loaded."
        )

    def regress_beta(self, embedding_512: np.ndarray) -> np.ndarray:
        """
        Passes a 512-dim ArcFace embedding through the MICA MLP regressor to predict 300-dim beta.
        """
        if self.regressor is None:
            raise RuntimeError("MICA regressor is not initialized.")

        t_emb = torch.from_numpy(embedding_512).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            beta = self.regressor(t_emb)
        return beta.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_single(self, det: Union[FaceDetection, np.ndarray]) -> np.ndarray:
        """
        Encodes a single view into a 300-dim beta vector.
        Accepts either FaceDetection or raw crop_112 BGR image.
        """
        if isinstance(det, np.ndarray):
            # Wrap raw BGR crop into minimal FaceDetection
            det = FaceDetection(
                bbox=np.zeros(4),
                landmarks_5pt=np.zeros((5, 2)),
                det_score=1.0,
                yaw_deg=0.0,
                crop_112=det,
                embedding=None
            )

        emb = self.extract_embedding(det)
        return self.regress_beta(emb)

    def encode_multiview(
        self,
        detections: List[FaceDetection],
        fusion_space: Optional[str] = None,
        temperature: Optional[float] = None,
        yaw_weight_exponent: Optional[int] = None
    ) -> np.ndarray:
        """
        Multi-view identity shape regression with confidence and pose weighting.
        
        Weighting Formulation:
            w_i = det_score_i * max(0, cos(yaw_i))^yaw_weight_exponent
            weights = softmax(w / temperature)
        
        Default Fusion (embedding space):
            bar_emb = sum(w_i * emb_i) / ||sum(w_i * emb_i)||
            beta = regressor(bar_emb)   [Run regressor ONCE]
            
        Fallback Fusion (beta space):
            beta = sum(w_i * beta_i)
        """
        valid_dets = [d for d in detections if d is not None]
        if not valid_dets:
            raise ValueError("No valid face detections provided for multi-view identity fusion.")

        space = fusion_space or self.fusion_space
        temp = temperature if temperature is not None else self.fusion_temperature
        yaw_exp = yaw_weight_exponent if yaw_weight_exponent is not None else self.yaw_weight_exponent

        if len(valid_dets) == 1:
            return self.encode_single(valid_dets[0])

        # 1. Compute view weights: det_score * cos^N(yaw)
        scores = []
        for det in valid_dets:
            yaw_rad = np.radians(det.yaw_deg)
            cos_yaw = max(0.0, float(np.cos(yaw_rad)))
            frontality = cos_yaw ** yaw_exp
            raw_weight = float(det.det_score) * frontality
            scores.append(raw_weight)

        scores = np.array(scores, dtype=np.float32)
        # Softmax with temperature
        scaled_scores = scores / max(temp, 1e-4)
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)

        # 2. Embedding space fusion (MICA standard, recommended)
        if space == 'embedding':
            fused_emb = np.zeros(512, dtype=np.float32)
            for w, det in zip(weights, valid_dets):
                emb = self.extract_embedding(det)
                fused_emb += w * emb

            # Re-normalize to unit hypersphere
            fused_emb_norm = fused_emb / (np.linalg.norm(fused_emb) + 1e-8)
            return self.regress_beta(fused_emb_norm)

        # 3. Beta space fusion (fallback)
        elif space == 'beta':
            betas = [self.encode_single(det) for det in valid_dets]
            fused_beta = np.zeros(300, dtype=np.float32)
            for w, b in zip(weights, betas):
                fused_beta += w * b
            return fused_beta

        else:
            raise ValueError(f"Unknown fusion_space '{space}'. Expected 'embedding' or 'beta'.")
