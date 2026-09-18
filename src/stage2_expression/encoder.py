import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from src.stage0_preprocess.detector import FaceDetection

class ExpressionEncoder:
    """
    Wraps SMIRK to extract 100-dim FLAME expression parameters (psi),
    15-dim pose parameters (theta), and coarse detail displacement maps.
    """
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            print(f"[Stage 2] Note: SMIRK checkpoint not found at {self.checkpoint_path}. Operating in neutral expression fallback.")
            return None

        import sys
        vendor_paths = [
            '/kaggle/working/pipeline/vendor/smirk',
            '/kaggle/working/face-geo-pipeline/vendor/smirk',
            './vendor/smirk'
        ]
        for p in vendor_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            model = torch.load(self.checkpoint_path, map_location=self.device)
            if hasattr(model, 'eval'):
                model.eval()
            return model
        except Exception as e:
            print(f"[Stage 2] Warning: SMIRK loading skipped ({e}). Using neutral fallback.")
            return None

    def encode(self, detection: FaceDetection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts:
            expression_psi: (100,) float32 array
            pose_theta: (15,) float32 array
            coarse_detail: (128, 128) float32 displacement map
        """
        if self.model is None or detection is None:
            return np.zeros(100, dtype=np.float32), np.zeros(15, dtype=np.float32), np.zeros((128, 128), dtype=np.float32)

        img = torch.from_numpy(detection.crop_224).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)
            if isinstance(output, dict):
                psi = output.get('expression', torch.zeros(1, 100)).squeeze(0).cpu().numpy()
                theta = output.get('pose', torch.zeros(1, 15)).squeeze(0).cpu().numpy()
                coarse = output.get('detail', torch.zeros(1, 1, 128, 128)).squeeze().cpu().numpy()
            else:
                psi = np.zeros(100, dtype=np.float32)
                theta = np.zeros(15, dtype=np.float32)
                coarse = np.zeros((128, 128), dtype=np.float32)

        return psi.astype(np.float32), theta.astype(np.float32), coarse.astype(np.float32)
