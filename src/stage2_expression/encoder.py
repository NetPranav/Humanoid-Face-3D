import numpy as np
from pathlib import Path
from typing import Tuple

try:
    import torch
except ImportError:
    torch = None

from src.stage0_preprocess.detector import FaceDetection

class ExpressionEncoder:
    """
    Wraps SMIRK to extract 100-dim FLAME expression parameters (psi),
    15-dim pose parameters (theta), and coarse detail displacement maps.
    """
    def __init__(self, checkpoint_path: str, device: str = 'cuda', allow_neutral: bool = False):
        if torch is None and not allow_neutral:
            raise RuntimeError("PyTorch is required for ExpressionEncoder. Install torch>=2.1.0.")
        self.device = device if (torch is not None and torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.checkpoint_path = Path(checkpoint_path)
        self.allow_neutral = allow_neutral
        self.model = self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            if self.allow_neutral:
                print(f"[Stage 2] Note: SMIRK checkpoint not found at {self.checkpoint_path}. Operating in neutral expression mode.")
                return None
            raise FileNotFoundError(
                f"SMIRK checkpoint not found at {self.checkpoint_path}. "
                "Download weights or run scripts/fetch_models.py."
            )

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
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            # If ckpt is a dict/OrderedDict with smirk_encoder keys, filter and strip prefix
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                state_dict = None

            if state_dict is not None and not hasattr(state_dict, 'eval'):
                # Extract encoder sub-dict
                encoder_dict = {
                    k.replace('smirk_encoder.', ''): v
                    for k, v in state_dict.items()
                    if k.startswith('smirk_encoder.')
                }
                # Construct SMIRK encoder if module exists in vendor
                try:
                    from smirk.models.smirk_encoder import SmirkEncoder
                    model = SmirkEncoder()
                    model.load_state_dict(encoder_dict if encoder_dict else state_dict)
                    model.to(self.device)
                    model.eval()
                    return model
                except Exception:
                    # Return state_dict if architecture module is unavailable
                    return state_dict

            if hasattr(ckpt, 'eval'):
                ckpt.eval()
            return ckpt
        except Exception as e:
            if self.allow_neutral:
                print(f"[Stage 2] Warning: SMIRK loading skipped ({e}). Using neutral fallback.")
                return None
            raise RuntimeError(f"Failed to load SMIRK model from {self.checkpoint_path}: {e}") from e

    def encode(self, detection: FaceDetection) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts:
            expression_psi: (100,) float32 array
            pose_theta: (15,) float32 array
        Note: Stage 2 is expression/pose only; all geometric micro-displacement
        is synthesized downstream in Stage 3.
        """
        if detection is None:
            raise ValueError("Cannot encode expression: detection is None.")

        if self.model is None:
            if self.allow_neutral:
                return np.zeros(100, dtype=np.float32), np.zeros(15, dtype=np.float32)
            raise RuntimeError("SMIRK expression encoder model is not loaded.")

        img = torch.from_numpy(detection.crop_224).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            if callable(self.model):
                output = self.model(img)
            else:
                output = {}

            if isinstance(output, dict):
                psi = output.get('expression', torch.zeros(1, 100)).squeeze(0).cpu().numpy()
                theta = output.get('pose', torch.zeros(1, 15)).squeeze(0).cpu().numpy()
            else:
                psi = np.zeros(100, dtype=np.float32)
                theta = np.zeros(15, dtype=np.float32)

        return psi.astype(np.float32), theta.astype(np.float32)
