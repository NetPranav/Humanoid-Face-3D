"""
Stage 0: semantic parsing of the portrait (MediaPipe multiclass selfie segmenter, Apache-2.0).

Classes: 0 background, 1 hair, 2 body skin (neck, ears, hands), 3 face skin (incl. eyes,
lips, brows), 4 clothes, 5 other (accessories). Stage 6 only samples texture from skin
classes, so hair, background and clothing never land in the skin UV (DOCS/01 R4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)
SKIN_CLASSES = (BODY_SKIN, FACE_SKIN)

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models_cache" / "mediapipe" / "selfie_multiclass_256x256.tflite"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)


class FaceParser:
    def __init__(self, model_path: Optional[Union[str, Path]] = None, max_side: int = 1536):
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"Parsing model not found at {path}. Download with:\n  curl -L -o {path} {_MODEL_URL}"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise RuntimeError("mediapipe is required for face parsing: pip install mediapipe") from e
        self._mp = mp
        opts = vision.ImageSegmenterOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(path)), output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(opts)
        self.max_side = max_side

    def parse(self, image_bgr: np.ndarray) -> np.ndarray:
        """Returns an (H, W) uint8 class map at the input resolution."""
        h, w = image_bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(image_bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else image_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        res = self.segmenter.segment(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        cls = np.squeeze(res.category_mask.numpy_view()).astype(np.uint8)
        if cls.shape != (h, w):
            cls = cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)
        return cls

    @staticmethod
    def skin_mask(classes: np.ndarray, erode_px: int = 0) -> np.ndarray:
        m = np.isin(classes, SKIN_CLASSES).astype(np.uint8)
        if erode_px > 0:
            m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
        return m.astype(bool)
