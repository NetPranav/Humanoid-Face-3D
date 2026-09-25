"""
Stage 0: dense facial landmarks with MediaPipe Face Landmarker (Apache-2.0).

478 points: the 468-vertex face mesh + 10 iris points (468-472 and 473-477), returned in
pixel coordinates of the input image (z in the same pixel units, relative depth), plus the
52 ARKit-style blendshape scores. Replaces InsightFace detection/landmarks on the commercial
path (DOCS/07 Phase C2).

MediaPipe is pinned to 0.10.35: 1.0.x crashes on macOS because its face detector opens a
Metal helper even with the CPU delegate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np

_MODEL = Path(__file__).resolve().parents[2] / "models_cache" / "mediapipe" / "face_landmarker.task"
_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
        "float16/latest/face_landmarker.task")

IRIS_A = slice(468, 473)       # centre + 4 contour points
IRIS_B = slice(473, 478)
# MediaPipe face-oval indices (silhouette-ish; down-weighted by the fitter)
FACE_OVAL = np.array([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
                      377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109])
# eye-opening and inner-lip rings (used to mask pixels, not as fit targets)
EYE_A = np.array([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])
EYE_B = np.array([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398])
LIPS_INNER = np.array([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191])


@dataclass
class MPLandmarks:
    points: np.ndarray                     # (478, 3) px (x, y, z)
    blendshapes: Dict[str, float] = field(default_factory=dict)
    image_hw: tuple = (0, 0)

    @property
    def iris_centres(self) -> np.ndarray:
        return np.stack([self.points[468, :2], self.points[473, :2]])

    @property
    def iod(self) -> float:
        """Inter-pupillary distance in pixels (iris centres)."""
        return float(np.linalg.norm(self.points[468, :2] - self.points[473, :2]))

    @property
    def bbox(self) -> np.ndarray:
        p = self.points[:468, :2]
        return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])

    def polygon_mask(self, idx: np.ndarray, dilate_px: int = 0) -> np.ndarray:
        h, w = self.image_hw
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(self.points[idx, :2]).astype(np.int32)], 1)
        if dilate_px > 0:
            m = cv2.dilate(m, np.ones((2 * dilate_px + 1,) * 2, np.uint8))
        return m.astype(bool)


class MediaPipeLandmarker:
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        path = Path(model_path) if model_path else _MODEL
        if not path.exists():
            raise FileNotFoundError(f"MediaPipe face landmarker not found at {path}. Download with:\n"
                                    f"  curl -L -o {path} {_URL}")
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(path)),
            output_face_blendshapes=True, num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, image_bgr: np.ndarray) -> Optional[MPLandmarks]:
        h, w = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        r = self.landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not r.face_landmarks:
            return None
        pts = np.array([[q.x * w, q.y * h, q.z * w] for q in r.face_landmarks[0]], np.float64)
        bs = {b.category_name: float(b.score) for b in r.face_blendshapes[0]} if r.face_blendshapes else {}
        return MPLandmarks(points=pts, blendshapes=bs, image_hw=(h, w))
