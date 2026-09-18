import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FaceDetection:
    bbox: np.ndarray          # (4,) [x1, y1, x2, y2]
    landmarks_5pt: np.ndarray # (5, 2) 5-point key landmarks
    det_score: float          # InsightFace detection confidence (used for beta fusion weights)
    yaw_deg: float            # Head yaw angle in degrees
    crop_112: np.ndarray      # (112, 112, 3) aligned crop for MICA / ArcFace
    crop_224: np.ndarray      # (224, 224, 3) aligned crop for SMIRK / EMOCA

class FaceDetector:
    """
    Detects faces, computes 5-point landmarks, estimates head orientation,
    and extracts canonical aligned crops for downstream networks.
    """
    def __init__(self, ctx_id: int = 0):
        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.has_insightface = True
        except Exception as e:
            print(f"[Stage 0] Warning: InsightFace initialization deferred ({e}).")
            self.app = None
            self.has_insightface = False

    def detect(self, image_bgr: np.ndarray) -> List[FaceDetection]:
        if not self.has_insightface or self.app is None:
            # Fallback simple crop if insightface isn't installed in environment
            h, w = image_bgr.shape[:2]
            c112 = cv2.resize(image_bgr, (112, 112))
            c224 = cv2.resize(image_bgr, (224, 224))
            return [FaceDetection(
                bbox=np.array([0, 0, w, h], dtype=np.float32),
                landmarks_5pt=np.zeros((5, 2), dtype=np.float32),
                det_score=0.99,
                yaw_deg=0.0,
                crop_112=c112,
                crop_224=c224,
            )]

        faces = self.app.get(image_bgr)
        results = []
        for face in faces:
            yaw = float(face.pose[1]) if hasattr(face, 'pose') and face.pose is not None else 0.0
            det = FaceDetection(
                bbox=face.bbox,
                landmarks_5pt=face.kps,
                det_score=float(face.det_score),
                yaw_deg=yaw,
                crop_112=self._align_crop(image_bgr, face.kps, size=112),
                crop_224=self._align_crop(image_bgr, face.kps, size=224),
            )
            results.append(det)
        return results

    def _align_crop(self, img: np.ndarray, kps: np.ndarray, size: int) -> np.ndarray:
        """Standard ArcFace similarity-transform alignment."""
        try:
            from insightface.utils import face_align
            return face_align.norm_crop(img, kps, image_size=size)
        except Exception:
            return cv2.resize(img, (size, size))

    def detect_single(self, image_bgr: np.ndarray) -> Optional[FaceDetection]:
        """Returns the highest-confidence face detection, or None."""
        dets = self.detect(image_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: d.det_score)
