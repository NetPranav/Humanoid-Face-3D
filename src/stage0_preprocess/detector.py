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
    crop_224: Optional[np.ndarray] = None  # (224, 224, 3) aligned crop for SMIRK / EMOCA
    embedding: Optional[np.ndarray] = None  # (512,) ArcFace normalized feature embedding
    pitch_deg: float = 0.0    # Head pitch angle in degrees
    roll_deg: float = 0.0     # Head roll angle in degrees

    @property
    def landmarks_5(self) -> np.ndarray:
        return self.landmarks_5pt

    @property
    def kps(self) -> np.ndarray:
        return self.landmarks_5pt

    @property
    def yaw(self) -> float:
        return self.yaw_deg

    @property
    def pitch(self) -> float:
        return self.pitch_deg

    @property
    def roll(self) -> float:
        return self.roll_deg

    @property
    def score(self) -> float:
        return self.det_score

class FaceDetector:
    """
    Detects faces, computes 5-point landmarks, estimates head orientation,
    and extracts canonical aligned crops for downstream networks.
    """
    def __init__(self, ctx_id: int = 0, allow_degraded: bool = False):
        self.allow_degraded = allow_degraded
        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.has_insightface = True
        except Exception as e:
            if not self.allow_degraded:
                raise RuntimeError(
                    "InsightFace is required for Stage 0 face detection and alignment. "
                    "Install with: pip install insightface onnxruntime-gpu"
                ) from e
            print(f"[Stage 0] Warning: Running in degraded mode (InsightFace missing: {e}).")
            self.app = None
            self.has_insightface = False

    def detect(self, image_bgr: np.ndarray) -> List[FaceDetection]:
        if not self.has_insightface or self.app is None:
            if not self.allow_degraded:
                raise RuntimeError("Face detection requested but InsightFace is not initialized.")
            h, w = image_bgr.shape[:2]
            return [FaceDetection(
                bbox=np.array([0, 0, w, h], dtype=np.float32),
                landmarks_5pt=np.zeros((5, 2), dtype=np.float32),
                det_score=0.99,
                yaw_deg=0.0,
                crop_112=cv2.resize(image_bgr, (112, 112)),
                crop_224=cv2.resize(image_bgr, (224, 224)),
                embedding=np.zeros(512, dtype=np.float32),
                pitch_deg=0.0,
                roll_deg=0.0,
            )]

        faces = self.app.get(image_bgr)
        if not faces and self.allow_degraded:
            h, w = image_bgr.shape[:2]
            return [FaceDetection(
                bbox=np.array([0, 0, w, h], dtype=np.float32),
                landmarks_5pt=np.zeros((5, 2), dtype=np.float32),
                det_score=0.99,
                yaw_deg=0.0,
                crop_112=cv2.resize(image_bgr, (112, 112)),
                crop_224=cv2.resize(image_bgr, (224, 224)),
                embedding=np.zeros(512, dtype=np.float32),
                pitch_deg=0.0,
                roll_deg=0.0,
            )]
        results = []
        for face in faces:
            emb = None
            if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                emb = face.normed_embedding
            elif hasattr(face, 'embedding') and face.embedding is not None:
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-12)

            # InsightFace buffalo_l populates pose as [pitch, yaw, roll]
            pose = getattr(face, 'pose', None)
            pitch = float(pose[0]) if pose is not None and len(pose) >= 1 else 0.0
            yaw = float(pose[1]) if pose is not None and len(pose) >= 2 else 0.0
            roll = float(pose[2]) if pose is not None and len(pose) >= 3 else 0.0

            det = FaceDetection(
                bbox=face.bbox,
                landmarks_5pt=face.kps,
                det_score=float(face.det_score),
                yaw_deg=yaw,
                crop_112=self._align_crop(image_bgr, face.kps, size=112),
                crop_224=self._align_crop(image_bgr, face.kps, size=224),
                embedding=emb,
                pitch_deg=pitch,
                roll_deg=roll,
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
