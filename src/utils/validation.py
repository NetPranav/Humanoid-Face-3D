import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional
from src.stage0_preprocess.detector import FaceDetector, FaceDetection

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detections: List[Optional[FaceDetection]] = field(default_factory=list)

def validate_inputs(
    photo_paths: List[str],
    detector: Optional[FaceDetector] = None,
    min_photos: int = 3,
    same_person_thresh: float = 0.40,
    min_face_confidence: float = 0.50
) -> ValidationResult:
    """
    Validates input photos prior to running reconstruction.
    Verifies face presence, detection confidence, pairwise identity consistency,
    pose coverage, and angular spread.
    Reuses pre-extracted ArcFace embeddings from FaceDetection.
    """
    result = ValidationResult(is_valid=True)
    if detector is None:
        detector = FaceDetector()

    images = []
    detections = []
    embeddings = []

    # 1. Face detection in each photo
    for idx, path in enumerate(photo_paths):
        img = cv2.imread(str(path))
        if img is None:
            result.errors.append(f"Could not load image at index {idx+1}: {path}")
            result.is_valid = False
            detections.append(None)
            continue

        det = detector.detect_single(img)
        if det is None:
            result.errors.append(f"No face detected in photo {idx+1} ({path}). Ensure face is visible and well-lit.")
            result.is_valid = False
            detections.append(None)
        elif det.det_score < min_face_confidence:
            result.errors.append(
                f"Face detection confidence in photo {idx+1} ({det.det_score:.2f}) is below threshold ({min_face_confidence:.2f})."
            )
            result.is_valid = False
            detections.append(None)
        else:
            detections.append(det)
            images.append(img)
            if det.embedding is not None:
                embeddings.append(det.embedding)

    result.detections = detections
    if not result.is_valid:
        return result

    # 2. Minimum photo count check
    if len(photo_paths) < min_photos:
        result.errors.append(f"Provided {len(photo_paths)} photos. A minimum of {min_photos} photos is required for robust 3D reconstruction.")
        result.is_valid = False

    # 3. Same person verification across all pairs
    if len(embeddings) == len(photo_paths):
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim < same_person_thresh:
                    result.errors.append(
                        f"Photos {i+1} and {j+1} appear to be different individuals (similarity: {sim:.2f}, threshold: {same_person_thresh:.2f})."
                    )
                    result.is_valid = False

    # 4. Near-frontal view existence (|yaw| < 30°)
    valid_yaws = [d.yaw_deg for d in detections if d is not None]
    if valid_yaws and not any(abs(y) < 30.0 for y in valid_yaws):
        result.errors.append("No near-frontal photo detected (yaw < 30°). Please include at least one front-facing portrait.")
        result.is_valid = False

    # 5. Angular coverage spread check
    if len(valid_yaws) >= 2 and (max(valid_yaws) - min(valid_yaws)) < 45.0:
        result.warnings.append("Input photos have limited angular diversity (< 45° spread). Adding quarter-profile or profile shots improves 3D depth.")

    return result
