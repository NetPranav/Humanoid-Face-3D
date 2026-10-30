"""
Input detail check (Stage 0): refuse photos that cannot support the requested reconstruction.

Two explicit checks, both raise `InsufficientDetailError` (never warn-and-continue, AGENTS.md rule 1):

1. Per photo: facial pixel density (inter-pupillary px -> px/mm) and sharpness of the skin
   (variance of the Laplacian at a canonical face scale, so it does not depend on image size).
2. Per photo set (>= 2 photos): cross-view feature tracking. SIFT keypoints inside the skin masks
   are matched between view pairs and verified with a RANSAC fundamental matrix. Few verified
   matches means the views share no trackable skin texture (blur, make-up, different sittings),
   which is exactly when multi-view photo-consistency has nothing to lock onto.

The fix the messages ask for is a higher-resolution / sharper ORIGINAL. Upscaling a small photo
does not restore detail; it invents texture, which the photo-consistency solver would turn into
false relief.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np

MEAN_IPD_MM = 63.0          # adult mean inter-pupillary distance
CANON_IOD_PX = 120.0        # face scale at which sharpness is measured


class InsufficientDetailError(ValueError):
    """The input photo(s) lack the detail needed for a faithful reconstruction."""


def px_per_mm(iod_px: float) -> float:
    return float(iod_px) / MEAN_IPD_MM


def skin_sharpness(image_bgr: np.ndarray, skin_mask: np.ndarray, iod_px: float) -> float:
    """Variance of the Laplacian over skin pixels after rescaling the face to a canonical size."""
    s = CANON_IOD_PX / max(float(iod_px), 1.0)
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=interp).astype(np.float32)
    m = cv2.resize(skin_mask.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if m.sum() < 500:
        return 0.0
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_32F)
    return float(lap[m].var())


def check_photo(name: str, image_bgr: np.ndarray, skin_mask: np.ndarray, iod_px: float, cfg: Dict) -> Dict:
    ppm = px_per_mm(iod_px)
    sharp = skin_sharpness(image_bgr, skin_mask, iod_px)
    need_ppm = float(cfg.get("min_px_per_mm", 3.0))
    need_sharp = float(cfg.get("min_sharpness", 2.0))
    if ppm < need_ppm:
        raise InsufficientDetailError(
            f"{name}: the face is only {ppm:.1f} px/mm (inter-pupillary {iod_px:.0f} px); at least "
            f"{need_ppm:.1f} px/mm ({need_ppm * MEAN_IPD_MM:.0f} px between the pupils) is needed to carry "
            f"skin detail. Provide a higher-resolution original or move closer to the face. Upscaling a small "
            f"photo will not recover scars or pores: it invents texture.")
    if sharp < need_sharp:
        raise InsufficientDetailError(
            f"{name}: the skin is too soft (sharpness {sharp:.1f}, need >= {need_sharp:.1f}). It looks blurred, "
            f"smoothed or heavily compressed. Provide a sharp, in-focus, uncompressed original; "
            f"beautify filters and upscalers remove exactly the detail this pipeline measures.")
    return {"px_per_mm": ppm, "sharpness": sharp}


def match_pair(img_a, mask_a, img_b, mask_b, n_features: int = 1500, ratio: float = 0.8) -> int:
    """Number of RANSAC-verified SIFT matches between two views, restricted to skin."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    ga, gb = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (img_a, img_b))
    ka, da = sift.detectAndCompute(ga, mask_a.astype(np.uint8) * 255)
    kb, db = sift.detectAndCompute(gb, mask_b.astype(np.uint8) * 255)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = []
    for pair in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        return len(good) if len(good) < 8 else 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 3.0, 0.99)
    return int(inl.sum()) if inl is not None else 0


def check_tracking(names: Sequence[str], images: Sequence[np.ndarray], skin_masks: Sequence[np.ndarray],
                   cfg: Dict) -> List[int]:
    """Every photo must track (>= min_matches verified skin matches) to at least one other photo."""
    n = len(images)
    if n < 2:
        return []
    need = int(cfg.get("min_track_matches", 12))
    best = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            c = match_pair(images[i], skin_masks[i], images[j], skin_masks[j])
            best[i], best[j] = max(best[i], c), max(best[j], c)
    for k, c in enumerate(best):
        if c < need:
            raise InsufficientDetailError(
                f"{names[k]}: only {c} skin features could be tracked to the other photos (need >= {need}). "
                f"Multi-view reconstruction has no texture to lock onto. Provide sharper originals of the same "
                f"face taken in one session with similar lighting and no beauty filter; or pass a single photo.")
    return best


def run_checks(names, images, skin_masks, iods, cfg: Dict) -> Dict:
    """Entry point from the pipeline. `cfg` is `stage0.detail_check`."""
    report = {"photos": [check_photo(n, im, m, iod, cfg) for n, im, m, iod in zip(names, images, skin_masks, iods)]}
    if cfg.get("tracking", True):
        report["tracked_matches"] = check_tracking(names, images, skin_masks, cfg)
    return report
