"""
Phase 3A: synthesize the subject's own skin grain into UV regions no photo observed.

The unseen area (back of head, scalp, far side, under the chin) previously got only a
smooth interpolated colour. Here its high-frequency layer is synthesized from the subject's
own observed, delit skin (forehead and cheeks) with patch-based image quilting (Efros &
Freeman 2001): candidates are chosen by SSD on the already-synthesized overlap, and overlaps
are combined with variance-preserving blending (Heitz & Neyret 2018) so the grain keeps its
contrast instead of averaging out. No colour is invented: the low-frequency colour still
comes from the observed texture.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * (m[..., None] if img.ndim == 3 else m), (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    den = np.maximum(den, 1e-6)
    return num / (den[..., None] if img.ndim == 3 else den)


def high_pass(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    hp = img.astype(np.float32) - masked_blur(img, mask, sigma)
    hp[~mask] = 0
    return hp


def _candidates(source: np.ndarray, patch: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Top-left corners of patches that lie completely inside `source`."""
    inside = cv2.erode(source.astype(np.uint8), np.ones((patch, patch), np.uint8), anchor=(0, 0),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(inside)
    if len(ys) == 0:
        return np.zeros((0, 2), np.int64)
    sel = rng.choice(len(ys), size=min(max_n, len(ys)), replace=False)
    return np.stack([ys[sel], xs[sel]], 1)


def quilt(
    exemplar: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    patch: int = 48,
    overlap: int = 12,
    n_pool: int = 4000,
    n_try: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """
    Fills `target` (R,R bool) with a quilt of patches taken from `exemplar` where `source` is
    True. Returns an (R,R,C) float32 image (zero outside target).
    """
    rng = np.random.default_rng(seed)
    R = target.shape[0]
    C = exemplar.shape[2]
    cand = _candidates(source, patch, n_pool, rng)
    while len(cand) < 50 and patch > 12:
        patch //= 2
        overlap = max(3, patch // 4)
        cand = _candidates(source, patch, n_pool, rng)
    if len(cand) == 0:
        raise ValueError("No clean skin area large enough to take texture patches from.")
    pool = np.stack([exemplar[y:y + patch, x:x + patch] for y, x in cand]).astype(np.float32)

    step = patch - overlap
    win1 = np.minimum(np.arange(patch) + 1, np.arange(patch)[::-1] + 1).astype(np.float32)
    win1 = np.minimum(win1 / overlap, 1.0)
    win = np.outer(win1, win1)

    acc = np.zeros((R + patch, R + patch, C), np.float32)
    wsq = np.zeros((R + patch, R + patch), np.float32)
    ys, xs = np.nonzero(target)
    y_lo, y_hi, x_lo, x_hi = ys.min(), ys.max(), xs.min(), xs.max()
    tgt = np.pad(target, ((0, patch), (0, patch)))
    for y0 in range(max(0, y_lo - overlap), y_hi + 1, step):
        for x0 in range(max(0, x_lo - overlap), x_hi + 1, step):
            if not tgt[y0:y0 + patch, x0:x0 + patch].any():
                continue
            filled = wsq[y0:y0 + patch, x0:x0 + patch] > 0
            idx = rng.choice(len(pool), size=min(n_try, len(pool)), replace=False)
            if filled.any():
                cur = acc[y0:y0 + patch, x0:x0 + patch] / np.sqrt(np.maximum(wsq[y0:y0 + patch, x0:x0 + patch], 1e-8))[..., None]
                diff = (pool[idx] - cur[None]) ** 2
                ssd = (diff.sum(-1) * filled[None]).sum((1, 2))
                best = ssd.min()
                ok = idx[ssd <= best * 1.1 + 1e-6]
                pick = rng.choice(ok)
            else:
                pick = idx[0]
            acc[y0:y0 + patch, x0:x0 + patch] += pool[pick] * win[..., None]
            wsq[y0:y0 + patch, x0:x0 + patch] += win ** 2
    out = acc[:R, :R] / np.sqrt(np.maximum(wsq[:R, :R], 1e-8))[..., None]
    out[~target] = 0
    return out


def synthesize_skin_detail(
    texture: np.ndarray,
    observed: np.ndarray,
    source_region: np.ndarray,
    target: np.ndarray,
    sigma_px: float = 5.0,
    patch: int = 48,
    seed: int = 0,
    gain: Optional[float] = 1.0,
) -> np.ndarray:
    """
    Returns the synthesized high-frequency layer (R,R,3) for `target`. The exemplar is the
    high-pass of `texture` over clean observed skin (`observed & source_region`), shrunk so
    patches never straddle a mask border.
    """
    src = observed & source_region
    src = cv2.erode(src.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if src.sum() < patch * patch * 4:
        raise ValueError("Too little observed clean skin to build a texture exemplar.")
    hp = high_pass(texture, observed, sigma_px)
    synth = quilt(hp, src, target, patch=patch, overlap=max(4, patch // 4), seed=seed)
    return synth * float(gain)
