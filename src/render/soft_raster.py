"""
Minimal CPU triangle rasterizer with a z-buffer (NumPy).

Used for (a) real occlusion testing during texture projection (v1's visibility test was
never called, DOCS/01 R4), (b) UV-space rasterization with per-texel face ids, and
(c) evaluation renders (overlays, turntables). Perspective-correct barycentrics.
A GPU rasterizer (nvdiffrast / PyTorch3D) replaces this in the Phase 1 fitter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RasterOut:
    face_id: np.ndarray      # (H, W) int32, -1 = empty
    bary: np.ndarray         # (H, W, 3) float32 perspective-correct barycentrics
    depth: np.ndarray        # (H, W) float32, +inf = empty

    @property
    def mask(self) -> np.ndarray:
        return self.face_id >= 0


def rasterize(
    xy: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    height: int,
    width: int,
    cull_backfaces: bool = False,
    face_subset: Optional[np.ndarray] = None,
) -> RasterOut:
    """
    xy : (V, 2) pixel coordinates (pixel centres at integer + 0.5 convention: x=0.5 is the
         centre of column 0). z : (V,) positive depth (larger = farther). For orthographic /
         UV rasterization pass z = 1.
    """
    face_id = np.full((height, width), -1, np.int32)
    depth = np.full((height, width), np.inf, np.float32)
    bary = np.zeros((height, width, 3), np.float32)

    fids = np.arange(len(faces)) if face_subset is None else np.asarray(face_subset)
    tri = faces[fids]
    p0, p1, p2 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    z0, z1, z2 = z[tri[:, 0]], z[tri[:, 1]], z[tri[:, 2]]
    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])

    keep = (np.abs(area) > 1e-12) & (z0 > 0) & (z1 > 0) & (z2 > 0)
    if cull_backfaces:
        keep &= area < 0  # image y points down: front faces are clockwise in pixel space
    xmin = np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0]) - 0.5).astype(np.int64)
    xmax = np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0]) - 0.5).astype(np.int64)
    ymin = np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1]) - 0.5).astype(np.int64)
    ymax = np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1]) - 0.5).astype(np.int64)
    xmin, ymin = np.clip(xmin, 0, width - 1), np.clip(ymin, 0, height - 1)
    xmax, ymax = np.clip(xmax, 0, width - 1), np.clip(ymax, 0, height - 1)
    keep &= (xmax >= xmin) & (ymax >= ymin)

    for k in np.nonzero(keep)[0]:
        xs = np.arange(xmin[k], xmax[k] + 1) + 0.5
        ys = np.arange(ymin[k], ymax[k] + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        a0, a1, a2 = p0[k], p1[k], p2[k]
        w0 = (a1[0] - X) * (a2[1] - Y) - (a1[1] - Y) * (a2[0] - X)
        w1 = (a2[0] - X) * (a0[1] - Y) - (a2[1] - Y) * (a0[0] - X)
        w2 = (a0[0] - X) * (a1[1] - Y) - (a0[1] - Y) * (a1[0] - X)
        w0, w1, w2 = w0 / area[k], w1 / area[k], w2 / area[k]
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        # perspective-correct: 1/z is affine in screen space
        iz = w0 / z0[k] + w1 / z1[k] + w2 / z2[k]
        zz = 1.0 / iz
        sl = (slice(ymin[k], ymax[k] + 1), slice(xmin[k], xmax[k] + 1))
        closer = inside & (zz < depth[sl])
        if not closer.any():
            continue
        depth[sl][closer] = zz[closer]
        face_id[sl][closer] = fids[k]
        b = np.stack([w0 / z0[k], w1 / z1[k], w2 / z2[k]], -1) * zz[..., None]
        bary[sl][closer] = b[closer]
    return RasterOut(face_id, bary, depth)


def interpolate(attr: np.ndarray, faces: np.ndarray, r: RasterOut) -> np.ndarray:
    """Interpolates a per-vertex attribute (V, C) over the raster -> (H, W, C)."""
    out = np.zeros(r.face_id.shape + (attr.shape[1],), np.float32)
    m = r.mask
    tri = faces[r.face_id[m]]
    b = r.bary[m]
    out[m] = (attr[tri] * b[..., None]).sum(1)
    return out
