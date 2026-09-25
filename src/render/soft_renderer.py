"""
Evaluation renderer: textured / shaded views of a head mesh (CPU).

Used by the Phase 0 evaluation harness for overlays (render over the photo with the
fitted camera) and turntables. Not a look-dev renderer: Lambert + ambient only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.render.soft_raster import interpolate, rasterize


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup of img at float pixel coords (pixel-centre = integer). Returns (N, C) float32."""
    n = len(x)
    cols = 1024
    rows = max(1, -(-n // cols))
    pad = rows * cols - n
    mx = np.pad(np.asarray(x, np.float32), (0, pad)).reshape(rows, cols)
    my = np.pad(np.asarray(y, np.float32), (0, pad)).reshape(rows, cols)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n].astype(np.float32)


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, f[:, i], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def project(vc: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = vc[:, 2]
    xy = np.stack([vc[:, 0] / z * K[0, 0] + K[0, 2], vc[:, 1] / z * K[1, 1] + K[1, 2]], 1)
    return xy, z


def render(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    hw: Tuple[int, int],
    uv: Optional[np.ndarray] = None,
    uv_faces: Optional[np.ndarray] = None,
    texture_bgr: Optional[np.ndarray] = None,
    base_color=(0.72, 0.72, 0.72),
    light_dir=(0.35, 0.45, 0.82),
    ambient: float = 0.45,
    textured_shading: bool = True,
    face_subset: Optional[np.ndarray] = None,
    sh_lum: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renders verts in OpenCV camera coordinates. Returns (bgr uint8, alpha bool).
    With a texture, shading multiplies the (already lit) photo texture lightly so the
    render stays comparable to the photo.
    """
    H, W = hw
    verts_cam = np.asarray(verts_cam, np.float64)
    xy, z = project(verts_cam, K)
    r = rasterize(xy, z, faces, H, W, face_subset=face_subset)
    m = r.mask
    n = vertex_normals(verts_cam, faces)
    N = interpolate(n, faces, r)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    # light_dir = direction the light travels (camera frame: +z away from camera)
    L = np.asarray(light_dir, np.float64)
    L /= np.linalg.norm(L)
    lam = np.clip(-(N * L).sum(-1), 0, 1)
    shade = ambient + (1 - ambient) * lam

    img = np.zeros((H, W, 3), np.float32)
    if texture_bgr is not None:
        th, tw = texture_bgr.shape[:2]
        tri_uv = uv_faces[r.face_id[m]]
        b = r.bary[m]
        uvp = (uv[tri_uv] * b[..., None]).sum(1)
        col = sample_bilinear(texture_bgr, uvp[:, 0] * tw - 0.5, (1.0 - uvp[:, 1]) * th - 0.5)
        if sh_lum is not None:
            # re-light a delit albedo with fitted first-order SH (camera frame), in linear light
            from src.stage6_texture.projector import linear_to_srgb, sh_basis, srgb_to_linear
            nn = N[m]
            shade_sh = np.clip(sh_basis(nn)[:, :4] @ np.asarray(sh_lum), 0.6, 1.6)
            col = linear_to_srgb(srgb_to_linear(col[:, ::-1]) * shade_sh[:, None])[:, ::-1].astype(np.float32)
        elif textured_shading:
            col *= (0.75 + 0.25 * shade[m])[:, None]
        img[m] = col
    else:
        img[m] = np.asarray(base_color, np.float32)[::-1] * 255.0 * shade[m][:, None]
    return np.clip(img, 0, 255).astype(np.uint8), m


def turntable_camera(
    verts_flame: np.ndarray,
    yaw_deg: float,
    hw: Tuple[int, int] = (768, 768),
    fov_deg: float = 12.0,
    fill: float = 0.8,
    pitch_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Places a FLAME-frame mesh (+y up, +z forward) in front of a camera rotated by yaw.
    Returns (verts_cam, K). Framing uses the head bounding box (neck excluded roughly).
    """
    H, W = hw
    v = np.asarray(verts_flame, np.float64)
    v = v - (v.min(0) + v.max(0)) / 2
    a = np.radians(yaw_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = np.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    vr = np.einsum("ij,vj->vi", Rx @ Ry, v)
    vr = vr * np.array([1.0, -1.0, -1.0])            # FLAME -> OpenCV camera axes
    vr[:, :2] -= (vr[:, :2].min(0) + vr[:, :2].max(0)) / 2
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    extent = max(np.ptp(vr[:, 0]), np.ptp(vr[:, 1]))
    dist = f * extent / (fill * min(H, W))
    vr[:, 2] += dist
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return vr, K
