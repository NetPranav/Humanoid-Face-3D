"""
Stage 6: per-texel UV texture backprojection (Phase 0 rewrite, DOCS/01 R2 + R4).

For each view:
  1. The *posed and expressed* mesh from Stage 2 (camera coordinates, fitted intrinsics)
     is rasterized into the photo to get a z-buffer. v1 projected onto the neutral mesh
     with a 5-point, f = width camera and never ran its occlusion test.
  2. Every UV texel is mapped to its 3D point on that mesh (UV-space rasterization with
     face ids), projected into the photo, and kept only if
       - it is the front-most surface at that pixel (z-buffer),
       - it faces the camera (cos > min_cos), and
       - the photo pixel is skin according to Stage 0 parsing (no hair, background, cloth).
  3. Kept texels are sampled bilinearly from the full-resolution photo and blended across
     views with weight cos^gamma.
No photo enhancement (v1's eye CLAHE/unsharp) is applied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import cv2
import numpy as np

from src.render.soft_raster import RasterOut, rasterize
from src.render.soft_renderer import project as project_points
from src.render.soft_renderer import sample_bilinear, vertex_normals


def sh_basis(n: np.ndarray) -> np.ndarray:
    """2nd-order real spherical-harmonic basis (9 terms, constants folded into the fit)."""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], 1)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c / 255.0, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return 255.0 * np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def fit_sh_shading(lin: np.ndarray, normals: np.ndarray, fit: np.ndarray, iters: int = 3,
                   clamp=(0.6, 1.6)) -> Dict[str, np.ndarray]:
    """
    First-order SH lighting on *luminance* from linear colours `lin` (N,3) and unit normals,
    fitted on the `fit` subset with albedo treated as constant there (robust: the worst 15%
    residuals are dropped each iteration, removing speculars, brows and moles).

    One grey shading for all channels, so delighting never shifts hue (colour cast is handled
    separately by white balance). Shading is normalized to median 1 on the fit set, so the
    average skin tone is kept, and clamped: coarse FLAME normals cannot justify stronger
    corrections (v1-style over-correction bleaches eye sockets and jaws).
    """
    Y = sh_basis(normals)[:, :4]
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    idx = np.nonzero(fit)[0]
    if len(idx) < 200:
        raise ValueError(f"Only {len(idx)} texels available to fit lighting; need at least 200.")
    sel = idx
    for _ in range(iters):
        L, *_ = np.linalg.lstsq(Y[sel], lum[sel], rcond=None)
        r = np.abs(Y[sel] @ L - lum[sel])
        sel = sel[r <= np.quantile(r, 0.85)]
    L = L / np.median((Y @ L)[fit])                     # normalized: shading = Y4(n) @ L, median 1
    shading = np.clip(Y @ L, clamp[0], clamp[1])
    return {"shading": np.repeat(shading[:, None], 3, 1), "coeffs": np.asarray(L), "clamp": clamp}


def gray_edge_illuminant(photo_bgr: np.ndarray, sigma: float = 2.0, p: float = 6.0) -> np.ndarray:
    """
    Illuminant colour (linear RGB, max-normalized) by the gray-edge hypothesis (van de Weijer,
    Gevers & Gijsenij 2007): the Minkowski-p mean of image derivatives is achromatic.
    """
    s = 1024.0 / max(photo_bgr.shape[:2])
    img = cv2.resize(photo_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else photo_bgr
    lin = srgb_to_linear(img[..., ::-1].astype(np.float64))
    lin = cv2.GaussianBlur(lin, (0, 0), sigma)
    gx = cv2.Sobel(lin, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(lin, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ok = (lin < 0.98).all(-1)                        # ignore clipped pixels
    e = np.power(np.power(mag[ok], p).mean(0), 1.0 / p)
    return e / e.max()


def white_balance_gains(photo_bgr: np.ndarray, max_gain: float = 1.25) -> np.ndarray:
    """Per-channel linear-RGB gains that neutralize the estimated illuminant (bounded)."""
    e = gray_edge_illuminant(photo_bgr)
    g = e.mean() / np.maximum(e, 1e-6)
    return np.clip(g / g.mean(), 1.0 / max_gain, max_gain)


def rasterize_uv(uv_coords: np.ndarray, uv_faces: np.ndarray, resolution: int) -> RasterOut:
    """UV-space raster: per-texel face id + barycentrics (texel centres, v up)."""
    xy = np.stack([uv_coords[:, 0] * resolution, (1.0 - uv_coords[:, 1]) * resolution], 1)
    return rasterize(xy, np.ones(len(uv_coords)), uv_faces, resolution, resolution)


class MultiViewTextureProjector:
    def __init__(
        self,
        texture_resolution: int = 2048,
        blend_gamma: float = 2.0,
        min_cos: float = 0.15,
        depth_tolerance_m: float = 0.004,
        zbuffer_max_side: int = 2048,
    ):
        self.resolution = int(texture_resolution)
        self.gamma = float(blend_gamma)
        self.min_cos = float(min_cos)
        self.depth_tol = float(depth_tolerance_m)
        self.zbuffer_max_side = int(zbuffer_max_side)
        self._uv_raster: Optional[RasterOut] = None
        self._uv_key = None

    def uv_raster(self, uv_coords: np.ndarray, uv_faces: np.ndarray) -> RasterOut:
        key = (uv_coords.shape, uv_faces.shape, float(uv_coords.sum()), self.resolution)
        if self._uv_raster is None or self._uv_key != key:
            self._uv_raster = rasterize_uv(uv_coords, uv_faces, self.resolution)
            self._uv_key = key
        return self._uv_raster

    def project(
        self,
        photos: Sequence[np.ndarray],
        vertices_cam: Sequence[np.ndarray],
        intrinsics_K: Sequence[np.ndarray],
        skin_masks: Sequence[np.ndarray],
        faces: np.ndarray,
        uv_coords: np.ndarray,
        uv_faces: np.ndarray,
        exclude_faces: Optional[np.ndarray] = None,
        delight_fit_region: Optional[np.ndarray] = None,
        delight_strength: Optional[np.ndarray] = None,
        white_balance: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        photos[i]       : (H, W, 3) uint8 BGR
        vertices_cam[i] : (V, 3) posed + expressed mesh in OpenCV camera coords (metres)
        intrinsics_K[i] : (3, 3)
        skin_masks[i]   : (H, W) bool, True where the photo shows skin
        exclude_faces   : optional (F,) bool, geometry faces that must not receive texture
        delight_fit_region : optional (R,R) bool UV mask of plain skin (no brows/eyes/lips). When
                          given, per-view SH lighting is fitted there and divided out, and the
                          result also contains 'albedo_rgb' (delit, average tone preserved).
        delight_strength : optional (R,R) float in [0,1]: exponent on the correction per texel
                          (lower where coarse normals are unreliable, e.g. eyes and lips).
        white_balance   : neutralize the light colour (gray-edge, bounded). Off by default: on single
                          photos clothing and backgrounds dominate the estimate (a blue suit
                          turned skin green in testing); proper colour-cast removal is Phase 3B.

        Returns projected_rgb (R,R,3) float32 BGR [0,255], projection_mask (R,R) uint8,
        weight_map (R,R) float32, uv_valid (R,R) bool, per_view_coverage (n_views,).
        """
        if not (len(photos) == len(vertices_cam) == len(intrinsics_K) == len(skin_masks)):
            raise ValueError("photos, vertices_cam, intrinsics_K and skin_masks must have equal length.")
        R = self.resolution
        uvr = self.uv_raster(uv_coords, uv_faces)
        texel = uvr.mask.copy()
        if exclude_faces is not None:
            texel &= ~np.asarray(exclude_faces, bool)[np.clip(uvr.face_id, 0, None)]
        ty, tx = np.nonzero(texel)
        fid = uvr.face_id[ty, tx]
        bary = uvr.bary[ty, tx].astype(np.float64)
        tri = faces[fid]

        accum = np.zeros((R, R, 3), np.float64)
        accum_alb = np.zeros((R, R, 3), np.float64)
        wsum = np.zeros((R, R), np.float64)
        coverage, sh_coeffs = [], []
        fit_texel = delight_fit_region[ty, tx] if delight_fit_region is not None else None
        strength_texel = delight_strength[ty, tx] if delight_strength is not None else None
        wb_gains = []
        for photo, vc, K, skin in zip(photos, vertices_cam, intrinsics_K, skin_masks):
            H, W = photo.shape[:2]
            vc = np.asarray(vc, np.float64)
            K = np.asarray(K, np.float64)
            pts = (vc[tri] * bary[..., None]).sum(1)
            vn = vertex_normals(vc, faces)
            nrm = (vn[tri] * bary[..., None]).sum(1)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            cos = -(nrm * (pts / np.linalg.norm(pts, axis=1, keepdims=True))).sum(1)

            # z-buffer at reduced resolution (occlusion is a low-frequency test)
            s = min(1.0, self.zbuffer_max_side / max(H, W))
            Ks = K.copy()
            Ks[:2] *= s
            hs, ws = int(round(H * s)), int(round(W * s))
            vxy, vz = project_points(vc, Ks)
            zb = rasterize(vxy, vz, faces, hs, ws).depth

            pxy, pz = project_points(pts, K)
            inb = (pxy[:, 0] >= 0) & (pxy[:, 0] < W - 1) & (pxy[:, 1] >= 0) & (pxy[:, 1] < H - 1)
            zx = np.clip((pxy[:, 0] * s).astype(np.int64), 0, ws - 1)
            zy = np.clip((pxy[:, 1] * s).astype(np.int64), 0, hs - 1)
            front = pz <= zb[zy, zx] + self.depth_tol
            skin_ok = skin[np.clip(pxy[:, 1].astype(np.int64), 0, H - 1),
                           np.clip(pxy[:, 0].astype(np.int64), 0, W - 1)]
            keep = inb & front & skin_ok & (cos > self.min_cos)
            coverage.append(float(keep.sum()) / max(1, len(keep)))
            if not keep.any():
                continue
            col = sample_bilinear(photo, pxy[keep, 0] - 0.5, pxy[keep, 1] - 0.5).astype(np.float64)
            w = np.clip(cos[keep], 0, 1) ** self.gamma
            np.add.at(accum, (ty[keep], tx[keep]), col * w[:, None])
            np.add.at(wsum, (ty[keep], tx[keep]), w)
            if fit_texel is not None:
                lin = srgb_to_linear(col[:, ::-1])                       # BGR -> linear RGB
                sh = fit_sh_shading(lin, nrm[keep], fit_texel[keep])
                corr = sh["shading"]
                if strength_texel is not None:
                    corr = corr ** strength_texel[keep][:, None]
                gains = white_balance_gains(photo) if white_balance else np.ones(3)
                wb_gains.append(gains)
                alb = linear_to_srgb(lin / corr * gains[None])[:, ::-1]   # back to BGR sRGB
                np.add.at(accum_alb, (ty[keep], tx[keep]), alb * w[:, None])
                sh_coeffs.append(sh["coeffs"])

        observed = wsum > 0
        rgb = np.zeros((R, R, 3), np.float32)
        rgb[observed] = (accum[observed] / wsum[observed, None]).astype(np.float32)
        out_extra = {}
        if fit_texel is not None:
            alb = np.zeros((R, R, 3), np.float32)
            alb[observed] = (accum_alb[observed] / wsum[observed, None]).astype(np.float32)
            out_extra = {"albedo_rgb": alb, "sh_coeffs": np.asarray(sh_coeffs, np.float32),
                         "white_balance_gains": np.asarray(wb_gains, np.float32)}
        return {
            **out_extra,
            "projected_rgb": rgb,
            "projection_mask": np.where(observed, 255, 0).astype(np.uint8),
            "weight_map": wsum.astype(np.float32),
            "uv_valid": uvr.mask,
            "per_view_coverage": np.asarray(coverage, np.float32),
        }

    def save_maps(self, result: Dict[str, np.ndarray], output_dir: Union[str, Path], prefix: str = "head") -> Dict[str, str]:
        tex_dir = Path(output_dir) / "textures"
        tex_dir.mkdir(parents=True, exist_ok=True)
        p_rgb = tex_dir / f"{prefix}_projected_raw.png"
        p_mask = tex_dir / f"{prefix}_projection_mask.png"
        cv2.imwrite(str(p_rgb), np.clip(result["projected_rgb"], 0, 255).astype(np.uint8))
        cv2.imwrite(str(p_mask), result["projection_mask"])
        return {"projected_texture": str(p_rgb), "projection_mask": str(p_mask)}
