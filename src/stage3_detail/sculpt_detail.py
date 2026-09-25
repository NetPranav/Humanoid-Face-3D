"""
Phase 4A: sculpt-quality geometric detail with a single `detail_level` knob (0–100).

Two sources, kept separate on purpose (DOCS/04 Phase 4):

  meso  (0.4–3 mm: wrinkles, folds, crow's feet, forehead and nasolabial lines)
        PHOTO-DERIVED. Band-passed log-luminance of the *delit* texture, "dark is deep"
        (Beeler et al. 2010, mesoscopic augmentation). Guard rails that v1 Tier 2 lacked:
          - runs on the registered, delit texture, never on raw photo lighting;
          - band-pass only (no low frequencies are ever integrated), so whole features
            cannot be embossed;
          - brows, eye openings, lip border, nose, ears and eyeballs are masked out
            (landmark + official FLAME region masks);
          - pigment is suppressed where chroma changes along with luminance (moles,
            freckles, stubble shadow are colour changes, shading is not);
          - only observed or symmetry-inferred texels contribute.

  micro (< 0.2 mm: pores, cross-hatched micro-grooves, lip striations)
        SYNTHESIZED. Below the pixel footprint of a portrait, so they are generated with
        region-dependent statistics, sized in millimetres using the per-texel scale map.

The knob scales both amplitudes and picks the subdivision level of the exported sculpt mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from src.stage7_delight.skin_synthesis import masked_blur


@dataclass
class DetailSettings:
    level: float                 # 0..100
    meso_gain: float             # multiplies photo-derived meso (0..1)
    micro_gain: float            # multiplies synthesized micro (0..1)
    subdivision: int             # Loop subdivision level of the exported sculpt mesh

    @classmethod
    def from_level(cls, level: float) -> "DetailSettings":
        lv = float(np.clip(level, 0, 100))
        sub = 1 if lv < 25 else 2 if lv < 60 else 3 if lv < 90 else 4
        return cls(level=lv, meso_gain=min(1.0, lv / 60.0), micro_gain=float(np.clip((lv - 20) / 80.0, 0, 1)),
                   subdivision=sub)


# physical scales (mm)
MESO_SCALES = ((0.45, 0.07), (0.9, 0.20), (1.8, 0.40))   # (crease width sigma mm, max groove depth mm)
PORE = {                          # region: (pores per mm^2, radius mm, depth mm)
    "nose": (1.4, 0.10, 0.060),
    "face": (0.9, 0.07, 0.040),
    "forehead": (0.8, 0.07, 0.040),
    "eye_region": (0.25, 0.045, 0.015),
    "scalp": (0.5, 0.05, 0.020),
    "neck": (0.5, 0.05, 0.020),
    "ears": (0.3, 0.04, 0.015),
}
GROOVE_MM = 0.020                 # cross-hatch groove depth
UNDULATION_MM = 0.015             # ~1 mm-scale skin unevenness
LIP_STRIATION_MM = 0.08


def _resize(a: np.ndarray, R: int, nearest: bool = False) -> np.ndarray:
    if a.shape[0] == R:
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(a.astype(np.float32) if not nearest else a.astype(np.uint8), (R, R), interpolation=interp)


def photo_meso(albedo_bgr: np.ndarray, usable: np.ndarray, mask: np.ndarray, texel_mm: float) -> np.ndarray:
    """
    Photo-derived wrinkle grooves (mm, <= 0) at the albedo resolution; 0 outside `mask`.

    Wrinkles are dark *lines* in the delit texture while pigment (blotches, freckles, pores)
    is mostly *blob*-shaped, so a plain band-pass turns skin tone into crumpled relief.
    Instead, multi-scale Hessian valley detection (Frangi et al. 1998 line measure) keeps
    elongated dark creases only; groove depth grows with the crease scale. Pigment is further
    suppressed where chroma changes along with luminance.
    """
    lab = cv2.cvtColor(albedo_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum = np.log(lab[..., 0] / 255.0 * 0.99 + 0.01)
    m = usable & mask
    lum = np.where(m, lum, masked_blur(lum, m, 3.0 / texel_mm))       # no edges from mask borders
    ab = lab[..., 1:]
    cmag = np.linalg.norm(masked_blur(ab, m, 0.4 / texel_mm) - masked_blur(ab, m, 3.0 / texel_mm), axis=-1)
    pigment_w = np.exp(-(cmag / (np.median(cmag[m]) * 3.0 + 1e-6)) ** 2)

    depth = np.zeros_like(lum)
    for sigma_mm, depth_mm in MESO_SCALES:
        s_px = sigma_mm / texel_mm
        g = cv2.GaussianBlur(lum, (0, 0), s_px)
        lxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * s_px ** 2
        lyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * s_px ** 2
        lxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * s_px ** 2
        tr, det = lxx + lyy, lxx * lyy - lxy ** 2
        disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc                      # l1 >= l2
        valley = l1 > 0                                            # dark line: strong positive curvature across
        rb = np.abs(l2) / (np.abs(l1) + 1e-9)
        S = np.sqrt(l1 ** 2 + l2 ** 2)
        c = np.percentile(S[m], 95) + 1e-9
        v = np.exp(-rb ** 2 / (2 * 0.5 ** 2)) * (1 - np.exp(-S ** 2 / (2 * (0.5 * c) ** 2))) * valley
        depth = np.maximum(depth, v * depth_mm)
    depth = _keep_elongated(depth, texel_mm)
    feather = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8 / texel_mm)
    return (-depth * pigment_w * feather).astype(np.float32)


def _keep_elongated(depth: np.ndarray, texel_mm: float, min_len_mm: float = 2.5, min_aspect: float = 2.0) -> np.ndarray:
    """
    Removes crease components that are short or round: the edges of moles and freckles give
    small closed rings, real wrinkles are long and thin. Weak responses are left untouched.
    """
    strong = depth > 0.25 * float(depth.max() + 1e-9)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    ys, xs = np.nonzero(strong)
    ids = lab[ys, xs]
    for i in range(1, n):
        sel = ids == i
        if sel.sum() < 3:
            continue
        pts = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
        ev = np.linalg.eigvalsh(np.cov(pts.T) + 1e-6 * np.eye(2))
        length_mm = 4.0 * np.sqrt(ev[1]) * texel_mm
        keep[i] = length_mm >= min_len_mm and np.sqrt(ev[1] / ev[0]) >= min_aspect
    drop = strong & ~keep[lab]
    drop = cv2.dilate(drop.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return np.where(drop, 0.0, depth).astype(np.float32)


def _oriented_kernel(length_px: float, width_px: float, angle_deg: float) -> np.ndarray:
    n = int(max(3, 2 * round(length_px * 1.5) + 1))
    y, x = np.mgrid[-(n // 2):n // 2 + 1, -(n // 2):n // 2 + 1].astype(np.float32)
    a = np.radians(angle_deg)
    u = x * np.cos(a) + y * np.sin(a)
    v = -x * np.sin(a) + y * np.cos(a)
    k = np.exp(-0.5 * (u / length_px) ** 2 - 0.5 * (v / max(width_px, 0.5)) ** 2)
    return k / k.sum()


def _unit_noise(shape, rng) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def synth_micro(regions: Dict[str, np.ndarray], texel_mm: np.ndarray, R: int, seed: int = 0) -> np.ndarray:
    """Synthesized micro displacement (mm) at resolution R."""
    rng = np.random.default_rng(seed)
    valid = _resize(regions["_valid"], R, nearest=True).astype(bool)
    tmm = _resize(texel_mm, R)
    tmm = np.where(valid, np.maximum(tmm, 1e-3), 1.0)
    face_mm = float(np.median(tmm[_resize(regions["face"], R, True).astype(bool)]))
    out = np.zeros((R, R), np.float32)

    # pores: Bernoulli-sampled pits with density per mm^2, Gaussian profile of the region's radius
    ears = regions["left_ear"] | regions["right_ear"]
    region_order = [("scalp", regions["scalp"]), ("neck", regions["neck"]), ("ears", ears),
                    ("face", regions["face"]), ("forehead", regions["forehead"]),
                    ("eye_region", regions["eye_region"]), ("nose", regions["nose"])]
    lips = _resize(regions["lips"] | regions["lips_landmark"], R, True).astype(bool)
    eyes = _resize(regions["left_eyeball"] | regions["right_eyeball"] | regions["eye_openings"], R, True).astype(bool)
    assigned = np.zeros((R, R), bool)
    for name, reg in reversed(region_order):          # most specific region wins
        m = _resize(reg, R, True).astype(bool) & valid & ~assigned & ~lips & ~eyes
        assigned |= m
        dens, rad, depth = PORE[name]
        p = dens * tmm ** 2
        hit = (rng.random((R, R)) < p) & m
        imp = np.zeros((R, R), np.float32)
        imp[hit] = rng.uniform(0.5, 1.0, hit.sum()).astype(np.float32)
        sig = rad / face_mm / 1.2
        pits = cv2.GaussianBlur(imp, (0, 0), max(sig, 0.6))
        pk = cv2.GaussianBlur(np.pad(np.ones((1, 1), np.float32), 20), (0, 0), max(sig, 0.6)).max()
        out -= pits / pk * depth * m

    skin = valid & ~eyes
    # cross-hatched micro-grooves (two families, ~0.6 mm long, thin)
    L, W = 0.6 / face_mm, 0.03 / face_mm
    for ang in (35.0, -35.0):
        n = cv2.filter2D(_unit_noise((R, R), rng), -1, _oriented_kernel(L, W, ang))
        n = n - cv2.GaussianBlur(n, (0, 0), L)
        n /= n.std() + 1e-6
        out -= np.clip(n - 1.0, 0, None) * GROOVE_MM * 0.5 * (skin & ~lips)
    # gentle undulation (~1 mm)
    und = cv2.GaussianBlur(_unit_noise((R, R), rng), (0, 0), 1.0 / face_mm)
    und /= und.std() + 1e-6
    out += und * UNDULATION_MM * skin
    # lip striations: vertical grooves in UV (FLAME lips are laid out horizontally)
    if lips.any():
        k = _oriented_kernel(2.5 / face_mm, 0.16 / face_mm, 90.0)
        s = cv2.filter2D(_unit_noise((R, R), rng), -1, k)
        s = s - cv2.GaussianBlur(s, (0, 0), 0.9 / face_mm)
        s /= s.std() + 1e-6
        out -= np.clip(s, 0, None) * LIP_STRIATION_MM * cv2.GaussianBlur(lips.astype(np.float32), (0, 0), 3)
    return out


def normal_map(disp_mm: np.ndarray, texel_mm: np.ndarray) -> np.ndarray:
    """OpenGL tangent-space normal map (RGB uint8) from displacement in mm (+v up)."""
    t = np.maximum(texel_mm, 1e-3)
    dx = cv2.Sobel(disp_mm, cv2.CV_32F, 1, 0, ksize=3) / 8.0 / t
    dy = -cv2.Sobel(disp_mm, cv2.CV_32F, 0, 1, ksize=3) / 8.0 / t   # image rows go down, v goes up
    n = np.stack([-dx, -dy, np.ones_like(dx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def build_detail(
    albedo_bgr: np.ndarray,
    provenance: np.ndarray,
    regions: Dict[str, np.ndarray],
    level: float,
    out_res: int = 4096,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Returns dict(displacement_mm (out_res²), normal_rgb, meso_mm, micro_mm, settings).
    `regions` is FaceGeoPipeline.uv_regions() at the albedo resolution (includes 'texel_mm').
    """
    st = DetailSettings.from_level(level)
    R0 = albedo_bgr.shape[0]
    texel_mm = regions["texel_mm"]
    valid = texel_mm > 0
    regs = dict(regions)
    regs["_valid"] = valid

    features = regions["brows"] | regions["eye_openings"] | regions["lip_border"] | regions["lips_landmark"]
    face_mm0 = float(np.median(texel_mm[regions["face"] & valid]))
    k = int(2 * round(2.0 / face_mm0) + 1)                       # ~2 mm margin around brows/eyes/lips
    features = cv2.dilate(features.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    meso_mask = (regions["face"] | regions["forehead"]) & ~features & ~regions["nose"] & \
        ~regions["left_eyeball"] & ~regions["right_eyeball"] & ~regions["lips"]
    meso_mask = cv2.erode(meso_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    usable = provenance >= 128                          # observed or symmetry-inferred
    face_mm = float(np.median(texel_mm[regions["face"] & valid]))
    meso = photo_meso(albedo_bgr, usable, meso_mask, face_mm) if st.meso_gain > 0 else np.zeros((R0, R0), np.float32)

    micro = synth_micro(regs, texel_mm, out_res, seed) if st.micro_gain > 0 else np.zeros((out_res, out_res), np.float32)
    disp = _resize(meso, out_res) * st.meso_gain + micro * st.micro_gain
    # nothing on the eyeballs; neck seam contract: zero at the FLAME boundary and bottom collar rows
    eyes = _resize(regions["left_eyeball"] | regions["right_eyeball"], out_res, True).astype(bool)
    seam = _resize(regions["boundary"], out_res, True).astype(np.float32)
    seam = cv2.GaussianBlur(cv2.dilate(seam, np.ones((9, 9), np.uint8)), (0, 0), 6)
    disp = disp * (1 - np.clip(seam * 2, 0, 1))
    disp[eyes] = 0
    disp[int(0.85 * out_res):, :] = 0
    tmm = _resize(texel_mm, out_res) * (R0 / out_res)
    return {
        "displacement_mm": disp.astype(np.float32),
        "normal_rgb": normal_map(disp, np.where(tmm > 0, tmm, 1.0)),
        "meso_mm": meso,
        "micro_mm": micro,
        "settings": st,
    }


def apply_to_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    disp_mm: np.ndarray,
    subdivision: int,
    pinned_vertices: Optional[np.ndarray] = None,
):
    """Loop-subdivides the neutral mesh and displaces it along smooth normals (metres)."""
    from src.render.soft_renderer import sample_bilinear, vertex_normals
    from src.utils.subdivision import loop_subdivide

    n0 = len(vertices)
    v, f, u, uf = loop_subdivide(vertices.astype(np.float64), faces, uv, uv_faces, levels=subdivision)
    vuv = np.zeros((len(v), 2))
    vuv[f.reshape(-1)] = u[uf.reshape(-1)]
    R = disp_mm.shape[0]
    d = sample_bilinear(disp_mm[..., None], vuv[:, 0] * R - 0.5, (1 - vuv[:, 1]) * R - 0.5)[:, 0]
    if pinned_vertices is not None:
        d[pinned_vertices[pinned_vertices < n0]] = 0   # original vertices keep their index after Loop
    n = vertex_normals(v, f)
    return v + n * (d[:, None] / 1000.0), f, u, uf
