"""
Stage 7 (Phase 0): provenance-aware UV completion. Replaces v1's procedural "delighting".

v1 invented colour (DOCS/01 R6): a fixed target luminance, painted sclera, a +12 red "ear
capillary boost" in fixed UV rectangles and seeded noise. That produced the pink/purple
cast. Phase 0 does no delighting (that needs a learned model, roadmap Phase 3) and only
fills unobserved texels, from data:

  low  = push-pull (normalized-convolution pyramid) interpolation of the observed texture,
         giving smooth lighting continuity across holes;
  high = detail of the *mirrored* texel (FLAME is near-symmetric), i.e. mirror - blur(mirror),
         used only where the mirrored texel was observed.

  filled = low + high   (where a mirror exists)   provenance 128 "inferred (symmetry)"
  filled = low + synth  (elsewhere, Phase 3A)      provenance  64 "synthesized": the subject's
                         own skin grain quilted from clean observed skin (skin_synthesis.py)
  filled = low          (no exemplar available)   provenance   0 "filled (interpolated)"
  observed texels are kept exactly                 provenance 255 "observed"
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np

from src.render.soft_raster import RasterOut

OBSERVED, INFERRED, SYNTHESIZED, FILLED = 255, 128, 64, 0


def push_pull(img: np.ndarray, weight: np.ndarray, levels: int = 12) -> np.ndarray:
    """Fills zero-weight pixels by pyramid normalized convolution (smooth, no seams)."""
    img = img.astype(np.float32)
    w = weight.astype(np.float32)
    pyr = []
    cur_i, cur_w = img * w[..., None], w
    for _ in range(levels):
        pyr.append((cur_i, cur_w))
        if min(cur_w.shape) <= 2:
            break
        cur_i = cv2.pyrDown(cur_i)
        cur_w = cv2.pyrDown(cur_w)
    acc_i, acc_w = pyr[-1]
    fill = acc_i / np.maximum(acc_w, 1e-8)[..., None]
    for lvl_i, lvl_w in reversed(pyr[:-1]):
        up = cv2.resize(fill, (lvl_w.shape[1], lvl_w.shape[0]), interpolation=cv2.INTER_LINEAR)
        own = lvl_i / np.maximum(lvl_w, 1e-8)[..., None]
        a = np.clip(lvl_w * 4.0, 0, 1)[..., None]
        fill = a * own + (1 - a) * up
    return fill


def mesh_harmonic_low(
    tex: np.ndarray,
    observed: np.ndarray,
    uv_raster: RasterOut,
    faces: np.ndarray,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    n_verts: int,
    sigma_px: float = 10.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Low-frequency colour for unobserved texels, interpolated on the *mesh* rather than in UV.

    UV-space interpolation fills each side of a UV seam from different data, which leaves a
    visible seam (e.g. down the back of the head). Here observed colour is averaged per
    vertex, unobserved vertices are solved harmonically over the mesh graph (vertices are
    shared across UV seams), and the vertex colours are rasterized back into UV.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from src.stage7_delight.skin_synthesis import masked_blur

    R = observed.shape[0]
    low_obs = masked_blur(tex, observed, sigma_px)
    obs_soft = cv2.erode(observed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    # sample each face corner slightly inside its triangle: vertex UVs sit on island borders,
    # where the eroded observation mask would otherwise never see them
    tri_uv = uv[uv_faces]                                   # (F,3,2)
    inner = 0.75 * tri_uv + 0.25 * tri_uv.mean(1, keepdims=True)
    cu = np.clip((inner[..., 0].reshape(-1) * R).astype(np.int64), 0, R - 1)
    cv = np.clip(((1 - inner[..., 1].reshape(-1)) * R).astype(np.int64), 0, R - 1)
    vid = faces.reshape(-1)
    ok = obs_soft[cv, cu]
    acc = np.zeros((n_verts, 3))
    cnt = np.zeros(n_verts)
    np.add.at(acc, vid[ok], low_obs[cv[ok], cu[ok]])
    np.add.at(cnt, vid[ok], 1)
    known = cnt > 0
    if not known.any():
        raise ValueError("No observed vertices to interpolate colour from.")
    val = np.zeros((n_verts, 3))
    val[known] = acc[known] / cnt[known, None]
    mean = val[known].mean(0)

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_verts, n_verts)).tocsr()
    A.data[:] = 1.0
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    unk = np.nonzero(~known)[0]
    kn = np.nonzero(known)[0]
    if len(unk):
        Luu = L[unk][:, unk] + eps * sp.eye(len(unk))
        rhs = -(L[unk][:, kn] @ val[kn]) + eps * mean[None]
        solve = spla.factorized(Luu.tocsc())
        val[unk] = np.stack([solve(rhs[:, c]) for c in range(3)], 1)

    out = np.zeros((R, R, 3), np.float32)
    m = uv_raster.mask
    tri = faces[uv_raster.face_id[m]]
    out[m] = (val[tri] * uv_raster.bary[m][..., None]).sum(1)
    return out


def build_mirror_map(
    uv_raster: RasterOut,
    template_vertices: np.ndarray,
    faces: np.ndarray,
    max_dist_m: float = 0.003,
) -> np.ndarray:
    """
    For every valid texel, the (y, x) of the texel at the mirrored (x -> -x) 3D position on
    the FLAME template, or (-1, -1). Computed once per UV resolution.
    """
    from scipy.spatial import cKDTree

    m = uv_raster.mask
    ty, tx = np.nonzero(m)
    tri = faces[uv_raster.face_id[ty, tx]]
    pos = (template_vertices[tri] * uv_raster.bary[ty, tx][..., None]).sum(1)
    tree = cKDTree(pos)
    mir = pos * np.array([-1.0, 1.0, 1.0])
    d, j = tree.query(mir, k=1)
    out = np.full(m.shape + (2,), -1, np.int32)
    ok = d < max_dist_m
    out[ty[ok], tx[ok], 0] = ty[j[ok]]
    out[ty[ok], tx[ok], 1] = tx[j[ok]]
    return out


class ProvenanceUVFill:
    def __init__(self, detail_sigma_px: float = 6.0, pad_px: int = 8):
        self.detail_sigma = float(detail_sigma_px)
        self.pad_px = int(pad_px)

    def fill(
        self,
        texture_bgr: np.ndarray,
        observed: np.ndarray,
        uv_valid: np.ndarray,
        mirror_map: np.ndarray,
        exemplar_region: Optional[np.ndarray] = None,
        seed: int = 0,
        mesh: Optional[dict] = None,
        closed_regions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        exemplar_region: (R,R) bool UV mask of plain skin (e.g. forehead + cheeks without brows,
        eyes, lips). When given, unseen texels without a mirrored observation receive the
        subject's own synthesized skin grain on top of the interpolated colour.
        mesh: optional dict(uv_raster, faces, uv, uv_faces, n_verts). When given, the colour of
        unseen regions is interpolated on the mesh (seam-free) instead of in UV space.
        closed_regions: optional {name: (R,R) bool}. Unseen texels inside such a region are
        interpolated only from observed texels of the same region (e.g. lips take lip colour,
        never skin or teeth), and get no synthesized skin grain.
        """
        tex = texture_bgr.astype(np.float32)
        obs = observed.astype(bool) & uv_valid
        if not obs.any():
            raise ValueError("No observed texels to fill from; projection produced an empty texture.")

        low = push_pull(tex, obs.astype(np.float32))
        if mesh is not None:
            low_mesh = mesh_harmonic_low(tex, obs, mesh["uv_raster"], mesh["faces"], mesh["uv"],
                                         mesh["uv_faces"], mesh["n_verts"])
            # keep the UV push-pull right at observed borders (continuity), the mesh field beyond
            dist = cv2.distanceTransform((~obs).astype(np.uint8), cv2.DIST_L2, 5)
            t = np.clip(dist / 24.0, 0, 1)[..., None]
            low = np.where(uv_valid[..., None], (1 - t) * low + t * low_mesh, low)

        # mirrored colour and its detail layer (normalized blur over observed texels only)
        my, mx = mirror_map[..., 0], mirror_map[..., 1]
        has_m = (my >= 0) & uv_valid
        has_m[has_m] &= obs[my[has_m], mx[has_m]]
        mir = np.zeros_like(tex)
        mir[has_m] = tex[my[has_m], mx[has_m]]
        wm = has_m.astype(np.float32)
        blur_num = cv2.GaussianBlur(mir * wm[..., None], (0, 0), self.detail_sigma)
        blur_den = cv2.GaussianBlur(wm, (0, 0), self.detail_sigma)
        mir_low = blur_num / np.maximum(blur_den, 1e-6)[..., None]
        high = np.where(has_m[..., None], mir - mir_low, 0.0)

        out = low.copy()
        need = uv_valid & ~obs
        inferred = need & has_m
        out[inferred] = low[inferred] + high[inferred]
        closed_any = np.zeros_like(obs)
        for reg in (closed_regions or {}).values():
            reg = reg & uv_valid
            src = obs & reg
            if src.sum() > 50:
                inside = push_pull(tex, src.astype(np.float32))
                fill_here = reg & ~obs
                out[fill_here] = inside[fill_here]
                closed_any |= reg
        inferred &= ~closed_any

        synthesized = np.zeros_like(obs)
        if exemplar_region is not None:
            from src.stage7_delight.skin_synthesis import synthesize_skin_detail
            synthesized = need & ~has_m & ~closed_any
            if synthesized.any():
                grain = synthesize_skin_detail(tex, obs, exemplar_region, synthesized, seed=seed)
                out[synthesized] = low[synthesized] + grain[synthesized]
        out[obs] = tex[obs]

        prov = np.zeros(obs.shape, np.uint8)
        prov[obs] = OBSERVED
        prov[inferred] = INFERRED
        prov[synthesized] = SYNTHESIZED
        prov[~uv_valid] = 0

        # pad colour beyond UV island borders so bilinear/mip lookups never see black
        if self.pad_px > 0:
            k = 2 * self.pad_px + 1
            dil = cv2.dilate(uv_valid.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
            outside = dil & ~uv_valid
            out[outside] = low[outside]
        return {
            "albedo_srgb": np.clip(out, 0, 255).astype(np.uint8),
            "provenance": prov,
            "fractions": {
                "observed": float(obs.sum() / max(1, uv_valid.sum())),
                "inferred_symmetry": float(inferred.sum() / max(1, uv_valid.sum())),
                "synthesized_own_skin": float(synthesized.sum() / max(1, uv_valid.sum())),
                "filled_interpolated": float((need & ~has_m & ~synthesized).sum() / max(1, uv_valid.sum())),
            },
        }

    @staticmethod
    def save_maps(result: Dict[str, np.ndarray], output_dir: Union[str, Path], prefix: str = "head") -> Dict[str, str]:
        tex_dir = Path(output_dir) / "textures"
        tex_dir.mkdir(parents=True, exist_ok=True)
        p_alb = tex_dir / f"{prefix}_albedo_diffuse.png"
        p_prov = tex_dir / f"{prefix}_provenance.png"
        cv2.imwrite(str(p_alb), result["albedo_srgb"])
        cv2.imwrite(str(p_prov), result["provenance"])
        return {"albedo_diffuse_png": str(p_alb), "albedo": str(p_alb), "provenance_map": str(p_prov)}


def load_or_build_mirror_map(
    cache_dir: Union[str, Path],
    uv_raster: RasterOut,
    template_vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    res = uv_raster.mask.shape[0]
    path = Path(cache_dir) / f"flame_uv_mirror_{res}.npy"
    if path.exists():
        mm = np.load(path)
        if mm.shape[:2] == uv_raster.mask.shape:
            return mm
    mm = build_mirror_map(uv_raster, template_vertices, faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mm)
    return mm
