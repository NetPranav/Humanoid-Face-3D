"""
Phase 0 evaluation harness (DOCS/04 §9).

Per run it writes
  eval/overlay_view{i}.jpg   photo | fitted geometry blended over photo | textured render over photo
                             | landmarks (green = detected, red = FLAME reprojection)
  eval/turntable.jpg         textured + shaded geometry at yaw 0/±30/60/90 (neutral mesh)
  report.json                metrics

Identity judge: InsightFace buffalo_l recognition (w600k_r50). Since the Phase 0 fix, no
loss or regressor in the pipeline consumes this network (MICA uses its own ArcFace), so it
is a held-out judge. Scores on *textured* renders mostly measure texture registration and
geometry at novel yaw; they are comparative (v1 vs v3), not absolute likeness scores.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.render.soft_renderer import render, turntable_camera

TURNTABLE_YAWS = (0, 30, -30, 60, 90)
JUDGE_YAWS = (0, 30, -30)
BG = 128


def _crop_square(img: np.ndarray, bbox: np.ndarray, scale: float = 1.15, out: int = 512) -> np.ndarray:
    x0, y0, x1, y1 = bbox.astype(float)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = scale * max(x1 - x0, y1 - y0)
    M = np.array([[out / (2 * r), 0, out / 2 - cx * out / (2 * r)], [0, out / (2 * r), out / 2 - cy * out / (2 * r)]])
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA, borderValue=(BG, BG, BG))


class IdentityJudge:
    """cos similarity between the photo's face and a rendered face, via buffalo_l recognition."""

    def __init__(self, face_app):
        self.app = face_app

    def embed(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        f = max(faces, key=lambda f: f.det_score)
        return f.normed_embedding

    def score(self, ref_emb: np.ndarray, img_bgr: np.ndarray) -> Optional[float]:
        e = self.embed(img_bgr)
        return None if e is None else float(np.dot(ref_emb, e))


def render_turntable(verts_flame, faces, uv, uv_faces, texture, yaws=TURNTABLE_YAWS, size=512, textured=True,
                     sh_lum=None):
    out = []
    for y in yaws:
        vc, K = turntable_camera(verts_flame, y, (size, size))
        img, m = render(vc, faces, K, (size, size), uv if textured else None,
                        uv_faces if textured else None, texture if textured else None, sh_lum=sh_lum)
        img[~m] = BG
        out.append(img)
    return out


def judge_turntable(judge: IdentityJudge, ref_emb, verts_flame, faces, uv, uv_faces, texture,
                    sh_lum=None, prefix: str = "id") -> Dict[str, Optional[float]]:
    imgs = render_turntable(verts_flame, faces, uv, uv_faces, texture, yaws=JUDGE_YAWS, sh_lum=sh_lum)
    return {f"{prefix}_yaw{y:+d}": judge.score(ref_emb, im) for y, im in zip(JUDGE_YAWS, imgs)}


def evaluate_run(
    out_dir: Path,
    photos: Sequence[np.ndarray],
    detections: Sequence,
    view_fits: Sequence,
    neutral_vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    albedo: np.ndarray,
    judge: Optional[IdentityJudge],
    extra: Optional[dict] = None,
    sh_lum: Optional[np.ndarray] = None,
) -> dict:
    """
    sh_lum: first-order SH luminance lighting fitted from the photo (view 0, camera frame). When
    the albedo is delit, 'id_relit_*' scores re-light it with that lighting: a delit texture is
    meant to be lit by a renderer, so this measures whether albedo + lighting still reproduce
    the person, separately from 'id_*' (albedo under the neutral evaluation light).
    """
    out_dir = Path(out_dir)
    ev = out_dir / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    faces = faces.astype(np.int64)
    uv_faces = uv_faces.astype(np.int64)
    report: Dict[str, object] = {"views": []}

    for i, (im, det, vf) in enumerate(zip(photos, detections, view_fits)):
        H, W = im.shape[:2]
        tex_img, tm = render(vf.vertices_cam, faces, vf.intrinsics.K, (H, W), uv, uv_faces, albedo)
        geo_img, gm = render(vf.vertices_cam, faces, vf.intrinsics.K, (H, W))
        over = im.copy()
        over[tm] = tex_img[tm]
        blend = im.copy()
        blend[gm] = (0.45 * im[gm] + 0.55 * geo_img[gm]).astype(np.uint8)
        lm = im.copy()
        rad = max(2, int(vf.metrics["iod_px"] / 60))
        for p in vf.lmk_target:
            cv2.circle(lm, tuple(int(v) for v in p), rad, (0, 255, 0), -1)
        for p in vf.lmk_pred:
            cv2.circle(lm, tuple(int(v) for v in p), rad, (0, 0, 255), -1)
        # render alone on grey at the input camera, for the judge
        solo = np.full_like(im, BG)
        solo[tm] = tex_img[tm]
        panels = [_crop_square(x, det.bbox) for x in (im, blend, over, lm)]
        cv2.imwrite(str(ev / f"overlay_view{i}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
        vrep = dict(vf.metrics)
        vrep["focal_source"] = vf.intrinsics.source
        if judge is not None and det.embedding is not None:
            vrep["id_input_view"] = judge.score(det.embedding, _crop_square(solo, det.bbox, 1.4))
            if sh_lum is not None:
                rl, rm = render(vf.vertices_cam, faces, vf.intrinsics.K, (H, W), uv, uv_faces, albedo, sh_lum=sh_lum)
                solo_r = np.full_like(im, BG)
                solo_r[rm] = rl[rm]
                vrep["id_relit_input_view"] = judge.score(det.embedding, _crop_square(solo_r, det.bbox, 1.4))
        report["views"].append(vrep)

    tt_tex = render_turntable(neutral_vertices, faces, uv, uv_faces, albedo)
    tt_geo = render_turntable(neutral_vertices, faces, uv, uv_faces, None, textured=False)
    cv2.imwrite(str(ev / "turntable.jpg"), np.vstack([np.hstack(tt_tex), np.hstack(tt_geo)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    if judge is not None:
        best = max(range(len(detections)), key=lambda k: detections[k].det_score)
        ref = detections[best].embedding
        report.update(judge_turntable(judge, ref, neutral_vertices, faces, uv, uv_faces, albedo))
        if sh_lum is not None:
            report.update(judge_turntable(judge, ref, neutral_vertices, faces, uv, uv_faces, albedo,
                                          sh_lum=sh_lum, prefix="id_relit"))
    report["lmk_err_iod_mean"] = float(np.mean([v["lmk_err_iod_mean"] for v in report["views"]]))
    if extra:
        report.update(extra)
    with open(out_dir / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return report
