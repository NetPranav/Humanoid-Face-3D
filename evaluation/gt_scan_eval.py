"""
Ground-truth geometry evaluation against a real head scan (default: CC-BY Lee Perry-Smith).

Two measurements:
  reconstruction_error()  photo -> fitted head vs. the scan the photo was rendered from.
                          Face region only (FLAME 'face' mask minus eyes: the scan's eyes are
                          closed). Similarity ICP alignment (scan units are arbitrary), errors
                          reported in mm after normalizing the scan to FLAME-mean size.
  model_capacity()        how closely a FLAME variant can represent this real person at all:
                          β, ψ, jaw and a similarity transform fitted directly to the scan in 3D.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import trimesh

from src.utils.flame_model import FLAMEModel
from src.utils.flame_regions import load_vertex_masks
from src.utils.flame_torch import FLAMETorch

ROOT = Path(__file__).resolve().parents[1]
F_CV = np.diag([1.0, -1.0, -1.0])


def load_scan_camera_frame(scan_glb: Path, camera_json: Path) -> trimesh.Trimesh:
    """Scan vertices in the OpenCV camera frame of a render made by scripts/render_gt_scan.py."""
    m = trimesh.load(str(scan_glb)).to_geometry()
    v = np.asarray(m.vertices, np.float64)
    v_bl = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)                 # glTF Y-up -> Blender Z-up
    cam = json.load(open(camera_json))
    M = np.asarray(cam["world_to_camera_blender"])
    vc = (M[:3, :3] @ v_bl.T).T + M[:3, 3]
    return trimesh.Trimesh((F_CV @ vc.T).T, np.asarray(m.faces), process=False)


def scan_landmarks_3d(scan: trimesh.Trimesh, photo_bgr: np.ndarray, K: np.ndarray, landmarker=None):
    """MediaPipe landmarks on the render, ray-cast onto the scan -> (478,3) points (nan = miss)."""
    from src.stage0_preprocess.landmarks_mp import MediaPipeLandmarker
    lm = (landmarker or MediaPipeLandmarker()).detect(photo_bgr)
    if lm is None:
        raise RuntimeError("No face found in the ground-truth render.")
    uv1 = np.c_[lm.points[:, :2], np.ones(len(lm.points))]
    dirs = (np.linalg.inv(K) @ uv1.T).T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    locs, ray_idx, _ = scan.ray.intersects_location(np.zeros_like(dirs), dirs, multiple_hits=False)
    out = np.full((len(dirs), 3), np.nan)
    out[ray_idx] = locs
    return out, lm


def umeyama(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def embedding_points(vertices: np.ndarray, faces: np.ndarray, emb_path: str) -> np.ndarray:
    e = np.load(emb_path)
    return (vertices[faces[e["face_idx"]]] * e["bary"][..., None]).sum(1)


def stable_landmarks(emb_path: str) -> np.ndarray:
    from src.stage0_preprocess.landmarks_mp import FACE_OVAL
    e = np.load(emb_path)
    ok = np.nan_to_num(e["spread_mm"], posinf=99) < 3.0
    ok[468:] = False
    ok[FACE_OVAL] = False
    return ok


def face_eval_vertices(flame: FLAMEModel) -> np.ndarray:
    vm = load_vertex_masks()
    face = np.setdiff1d(vm["face"], np.concatenate([vm["eye_region"], vm["left_eyeball"], vm["right_eyeball"]]))
    return face


def _similarity_icp(src: np.ndarray, target: trimesh.Trimesh, iters: int = 40, init=None):
    """Aligns point set src to the target surface with scale; returns aligned points and scale.
    init: optional (s, R, t) similarity applied first (e.g. from landmark correspondences)."""
    s_tot = 1.0
    x = src.copy()
    if init is not None:
        s0, R0, t0 = init
        x = (s0 * (R0 @ x.T)).T + t0
        s_tot = s0
    for _ in range(iters):
        cp, d, _ = trimesh.proximity.closest_point(target, x)
        keep = d < np.percentile(d, 90)
        a, b = x[keep], cp[keep]
        mu_a, mu_b = a.mean(0), b.mean(0)
        A, B = a - mu_a, b - mu_b
        U, S, Vt = np.linalg.svd(B.T @ A / len(a))
        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[2, 2] = -1
        R = U @ D @ Vt
        s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
        x = (s * (R @ (x - mu_a).T)).T + mu_b
        s_tot *= s
    return x, s_tot


def reconstruction_error(recon_vertices: np.ndarray, flame: FLAMEModel, scan: trimesh.Trimesh,
                         scan_to_mm: float, scan_lmk3d: np.ndarray, emb_path: str) -> Dict[str, float]:
    """
    recon_vertices: fitted (posed) FLAME vertices in the camera frame (metres).
    scan_to_mm: mm per scan unit (from model_capacity: 1000 / scan_units_per_m).
    """
    idx = face_eval_vertices(flame)
    faces = flame.faces.astype(np.int64)
    ok = stable_landmarks(emb_path) & np.isfinite(scan_lmk3d).all(1)
    init = umeyama(embedding_points(recon_vertices, faces, emb_path)[ok], scan_lmk3d[ok])
    aligned, s = _similarity_icp(recon_vertices[idx], scan, init=init)
    _, d, _ = trimesh.proximity.closest_point(scan, aligned)
    d_mm = d * scan_to_mm
    return {"face_err_mean_mm": float(d_mm.mean()), "face_err_median_mm": float(np.median(d_mm)),
            "face_err_p90_mm": float(np.percentile(d_mm, 90)), "icp_scale": float(s)}


def model_capacity(flame: FLAMEModel, scan: trimesh.Trimesh, scan_lmk3d: np.ndarray, emb_path: str,
                   iters: int = 300, n_shape: int = 300) -> Dict[str, float]:
    """
    Fits β (n_shape), ψ (100), jaw and a similarity transform of FLAME directly to the scan
    surface (face region), alternating closest-point matching and gradient steps. Errors are
    reported in mm of FLAME's metric frame (the fitted similarity scale converts scan units).
    """
    import cv2
    from src.utils.flame_torch import batch_rodrigues

    layer = FLAMETorch(flame)
    idx = torch.as_tensor(face_eval_vertices(flame))
    faces = flame.faces.astype(np.int64)
    with torch.no_grad():
        vall = layer(torch.zeros(300)).numpy().astype(np.float64)
    v0 = vall[idx.numpy()]
    ok = stable_landmarks(emb_path) & np.isfinite(scan_lmk3d).all(1)
    init = umeyama(embedding_points(vall, faces, emb_path)[ok], scan_lmk3d[ok])
    x0, _ = _similarity_icp(v0, scan, init=init)
    mu_a, mu_b = v0.mean(0), x0.mean(0)
    A, B = v0 - mu_a, x0 - mu_b
    U, S, Vt = np.linalg.svd(B.T @ A / len(A))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R0 = U @ D @ Vt
    sc0 = float(np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean())
    aa = torch.tensor(cv2.Rodrigues(R0)[0].reshape(3), dtype=torch.float32, requires_grad=True)
    ls = torch.tensor([np.log(sc0)], dtype=torch.float32, requires_grad=True)
    t = torch.tensor(mu_b - sc0 * R0 @ mu_a, dtype=torch.float32, requires_grad=True)
    beta = torch.zeros(n_shape, requires_grad=True)
    psi = torch.zeros(100, requires_grad=True)
    jaw = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([beta, psi, jaw, aa, ls, t], lr=0.01)

    def verts():
        pose = torch.cat([torch.zeros(6), jaw, torch.zeros(6)])
        fb = torch.cat([beta, torch.zeros(300 - n_shape)]) if n_shape < 300 else beta
        return torch.exp(ls) * (layer(fb, psi, pose)[idx] @ batch_rodrigues(aa).T) + t

    for it in range(iters):
        x = verts()
        if it % 10 == 0:
            cp, d, _ = trimesh.proximity.closest_point(scan, x.detach().numpy())
            tgt = torch.as_tensor(cp, dtype=torch.float32)
            w = torch.as_tensor((d < np.percentile(d, 95)).astype(np.float32))
        loss = ((x - tgt).pow(2).sum(1) * w).sum() / w.sum() / torch.exp(ls) ** 2 \
            + 1e-6 * (beta.pow(2).sum() + psi.pow(2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = verts().numpy()
        scale = float(torch.exp(ls))                 # scan units per metre of FLAME
    _, d, _ = trimesh.proximity.closest_point(scan, x)
    d_mm = d / scale * 1000.0
    return {"capacity_mean_mm": float(d_mm.mean()), "capacity_median_mm": float(np.median(d_mm)),
            "capacity_p90_mm": float(np.percentile(d_mm, 90)), "scan_units_per_m": scale,
            "beta_norm": float(beta.detach().norm())}
