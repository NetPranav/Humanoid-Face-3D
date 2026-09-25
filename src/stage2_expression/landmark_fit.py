"""
Stage 2 (Phase 0): per-view camera + head pose + expression + jaw fit to 68 landmarks.

Replaces two v1 components (DOCS/01 R2, R3):
  * the dead SMIRK wrapper that silently returned ψ = θ = 0, and
  * 5-point solvePnP with f = image width.

Given MICA's β (held fixed in Phase 0), this solves for
    R, t (head in camera), focal scale s_f (EXIF prior, bounded [0.5, 2]),
    ψ[:n_expr], jaw rotation
by minimizing IOD-normalized reprojection error of FLAME's 68-landmark embedding against
InsightFace's 68 landmarks, under a full pinhole camera. The result gives the posed and
expressed mesh that Stage 6 projects the photo onto, while export stays neutral.

Frames: FLAME (+y up, +z out of face) -> OpenCV camera (+y down, +z forward) via
X_cam = F · R · X + t, with F = diag(1, -1, -1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.stage0_preprocess.camera import Intrinsics
from src.utils.flame_model import FLAMEModel
from src.utils.flame_torch import FLAMETorch, batch_rodrigues

F_FLIP = np.diag([1.0, -1.0, -1.0])

# Landmark groups (iBUG-68 ordering, shared by FLAME's embedding and InsightFace 1k3d68)
JAW = slice(0, 17)
INNER = slice(17, 68)


def load_landmark_embedding(path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path else root / "data" / "flame_model" / "landmark_embedding_mica.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Create it with: python3 scripts/extract_flame_from_mica.py"
        )
    return dict(np.load(p))


@dataclass
class ViewFit:
    R: np.ndarray                  # (3,3) FLAME -> camera rotation (includes the axis flip)
    t: np.ndarray                  # (3,) metres
    intrinsics: Intrinsics
    psi: np.ndarray                # (100,)
    jaw: np.ndarray                # (3,) axis-angle
    vertices_cam: np.ndarray       # (V,3) posed + expressed mesh in camera coordinates
    vertices_flame: np.ndarray     # (V,3) expressed mesh in FLAME frame (before R, t)
    lmk_pred: np.ndarray           # (68,2) px
    lmk_target: np.ndarray         # (68,2) px
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def pose_theta(self) -> np.ndarray:
        th = np.zeros(15, np.float32)
        th[6:9] = self.jaw
        return th

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        K = self.intrinsics.K
        z = np.clip(pts_cam[:, 2:3], 1e-6, None)
        return (pts_cam[:, :2] / z) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]

    def to_dict(self) -> dict:
        return {
            "R": self.R.tolist(), "t": self.t.tolist(), "intrinsics": self.intrinsics.to_dict(),
            "psi": self.psi.tolist(), "jaw": self.jaw.tolist(), "metrics": self.metrics,
        }


class LandmarkFitter:
    def __init__(
        self,
        flame: FLAMEModel,
        n_expr: int = 50,
        iters: int = 400,
        lr: float = 0.01,
        w_jaw_contour: float = 0.5,
        lambda_expr: float = 2e-3,
        lambda_jaw: float = 1e-1,
        lambda_focal: float = 1e-2,
        fit_focal: bool = True,
        embedding: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.flame = flame
        self.layer = FLAMETorch(flame)
        self.n_expr = n_expr
        self.iters = iters
        self.lr = lr
        self.w_jaw = w_jaw_contour
        self.lambda_expr = lambda_expr
        self.lambda_jaw = lambda_jaw
        self.lambda_focal = lambda_focal
        self.fit_focal = fit_focal
        emb = embedding or load_landmark_embedding()
        faces = flame.faces
        idx = emb["full_lmk_faces_idx"].astype(np.int64).reshape(-1)
        bary = emb["full_lmk_bary_coords"].reshape(-1, 3)
        self.lmk_vidx = torch.as_tensor(faces[idx].astype(np.int64))          # (68,3)
        self.lmk_bary = torch.as_tensor(bary, dtype=torch.float32)            # (68,3)

    # ------------------------------------------------------------------
    def landmarks_3d(self, verts: torch.Tensor) -> torch.Tensor:
        return (verts[self.lmk_vidx] * self.lmk_bary[..., None]).sum(1)

    def _init_pnp(self, lmk3d_flame: np.ndarray, lmk2d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = (F_FLIP @ lmk3d_flame.T).T.astype(np.float64)
        sel = np.r_[17:68]  # inner landmarks: no jaw silhouette ambiguity
        ok, rvec, tvec = cv2.solvePnP(pts[sel], lmk2d[sel].astype(np.float64), K, np.zeros(4),
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise RuntimeError("solvePnP failed to initialize the head pose.")
        R_cv, _ = cv2.Rodrigues(rvec)
        # X_cam = R_cv F X + t  ->  write as F R X + t with R = F R_cv F
        R = F_FLIP @ R_cv @ F_FLIP
        return R, tvec.reshape(3)

    def fit(
        self,
        beta: np.ndarray,
        lmk68_px: np.ndarray,
        intrinsics: Intrinsics,
    ) -> ViewFit:
        lmk2d = np.asarray(lmk68_px, np.float64)[:, :2]
        target = torch.as_tensor(lmk2d, dtype=torch.float32)
        iod = float(np.linalg.norm(lmk2d[36:42].mean(0) - lmk2d[42:48].mean(0)))
        if iod < 5:
            raise ValueError(f"Inter-ocular distance {iod:.1f}px is too small for a reliable fit.")

        beta_t = torch.as_tensor(beta, dtype=torch.float32)
        with torch.no_grad():
            v0 = self.layer(beta_t)
            l3d0 = self.landmarks_3d(v0).numpy()
        R0, t0 = self._init_pnp(l3d0, lmk2d, intrinsics.K)
        aa0, _ = cv2.Rodrigues(R0)

        aa = torch.tensor(aa0.reshape(3), dtype=torch.float32, requires_grad=True)
        t = torch.tensor(t0, dtype=torch.float32, requires_grad=True)
        log_sf = torch.zeros(1, requires_grad=self.fit_focal)
        psi = torch.zeros(self.n_expr, requires_grad=True)
        jaw = torch.zeros(3, requires_grad=True)
        Fm = torch.as_tensor(F_FLIP, dtype=torch.float32)
        fx0, fy0, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

        w = torch.ones(68)
        w[JAW] = self.w_jaw
        params = [aa, t, psi, jaw] + ([log_sf] if self.fit_focal else [])
        opt = torch.optim.Adam(params, lr=self.lr)

        def forward():
            pose = torch.zeros(15)
            pose = torch.cat([pose[:6], jaw, pose[9:]])
            v = self.layer(beta_t, psi, pose)
            R = Fm @ batch_rodrigues(aa)
            sf = torch.exp(torch.clamp(log_sf, -0.69, 0.69))
            # keep the subject's apparent size fixed while focal changes: scale depth with f
            t_eff = torch.cat([t[:2], t[2:] * sf])
            vc = v @ R.T + t_eff
            l3 = self.landmarks_3d(vc)
            z = torch.clamp(l3[:, 2:3], min=1e-4)
            proj = torch.cat([l3[:, :1] / z * fx0 * sf + cx, l3[:, 1:2] / z * fy0 * sf + cy], 1)
            return v, vc, proj, sf

        # translation is in metres while landmarks are pixels: scale the lr for t by depth
        for i in range(self.iters):
            opt.zero_grad()
            _, _, proj, sf = forward()
            err = ((proj - target) / iod).pow(2).sum(1)
            loss = (w * err).sum() / w.sum()
            loss = loss + self.lambda_expr * psi.pow(2).sum() + self.lambda_jaw * jaw.pow(2).sum()
            if self.fit_focal:
                loss = loss + self.lambda_focal * log_sf.pow(2).sum()
            loss.backward()
            t.grad[2] *= float(t0[2]) ** 2
            t.grad[:2] *= float(t0[2]) ** 2 * 0.01
            opt.step()
            with torch.no_grad():
                jaw[1:].clamp_(-0.05, 0.05)       # jaw mostly opens about x
                jaw[0].clamp_(-0.05, 0.5)

        with torch.no_grad():
            v, vc, proj, sf = forward()
            R = (Fm @ batch_rodrigues(aa)).numpy()
            t_eff = vc.mean(0) - (v @ torch.as_tensor(R).T).mean(0)
            per = (np.linalg.norm(proj.numpy() - lmk2d, axis=1) / iod)
            psi_full = np.zeros(100, np.float32)
            psi_full[: self.n_expr] = psi.numpy()
            K_fit = Intrinsics(fx0 * float(sf), fy0 * float(sf), cx, cy,
                               intrinsics.source + ("+fit" if self.fit_focal else ""))
            metrics = {
                "lmk_err_iod_mean": float(per.mean()),
                "lmk_err_iod_inner": float(per[INNER].mean()),
                "lmk_err_iod_jaw": float(per[JAW].mean()),
                "iod_px": iod,
                "focal_scale": float(sf),
                "subject_distance_m": float(t_eff[2]),
                "yaw_deg": float(np.degrees(np.arctan2(-R[2, 0], R[2, 2]))),
            }
            return ViewFit(
                R=R, t=t_eff.numpy(), intrinsics=K_fit, psi=psi_full, jaw=jaw.numpy().copy(),
                vertices_cam=vc.numpy(), vertices_flame=v.numpy(), lmk_pred=proj.numpy(),
                lmk_target=lmk2d.astype(np.float32), metrics=metrics,
            )
