"""
Stage 1.5: Non-Linear Contour & Landmark Laplacian Mesh Deformer.

Deforms FLAME base mesh non-linearly to break through the 300-D linear PCA ceiling.
Aligns facial silhouette, mandibular jawline, chin prominence, cheekbones,
and brow overhang to true photographic landmarks under Graph Laplacian smoothness
regularization and strict Rule 4 collar pinning constraints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.stage0_preprocess.detector import FaceDetection
from src.stage1_5_residual.residual_net import build_laplacian_matrix
from src.stage6_texture.projector import estimate_camera_projection_matrix, extract_flame_5_landmarks
from src.utils.flame_model import FLAMEModel, N_VERTS

logger = logging.getLogger(__name__)


def load_flame_landmark_matrix(
    faces: np.ndarray,
    n_verts: int,
    embedding_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Constructs the linear mapping matrix M_lmk in R^{68 x N_verts} from FLAME's
    landmark barycentric embedding, such that L_3d = M_lmk @ V.
    """
    candidates = []
    if embedding_path:
        candidates.append(Path(embedding_path))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend([
        root / 'data' / 'flame_model' / 'landmark_embedding.npy',
        root / 'vendor' / 'MICA' / 'data' / 'FLAME2020' / 'landmark_embedding.npy',
        root / 'data' / 'FLAME2020' / 'landmark_embedding.npy',
    ])

    for cand in candidates:
        if cand.exists():
            try:
                emb = np.load(cand, allow_pickle=True, encoding='latin1')
                if hasattr(emb, 'item'):
                    emb = emb.item()
                elif isinstance(emb, np.ndarray) and emb.dtype == object:
                    emb = emb[()]

                lmk_faces = emb['full_lmk_faces_idx']
                if lmk_faces.ndim > 1:
                    lmk_faces = lmk_faces[0]
                lmk_bary = emb['full_lmk_bary_coords']
                if lmk_bary.ndim > 2:
                    lmk_bary = lmk_bary[0]

                M = np.zeros((68, n_verts), dtype=np.float32)
                for i in range(min(68, len(lmk_faces))):
                    f_idx = lmk_faces[i]
                    if f_idx < len(faces):
                        f = faces[f_idx]
                        for k in range(3):
                            if f[k] < n_verts:
                                M[i, f[k]] += float(lmk_bary[i, k])
                    if np.sum(M[i]) == 0 and n_verts > 0:
                        M[i, i % n_verts] = 1.0
                return M
            except Exception as e:
                logger.warning("Failed loading landmark embedding %s: %s", cand, e)

    raise FileNotFoundError(
        "Could not find FLAME landmark_embedding.npy in data/flame_model/ or vendor/MICA/."
    )


class NonLinearContourDeformer:
    """
    Solves for per-vertex non-linear displacement deltas ΔV to snap the FLAME mesh
    to the photographic 2D/3D landmarks while guaranteeing C1 organic smoothness.
    """

    def __init__(
        self,
        flame_model: FLAMEModel,
        lr: float = 2e-4,
        n_iterations: int = 80,
        lambda_laplacian: float = 600.0,
        lambda_reg: float = 60.0,
        max_displacement_mm: float = 18.0,
        device: Optional[str] = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for NonLinearContourDeformer.")

        self.flame = flame_model
        self.lr = lr
        self.n_iterations = n_iterations
        self.lambda_lap = lambda_laplacian
        self.lambda_reg = lambda_reg
        self.max_disp_m = max_displacement_mm / 1000.0  # convert mm to meters

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 1. Mesh topology and combinatorial Laplacian
        L_np = build_laplacian_matrix(self.flame.faces, N_VERTS)
        self.L = torch.from_numpy(L_np).float().to(self.device)

        # 2. Barycentric landmark projection matrix (68 x N_VERTS)
        M_np = load_flame_landmark_matrix(self.flame.faces, N_VERTS)
        self.M = torch.from_numpy(M_np).float().to(self.device)

        # 3. Collar pinning mask: lowest 20% along Y has ΔV ≡ 0.000000 mm
        v_temp = self.flame.v_template
        y_min, y_max = v_temp[:, 1].min(), v_temp[:, 1].max()
        y_norm = (v_temp[:, 1] - y_min) / (y_max - y_min + 1e-8)

        pinning_mask = np.ones((N_VERTS, 1), dtype=np.float32)
        pinning_mask[y_norm <= 0.20] = 0.0
        transition = (y_norm > 0.20) & (y_norm < 0.32)
        t = (y_norm[transition] - 0.20) / 0.12
        pinning_mask[transition, 0] = 3 * t**2 - 2 * t**3  # Hermite C1 ramp
        self.pinning_mask = torch.from_numpy(pinning_mask).float().to(self.device)
        self.y_norm = y_norm

    def deform(
        self,
        base_vertices: np.ndarray,
        detections: List[FaceDetection],
        image_shapes: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deform base FLAME vertices to minimize landmark & contour reprojection error.

        Parameters
        ----------
        base_vertices : (N_VERTS, 3) base neutral vertices from Stage 1/2.
        detections : list of FaceDetection objects with landmark_3d_68 or landmarks_5pt.
        image_shapes : list of (H, W) corresponding to detections.

        Returns
        -------
        deformed_vertices : (N_VERTS, 3) non-linearly deformed mesh vertices.
        delta_v : (N_VERTS, 3) per-vertex displacement vectors in meters.
        """
        valid_pairs = []
        for i, det in enumerate(detections):
            if det is None:
                continue
            # Need 68 landmarks (or fallback to 5pt)
            lm_target = getattr(det, 'landmark_3d_68', None)
            if lm_target is None:
                continue
            img_shape = image_shapes[i] if image_shapes and i < len(image_shapes) else (2048, 2048)
            valid_pairs.append((det, lm_target, img_shape))

        if not valid_pairs:
            logger.info("[Stage 1.5] No valid 68-point landmarks found. Returning base mesh unmodified.")
            return base_vertices.copy(), np.zeros_like(base_vertices)

        # Build camera projection matrices for valid views
        cam_projections = []
        target_landmarks_2d = []
        weights_list = []

        mesh_5_lmk = extract_flame_5_landmarks(base_vertices, self.flame.faces)

        for det, lm_68, shape in valid_pairs:
            P = estimate_camera_projection_matrix(
                landmarks_5=det.landmarks_5pt,
                yaw_deg=det.yaw_deg,
                pitch_deg=det.pitch_deg,
                roll_deg=det.roll_deg,
                image_shape=shape,
                mesh_3d_landmarks=mesh_5_lmk,
            )
            cam_projections.append(torch.from_numpy(P).float().to(self.device))
            target_landmarks_2d.append(torch.from_numpy(lm_68[:, :2]).float().to(self.device))

            # Feature weight profile across 68 landmarks:
            # 0..16: Mandibular jawline silhouette (weight 2.5x)
            # 8: Chin apex (weight 4.0x)
            # 17..26: Brow overhang (weight 1.8x)
            # 27..35: Nose bridge and tip (weight 2.0x)
            # 36..47: Eye corners (weight 2.0x)
            # 48..67: Mouth commissures (weight 1.5x)
            w = torch.ones(68, device=self.device)
            w[0:17] = 2.5
            w[8] = 4.0
            w[17:27] = 1.8
            w[27:36] = 2.0
            w[36:48] = 2.0
            w[48:68] = 1.5
            weights_list.append(w)

        # Optimization variable: delta_v initialized to zero
        v_base = torch.from_numpy(base_vertices).float().to(self.device)
        delta_v = torch.zeros_like(v_base, requires_grad=True)
        optimizer = torch.optim.Adam([delta_v], lr=self.lr)

        for _ in range(self.n_iterations):
            optimizer.zero_grad()

            # Enforce collar pinning and clamp displacement magnitude
            dv_pinned = delta_v * self.pinning_mask
            dv_clamped = torch.clamp(dv_pinned, -self.max_disp_m, self.max_disp_m)
            v_curr = v_base + dv_clamped

            # Compute 3D landmarks
            lm3d = self.M @ v_curr  # (68, 3)
            homo = torch.cat([lm3d, torch.ones((68, 1), device=self.device)], dim=-1)  # (68, 4)

            # Reprojection loss across all camera views
            total_lmk_loss = torch.tensor(0.0, device=self.device)
            for P_cam, target_2d, w in zip(cam_projections, target_landmarks_2d, weights_list):
                proj = homo @ P_cam.T  # (68, 3)
                z = proj[:, 2:3]
                z_safe = torch.where(z.abs() < 1e-4, torch.full_like(z, 1e-4), z)
                pred_2d = proj[:, :2] / z_safe

                diff_sq = (pred_2d - target_2d) ** 2
                loss_cam = (w.unsqueeze(-1) * diff_sq).mean()
                total_lmk_loss = total_lmk_loss + loss_cam

            total_lmk_loss = total_lmk_loss / len(cam_projections)

            # Smoothness: Laplacian regularization ||L · ΔV||²
            Lv = self.L @ dv_clamped
            loss_lap = torch.mean(Lv ** 2) * self.lambda_lap

            # Regularization: prevent drifting from base geometry
            loss_reg = torch.mean(dv_clamped ** 2) * self.lambda_reg

            total_loss = total_lmk_loss + loss_lap + loss_reg
            total_loss.backward()
            optimizer.step()

        # Final displacement calculation
        with torch.no_grad():
            dv_final = torch.clamp(delta_v * self.pinning_mask, -self.max_disp_m, self.max_disp_m)
            v_final = v_base + dv_final

            # Hard guarantee: bitwise zero on collar
            collar_indices = np.where(self.y_norm <= 0.20)[0]
            dv_np = dv_final.cpu().numpy()
            dv_np[collar_indices] = 0.0

            v_final_np = base_vertices + dv_np

            max_disp_mm = float(np.max(np.abs(dv_np)) * 1000.0)
            collar_disp_mm = float(np.max(np.abs(dv_np[collar_indices])) * 1000.0)
            logger.info(
                "[Stage 1.5] Non-linear deformation complete: max displacement = %.2f mm, collar displacement = %.6f mm",
                max_disp_mm, collar_disp_mm
            )

        return v_final_np, dv_np
