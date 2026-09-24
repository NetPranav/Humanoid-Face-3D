"""
Stage 6: Multi-View UV Texture Projection Engine.

Projects input photographs onto the 3D FLAME mesh UV parameterisation using
per-view weak-perspective camera matrices, angle-weighted cosine blending,
and z-buffer-based visibility testing.

All operations are pure NumPy/OpenCV — no GPU required (~200 MB RAM, ~2.5s).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera projection estimation
# ---------------------------------------------------------------------------

CANONICAL_5_LANDMARKS = np.array([
    [-0.0311,  0.0234,  0.0352],  # Right eye (InsightFace kps[0], viewer left)
    [ 0.0320,  0.0225,  0.0348],  # Left eye  (InsightFace kps[1], viewer right)
    [ 0.0005, -0.0056,  0.0732],  # Nose tip  (InsightFace kps[2])
    [-0.0245, -0.0443,  0.0473],  # Right mouth corner (InsightFace kps[3])
    [ 0.0238, -0.0443,  0.0473],  # Left mouth corner  (InsightFace kps[4])
], dtype=np.float64)


def extract_flame_5_landmarks(
    vertices: np.ndarray,
    faces: np.ndarray,
    embedding_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Extracts the 5 canonical facial landmark 3D positions directly from the
    subject's reconstructed 3D head mesh using the official FLAME landmark embedding.
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

                if np.max(lmk_faces) < len(faces):
                    lmk_3d_68 = np.zeros((68, 3), dtype=np.float64)
                    for i in range(68):
                        f_idx = lmk_faces[i]
                        f = faces[f_idx]
                        lmk_3d_68[i] = (
                            lmk_bary[i, 0] * vertices[f[0]] +
                            lmk_bary[i, 1] * vertices[f[1]] +
                            lmk_bary[i, 2] * vertices[f[2]]
                        )
                    r_eye = np.mean(lmk_3d_68[36:42], axis=0)
                    l_eye = np.mean(lmk_3d_68[42:48], axis=0)
                    nose = lmk_3d_68[30]
                    r_mouth = lmk_3d_68[48]
                    l_mouth = lmk_3d_68[54]
                    return np.array([r_eye, l_eye, nose, r_mouth, l_mouth], dtype=np.float64)
            except Exception:
                pass

    return CANONICAL_5_LANDMARKS.copy()


def estimate_camera_projection_matrix(
    landmarks_5: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    image_shape: Tuple[int, int],
    mesh_3d_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Converts Stage 0's 5-point landmarks + Euler angles into a weak-perspective
    3×4 projection matrix P suitable for projecting FLAME 3D vertices (in metres)
    into 2D image pixel coordinates.

    Parameters
    ----------
    landmarks_5 : (5, 2)  — left eye, right eye, nose tip, left mouth, right mouth
    yaw_deg, pitch_deg, roll_deg : Head pose Euler angles (degrees)
    image_shape : (H, W) of the source photograph
    mesh_3d_landmarks : (5, 3) optional subject-specific 3D landmark coordinates

    Returns
    -------
    P : (3, 4) projection matrix   x_px = P @ [X, Y, Z, 1]^T
    """
    h, w = image_shape[:2]

    # --- Canonical or Mesh-Derived 3D FLAME landmark positions (metres) ---
    if mesh_3d_landmarks is not None and len(mesh_3d_landmarks) == 5:
        canonical_3d = mesh_3d_landmarks.astype(np.float64)
    else:
        canonical_3d = CANONICAL_5_LANDMARKS.copy()

    # --- Build rotation from Euler angles ---
    yaw   = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll  = np.radians(roll_deg)

    Ry = np.array([
        [ np.cos(yaw),  0, np.sin(yaw)],
        [ 0,            1, 0           ],
        [-np.sin(yaw),  0, np.cos(yaw)],
    ])
    Rx = np.array([
        [1, 0,             0            ],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0,             0,            1],
    ])
    R = Rz @ Rx @ Ry   # extrinsic rotation (yaw-pitch-roll order)

    # --- Solve weak-perspective via PnP (DLT) ---
    # Use solvePnP for a stable solution
    lm_2d = landmarks_5.astype(np.float64).reshape(5, 1, 2)
    cam_matrix = np.array([
        [w, 0, w / 2.0],
        [0, w, h / 2.0],
        [0, 0, 1.0    ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    success = False
    try:
        success, rvec, tvec = cv2.solvePnP(
            canonical_3d, lm_2d, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                canonical_3d, lm_2d, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
    except Exception:
        success = False

    if not success:
        # Fallback to weak-perspective from 2D/3D correspondence
        return _fallback_weak_perspective(canonical_3d, landmarks_5, image_shape)

    R_pnp, _ = cv2.Rodrigues(rvec)
    T_pnp = tvec.reshape(3, 1)

    # Build full 3×4 projection: P = K @ [R | t]
    Rt = np.hstack([R_pnp, T_pnp])  # (3, 4)
    P = cam_matrix @ Rt
    return P.astype(np.float64)


def _fallback_weak_perspective(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Least-squares weak-perspective 3×4 matrix from 2D-3D pairs."""
    n = len(pts_3d)
    A = np.zeros((2 * n, 8), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        X, Y, Z = pts_3d[i]
        u, v = pts_2d[i]
        A[2 * i]     = [X, Y, Z, 1, 0, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1]
        b[2 * i]     = u
        b[2 * i + 1] = v

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    P = np.zeros((3, 4), dtype=np.float64)
    P[0, :] = result[:4]
    P[1, :] = result[4:]
    P[2, :] = [0, 0, 0, 1]
    return P


# ---------------------------------------------------------------------------
# Z-buffer visibility test
# ---------------------------------------------------------------------------

def _compute_visibility_mask(
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    P: np.ndarray,
    image_shape: Tuple[int, int],
    epsilon: float = 0.005,
) -> np.ndarray:
    """
    Returns a boolean mask (N_vertices,) indicating which vertices are visible
    from the camera defined by projection matrix P (i.e. not occluded).
    """
    h, w = image_shape[:2]
    n_verts = len(vertices_3d)

    # Project vertices to 2D using explicit coordinate dot products to prevent BLAS warnings
    proj_x = vertices_3d[:, 0] * P[0, 0] + vertices_3d[:, 1] * P[0, 1] + vertices_3d[:, 2] * P[0, 2] + P[0, 3]
    proj_y = vertices_3d[:, 0] * P[1, 0] + vertices_3d[:, 1] * P[1, 1] + vertices_3d[:, 2] * P[1, 2] + P[1, 3]
    proj_z = vertices_3d[:, 0] * P[2, 0] + vertices_3d[:, 1] * P[2, 1] + vertices_3d[:, 2] * P[2, 2] + P[2, 3]

    # Handle perspective division
    z = proj_z.copy()
    z[z == 0] = 1e-8
    px = (proj_x / z).astype(np.float32)
    py = (proj_y / z).astype(np.float32)

    # Clip to image bounds
    px_int = np.clip(np.round(px).astype(np.int32), 0, w - 1)
    py_int = np.clip(np.round(py).astype(np.int32), 0, h - 1)

    # Build depth buffer
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
    for i in range(n_verts):
        xi, yi = px_int[i], py_int[i]
        if z[i] < depth_buffer[yi, xi]:
            depth_buffer[yi, xi] = z[i]

    # Mark visible vertices
    visible = np.zeros(n_verts, dtype=bool)
    for i in range(n_verts):
        xi, yi = px_int[i], py_int[i]
        if abs(z[i] - depth_buffer[yi, xi]) < epsilon:
            visible[i] = True

    return visible


# ---------------------------------------------------------------------------
# Multi-view texture projector
# ---------------------------------------------------------------------------

class MultiViewTextureProjector:
    """
    Projects multiple input photographs onto the 3D FLAME mesh UV space
    using per-view projection matrices, visibility masks, and angle-weighted
    cosine blending.

    Parameters
    ----------
    texture_resolution : int
        Output texture map resolution (e.g. 2048 for 2048×2048).
    blend_gamma : float
        Cosine falloff exponent for view blending (higher = sharper selection).
    visibility_epsilon : float
        Z-buffer depth epsilon for occlusion testing.
    """

    def __init__(
        self,
        texture_resolution: int = 2048,
        blend_gamma: float = 2.0,
        visibility_epsilon: float = 0.005,
    ):
        self.resolution = texture_resolution
        self.gamma = blend_gamma
        self.vis_eps = visibility_epsilon

    def project(
        self,
        photos: List[np.ndarray],
        detections: list,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_normals: np.ndarray,
        uv_coords: np.ndarray,
        uv_faces: np.ndarray,
        flame_faces: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Project all views onto the UV texture map.

        Parameters
        ----------
        photos : list of (H, W, 3) uint8 BGR images
        detections : list of FaceDetection objects (with .landmarks_5, .yaw, .pitch, .roll)
        vertices : (V, 3) mesh vertices in metres
        faces : (F, 3) triangle face indices
        vertex_normals : (V, 3) unit vertex normals
        uv_coords : (UV, 2) UV texture coordinates in [0, 1]
        uv_faces : (TF, 3) UV face indices
        flame_faces : (F, 3) geometry face indices (may differ from uv_faces)

        Returns
        -------
        dict with keys:
            'projected_rgb'   : (R, R, 3) float32 [0, 255] — blended texture
            'projection_mask' : (R, R) uint8 — 255 where data, 0 where unseen
            'weight_map'      : (R, R) float32 — total accumulated weight per texel
        """
        R = self.resolution
        accum_rgb = np.zeros((R, R, 3), dtype=np.float64)
        accum_weight = np.zeros((R, R), dtype=np.float64)

        if len(photos) == 0:
            return {
                'projected_rgb': np.zeros((R, R, 3), dtype=np.float32),
                'projection_mask': np.zeros((R, R), dtype=np.uint8),
                'weight_map': np.zeros((R, R), dtype=np.float32),
            }

        if flame_faces is None:
            flame_faces = faces

        # Precompute per-texel 3D positions and surface normals across the UV map
        from src.stage3_detail.rasterizer import rasterize_uv_maps
        _, pos_map, norm_enc, valid_mask_tex = rasterize_uv_maps(
            flame_verts_m=vertices,
            flame_normals=vertex_normals,
            uv_coords=uv_coords,
            uv_faces=uv_faces,
            flame_faces=flame_faces,
            resolution=R,
        )

        valid_indices = np.where(valid_mask_tex > 0)
        y_valid, x_valid = valid_indices[0], valid_indices[1]

        if len(y_valid) == 0:
            return {
                'projected_rgb': np.zeros((R, R, 3), dtype=np.float32),
                'projection_mask': np.zeros((R, R), dtype=np.uint8),
                'weight_map': np.zeros((R, R), dtype=np.float32),
            }

        pts_3d = pos_map[y_valid, x_valid]  # (K, 3)
        surf_normals = norm_enc[y_valid, x_valid] * 2.0 - 1.0  # Decoded from [0, 1]
        n_len = np.linalg.norm(surf_normals, axis=1, keepdims=True) + 1e-8
        surf_normals = surf_normals / n_len

        # Extract subject-specific 3D landmarks from reconstructed mesh
        mesh_5_lmk = extract_flame_5_landmarks(vertices, flame_faces)

        for idx, (photo, det) in enumerate(zip(photos, detections)):
            if det is None or photo is None:
                continue

            h_img, w_img = photo.shape[:2]

            # Extract pose info from detection
            landmarks_5 = getattr(det, 'landmarks_5', None)
            if landmarks_5 is None:
                landmarks_5 = getattr(det, 'landmarks_5pt', None)
            if landmarks_5 is None:
                landmarks_5 = getattr(det, 'kps', None)
            if landmarks_5 is None:
                continue

            yaw = getattr(det, 'yaw', None)
            if yaw is None:
                yaw = getattr(det, 'yaw_deg', 0.0)
            pitch = getattr(det, 'pitch', None)
            if pitch is None:
                pitch = getattr(det, 'pitch_deg', 0.0)
            roll = getattr(det, 'roll', None)
            if roll is None:
                roll = getattr(det, 'roll_deg', 0.0)

            # Build projection matrix for this view with exact 3D landmarks
            P = estimate_camera_projection_matrix(
                landmarks_5, yaw, pitch, roll, (h_img, w_img),
                mesh_3d_landmarks=mesh_5_lmk
            )

            # Eye & iris local contrast enhancement on input photo
            proc_photo = photo.copy()
            if landmarks_5 is not None and len(landmarks_5) >= 2:
                try:
                    r_eye_px, l_eye_px = landmarks_5[0], landmarks_5[1]
                    iod = float(np.linalg.norm(r_eye_px - l_eye_px))
                    eye_radius = int(max(10, iod * 0.22))
                    for eye_pt in (r_eye_px, l_eye_px):
                        ex, ey = int(round(eye_pt[0])), int(round(eye_pt[1]))
                        x0_c = max(0, ex - eye_radius)
                        x1_c = min(w_img, ex + eye_radius)
                        y0_c = max(0, ey - eye_radius)
                        y1_c = min(h_img, ey + eye_radius)
                        if x1_c > x0_c + 4 and y1_c > y0_c + 4:
                            crop = proc_photo[y0_c:y1_c, x0_c:x1_c]
                            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                            l_ch, a_ch, b_ch = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                            l_clahe = clahe.apply(l_ch)
                            blurred = cv2.GaussianBlur(l_clahe, (0, 0), 1.5)
                            l_sharp = cv2.addWeighted(l_clahe, 1.4, blurred, -0.4, 0)
                            lab_sharp = cv2.merge([l_sharp, a_ch, b_ch])
                            enh_bgr = cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2BGR)
                            yy, xx = np.ogrid[y0_c:y1_c, x0_c:x1_c]
                            dist = np.sqrt((xx - ex)**2 + (yy - ey)**2) / eye_radius
                            alpha = np.clip(1.0 - dist, 0.0, 1.0)[:, :, None]
                            proc_photo[y0_c:y1_c, x0_c:x1_c] = np.clip(
                                proc_photo[y0_c:y1_c, x0_c:x1_c].astype(np.float32) * (1.0 - alpha) +
                                enh_bgr.astype(np.float32) * alpha, 0, 255
                            ).astype(np.uint8)
                except Exception:
                    proc_photo = photo

            # Camera position from extrinsic parameters
            R_mat = P[:3, :3]
            t_vec = P[:3, 3]
            try:
                cam_pos = -np.linalg.inv(R_mat) @ t_vec
            except np.linalg.LinAlgError:
                cam_pos = np.array([0, 0, 1.0])

            # Direct per-texel 3D -> 2D projection (explicit coordinates, no BLAS issues)
            proj_x = pts_3d[:, 0] * P[0, 0] + pts_3d[:, 1] * P[0, 1] + pts_3d[:, 2] * P[0, 2] + P[0, 3]
            proj_y = pts_3d[:, 0] * P[1, 0] + pts_3d[:, 1] * P[1, 1] + pts_3d[:, 2] * P[1, 2] + P[1, 3]
            proj_z = pts_3d[:, 0] * P[2, 0] + pts_3d[:, 1] * P[2, 1] + pts_3d[:, 2] * P[2, 2] + P[2, 3]
            proj_z_safe = np.where(np.abs(proj_z) > 1e-6, proj_z, 1e-6)

            px = (proj_x / proj_z_safe).astype(np.float32)
            py = (proj_y / proj_z_safe).astype(np.float32)

            # View direction & cosine surface angle
            view_dirs = cam_pos[np.newaxis, :] - pts_3d  # (K, 3)
            view_lens = np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8
            view_dirs_unit = view_dirs / view_lens
            cos_angles = np.sum(surf_normals * view_dirs_unit, axis=1)

            # Exclude collar clothing region (lowest 16% along Y is clothing/collar)
            y_min_v, y_max_v = vertices[:, 1].min(), vertices[:, 1].max()
            y_norm_pts = (pts_3d[:, 1] - y_min_v) / (y_max_v - y_min_v + 1e-8)

            # In-bounds and front-facing condition (cos_angles > 0.05 eliminates backfaces)
            valid_sample = (
                (cos_angles > 0.05) &
                (px >= 0) & (px < w_img - 1) &
                (py >= 0) & (py < h_img - 1) &
                (y_norm_pts > 0.16)
            )

            if not np.any(valid_sample):
                continue

            sub_idx = np.where(valid_sample)[0]
            sub_px = px[sub_idx]
            sub_py = py[sub_idx]
            sub_cos = cos_angles[sub_idx]

            # Vectorized bilinear sampling from photo
            x0 = np.floor(sub_px).astype(np.int32)
            y0 = np.floor(sub_py).astype(np.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            dx = (sub_px - x0)[:, None]
            dy = (sub_py - y0)[:, None]

            c00 = proc_photo[y0, x0].astype(np.float64)
            c01 = proc_photo[y0, x1].astype(np.float64)
            c10 = proc_photo[y1, x0].astype(np.float64)
            c11 = proc_photo[y1, x1].astype(np.float64)

            sampled_color = (
                (1.0 - dx) * (1.0 - dy) * c00 +
                dx * (1.0 - dy) * c01 +
                (1.0 - dx) * dy * c10 +
                dx * dy * c11
            )

            det_score = float(getattr(det, 'det_score', 1.0))
            sample_weights = (sub_cos ** self.gamma) * det_score

            tex_y = y_valid[sub_idx]
            tex_x = x_valid[sub_idx]

            accum_rgb[tex_y, tex_x] += sampled_color * sample_weights[:, None]
            accum_weight[tex_y, tex_x] += sample_weights

            print(f"  [Stage 6] View {idx+1}/{len(photos)}: "
                  f"{len(sub_idx):,} projected texels sampled, "
                  f"yaw={yaw:.1f}°")

        # Normalize accumulated colours
        valid_mask = accum_weight > 0
        projected_rgb = np.zeros((R, R, 3), dtype=np.float32)
        for c in range(3):
            projected_rgb[:, :, c] = np.where(
                valid_mask,
                accum_rgb[:, :, c] / (accum_weight + 1e-10),
                0.0
            )

        # Inpaint internal cavity / occluded regions within the facial mesh hull (mouth cavity, ear depth)
        unseen_hull = (valid_mask_tex > 0) & (~valid_mask)
        if np.any(unseen_hull) and np.any(valid_mask):
            inpaint_mask = unseen_hull.astype(np.uint8) * 255
            rgb_uint8 = np.clip(projected_rgb, 0, 255).astype(np.uint8)
            inpainted_bgr = cv2.inpaint(rgb_uint8, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            projected_rgb[unseen_hull] = inpainted_bgr[unseen_hull].astype(np.float32)
            valid_mask = valid_mask | unseen_hull

        projection_mask = np.where(valid_mask, 255, 0).astype(np.uint8)

        return {
            'projected_rgb': projected_rgb.astype(np.float32),
            'projection_mask': projection_mask,
            'weight_map': accum_weight.astype(np.float32),
        }

    def save_maps(
        self,
        result: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "head",
    ) -> Dict[str, str]:
        """Save projected texture and mask to disk."""
        out = Path(output_dir)
        textures_dir = out / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)

        rgb = result['projected_rgb']
        mask = result['projection_mask']

        # Convert float RGB to uint8 BGR for OpenCV
        rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)

        projected_path = textures_dir / f"{prefix}_projected_raw.png"
        mask_path = textures_dir / f"{prefix}_projection_mask.png"

        cv2.imwrite(str(projected_path), rgb_uint8)
        cv2.imwrite(str(mask_path), mask)

        return {
            'projected_texture': str(projected_path),
            'projection_mask': str(mask_path),
        }
