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

def estimate_camera_projection_matrix(
    landmarks_5: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    image_shape: Tuple[int, int],
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

    Returns
    -------
    P : (3, 4) projection matrix   x_px = P @ [X, Y, Z, 1]^T
    """
    h, w = image_shape[:2]

    # --- Canonical 3D FLAME landmark positions (metres, neutral pose) ---
    # Order: left eye, right eye, nose tip, left mouth corner, right mouth corner
    canonical_3d = np.array([
        [-0.0300,  0.0337,  0.0580],   # left eye centre
        [ 0.0300,  0.0337,  0.0580],   # right eye centre
        [ 0.0000, -0.0110,  0.0830],   # nose tip
        [-0.0210, -0.0380,  0.0680],   # left mouth corner
        [ 0.0210, -0.0380,  0.0680],   # right mouth corner
    ], dtype=np.float64)

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
    # Rotate canonical landmarks
    rotated_3d = (R @ canonical_3d.T).T   # (5, 3)

    # Use solvePnP for a more stable solution
    lm_2d = landmarks_5.astype(np.float64).reshape(5, 1, 2)
    cam_matrix = np.array([
        [w, 0, w / 2.0],
        [0, w, h / 2.0],
        [0, 0, 1.0    ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        canonical_3d, lm_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

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

    Uses a simple per-vertex depth-buffer approach:
    1. Project all vertices into pixel space.
    2. For each pixel, record the minimum depth vertex.
    3. A vertex is visible if its depth is within `epsilon` of the min depth at its pixel.
    """
    h, w = image_shape[:2]
    n_verts = len(vertices_3d)

    # Project vertices to 2D
    pts_h = np.hstack([vertices_3d, np.ones((n_verts, 1))])  # (N, 4)
    proj = (P @ pts_h.T).T  # (N, 3)

    # Handle perspective division
    z = proj[:, 2].copy()
    z[z == 0] = 1e-8
    px = (proj[:, 0] / z).astype(np.float32)
    py = (proj[:, 1] / z).astype(np.float32)

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

        if flame_faces is None:
            flame_faces = faces

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

            # Build projection matrix for this view
            P = estimate_camera_projection_matrix(
                landmarks_5, yaw, pitch, roll, (h_img, w_img)
            )

            # Compute visibility (which vertices are not occluded)
            visible = _compute_visibility_mask(
                vertices, faces, P, (h_img, w_img), self.vis_eps
            )

            # Compute camera direction for angle weighting
            # Camera position: C = -R^T @ t (extract from P)
            # For weak perspective, approximate as the negative of the
            # translation component of P
            R_mat = P[:3, :3]
            t_vec = P[:3, 3]
            try:
                cam_pos = -np.linalg.inv(R_mat) @ t_vec
            except np.linalg.LinAlgError:
                cam_pos = np.array([0, 0, 1.0])

            # Compute per-vertex view direction
            view_dirs = cam_pos[np.newaxis, :] - vertices  # (V, 3)
            view_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8
            view_dirs_unit = view_dirs / view_norms

            # Cosine between surface normal and view direction
            cos_angles = np.sum(vertex_normals * view_dirs_unit, axis=1)  # (V,)
            cos_angles = np.clip(cos_angles, 0.0, 1.0)

            # Per-vertex weight: cosine^gamma * det_score * visibility
            det_score = getattr(det, 'det_score', 1.0)
            per_vertex_weight = (cos_angles ** self.gamma) * det_score * visible.astype(np.float64)

            # Project each vertex into 2D image space for colour sampling
            pts_h = np.hstack([vertices, np.ones((len(vertices), 1))])  # (V, 4)
            proj_2d = (P @ pts_h.T).T  # (V, 3)
            z_proj = proj_2d[:, 2].copy()
            z_proj[z_proj == 0] = 1e-8
            px_x = proj_2d[:, 0] / z_proj   # (V,) pixel x
            px_y = proj_2d[:, 1] / z_proj   # (V,) pixel y

            # Sample pixel colors per vertex (bilinear)
            per_vertex_color = np.zeros((len(vertices), 3), dtype=np.float64)
            for vi in range(len(vertices)):
                if per_vertex_weight[vi] <= 0:
                    continue
                x_f, y_f = float(px_x[vi]), float(px_y[vi])
                x0, y0 = int(np.floor(x_f)), int(np.floor(y_f))
                x1, y1 = x0 + 1, y0 + 1

                if x0 < 0 or y0 < 0 or x1 >= w_img or y1 >= h_img:
                    per_vertex_weight[vi] = 0
                    continue

                # Bilinear interpolation
                dx, dy = x_f - x0, y_f - y0
                c00 = photo[y0, x0].astype(np.float64)
                c01 = photo[y0, x1].astype(np.float64)
                c10 = photo[y1, x0].astype(np.float64)
                c11 = photo[y1, x1].astype(np.float64)
                color = (c00 * (1 - dx) * (1 - dy) +
                         c01 * dx * (1 - dy) +
                         c10 * (1 - dx) * dy +
                         c11 * dx * dy)
                per_vertex_color[vi] = color

            # Rasterize into UV space with barycentric interpolation
            self._rasterize_view(
                uv_coords, uv_faces, flame_faces,
                per_vertex_color, per_vertex_weight,
                accum_rgb, accum_weight
            )

            print(f"  [Stage 6] View {idx+1}/{len(photos)}: "
                  f"{int(np.sum(visible))}/{len(vertices)} visible vertices, "
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

        projection_mask = np.where(valid_mask, 255, 0).astype(np.uint8)

        return {
            'projected_rgb': projected_rgb.astype(np.float32),
            'projection_mask': projection_mask,
            'weight_map': accum_weight.astype(np.float32),
        }

    def _rasterize_view(
        self,
        uv_coords: np.ndarray,
        uv_faces: np.ndarray,
        geom_faces: np.ndarray,
        vertex_colors: np.ndarray,
        vertex_weights: np.ndarray,
        accum_rgb: np.ndarray,
        accum_weight: np.ndarray,
    ) -> None:
        """Rasterize a single view's vertex data into UV space via barycentric interpolation."""
        R = self.resolution
        uv_px = uv_coords.copy().astype(np.float64)
        uv_px[:, 0] = np.clip(uv_px[:, 0] * (R - 1), 0, R - 1)
        uv_px[:, 1] = np.clip((1.0 - uv_px[:, 1]) * (R - 1), 0, R - 1)

        n_faces = min(len(uv_faces), len(geom_faces))

        for i in range(n_faces):
            tri_uv = uv_faces[i]
            tri_geom = geom_faces[i]

            # Skip if all weights are zero
            w0_v = vertex_weights[tri_geom[0]]
            w1_v = vertex_weights[tri_geom[1]]
            w2_v = vertex_weights[tri_geom[2]]
            if w0_v <= 0 and w1_v <= 0 and w2_v <= 0:
                continue

            p0 = uv_px[tri_uv[0]]
            p1 = uv_px[tri_uv[1]]
            p2 = uv_px[tri_uv[2]]

            xmin = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
            xmax = min(R - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
            ymin = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
            ymax = min(R - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

            if xmax <= xmin or ymax <= ymin:
                continue

            area = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
            if abs(area) < 1e-6:
                continue

            xs, ys = np.meshgrid(
                np.arange(xmin, xmax + 1, dtype=np.float64),
                np.arange(ymin, ymax + 1, dtype=np.float64)
            )
            bary0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / area
            bary1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / area
            bary2 = 1.0 - bary0 - bary1

            inside = (bary0 >= 0) & (bary1 >= 0) & (bary2 >= 0)
            if not np.any(inside):
                continue

            yc = ys[inside].astype(np.int32)
            xc = xs[inside].astype(np.int32)
            b0 = bary0[inside]
            b1 = bary1[inside]
            b2 = bary2[inside]

            # Interpolate weight
            w_interp = (b0 * w0_v + b1 * w1_v + b2 * w2_v)

            # Interpolate colour
            c0 = vertex_colors[tri_geom[0]]   # (3,)
            c1 = vertex_colors[tri_geom[1]]
            c2 = vertex_colors[tri_geom[2]]
            color_interp = (b0[:, None] * c0 + b1[:, None] * c1 + b2[:, None] * c2)  # (K, 3)

            # Accumulate weighted colour
            for j in range(len(yc)):
                if w_interp[j] > 0:
                    accum_rgb[yc[j], xc[j]] += color_interp[j] * w_interp[j]
                    accum_weight[yc[j], xc[j]] += w_interp[j]

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
