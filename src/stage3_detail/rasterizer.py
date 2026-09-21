"""
Canonical UV rasterization module for Stage 3 Micro-Detail GAN.
Provides barycentric triangle rasterization over the FLAME UV parameterization.
"""
from typing import Tuple, Optional
from pathlib import Path
import numpy as np


def load_flame_uv_layout(uv_template_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads canonical FLAME UV coordinates and texture faces.
    Resolves automatically from standard repository paths if not explicitly provided.
    """
    candidates = []
    if uv_template_path:
        candidates.append(Path(uv_template_path))

    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend([
        root / 'vendor' / 'MICA' / 'data' / 'FLAME2020' / 'head_template.obj',
        root / 'data' / 'FLAME2020' / 'head_template.obj',
        root / 'data' / 'flame_model' / 'head_template.obj',
        root / 'data' / 'flame_model' / 'FLAME_texture.npz',
    ])

    resolved_path = None
    for cand in candidates:
        if cand.exists():
            resolved_path = cand
            break

    if resolved_path is None:
        raise FileNotFoundError(
            "FLAME UV template not found. Please provide head_template.obj or FLAME_texture.npz."
        )

    if resolved_path.suffix == '.npz':
        data = np.load(resolved_path)
        vt = data.get('vt', data.get('uv_coords'))
        ft = data.get('ft', data.get('uv_faces'))
        return vt.astype(np.float32), ft.astype(np.int32)
    elif resolved_path.suffix == '.obj':
        v_list = []
        vt_list = []
        ft_list = []
        f_raw_list = []
        with open(resolved_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()[1:4]
                    v_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
                elif line.startswith('vt '):
                    parts = line.strip().split()
                    vt_list.append([float(parts[1]), float(parts[2])])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:4]
                    face_uvs = []
                    face_verts = []
                    for pt in parts:
                        vals = pt.split('/')
                        face_verts.append(int(vals[0]) - 1)
                        if len(vals) > 1 and vals[1]:
                            face_uvs.append(int(vals[1]) - 1)
                    if len(face_uvs) == 3:
                        ft_list.append(face_uvs)
                    if len(face_verts) == 3:
                        f_raw_list.append(face_verts)

        if len(vt_list) > 0 and len(ft_list) > 0:
            return np.array(vt_list, dtype=np.float32), np.array(ft_list, dtype=np.int32)

        # Fallback to cylindrical unwrapping of 3D vertices
        verts = np.array(v_list, dtype=np.float32)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        u = (np.arctan2(x, -z) + np.pi) / (2.0 * np.pi)
        y_min, y_max = y.min(), y.max()
        v = (y - y_min) / (y_max - y_min + 1e-8)
        uv_coords = np.stack([u, v], axis=1).astype(np.float32)
        uv_faces = np.array(f_raw_list, dtype=np.int32)
        return uv_coords, uv_faces
    else:
        raise ValueError(f"Unsupported UV template format: {resolved_path.suffix}")


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes area-weighted vertex normals from triangle mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    vertex_normals = np.zeros_like(vertices, dtype=np.float32)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
        
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vertex_normals / norms


def rasterize_uv_maps(
    flame_verts_m: np.ndarray,
    flame_normals: np.ndarray,
    uv_coords: np.ndarray,
    uv_faces: np.ndarray,
    flame_faces: Optional[np.ndarray] = None,
    disp_mm: Optional[np.ndarray] = None,
    hit_mask: Optional[np.ndarray] = None,
    resolution: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Barycentric triangle rasterization over FLAME UV layout.
    Rasterizes displacement (if given), coarse 3D position, surface normal, and validity mask.
    """
    disp_map = np.zeros((resolution, resolution), dtype=np.float32)
    pos_map = np.zeros((resolution, resolution, 3), dtype=np.float32)
    norm_map = np.zeros((resolution, resolution, 3), dtype=np.float32)
    mask_map = np.zeros((resolution, resolution), dtype=np.uint8)

    if hit_mask is None:
        hit_mask = np.ones(len(flame_verts_m), dtype=bool)
    if disp_mm is None:
        disp_mm = np.zeros(len(flame_verts_m), dtype=np.float32)

    uv_px = uv_coords.copy()
    uv_px[:, 0] = np.clip(uv_px[:, 0] * (resolution - 1), 0, resolution - 1)
    uv_px[:, 1] = np.clip((1.0 - uv_px[:, 1]) * (resolution - 1), 0, resolution - 1)

    for i, tri_uv in enumerate(uv_faces):
        if flame_faces is not None and i >= len(flame_faces):
            break
        p0, p1, p2 = uv_px[tri_uv[0]], uv_px[tri_uv[1]], uv_px[tri_uv[2]]
        v_idx = flame_faces[i] if flame_faces is not None else tri_uv[:3]
        if not np.all(hit_mask[v_idx]):
            continue

        xmin = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        xmax = min(resolution - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        ymin = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        ymax = min(resolution - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

        if xmax <= xmin or ymax <= ymin:
            continue

        area = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(area) < 1e-6:
            continue

        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        w0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / area
        w1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / area
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(inside):
            continue

        y_coords = ys[inside]
        x_coords = xs[inside]
        w0_in = w0[inside]
        w1_in = w1[inside]
        w2_in = w2[inside]

        disp_map[y_coords, x_coords] = (
            w0_in * disp_mm[v_idx[0]] + w1_in * disp_mm[v_idx[1]] + w2_in * disp_mm[v_idx[2]]
        )
        pos_map[y_coords, x_coords] = (
            w0_in[:, None] * flame_verts_m[v_idx[0]] +
            w1_in[:, None] * flame_verts_m[v_idx[1]] +
            w2_in[:, None] * flame_verts_m[v_idx[2]]
        )
        n_interp = (
            w0_in[:, None] * flame_normals[v_idx[0]] +
            w1_in[:, None] * flame_normals[v_idx[1]] +
            w2_in[:, None] * flame_normals[v_idx[2]]
        )
        n_len = np.linalg.norm(n_interp, axis=1, keepdims=True) + 1e-8
        norm_map[y_coords, x_coords] = (n_interp / n_len + 1.0) * 0.5
        mask_map[y_coords, x_coords] = 255

    return disp_map, pos_map, norm_map, mask_map
