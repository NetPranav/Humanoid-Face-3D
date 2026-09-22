"""
High-Performance Vectorized Loop Subdivision and Displacement Mapping Engine.

Provides:
- Exact Loop subdivision for 2-manifold triangle meshes with boundary.
- Synchronized UV coordinate subdivision to preserve texture mapping.
- Neck Seam Contract enforcement (bottom 20% Y collar pinned to 0 delta).
- Bilinear displacement sampling from 16-bit / float UV displacement maps.
- Smooth normal computation and high-resolution OBJ export.
"""

from typing import Tuple, Optional, Dict
from pathlib import Path
import numpy as np


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Computes area-weighted smooth vertex normals for a triangle mesh.
    
    Args:
        vertices: (N, 3) float32 array
        faces: (M, 3) int32 array
        
    Returns:
        normals: (N, 3) float32 normalized unit vectors
    """
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


def loop_subdivide_step(
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    uv_faces: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Performs one step of Loop subdivision on a triangular mesh.
    
    For 3D vertices:
      - Interior edge: 3/8*(A+B) + 1/8*(C+D)
      - Boundary edge: 1/2*(A+B)
      - Interior vertex: (1 - k*beta)*V + beta*sum(neighbors)
      - Boundary vertex: 3/4*V + 1/8*(B1 + B2)
      
    For UV coordinates (if provided):
      - Linearly splits UV edges to preserve texture layout without distortion.
    """
    n_verts = len(verts)
    n_faces = len(faces)
    
    # 1. Build directed half-edges: (3*N_f, 2)
    h0 = faces[:, [0, 1]]
    h1 = faces[:, [1, 2]]
    h2 = faces[:, [2, 0]]
    all_half_edges = np.vstack([h0, h1, h2])
    
    # Opposite vertices in the original triangles: (3*N_f,)
    opp_verts = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    
    # 2. Canonical edge keys (u < v)
    u = np.minimum(all_half_edges[:, 0], all_half_edges[:, 1])
    v = np.maximum(all_half_edges[:, 0], all_half_edges[:, 1])
    edge_keys = (u.astype(np.int64) << 32) | v.astype(np.int64)
    
    unq_keys, first_indices, inverse_indices, counts = np.unique(
        edge_keys, return_index=True, return_inverse=True, return_counts=True
    )
    n_edges = len(unq_keys)
    
    unq_u = (unq_keys >> 32).astype(np.int32)
    unq_v = (unq_keys & 0xFFFFFFFF).astype(np.int32)
    
    # 3. Compute edge vertex positions
    edge_verts = np.zeros((n_edges, 3), dtype=np.float32)
    
    # Group half-edges by unique edge index to find opposite vertices
    sort_order = np.argsort(inverse_indices)
    sorted_inv = inverse_indices[sort_order]
    sorted_opp = opp_verts[sort_order]
    
    # Offsets for each edge
    idx0 = np.searchsorted(sorted_inv, np.arange(n_edges), side='left')
    idx1 = np.minimum(idx0 + 1, len(sorted_inv) - 1)
    
    opp_c = sorted_opp[idx0]
    opp_d = sorted_opp[idx1]
    
    interior_mask = (counts == 2)
    boundary_mask = (counts == 1)
    
    # Boundary edge: 1/2 * (A + B)
    edge_verts[boundary_mask] = 0.5 * (verts[unq_u[boundary_mask]] + verts[unq_v[boundary_mask]])
    # Interior edge: 3/8 * (A + B) + 1/8 * (C + D)
    edge_verts[interior_mask] = (
        0.375 * (verts[unq_u[interior_mask]] + verts[unq_v[interior_mask]]) +
        0.125 * (verts[opp_c[interior_mask]] + verts[opp_d[interior_mask]])
    )
    
    # 4. Update existing vertices
    neighbor_sum = np.zeros_like(verts)
    degree = np.zeros(n_verts, dtype=np.float32)
    
    np.add.at(neighbor_sum, unq_u, verts[unq_v])
    np.add.at(neighbor_sum, unq_v, verts[unq_u])
    np.add.at(degree, unq_u, 1.0)
    np.add.at(degree, unq_v, 1.0)
    
    # Standard Loop beta formula
    k = np.maximum(degree, 1.0)
    cos_term = 0.375 + 0.25 * np.cos(2.0 * np.pi / k)
    beta = (1.0 / k) * (0.625 - cos_term * cos_term)
    
    updated_verts = (1.0 - k[:, None] * beta[:, None]) * verts + beta[:, None] * neighbor_sum
    
    # Handle boundary vertices: 3/4 * V + 1/8 * (B1 + B2)
    b_u = unq_u[boundary_mask]
    b_v = unq_v[boundary_mask]
    if len(b_u) > 0:
        b_neighbor_sum = np.zeros_like(verts)
        b_degree = np.zeros(n_verts, dtype=np.float32)
        np.add.at(b_neighbor_sum, b_u, verts[b_v])
        np.add.at(b_neighbor_sum, b_v, verts[b_u])
        np.add.at(b_degree, b_u, 1.0)
        np.add.at(b_degree, b_v, 1.0)
        
        b2_mask = (b_degree == 2.0)
        updated_verts[b2_mask] = 0.75 * verts[b2_mask] + 0.125 * b_neighbor_sum[b2_mask]
        
    new_verts = np.vstack([updated_verts, edge_verts])
    
    # 5. Construct new faces (4 sub-triangles per face)
    e0_idx = inverse_indices[0:n_faces] + n_verts
    e1_idx = inverse_indices[n_faces:2*n_faces] + n_verts
    e2_idx = inverse_indices[2*n_faces:3*n_faces] + n_verts
    
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    
    f0 = np.stack([v0, e0_idx, e2_idx], axis=1)
    f1 = np.stack([v1, e1_idx, e0_idx], axis=1)
    f2 = np.stack([v2, e2_idx, e1_idx], axis=1)
    f3 = np.stack([e0_idx, e1_idx, e2_idx], axis=1)
    new_faces = np.vstack([f0, f1, f2, f3])
    
    # 6. Synchronized UV subdivision (if provided)
    new_uvs = None
    new_uv_faces = None
    if uvs is not None and uv_faces is not None:
        n_uvs = len(uvs)
        uv_h0 = uv_faces[:, [0, 1]]
        uv_h1 = uv_faces[:, [1, 2]]
        uv_h2 = uv_faces[:, [2, 0]]
        all_uv_h = np.vstack([uv_h0, uv_h1, uv_h2])
        
        uv_u = np.minimum(all_uv_h[:, 0], all_uv_h[:, 1])
        uv_v = np.maximum(all_uv_h[:, 0], all_uv_h[:, 1])
        uv_edge_keys = (uv_u.astype(np.int64) << 32) | uv_v.astype(np.int64)
        
        unq_uv_keys, uv_inv = np.unique(uv_edge_keys, return_inverse=True)
        unq_uv_u = (unq_uv_keys >> 32).astype(np.int32)
        unq_uv_v = (unq_uv_keys & 0xFFFFFFFF).astype(np.int32)
        
        # Edge UVs are strictly linear midpoints to prevent texture warping
        edge_uvs = 0.5 * (uvs[unq_uv_u] + uvs[unq_uv_v])
        new_uvs = np.vstack([uvs, edge_uvs])
        
        uv_e0_idx = uv_inv[0:n_faces] + n_uvs
        uv_e1_idx = uv_inv[n_faces:2*n_faces] + n_uvs
        uv_e2_idx = uv_inv[2*n_faces:3*n_faces] + n_uvs
        
        vt0 = uv_faces[:, 0]
        vt1 = uv_faces[:, 1]
        vt2 = uv_faces[:, 2]
        
        uv_f0 = np.stack([vt0, uv_e0_idx, uv_e2_idx], axis=1)
        uv_f1 = np.stack([vt1, uv_e1_idx, uv_e0_idx], axis=1)
        uv_f2 = np.stack([vt2, uv_e2_idx, uv_e1_idx], axis=1)
        uv_f3 = np.stack([uv_e0_idx, uv_e1_idx, uv_e2_idx], axis=1)
        new_uv_faces = np.vstack([uv_f0, uv_f1, uv_f2, uv_f3])
        
    return new_verts, new_faces, new_uvs, new_uv_faces


def loop_subdivide(
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    uv_faces: Optional[np.ndarray] = None,
    levels: int = 3
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Subdivides a triangle mesh `levels` times using Loop subdivision.
    
    For FLAME (5,023 verts, 9,976 faces):
      Level 1: ~20k verts, ~40k faces
      Level 2: ~80k verts, ~160k faces
      Level 3: ~320k verts, ~640k faces (Film quality target)
    """
    curr_v = verts.copy()
    curr_f = faces.copy()
    curr_uv = uvs.copy() if uvs is not None else None
    curr_uv_f = uv_faces.copy() if uv_faces is not None else None
    
    for _ in range(levels):
        curr_v, curr_f, curr_uv, curr_uv_f = loop_subdivide_step(
            curr_v, curr_f, curr_uv, curr_uv_f
        )
        
    return curr_v, curr_f, curr_uv, curr_uv_f


def sample_bilinear_2d(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear sampling of a 2D image (H, W) or (H, W, C) at normalized UV coords in [0, 1].
    UV convention: u in [0, 1] maps to X in [0, W-1], v in [0, 1] maps to Y in [H-1, 0] (OpenGL style).
    """
    h, w = img.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    
    px = u * (w - 1)
    py = (1.0 - v) * (h - 1)
    
    x0 = np.floor(px).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(py).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    
    wx = (px - x0)
    wy = (py - y0)
    
    if img.ndim == 2:
        val = (
            (1.0 - wx) * (1.0 - wy) * img[y0, x0] +
            wx * (1.0 - wy) * img[y0, x1] +
            (1.0 - wx) * wy * img[y1, x0] +
            wx * wy * img[y1, x1]
        )
    else:
        wx = wx[:, None]
        wy = wy[:, None]
        val = (
            (1.0 - wx) * (1.0 - wy) * img[y0, x0] +
            wx * (1.0 - wy) * img[y0, x1] +
            (1.0 - wx) * wy * img[y1, x0] +
            wx * wy * img[y1, x1]
        )
    return val.astype(np.float32)


def apply_displacement_to_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    disp_map_mm: np.ndarray,
    neck_pinning: bool = True,
    disp_scale_meters: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies high-resolution displacement map onto the subdivided mesh along smooth vertex normals.
    
    Args:
        verts: (N, 3) Subdivided vertices in meters
        faces: (M, 3) Subdivided faces
        uvs: (K, 2) Subdivided UV coordinates
        uv_faces: (M, 3) Subdivided UV face indices
        disp_map_mm: (H, W) Signed displacement map in millimeters
        neck_pinning: Enforce Neck Seam Contract (delta=0 for bottom 20% Y)
        disp_scale_meters: Conversion from displacement map units (mm) to mesh units (m)
        
    Returns:
        displaced_verts: (N, 3) float32
        displaced_normals: (N, 3) float32 smooth vertex normals
    """
    n_verts = len(verts)
    
    # 1. Map each 3D vertex to its primary UV coordinate
    vert_uvs = np.zeros((n_verts, 2), dtype=np.float32)
    vert_uvs[faces[:, 0]] = uvs[uv_faces[:, 0]]
    vert_uvs[faces[:, 1]] = uvs[uv_faces[:, 1]]
    vert_uvs[faces[:, 2]] = uvs[uv_faces[:, 2]]
    
    # 2. Sample displacement at each vertex's UV
    sampled_disp_mm = sample_bilinear_2d(disp_map_mm, vert_uvs)
    
    # 3. Enforce Neck Seam Contract: delta == 0.0 for y_norm <= 0.20
    if neck_pinning:
        y = verts[:, 1]
        y_min, y_max = y.min(), y.max()
        y_span = max(y_max - y_min, 1e-6)
        y_norm = (y - y_min) / y_span
        neck_weight = np.clip((y_norm - 0.20) / 0.05, 0.0, 1.0)
        sampled_disp_mm = sampled_disp_mm * neck_weight
        
    # 4. Compute smooth vertex normals on the subdivided mesh
    smooth_normals = compute_vertex_normals(verts, faces)
    
    # 5. Displace along surface normals
    displaced_verts = verts + smooth_normals * (sampled_disp_mm[:, None] * disp_scale_meters)
    
    # 6. Recompute normals on the displaced mesh
    displaced_normals = compute_vertex_normals(displaced_verts, faces)
    
    return displaced_verts, displaced_normals


def export_obj_with_uvs(
    filepath: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    uv_faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None
) -> None:
    """
    Exports a production-grade Wavefront OBJ file with vertices, texture coordinates,
    and vertex normals.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    has_uv = uvs is not None and uv_faces is not None and len(uvs) > 0 and len(uv_faces) == len(faces)
    has_vn = normals is not None and len(normals) == len(verts)
    
    with open(filepath, 'w') as f:
        f.write("# Production 3D Head Mesh (Loop Subdivided & Displaced)\n")
        f.write(f"# Vertices: {len(verts)}, Triangles: {len(faces)}\n")
        
        # Vertices
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Texture coordinates
        if has_uv:
            for vt in uvs:
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
                
        # Vertex normals
        if has_vn:
            for vn in normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
                
        # Faces
        if has_uv and has_vn:
            for i in range(len(faces)):
                fv = faces[i] + 1
                fvt = uv_faces[i] + 1
                f.write(f"f {fv[0]}/{fvt[0]}/{fv[0]} {fv[1]}/{fvt[1]}/{fv[1]} {fv[2]}/{fvt[2]}/{fv[2]}\n")
        elif has_uv:
            for i in range(len(faces)):
                fv = faces[i] + 1
                fvt = uv_faces[i] + 1
                f.write(f"f {fv[0]}/{fvt[0]} {fv[1]}/{fvt[1]} {fv[2]}/{fvt[2]}\n")
        else:
            for fv in faces + 1:
                f.write(f"f {fv[0]} {fv[1]} {fv[2]}\n")
