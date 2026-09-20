"""
Multi-Resolution LOD Decimation & Shape Key Re-projection Engine.

Generates 4 production Level-of-Detail (LOD) tiers for Unreal Engine 5:
  LOD0: Full resolution (100% detail, ~24.5k tris or base mesh)
  LOD1: High quality (~5,000 tris, ~20%)
  LOD2: Medium distance (~2,000 tris, ~8%)
  LOD3: Impostor / low distance (~500 tris, ~2%)

CRITICAL: Blender's built-in Decimate modifier strips all shape keys.
This engine solves the problem by computing barycentric correspondence from
decimated vertices back to the LOD0 surface, re-projecting all 52 ARKit blendshapes
so that every LOD tier retains 100% of its facial animation capabilities.
"""
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import numpy as np

from src.stage5_export.retopology import compute_barycentric_weights_triangle


LOD_TRIANGLE_TARGETS = {
    'LOD0': None,      # Full resolution
    'LOD1': 5000,
    'LOD2': 2000,
    'LOD3': 500,
}


def decimate_mesh_spatial_clustering(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_triangles: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure-NumPy spatial grid vertex clustering decimation.
    Serves as a robust, zero-dependency decimation fallback when trimesh/open3d
    is not installed. Simplifies geometry to approximately the target triangle count.
    """
    if len(faces) <= target_triangles:
        return vertices.copy(), faces.copy()

    # Determine spatial grid cell size to achieve target vertex reduction
    # Number of triangles is roughly 2 * number of vertices on closed 2-manifold
    target_vertices = max(target_triangles // 2, 50)
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_extent = np.maximum(bbox_max - bbox_min, 1e-4)

    # Estimate number of grid cells per axis
    volume = np.prod(bbox_extent)
    cell_vol = volume / target_vertices
    cell_size = np.cbrt(cell_vol)

    # Grid coordinates
    grid_coords = np.floor((vertices - bbox_min) / cell_size).astype(np.int32)
    # Hash each vertex to unique cell key
    keys = (
        grid_coords[:, 0].astype(np.int64) * 73856093 ^
        grid_coords[:, 1].astype(np.int64) * 19349663 ^
        grid_coords[:, 2].astype(np.int64) * 83492791
    )

    unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
    n_new_verts = len(unique_keys)

    # Compute centroid of vertices falling in each cell
    new_vertices = np.zeros((n_new_verts, 3), dtype=np.float32)
    counts = np.zeros(n_new_verts, dtype=np.float32)
    for i, cluster_idx in enumerate(inverse_indices):
        new_vertices[cluster_idx] += vertices[i]
        counts[cluster_idx] += 1.0

    counts = np.maximum(counts, 1.0)[:, None]
    new_vertices /= counts

    # Remap faces
    remapped_faces = inverse_indices[faces]
    # Remove degenerate triangles (faces with duplicate vertex indices)
    valid_face_mask = (
        (remapped_faces[:, 0] != remapped_faces[:, 1]) &
        (remapped_faces[:, 1] != remapped_faces[:, 2]) &
        (remapped_faces[:, 2] != remapped_faces[:, 0])
    )
    new_faces = remapped_faces[valid_face_mask]

    # Remove duplicated faces
    sorted_faces = np.sort(new_faces, axis=1)
    _, unique_face_indices = np.unique(sorted_faces, axis=0, return_index=True)
    new_faces = new_faces[unique_face_indices]

    return new_vertices.astype(np.float32), new_faces.astype(np.int32)


def decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_triangles: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimates mesh to approximately target_triangles.
    Uses trimesh quadric simplification if available; otherwise falls back
    to pure-NumPy spatial clustering.
    """
    if len(faces) <= target_triangles:
        return vertices.copy(), faces.copy()

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # Try trimesh quadric decimation
        ratio = target_triangles / len(faces)
        decimated = mesh.simplify_quadric_decimation(face_count=target_triangles)
        return np.asarray(decimated.vertices, dtype=np.float32), np.asarray(decimated.faces, dtype=np.int32)
    except Exception:
        # Fallback to pure-NumPy clustering
        return decimate_mesh_spatial_clustering(vertices, faces, target_triangles)


def reproject_blendshapes_to_lod(
    lod0_vertices: np.ndarray,
    lod0_faces: np.ndarray,
    lod0_blendshapes: Dict[str, np.ndarray],
    lod_vertices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Projects all 52 ARKit blendshapes from LOD0 topology onto decimated LOD topology.
    For each LOD vertex, computes its barycentric coordinates on the closest LOD0 triangle,
    then evaluates the interpolated delta displacement vector.
    """
    n_lod_verts = len(lod_vertices)

    # Pre-calculate LOD0 triangle centroids
    v0 = lod0_vertices[lod0_faces[:, 0]]
    v1 = lod0_vertices[lod0_faces[:, 1]]
    v2 = lod0_vertices[lod0_faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    # For each LOD vertex, find closest triangle and barycentric weights
    tri_indices = np.zeros(n_lod_verts, dtype=np.int32)
    bary_weights = np.zeros((n_lod_verts, 3), dtype=np.float32)

    for i in range(n_lod_verts):
        pt = lod_vertices[i]
        dists_sq = np.sum((centroids - pt) ** 2, axis=1)
        best_tri = int(np.argmin(dists_sq))
        tri_indices[i] = best_tri

        u, v, w = compute_barycentric_weights_triangle(
            pt,
            v0[best_tri],
            v1[best_tri],
            v2[best_tri]
        )
        bary_weights[i] = [u, v, w]

    # Re-project each blendshape delta
    lod_blendshapes: Dict[str, np.ndarray] = {}
    f_v0 = lod0_faces[tri_indices, 0]
    f_v1 = lod0_faces[tri_indices, 1]
    f_v2 = lod0_faces[tri_indices, 2]

    u = bary_weights[:, 0, None]
    v = bary_weights[:, 1, None]
    w = bary_weights[:, 2, None]

    for name, delta0 in lod0_blendshapes.items():
        delta_reprojected = (
            u * delta0[f_v0] +
            v * delta0[f_v1] +
            w * delta0[f_v2]
        )
        lod_blendshapes[name] = delta_reprojected.astype(np.float32)

    return lod_blendshapes


def generate_lod_chain(
    lod0_vertices: np.ndarray,
    lod0_faces: np.ndarray,
    lod0_blendshapes: Dict[str, np.ndarray],
    targets: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """
    Generates full multi-resolution LOD chain (LOD0 -> LOD3) with preserved ARKit-52 blendshapes.
    Returns:
    {
      'LOD0': {'vertices': ..., 'faces': ..., 'blendshapes': ...},
      'LOD1': {'vertices': ..., 'faces': ..., 'blendshapes': ...},
      'LOD2': {'vertices': ..., 'faces': ..., 'blendshapes': ...},
      'LOD3': {'vertices': ..., 'faces': ..., 'blendshapes': ...}
    }
    """
    targets = targets or LOD_TRIANGLE_TARGETS
    chain = {}

    # LOD0 is full resolution
    chain['LOD0'] = {
        'vertices': lod0_vertices.copy(),
        'faces': lod0_faces.copy(),
        'blendshapes': {k: v.copy() for k, v in lod0_blendshapes.items()},
    }

    current_v = lod0_vertices
    current_f = lod0_faces

    for lod_name in ['LOD1', 'LOD2', 'LOD3']:
        target_tri = targets.get(lod_name)
        if target_tri is None or len(current_f) <= target_tri:
            # If current mesh is already smaller than target, keep current
            lod_v, lod_f = current_v.copy(), current_f.copy()
        else:
            lod_v, lod_f = decimate_mesh(current_v, current_f, target_tri)

        # Reproject all shape keys from LOD0 onto this decimated topology
        lod_bs = reproject_blendshapes_to_lod(
            lod0_vertices=lod0_vertices,
            lod0_faces=lod0_faces,
            lod0_blendshapes=lod0_blendshapes,
            lod_vertices=lod_v
        )

        chain[lod_name] = {
            'vertices': lod_v,
            'faces': lod_f,
            'blendshapes': lod_bs,
        }
        current_v = lod_v
        current_f = lod_f

    return chain


def export_lod_chain_assets(
    lod_chain: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    output_dir: Union[str, Path]
) -> Dict[str, Dict[str, str]]:
    """
    Exports each LOD mesh as an OBJ and its associated ARKit shape keys as JSON.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for lod_name, data in lod_chain.items():
        v = data['vertices']
        f = data['faces']
        bs = data['blendshapes']

        obj_path = out_dir / f"face_{lod_name.lower()}.obj"
        with open(obj_path, "w") as fp:
            fp.write(f"# Face Mesh {lod_name} - Triangles: {len(f)}\n")
            for vert in v:
                fp.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
            for face in f + 1:
                fp.write(f"f {face[0]} {face[1]} {face[2]}\n")

        bs_path = out_dir / f"blendshapes_{lod_name.lower()}.json"
        bs_payload = {
            "version": "1.0",
            "lod": lod_name,
            "num_vertices": int(len(v)),
            "num_triangles": int(len(f)),
            "blendshapes": {k: np.round(val, decimals=6).tolist() for k, val in bs.items()}
        }
        with open(bs_path, "w") as fp:
            json.dump(bs_payload, fp, indent=2)

        manifest[lod_name] = {
            'mesh_obj': str(obj_path.resolve()),
            'blendshapes_json': str(bs_path.resolve()),
            'num_vertices': len(v),
            'num_triangles': len(f),
            'shape_keys_count': len(bs),
        }

    return manifest
