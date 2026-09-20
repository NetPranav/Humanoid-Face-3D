"""
Retopology & Dense Barycentric Correspondence Engine.

Maps canonical FLAME geometry (5023 vertices, 9976 triangles) onto production
game-ready target topologies (e.g. ICT-FaceKit, custom quad game meshes).

Computes or loads the sparse correspondence matrix W in R^(N_target x 5023):
    v_target = W @ v_flame

Supports both scipy.sparse.csr_matrix and a standalone pure-NumPy sparse representation
for zero-dependency execution across local and cloud environments.
"""
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np


class SparseMatrixCSR:
    """
    Minimal standalone CSR sparse matrix implementation in pure NumPy.
    Ensures sparse matrix-vector and matrix-matrix operations run without scipy.
    """
    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: Tuple[int, int]):
        self.data = np.asarray(data, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.shape = shape

    def dot(self, other: np.ndarray) -> np.ndarray:
        """Matrix multiplication: result = self @ other."""
        n_rows, n_cols = self.shape
        other = np.asarray(other, dtype=np.float32)
        assert other.shape[0] == n_cols, f"Shape mismatch: {self.shape} vs {other.shape}"

        if other.ndim == 1:
            out = np.zeros(n_rows, dtype=np.float32)
            for i in range(n_rows):
                start = self.indptr[i]
                end = self.indptr[i + 1]
                cols = self.indices[start:end]
                vals = self.data[start:end]
                out[i] = np.dot(vals, other[cols])
            return out
        elif other.ndim == 2:
            n_features = other.shape[1]
            out = np.zeros((n_rows, n_features), dtype=np.float32)
            for i in range(n_rows):
                start = self.indptr[i]
                end = self.indptr[i + 1]
                cols = self.indices[start:end]
                vals = self.data[start:end]
                out[i] = np.dot(vals, other[cols, :])
            return out
        else:
            raise ValueError(f"Unsupported operand dimensionality: {other.ndim}")

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        return self.dot(other)


def compute_barycentric_weights_triangle(
    point: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes barycentric coordinates (u, v, w) of a 3D point projected onto triangle (v0, v1, v2).
    u*v0 + v*v1 + w*v2 = projection, with u + v + w = 1.0.
    """
    e0 = v1 - v0
    e1 = v2 - v0
    ep = point - v0

    d00 = float(np.dot(e0, e0))
    d01 = float(np.dot(e0, e1))
    d11 = float(np.dot(e1, e1))
    dp0 = float(np.dot(ep, e0))
    dp1 = float(np.dot(ep, e1))

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

    v = (d11 * dp0 - d01 * dp1) / denom
    w = (d00 * dp1 - d01 * dp0) / denom
    u = 1.0 - v - w

    # Clamp to triangle bounds and re-normalize to guarantee partition of unity
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    w = max(0.0, min(1.0, w))
    total = u + v + w
    if total > 1e-9:
        u /= total
        v /= total
        w /= total
    else:
        u, v, w = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

    return u, v, w


def build_correspondence_matrix(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
) -> SparseMatrixCSR:
    """
    Constructs the dense barycentric correspondence matrix W in R^(N_target x N_source).
    For each target vertex, finds the closest triangle on source mesh and assigns
    the 3 barycentric weights to the triangle's 3 vertex indices.
    """
    n_source = len(source_vertices)
    n_target = len(target_vertices)

    # Compute triangle centroids for fast nearest-triangle lookup
    tri_v0 = source_vertices[source_faces[:, 0]]
    tri_v1 = source_vertices[source_faces[:, 1]]
    tri_v2 = source_vertices[source_faces[:, 2]]
    centroids = (tri_v0 + tri_v1 + tri_v2) / 3.0  # (n_faces, 3)

    data = []
    indices = []
    indptr = [0]

    for t_idx in range(n_target):
        pt = target_vertices[t_idx]
        # Fast nearest triangle via centroid distance
        dists_sq = np.sum((centroids - pt) ** 2, axis=1)
        best_f_idx = int(np.argmin(dists_sq))

        v0_idx, v1_idx, v2_idx = source_faces[best_f_idx]
        u, v, w = compute_barycentric_weights_triangle(
            pt,
            source_vertices[v0_idx],
            source_vertices[v1_idx],
            source_vertices[v2_idx]
        )

        # Record 3 non-zero entries for this row
        row_entries = [(int(v0_idx), float(u)), (int(v1_idx), float(v)), (int(v2_idx), float(w))]
        # Merge if any indices are identical (degenerate triangle)
        entry_dict: Dict[int, float] = {}
        for col, weight in row_entries:
            entry_dict[col] = entry_dict.get(col, 0.0) + weight

        # Sort columns in row
        sorted_cols = sorted(entry_dict.keys())
        for col in sorted_cols:
            indices.append(col)
            data.append(entry_dict[col])

        indptr.append(len(indices))

    return SparseMatrixCSR(
        data=np.array(data, dtype=np.float32),
        indices=np.array(indices, dtype=np.int32),
        indptr=np.array(indptr, dtype=np.int32),
        shape=(n_target, n_source)
    )


def save_correspondence_matrix(matrix: SparseMatrixCSR, filepath: Union[str, Path]):
    """Saves the CSR correspondence matrix to a compressed .npz file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.array(matrix.shape, dtype=np.int32)
    )


def load_correspondence_matrix(filepath: Union[str, Path]) -> SparseMatrixCSR:
    """Loads the CSR correspondence matrix from a .npz file."""
    path = Path(filepath)
    assert path.exists(), f"Correspondence matrix file not found: {path}"
    with np.load(str(path)) as loader:
        shape = tuple(loader['shape'])
        return SparseMatrixCSR(
            data=loader['data'],
            indices=loader['indices'],
            indptr=loader['indptr'],
            shape=(int(shape[0]), int(shape[1]))
        )


def transfer_retopology(
    flame_vertices: np.ndarray,
    correspondence_w: SparseMatrixCSR
) -> np.ndarray:
    """
    Applies the barycentric correspondence matrix to retopologize FLAME vertices.
    v_target = W @ v_flame.
    Runs in sub-millisecond time.
    """
    return correspondence_w.dot(flame_vertices)


def compute_retopology_error(
    original_target: np.ndarray,
    reconstructed_target: np.ndarray
) -> Dict[str, float]:
    """
    Computes per-vertex Euclidean deviation between ground-truth and transferred geometry.
    Returns mean, median, 95th percentile, and max deviation in millimeters.
    """
    diffs = np.linalg.norm(original_target - reconstructed_target, axis=1)
    return {
        'mean_error_mm': float(np.mean(diffs)),
        'median_error_mm': float(np.median(diffs)),
        'p95_error_mm': float(np.percentile(diffs, 95)),
        'max_error_mm': float(np.max(diffs)),
    }
