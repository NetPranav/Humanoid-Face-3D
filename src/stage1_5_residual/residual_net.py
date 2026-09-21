"""
Stage 1.5: Macro-Shape Residual Correction Network.

Predicts non-linear per-vertex displacement corrections on top of FLAME's
linear PCA output to break through the representational ceiling.

Why this exists:
    FLAME's 300 PCA shape directions are global — they affect the entire head.
    Localized morphological traits (masseter hypertrophy, buccal fat pads,
    localized jowl mass, ethnic nasal bridge differences) cannot be represented
    independently in a linear subspace without triggering unintended global
    deformations. This network learns the residual between what FLAME's linear
    fit produces and what the actual 3D scan shows.

Architecture:
    Multi-scale Graph Convolution on the FLAME mesh connectivity:
        Input:  FLAME vertices V(β) ∈ R^{5023×3} + multi-view identity features
        Output: Coarse vertex displacements ΔV_residual ∈ R^{5023×3}
        Final:  V_corrected = V_FLAME(β) + ΔV_residual

Constraints:
    1. Collar pinning: ΔV[i] ≡ 0 for collar vertices (y_norm ≤ 0.20)
    2. Laplacian smoothness: ||L · ΔV||² penalizes high-frequency spiking
    3. Magnitude clamp: ||ΔV||_∞ ≤ 25.0 mm
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    class _MockModule:
        pass
    class _MockNN:
        Module = _MockModule
    nn = _MockNN()
    F = None

from src.utils.flame_model import FLAMEModel, N_VERTS, N_SHAPE


def build_adjacency_from_faces(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """
    Build symmetric adjacency list from face array.

    Returns:
        adj: list of sets, adj[i] = set of vertex indices adjacent to vertex i
    """
    adj = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])
    return adj


def build_laplacian_matrix(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """
    Build the combinatorial graph Laplacian L = D - A for the mesh.

    Used in Laplacian smoothness regularization to prevent high-frequency
    spiking in the predicted residual displacements.
    """
    adj = build_adjacency_from_faces(faces, n_verts)

    # Sparse construction for efficiency
    rows, cols, vals = [], [], []
    for i in range(n_verts):
        neighbors = list(adj[i])
        degree = len(neighbors)
        rows.append(i)
        cols.append(i)
        vals.append(float(degree))
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            vals.append(-1.0)

    L = np.zeros((n_verts, n_verts), dtype=np.float32)
    for r, c, v in zip(rows, cols, vals):
        L[r, c] = v

    return L


class GraphConvBlock(nn.Module):
    """
    Single graph convolution layer operating on mesh vertices.

    Aggregates features from 1-ring neighbors using the adjacency structure
    of the FLAME mesh topology. Uses InstanceNorm + residual connections
    for stable training.
    """

    def __init__(self, in_features: int, out_features: int, n_verts: int):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.use_residual = (in_features == out_features)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_verts, C) vertex features
            adj_matrix: (N_verts, N_verts) normalized adjacency matrix

        Returns:
            (B, N_verts, C_out) updated vertex features
        """
        h_self = self.linear_self(x)
        # Aggregate neighbor features: adj_matrix @ x
        h_neigh = torch.matmul(adj_matrix, x)
        h_neigh = self.linear_neigh(h_neigh)
        h = self.norm(h_self + h_neigh)
        h = self.activation(h)
        if self.use_residual:
            h = h + x
        return h


class MacroShapeResidualNetwork(nn.Module):
    """
    Graph convolutional network predicting per-vertex macro-shape corrections.

    Takes FLAME vertices + identity features and outputs ΔV_residual that
    captures morphology outside the linear PCA span (heavy jaws, fat cheeks,
    ethnic bone structure differences).
    """

    def __init__(
        self,
        flame_model: FLAMEModel,
        identity_dim: int = 512,
        hidden_dim: int = 128,
        n_layers: int = 4,
        max_displacement_mm: float = 25.0,
    ):
        super().__init__()
        if torch is None:
            raise RuntimeError("PyTorch required for MacroShapeResidualNetwork")

        self.n_verts = N_VERTS
        self.max_displacement = max_displacement_mm
        self.hidden_dim = hidden_dim

        # Build normalized adjacency from FLAME topology
        adj = build_adjacency_from_faces(flame_model.faces, N_VERTS)
        adj_dense = np.zeros((N_VERTS, N_VERTS), dtype=np.float32)
        for i in range(N_VERTS):
            neighbors = list(adj[i])
            if len(neighbors) > 0:
                weight = 1.0 / len(neighbors)
                for j in neighbors:
                    adj_dense[i, j] = weight
        self.register_buffer('adj_matrix', torch.from_numpy(adj_dense))

        # Build Laplacian for smoothness regularization
        L = build_laplacian_matrix(flame_model.faces, N_VERTS)
        self.register_buffer('laplacian', torch.from_numpy(L))

        # Collar pinning mask: vertices with y_norm ≤ 0.20 get zero displacement
        v_temp = flame_model.v_template
        y_min, y_max = v_temp[:, 1].min(), v_temp[:, 1].max()
        y_norm = (v_temp[:, 1] - y_min) / (y_max - y_min + 1e-8)

        # Smooth transition: hard zero below 0.20, linear ramp 0.20→0.32
        pinning_mask = np.ones(N_VERTS, dtype=np.float32)
        pinning_mask[y_norm <= 0.20] = 0.0
        transition = (y_norm > 0.20) & (y_norm < 0.32)
        t = (y_norm[transition] - 0.20) / 0.12
        pinning_mask[transition] = 3 * t**2 - 2 * t**3  # Hermite C1 blend
        self.register_buffer('pinning_mask', torch.from_numpy(pinning_mask).unsqueeze(-1))

        # Input projection: vertex coords (3) + identity features (broadcast)
        vertex_input_dim = 3 + identity_dim

        self.input_proj = nn.Linear(vertex_input_dim, hidden_dim)

        # Graph convolution stack
        self.gc_layers = nn.ModuleList([
            GraphConvBlock(hidden_dim, hidden_dim, N_VERTS)
            for _ in range(n_layers)
        ])

        # Output head: predict (dx, dy, dz) per vertex
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # Initialize output to near-zero (residual should start small)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        flame_vertices: torch.Tensor,
        identity_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict macro-shape residual displacement.

        Args:
            flame_vertices: (B, 5023, 3) FLAME-decoded vertices
            identity_features: (B, 512) multi-view identity embedding

        Returns:
            corrected_vertices: (B, 5023, 3) = flame_vertices + ΔV_residual
            delta_v: (B, 5023, 3) the raw residual displacement
        """
        B = flame_vertices.shape[0]

        # Broadcast identity features to each vertex
        id_expanded = identity_features.unsqueeze(1).expand(B, self.n_verts, -1)

        # Concatenate per-vertex position + global identity
        vertex_input = torch.cat([flame_vertices, id_expanded], dim=-1)

        # Input projection
        h = self.input_proj(vertex_input)

        # Graph convolution layers
        for gc in self.gc_layers:
            h = gc(h, self.adj_matrix)

        # Predict per-vertex displacement
        delta_v = self.output_head(h)  # (B, 5023, 3)

        # Clamp displacement magnitude
        delta_v = torch.clamp(delta_v, -self.max_displacement, self.max_displacement)

        # Apply collar pinning mask (strict zero at neck boundary)
        delta_v = delta_v * self.pinning_mask

        corrected_vertices = flame_vertices + delta_v

        return corrected_vertices, delta_v

    def compute_laplacian_loss(self, delta_v: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness regularization.

        Penalizes high-frequency vertex displacement spiking to ensure
        the residual correction is organically smooth.

        Args:
            delta_v: (B, 5023, 3) displacement vectors

        Returns:
            scalar loss: ||L · ΔV||²
        """
        # L @ delta_v for each spatial dim
        Lv = torch.matmul(self.laplacian, delta_v)  # (B, 5023, 3)
        return torch.mean(Lv ** 2)

    def compute_collar_violation(self, delta_v: torch.Tensor) -> torch.Tensor:
        """
        Verify collar pinning contract is maintained.

        Returns max absolute displacement in the collar region (should be 0.0).
        """
        collar_mask = (self.pinning_mask.squeeze(-1) == 0.0)
        if collar_mask.any():
            collar_disp = torch.abs(delta_v[:, collar_mask]).max()
            return collar_disp
        return torch.tensor(0.0)


# Alias for concise import
MacroShapeResidualNet = MacroShapeResidualNetwork

