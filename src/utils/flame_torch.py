"""
Differentiable FLAME layer (PyTorch) sharing the arrays of src/utils/flame_model.FLAMEModel.

Pose vector layout matches FLAME: [global(3), neck(3), jaw(3), eye_L(3), eye_R(3)].
Coordinates are metres, FLAME frame: +y up, +z out of the face.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.utils.flame_model import FLAMEModel, N_EXPR, N_JOINTS, N_SHAPE


def batch_rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3)."""
    angle = torch.linalg.norm(aa + 1e-8, dim=-1, keepdim=True)
    k = aa / angle
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    K = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    s = torch.sin(angle)[..., None]
    c = torch.cos(angle)[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


class FLAMETorch(torch.nn.Module):
    def __init__(self, flame: FLAMEModel):
        super().__init__()
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.register_buffer("v_template", f32(flame.v_template))
        self.register_buffer("shapedirs", f32(flame.shapedirs))       # (V,3,300)
        self.register_buffer("exprdirs", f32(flame.exprdirs))         # (V,3,100)
        self.register_buffer("posedirs", f32(flame.posedirs))         # (V,3,36)
        self.register_buffer("J_regressor", f32(flame.J_regressor))   # (5,V)
        self.register_buffer("weights", f32(flame.weights))           # (V,5)
        self.register_buffer("faces", torch.as_tensor(flame.faces, dtype=torch.long))
        self.parents = [int(p) for p in flame.parents]

    def forward(
        self,
        beta: Optional[torch.Tensor] = None,
        psi: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (V, 3) vertices for a single subject (no batch dim, kept simple on purpose)."""
        v = self.v_template
        if beta is not None:
            b = beta.reshape(-1)
            v = v + torch.einsum("vck,k->vc", self.shapedirs[..., : b.numel()], b)
        if psi is not None:
            e = psi.reshape(-1)
            v = v + torch.einsum("vck,k->vc", self.exprdirs[..., : e.numel()], e)
        if pose is None:
            return v

        rot = batch_rodrigues(pose.reshape(N_JOINTS, 3))                 # (5,3,3)
        eye = torch.eye(3, dtype=v.dtype, device=v.device)
        pose_feat = (rot[1:] - eye).reshape(-1)
        v = v + torch.einsum("vck,k->vc", self.posedirs, pose_feat)

        J = self.J_regressor @ v                                          # (5,3)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=v.dtype, device=v.device)
        A = []
        for i in range(N_JOINTS):
            p = self.parents[i]
            rel = J[i] - (J[p] if p >= 0 else torch.zeros_like(J[i]))
            T = torch.cat([torch.cat([rot[i], rel[:, None]], 1), bottom], 0)
            A.append(T if p < 0 else A[p] @ T)
        A = torch.stack(A)                                                # (5,4,4)
        rest = torch.cat([J, torch.zeros(N_JOINTS, 1, dtype=v.dtype, device=v.device)], 1)
        corr = torch.einsum("jmn,jn->jm", A[:, :3, :], rest)              # (5,3)
        A = torch.cat([A[:, :3, :3], (A[:, :3, 3] - corr)[..., None]], 2)  # (5,3,4)
        A = torch.cat([A, bottom.expand(N_JOINTS, 1, 4)], 1)
        T = torch.einsum("vj,jmn->vmn", self.weights, A)
        vh = torch.cat([v, torch.ones(v.shape[0], 1, dtype=v.dtype, device=v.device)], 1)
        return torch.einsum("vmn,vn->vm", T, vh)[:, :3]


def load_flame(model_path: Optional[Union[str, Path]] = None) -> FLAMEModel:
    """Loads FLAME from data/flame_model/generic_model.pkl (see scripts/extract_flame_from_mica.py)."""
    root = Path(__file__).resolve().parents[2]
    path = Path(model_path) if model_path else root / "data" / "flame_model" / "generic_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"FLAME model not found at {path}. Create it from the MICA checkpoint with:\n"
            "  python3 scripts/extract_flame_from_mica.py --mica models_cache/mica/mica.tar"
        )
    return FLAMEModel(str(path))


__all__ = ["FLAMETorch", "batch_rodrigues", "load_flame", "N_SHAPE", "N_EXPR"]
