"""
FLAME 2020 / 2023 head-model decoder (pure NumPy).

Fixes vs. the previous version:
  1. There is NO 'exprdirs' key in generic_model.pkl. shapedirs is (5023, 3, 400):
     [..., :300] = identity, [..., 300:400] = expression.
  2. The `find_class -> np.array` chumpy trick does not work. pickle's NEWOBJ
     opcode requires a *type*; np.array is a builtin function, so you get
     "UnpicklingError: NEWOBJ class argument must be a type". Two working
     strategies are implemented below.
  3. FLAME is in METRES, not millimetres. v_template spans roughly -0.13..0.13.
  4. decode() silently ignored `theta`. It now either applies linear blend
     skinning properly or raises, instead of returning a mesh that quietly
     has no jaw pose.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# FLAME constants
N_VERTS = 5023
N_FACES = 9976
N_SHAPE = 300
N_EXPR = 100
N_JOINTS = 5  # global, neck, jaw, left eye, right eye


# ---------------------------------------------------------------------------
# chumpy handling
# ---------------------------------------------------------------------------

def _install_numpy_aliases() -> None:
    """chumpy 0.70 imports np.bool/np.int/np.float/... which numpy removed in 1.24.
    Re-add them as aliases so `import chumpy` succeeds. Harmless if unused."""
    for name, target in (("bool", bool), ("int", int), ("float", float),
                         ("complex", complex), ("object", object),
                         ("str", str), ("unicode", str)):
        if name not in np.__dict__:
            setattr(np, name, target)


class _ChumpyStub(np.ndarray):
    """Fallback stand-in for chumpy.Ch when chumpy itself cannot be imported.

    Ch keeps its numeric payload in instance state under 'x'. We capture the
    state dict and expose it via `_payload` for `_as_array` to unwrap.
    """

    def __new__(cls, *args, **kwargs):
        return np.zeros(0).view(cls)

    def __setstate__(self, state):
        if isinstance(state, dict):
            object.__setattr__(self, "_payload", state)

    def __array_finalize__(self, obj):
        if obj is not None and not hasattr(self, "_payload"):
            object.__setattr__(self, "_payload", {})


class _FlameUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("chumpy"):
            return _ChumpyStub
        # Older pickles reference scipy.sparse submodules that moved in scipy>=1.8
        if module.startswith("scipy.sparse.") and name.endswith("_matrix"):
            import scipy.sparse as sp
            return getattr(sp, name, None) or super().find_class(module, name)
        return super().find_class(module, name)


def _as_array(x, dtype=np.float32) -> np.ndarray:
    """Coerce chumpy Ch / _ChumpyStub / scipy sparse / ndarray to a dense array."""
    if hasattr(x, "toarray"):                       # scipy sparse
        return np.asarray(x.toarray(), dtype=dtype)
    if isinstance(x, _ChumpyStub):
        payload = getattr(x, "_payload", {})
        for key in ("x", "r", "_data"):
            if key in payload:
                return np.asarray(payload[key], dtype=dtype)
        raise ValueError("Could not unwrap chumpy payload; install chumpy instead.")
    if hasattr(x, "r"):                             # real chumpy.Ch
        return np.asarray(x.r, dtype=dtype)
    return np.asarray(x, dtype=dtype)


def load_flame_pickle(model_path: str | Path) -> dict:
    """Load generic_model.pkl. Prefers real chumpy; falls back to the stub."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"FLAME model not found at: {model_path}")

    _install_numpy_aliases()
    try:
        import chumpy  # noqa: F401  (registers the real classes for pickle)
        with open(model_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception:
        with open(model_path, "rb") as f:
            return _FlameUnpickler(f, encoding="latin1").load()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FLAMEModel:
    """Pure-NumPy FLAME decoder.

    Coordinates are METRES (FLAME's native unit). Use `scale_to_mm=True` only
    if a downstream consumer explicitly needs millimetres — and then be
    consistent everywhere, including the displacement normalisation stats.
    """

    def __init__(self, model_path: str, scale_to_mm: bool = False):
        data = load_flame_pickle(model_path)

        self.unit_scale = 1000.0 if scale_to_mm else 1.0
        self.units = "mm" if scale_to_mm else "m"

        self.v_template = _as_array(data["v_template"]) * self.unit_scale       # (5023, 3)

        shapedirs = _as_array(data["shapedirs"])                                # (5023, 3, 400)
        if shapedirs.shape[-1] < N_SHAPE + N_EXPR:
            raise ValueError(
                f"shapedirs has {shapedirs.shape[-1]} components; expected >= 400. "
                "Are you loading a reduced/converted FLAME variant?"
            )
        self.shapedirs = shapedirs[:, :, :N_SHAPE] * self.unit_scale            # identity
        self.exprdirs = shapedirs[:, :, N_SHAPE:N_SHAPE + N_EXPR] * self.unit_scale  # expression

        self.posedirs = _as_array(data["posedirs"]) * self.unit_scale           # (5023, 3, 36)
        self.J_regressor = _as_array(data["J_regressor"])                       # (5, 5023)
        self.weights = _as_array(data["weights"])                               # (5023, 5)
        self.faces = np.asarray(_as_array(data["f"], dtype=np.int64), dtype=np.int32)
        self.kintree_table = np.asarray(_as_array(data["kintree_table"], dtype=np.int64), dtype=np.int32)

        self.parents = self.kintree_table[0].copy()
        self.parents[0] = -1

        self._self_check()

    def _self_check(self) -> None:
        assert self.v_template.shape == (N_VERTS, 3), \
            f"Expected ({N_VERTS}, 3) template, got {self.v_template.shape}"
        assert not np.isnan(self.v_template).any(), "NaN in FLAME template"
        assert self.faces.shape == (N_FACES, 3), \
            f"Expected ({N_FACES}, 3) faces, got {self.faces.shape}"
        assert self.shapedirs.shape == (N_VERTS, 3, N_SHAPE)
        assert self.exprdirs.shape == (N_VERTS, 3, N_EXPR)
        lo, hi = float(self.v_template.min()), float(self.v_template.max())
        expected = (-0.25, 0.25) if self.units == "m" else (-250.0, 250.0)
        assert expected[0] < lo and hi < expected[1], (
            f"Template range {lo:.4f}..{hi:.4f} {self.units} is outside the expected "
            f"{expected} — unit mismatch, check scale_to_mm."
        )

    # -- pose helpers --------------------------------------------------------

    @staticmethod
    def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
        """(J, 3) axis-angle -> (J, 3, 3) rotation matrices."""
        theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)
        safe = np.where(theta < 1e-8, 1.0, theta)
        k = axis_angle / safe
        K = np.zeros((axis_angle.shape[0], 3, 3), dtype=np.float32)
        K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
        K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
        eye = np.eye(3, dtype=np.float32)[None]
        s = np.sin(theta)[:, :, None]
        c = np.cos(theta)[:, :, None]
        R = eye + s * K + (1.0 - c) * (K @ K)
        return np.where(theta[:, :, None] < 1e-8, eye, R).astype(np.float32)

    def _lbs(self, v_posed: np.ndarray, rot_mats: np.ndarray) -> np.ndarray:
        J = np.einsum("jv,vc->jc", self.J_regressor, v_posed)           # (5, 3)
        A = np.zeros((N_JOINTS, 4, 4), dtype=np.float32)
        for i in range(N_JOINTS):
            parent = self.parents[i]
            local = np.eye(4, dtype=np.float32)
            local[:3, :3] = rot_mats[i]
            local[:3, 3] = J[i] - (J[parent] if parent >= 0 else 0.0)
            A[i] = local if parent < 0 else A[parent] @ local
        # remove the rest-pose offset so a zero pose is the identity transform
        for i in range(N_JOINTS):
            rest = np.concatenate([J[i], [0.0]]).astype(np.float32)
            A[i, :, 3] -= A[i] @ rest
        T = np.einsum("vj,jmn->vmn", self.weights, A)                    # (5023, 4, 4)
        v_h = np.concatenate([v_posed, np.ones((N_VERTS, 1), np.float32)], axis=1)
        return np.einsum("vmn,vn->vm", T, v_h)[:, :3].astype(np.float32)

    # -- decoding ------------------------------------------------------------

    @staticmethod
    def _fit(vec, dim: int) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).ravel()[:dim]
        return np.pad(v, (0, dim - len(v))) if len(v) < dim else v

    def decode(
        self,
        beta: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        apply_pose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        beta  : (300,) identity coefficients
        psi   : (100,) expression coefficients
        theta : (15,) axis-angle pose = [global(3), neck(3), jaw(3), eyeL(3), eyeR(3)]

        Returns (vertices (5023,3) float32, faces (9976,3) int32).
        """
        v = self.v_template.copy()

        if beta is not None:
            v += np.einsum("ijk,k->ij", self.shapedirs, self._fit(beta, N_SHAPE))
        if psi is not None:
            v += np.einsum("ijk,k->ij", self.exprdirs, self._fit(psi, N_EXPR))

        if theta is None or not apply_pose:
            return v.astype(np.float32), self.faces

        t = self._fit(theta, N_JOINTS * 3).reshape(N_JOINTS, 3)
        rot = self._rodrigues(t)

        # pose-corrective blendshapes: (R_1..R_4 - I) flattened -> 36 coefficients
        pose_feat = (rot[1:] - np.eye(3, dtype=np.float32)[None]).reshape(-1)
        v = v + np.einsum("ijk,k->ij", self.posedirs, pose_feat)

        return self._lbs(v, rot), self.faces

    def decode_neutral(self, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Identity-only canonical base mesh (psi = 0, theta = 0).

        This is the mesh that must be exported, and the mesh the Stage-3
        conditioning maps are rasterised from. Never bake expression in here.
        """
        return self.decode(beta=beta, psi=None, theta=None)

    def vertex_normals(self, vertices: np.ndarray) -> np.ndarray:
        """Area-weighted vertex normals, vectorised (the per-face Python loop in
        build_uv_displacement_dataset.py is ~200x slower than this)."""
        v0, v1, v2 = (vertices[self.faces[:, i]] for i in range(3))
        fn = np.cross(v1 - v0, v2 - v0)                 # unnormalised == area-weighted
        vn = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(vn, self.faces[:, i], fn)
        norm = np.linalg.norm(vn, axis=1, keepdims=True)
        safe_norm = np.where(norm > 1e-12, norm, 1.0)
        return np.where(norm > 1e-12, vn / safe_norm, np.array([0.0, 0.0, 1.0], dtype=vertices.dtype))


if __name__ == "__main__":
    import sys
    m = FLAMEModel(sys.argv[1])
    v, f = m.decode_neutral(np.zeros(N_SHAPE, np.float32))
    print(f"verts={v.shape} faces={f.shape} nan={np.isnan(v).any()} "
          f"range={v.min():.4f}..{v.max():.4f} {m.units}")
    vp, _ = m.decode(np.zeros(N_SHAPE), np.zeros(N_EXPR), np.zeros(15))
    print(f"zero-pose LBS max deviation from neutral: {np.abs(vp - v).max():.2e} (should be ~0)")