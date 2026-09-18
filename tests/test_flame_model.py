import unittest
import tempfile
import pickle
import numpy as np
from pathlib import Path

from src.utils.flame_model import (
    FLAMEModel,
    _install_numpy_aliases,
    _as_array,
    _ChumpyStub,
    N_VERTS,
    N_FACES,
    N_SHAPE,
    N_EXPR,
    N_JOINTS,
)

class TestFLAMEModel(unittest.TestCase):
    def setUp(self):
        """Generates valid synthetic FLAME 2020/2023 pickle dictionary in a temp file."""
        rng = np.random.default_rng(42)

        v_template = rng.uniform(-0.12, 0.12, (N_VERTS, 3)).astype(np.float32)
        # shapedirs must be (5023, 3, 400) - 300 shape + 100 expr
        shapedirs = rng.normal(0, 0.001, (N_VERTS, 3, N_SHAPE + N_EXPR)).astype(np.float32)
        posedirs = rng.normal(0, 0.001, (N_VERTS, 3, 36)).astype(np.float32)

        # J_regressor: 5 joints from 5023 vertices
        J_reg = np.zeros((N_JOINTS, N_VERTS), dtype=np.float32)
        for j in range(N_JOINTS):
            J_reg[j, j * 100:(j + 1) * 100] = 1.0 / 100.0

        # Weights: skinning weights summing to 1
        weights = rng.uniform(0.01, 1.0, (N_VERTS, N_JOINTS)).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)

        # Faces: triangle indices
        faces = rng.integers(0, N_VERTS, (N_FACES, 3)).astype(np.int64)

        # Kinematic tree: parent relationships
        kintree_table = np.array([
            [-1, 0, 0, 0, 0],
            [0, 1, 2, 3, 4]
        ], dtype=np.int64)

        data = {
            'v_template': v_template,
            'shapedirs': shapedirs,
            'posedirs': posedirs,
            'J_regressor': J_reg,
            'weights': weights,
            'f': faces,
            'kintree_table': kintree_table,
        }

        self.temp_dir = tempfile.TemporaryDirectory()
        self.pkl_path = Path(self.temp_dir.name) / "generic_model.pkl"
        with open(self.pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_numpy_aliases(self):
        """Verify numpy compatibility shim re-adds deprecated aliases on NumPy 2.x."""
        _install_numpy_aliases()
        self.assertTrue(hasattr(np, "bool"))
        self.assertTrue(hasattr(np, "int"))
        self.assertTrue(hasattr(np, "float"))

    def test_chumpy_stub_unwrap(self):
        """Verify _as_array correctly unwraps _ChumpyStub payload."""
        stub = _ChumpyStub()
        stub._payload = {'x': np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        arr = _as_array(stub)
        self.assertIsInstance(arr, np.ndarray)
        self.assertTrue(np.allclose(arr, [1.0, 2.0, 3.0]))

    def test_flame_loader_init(self):
        """Verify FLAMEModel initializes with shapedirs unpacking and native metres."""
        flame = FLAMEModel(str(self.pkl_path), scale_to_mm=False)

        self.assertEqual(flame.v_template.shape, (N_VERTS, 3))
        self.assertEqual(flame.faces.shape, (N_FACES, 3))
        self.assertEqual(flame.shapedirs.shape, (N_VERTS, 3, N_SHAPE))
        self.assertEqual(flame.exprdirs.shape, (N_VERTS, 3, N_EXPR))
        self.assertEqual(flame.units, "m")
        self.assertFalse(np.isnan(flame.v_template).any())

    def test_flame_scale_to_mm(self):
        """Verify scale_to_mm converts dimensions from metres to millimetres."""
        flame_m = FLAMEModel(str(self.pkl_path), scale_to_mm=False)
        flame_mm = FLAMEModel(str(self.pkl_path), scale_to_mm=True)

        self.assertEqual(flame_mm.units, "mm")
        self.assertTrue(np.allclose(flame_mm.v_template, flame_m.v_template * 1000.0))

    def test_flame_decode_neutral(self):
        """Verify neutral mesh decoding with shape parameters."""
        flame = FLAMEModel(str(self.pkl_path))
        beta = np.ones(N_SHAPE, dtype=np.float32) * 0.5
        v_neutral, f = flame.decode_neutral(beta)

        self.assertEqual(v_neutral.shape, (N_VERTS, 3))
        self.assertEqual(f.shape, (N_FACES, 3))
        self.assertFalse(np.allclose(v_neutral, flame.v_template))

    def test_flame_zero_pose_lbs_identity(self):
        """Verify that zero pose (theta=0) in LBS reproduces neutral decoding."""
        flame = FLAMEModel(str(self.pkl_path))
        beta = np.zeros(N_SHAPE, dtype=np.float32)
        psi = np.zeros(N_EXPR, dtype=np.float32)
        theta_zero = np.zeros(15, dtype=np.float32)

        v_neutral, _ = flame.decode_neutral(beta)
        v_posed, _ = flame.decode(beta=beta, psi=psi, theta=theta_zero)

        diff = np.abs(v_posed - v_neutral).max()
        self.assertLess(diff, 1e-6, f"Zero-pose LBS deviation too large: {diff}")

    def test_flame_jaw_pose_moves_mesh(self):
        """Verify that articulating the jaw joint (theta[6] != 0) modifies mesh coordinates."""
        flame = FLAMEModel(str(self.pkl_path))
        beta = np.zeros(N_SHAPE, dtype=np.float32)
        psi = np.zeros(N_EXPR, dtype=np.float32)

        theta_jaw = np.zeros(15, dtype=np.float32)
        theta_jaw[6] = 0.25  # Jaw opening rotation in radians

        v_neutral, _ = flame.decode_neutral(beta)
        v_jaw, _ = flame.decode(beta=beta, psi=psi, theta=theta_jaw)

        diff = np.abs(v_jaw - v_neutral).max()
        self.assertGreater(diff, 1e-4, "Jaw articulation failed to move any vertices")

    def test_vertex_normals_vectorised(self):
        """Verify vectorised vertex normals are unit-length and free of NaNs."""
        flame = FLAMEModel(str(self.pkl_path))
        v, _ = flame.decode_neutral(np.zeros(N_SHAPE))
        normals = flame.vertex_normals(v)

        self.assertEqual(normals.shape, (N_VERTS, 3))
        self.assertFalse(np.isnan(normals).any())
        norms = np.linalg.norm(normals, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-4))

if __name__ == "__main__":
    unittest.main()
