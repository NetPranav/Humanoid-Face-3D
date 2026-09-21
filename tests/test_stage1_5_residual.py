import unittest
import numpy as np
import tempfile
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from src.utils.flame_model import N_VERTS, N_SHAPE
from src.stage1_5_residual.residual_net import (
    MacroShapeResidualNet,
    MacroShapeResidualNetwork,
    build_adjacency_from_faces,
    build_laplacian_matrix,
)
from src.stage1_5_residual.trainer import ResidualScanDataset


class MockFLAME:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.v_template = rng.uniform(-0.12, 0.12, (N_VERTS, 3)).astype(np.float32)
        self.shapedirs = rng.normal(0, 0.001, (N_VERTS, 3, N_SHAPE)).astype(np.float32)
        # Connected ring topology so every vertex has neighbors
        faces = []
        for i in range(N_VERTS - 2):
            faces.append([i, i + 1, i + 2])
        faces.append([N_VERTS - 2, N_VERTS - 1, 0])
        faces.append([N_VERTS - 1, 0, 1])
        self.faces = np.array(faces, dtype=np.int64)


class TestStage15Residual(unittest.TestCase):
    def setUp(self):
        self.flame = MockFLAME()

    def test_adjacency_and_laplacian_construction(self):
        """Test adjacency list and Laplacian matrix construction on mesh faces."""
        faces = self.flame.faces
        n_verts = N_VERTS

        adj = build_adjacency_from_faces(faces, n_verts)
        self.assertEqual(len(adj), n_verts)
        # Every vertex should have neighbors
        self.assertGreater(len(adj[0]), 0)

        L = build_laplacian_matrix(faces, n_verts)
        self.assertEqual(L.shape, (n_verts, n_verts))
        # Laplacian row sum should be zero
        row_sums = np.abs(np.sum(L, axis=1))
        self.assertLess(np.max(row_sums), 1e-4)

    @unittest.skipIf(torch is None, "PyTorch required for neural network tests")
    def test_residual_net_forward_and_shapes(self):
        """Test that MacroShapeResidualNet returns expected shapes and preserves base topology."""
        model = MacroShapeResidualNet(flame_model=self.flame)
        model.eval()

        B = 2
        flame_v = torch.from_numpy(self.flame.v_template).float().unsqueeze(0).expand(B, -1, -1)
        id_feat = torch.randn(B, 512)

        with torch.no_grad():
            corr_v, delta_v = model(flame_v, id_feat)

        self.assertEqual(corr_v.shape, (B, N_VERTS, 3))
        self.assertEqual(delta_v.shape, (B, N_VERTS, 3))

    @unittest.skipIf(torch is None, "PyTorch required for neural network tests")
    def test_neck_collar_pinning_contract(self):
        """CRITICAL: Test that collar vertices strictly maintain delta_v == 0 (Neck Seam Contract)."""
        model = MacroShapeResidualNet(flame_model=self.flame)
        with torch.no_grad():
            for p in model.output_head.parameters():
                p.fill_(5.0)

        B = 2
        flame_v = torch.from_numpy(self.flame.v_template).float().unsqueeze(0).expand(B, -1, -1)
        id_feat = torch.randn(B, 512)

        _, delta_v = model(flame_v, id_feat)

        collar_violation = model.compute_collar_violation(delta_v)
        self.assertEqual(float(collar_violation.item()), 0.0)

    @unittest.skipIf(torch is None, "PyTorch required for neural network tests")
    def test_laplacian_smoothness_loss(self):
        """Test Laplacian loss penalizes unsmooth displacements."""
        model = MacroShapeResidualNet(flame_model=self.flame)
        B = 1
        smooth_disp = torch.ones(B, N_VERTS, 3) * 2.0
        loss_smooth = model.compute_laplacian_loss(smooth_disp)
        self.assertLess(float(loss_smooth.item()), 1e-4)

        noisy_disp = torch.randn(B, N_VERTS, 3) * 2.0
        loss_noisy = model.compute_laplacian_loss(noisy_disp)
        self.assertGreater(float(loss_noisy.item()), float(loss_smooth.item()))

    @unittest.skipIf(torch is None, "PyTorch required for dataset tests")
    def test_residual_scan_dataset_stratification(self):
        """Test ResidualScanDataset enforces subject-level stratification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            for s in range(6):
                for v in range(2):
                    f = tmppath / f"subj{s:02d}_view{v}.npz"
                    np.savez(
                        f,
                        flame_vertices=np.zeros((N_VERTS, 3), dtype=np.float32),
                        feature=np.zeros(512, dtype=np.float32),
                        gt_vertices=np.zeros((N_VERTS, 3), dtype=np.float32)
                    )

            train_ds = ResidualScanDataset(str(tmppath), split='train', val_ratio=0.33, seed=42)
            val_ds = ResidualScanDataset(str(tmppath), split='val', val_ratio=0.33, seed=42)

            train_subjs = {Path(p).stem.split('_')[0] for p in train_ds.samples}
            val_subjs = {Path(p).stem.split('_')[0] for p in val_ds.samples}

            self.assertEqual(len(train_subjs.intersection(val_subjs)), 0)
            self.assertGreater(len(train_subjs), 0)
            self.assertGreater(len(val_subjs), 0)


if __name__ == '__main__':
    unittest.main()
