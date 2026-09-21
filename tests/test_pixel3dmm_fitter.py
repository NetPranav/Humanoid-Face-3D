import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.utils.flame_model import N_VERTS, N_SHAPE
from src.stage1_identity.pixel3dmm_fitter import (
    PixelNormalPredictor,
    PixelUVPredictor,
    DenseFLAMEFitter,
)


class MockFLAME:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.v_template = rng.uniform(-0.12, 0.12, (N_VERTS, 3)).astype(np.float32)
        self.shapedirs = rng.normal(0, 0.001, (N_VERTS, 3, N_SHAPE)).astype(np.float32)
        faces = []
        for i in range(N_VERTS - 2):
            faces.append([i, i + 1, i + 2])
        faces.append([N_VERTS - 2, N_VERTS - 1, 0])
        faces.append([N_VERTS - 1, 0, 1])
        self.faces = np.array(faces, dtype=np.int64)


class TestPixel3DMMFitter(unittest.TestCase):
    def setUp(self):
        self.flame = MockFLAME()

    @unittest.skipIf(torch is None, "PyTorch required for neural network tests")
    def test_pixel_normal_decoder_shapes(self):
        """Test PixelNormalPredictor decoder produces 3-channel unit normal vectors."""
        predictor = PixelNormalPredictor()
        B, D, H, W = 1, 384, 128, 128
        spatial_features = torch.randn(B, D, H, W)
        raw_normals = predictor.decoder(spatial_features)
        self.assertEqual(raw_normals.shape, (B, 3, H, W))

        unit_normals = torch.nn.functional.normalize(raw_normals, dim=1, eps=1e-6)
        norms = torch.norm(unit_normals, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    @unittest.skipIf(torch is None, "PyTorch required for neural network tests")
    def test_pixel_uv_decoder_shapes(self):
        """Test PixelUVPredictor decoder produces 2-channel coordinates bounded in [0, 1]."""
        predictor = PixelUVPredictor()
        B, D, H, W = 1, 384, 128, 128
        spatial_features = torch.randn(B, D, H, W)
        uv_coords = predictor.decoder(spatial_features)
        self.assertEqual(uv_coords.shape, (B, 2, H, W))
        self.assertTrue((uv_coords >= 0.0).all())
        self.assertTrue((uv_coords <= 1.0).all())

    @unittest.skipIf(torch is None, "PyTorch required for dense fitting tests")
    def test_dense_flame_fitter_forward_and_loss(self):
        """Test DenseFLAMEFitter differentiable forward and normal consistency computation."""
        fitter = DenseFLAMEFitter(
            flame_model=self.flame,
            device='cpu',
            n_iterations=5,
            lr_beta=0.01
        )

        beta = torch.zeros(N_SHAPE, requires_grad=True)
        verts = fitter._flame_forward(beta)
        self.assertEqual(verts.shape, (N_VERTS, 3))

        normals = fitter._compute_vertex_normals(verts, fitter.faces)
        self.assertEqual(normals.shape, (N_VERTS, 3))
        norms = torch.norm(normals, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    @unittest.skipIf(torch is None, "PyTorch required for dense fitting tests")
    def test_dense_flame_fitter_fit_execution(self):
        """Test that DenseFLAMEFitter.fit executes and returns valid updated beta."""
        fitter = DenseFLAMEFitter(
            flame_model=self.flame,
            device='cpu',
            n_iterations=3,
            lr_beta=0.01
        )
        mock_img = np.zeros((256, 256, 3), dtype=np.uint8)
        init_beta = np.ones(N_SHAPE, dtype=np.float32) * 0.1

        res = fitter.fit(mock_img, initial_beta=init_beta)
        self.assertIn('beta', res)
        self.assertIn('vertices', res)
        self.assertIn('normals', res)
        self.assertEqual(res['beta'].shape, (N_SHAPE,))
        self.assertEqual(res['vertices'].shape, (N_VERTS, 3))


if __name__ == '__main__':
    unittest.main()
