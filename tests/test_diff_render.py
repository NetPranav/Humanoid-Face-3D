import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.utils.flame_model import N_VERTS
from src.stage1_identity.diff_render import (
    SoftSilhouetteRenderer,
    DifferentiableRenderLoss,
)


class MockFLAME:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.v_template = rng.uniform(-0.12, 0.12, (N_VERTS, 3)).astype(np.float32)
        faces = []
        for i in range(N_VERTS - 2):
            faces.append([i, i + 1, i + 2])
        faces.append([N_VERTS - 2, N_VERTS - 1, 0])
        faces.append([N_VERTS - 1, 0, 1])
        self.faces = np.array(faces, dtype=np.int64)


class TestDiffRender(unittest.TestCase):
    def setUp(self):
        self.flame = MockFLAME()

    @unittest.skipIf(torch is None, "PyTorch required for differentiable render tests")
    def test_soft_silhouette_renderer_shape(self):
        """Test that SoftSilhouetteRenderer outputs (B, 1, H, W) binary masks."""
        renderer = SoftSilhouetteRenderer(image_size=128, device='cpu')
        B = 2
        verts = torch.from_numpy(self.flame.v_template).float().unsqueeze(0).expand(B, -1, -1)
        faces = torch.from_numpy(self.flame.faces).long()

        sil = renderer.render_silhouette(verts, faces)
        self.assertEqual(sil.shape, (B, 1, 128, 128))
        self.assertTrue((sil >= 0.0).all())
        self.assertTrue((sil <= 1.0).all())

    @unittest.skipIf(torch is None, "PyTorch required for differentiable render tests")
    def test_silhouette_iou_loss(self):
        """Test silhouette IoU loss correctly computes 1 - IoU."""
        loss_mod = DifferentiableRenderLoss(device='cpu')
        B, C, H, W = 2, 1, 64, 64

        mask_a = torch.ones(B, C, H, W)
        mask_b = torch.ones(B, C, H, W)
        loss = loss_mod.silhouette_iou_loss(mask_a, mask_b)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=4)

        mask_c = torch.zeros(B, C, H, W)
        mask_c[:, :, :32, :] = 1.0
        mask_d = torch.zeros(B, C, H, W)
        mask_d[:, :, 32:, :] = 1.0
        loss_disjoint = loss_mod.silhouette_iou_loss(mask_c, mask_d)
        self.assertAlmostEqual(float(loss_disjoint.item()), 1.0, places=3)

    @unittest.skipIf(torch is None, "PyTorch required for differentiable render tests")
    def test_landmark_reprojection_loss(self):
        """Test landmark reprojection error calculation."""
        loss_mod = DifferentiableRenderLoss(device='cpu')
        B = 1
        verts = torch.from_numpy(self.flame.v_template).float().unsqueeze(0)
        lmk_indices = torch.tensor([0, 10, 20, 30, 40], dtype=torch.long)

        lmks_2d = verts[:, lmk_indices, :2]
        loss_zero = loss_mod.landmark_reprojection_loss(verts, lmks_2d, lmk_indices)
        self.assertAlmostEqual(float(loss_zero.item()), 0.0, places=4)

        lmks_perturbed = lmks_2d + 0.1
        loss_pert = loss_mod.landmark_reprojection_loss(verts, lmks_perturbed, lmk_indices)
        self.assertGreater(float(loss_pert.item()), 0.0)


if __name__ == '__main__':
    unittest.main()
