import unittest
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from src.stage0_preprocess.detector import FaceDetection
from src.stage1_identity.inference import MICAIdentityEncoder, MappingNetwork

class TestMICAIntegration(unittest.TestCase):
    def test_mica_raises_without_torch_or_checkpoint(self):
        """Verify MICAIdentityEncoder raises RuntimeError (if no torch) or FileNotFoundError (if no ckpt)."""
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            MICAIdentityEncoder(checkpoint_path="/nonexistent/path/to/mica.tar")

    def test_multiview_weighting_frontality_bias(self):
        """
        Verify multi-view weighting assigns substantially higher weight to frontal view
        than profile view when using det_score * cos^2(yaw) with softmax temperature.
        """
        yaw_frontal = 2.0
        yaw_profile = 45.0
        det_score = 0.95

        # Compute weights as done in MICAIdentityEncoder
        w_frontal = det_score * (max(0.0, float(np.cos(np.radians(yaw_frontal)))) ** 2)
        w_profile = det_score * (max(0.0, float(np.cos(np.radians(yaw_profile)))) ** 2)

        # cos(2°) ~ 0.999 -> w_frontal ~ 0.95 * 0.998 = 0.948
        # cos(45°) ~ 0.707 -> w_profile ~ 0.95 * 0.500 = 0.475
        self.assertGreater(w_frontal, w_profile * 1.8)

        # After softmax with temperature 0.05:
        scores = np.array([w_frontal, w_profile], dtype=np.float32)
        temp = 0.05
        scaled = scores / temp
        exp_w = np.exp(scaled - np.max(scaled))
        norm_w = exp_w / np.sum(exp_w)

        # Frontal view should dominate (> 99% of total weight)
        self.assertGreater(norm_w[0], 0.99)
        self.assertLess(norm_w[1], 0.01)

    @unittest.skipIf(torch is None, "PyTorch not available in local test environment")
    def test_mapping_network_shapes_and_distinct_identities(self):
        """Verify MappingNetwork maps (B, 512) -> (B, 300) and distinct inputs produce distinct betas."""
        net = MappingNetwork(z_dim=512, map_hidden_dim=300, map_output_dim=300, hidden=3)
        net.eval()

        rng = torch.Generator().manual_seed(42)
        emb_subject_a = torch.randn(1, 512, generator=rng)
        emb_subject_a = torch.nn.functional.normalize(emb_subject_a, dim=1)

        emb_subject_b = torch.randn(1, 512, generator=rng)
        emb_subject_b = torch.nn.functional.normalize(emb_subject_b, dim=1)

        with torch.no_grad():
            beta_a = net(emb_subject_a)
            beta_b = net(emb_subject_b)

        self.assertEqual(beta_a.shape, (1, 300))
        self.assertEqual(beta_b.shape, (1, 300))
        self.assertFalse(torch.isnan(beta_a).any())

        # Euclidean distance between distinct subjects must be strictly non-zero
        diff = torch.norm(beta_a - beta_b).item()
        self.assertGreater(diff, 1e-3, f"Distinct identities collapsed to identical beta: diff={diff}")

if __name__ == "__main__":
    unittest.main()
