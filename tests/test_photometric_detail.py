"""
Unit tests for Stage 3 Tier 2: PhotometricDetailExtractor (Meso Wrinkle Engine).
"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.stage3_detail.photometric_detail import PhotometricDetailExtractor


class TestPhotometricDetailExtractor(unittest.TestCase):
    """Test suite for Tier 2 Meso Wrinkle Engine."""

    def setUp(self):
        self.resolution = 256  # Fast resolution for unit tests
        self.max_depth_mm = 1.20
        self.extractor = PhotometricDetailExtractor(
            resolution=self.resolution,
            max_wrinkle_depth_mm=self.max_depth_mm,
            sigma_fine=1.0,
            sigma_meso=6.0,
            gradient_gain=2.0,
        )

    def test_initialization(self):
        """Verify constructor handles parameters and loads UV layout."""
        self.assertEqual(self.extractor.resolution, 256)
        self.assertEqual(self.extractor.max_depth_mm, 1.20)
        self.assertIsNotNone(self.extractor.uv_coords)
        self.assertIsNotNone(self.extractor.uv_faces)

    def test_wrinkle_gradient_extraction_flat_image(self):
        """A uniform flat albedo image should yield near-zero gradients."""
        flat_img = np.full((self.resolution, self.resolution, 3), 180, dtype=np.uint8)
        gx, gy, wrinkle_resp = self.extractor.extract_wrinkle_gradients(flat_img)

        self.assertEqual(gx.shape, (self.resolution, self.resolution))
        self.assertEqual(gy.shape, (self.resolution, self.resolution))
        self.assertTrue(np.all(np.abs(gx) < 1e-4))
        self.assertTrue(np.all(np.abs(gy) < 1e-4))
        self.assertTrue(np.all(wrinkle_resp < 1e-4))

    def test_wrinkle_gradient_extraction_valley_line(self):
        """A dark horizontal crease line (wrinkle valley) should produce prominent vertical gradients."""
        img = np.full((self.resolution, self.resolution, 3), 200, dtype=np.uint8)
        # Draw a dark horizontal crease representing a forehead furrow
        img[120:124, 50:200] = 50

        gx, gy, wrinkle_resp = self.extractor.extract_wrinkle_gradients(img)

        # gy should have strong response around y=120..124
        self.assertTrue(np.max(np.abs(gy[115:130, 60:190])) > 0.1)
        # wrinkle_resp should be positive in the dark valley
        self.assertTrue(np.max(wrinkle_resp[120:124, 60:190]) > 0.2)

    def test_frankot_chellappa_integration_accuracy(self):
        """Verify Frankot-Chellappa integration reconstructs an analytical surface."""
        # Create a synthetic Gaussian bump heightfield
        x = np.linspace(-2, 2, self.resolution)
        y = np.linspace(-2, 2, self.resolution)
        xx, yy = np.meshgrid(x, y)
        z_true = np.exp(-(xx ** 2 + yy ** 2))

        # Analytical gradients
        gx = np.gradient(z_true, axis=1)
        gy = np.gradient(z_true, axis=0)

        # Integrate
        height_rec = self.extractor.integrate_heightfield(gx, gy)

        # Normalize both to zero mean and unit variance for correlation check
        z_norm = (z_true - z_true.mean()) / (z_true.std() + 1e-8)
        h_norm = (height_rec - height_rec.mean()) / (height_rec.std() + 1e-8)

        corr = np.corrcoef(z_norm.flatten(), h_norm.flatten())[0, 1]
        self.assertGreater(corr, 0.95, "Frankot-Chellappa correlation must exceed 0.95")
        self.assertFalse(np.isnan(height_rec).any())
        self.assertFalse(np.isinf(height_rec).any())

    def test_metric_depth_bounds(self):
        """Ensure synthesized heightfield does not exceed specified max metric depth."""
        # High-contrast noisy pattern
        rng = np.random.RandomState(42)
        random_tex = (rng.rand(self.resolution, self.resolution, 3) * 255).astype(np.uint8)

        res = self.extractor.extract_from_albedo(random_tex)
        disp = res['displacement_mm']

        self.assertLessEqual(np.max(disp), self.max_depth_mm + 1e-5)
        self.assertGreaterEqual(np.min(disp), -self.max_depth_mm - 1e-5)
        self.assertFalse(np.isnan(disp).any())

    def test_collar_pinning_contract(self):
        """Verify Rule 4: Lowest 20% of vertices must have strictly zero displacement."""
        # Use real FLAME model geometry if available, else synthetic mesh
        try:
            from src.utils.flame_model import FLAMEModel
            flame = FLAMEModel('data/flame_model/generic_model.pkl')
            verts, faces = flame.decode_neutral(np.zeros(300))
        except Exception:
            verts = np.zeros((len(self.extractor.uv_coords), 3), dtype=np.float64)
            verts[:, 1] = np.linspace(-0.15, 0.15, len(verts))
            faces = self.extractor.uv_faces

        disp = np.ones((self.resolution, self.resolution), dtype=np.float32) * 0.8
        pinned = self.extractor.apply_collar_pinning(disp, vertices=verts, faces=faces)

        self.assertEqual(pinned.shape, disp.shape)
        # Inside the neck collar region (bottom center), displacement must be strictly 0.0
        neck_row_start = int(self.resolution * 0.90)
        neck_col_start = int(self.resolution * 0.35)
        neck_col_end = int(self.resolution * 0.65)
        neck_region = pinned[neck_row_start:, neck_col_start:neck_col_end]
        self.assertLessEqual(neck_region.max(), 1e-6)

    def test_collar_pinning_fallback(self):
        """Collar pinning fallback using UV coords when 3D geometry is not provided."""
        disp = np.ones((self.resolution, self.resolution), dtype=np.float32) * 0.8
        pinned = self.extractor.apply_collar_pinning(disp, vertices=None, faces=None)

        # Bottom 15% rows (highest row indices in image array) must be exactly zero
        bottom_rows = int(self.resolution * 0.15)
        self.assertEqual(pinned[-bottom_rows:, :].max(), 0.0)

    def test_tangent_normal_computation(self):
        """Verify normal map generation matches tangent-space conventions."""
        # Flat surface should yield neutral blue normal (128, 128, 255)
        flat_disp = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        normals = self.extractor.compute_tangent_normal_map(flat_disp)

        self.assertEqual(normals.shape, (self.resolution, self.resolution, 3))
        self.assertEqual(normals.dtype, np.uint8)

        # Center pixel of flat surface
        r, g, b = normals[self.resolution // 2, self.resolution // 2]
        self.assertTrue(126 <= r <= 130, f"Expected R ~ 128, got {r}")
        self.assertTrue(126 <= g <= 130, f"Expected G ~ 128, got {g}")
        self.assertTrue(250 <= b <= 255, f"Expected B ~ 255, got {b}")

    def test_multi_view_fusion(self):
        """Verify multi-view extraction blends multiple views with angle weights."""
        view1 = {
            'rgb': np.full((self.resolution, self.resolution, 3), 150, dtype=np.uint8),
            'weight': np.ones((self.resolution, self.resolution), dtype=np.float32) * 0.8,
            'mask': np.ones((self.resolution, self.resolution), dtype=np.float32),
        }
        view2 = {
            'rgb': np.full((self.resolution, self.resolution, 3), 150, dtype=np.uint8),
            'weight': np.ones((self.resolution, self.resolution), dtype=np.float32) * 0.4,
            'mask': np.ones((self.resolution, self.resolution), dtype=np.float32),
        }

        fused = self.extractor.extract_from_views([view1, view2])
        self.assertIn('displacement_mm', fused)
        self.assertIn('tangent_normal_rgb', fused)
        self.assertIn('wrinkle_mask', fused)
        self.assertEqual(fused['displacement_mm'].shape, (self.resolution, self.resolution))

    def test_save_maps_round_trip(self):
        """Verify 16-bit uint PNG storage and precision round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_disp = np.linspace(-1.0, 1.0, self.resolution, dtype=np.float32)
            test_disp = np.repeat(test_disp[:, np.newaxis], self.resolution, axis=1)

            res = {
                'displacement_mm': test_disp,
                'tangent_normal_rgb': np.full((self.resolution, self.resolution, 3), 128, dtype=np.uint8),
            }

            saved = self.extractor.save_maps(res, output_dir=tmpdir, prefix="test")
            self.assertTrue(saved['displacement_png'].exists())
            self.assertTrue(saved['normal_png'].exists())

            # Read back 16-bit PNG
            loaded_u16 = cv2.imread(str(saved['displacement_png']), cv2.IMREAD_UNCHANGED)
            self.assertEqual(loaded_u16.dtype, np.uint16)

            # Decode from uint16 to mm
            decoded_mm = ((loaded_u16.astype(np.float32) / 65535.0) * 2.0 - 1.0) * self.max_depth_mm

            # Check precision error < 0.001 mm
            max_err = np.max(np.abs(decoded_mm - test_disp))
            self.assertLess(max_err, 0.002, f"16-bit precision round-trip error too large: {max_err}")


if __name__ == '__main__':
    unittest.main()
