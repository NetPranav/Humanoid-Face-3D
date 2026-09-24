"""
Unit tests for Stage 7: AI Delighting + UV Inpainting.
"""
import unittest
import numpy as np


class TestUVInpainter(unittest.TestCase):
    """Tests for the procedural Gaussian dilation inpainter."""

    def setUp(self):
        from src.stage7_delight.delight_net import UVInpainter
        self.inpainter = UVInpainter(iterations=20, kernel_size=5)

    def test_preserves_original_pixels(self):
        """Original pixel values must be preserved exactly where mask is valid."""
        texture = np.random.rand(64, 64, 3).astype(np.float32) * 255
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        # Set some pixels as missing
        mask[20:30, 20:30] = 0

        original_valid = texture.copy()
        filled, filled_mask = self.inpainter.inpaint(texture, mask)

        # Check that valid pixels are exactly preserved
        valid = mask > 127
        np.testing.assert_array_almost_equal(
            filled[valid], original_valid[valid], decimal=5,
            err_msg="Inpainter modified original valid pixels!"
        )

    def test_fills_missing_regions(self):
        """After inpainting, previously-zero regions should have data."""
        texture = np.full((64, 64, 3), 128.0, dtype=np.float32)
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        mask[25:35, 25:35] = 0  # 10x10 hole
        texture[25:35, 25:35] = 0  # Black hole

        filled, filled_mask = self.inpainter.inpaint(texture, mask)

        # The hole should now have non-zero values
        hole_mean = filled[25:35, 25:35].mean()
        self.assertGreater(hole_mean, 50.0, "Inpainter failed to fill the hole")

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        texture = np.random.rand(128, 128, 3).astype(np.float32) * 255
        mask = np.ones((128, 128), dtype=np.uint8) * 255
        mask[40:80, 40:80] = 0

        filled, filled_mask = self.inpainter.inpaint(texture, mask)
        self.assertEqual(filled.shape, texture.shape)
        self.assertEqual(filled_mask.shape, mask.shape)

    def test_fully_valid_mask_unchanged(self):
        """If mask is all valid, output should equal input exactly."""
        texture = np.random.rand(32, 32, 3).astype(np.float32) * 255
        mask = np.ones((32, 32), dtype=np.uint8) * 255

        filled, filled_mask = self.inpainter.inpaint(texture, mask)
        np.testing.assert_array_almost_equal(filled, texture, decimal=5)

    def test_uint8_input_produces_uint8_output(self):
        """uint8 input should produce uint8 output."""
        texture = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        mask[10:20, 10:20] = 0

        filled, _ = self.inpainter.inpaint(texture, mask)
        self.assertEqual(filled.dtype, np.uint8)


class TestDelightUNet(unittest.TestCase):
    """Tests for the DelightUNet architecture."""

    def test_import_without_torch(self):
        """Module should be importable even if torch is not available."""
        # This test verifies the import guard works
        from src.stage7_delight.delight_net import HAS_TORCH
        # Just check the flag exists — actual torch availability varies
        self.assertIsInstance(HAS_TORCH, bool)

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('torch') is not None,
        "PyTorch not available"
    )
    def test_forward_pass_shape(self):
        """U-Net forward pass should produce correct output shape."""
        import torch
        from src.stage7_delight.delight_net import DelightUNet

        model = DelightUNet(in_channels=6, out_channels=3, base_channels=16)
        model.eval()

        # Small input for speed (must be divisible by 64)
        x = torch.randn(1, 6, 64, 64)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 3, 64, 64))

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('torch') is not None,
        "PyTorch not available"
    )
    def test_output_bounded_0_1(self):
        """Output should be in [0, 1] due to Sigmoid activation."""
        import torch
        from src.stage7_delight.delight_net import DelightUNet

        model = DelightUNet(in_channels=6, out_channels=3, base_channels=16)
        model.eval()

        x = torch.randn(1, 6, 64, 64)
        with torch.no_grad():
            out = model(x)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)


class TestDelightingPipeline(unittest.TestCase):
    """Tests for the full delighting pipeline orchestrator."""

    def test_gamma_fallback_produces_valid_output(self):
        """Gamma-correction fallback should produce valid albedo."""
        from src.stage7_delight.delight_net import DelightingPipeline

        pipeline = DelightingPipeline(
            delight_checkpoint=None,
            inpaint_method='procedural',
            inpaint_iterations=10,
        )

        texture = np.random.rand(64, 64, 3).astype(np.float32) * 255
        mask = np.ones((64, 64), dtype=np.uint8) * 255

        result = pipeline.process(texture, mask)

        self.assertIn('albedo_srgb', result)
        self.assertIn('albedo_linear', result)
        self.assertEqual(result['albedo_srgb'].shape, (64, 64, 3))
        self.assertEqual(result['albedo_srgb'].dtype, np.uint8)
        self.assertEqual(result['albedo_linear'].dtype, np.float32)

    def test_no_nan_in_output(self):
        """Output albedo should contain no NaN values."""
        from src.stage7_delight.delight_net import DelightingPipeline

        pipeline = DelightingPipeline(
            delight_checkpoint=None,
            inpaint_method='procedural',
        )

        texture = np.random.rand(32, 32, 3).astype(np.float32) * 255
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        mask[10:20, 10:20] = 0

        result = pipeline.process(texture, mask)

        self.assertFalse(np.any(np.isnan(result['albedo_linear'])),
                         "NaN values found in albedo output")
        self.assertFalse(np.any(np.isnan(result['albedo_srgb'].astype(np.float32))),
                         "NaN values found in sRGB output")

    def test_save_maps_creates_files(self):
        """save_maps should write albedo files to disk."""
        import tempfile
        from src.stage7_delight.delight_net import DelightingPipeline

        pipeline = DelightingPipeline(delight_checkpoint=None)
        result = {
            'albedo_srgb': (np.random.rand(32, 32, 3) * 255).astype(np.uint8),
            'albedo_linear': np.random.rand(32, 32, 3).astype(np.float32),
            'inpainted_mask': np.ones((32, 32), dtype=np.uint8) * 255,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = pipeline.save_maps(result, tmpdir)
            self.assertIn('albedo_diffuse', paths)
            from pathlib import Path
            self.assertTrue(Path(paths['albedo_diffuse']).exists())


if __name__ == '__main__':
    unittest.main()
