"""
Unit tests for Stage 3 Tier 3: AnatomicalPoreSynthesizer (4K Micro-Detail Engine).
"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.stage3_detail.anatomical_pores import AnatomicalPoreSynthesizer
from src.utils.flame_model import FLAMEModel


class TestAnatomicalPoreSynthesizer(unittest.TestCase):
    """Test suite for Tier 3 4K Anatomical Pore Synthesis."""

    def setUp(self):
        self.resolution = 256  # Fast resolution for unit tests
        self.synthesizer = AnatomicalPoreSynthesizer(
            resolution=self.resolution,
            pore_density_scale=1.0,
            random_seed=42,
        )
        try:
            self.flame = FLAMEModel('data/flame_model/generic_model.pkl')
            self.verts, self.faces = self.flame.decode_neutral(np.zeros(300))
        except Exception:
            self.flame = None
            self.verts = None
            self.faces = None

    def test_initialization(self):
        """Verify synthesizer loads UV parameterization and sets attributes."""
        self.assertEqual(self.synthesizer.resolution, 256)
        self.assertIsNotNone(self.synthesizer.uv_coords)
        self.assertIsNotNone(self.synthesizer.uv_faces)

    def test_follicular_pore_profile(self):
        """Verify T-zone sebaceous follicles have negative pit depressions and positive annular rims."""
        t_pores = self.synthesizer.generate_follicular_pores(
            resolution=self.resolution,
            grid_dim=16,
            pore_scale=3.5,
            rim_weight=0.30,
            seed=42,
        )
        self.assertEqual(t_pores.shape, (self.resolution, self.resolution))
        # Minimum must be deep pit (~ -1.0)
        self.assertLess(t_pores.min(), -0.7)
        # Maximum must be raised annular rim (> 0.05)
        self.assertGreater(t_pores.max(), 0.05)
        self.assertFalse(np.isnan(t_pores).any())

    def test_anisotropic_cheek_grain(self):
        """Verify cheek micro-grain generates cellular boundaries with directional orientation."""
        grain = self.synthesizer.generate_anisotropic_micro_grain(
            resolution=self.resolution,
            grid_dim=32,
            stretch_factor=1.8,
            angle_deg=35.0,
            seed=101,
        )
        self.assertEqual(grain.shape, (self.resolution, self.resolution))
        self.assertTrue(-1.0 <= grain.min() <= 0.0)
        self.assertTrue(0.0 <= grain.max() <= 1.0)
        self.assertFalse(np.isnan(grain).any())

    def test_lip_striations_orientation(self):
        """Verify vermilion lip striations are predominantly oriented vertically."""
        lips = self.synthesizer.generate_lip_striations(
            resolution=self.resolution,
            frequency_x=32,
            frequency_y=6,
            seed=202,
        )
        self.assertEqual(lips.shape, (self.resolution, self.resolution))
        # Compute directional gradients: dx (horizontal derivative) should have higher variance than dy
        dx = np.abs(np.gradient(lips, axis=1))
        dy = np.abs(np.gradient(lips, axis=0))
        self.assertGreater(dx.mean(), dy.mean() * 1.5, "Lip striations must have stronger lateral gradients than vertical")

    def test_end_to_end_synthesis_outputs(self):
        """Verify full synthesis pipeline produces all required maps and bounded values."""
        res = self.synthesizer.synthesize(
            vertices=self.verts,
            faces=self.faces,
            resolution=self.resolution,
        )

        self.assertIn('displacement_mm', res)
        self.assertIn('tangent_normal_rgb', res)
        self.assertIn('pore_density_map', res)
        self.assertIn('zone_masks', res)

        disp = res['displacement_mm']
        self.assertEqual(disp.shape, (self.resolution, self.resolution))
        self.assertEqual(disp.dtype, np.float32)

        # Micro-displacement must be sub-millimeter (-0.5 mm to +0.5 mm)
        self.assertGreater(disp.min(), -0.6)
        self.assertLess(disp.max(), 0.6)
        self.assertFalse(np.isnan(disp).any())

        # Normal map must be (H, W, 3) uint8
        normals = res['tangent_normal_rgb']
        self.assertEqual(normals.shape, (self.resolution, self.resolution, 3))
        self.assertEqual(normals.dtype, np.uint8)

    def test_collar_boundary_pinning_rule4(self):
        """Verify System Invariant Rule 4: Collar boundary is strictly 0.000000 mm."""
        res = self.synthesizer.synthesize(
            vertices=self.verts,
            faces=self.faces,
            resolution=self.resolution,
        )
        disp = res['displacement_mm']
        masks = res['zone_masks']

        neck_pin = masks.get('neck_pinning', masks.get('neck_collar'))
        self.assertIsNotNone(neck_pin)

        # Everywhere where neck_pinning > 0.8, displacement must be strictly zero
        pinned_pixels = neck_pin > 0.8
        if np.any(pinned_pixels):
            self.assertEqual(disp[pinned_pixels].max(), 0.0)
            self.assertEqual(disp[pinned_pixels].min(), 0.0)

    def test_multi_resolution_support(self):
        """Verify synthesis scales to different resolutions (e.g. 512)."""
        res_512 = self.synthesizer.synthesize(
            vertices=self.verts,
            faces=self.faces,
            resolution=512,
        )
        self.assertEqual(res_512['displacement_mm'].shape, (512, 512))
        self.assertEqual(res_512['tangent_normal_rgb'].shape, (512, 512, 3))

    def test_save_maps_round_trip(self):
        """Verify 16-bit uint PNG storage and decoding precision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            res = self.synthesizer.synthesize(
                vertices=self.verts,
                faces=self.faces,
                resolution=self.resolution,
            )
            saved = self.synthesizer.save_maps(res, output_dir=tmpdir, prefix="test_micro", max_scale_mm=0.5)
            self.assertTrue(saved['displacement_png'].exists())
            self.assertTrue(saved['normal_png'].exists())

            # Read back 16-bit PNG
            u16 = cv2.imread(str(saved['displacement_png']), cv2.IMREAD_UNCHANGED)
            self.assertEqual(u16.dtype, np.uint16)

            # Decode to mm
            decoded_mm = ((u16.astype(np.float32) / 65535.0) * 2.0 - 1.0) * 0.5
            err = np.max(np.abs(decoded_mm - res['displacement_mm']))
            self.assertLess(err, 0.001, f"Round-trip decoding error too high: {err}")


if __name__ == '__main__':
    unittest.main()
