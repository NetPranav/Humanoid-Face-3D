"""
Unit tests for Step 3: MultiTierDetailFusion & PBR Material Coupling.
"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.stage3_detail.fusion import MultiTierDetailFusion
from src.stage8_pbr.material_stack import CavityMapGenerator


class TestPBRCoupling(unittest.TestCase):
    """Test suite for Multi-Tier Geometry & PBR Material Coupling."""

    def setUp(self):
        self.resolution = 256
        self.max_scale_mm = 5.0
        self.fusion = MultiTierDetailFusion(
            resolution=self.resolution,
            max_scale_mm=self.max_scale_mm,
            cavity_strength=0.40,
            cavity_blur_sigma=0.8,
        )

        # Create synthetic masks
        self.zone_masks = {
            'valid': np.ones((self.resolution, self.resolution), dtype=np.float32),
            't_zone': np.zeros((self.resolution, self.resolution), dtype=np.float32),
            'cheeks': np.zeros((self.resolution, self.resolution), dtype=np.float32),
            'lips': np.zeros((self.resolution, self.resolution), dtype=np.float32),
            'periorbital': np.zeros((self.resolution, self.resolution), dtype=np.float32),
        }
        # T-zone in center column
        self.zone_masks['t_zone'][:, 100:156] = 1.0
        # Cheeks in lateral regions
        self.zone_masks['cheeks'][:, 20:80] = 1.0
        self.zone_masks['cheeks'][:, 176:236] = 1.0

    def test_initialization(self):
        """Verify fusion constructor stores settings."""
        self.assertEqual(self.fusion.resolution, 256)
        self.assertEqual(self.fusion.max_scale_mm, 5.0)
        self.assertEqual(self.fusion.cavity_strength, 0.40)

    def test_displacement_fusion_weights(self):
        """Verify multi-tier displacements composite linearly by specified weights."""
        meso = np.full((self.resolution, self.resolution), 0.5, dtype=np.float32)
        micro = np.full((self.resolution, self.resolution), 0.1, dtype=np.float32)
        macro = np.full((self.resolution, self.resolution), 0.2, dtype=np.float32)

        total = self.fusion.fuse_displacements(
            meso_disp_mm=meso,
            micro_disp_mm=micro,
            macro_disp_mm=macro,
            weight_macro=1.0,
            weight_meso=1.0,
            weight_micro=1.0,
        )

        expected = 0.5 + 0.1 + 0.2
        self.assertAlmostEqual(float(total[10, 10]), expected, places=4)
        self.assertEqual(total.shape, (self.resolution, self.resolution))

    def test_displacement_bounds_clipping(self):
        """Ensure fused displacements do not exceed maximum metric millimeter scale."""
        huge_meso = np.full((self.resolution, self.resolution), 10.0, dtype=np.float32)
        huge_micro = np.full((self.resolution, self.resolution), 5.0, dtype=np.float32)

        total = self.fusion.fuse_displacements(
            meso_disp_mm=huge_meso,
            micro_disp_mm=huge_micro,
        )

        self.assertLessEqual(total.max(), self.max_scale_mm)
        self.assertGreaterEqual(total.min(), -self.max_scale_mm)

    def test_tangent_normal_computation(self):
        """Verify tangent normal map generation produces normalized unit vectors."""
        # Flat surface -> neutral blue normal (128, 128, 255)
        flat = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        normals = self.fusion.compute_tangent_normal_map(flat)

        self.assertEqual(normals.shape, (self.resolution, self.resolution, 3))
        self.assertEqual(normals.dtype, np.uint8)
        self.assertTrue(126 <= normals[100, 100, 0] <= 130)  # R ~ 128
        self.assertTrue(126 <= normals[100, 100, 1] <= 130)  # G ~ 128
        self.assertTrue(250 <= normals[100, 100, 2] <= 255)  # B ~ 255

    def test_micro_cavity_darkening_on_pore_pit(self):
        """Verify cavity map darkens (< 1.0) on concave pore depressions and stays 1.0 on flat skin."""
        disp = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        # Create a deep pore depression at (128, 128)
        disp[126:130, 126:130] = -0.30

        cavity = self.fusion.compute_cavity_ao_map(disp)

        self.assertEqual(cavity.shape, (self.resolution, self.resolution))
        # Deep pit should have cavity < 0.90 (darkened)
        self.assertLess(cavity[128, 128], 0.90)
        # Far from pit, flat surface should be 1.0
        self.assertAlmostEqual(float(cavity[10, 10]), 1.0, places=3)
        self.assertTrue(0.0 <= cavity.min() <= cavity.max() <= 1.0)

    def test_dual_lobe_roughness_anatomical_variation(self):
        """Verify T-zone receives lower roughness (glossier) than cheeks."""
        roughness_dict = self.fusion.compute_dual_lobe_roughness(self.zone_masks)

        base_rough = roughness_dict['base_roughness']
        coat_rough = roughness_dict['coat_roughness']

        # T-zone column (col 128) vs Cheeks (col 50)
        tzone_base = base_rough[100, 128]
        cheeks_base = base_rough[100, 50]

        self.assertLess(tzone_base, cheeks_base, "T-zone must be less rough (glossier) than cheeks")

        # Coat sheen in T-zone should be sharp (< 0.25)
        self.assertLessEqual(coat_rough[100, 128], 0.25)

    def test_full_coupling_pipeline(self):
        """Verify end-to-end PBR coupling returns all required maps with correct types."""
        meso = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        micro = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        res = self.fusion.couple_pbr_material_stack(
            meso_disp_mm=meso,
            micro_disp_mm=micro,
            zone_masks=self.zone_masks,
        )

        self.assertIn('composite_displacement_mm', res)
        self.assertIn('composite_normal_rgb', res)
        self.assertIn('cavity_ao_map', res)
        self.assertIn('base_roughness', res)
        self.assertIn('coat_roughness', res)

    def test_save_maps_round_trip(self):
        """Verify 16-bit uint displacement and PBR texture map serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meso = np.linspace(-0.5, 0.5, self.resolution, dtype=np.float32)
            meso = np.repeat(meso[:, np.newaxis], self.resolution, axis=1)
            micro = np.zeros((self.resolution, self.resolution), dtype=np.float32)

            res = self.fusion.couple_pbr_material_stack(
                meso_disp_mm=meso,
                micro_disp_mm=micro,
                zone_masks=self.zone_masks,
            )

            saved = self.fusion.save_maps(res, output_dir=tmpdir, prefix="film_test")
            self.assertTrue(saved['displacement_png'].exists())
            self.assertTrue(saved['normal_png'].exists())
            self.assertTrue(saved['cavity_ao_png'].exists())
            self.assertTrue(saved['roughness_base_png'].exists())
            self.assertTrue(saved['roughness_coat_png'].exists())

            # Read back 16-bit PNG
            u16 = cv2.imread(str(saved['displacement_png']), cv2.IMREAD_UNCHANGED)
            self.assertEqual(u16.dtype, np.uint16)

            decoded_mm = ((u16.astype(np.float32) / 65535.0) * 2.0 - 1.0) * self.max_scale_mm
            err = np.max(np.abs(decoded_mm - res['composite_displacement_mm']))
            self.assertLess(err, 0.002, f"Round-trip decoding error too high: {err}")

    def test_cavity_map_generator_modes(self):
        """Verify CavityMapGenerator supports both film_ao and curvature modes."""
        disp = np.zeros((64, 64), dtype=np.float32)
        disp[30:34, 30:34] = -0.5

        # Film AO mode: flat is 1.0, pits darken
        gen_ao = CavityMapGenerator(strength=0.35, mode="film_ao")
        cav_ao = gen_ao.generate(disp)
        self.assertAlmostEqual(float(cav_ao[5, 5]), 1.0, places=3)
        self.assertLess(cav_ao[32, 32], 0.9)

        # Curvature mode: flat is 0.5
        gen_curv = CavityMapGenerator(strength=0.35, mode="curvature")
        cav_curv = gen_curv.generate(disp)
        self.assertAlmostEqual(float(cav_curv[5, 5]), 0.5, places=3)


if __name__ == '__main__':
    unittest.main()
