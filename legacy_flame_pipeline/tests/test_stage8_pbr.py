"""
Unit tests for Stage 8: PBR Material Stack Derivation.
"""
import unittest
import numpy as np


class TestRoughnessMapGenerator(unittest.TestCase):
    """Tests for anatomical zone roughness generation."""

    def test_roughness_bounded_0_1(self):
        """Roughness values must be in [0, 1]."""
        from src.stage8_pbr.material_stack import RoughnessMapGenerator

        gen = RoughnessMapGenerator()
        zone_masks = {
            'valid': np.ones((64, 64), dtype=np.float32),
            't_zone': np.zeros((64, 64), dtype=np.float32),
            'cheeks': np.zeros((64, 64), dtype=np.float32),
            'lips': np.zeros((64, 64), dtype=np.float32),
            'periorbital': np.zeros((64, 64), dtype=np.float32),
        }
        roughness = gen.generate(zone_masks)
        self.assertTrue(np.all(roughness >= 0.0))
        self.assertTrue(np.all(roughness <= 1.0))

    def test_t_zone_smoother_than_cheeks(self):
        """T-zone roughness should be lower (smoother) than cheeks."""
        from src.stage8_pbr.material_stack import RoughnessMapGenerator

        gen = RoughnessMapGenerator()

        # Create a mask where top half is T-zone, bottom half is cheeks
        t_zone = np.zeros((64, 64), dtype=np.float32)
        t_zone[:32, :] = 1.0
        cheeks = np.zeros((64, 64), dtype=np.float32)
        cheeks[32:, :] = 1.0

        zone_masks = {
            'valid': np.ones((64, 64), dtype=np.float32),
            't_zone': t_zone,
            'cheeks': cheeks,
            'lips': np.zeros((64, 64), dtype=np.float32),
            'periorbital': np.zeros((64, 64), dtype=np.float32),
        }
        roughness = gen.generate(zone_masks)

        t_zone_mean = roughness[:32, :].mean()
        cheeks_mean = roughness[32:, :].mean()
        self.assertLess(t_zone_mean, cheeks_mean,
                        f"T-zone ({t_zone_mean:.3f}) should be smoother than cheeks ({cheeks_mean:.3f})")

    def test_displacement_variation_affects_roughness(self):
        """Adding displacement should change the roughness map."""
        from src.stage8_pbr.material_stack import RoughnessMapGenerator

        gen = RoughnessMapGenerator(displacement_variation=0.1)
        zone_masks = {
            'valid': np.ones((64, 64), dtype=np.float32),
            't_zone': np.zeros((64, 64), dtype=np.float32),
            'cheeks': np.zeros((64, 64), dtype=np.float32),
            'lips': np.zeros((64, 64), dtype=np.float32),
            'periorbital': np.zeros((64, 64), dtype=np.float32),
        }

        # Without displacement
        r_no_disp = gen.generate(zone_masks, displacement_map=None)

        # With displacement
        disp = np.random.randn(64, 64).astype(np.float32) * 0.5
        r_with_disp = gen.generate(zone_masks, displacement_map=disp)

        self.assertFalse(np.allclose(r_no_disp, r_with_disp, atol=1e-4),
                         "Displacement should affect roughness values")


class TestCavityMapGenerator(unittest.TestCase):
    """Tests for displacement-derived cavity/AO map."""

    def test_flat_displacement_produces_neutral_cavity(self):
        """Flat displacement (zero curvature) should produce ~0.5 cavity."""
        from src.stage8_pbr.material_stack import CavityMapGenerator

        gen = CavityMapGenerator(strength=0.35)
        flat_disp = np.ones((64, 64), dtype=np.float32) * 0.5  # Constant
        cavity = gen.generate(flat_disp)

        # Should be very close to 0.5 (neutral) since Laplacian of constant is 0
        self.assertAlmostEqual(cavity.mean(), 0.5, delta=0.05,
                               msg="Flat displacement should produce neutral cavity")

    def test_cavity_bounded_0_1(self):
        """Cavity values must be clamped to [0, 1]."""
        from src.stage8_pbr.material_stack import CavityMapGenerator

        gen = CavityMapGenerator(strength=0.35)
        # Random displacement with strong variation
        disp = np.random.randn(64, 64).astype(np.float32) * 2.0
        cavity = gen.generate(disp)

        self.assertTrue(np.all(cavity >= 0.0))
        self.assertTrue(np.all(cavity <= 1.0))

    def test_symmetric_input_produces_symmetric_output(self):
        """Symmetric displacement should produce symmetric cavity."""
        from src.stage8_pbr.material_stack import CavityMapGenerator

        gen = CavityMapGenerator(strength=0.35, blur_sigma=0)

        # Create symmetric displacement (mirror around center column)
        half = np.random.randn(64, 32).astype(np.float32)
        disp = np.hstack([half, half[:, ::-1]])
        cavity = gen.generate(disp)

        # Check approximate symmetry (allowing small numerical differences)
        left = cavity[:, :32]
        right = cavity[:, 32:][:, ::-1]
        np.testing.assert_array_almost_equal(
            left, right, decimal=3,
            err_msg="Symmetric displacement should produce symmetric cavity"
        )


class TestSSSThicknessGenerator(unittest.TestCase):
    """Tests for subsurface scattering thickness estimation."""

    def test_output_bounded_0_1(self):
        """SSS values must be in [0, 1] when normalized."""
        from src.stage8_pbr.material_stack import SSSThicknessGenerator

        gen = SSSThicknessGenerator(normalize=True)

        # Simple cube-like mesh (8 vertices, 12 triangles)
        vertices = np.array([
            [-0.05, -0.05, -0.05], [0.05, -0.05, -0.05],
            [0.05, 0.05, -0.05], [-0.05, 0.05, -0.05],
            [-0.05, -0.05, 0.05], [0.05, -0.05, 0.05],
            [0.05, 0.05, 0.05], [-0.05, 0.05, 0.05],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)
        normals = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        uv = np.array([
            [0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0],
            [0.0, 0.5], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5],
        ], dtype=np.float32)
        uv_faces = faces.copy()

        sss = gen.generate(vertices, faces, normals, uv, uv_faces, faces, resolution=32)

        self.assertTrue(np.all(sss >= 0.0))
        self.assertTrue(np.all(sss <= 1.0))

    def test_output_shape(self):
        """SSS map shape must match requested resolution."""
        from src.stage8_pbr.material_stack import SSSThicknessGenerator

        gen = SSSThicknessGenerator(normalize=True)

        # Minimal triangle
        v = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        n = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        sss = gen.generate(v, f, n, uv, f, f, resolution=64)
        self.assertEqual(sss.shape, (64, 64))


class TestPBRMaterialStack(unittest.TestCase):
    """Tests for the full PBR material stack orchestrator."""

    def test_generate_returns_all_maps(self):
        """generate() should return roughness, cavity, and sss_thickness."""
        from src.stage8_pbr.material_stack import PBRMaterialStack

        stack = PBRMaterialStack()

        # Minimal triangle mesh
        v = np.array([[0, 0, 0.1], [0.1, 0, 0.1], [0, 0.1, 0.1]], dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        n = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        result = stack.generate(v, f, n, uv, f, f, resolution=32)

        self.assertIn('roughness', result)
        self.assertIn('cavity', result)
        self.assertIn('sss_thickness', result)
        self.assertEqual(result['roughness'].shape, (32, 32))
        self.assertEqual(result['cavity'].shape, (32, 32))
        self.assertEqual(result['sss_thickness'].shape, (32, 32))

    def test_save_maps_creates_files(self):
        """save_maps should write all PBR textures to disk."""
        import tempfile
        from src.stage8_pbr.material_stack import PBRMaterialStack

        stack = PBRMaterialStack()
        result = {
            'roughness': np.random.rand(32, 32).astype(np.float32),
            'cavity': np.random.rand(32, 32).astype(np.float32),
            'sss_thickness': np.random.rand(32, 32).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = stack.save_maps(result, tmpdir)
            self.assertIn('roughness_map', paths)
            self.assertIn('cavity_ao_map', paths)
            self.assertIn('sss_thickness_map', paths)
            from pathlib import Path
            for key, path in paths.items():
                self.assertTrue(Path(path).exists(), f"{key} file not created: {path}")


if __name__ == '__main__':
    unittest.main()
