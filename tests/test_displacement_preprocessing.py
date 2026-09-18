import unittest
import numpy as np
from scripts.build_uv_displacement_dataset import (
    encode_displacement_16bit,
    decode_displacement_16bit,
    rasterize_uv_maps,
)

class TestDisplacementPreprocessing(unittest.TestCase):
    def test_lossless_16bit_roundtrip_precision(self):
        """Verify 16-bit uint PNG displacement encoding round-trip error is strictly < 1/65535."""
        p99_mm = 2.0
        # Generate 1000 random displacement values in range [-2.0, 2.0] mm
        rng = np.random.default_rng(42)
        disp_mm = rng.uniform(-2.0, 2.0, 1000).astype(np.float32)

        # 1. Encode to uint16
        u16 = encode_displacement_16bit(disp_mm, p99_mm)
        self.assertEqual(u16.dtype, np.uint16)
        self.assertTrue(np.all(u16 >= 0))
        self.assertTrue(np.all(u16 <= 65535))

        # 2. Decode back to [-1, 1] normalized float
        decoded_norm = decode_displacement_16bit(u16)
        ground_truth_norm = disp_mm / p99_mm

        max_err = np.max(np.abs(decoded_norm - ground_truth_norm))
        # Discretization resolution of 16-bit uint is 1.0 / 65535
        self.assertLess(max_err, 1.0 / 65535.0, f"Round-trip error {max_err} exceeds 1/65535 threshold")

        # 3. Decode back to mm
        decoded_mm = decode_displacement_16bit(u16, p99_mm=p99_mm)
        max_err_mm = np.max(np.abs(decoded_mm - disp_mm))
        self.assertLess(max_err_mm, (p99_mm / 65535.0) * 1.01)

    def test_barycentric_uv_rasterization(self):
        """Verify barycentric UV rasterization produces filled mask and interpolated displacement."""
        resolution = 64
        # 3 vertices of a single triangle in UV space
        uv_coords = np.array([
            [0.2, 0.2],
            [0.8, 0.2],
            [0.5, 0.8]
        ], dtype=np.float32)
        uv_faces = np.array([[0, 1, 2]], dtype=np.int64)

        flame_verts = np.array([
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.0, 0.05, 0.0]
        ], dtype=np.float32)
        flame_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        disp_mm = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        hit_mask = np.array([True, True, True], dtype=bool)

        disp_map, pos_map, norm_map, mask_map = rasterize_uv_maps(
            flame_verts_m=flame_verts,
            flame_normals=flame_normals,
            disp_mm=disp_mm,
            hit_mask=hit_mask,
            uv_coords=uv_coords,
            uv_faces=uv_faces,
            resolution=resolution
        )

        self.assertEqual(disp_map.shape, (resolution, resolution))
        self.assertEqual(mask_map.shape, (resolution, resolution))
        self.assertEqual(pos_map.shape, (resolution, resolution, 3))
        self.assertEqual(norm_map.shape, (resolution, resolution, 3))

        # Triangle interior must contain rasterized pixels
        valid_pixels = (mask_map > 0)
        self.assertTrue(np.any(valid_pixels))

        # Interpolated displacement values must lie within [min, max] of triangle vertices
        disp_in_triangle = disp_map[valid_pixels]
        self.assertGreaterEqual(disp_in_triangle.min(), 0.99)
        self.assertLessEqual(disp_in_triangle.max(), 3.01)

if __name__ == "__main__":
    unittest.main()
