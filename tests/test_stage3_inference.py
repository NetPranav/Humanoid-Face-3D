from __future__ import annotations
"""
Unit tests for Stage 3 High-Frequency Detail GAN Inference Module.
"""
import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
import json

try:
    import torch
except ImportError:
    torch = None

from src.stage3_detail.inference import DetailSynthesizer
from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.rasterizer import (
    load_flame_uv_layout,
    load_flame_geometry_faces,
    compute_vertex_normals,
    rasterize_uv_maps
)


class TestStage3Inference(unittest.TestCase):

    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.ckpt_path = self.root / 'models_cache' / 'stage3_detail' / 'ema_generator.pt'
        self.stats_path = self.root / 'models_cache' / 'stage3_detail' / 'normalization_stats.json'

        # Synthetic minimal test mesh (5023 vertices to match FLAME parameterization)
        np.random.seed(42)
        self.verts = np.random.randn(5023, 3).astype(np.float32) * 0.1
        self.faces = load_flame_geometry_faces()

    @unittest.skipIf(torch is None, "PyTorch required for DetailSynthesizer inference")
    def test_zero_silent_fallbacks_on_missing_weights(self):
        """Invariant 1: Zero silent fallbacks when weights are missing."""
        with self.assertRaises(FileNotFoundError):
            DetailSynthesizer(checkpoint_path="/nonexistent/path/to/ema_generator.pt")

    def test_uv_rasterizer_geometry(self):
        """Validates barycentric triangle rasterizer over FLAME UV layout."""
        uv_coords, uv_faces = load_flame_uv_layout()
        self.assertGreater(len(uv_coords), 0)
        self.assertGreater(len(uv_faces), 0)

        normals = compute_vertex_normals(self.verts, self.faces)
        self.assertEqual(normals.shape, (5023, 3))
        # Normals must have unit length
        lengths = np.linalg.norm(normals[self.faces.flatten()], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-4)

        disp_m, pos_m, norm_m, mask_m = rasterize_uv_maps(
            flame_verts_m=self.verts,
            flame_normals=normals,
            uv_coords=uv_coords,
            uv_faces=uv_faces,
            flame_faces=self.faces,
            resolution=128
        )
        self.assertEqual(pos_m.shape, (128, 128, 3))
        self.assertEqual(norm_m.shape, (128, 128, 3))
        self.assertEqual(mask_m.shape, (128, 128))

    @unittest.skipIf(torch is None, "PyTorch required for DetailSynthesizer inference")
    @unittest.skipUnless(
        os.path.exists('models_cache/stage3_detail/ema_generator.pt'),
        "Trained Detail GAN checkpoint required for full forward test"
    )
    def test_detail_synthesis_forward_and_formats(self):
        """Tests full synthesis forward pass, 16-bit uint encoding, and normal map generation."""
        synthesizer = DetailSynthesizer(
            checkpoint_path=str(self.ckpt_path),
            stats_path=str(self.stats_path),
            device='cpu'
        )
        with open(self.stats_path) as f:
            stats_data = json.load(f)
            expected_p99 = float(stats_data.get('p99_displacement_mm', stats_data.get('p99_mm', 1.1465)))
        self.assertAlmostEqual(synthesizer.p99_mm, expected_p99, places=3)

        res = 128  # Fast resolution for unit test
        out = synthesizer.synthesize(
            neutral_vertices=self.verts,
            faces=self.faces,
            resolution=res
        )

        self.assertIn('disp_mm', out)
        self.assertIn('disp_norm', out)
        self.assertIn('disp_uint16', out)
        self.assertIn('normal_map_rgb', out)
        self.assertIn('mask', out)

        disp_uint16 = out['disp_uint16']
        self.assertEqual(disp_uint16.shape, (res, res))
        self.assertEqual(disp_uint16.dtype, np.uint16)

        # Normal map checks
        norm_rgb = out['normal_map_rgb']
        self.assertEqual(norm_rgb.shape, (res, res, 3))
        self.assertEqual(norm_rgb.dtype, np.uint8)

        # Inactive UV pixels must be neutral flat normal RGB(128, 128, 255)
        inactive = (out['mask'] == 0)
        if np.any(inactive):
            sample_pixel = norm_rgb[inactive][0]
            self.assertEqual(list(sample_pixel), [128, 128, 255])

    @unittest.skipIf(torch is None, "PyTorch required for DetailSynthesizer inference")
    @unittest.skipUnless(
        os.path.exists('models_cache/stage3_detail/ema_generator.pt'),
        "Trained Detail GAN checkpoint required for disk output test"
    )
    def test_save_maps_disk_outputs(self):
        """Tests lossless 16-bit PNG write/read roundtrip and color maps."""
        synthesizer = DetailSynthesizer(
            checkpoint_path=str(self.ckpt_path),
            stats_path=str(self.stats_path),
            device='cpu'
        )
        out = synthesizer.synthesize(
            neutral_vertices=self.verts,
            faces=self.faces,
            resolution=128
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = DetailSynthesizer.save_maps(out, tmpdir, prefix="test_subject")
            self.assertTrue(os.path.exists(paths['displacement_16bit']))
            self.assertTrue(os.path.exists(paths['normal_map']))
            self.assertTrue(os.path.exists(paths['displacement_preview']))

            # Verify 16-bit depth on disk
            loaded_16bit = cv2.imread(paths['displacement_16bit'], cv2.IMREAD_UNCHANGED)
            self.assertEqual(loaded_16bit.dtype, np.uint16)
            self.assertEqual(loaded_16bit.shape, (128, 128))
            np.testing.assert_array_equal(loaded_16bit, out['disp_uint16'])


    def test_conditioning_maps_no_wireframe_leakage(self):
        """
        Validates that C² Gaussian smoothing eliminates polygonal triangle-edge
        artifacts from conditioning maps. The Laplacian (second derivative) of
        the smoothed position map must stay below 0.02 — wireframe leakage
        produces sharp spikes well above this threshold.
        """
        from src.utils.subdivision import loop_subdivide

        uv_coords, uv_faces = load_flame_uv_layout()
        normals = compute_vertex_normals(self.verts, self.faces)

        # Subdivide once to match inference.py behavior for <15k vertex meshes
        v_sub, f_sub, uv_sub, uv_f_sub = loop_subdivide(
            self.verts, self.faces, uv_coords, uv_faces, levels=1
        )
        normals_sub = compute_vertex_normals(v_sub, f_sub)

        _, pos_map, norm_map, mask_map = rasterize_uv_maps(
            flame_verts_m=v_sub,
            flame_normals=normals_sub,
            uv_coords=uv_sub,
            uv_faces=uv_f_sub,
            flame_faces=f_sub,
            resolution=256
        )

        # Apply the same C² normalized convolution smoothing as inference.py
        COND_SIGMA = 8.0
        valid_px = (mask_map > 0)
        if np.any(valid_px):
            mask_f = valid_px.astype(np.float32)
            pos_signal = pos_map * mask_f[..., None]
            pos_blurred = cv2.GaussianBlur(pos_signal, (0, 0), sigmaX=COND_SIGMA, sigmaY=COND_SIGMA)
            mask_blurred = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=COND_SIGMA, sigmaY=COND_SIGMA)
            safe_mask = (mask_blurred > 0.01)
            pos_smooth = np.zeros_like(pos_map)
            pos_smooth[safe_mask] = pos_blurred[safe_mask] / mask_blurred[safe_mask, None]
            pos_map[valid_px] = pos_smooth[valid_px]

        # Compute Laplacian magnitude of the smoothed position map
        laplacian_mag = np.zeros(pos_map.shape[:2], dtype=np.float32)
        for ch in range(3):
            lap = cv2.Laplacian(pos_map[:, :, ch], cv2.CV_32F, ksize=3)
            laplacian_mag += lap ** 2
        laplacian_mag = np.sqrt(laplacian_mag)

        # Only check within the valid facial mask interior (erode to exclude boundary)
        kernel = np.ones((5, 5), np.uint8)
        interior = cv2.erode(mask_map, kernel, iterations=2)
        interior_px = (interior > 0)

        if np.any(interior_px):
            max_laplacian = float(laplacian_mag[interior_px].max())
            self.assertLess(
                max_laplacian, 0.02,
                f"Wireframe leakage detected! Max Laplacian magnitude {max_laplacian:.4f} "
                f"exceeds 0.02 threshold inside facial mask interior."
            )


if __name__ == '__main__':
    unittest.main()
