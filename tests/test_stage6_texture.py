"""
Tests for Stage 6 (Phase 0): per-texel backprojection with z-buffer occlusion and skin masks,
plus the CPU rasterizer it relies on.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.render.soft_raster import interpolate, rasterize
from src.stage6_texture.projector import MultiViewTextureProjector, rasterize_uv

K = np.array([[500.0, 0, 64], [0, 500.0, 64], [0, 0, 1]])


def quad(z, x0=-0.05, x1=0.05, y0=-0.05, y1=0.05, u0=0.0, u1=1.0):
    """Camera-facing square at depth z (OpenCV frame), with a UV rectangle."""
    v = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    f = np.array([[0, 2, 1], [0, 3, 2]])  # outward normal towards the camera (-z)
    uv = np.array([[u0, 1.0], [u1, 1.0], [u1, 0.0], [u0, 0.0]], float)
    return v, f, uv


class TestRasterizer(unittest.TestCase):
    def test_nearest_surface_wins(self):
        xy = np.array([[10, 10], [110, 10], [60, 110], [10, 10], [110, 10], [60, 110]], float)
        z = np.array([1, 1, 1, 2, 2, 2], float)
        faces = np.array([[3, 4, 5], [0, 1, 2]])  # far triangle drawn first
        r = rasterize(xy, z, faces, 128, 128)
        self.assertEqual(r.face_id[50, 60], 1)
        self.assertAlmostEqual(float(r.depth[50, 60]), 1.0, places=5)

    def test_barycentrics_interpolate_attributes(self):
        xy = np.array([[0, 0], [128, 0], [0, 128]], float)
        r = rasterize(xy, np.ones(3), np.array([[0, 1, 2]]), 128, 128)
        attr = np.array([[0.0], [1.0], [0.0]])
        a = interpolate(attr, np.array([[0, 1, 2]]), r)
        self.assertAlmostEqual(float(a[10, 64, 0]), 64.5 / 128, places=2)


class TestProjector(unittest.TestCase):
    def setUp(self):
        self.proj = MultiViewTextureProjector(texture_resolution=64, zbuffer_max_side=128)

    def test_textures_visible_camera_facing_plane(self):
        v, f, uv = quad(1.0)
        photo = np.full((128, 128, 3), 200, np.uint8)
        skin = np.ones((128, 128), bool)
        res = self.proj.project([photo], [v], [K], [skin], f, uv, f)
        self.assertGreater(res["per_view_coverage"][0], 0.95)
        m = res["projection_mask"] > 0
        self.assertTrue(np.allclose(res["projected_rgb"][m], 200, atol=1))

    def test_occluded_surface_gets_no_texture(self):
        front, ff, uvf_ = quad(1.0, u0=0.0, u1=0.5)
        back, fb, uvb = quad(1.5, x0=-0.03, x1=0.03, y0=-0.03, y1=0.03, u0=0.5, u1=1.0)
        v = np.vstack([front, back])
        faces = np.vstack([ff, fb + 4])
        uv = np.vstack([uvf_, uvb])
        photo = np.full((128, 128, 3), 100, np.uint8)
        res = self.proj.project([photo], [v], [K], [np.ones((128, 128), bool)], faces, uv, faces)
        mask = res["projection_mask"] > 0
        self.assertTrue(mask[:, :30].any(), "front quad should be textured")
        self.assertFalse(mask[:, 34:].any(), "quad hidden behind the front quad must stay untextured")

    def test_non_skin_pixels_are_excluded(self):
        v, f, uv = quad(1.0)
        photo = np.full((128, 128, 3), 50, np.uint8)
        skin = np.zeros((128, 128), bool)
        skin[:, :64] = True
        res = self.proj.project([photo], [v], [K], [skin], f, uv, f)
        cov = res["per_view_coverage"][0]
        self.assertGreater(cov, 0.35)
        self.assertLess(cov, 0.65)

    def test_mismatched_inputs_raise(self):
        v, f, uv = quad(1.0)
        with self.assertRaises(ValueError):
            self.proj.project([np.zeros((8, 8, 3), np.uint8)], [v, v], [K], [np.ones((8, 8), bool)], f, uv, f)

    def test_save_maps_creates_files(self):
        v, f, uv = quad(1.0)
        res = self.proj.project([np.full((128, 128, 3), 90, np.uint8)], [v], [K],
                                [np.ones((128, 128), bool)], f, uv, f)
        with tempfile.TemporaryDirectory() as d:
            paths = self.proj.save_maps(res, d)
            for p in paths.values():
                self.assertTrue(Path(p).exists())

    def test_uv_raster_covers_uv_triangle(self):
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        r = rasterize_uv(uv, np.array([[0, 1, 2]]), 32)
        self.assertAlmostEqual(r.mask.mean(), 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
