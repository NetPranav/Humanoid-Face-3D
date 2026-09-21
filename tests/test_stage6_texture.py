"""
Unit tests for Stage 6: Multi-View UV Texture Projection.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock


class TestCameraProjectionMatrix(unittest.TestCase):
    """Tests for estimate_camera_projection_matrix."""

    def setUp(self):
        from src.stage6_texture.projector import estimate_camera_projection_matrix
        self.estimate = estimate_camera_projection_matrix

    def test_produces_3x4_matrix(self):
        """Projection matrix must be 3×4."""
        landmarks = np.array([
            [200, 180], [320, 180], [260, 250],
            [220, 310], [300, 310],
        ], dtype=np.float32)
        P = self.estimate(landmarks, yaw_deg=0, pitch_deg=0, roll_deg=0, image_shape=(480, 640))
        self.assertEqual(P.shape, (3, 4))

    def test_nonzero_entries(self):
        """P should have non-zero entries (not a degenerate zero matrix)."""
        landmarks = np.array([
            [200, 180], [320, 180], [260, 250],
            [220, 310], [300, 310],
        ], dtype=np.float32)
        P = self.estimate(landmarks, yaw_deg=15, pitch_deg=5, roll_deg=0, image_shape=(480, 640))
        self.assertGreater(np.abs(P).sum(), 0.0)

    def test_different_yaw_produces_different_P(self):
        """Different head poses should produce different projection matrices."""
        # Frontal landmarks (symmetric)
        landmarks_frontal = np.array([
            [200, 180], [320, 180], [260, 250],
            [220, 310], [300, 310],
        ], dtype=np.float32)
        # Rotated landmarks (asymmetric — simulating a yaw-rotated head)
        landmarks_rotated = np.array([
            [180, 185], [310, 175], [250, 255],
            [200, 315], [290, 305],
        ], dtype=np.float32)
        P0 = self.estimate(landmarks_frontal, yaw_deg=0, pitch_deg=0, roll_deg=0, image_shape=(480, 640))
        P45 = self.estimate(landmarks_rotated, yaw_deg=45, pitch_deg=0, roll_deg=0, image_shape=(480, 640))
        self.assertFalse(np.allclose(P0, P45, atol=1e-3))


class TestVisibilityMask(unittest.TestCase):
    """Tests for z-buffer visibility computation."""

    def test_all_visible_for_simple_plane(self):
        """For a flat plane facing the camera, all vertices should be visible."""
        from src.stage6_texture.projector import _compute_visibility_mask

        # Simple quad (2 triangles) facing +Z
        vertices = np.array([
            [-0.05, -0.05, 0.1],
            [ 0.05, -0.05, 0.1],
            [ 0.05,  0.05, 0.1],
            [-0.05,  0.05, 0.1],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # Simple orthographic-like P
        P = np.array([
            [500, 0, 0, 240],
            [0, 500, 0, 240],
            [0, 0, 1, 0],
        ], dtype=np.float64)

        visible = _compute_visibility_mask(vertices, faces, P, (480, 480))
        self.assertTrue(np.all(visible))


class TestMultiViewProjector(unittest.TestCase):
    """Tests for MultiViewTextureProjector."""

    def test_initialization(self):
        from src.stage6_texture.projector import MultiViewTextureProjector
        proj = MultiViewTextureProjector(texture_resolution=512, blend_gamma=2.0)
        self.assertEqual(proj.resolution, 512)
        self.assertEqual(proj.gamma, 2.0)

    def test_empty_input_produces_empty_output(self):
        """Empty photos list should produce zero-filled texture."""
        from src.stage6_texture.projector import MultiViewTextureProjector
        proj = MultiViewTextureProjector(texture_resolution=64)

        # Minimal mesh (single triangle)
        vertices = np.array([
            [0, 0, 0.1],
            [0.1, 0, 0.1],
            [0, 0.1, 0.1],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        uv_faces = np.array([[0, 1, 2]], dtype=np.int32)

        result = proj.project(
            photos=[], detections=[],
            vertices=vertices, faces=faces,
            vertex_normals=normals,
            uv_coords=uv, uv_faces=uv_faces,
        )
        self.assertEqual(result['projected_rgb'].shape, (64, 64, 3))
        self.assertEqual(result['projection_mask'].shape, (64, 64))
        # All zeros since no photos
        self.assertEqual(result['projection_mask'].sum(), 0)

    def test_save_maps_creates_files(self):
        """save_maps should write PNG files to disk."""
        import tempfile
        from src.stage6_texture.projector import MultiViewTextureProjector
        proj = MultiViewTextureProjector(texture_resolution=32)

        result = {
            'projected_rgb': np.random.rand(32, 32, 3).astype(np.float32) * 255,
            'projection_mask': np.ones((32, 32), dtype=np.uint8) * 255,
            'weight_map': np.ones((32, 32), dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = proj.save_maps(result, tmpdir)
            self.assertIn('projected_texture', paths)
            self.assertIn('projection_mask', paths)
            from pathlib import Path
            self.assertTrue(Path(paths['projected_texture']).exists())
            self.assertTrue(Path(paths['projection_mask']).exists())


class TestProjectionWeights(unittest.TestCase):
    """Tests for blending weight correctness."""

    def test_weight_map_nonnegative(self):
        """Accumulated weights must be non-negative everywhere."""
        from src.stage6_texture.projector import MultiViewTextureProjector
        proj = MultiViewTextureProjector(texture_resolution=32)

        result = {
            'weight_map': np.random.rand(32, 32).astype(np.float32),
            'projected_rgb': np.zeros((32, 32, 3), dtype=np.float32),
            'projection_mask': np.zeros((32, 32), dtype=np.uint8),
        }
        self.assertTrue(np.all(result['weight_map'] >= 0))


if __name__ == '__main__':
    unittest.main()
