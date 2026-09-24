import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path

from evaluation.identity_score import (
    IdentityEvaluator,
    compute_identity_score,
    render_neutral_preview,
)

class MockEvaluator:
    def __init__(self, fixed_score: float = 0.85):
        self.fixed_score = fixed_score
        self.evaluate_calls = []

    def evaluate(self, render_bgr: np.ndarray, photo_bgr: np.ndarray) -> float:
        self.evaluate_calls.append((render_bgr, photo_bgr))
        return self.fixed_score

class TestIdentityScore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluator_raises_when_insightface_missing(self):
        """Verify IdentityEvaluator raises RuntimeError if InsightFace cannot be loaded."""
        try:
            evaluator = IdentityEvaluator()
            # If insightface is installed, verify has_insightface flag
            self.assertTrue(evaluator.has_insightface)
        except RuntimeError as e:
            self.assertIn("InsightFace is required", str(e))

    def test_compute_identity_score_raises_on_missing_preview(self):
        """Verify compute_identity_score raises FileNotFoundError when preview PNG does not exist."""
        mesh_path = self.temp_path / "model.obj"
        mesh_path.write_text("# dummy obj")
        photo_path = self.temp_path / "photo.jpg"
        photo_path.write_text("dummy")

        # Preview model.png is intentionally missing
        with self.assertRaises(FileNotFoundError) as ctx:
            compute_identity_score(str(mesh_path), str(photo_path))
        self.assertIn("No rendered preview found", str(ctx.exception))

    def test_compute_identity_score_raises_on_missing_photo(self):
        """Verify compute_identity_score raises FileNotFoundError when source photo does not exist."""
        mesh_path = self.temp_path / "model.obj"
        mesh_path.write_text("# dummy obj")
        preview_path = self.temp_path / "model.png"
        preview_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(preview_path), preview_img)

        missing_photo = self.temp_path / "nonexistent_photo.jpg"

        mock_eval = MockEvaluator()
        with self.assertRaises(FileNotFoundError) as ctx:
            compute_identity_score(str(mesh_path), str(missing_photo), evaluator=mock_eval)
        self.assertIn("Could not read source photograph", str(ctx.exception))

    def test_render_neutral_preview_generates_image(self):
        """Verify render_neutral_preview produces a valid PNG from an OBJ file."""
        mesh_path = self.temp_path / "simple_triangle.obj"
        with open(mesh_path, "w") as f:
            f.write("v -0.05 -0.05 0.0\n")
            f.write("v 0.05 -0.05 0.0\n")
            f.write("v 0.0 0.05 0.0\n")
            f.write("f 1 2 3\n")

        out_png = self.temp_path / "simple_triangle.png"
        render_neutral_preview(str(mesh_path), str(out_png), size=128)

        self.assertTrue(out_png.exists())
        img = cv2.imread(str(out_png))
        self.assertEqual(img.shape, (128, 128, 3))
        # Verify rendered triangle has drawn pixels (canvas is 35, triangle is 180)
        self.assertTrue(np.any(img > 50))

    def test_compute_identity_score_success_with_evaluator(self):
        """Verify compute_identity_score runs correctly when preview and photo exist."""
        mesh_path = self.temp_path / "model.obj"
        mesh_path.write_text("# dummy obj")
        preview_path = self.temp_path / "model.png"
        preview_img = np.full((128, 128, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(preview_path), preview_img)

        photo_path = self.temp_path / "photo.jpg"
        photo_img = np.full((128, 128, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(photo_path), photo_img)

        mock_eval = MockEvaluator(fixed_score=0.91)
        score = compute_identity_score(str(mesh_path), str(photo_path), evaluator=mock_eval)

        self.assertEqual(score, 0.91)
        self.assertEqual(len(mock_eval.evaluate_calls), 1)

    def test_calibrate_identity_threshold_calculation(self):
        """Verify calibrate_identity_threshold correctly summarizes statistics and sets threshold."""
        from evaluation.identity_score import calibrate_identity_threshold

        mesh_path = self.temp_path / "model.obj"
        mesh_path.write_text("# dummy obj")
        preview_path = self.temp_path / "model.png"
        preview_img = np.full((128, 128, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(preview_path), preview_img)

        photo_path = self.temp_path / "photo.jpg"
        photo_img = np.full((128, 128, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(photo_path), photo_img)

        mock_eval = MockEvaluator(fixed_score=0.82)
        calib = calibrate_identity_threshold(
            positive_pairs=[(str(mesh_path), str(photo_path))],
            evaluator=mock_eval
        )

        self.assertEqual(calib['positive_count'], 1)
        self.assertAlmostEqual(calib['positive_mean'], 0.82)
        self.assertGreaterEqual(calib['recommended_threshold'], 0.40)

if __name__ == "__main__":
    unittest.main()
