import unittest
import tempfile
import numpy as np
from pathlib import Path

from src.stage0_preprocess.detector import FaceDetector
from src.stage1_identity.inference import MICAIdentityEncoder
from src.pipeline import FaceGeoPipeline

class TestSilentFallbacksRemoved(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detector_raises_without_insightface_by_default(self):
        """Verify FaceDetector raises RuntimeError when insightface is missing and allow_degraded is False."""
        # When insightface is not in environment or fails to load, it should raise
        try:
            detector = FaceDetector(allow_degraded=False)
            # If insightface happened to be installed, detect should work normally
            self.assertTrue(detector.has_insightface)
        except RuntimeError as e:
            self.assertIn("InsightFace is required", str(e))

    def test_pipeline_raises_on_missing_flame(self):
        """Verify FaceGeoPipeline refuses to synthesize zero-mesh when FLAME is missing."""
        cfg_path = self.temp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            f.write("pipeline: {}\nallow_degraded: true\n")

        # Point to non-existent model directory
        pipe = FaceGeoPipeline(str(cfg_path), model_dir=str(self.temp_path / "nonexistent"))
        self.assertIsNone(pipe.flame)

        # Mock dummy input photo
        dummy_photo = self.temp_path / "front.jpg"
        dummy_photo.write_text("fake image content")

        # Running without FLAME model must raise RuntimeError, never write 5023 zero vertices
        with self.assertRaises(Exception) as ctx:
            pipe.run([str(dummy_photo)], str(self.temp_path / "output"))
        # Must fail at validation or flame initialization, not write degenerate OBJ
        self.assertFalse((self.temp_path / "output/head_mesh.obj").exists())

    def test_mica_raises_on_missing_checkpoint(self):
        """Verify MICAIdentityEncoder raises FileNotFoundError instead of returning zeros(300)."""
        missing_ckpt = self.temp_path / "missing_mica.tar"
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            MICAIdentityEncoder(checkpoint_path=str(missing_ckpt))

if __name__ == "__main__":
    unittest.main()
