import unittest
import tempfile
import json
import numpy as np
import cv2
from pathlib import Path

from src.stage0_preprocess.detector import FaceDetection
from scripts.ingest_capture_session import ingest_session, EXPECTED_VIEWS


class MockCaptureDetector:
    """Mock detector for ingest testing without insightface or GPU dependencies."""
    def __init__(self):
        self.base_embedding = np.ones(512, dtype=np.float32)
        self.base_embedding /= np.linalg.norm(self.base_embedding)

    def detect_single(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        # Infer yaw from synthetic fill value
        val = int(image_bgr[0, 0, 0])
        if val == 100:
            yaw = 0.0
        elif val == 120:
            yaw = -45.0
        elif val == 140:
            yaw = 45.0
        else:
            yaw = 0.0

        return FaceDetection(
            bbox=np.array([0, 0, w, h], dtype=np.float32),
            landmarks_5pt=np.zeros((5, 2), dtype=np.float32),
            det_score=0.98,
            yaw_deg=yaw,
            crop_112=np.zeros((112, 112, 3), dtype=np.uint8),
            crop_224=np.zeros((224, 224, 3), dtype=np.uint8),
            embedding=self.base_embedding.copy(),
        )


class TestCaptureIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_path = Path(self.temp_dir.name) / "SESSION_001"
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.detector = MockCaptureDetector()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_face_photo(self, filename: str, fill_val: int = 120):
        img_path = self.session_path / filename
        img = np.full((256, 256, 3), fill_val, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return img_path

    def test_ingest_fails_on_missing_required_views(self):
        """Verify ingest_session raises FileNotFoundError when required views (e.g. left45) are missing."""
        # Only create frontal
        self._create_synthetic_face_photo("view_frontal.png")

        with self.assertRaises(FileNotFoundError) as ctx:
            ingest_session(str(self.session_path), detector=self.detector)
        self.assertIn("Missing required camera views", str(ctx.exception))

    def test_ingest_succeeds_on_complete_session(self):
        """Verify ingest_session processes complete set and writes session_manifest.json."""
        self._create_synthetic_face_photo("view_frontal.png", fill_val=100)
        self._create_synthetic_face_photo("view_left45.png", fill_val=120)
        self._create_synthetic_face_photo("view_right45.png", fill_val=140)

        manifest = ingest_session(str(self.session_path), detector=self.detector)
        self.assertIn('views', manifest)
        self.assertIn('frontal', manifest['views'])
        self.assertIn('left45', manifest['views'])
        self.assertIn('right45', manifest['views'])

        manifest_file = self.session_path / "session_manifest.json"
        self.assertTrue(manifest_file.exists())

        with open(manifest_file) as f:
            data = json.load(f)
        self.assertEqual(data['session_id'], "SESSION_001")
        self.assertTrue(data['is_valid'])


if __name__ == '__main__':
    unittest.main()
