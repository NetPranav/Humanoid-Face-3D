import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path

from src.stage0_preprocess.detector import FaceDetection, FaceDetector
from src.utils.validation import validate_inputs, ValidationResult

class MockFaceDetector:
    """Mock detector to simulate various real-world validation scenarios."""
    def __init__(self, behavior_map=None):
        self.behavior_map = behavior_map or {}

    def detect_single(self, img: np.ndarray):
        tag = int(np.round(img.mean()))
        if tag in self.behavior_map:
            return self.behavior_map[tag]
        # Default: high confidence, frontal face, unit embedding
        return FaceDetection(
            bbox=np.array([10, 10, 100, 100]),
            landmarks_5pt=np.zeros((5, 2)),
            det_score=0.98,
            yaw_deg=5.0,
            crop_112=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_image(self, filename: str, fill_val: int = 128) -> str:
        img_path = self.dir_path / filename
        img = np.full((200, 200, 3), fill_val, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return str(img_path)

    def test_validation_fails_on_missing_photo_file(self):
        """Verify validation rejects non-existent image paths."""
        photos = [
            str(self.dir_path / "nonexistent_1.jpg"),
            str(self.dir_path / "nonexistent_2.jpg"),
            str(self.dir_path / "nonexistent_3.jpg"),
        ]
        detector = MockFaceDetector()
        res = validate_inputs(photos, detector=detector)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("Could not load image" in err for err in res.errors))

    def test_validation_fails_on_insufficient_photos(self):
        """Verify validation rejects fewer photos than min_photos."""
        p1 = self._create_dummy_image("p1.jpg")
        p2 = self._create_dummy_image("p2.jpg")
        detector = MockFaceDetector()

        res = validate_inputs([p1, p2], detector=detector, min_photos=3)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("minimum of 3 photos is required" in err for err in res.errors))

    def test_validation_fails_when_face_not_detected(self):
        """Verify validation rejects photos where detector returns None."""
        p1 = self._create_dummy_image("p1.jpg", fill_val=10)
        p2 = self._create_dummy_image("p2.jpg", fill_val=20)
        p3 = self._create_dummy_image("p3.jpg", fill_val=30)

        # Map fill_val 20 to return None (no face detected)
        detector = MockFaceDetector(behavior_map={20: None})

        res = validate_inputs([p1, p2, p3], detector=detector)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("No face detected" in err for err in res.errors))

    def test_validation_fails_on_low_confidence_detection(self):
        """Verify validation rejects face detections below min_face_confidence."""
        p1 = self._create_dummy_image("p1.jpg", fill_val=10)
        p2 = self._create_dummy_image("p2.jpg", fill_val=20)
        p3 = self._create_dummy_image("p3.jpg", fill_val=30)

        low_conf_det = FaceDetection(
            bbox=np.array([10, 10, 100, 100]),
            landmarks_5pt=np.zeros((5, 2)),
            det_score=0.35,  # Below 0.50 threshold
            yaw_deg=5.0,
            crop_112=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        detector = MockFaceDetector(behavior_map={20: low_conf_det})

        res = validate_inputs([p1, p2, p3], detector=detector, min_face_confidence=0.50)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("below threshold" in err for err in res.errors))

    def test_validation_fails_on_different_individuals(self):
        """Verify validation rejects photos with conflicting identity embeddings."""
        p1 = self._create_dummy_image("p1.jpg", fill_val=10)
        p2 = self._create_dummy_image("p2.jpg", fill_val=20)
        p3 = self._create_dummy_image("p3.jpg", fill_val=30)

        # Person A embedding = [1, 0, 0], Person B embedding = [0, 1, 0] -> cosine similarity = 0.0
        det_person_b = FaceDetection(
            bbox=np.array([10, 10, 100, 100]),
            landmarks_5pt=np.zeros((5, 2)),
            det_score=0.95,
            yaw_deg=0.0,
            crop_112=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        detector = MockFaceDetector(behavior_map={30: det_person_b})

        res = validate_inputs([p1, p2, p3], detector=detector, same_person_thresh=0.40)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("appear to be different individuals" in err for err in res.errors))

    def test_validation_fails_when_no_frontal_photo(self):
        """Verify validation rejects set if all photos have |yaw| >= 30 degrees."""
        p1 = self._create_dummy_image("p1.jpg", fill_val=10)
        p2 = self._create_dummy_image("p2.jpg", fill_val=20)
        p3 = self._create_dummy_image("p3.jpg", fill_val=30)

        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        det_profile_1 = FaceDetection(np.zeros(4), np.zeros((5,2)), 0.95, yaw_deg=45.0, crop_112=np.zeros((112,112,3)), embedding=emb)
        det_profile_2 = FaceDetection(np.zeros(4), np.zeros((5,2)), 0.95, yaw_deg=-50.0, crop_112=np.zeros((112,112,3)), embedding=emb)
        det_profile_3 = FaceDetection(np.zeros(4), np.zeros((5,2)), 0.95, yaw_deg=35.0, crop_112=np.zeros((112,112,3)), embedding=emb)

        detector = MockFaceDetector(behavior_map={
            10: det_profile_1,
            20: det_profile_2,
            30: det_profile_3,
        })

        res = validate_inputs([p1, p2, p3], detector=detector)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("No near-frontal photo detected" in err for err in res.errors))

    def test_validation_succeeds_on_valid_multi_view_set(self):
        """Verify validation passes on 3 consistent photos with frontal and profile coverage."""
        p1 = self._create_dummy_image("p1.jpg", fill_val=10)
        p2 = self._create_dummy_image("p2.jpg", fill_val=20)
        p3 = self._create_dummy_image("p3.jpg", fill_val=30)

        emb = np.array([0.8, 0.6, 0.0], dtype=np.float32)  # Unit length
        det_frontal = FaceDetection(np.zeros(4), np.zeros((5,2)), 0.98, yaw_deg=2.0, crop_112=np.zeros((112,112,3)), embedding=emb)
        det_left = FaceDetection(np.zeros(4), np.zeros((5,2)), 0.92, yaw_deg=-25.0, crop_112=np.zeros((112,112,3)), embedding=emb)
        det_right = FaceDetection(np.zeros(4), np.zeros((5,2)), 0.91, yaw_deg=30.0, crop_112=np.zeros((112,112,3)), embedding=emb)

        detector = MockFaceDetector(behavior_map={
            10: det_frontal,
            20: det_left,
            30: det_right,
        })

        res = validate_inputs([p1, p2, p3], detector=detector)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)
        self.assertEqual(len(res.detections), 3)

    def test_face_detector_pose_yaw_extraction(self):
        """Verify FaceDetector correctly extracts yaw from face.pose without crashing on undefined yaw."""
        class MockFace:
            bbox = np.array([10, 10, 100, 100])
            kps = np.zeros((5, 2))
            det_score = 0.99
            pose = np.array([5.0, 32.5, -2.0])  # [pitch, yaw, roll]
            normed_embedding = np.ones(512, dtype=np.float32)

        class MockApp:
            def get(self, img):
                return [MockFace()]

        detector = FaceDetector.__new__(FaceDetector)
        detector.has_insightface = True
        detector.app = MockApp()

        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = detector.detect(dummy_img)

        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0].yaw_deg, 32.5, places=2)

if __name__ == "__main__":
    unittest.main()
