"""
Unit tests for scripts/extract_arcface_features.py.
"""
import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from scripts.extract_arcface_features import extract_features_for_dataset, find_ground_truth_beta
from src.stage1_identity.data import MICAIdentityDataset


class TestExtractArcFaceFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.img_dir = self.root / "images"
        self.out_dir = self.root / "features"
        self.beta_dir = self.root / "betas"

        self.img_dir.mkdir(parents=True)
        self.out_dir.mkdir(parents=True)
        self.beta_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_ground_truth_beta(self):
        """Verify find_ground_truth_beta loads correctly formatted (300,) beta array."""
        subj_id = "subject042"
        gt_beta = np.linspace(-1.0, 1.0, 300, dtype=np.float32)

        # Save to beta_dir
        np.save(str(self.beta_dir / f"{subj_id}.npy"), {'beta': gt_beta})

        loaded_beta = find_ground_truth_beta(self.beta_dir, subj_id, f"{subj_id}_frontal")
        self.assertEqual(loaded_beta.shape, (300,))
        np.testing.assert_allclose(loaded_beta, gt_beta, atol=1e-6)

        # Nonexistent returns zeros
        dummy_beta = find_ground_truth_beta(self.beta_dir, "nonexistent", "nonexistent")
        self.assertEqual(dummy_beta.shape, (300,))
        self.assertTrue(np.all(dummy_beta == 0.0))

    def test_extract_features_pipeline(self):
        """Verify extract_features_for_dataset processes images and produces loadable MICAIdentityDataset samples."""
        # Create dummy images for 3 subjects
        for i in range(3):
            subj_id = f"sub{i:03d}"
            # Create a synthetic 112x112 image
            img = np.full((112, 112, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(self.img_dir / f"{subj_id}_frontal.jpg"), img)
            # Create matching beta
            beta = np.ones(300, dtype=np.float32) * (i + 1.0)
            np.save(str(self.beta_dir / f"{subj_id}.npy"), {'beta': beta})

        # Run extraction in degraded mode (for CPU test suite without GPU insightface)
        n_processed = extract_features_for_dataset(
            image_dir=str(self.img_dir),
            output_dir=str(self.out_dir),
            beta_dir=str(self.beta_dir),
            allow_degraded=True,
            device='cpu'
        )
        self.assertEqual(n_processed, 3)

        # Verify output files exist
        out_files = list(self.out_dir.glob("*.npz"))
        self.assertEqual(len(out_files), 3)

        # Verify data structure in raw .npz
        sample_npz = np.load(out_files[0])
        self.assertIn('embedding', sample_npz)
        self.assertIn('beta', sample_npz)
        self.assertEqual(sample_npz['embedding'].shape, (512,))
        self.assertEqual(sample_npz['beta'].shape, (300,))

        # Verify loadable by MICAIdentityDataset if PyTorch is installed
        if torch is not None:
            dataset = MICAIdentityDataset(data_dir=str(self.out_dir), split='train', val_ratio=0.0)
            self.assertEqual(len(dataset), 3)
            item = dataset[0]
            self.assertIn('feature', item)
            self.assertIn('beta_gt', item)
            self.assertEqual(item['feature'].shape, (512,))
            self.assertEqual(item['beta_gt'].shape, (300,))


if __name__ == '__main__':
    unittest.main()
