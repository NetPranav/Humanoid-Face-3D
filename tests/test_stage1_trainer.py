import unittest
import tempfile
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from src.stage1_identity.data import MICAIdentityDataset

class TestStage1TrainerAndData(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @unittest.skipIf(torch is None, "PyTorch required for dataset testing")
    def test_dataset_raises_on_empty_directory(self):
        """Verify MICAIdentityDataset raises RuntimeError when no sample files exist."""
        with self.assertRaises(RuntimeError) as ctx:
            MICAIdentityDataset(str(self.temp_path))
        self.assertIn("No valid identity samples found", str(ctx.exception))

    @unittest.skipIf(torch is None, "PyTorch required for dataset testing")
    def test_subject_stratified_split_prevents_leakage(self):
        """Verify all views for a given subject remain strictly in one split (no leakage)."""
        rng = np.random.default_rng(42)
        # Create 10 subjects with 2 views each
        for i in range(10):
            subj_id = f"subj{i:02d}"
            for v in range(2):
                file_path = self.temp_path / f"{subj_id}_view{v}.npz"
                emb = rng.normal(0, 1, 512).astype(np.float32)
                beta = rng.normal(0, 1, 300).astype(np.float32)
                np.savez(file_path, embedding=emb, beta=beta)

        train_ds = MICAIdentityDataset(str(self.temp_path), split='train', val_ratio=0.2, seed=42)
        val_ds = MICAIdentityDataset(str(self.temp_path), split='val', val_ratio=0.2, seed=42)

        train_subjects = {Path(p).stem.split('_')[0] for p in train_ds.samples}
        val_subjects = {Path(p).stem.split('_')[0] for p in val_ds.samples}

        # Assert no intersection between train and val subjects
        intersection = train_subjects.intersection(val_subjects)
        self.assertEqual(len(intersection), 0, f"Subject leakage detected: {intersection}")
        self.assertGreater(len(train_subjects), 0)
        self.assertGreater(len(val_subjects), 0)
        self.assertEqual(len(train_subjects) + len(val_subjects), 10)

    @unittest.skipIf(torch is None, "PyTorch required for tensor testing")
    def test_dataset_item_format(self):
        """Verify dataset returns expected tensor dimensions and normalized embeddings."""
        rng = np.random.default_rng(42)
        file_path = self.temp_path / "subj01_view0.npz"
        emb = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32) # norm = 5.0
        beta = rng.normal(0, 1, 300).astype(np.float32)
        np.savez(file_path, embedding=emb, beta=beta)

        ds = MICAIdentityDataset(str(self.temp_path), split='train', val_ratio=0.0)
        sample = ds[0]

        self.assertIn('feature', sample)
        self.assertIn('beta_gt', sample)
        self.assertEqual(sample['feature'].shape, (512,))
        self.assertEqual(sample['beta_gt'].shape, (300,))

        # Verify unit normalization on embedding: norm([3/5, 4/5, ...]) == 1.0
        feature_norm = torch.norm(sample['feature']).item()
        self.assertAlmostEqual(feature_norm, 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
