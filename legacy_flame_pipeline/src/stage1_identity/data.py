import os
import glob
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


class MICAIdentityDataset(Dataset):
    """
    Dataset for fine-tuning MICA identity regressor on registered FLAME scans.
    Pairs 512-dim ArcFace feature embeddings (or 112x112 aligned face crops)
    with ground-truth 300-dim FLAME shape coefficients (beta).

    Features subject-stratified splitting to prevent identity leakage across train/val sets.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        val_ratio: float = 0.1,
        seed: int = 42,
        use_embeddings: bool = True
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for MICAIdentityDataset. Install torch>=2.1.0.")

        self.data_dir = Path(data_dir)
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.use_embeddings = use_embeddings

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.samples = self._load_and_split()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid identity samples found in {self.data_dir} for split='{split}'. "
                "Expected paired *.npy or *.npz containing beta and embedding/image."
            )

    def _load_and_split(self) -> List[Path]:
        """
        Discovers all sample files and splits them strictly by Subject ID.
        Expected naming: {subject_id}_{view_id}.npz or {subject_id}.npz
        """
        all_files = sorted(list(self.data_dir.glob("*.npz")) + list(self.data_dir.glob("*.npy")))
        if not all_files:
            return []

        # Group files by subject identifier
        subject_map: Dict[str, List[Path]] = {}
        for f in all_files:
            # Subject ID is prefix before first underscore, or stem if no underscore
            subj_id = f.stem.split('_')[0]
            if subj_id not in subject_map:
                subject_map[subj_id] = []
            subject_map[subj_id].append(f)

        unique_subjects = sorted(list(subject_map.keys()))
        rng = np.random.default_rng(self.seed)
        shuffled_subjects = rng.permutation(unique_subjects).tolist()

        if self.val_ratio <= 0.0:
            n_val = 0
        else:
            n_val = max(1, int(len(shuffled_subjects) * self.val_ratio))
        val_subjects = set(shuffled_subjects[:n_val])
        train_subjects = set(shuffled_subjects[n_val:])

        selected_subjects = train_subjects if self.split == 'train' else val_subjects

        split_samples = []
        for s in selected_subjects:
            split_samples.extend(subject_map[s])

        return sorted(split_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.samples[idx]
        data = np.load(file_path, allow_pickle=True)

        if isinstance(data, np.lib.npyio.NpzFile):
            # .npz archive
            beta_gt = data['beta'].astype(np.float32)  # (300,)
            if self.use_embeddings and 'embedding' in data:
                feature = data['embedding'].astype(np.float32)  # (512,)
                feature = feature / (np.linalg.norm(feature) + 1e-8)
            elif 'crop_112' in data:
                # Image crop in OpenCV BGR -> convert to RGB and normalize [-1, 1]
                crop = data['crop_112'][:, :, ::-1].astype(np.float32)
                feature = (crop - 127.5) / 127.5
                feature = np.transpose(feature, (2, 0, 1))  # (3, 112, 112)
            else:
                raise KeyError(f"Sample {file_path} missing 'embedding' or 'crop_112' array.")
        else:
            # Single .npy dict
            d = data.item() if data.ndim == 0 else data
            beta_gt = d['beta'].astype(np.float32)
            feature = d['embedding'].astype(np.float32)
            feature = feature / (np.linalg.norm(feature) + 1e-8)

        return {
            'feature': torch.from_numpy(feature),
            'beta_gt': torch.from_numpy(beta_gt[:300]),
            'sample_name': file_path.stem
        }
