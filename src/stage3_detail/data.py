from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import random

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    class Dataset:
        pass

class UVDisplacementDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed UV displacement map pairs,
    neutral position & normal conditioning maps, and facial validity masks.
    Enforces subject-stratified splits and strict [-1, 1] normalization matching tanh.
    """
    def __init__(self, data_dir: str, is_train: bool = True, target_resolution: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.target_resolution = target_resolution
        self.disp_files = sorted(list(self.data_dir.glob('*_disp.png')))
        if not self.disp_files:
            raise RuntimeError(
                f"No *_disp.png files found in {data_dir}. "
                "Ensure scripts/build_uv_displacement_dataset.py has executed successfully."
            )

        # Robust subject ID extraction (supports 'subject_001_...', 'sub001_...', 'carell_...')
        def get_subject_id(p: Path) -> str:
            stem = p.stem
            if stem.startswith('subject_'):
                return '_'.join(stem.split('_')[:2])
            return stem.split('_')[0]

        self.get_subject_id = get_subject_id
        subjects = sorted({get_subject_id(p) for p in self.disp_files})
        rng = random.Random(1337)
        rng.shuffle(subjects)
        if len(subjects) <= 1:
            # If only 1 subject exists (e.g. pilot/testing), use for train or allow overfit validation
            val_subjects = set() if is_train else set(subjects)
        else:
            val_count = max(1, int(len(subjects) * 0.15))
            val_subjects = set(subjects[:val_count])

        self.files = [
            p for p in self.disp_files
            if (get_subject_id(p) in val_subjects) != is_train
        ]

        if len(self.files) == 0:
            if is_train:
                raise RuntimeError(
                    f"Zero training samples available in {data_dir}. "
                    f"Total subjects found: {len(subjects)}. At least 2 subjects required for strict train/val split."
                )
            else:
                # In pilot/single-subject mode, fallback to using train sample for validation evaluation
                self.files = self.disp_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        disp_path = self.files[idx]
        stem = disp_path.stem.replace('_disp', '')
        pos_path = self.data_dir / f"{stem}_pos.png"
        norm_path = self.data_dir / f"{stem}_norm.png"
        mask_path = self.data_dir / f"{stem}_mask.png"
        meta_path = self.data_dir / f"{stem}_meta.npz"
        maps_path = self.data_dir / f"{stem}_maps.npz"

        # Read displacement with 16-bit -> [-1, 1] contract
        raw_disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if raw_disp.dtype == np.uint16:
            disp = (raw_disp.astype(np.float32) / 65535.0) * 2.0 - 1.0
        else:
            disp = raw_disp.astype(np.float32)
            if disp.max() > 1.0:
                disp = (disp / 255.0) * 2.0 - 1.0
        pos = cv2.imread(str(pos_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        norm = cv2.imread(str(norm_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        if self.target_resolution is not None:
            r = self.target_resolution
            disp = cv2.resize(disp, (r, r), interpolation=cv2.INTER_LINEAR)
            pos = cv2.resize(pos, (r, r), interpolation=cv2.INTER_LINEAR)
            norm = cv2.resize(norm, (r, r), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (r, r), interpolation=cv2.INTER_NEAREST)

        disp = torch.from_numpy(disp).unsqueeze(0)
        pos = torch.from_numpy(pos).permute(2, 0, 1)
        norm = torch.from_numpy(norm).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Load metadata or maps
        resolved_npz = meta_path if meta_path.exists() else (maps_path if maps_path.exists() else None)
        if resolved_npz is not None:
            meta = np.load(resolved_npz)
            beta = torch.from_numpy(meta['beta']).float() if 'beta' in meta else torch.zeros(300, dtype=torch.float32)
            psi = torch.from_numpy(meta['psi']).float() if 'psi' in meta else torch.zeros(100, dtype=torch.float32)
            if 'per_view_feats' in meta:
                pv = meta['per_view_feats']
                if pv.ndim == 1:
                    pv = pv[np.newaxis, :]
                per_view_feats = torch.from_numpy(pv).float()
            else:
                per_view_feats = torch.zeros((1, 512), dtype=torch.float32)
        else:
            beta = torch.zeros(300, dtype=torch.float32)
            psi = torch.zeros(100, dtype=torch.float32)
            per_view_feats = torch.zeros((1, 512), dtype=torch.float32)

        return {
            'disp': disp,
            'pos': pos,
            'norm': norm,
            'mask': mask,
            'beta': beta,
            'psi': psi,
            'per_view_feats': per_view_feats,
        }
