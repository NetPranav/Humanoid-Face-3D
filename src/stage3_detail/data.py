from pathlib import Path
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
    def __init__(self, data_dir: str, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.disp_files = sorted(list(self.data_dir.glob('*_disp.png')))
        if not self.disp_files:
            raise RuntimeError(
                f"No *_disp.png files found in {data_dir}. "
                "Ensure scripts/build_uv_displacement_dataset.py has executed successfully."
            )

        # Subject-stratified split (prevents same subject appearing in both train and val)
        subjects = sorted({p.stem.split('_')[0] for p in self.disp_files})
        rng = random.Random(1337)
        rng.shuffle(subjects)
        val_count = max(1, int(len(subjects) * 0.1))
        val_subjects = set(subjects[:val_count])

        self.files = [
            p for p in self.disp_files
            if (p.stem.split('_')[0] in val_subjects) != is_train
        ]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        disp_path = self.files[idx]
        stem = disp_path.stem.replace('_disp', '')
        pos_path = self.data_dir / f"{stem}_pos.png"
        norm_path = self.data_dir / f"{stem}_norm.png"
        mask_path = self.data_dir / f"{stem}_mask.png"
        meta_path = self.data_dir / f"{stem}_meta.npz"

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

        disp = torch.from_numpy(disp).unsqueeze(0)
        pos = torch.from_numpy(pos).permute(2, 0, 1)
        norm = torch.from_numpy(norm).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if meta_path.exists():
            meta = np.load(meta_path)
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
