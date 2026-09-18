import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

class UVDisplacementDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed UV displacement map pairs,
    neutral position & normal conditioning maps, and facial validity masks.
    """
    def __init__(self, data_dir: str, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.disp_files = sorted(list(self.data_dir.glob('*_disp.png')))
        if not self.disp_files:
            # Provide dummy fallback sample for local pipeline testing / dry runs
            self.files = []
        else:
            split_idx = int(0.9 * len(self.disp_files))
            self.files = self.disp_files[:split_idx] if is_train else self.disp_files[split_idx:]

    def __len__(self) -> int:
        return max(len(self.files), 1)

    def __getitem__(self, idx: int) -> dict:
        if not self.files:
            # Fallback zero-filled tensors
            return {
                'disp': torch.zeros((1, 512, 512), dtype=torch.float32),
                'pos': torch.zeros((3, 512, 512), dtype=torch.float32),
                'norm': torch.zeros((3, 512, 512), dtype=torch.float32),
                'mask': torch.ones((1, 512, 512), dtype=torch.float32),
                'beta': torch.zeros(300, dtype=torch.float32),
                'psi': torch.zeros(100, dtype=torch.float32),
                'per_view_feats': torch.zeros((3, 512), dtype=torch.float32),
            }

        disp_path = self.files[idx]
        stem = disp_path.stem.replace('_disp', '')
        pos_path = self.data_dir / f"{stem}_pos.png"
        norm_path = self.data_dir / f"{stem}_norm.png"
        mask_path = self.data_dir / f"{stem}_mask.png"
        meta_path = self.data_dir / f"{stem}_meta.npz"

        disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        pos = cv2.imread(str(pos_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        norm = cv2.imread(str(norm_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        disp = torch.from_numpy(disp).unsqueeze(0)
        pos = torch.from_numpy(pos).permute(2, 0, 1)
        norm = torch.from_numpy(norm).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if meta_path.exists():
            meta = np.load(meta_path)
            beta = torch.from_numpy(meta['beta']).float()
            psi = torch.from_numpy(meta['psi']).float()
            per_view_feats = torch.from_numpy(meta['per_view_feats']).float()
        else:
            beta = torch.zeros(300, dtype=torch.float32)
            psi = torch.zeros(100, dtype=torch.float32)
            per_view_feats = torch.zeros((3, 512), dtype=torch.float32)

        return {
            'disp': disp,
            'pos': pos,
            'norm': norm,
            'mask': mask,
            'beta': beta,
            'psi': psi,
            'per_view_feats': per_view_feats,
        }
