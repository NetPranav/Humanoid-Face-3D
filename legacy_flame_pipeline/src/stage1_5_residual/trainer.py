"""
Trainer for Stage 1.5: Macro-Shape Non-Linear Residual Network.

Trains the graph convolutional vertex displacement predictor to correct
FLAME's linear PCA inaccuracies against ground-truth 3D head scans.

Adheres strictly to Humanoid-Face-3D Kaggle & System Invariants:
1. Zero silent fallbacks — explicit FileNotFoundError/RuntimeError.
2. Inode/Disk quota — sliding window of 2 checkpoints + checkpoint_latest.pt.
3. Resumability & Emergency checkpoint triggered at 11.5 hours.
4. Neck Seam Contract — collar vertices (y_norm <= 0.20) strictly pinned to 0.
5. Dynamic GPU topology — auto-detects single-GPU or multi-GPU (T4x2/DDP).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
except ImportError:
    torch = None
    nn = None
    F = None
    Dataset = object

from src.utils.flame_model import FLAMEModel, N_VERTS, N_SHAPE
from src.stage1_5_residual.residual_net import MacroShapeResidualNet


class ResidualScanDataset(Dataset):
    """
    Dataset pairing FLAME base vertices / shape parameters and ArcFace embeddings
    with ground-truth registered 3D scan vertices (5,023 vertices).
    """

    def __init__(self, data_dir: str, split: str = 'train', val_ratio: float = 0.1, seed: int = 42):
        if torch is None:
            raise RuntimeError("PyTorch required for ResidualScanDataset")

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Scan dataset directory not found: {self.data_dir}")

        all_files = sorted(list(self.data_dir.glob("*.npz")) + list(self.data_dir.glob("*.npy")))
        if not all_files:
            raise RuntimeError(
                f"No scan samples found in {self.data_dir}. "
                "Expected *.npz files with 'flame_vertices', 'feature', and 'gt_vertices'."
            )

        # Stratify by subject ID
        subject_map: Dict[str, List[Path]] = {}
        for f in all_files:
            subj_id = f.stem.split('_')[0]
            if subj_id not in subject_map:
                subject_map[subj_id] = []
            subject_map[subj_id].append(f)

        unique_subjects = sorted(list(subject_map.keys()))
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_subjects).tolist()

        n_val = max(1, int(len(shuffled) * val_ratio))
        val_subjects = set(shuffled[:n_val])
        train_subjects = set(shuffled[n_val:])

        chosen_subjects = train_subjects if split == 'train' else val_subjects
        self.samples = []
        for s in chosen_subjects:
            self.samples.extend(subject_map[s])
        self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath = self.samples[idx]
        data = np.load(fpath, allow_pickle=True)

        flame_v = data['flame_vertices'].astype(np.float32)  # (5023, 3)
        feature = data['feature'].astype(np.float32)        # (512,)
        gt_v = data['gt_vertices'].astype(np.float32)        # (5023, 3)

        # Normalize feature if not already unit length
        fnorm = np.linalg.norm(feature)
        if fnorm > 1e-8:
            feature = feature / fnorm

        return {
            'flame_vertices': torch.from_numpy(flame_v),
            'feature': torch.from_numpy(feature),
            'gt_vertices': torch.from_numpy(gt_v),
        }


class Stage15Trainer:
    """
    Production trainer for Stage 1.5 Macro-Shape Residual Network.
    """

    def __init__(
        self,
        flame_model: FLAMEModel,
        data_dir: str,
        output_dir: str,
        pretrained_ckpt: Optional[str] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_laplacian: float = 0.1,
        lambda_reg: float = 0.01,
        batch_size: int = 8,
        max_epochs: int = 50,
        max_hours: float = 11.5,
    ):
        if torch is None:
            raise RuntimeError("PyTorch required for Stage15Trainer")

        self.flame = flame_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_seconds = max_hours * 3600.0
        self.start_time = time.time()

        self.lambda_laplacian = lambda_laplacian
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        # DDP & Device Setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = (self.local_rank == 0)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            if self.world_size > 1 and not dist.is_initialized():
                dist.init_process_group(backend="nccl")
        else:
            self.device = torch.device("cpu")

        # Build Model
        self.model = MacroShapeResidualNet(flame_model=self.flame).to(self.device)
        if pretrained_ckpt and Path(pretrained_ckpt).exists():
            if self.is_main:
                print(f"[Stage1.5 Trainer] Loading weights from: {pretrained_ckpt}")
            ckpt = torch.load(pretrained_ckpt, map_location=self.device)
            state = ckpt.get('state_dict', ckpt)
            self.model.load_state_dict(state, strict=False)

        if self.world_size > 1:
            self.ddp_model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.ddp_model = self.model

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=1e-6)
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Datasets
        self.train_dataset = ResidualScanDataset(data_dir, split='train')
        self.val_dataset = ResidualScanDataset(data_dir, split='val')

        sampler = DistributedSampler(self.train_dataset) if self.world_size > 1 else None
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.step_checkpoints: List[Path] = []
        self.best_val_loss = float('inf')

    def compute_loss(
        self,
        pred_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
        delta_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Composite loss:
        1. L1 vertex position loss in mm: ||pred - gt||
        2. Laplacian smoothness loss: ||L . delta_v||^2
        3. Regularization: mean(||delta_v||^2) to keep displacement bounded
        """
        # Vertex loss in millimeters (FLAME native in meters -> * 1000)
        loss_vert = torch.mean(torch.abs(pred_vertices - gt_vertices)) * 1000.0

        # Laplacian smoothness loss
        raw_model = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
        loss_lap = raw_model.compute_laplacian_loss(delta_v)

        # Magnitude regularization
        loss_reg = torch.mean(delta_v ** 2)

        total_loss = loss_vert + self.lambda_laplacian * loss_lap + self.lambda_reg * loss_reg

        # Invariant check: verify collar violation
        collar_viol = raw_model.compute_collar_violation(delta_v)
        if collar_viol.item() > 1e-4:
            raise RuntimeError(
                f"[Neck Seam Contract Violation] Max collar displacement {collar_viol.item():.6f} > 0.0! "
                "Collar vertices MUST be strictly pinned to 0."
            )

        metrics = {
            'loss_vert_mm': float(loss_vert.item()),
            'loss_lap': float(loss_lap.item()),
            'loss_reg': float(loss_reg.item()),
            'loss_total': float(total_loss.item()),
            'collar_max_disp': float(collar_viol.item()),
        }
        return total_loss, metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.ddp_model.train()
        acc_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            flame_v = batch['flame_vertices'].to(self.device, non_blocking=True)
            feature = batch['feature'].to(self.device, non_blocking=True)
            gt_v = batch['gt_vertices'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                pred_v, delta_v = self.ddp_model(flame_v, feature)
                loss, metrics = self.compute_loss(pred_v, gt_v, delta_v)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc_loss += loss.item()
            n_batches += 1

            # 11.5h Emergency checkpoint safeguard
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_seconds:
                if self.is_main:
                    print(f"\n[Safeguard Alert] Session reached {elapsed/3600:.2f}h. Triggering emergency checkpoint...")
                self.save_checkpoint(epoch, is_emergency=True)
                sys.exit(0)

        return {'train_loss': acc_loss / max(1, n_batches)}

    def save_checkpoint(self, epoch: int, is_emergency: bool = False):
        if not self.is_main:
            return

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        if is_emergency:
            path = self.output_dir / 'checkpoint_emergency.pt'
            torch.save(state, path)
            print(f"[Stage 1.5] Emergency checkpoint saved to {path}")
            return

        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(state, latest_path)

        # Sliding window of 2 step checkpoints
        step_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, step_path)
        self.step_checkpoints.append(step_path)
        while len(self.step_checkpoints) > 2:
            old = self.step_checkpoints.pop(0)
            if old.exists():
                old.unlink()
