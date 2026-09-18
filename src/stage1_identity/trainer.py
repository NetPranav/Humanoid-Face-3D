import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
except ImportError:
    torch = None

from src.stage1_identity.inference import MappingNetwork
from src.stage1_identity.data import MICAIdentityDataset
from src.utils.flame_model import FLAMEModel, N_VERTS, N_SHAPE


def setup_ddp() -> Tuple[int, int, int]:
    """Initializes distributed process group if running under torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class MICAIdentityTrainer:
    """
    Fine-tuning trainer for MICA 512->300 MLP regressor on registered demographic scans.
    Supports PyTorch 2.4+ AMP (torch.amp.autocast/GradScaler) and DDP multi-GPU (T4x2).
    Includes 11.5-hour Kaggle wall-clock safeguard.
    """
    def __init__(
        self,
        data_dir: str,
        flame_model_path: str,
        checkpoint_dir: str,
        pretrained_ckpt: Optional[str] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        vert_loss_lambda: float = 1.0,
        batch_size: int = 16,
        max_epochs: int = 50,
        max_hours: float = 11.5
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for MICAIdentityTrainer. Install torch>=2.1.0.")

        self.rank, self.world_size, self.local_rank = setup_ddp()
        self.is_main = (self.rank == 0)
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.lr = lr
        self.weight_decay = weight_decay
        self.vert_loss_lambda = vert_loss_lambda
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_seconds = max_hours * 3600.0
        self.start_time = time.time()

        # 1. Load FLAME shapedirs for 3D vertex loss
        self.flame = FLAMEModel(flame_model_path, scale_to_mm=False)
        # shapedirs shape: (5023, 3, 300)
        self.shapedirs = torch.from_numpy(self.flame.shapedirs).float().to(self.device)

        # 2. Build model
        self.model = MappingNetwork(z_dim=512, map_hidden_dim=300, map_output_dim=300, hidden=3).to(self.device)
        if pretrained_ckpt and Path(pretrained_ckpt).exists():
            if self.is_main:
                print(f"[Trainer] Loading pretrained weights from {pretrained_ckpt}...")
            ckpt = torch.load(pretrained_ckpt, map_location=self.device)
            state = ckpt.get('flameModel', ckpt.get('state_dict', ckpt))
            clean_state = {k.replace('regressor.', ''): v for k, v in state.items() if 'generator' not in k}
            self.model.load_state_dict(clean_state, strict=False)

        if self.world_size > 1:
            self.ddp_model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.ddp_model = self.model

        # 3. Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=1e-6)

        # 4. Modern PyTorch 2.4+ AMP GradScaler
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # 5. Datasets & Loaders
        self.train_dataset = MICAIdentityDataset(str(self.data_dir), split='train', val_ratio=0.1)
        self.val_dataset = MICAIdentityDataset(str(self.data_dir), split='val', val_ratio=0.1)

        self.train_sampler = DistributedSampler(self.train_dataset) if self.world_size > 1 else None
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.best_val_loss = float('inf')

    def compute_loss(self, pred_beta: torch.Tensor, gt_beta: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Computes combined L1 shape parameter loss and L1 3D vertex reconstruction loss.
        """
        # Shape coefficient loss
        loss_beta = torch.mean(torch.abs(pred_beta - gt_beta))

        # 3D Vertex loss in millimetres (native FLAME units in metres -> multiply by 1000)
        # shapedirs: (5023, 3, 300)
        beta_diff = pred_beta - gt_beta  # (B, 300)
        vert_diff = torch.einsum('bk,vck->bvc', beta_diff, self.shapedirs) * 1000.0  # (B, 5023, 3) in mm
        loss_vert = torch.mean(torch.abs(vert_diff))

        total_loss = loss_beta + self.vert_loss_lambda * loss_vert
        return total_loss, float(loss_beta.item()), float(loss_vert.item())

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.ddp_model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        total_loss_acc = 0.0
        beta_loss_acc = 0.0
        vert_loss_acc = 0.0
        n_batches = 0

        for batch in self.train_loader:
            features = batch['feature'].to(self.device, non_blocking=True)
            beta_gt = batch['beta_gt'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                pred_beta = self.ddp_model(features)
                loss, l_b, l_v = self.compute_loss(pred_beta, beta_gt)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_acc += loss.item()
            beta_loss_acc += l_b
            vert_loss_acc += l_v
            n_batches += 1

            # Safeguard: Check session wall-clock timer
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_seconds:
                if self.is_main:
                    print(f"\n[Safeguard Alert] Session reached {elapsed/3600:.2f}h. Triggering emergency checkpoint...")
                self.save_checkpoint(epoch, is_emergency=True)
                sys.exit(0)

        self.scheduler.step()

        return {
            'loss': total_loss_acc / max(1, n_batches),
            'loss_beta': beta_loss_acc / max(1, n_batches),
            'loss_vert_mm': vert_loss_acc / max(1, n_batches)
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.ddp_model.eval()
        total_loss_acc = 0.0
        beta_loss_acc = 0.0
        vert_loss_acc = 0.0
        n_batches = 0

        for batch in self.val_loader:
            features = batch['feature'].to(self.device, non_blocking=True)
            beta_gt = batch['beta_gt'].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                pred_beta = self.ddp_model(features)
                loss, l_b, l_v = self.compute_loss(pred_beta, beta_gt)

            total_loss_acc += loss.item()
            beta_loss_acc += l_b
            vert_loss_acc += l_v
            n_batches += 1

        val_loss = total_loss_acc / max(1, n_batches)
        return {
            'val_loss': val_loss,
            'val_loss_beta': beta_loss_acc / max(1, n_batches),
            'val_vert_mm': vert_loss_acc / max(1, n_batches)
        }

    def save_checkpoint(self, epoch: int, is_emergency: bool = False):
        if not self.is_main:
            return

        state = {
            'epoch': epoch,
            'regressor': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'flame_version': self.flame.version if hasattr(self.flame, 'version') else 'FLAME2020',
        }

        tag = 'emergency' if is_emergency else f'epoch_{epoch:03d}'
        ckpt_path = self.checkpoint_dir / f'regressor_{tag}.pt'
        torch.save(state, ckpt_path)
        torch.save(state, self.checkpoint_dir / 'regressor_latest.pt')
        print(f"[Trainer] Checkpoint saved: {ckpt_path}")

    def fit(self):
        if self.is_main:
            print(f"[Trainer] Starting Stage 1 Fine-Tuning across {self.world_size} device(s)...")
            print(f"Train samples: {len(self.train_dataset)} | Val samples: {len(self.val_dataset)}")

        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            if self.is_main:
                print(
                    f"Epoch {epoch:02d}/{self.max_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} (Vert: {train_metrics['loss_vert_mm']:.3f}mm) | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} (Vert: {val_metrics['val_vert_mm']:.3f}mm) | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    torch.save(self.model.state_dict(), self.checkpoint_dir / 'regressor_best.pt')
                    print(f"  --> Saved new best checkpoint (Val Vert: {val_metrics['val_vert_mm']:.3f}mm)")

                if epoch % 10 == 0:
                    self.save_checkpoint(epoch)

        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MICA Identity Regressor")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to preprocessed training data (.npz)")
    parser.add_argument('--flame_path', type=str, default='data/flame_model/generic_model.pkl')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/stage1_identity')
    parser.add_argument('--pretrained', type=str, default=None, help="Path to initial MICA checkpoint")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_hours', type=float, default=11.5)
    args = parser.parse_args()

    trainer = MICAIdentityTrainer(
        data_dir=args.data_dir,
        flame_model_path=args.flame_path,
        checkpoint_dir=args.checkpoint_dir,
        pretrained_ckpt=args.pretrained,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        max_hours=args.max_hours
    )
    trainer.fit()


if __name__ == '__main__':
    main()
