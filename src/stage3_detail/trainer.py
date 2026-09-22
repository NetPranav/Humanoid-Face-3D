from __future__ import annotations
import copy, time, os, json, sys
import argparse
from pathlib import Path

# Ensure repository root is on sys.path for standalone and torchrun execution
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import torch
    import torch.distributed as dist
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except ImportError:
    torch = None
    dist = None
    GradScaler = None
    autocast = None
    DataLoader = None
    DistributedSampler = None

from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.discriminator import DetailDiscriminator
from src.stage3_detail.losses import (
    adversarial_loss_g, adversarial_loss_d,
    r1_gradient_penalty, reconstruction_loss_masked,
    identity_preservation_loss,
)
from src.stage3_detail.data import UVDisplacementDataset

HYPERPARAMS = {
    'lr_g': 2e-4,
    'lr_d': 2e-4,
    'adam_betas': (0.0, 0.99),
    'recon_lambda_start': 100.0,
    'recon_lambda_end': 10.0,
    'recon_anneal_steps': 50000,
    'r1_gamma': 10.0,
    'id_lambda': 0.0,  # Kept at 0.0 until differentiable mesh rendering (nvdiffrast) is integrated
    'ema_decay': 0.999,
    'checkpoint_every': 500,
    'max_session_hours': 11.5,
}

def get_recon_lambda(step: int, cfg: dict) -> float:
    t = min(step / cfg['recon_anneal_steps'], 1.0)
    return cfg['recon_lambda_start'] + t * (cfg['recon_lambda_end'] - cfg['recon_lambda_start'])

def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999):
    """Updates EMA model parameters and copies all buffers."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)

def main():
    parser = argparse.ArgumentParser(description="Stage 3 Detail GAN Training")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to preprocessed UV displacement dataset")
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--arcface_checkpoint', type=str, default=None, help="Path to pretrained ArcFace weights")
    parser.add_argument('--id_lambda', type=float, default=0.0, help="Weight for identity loss (keep 0.0 until render pass is wired)")
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=512, help="Target training resolution (e.g. 512 or 1024)")
    parser.add_argument('--checkpoint_every', type=int, default=250, help="Steps between checkpoint saves")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint_latest.pt to resume training")
    args = parser.parse_args()

    HYPERPARAMS['id_lambda'] = args.id_lambda
    HYPERPARAMS['checkpoint_every'] = args.checkpoint_every

    # Multi-GPU DDP setup
    is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if is_distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    generator = DetailGenerator().to(device)
    discriminator = DetailDiscriminator().to(device)
    ema_generator = copy.deepcopy(generator).eval()

    # Optional frozen ArcFace identity preservation model
    arcface_model = None
    if args.arcface_checkpoint and os.path.exists(args.arcface_checkpoint):
        try:
            import sys
            from pathlib import Path
            mica_dir = Path(__file__).resolve().parents[2] / 'vendor' / 'MICA'
            if str(mica_dir) not in sys.path:
                sys.path.insert(0, str(mica_dir))
            from models.arcface import Arcface
            arcface_model = Arcface().to(device)
            try:
                ckpt = torch.load(args.arcface_checkpoint, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(args.arcface_checkpoint, map_location=device)
            state_dict = ckpt['arcface'] if isinstance(ckpt, dict) and 'arcface' in ckpt else ckpt
            arcface_model.load_state_dict(state_dict)
            arcface_model.eval()
            for p in arcface_model.parameters():
                p.requires_grad = False
            if rank == 0:
                print(f"[Stage 3] Initialized frozen ArcFace identity preservation model from {args.arcface_checkpoint}")
        except Exception as e:
            if rank == 0:
                print(f"[Stage 3] Note: Could not load ArcFace model from {args.arcface_checkpoint}: {e}")
            arcface_model = None

    if is_distributed:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[local_rank])
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank])
        g_raw = generator.module
        d_raw = discriminator.module
    else:
        g_raw = generator
        d_raw = discriminator

    opt_g = torch.optim.Adam(generator.parameters(), lr=HYPERPARAMS['lr_g'], betas=HYPERPARAMS['adam_betas'])
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=HYPERPARAMS['lr_d'], betas=HYPERPARAMS['adam_betas'])
    scaler_g = GradScaler('cuda', enabled=torch.cuda.is_available())
    scaler_d = GradScaler('cuda', enabled=torch.cuda.is_available())

    dataset = UVDisplacementDataset(args.data_dir, is_train=True, target_resolution=args.resolution)
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    start_time = time.time()
    step = 0
    saved_step_ckpts = []

    # Invariant 3: Resumability from checkpoint_latest.pt
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and 'generator' in ckpt:
            g_raw.load_state_dict(ckpt['generator'])
            if 'discriminator' in ckpt:
                d_raw.load_state_dict(ckpt['discriminator'])
            if 'opt_g' in ckpt:
                opt_g.load_state_dict(ckpt['opt_g'])
            if 'opt_d' in ckpt:
                opt_d.load_state_dict(ckpt['opt_d'])
            if 'scaler_g' in ckpt and scaler_g is not None and ckpt['scaler_g'] is not None:
                scaler_g.load_state_dict(ckpt['scaler_g'])
            if 'scaler_d' in ckpt and scaler_d is not None and ckpt['scaler_d'] is not None:
                scaler_d.load_state_dict(ckpt['scaler_d'])
            if 'ema_generator' in ckpt:
                ema_generator.load_state_dict(ckpt['ema_generator'])
            step = ckpt.get('step', 0)
            if rank == 0:
                print(f"[Stage 3] Resumed training state from {args.resume} at step {step}")
        elif isinstance(ckpt, dict):
            g_raw.load_state_dict(ckpt)
            if rank == 0:
                print(f"[Stage 3] Loaded generator weights from {args.resume}")

    while step < args.total_steps:
        if sampler:
            sampler.set_epoch(step)
        for batch in dataloader:
            step += 1
            pos_map = batch['pos'].to(device)
            norm_map = batch['norm'].to(device)
            real_disp = batch['disp'].to(device)
            mask = batch['mask'].to(device)
            per_view_feats = batch['per_view_feats'].to(device)
            beta = batch['beta'].to(device)
            psi = batch['psi'].to(device)

            # ── 1. Train Discriminator ─────────────────────────────
            opt_d.zero_grad()
            with autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                fake_disp = generator(pos_map, norm_map, per_view_feats, beta, psi).detach()
                d_real, _, _ = discriminator(real_disp)
                d_fake, _, _ = discriminator(fake_disp)
                d_loss = adversarial_loss_d(d_real, d_fake)

            scaler_d.scale(d_loss).backward()

            # Lazy R1 gradient penalty computed strictly in FP32 (accumulate with d_loss gradients)
            if step % 16 == 0:
                r1_loss = r1_gradient_penalty(d_raw, real_disp, gamma=HYPERPARAMS['r1_gamma'])
                scaler_d.scale(r1_loss * 16.0).backward()

            scaler_d.step(opt_d)
            scaler_d.update()

            # ── 2. Train Generator ─────────────────────────────────
            opt_g.zero_grad()
            with autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                fake_disp = generator(pos_map, norm_map, per_view_feats, beta, psi)
                d_fake_for_g, _, _ = discriminator(fake_disp)

                g_adv = adversarial_loss_g(d_fake_for_g)
                recon_lambda = get_recon_lambda(step, HYPERPARAMS)
                recon_loss = reconstruction_loss_masked(fake_disp, real_disp, mask)
                total_g_loss = g_adv + recon_lambda * recon_loss

                id_loss_val = 0.0
                if arcface_model is not None and 'crop' in batch and HYPERPARAMS['id_lambda'] > 0.0:
                    target_photo = batch['crop'].to(device)
                    id_loss = identity_preservation_loss(fake_disp, target_photo, arcface_model)
                    total_g_loss = total_g_loss + HYPERPARAMS['id_lambda'] * id_loss
                    id_loss_val = float(id_loss.item())

            scaler_g.scale(total_g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            # ── 3. EMA & Checkpoints ──────────────────────────────
            if rank == 0:
                update_ema(ema_generator, g_raw, decay=HYPERPARAMS['ema_decay'])

                # Log progress & diversity metrics every 100 steps
                if step % 100 == 0 or step == 1:
                    batch_std = fake_disp.std().item()
                    d_acc_real = (d_real > 0).float().mean().item() * 100.0
                    d_acc_fake = (d_fake < 0).float().mean().item() * 100.0
                    warning_msg = " [WARN: mode collapse risk]" if batch_std < 0.01 else ""
                    id_msg = f", Id: {id_loss_val:.4f}" if id_loss_val > 0 else ""
                    print(
                        f"Step {step:06d}/{args.total_steps} | "
                        f"D Loss: {d_loss.item():.4f} (R:{d_acc_real:.1f}% F:{d_acc_fake:.1f}%) | "
                        f"G Loss: {total_g_loss.item():.4f} (Adv: {g_adv.item():.4f}, Recon: {recon_loss.item():.4f}{id_msg}, λ: {recon_lambda:.1f}) | "
                        f"Disp Std: {batch_std:.4f}{warning_msg}"
                    )

                if step % HYPERPARAMS['checkpoint_every'] == 0:
                    step_ckpt = f"{args.checkpoint_dir}/generator_step_{step:06d}.pt"
                    torch.save(g_raw.state_dict(), step_ckpt)
                    torch.save({
                        'step': step,
                        'generator': g_raw.state_dict(),
                        'discriminator': d_raw.state_dict(),
                        'opt_g': opt_g.state_dict(),
                        'opt_d': opt_d.state_dict(),
                        'scaler_g': scaler_g.state_dict() if scaler_g else None,
                        'scaler_d': scaler_d.state_dict() if scaler_d else None,
                        'ema_generator': ema_generator.state_dict(),
                    }, f"{args.checkpoint_dir}/checkpoint_latest.pt")
                    torch.save(g_raw.state_dict(), f"{args.checkpoint_dir}/generator_latest.pt")
                    torch.save(ema_generator.state_dict(), f"{args.checkpoint_dir}/ema_generator.pt")

                    # Quota Guard: Prune older step checkpoints to respect Kaggle's 19.5GB limit
                    saved_step_ckpts.append(step_ckpt)
                    while len(saved_step_ckpts) > 2:
                        old_path = saved_step_ckpts.pop(0)
                        if os.path.exists(old_path):
                            try:
                                os.remove(old_path)
                            except OSError:
                                pass

                    print(f"--> Checkpoint saved at step {step:06d} (latest & ema_generator retained, older step files pruned)")
                # Save periodic checkpoint at rank 0
                pass

            # ── 4. Graceful Session Timeout Guard (All Ranks) ─────
            should_stop = torch.tensor([1 if (time.time() - start_time) / 3600 > HYPERPARAMS['max_session_hours'] else 0], device=device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(should_stop, src=0)

            if should_stop.item() == 1:
                if rank == 0:
                    torch.save({
                        'step': step,
                        'generator': g_raw.state_dict(),
                        'discriminator': d_raw.state_dict(),
                        'opt_g': opt_g.state_dict(),
                        'opt_d': opt_d.state_dict(),
                        'scaler_g': scaler_g.state_dict() if scaler_g else None,
                        'scaler_d': scaler_d.state_dict() if scaler_d else None,
                        'ema_generator': ema_generator.state_dict(),
                    }, f"{args.checkpoint_dir}/checkpoint_latest.pt")
                    torch.save(ema_generator.state_dict(), f"{args.checkpoint_dir}/ema_generator.pt")
                    torch.save(g_raw.state_dict(), f"{args.checkpoint_dir}/generator_latest.pt")
                    print(f"Session approaching 11.5 hours. Emergency checkpoint saved at step {step}.")

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    torch.distributed.destroy_process_group()
                return

if __name__ == '__main__':
    main()
