import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import copy, time, os, json
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.discriminator import DetailDiscriminator
from src.stage3_detail.losses import (
    adversarial_loss_g, adversarial_loss_d,
    r1_gradient_penalty, reconstruction_loss_masked,
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
    'ema_decay': 0.999,
    'checkpoint_every': 500,
    'max_session_hours': 11.5,
}

def get_recon_lambda(step: int, cfg: dict) -> float:
    t = min(step / cfg['recon_anneal_steps'], 1.0)
    return cfg['recon_lambda_start'] + t * (cfg['recon_lambda_end'] - cfg['recon_lambda_start'])

def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

def main():
    parser = argparse.ArgumentParser(description="Stage 3 Detail GAN Training")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to preprocessed UV displacement dataset")
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

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
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    dataset = UVDisplacementDataset(args.data_dir, is_train=True)
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    start_time = time.time()
    step = 0

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
            with autocast(dtype=torch.float16):
                fake_disp = generator(pos_map, norm_map, per_view_feats, beta, psi).detach()
                d_real, _, _ = discriminator(real_disp)
                d_fake, _, _ = discriminator(fake_disp)
                d_loss = adversarial_loss_d(d_real, d_fake)

            scaler_d.scale(d_loss).backward()

            # Lazy R1 gradient penalty computed strictly in FP32
            if step % 16 == 0:
                opt_d.zero_grad()
                r1_loss = r1_gradient_penalty(d_raw, real_disp, gamma=HYPERPARAMS['r1_gamma'])
                scaler_d.scale(r1_loss * 16.0).backward()

            scaler_d.step(opt_d)
            scaler_d.update()

            # ── 2. Train Generator ─────────────────────────────────
            opt_g.zero_grad()
            with autocast(dtype=torch.float16):
                fake_disp = generator(pos_map, norm_map, per_view_feats, beta, psi)
                d_fake_for_g, _, _ = discriminator(fake_disp)

                g_adv = adversarial_loss_g(d_fake_for_g)
                recon_lambda = get_recon_lambda(step, HYPERPARAMS)
                recon_loss = reconstruction_loss_masked(fake_disp, real_disp, mask)
                total_g_loss = g_adv + recon_lambda * recon_loss

            scaler_g.scale(total_g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            # ── 3. EMA & Checkpoints ──────────────────────────────
            if rank == 0:
                update_ema(ema_generator, g_raw, decay=HYPERPARAMS['ema_decay'])

                if step % HYPERPARAMS['checkpoint_every'] == 0:
                    torch.save(g_raw.state_dict(), f"{args.checkpoint_dir}/generator_step_{step:06d}.pt")
                    torch.save(ema_generator.state_dict(), f"{args.checkpoint_dir}/ema_generator.pt")
                    print(f"Step {step:06d} | D Loss: {d_loss.item():.4f} | G Loss: {total_g_loss.item():.4f} | Recon: {recon_loss.item():.4f}")

                # Emergency checkpoint before Kaggle 12hr session kill
                if (time.time() - start_time) / 3600 > HYPERPARAMS['max_session_hours']:
                    torch.save(ema_generator.state_dict(), f"{args.checkpoint_dir}/ema_generator.pt")
                    print(f"Session approaching 11.5 hours. Emergency checkpoint saved at step {step}.")
                    return

if __name__ == '__main__':
    main()
