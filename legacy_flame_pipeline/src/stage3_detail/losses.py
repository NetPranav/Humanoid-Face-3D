from __future__ import annotations
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

def adversarial_loss_g(d_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss."""
    return F.softplus(-d_fake).mean()

def adversarial_loss_d(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating discriminator loss."""
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

def r1_gradient_penalty(discriminator: torch.nn.Module, real_samples: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """
    R1 gradient penalty regularizing discriminator Lipschitz continuity.
    CRITICAL: Executed strictly in FP32 without autocast to prevent NaN crashes on T4 GPUs.
    """
    real_samples = real_samples.detach().requires_grad_(True)

    # Modern torch.amp.autocast syntax
    device_type = 'cuda' if real_samples.is_cuda else 'cpu'
    with torch.amp.autocast(device_type, enabled=False):
        d_real, _, _ = discriminator(real_samples.float())
        gradients = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real_samples,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        r1 = gradients.pow(2).sum([1, 2, 3]).mean()

    return (gamma / 2.0) * r1

def reconstruction_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    L1 reconstruction loss restricted strictly to valid facial UV regions.
    Prevents generator from learning boundary smoothing over invalid UV unwraps.
    """
    assert mask.sum() > 0, "Validity mask is empty. Verify preprocessing dataset validity masks."
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() + 1e-8)

from typing import Optional

def identity_preservation_loss(
    mesh_render: torch.Tensor,
    input_photo_crop: torch.Tensor,
    arcface_model: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """
    Cosine distance between rendered reconstructed face and input portrait.
    Also accepts direct feature/embedding tensors if pre-extracted.
    """
    if arcface_model is not None:
        with torch.no_grad():
            target_emb = arcface_model(input_photo_crop)
        pred_emb = arcface_model(mesh_render)
    else:
        pred_emb = mesh_render
        target_emb = input_photo_crop

    target_emb = F.normalize(target_emb.float(), dim=-1)
    pred_emb = F.normalize(pred_emb.float(), dim=-1)

    return 1.0 - (target_emb * pred_emb).sum(dim=-1).mean()

def photometric_consistency_loss(shaded_render: torch.Tensor, shaded_target: torch.Tensor) -> torch.Tensor:
    """Penalizes directional illumination shading discrepancies to prevent surface inversion."""
    return F.l1_loss(shaded_render, shaded_target)


def high_frequency_fft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    high_pass_cutoff: float = 0.1
) -> torch.Tensor:
    """
    Frequency-domain spectral loss using 2D Fast Fourier Transform (FFT).
    Directly penalizes the loss of high-frequency power (pores, wrinkles, micro-creases)
    to prevent GAN mode collapse under L1 dominance.
    """
    if mask is not None:
        p = pred * mask
        t = target * mask
    else:
        p = pred
        t = target

    # 2D Real FFT across spatial dimensions (H, W)
    fft_p = torch.fft.rfft2(p.float(), norm='ortho')
    fft_t = torch.fft.rfft2(t.float(), norm='ortho')

    # Log magnitude spectra
    mag_p = torch.log(torch.abs(fft_p) + 1e-6)
    mag_t = torch.log(torch.abs(fft_t) + 1e-6)

    # High-pass filter mask: frequency radii using original spatial dimensions
    orig_h, orig_w = pred.shape[-2], pred.shape[-1]
    fy = torch.fft.fftfreq(orig_h, device=pred.device)[:, None]
    fx = torch.fft.rfftfreq(orig_w, device=pred.device)[None, :]
    freq_radius = torch.sqrt(fy ** 2 + fx ** 2)
    hp_weight = (freq_radius >= high_pass_cutoff).float()

    diff = torch.abs(mag_p - mag_t) * hp_weight
    return diff.mean()


def gradient_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Penalizes first-order spatial gradient discrepancies along X and Y axes.
    Preserves razor-sharp skin pores and micro-edges.
    """
    # Horizontal gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Vertical gradients
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = torch.abs(pred_dx - target_dx)
    loss_y = torch.abs(pred_dy - target_dy)

    if mask is not None:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        loss = (loss_x * mask_x).sum() / (mask_x.sum() + 1e-8) + (loss_y * mask_y).sum() / (mask_y.sum() + 1e-8)
    else:
        loss = loss_x.mean() + loss_y.mean()

    return loss

