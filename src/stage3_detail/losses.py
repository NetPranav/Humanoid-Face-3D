import torch
import torch.nn.functional as F

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

    with torch.cuda.amp.autocast(enabled=False):
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

def identity_preservation_loss(mesh_render: torch.Tensor, input_photo_crop: torch.Tensor, arcface_model: torch.nn.Module) -> torch.Tensor:
    """Cosine distance between rendered reconstructed face and input portrait."""
    with torch.no_grad():
        target_emb = arcface_model(input_photo_crop)

    pred_emb = arcface_model(mesh_render)
    target_emb = F.normalize(target_emb, dim=-1)
    pred_emb = F.normalize(pred_emb, dim=-1)

    return 1.0 - (target_emb * pred_emb).sum(dim=-1).mean()
