from __future__ import annotations
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import spectral_norm
except ImportError:
    torch = None
    class _MockModule:
        pass
    class _MockNN:
        Module = _MockModule
    nn = _MockNN()
    def spectral_norm(module, *args, **kwargs):
        return module
from typing import Tuple, Optional

class DetailDiscriminator(nn.Module):
    """
    PatchGAN discriminator with Spectral Normalization for UV displacement maps.
    Outputs a 2D spatial grid of real/fake logits (evaluating local 70x70 patches),
    with optional auxiliary heads for identity and expression classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_identities: Optional[int] = None,
        num_expressions: Optional[int] = None
    ):
        super().__init__()
        bc = base_channels

        def conv_block(in_c: int, out_c: int, stride: int = 2) -> nn.Sequential:
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, bc, kernel_size=4, stride=2, padding=1))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv_block(bc, bc * 2, stride=2)      # 256 -> 128
        self.conv3 = conv_block(bc * 2, bc * 4, stride=2)  # 128 -> 64
        self.conv4 = conv_block(bc * 4, bc * 8, stride=1)  # 64 -> 63

        # PatchGAN real/fake patch grid output
        self.out_patch = spectral_norm(nn.Conv2d(bc * 8, 1, kernel_size=4, stride=1, padding=1))

        # Auxiliary classifier heads
        self.id_head = nn.Linear(bc * 8, num_identities) if num_identities else None
        self.expr_head = nn.Linear(bc * 8, num_expressions) if num_expressions else None
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = self.act1(self.conv1(x))
        h = self.conv2(h)
        h = self.conv3(h)
        feat = self.conv4(h)

        out_patch = self.out_patch(feat)

        out_id = None
        out_expr = None
        if self.id_head is not None or self.expr_head is not None:
            pooled = self.gap(feat).flatten(1)
            if self.id_head is not None:
                out_id = self.id_head(pooled)
            if self.expr_head is not None:
                out_expr = self.expr_head(pooled)

        return out_patch, out_id, out_expr
