import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewAttention(nn.Module):
    """
    Cross-attention mechanism: UV-space bottleneck spatial queries attend to
    pooled per-view camera image features to resolve fine geometric details.
    """
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

    def forward(self, query_features: torch.Tensor, per_view_features: torch.Tensor) -> torch.Tensor:
        """
        query_features: (B, HW, C) - flattened bottleneck spatial features
        per_view_features: (B, N, C) - pooled image features from N views
        """
        kv = self.feature_proj(per_view_features)
        out, _ = self.cross_attn(query_features, kv, kv)
        return out

class DetailGenerator(nn.Module):
    """
    U-Net generator synthesizing 512x512 UV displacement maps.
    Conditioned on coarse position and normal maps rasterized from the neutral base mesh,
    multi-view features, and identity/expression codes via AdaIN.
    """
    def __init__(
        self,
        in_channels: int = 6,        # position (3) + normal (3)
        out_channels: int = 1,       # scalar displacement
        base_channels: int = 64,
        backbone_feature_dim: int = 512,
        identity_dim: int = 300,
        expression_dim: int = 100,
    ):
        super().__init__()
        bc = base_channels

        # Encoder path (512 -> 256 -> 128 -> 64 -> 32 -> 16)
        self.enc1 = self._conv_block(in_channels, bc)
        self.enc2 = self._conv_block(bc, bc * 2)
        self.enc3 = self._conv_block(bc * 2, bc * 4)
        self.enc4 = self._conv_block(bc * 4, bc * 8)
        self.enc5 = self._conv_block(bc * 8, bc * 8)
        self.pool = nn.AvgPool2d(2)

        # Cross-attention bottleneck
        bottleneck_c = bc * 8
        self.mv_attn = MultiViewAttention(bottleneck_c)

        # AdaIN identity and expression conditioning projection
        self.style_proj = nn.Linear(identity_dim + expression_dim, bottleneck_c * 2)

        # Decoder path (bilinear interpolation + conv to eliminate checkerboard artifacts)
        self.dec5 = self._conv_block(bc * 16, bc * 8)
        self.dec4 = self._conv_block(bc * 16, bc * 4)
        self.dec3 = self._conv_block(bc * 8, bc * 2)
        self.dec2 = self._conv_block(bc * 4, bc)
        self.dec1 = nn.Conv2d(bc * 2, out_channels, kernel_size=1)

        self.out_act = nn.Tanh()  # Outputs bounded in [-1, 1]

    def _conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def adain(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        mean, std = style.chunk(2, dim=1)
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        std = std.unsqueeze(-1).unsqueeze(-1) + 1e-8
        x_norm = (x - x.mean([2, 3], keepdim=True)) / (x.std([2, 3], keepdim=True) + 1e-8)
        return x_norm * std + mean

    def forward(
        self,
        pos_map: torch.Tensor,
        norm_map: torch.Tensor,
        per_view_feats: torch.Tensor,
        beta: torch.Tensor,
        psi: torch.Tensor
    ) -> torch.Tensor:
        """
        Synthesizes UV displacement map.
        pos_map: (B, 3, 512, 512) neutral coarse position map
        norm_map: (B, 3, 512, 512) neutral coarse normal map
        per_view_feats: (B, N, C) pooled view features
        beta: (B, 300) identity parameters
        psi: (B, 100) expression parameters
        """
        x = torch.cat([pos_map, norm_map], dim=1)  # (B, 6, 512, 512)

        # Style injection
        style = self.style_proj(torch.cat([beta, psi], dim=1))

        # Downsample
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        bottleneck = self.pool(e5)

        # Modulate with style vector
        bottleneck = self.adain(bottleneck, style)

        # Cross-attention across views
        B, C, H, W = bottleneck.shape
        q = bottleneck.reshape(B, C, H * W).permute(0, 2, 1)
        q = self.mv_attn(q, per_view_feats)
        bottleneck = q.permute(0, 2, 1).reshape(B, C, H, W)

        def up(t): return F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)

        # Upsample with skip connections
        d5 = self.dec5(torch.cat([up(bottleneck), e5], dim=1))
        d4 = self.dec4(torch.cat([up(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([up(d3), e2], dim=1))
        out = self.dec1(torch.cat([up(d2), e1], dim=1))

        return self.out_act(out)
