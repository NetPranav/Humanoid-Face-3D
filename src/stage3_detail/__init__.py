"""
Stage 3 Micro-Detail GAN Module.
"""
from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.discriminator import DetailDiscriminator
from src.stage3_detail.inference import DetailSynthesizer
from src.stage3_detail.rasterizer import (
    load_flame_uv_layout,
    compute_vertex_normals,
    rasterize_uv_maps
)

MultiScalePatchGAN = DetailDiscriminator

__all__ = [
    'DetailGenerator',
    'DetailDiscriminator',
    'MultiScalePatchGAN',
    'DetailSynthesizer',
    'load_flame_uv_layout',
    'compute_vertex_normals',
    'rasterize_uv_maps',
]
