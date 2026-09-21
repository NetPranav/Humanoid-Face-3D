"""
Stage 7: AI Delighting + UV Inpainting.

Removes environment lighting from projected textures to produce clean
diffuse albedo maps, and fills unseen UV regions using procedural
Gaussian dilation or optional neural inpainting (LaMa).
"""
from src.stage7_delight.delight_net import (
    DelightingPipeline,
    DelightUNet,
    UVInpainter,
)

__all__ = [
    "DelightingPipeline",
    "DelightUNet",
    "UVInpainter",
]
