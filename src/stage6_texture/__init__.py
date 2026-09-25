"""
Stage 6: per-texel UV texture backprojection onto the fitted (posed + expressed) mesh,
with z-buffer occlusion and skin-parsing masks. See projector.py.
"""
from src.stage6_texture.projector import MultiViewTextureProjector, rasterize_uv

__all__ = ["MultiViewTextureProjector", "rasterize_uv"]
