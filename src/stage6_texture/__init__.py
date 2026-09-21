"""
Stage 6: Multi-View UV Texture Projection.

Backprojects input photos onto the 3D FLAME mesh UV space using
per-view camera projection, angle-weighted cosine blending, and
z-buffer visibility testing. Produces a raw projected texture map
ready for delighting in Stage 7.
"""
from src.stage6_texture.projector import (
    MultiViewTextureProjector,
    estimate_camera_projection_matrix,
)

__all__ = [
    "MultiViewTextureProjector",
    "estimate_camera_projection_matrix",
]
