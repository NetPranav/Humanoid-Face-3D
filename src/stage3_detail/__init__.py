"""
Stage 3: geometric detail.

  sculpt_detail.py  Phase 4A sculpt detail (photo-derived creases + synthesized micro detail)
  fusion.py         roughness / cavity / coat maps from the composite displacement
  rasterizer.py     FLAME UV layout loading and UV-space rasterization helpers
"""
from src.stage3_detail.fusion import MultiTierDetailFusion
from src.stage3_detail.rasterizer import (
    compute_vertex_normals,
    load_flame_geometry_faces,
    load_flame_uv_layout,
    rasterize_uv_maps,
)

__all__ = [
    "MultiTierDetailFusion",
    "load_flame_uv_layout",
    "load_flame_geometry_faces",
    "compute_vertex_normals",
    "rasterize_uv_maps",
]
