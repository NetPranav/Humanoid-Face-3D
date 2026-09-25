"""
Stage 7: UV completion (Phase 3A).

  uv_fill.py         provenance-aware fill: observed / mirrored / synthesized / interpolated,
                     seam-free mesh-harmonic colour, lip-only lip fill
  skin_synthesis.py  the subject's own skin grain quilted into unseen areas

Delighting itself happens during projection (src/stage6_texture/projector.py, fitted SH).
"""
from src.stage7_delight.uv_fill import ProvenanceUVFill, load_or_build_mirror_map

__all__ = ["ProvenanceUVFill", "load_or_build_mirror_map"]
