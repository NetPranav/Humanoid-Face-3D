"""
Stage 4: Facial Hair & Stubble Geometry Subsystem.

Provides procedural follicular stubble micro-displacement and static 3D hair card polygonal mesh
generation for Unreal Engine 5 and game-engine humanoid assets.
"""
from src.stage4_facial_hair.regions import (
    FacialHairConfig,
    FacialHairRegionSegmenter,
    resolve_hair_config,
    FACIAL_HAIR_PRESETS,
    compute_collar_pinning_mask,
    compute_normalized_coordinates,
)
from src.stage4_facial_hair.stubble import (
    ProceduralStubbleEngine,
)
from src.stage4_facial_hair.cards import (
    HairCardGenerator,
)
from src.stage4_facial_hair.generator import (
    FacialHairGenerator,
)

__all__ = [
    "FacialHairConfig",
    "FacialHairRegionSegmenter",
    "resolve_hair_config",
    "FACIAL_HAIR_PRESETS",
    "compute_collar_pinning_mask",
    "compute_normalized_coordinates",
    "ProceduralStubbleEngine",
    "HairCardGenerator",
    "FacialHairGenerator",
]
