"""
Stage 8: PBR Material Stack Derivation.

Generates physically based rendering maps (roughness, cavity/AO, SSS thickness)
from existing geometry and albedo data. All operations are procedural —
no neural networks or GPU training required.
"""
from src.stage8_pbr.material_stack import (
    PBRMaterialStack,
    RoughnessMapGenerator,
    CavityMapGenerator,
    SSSThicknessGenerator,
)

__all__ = [
    "PBRMaterialStack",
    "RoughnessMapGenerator",
    "CavityMapGenerator",
    "SSSThicknessGenerator",
]
