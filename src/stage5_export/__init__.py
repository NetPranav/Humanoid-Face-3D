"""
Stage 5 Production Retopology, Rigging, LOD, and Game Engine Export.
"""
from src.stage5_export.retopology import (
    SparseMatrixCSR,
    build_correspondence_matrix,
    save_correspondence_matrix,
    load_correspondence_matrix,
    transfer_retopology,
    compute_retopology_error
)
from src.stage5_export.blendshapes import (
    ARKIT_52_NAMES,
    synthesize_canonical_arkit_deltas,
    transfer_blendshapes_deformation,
    export_blendshapes_json,
    load_blendshapes_json
)
from src.stage5_export.lod import (
    generate_lod_chain,
    export_lod_chain_assets,
    reproject_blendshapes_to_lod,
    decimate_mesh,
    LOD_TRIANGLE_TARGETS
)
from src.stage5_export.armature import (
    JOINT_NAMES,
    JOINT_HIERARCHY,
    estimate_anatomical_joint_centers,
    compute_linear_skinning_weights,
    export_armature_json,
    load_armature_json
)
from src.stage5_export.stylize import (
    StylizationParameters,
    STYLIZATION_PRESETS,
    resolve_stylization_params,
    FaceStylizer
)
from src.stage5_export.exporter import Stage5Exporter
from src.stage5_export.fbx_packager import FBXPackager
from src.stage5_export.metahuman_bridge import MetaHumanBridgeExporter

__all__ = [
    "SparseMatrixCSR",
    "build_correspondence_matrix",
    "save_correspondence_matrix",
    "load_correspondence_matrix",
    "transfer_retopology",
    "compute_retopology_error",
    "ARKIT_52_NAMES",
    "synthesize_canonical_arkit_deltas",
    "transfer_blendshapes_deformation",
    "export_blendshapes_json",
    "load_blendshapes_json",
    "generate_lod_chain",
    "export_lod_chain_assets",
    "reproject_blendshapes_to_lod",
    "decimate_mesh",
    "LOD_TRIANGLE_TARGETS",
    "JOINT_NAMES",
    "JOINT_HIERARCHY",
    "estimate_anatomical_joint_centers",
    "compute_linear_skinning_weights",
    "export_armature_json",
    "load_armature_json",
    "StylizationParameters",
    "STYLIZATION_PRESETS",
    "resolve_stylization_params",
    "FaceStylizer",
    "Stage5Exporter",
    "FBXPackager",
    "MetaHumanBridgeExporter",
]

