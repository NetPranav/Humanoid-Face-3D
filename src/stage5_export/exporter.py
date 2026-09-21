"""
Stage 5 Master Exporter Orchestrator.

Integrates retopology, ARKit-52 blendshape generation, multi-resolution LOD decimation,
skeletal armature rigging, and headless Blender FBX export into a unified pipeline.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np

from src.stage5_export.retopology import (
    SparseMatrixCSR,
    transfer_retopology,
    load_correspondence_matrix
)
from src.stage5_export.blendshapes import (
    synthesize_canonical_arkit_deltas,
    transfer_blendshapes_deformation,
    export_blendshapes_json,
    ARKIT_52_NAMES
)
from src.stage5_export.lod import (
    generate_lod_chain,
    export_lod_chain_assets,
    LOD_TRIANGLE_TARGETS
)
from src.stage5_export.armature import (
    estimate_anatomical_joint_centers,
    compute_linear_skinning_weights,
    export_armature_json
)
from src.stage5_export.stylize import (
    FaceStylizer,
    StylizationParameters,
    resolve_stylization_params
)


class Stage5Exporter:
    """
    Production export engine for game-engine-ready facial assets.
    """
    def __init__(
        self,
        correspondence_w_path: Optional[str] = None,
        enable_lods: bool = True,
        enable_armature: bool = True,
    ):
        self.correspondence_w: Optional[SparseMatrixCSR] = None
        if correspondence_w_path and Path(correspondence_w_path).exists():
            print(f"[Stage 5] Loading retopology correspondence matrix from: {correspondence_w_path}")
            self.correspondence_w = load_correspondence_matrix(correspondence_w_path)

        self.enable_lods = enable_lods
        self.enable_armature = enable_armature

    def export_production_asset(
        self,
        neutral_vertices: np.ndarray,
        faces: np.ndarray,
        output_dir: Union[str, Path],
        template_blendshapes: Optional[Dict[str, np.ndarray]] = None,
        export_fbx: bool = True,
        stylization_params: Optional[Union[StylizationParameters, Dict[str, float], str]] = None,
        detail_maps: Optional[Dict[str, str]] = None,
        facial_hair_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full Stage 5 production export workflow.
        Returns a dictionary manifest detailing all generated assets.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print("\n[Stage 5 Export] Initiating game asset generation...")

        # 1. Retopology (Optional target correspondence mapping)
        if self.correspondence_w is not None:
            print("[Stage 5] Applying retopology correspondence matrix W...")
            active_vertices = transfer_retopology(neutral_vertices, self.correspondence_w)
            active_faces = faces  # Target faces should match target topology
        else:
            print(f"[Stage 5] Using base topology ({len(neutral_vertices)} vertices, {len(faces)} faces)...")
            active_vertices = neutral_vertices.copy()
            active_faces = faces.copy()

        # 1.5 Parametric Facial Stylization (Heroic / Chiseled / Sliders)
        stylization_info = None
        if stylization_params is not None:
            resolved_p = resolve_stylization_params(stylization_params)
            if not resolved_p.is_neutral():
                stylizer = FaceStylizer()
                active_vertices, stylize_deltas = stylizer.apply_stylization(active_vertices, resolved_p)
                max_disp = float(np.max(np.linalg.norm(stylize_deltas, axis=1)))
                print(f"[Stage 5] Applied parametric facial stylization (max disp: {max_disp:.3f})")
                stylization_info = {
                    "parameters": resolved_p.to_dict(),
                    "max_displacement": max_disp
                }

        # 2. ARKit-52 Blendshape Generation

        print("[Stage 5] Synthesizing ARKit-52 semantic blendshapes...")
        if template_blendshapes is not None:
            blendshapes = transfer_blendshapes_deformation(
                template_neutral=neutral_vertices,
                template_deltas=template_blendshapes,
                subject_neutral=active_vertices
            )
        else:
            blendshapes = synthesize_canonical_arkit_deltas(active_vertices)

        bs_json_path = out_path / "blendshapes_arkit52.json"
        export_blendshapes_json(blendshapes, bs_json_path)
        print(f"[Stage 5] Saved 52 ARKit blendshapes to: {bs_json_path.name}")

        # 3. Multi-resolution LOD Generation (LOD0 -> LOD3)
        lod_manifest = {}
        if self.enable_lods:
            print("[Stage 5] Generating 4-tier LOD chain (LOD0, LOD1, LOD2, LOD3) with preserved shape keys...")
            lod_chain = generate_lod_chain(
                lod0_vertices=active_vertices,
                lod0_faces=active_faces,
                lod0_blendshapes=blendshapes
            )
            lod_manifest = export_lod_chain_assets(lod_chain, out_path)
            for lod_tier, info in lod_manifest.items():
                print(f"  - {lod_tier}: {info['num_triangles']} tris, {info['num_vertices']} verts, {info['shape_keys_count']} shape keys")

        # 4. Skeletal Armature Rigging
        armature_path = None
        if self.enable_armature:
            print("[Stage 5] Estimating 5-joint anatomical armature & computing LBS skinning weights...")
            joints = estimate_anatomical_joint_centers(active_vertices)
            skin_weights = compute_linear_skinning_weights(active_vertices, joints)
            armature_file = out_path / "armature_rig.json"
            export_armature_json(joints, skin_weights, armature_file)
            armature_path = str(armature_file.resolve())
            print(f"[Stage 5] Saved skeletal rig to: {armature_file.name}")

        # 5. Export base OBJ
        base_obj_path = out_path / "head_mesh_neutral.obj"
        with open(base_obj_path, "w") as fp:
            fp.write(f"# Face Geometry Pipeline - Neutral Base Mesh\n")
            for vert in active_vertices:
                fp.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
            for face in active_faces + 1:
                fp.write(f"f {face[0]} {face[1]} {face[2]}\n")

        # 6. Headless Blender Production FBX Packaging (Mesh + Armature + ARKit-52 + Hair Cards + LODs + Materials)
        fbx_output_path = out_path / "head_mesh_ue5_livelink.fbx"
        fbx_manifest = {}

        if export_fbx:
            from src.stage5_export.fbx_packager import FBXPackager

            lod_manifest_file = out_path / "lod_manifest.json"
            if lod_manifest:
                with open(lod_manifest_file, "w") as fp:
                    json.dump(lod_manifest, fp, indent=2)

            hair_cards_obj = None
            hair_skinning_json = None
            if facial_hair_manifest and isinstance(facial_hair_manifest, dict):
                cards_info = facial_hair_manifest.get("hair_cards", {}).get("files", {})
                hair_cards_obj = cards_info.get("cards_obj")
                hair_skinning_json = cards_info.get("skinning_json")

            normal_map_file = None
            disp_map_file = None
            if detail_maps and isinstance(detail_maps, dict):
                normal_map_file = detail_maps.get("normal_png") or detail_maps.get("normal_map") or detail_maps.get("stubble_normal")
                disp_map_file = detail_maps.get("displacement_png") or detail_maps.get("displacement_map") or detail_maps.get("stubble_displacement")

            render_preview_path = out_path / "preview_render.png"

            packager = FBXPackager()
            fbx_manifest = packager.package_fbx(
                mesh_obj=base_obj_path,
                blendshapes_json=bs_json_path,
                output_fbx=fbx_output_path,
                armature_json=armature_path,
                lod_manifest=str(lod_manifest_file) if lod_manifest else None,
                hair_cards_obj=hair_cards_obj,
                hair_skinning_json=hair_skinning_json,
                normal_map=normal_map_file,
                displacement_map=disp_map_file,
                render_preview=render_preview_path,
            )

        # 7. Package complete export manifest
        manifest = {
            "status": "success",
            "neutral_base_obj": str(base_obj_path.resolve()),
            "blendshapes_json": str(bs_json_path.resolve()),
            "armature_json": armature_path,
            "fbx_file": fbx_manifest.get("fbx_file"),
            "fbx_packaging": fbx_manifest,
            "lods": lod_manifest,
            "stylization": stylization_info,
            "detail_maps": detail_maps,
            "facial_hair": facial_hair_manifest,
            "num_vertices": int(len(active_vertices)),
            "num_triangles": int(len(active_faces)),
            "arkit_blendshapes_count": len(blendshapes),
        }

        manifest_file = out_path / "stage5_export_manifest.json"
        with open(manifest_file, "w") as fp:
            json.dump(manifest, fp, indent=2)

        print(f"[Stage 5 Export Complete] Manifest saved to: {manifest_file.name}\n")
        return manifest
