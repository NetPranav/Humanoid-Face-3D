"""
Stage 4 Master Facial Hair & Stubble Generator Orchestrator.

Integrates anatomical region segmentation, procedural stubble micro-displacement,
and static 3D hair card polygonal mesh generation into a unified subsystem.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np

from src.stage4_facial_hair.regions import (
    FacialHairConfig,
    FacialHairRegionSegmenter,
    resolve_hair_config,
    FACIAL_HAIR_PRESETS,
)
from src.stage4_facial_hair.stubble import ProceduralStubbleEngine
from src.stage4_facial_hair.cards import HairCardGenerator


class FacialHairGenerator:
    """
    Master generator for Stage 4 facial hair and stubble assets.
    """
    def __init__(self, neck_collar_threshold: float = 0.20):
        self.neck_collar_threshold = float(neck_collar_threshold)
        self.segmenter = FacialHairRegionSegmenter(neck_collar_threshold=neck_collar_threshold)
        self.stubble_engine = ProceduralStubbleEngine(neck_collar_threshold=neck_collar_threshold)
        self.card_generator = HairCardGenerator(neck_collar_threshold=neck_collar_threshold)

    def generate(
        self,
        neutral_vertices: np.ndarray,
        faces: np.ndarray,
        detail_displacement_mm: Optional[np.ndarray] = None,
        config: Optional[Union[FacialHairConfig, Dict[str, Any], str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        resolution: int = 512,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end facial hair and stubble generation.

        Args:
            neutral_vertices: (N, 3) float array of base head vertices.
            faces: (F, 3) int array of triangle face indices.
            detail_displacement_mm: (H, W) optional incoming Stage 3 displacement map in mm.
            config: Preset name ('stubble', 'full_beard', etc.), FacialHairConfig, or dict.
            output_dir: Target asset directory (e.g. outputs/SESSION_001).
            resolution: Displacement and texture map resolution (default 512).

        Returns:
            dict containing:
              - 'stubble_displacement_mm': (H, W) float32 combined displacement map
              - 'stubble_normal_map': (H, W, 3) uint8 RGB tangent normal map
              - 'stubble_only_mm': (H, W) float32 stubble relief map
              - 'cards_data': dict of hair card geometry arrays
              - 'manifest': dict of generated file paths and metrics
        """
        cfg = resolve_hair_config(config)
        hair_out_dir = Path(output_dir) / "facial_hair" if output_dir else None
        if hair_out_dir:
            hair_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Stage 4] Generating facial hair & stubble (clean_shaven={cfg.is_clean_shaven()})...")

        # 1. Procedural Stubble Micro-Displacement
        stubble_res = self.stubble_engine.generate_stubble_displacement(
            vertices=neutral_vertices,
            faces=faces,
            base_displacement_mm=detail_displacement_mm,
            config=cfg,
            resolution=resolution,
        )

        stubble_files = {}
        if hair_out_dir and cfg.generate_stubble and not cfg.is_clean_shaven():
            stubble_files = self.stubble_engine.save_maps(
                stubble_data=stubble_res,
                output_dir=hair_out_dir,
                prefix="head_stubble",
            )
            print(f"[Stage 4] Saved stubble displacement & normal maps to: {hair_out_dir.name}/")

        # 2. Static 3D Hair Card Polygonal Mesh
        cards_data = self.card_generator.generate_hair_cards(
            vertices=neutral_vertices,
            faces=faces,
            config=cfg,
        )

        card_files = {}
        if hair_out_dir and cards_data['num_cards'] > 0:
            obj_path = hair_out_dir / "facial_hair_cards.obj"
            obj_saved = self.card_generator.export_cards_obj(cards_data, obj_path)
            tex_saved = self.card_generator.export_card_textures(hair_out_dir, resolution=512)

            # Skinning weights JSON
            skin_path = hair_out_dir / "hair_cards_skinning.json"
            skin_payload = {
                "bone_names": ["neck", "head", "jaw", "eye_L", "eye_R"],
                "num_vertices": int(len(cards_data['card_vertices'])),
                "num_cards": int(cards_data['num_cards']),
                "skinning_weights": cards_data['skinning_weights'].tolist(),
            }
            with open(skin_path, 'w') as f:
                json.dump(skin_payload, f, indent=2)

            card_files = {
                "cards_obj": obj_saved,
                "alpha_png": tex_saved["alpha_png"],
                "normal_png": tex_saved["normal_png"],
                "skinning_json": str(skin_path.resolve()),
            }
            print(f"[Stage 4] Saved {cards_data['num_cards']} 3D hair cards and UE5 textures.")

        # 3. Assemble Subsystem Manifest
        manifest = {
            "status": "success",
            "is_clean_shaven": cfg.is_clean_shaven(),
            "config": cfg.to_dict(),
            "stubble": {
                "generated": bool(cfg.generate_stubble and not cfg.is_clean_shaven()),
                "max_stubble_relief_mm": float(np.max(stubble_res['stubble_only_mm'])),
                "maps": stubble_files,
            },
            "hair_cards": {
                "generated": bool(cards_data['num_cards'] > 0),
                "num_cards": int(cards_data['num_cards']),
                "num_vertices": int(len(cards_data['card_vertices'])),
                "num_triangles": int(len(cards_data['card_faces'])),
                "files": card_files,
            },
        }

        if hair_out_dir:
            manifest_path = hair_out_dir / "facial_hair_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

        return {
            "stubble_displacement_mm": stubble_res['displacement_mm'],
            "stubble_normal_map": stubble_res['normal_map'],
            "stubble_only_mm": stubble_res['stubble_only_mm'],
            "cards_data": cards_data,
            "manifest": manifest,
        }
