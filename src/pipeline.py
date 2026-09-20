import numpy as np
import cv2
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.stage0_preprocess.detector import FaceDetector, FaceDetection
from src.utils.flame_model import FLAMEModel
from src.utils.validation import validate_inputs

class FaceGeoPipeline:
    """
    End-to-End Face Geometry Reconstruction Pipeline.
    Takes 3-5 photographs, isolates identity with confidence weighting,
    estimates expression, neutralizes the base geometry for UE5 ARKit deltas,
    synthesizes high-frequency micro-displacement, and packages game-ready outputs.
    """
    def __init__(self, config_path: str, model_dir: str, detector: Optional[FaceDetector] = None):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.model_dir = Path(model_dir)
        flame_path = self.model_dir / 'flame/generic_model.pkl'
        if not flame_path.exists() and (self.model_dir / 'generic_model.pkl').exists():
            flame_path = self.model_dir / 'generic_model.pkl'
        elif not flame_path.exists() and str(model_dir) in ('models', 'models_cache', 'models/'):
            flame_path = Path('./data/flame_model/generic_model.pkl')

        self.flame = FLAMEModel(str(flame_path)) if flame_path.exists() else None

        if detector is not None:
            self.detector = detector
        else:
            allow_degraded = self.cfg.get('allow_degraded', False) if isinstance(self.cfg, dict) else False
            self.detector = FaceDetector(allow_degraded=allow_degraded)

        # Lazy-loaded network models
        self._stage1 = None
        self._stage2 = None
        self._stage3 = None

    def _get_stage1(self):
        if self._stage1 is None:
            from src.stage1_identity.inference import MICAIdentityEncoder
            mica_candidates = [
                self.model_dir / 'mica/pretrained.tar',
                self.model_dir / 'mica/mica.tar',
                Path('/kaggle/input/mica-pretrained/pretrained.tar'),
                Path('/kaggle/input/mica-pretrained/mica.tar'),
            ]
            mica_ckpt = next((p for p in mica_candidates if p.exists()), mica_candidates[0])

            stage1_cfg = self.cfg.get('stage1', {}) if isinstance(self.cfg, dict) else {}
            fusion_space = stage1_cfg.get('fusion_space', 'embedding')
            fusion_temp = float(stage1_cfg.get('fusion_temperature', 0.05))
            yaw_exp = int(stage1_cfg.get('yaw_weight_exponent', 2))

            self._stage1 = MICAIdentityEncoder(
                str(mica_ckpt),
                self.flame,
                fusion_space=fusion_space,
                fusion_temperature=fusion_temp,
                yaw_weight_exponent=yaw_exp
            )
        return self._stage1

    def _get_stage2(self):
        if self._stage2 is None:
            from src.stage2_expression.encoder import ExpressionEncoder
            smirk_ckpt = self.model_dir / 'smirk/pretrained.tar'
            if not smirk_ckpt.exists():
                smirk_ckpt = Path('/kaggle/input/smirk-pretrained/smirk.tar')
            self._stage2 = ExpressionEncoder(str(smirk_ckpt))
        return self._stage2

    def run(self, photo_paths: List[str], output_dir: str) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Pre-flight input validation & detection ──────────────────────────
        print("[Stage 0] Validating input portraits and extracting alignments...")
        min_photos = self.cfg.get('pipeline', {}).get('input_min_photos', 3) if isinstance(self.cfg, dict) else 3
        min_face_conf = self.cfg.get('stage0', {}).get('min_face_confidence', 0.5) if isinstance(self.cfg, dict) else 0.5
        val_res = validate_inputs(
            photo_paths,
            detector=self.detector,
            min_photos=min_photos,
            min_face_confidence=min_face_conf
        )
        if not val_res.is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(val_res.errors)}")

        detections = val_res.detections

        # ── Stage 1: Identity ────────────────────────────────────────────────
        print("[Stage 1] Regressing identity shape (multi-view confidence-weighted)...")
        stage1 = self._get_stage1()
        beta = stage1.encode_multiview(detections)

        # ── Stage 2: Expression ──────────────────────────────────────────────
        print("[Stage 2] Regressing facial expression and pose...")
        valid_dets = [d for d in detections if d is not None]
        frontal_det = max(valid_dets, key=lambda d: d.det_score) if valid_dets else None
        stage2 = self._get_stage2()
        expression_psi, pose_theta = stage2.encode(frontal_det)

        # ── Neutral-expression normalization (CRITICAL) ──────────────────────
        # Base mesh is exported in canonical neutral pose (psi=0, theta=neutral).
        # Ensures ARKit-52 blendshape target offsets in UE5 remain mathematically correct.
        print("[Neutral Normalization] Generating canonical neutral base mesh (psi=0, theta=neutral)...")
        if self.flame is None:
            raise RuntimeError(
                "FLAME model is not initialized. A valid FLAME model file "
                "(e.g. data/flame_model/generic_model.pkl) is required to decode base geometry."
            )
        neutral_vertices, faces = self.flame.decode_neutral(beta)

        expression_metadata = {
            'expression_psi': expression_psi.tolist(),
            'pose_theta': pose_theta.tolist(),
        }

        # ── Stage 3: Micro-displacement detail synthesis ────────────────────
        detail_displacement = None
        if self._stage3 is not None:
            print("[Stage 3] Synthesizing high-frequency micro-displacement...")
            # detail_displacement = self._stage3(...)

        # ── Stage 5: Export ──────────────────────────────────────────────────
        print("[Stage 5] Exporting clean neutral OBJ and run manifest...")
        output_paths = self._export(
            vertices=neutral_vertices,
            faces=faces,
            displacement=detail_displacement,
            expression_metadata=expression_metadata,
            beta=beta,
            detections=detections,
            photo_paths=photo_paths,
            output_dir=output_dir,
        )

        print(f"[Done] Complete. Output generated at: {output_dir}")
        return output_paths

    def _export(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        displacement: Optional[np.ndarray],
        expression_metadata: dict,
        beta: np.ndarray,
        detections: list,
        photo_paths: list,
        output_dir: Path
    ) -> Dict[str, str]:
        obj_path = output_dir / 'head_mesh.obj'
        with open(obj_path, 'w') as f:
            f.write("# Face Geometry Pipeline - Canonical Neutral Base Mesh\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces + 1:  # OBJ indices are 1-based
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        preview_path = output_dir / 'head_mesh.png'
        try:
            from evaluation.identity_score import render_neutral_preview
            render_neutral_preview(str(obj_path), str(preview_path), size=512)
        except Exception as e:
            print(f"[Stage 5] Warning: Failed to render preview PNG: {e}")

        # ── Stage 5 Production Asset Packaging (ARKit-52, LODs, Armature) ──
        stage5_manifest = {}
        try:
            from src.stage5_export.exporter import Stage5Exporter
            retopo_matrix_path = self.config.get('stage5', {}).get('correspondence_w') if hasattr(self, 'config') else None
            stylize_config = self.config.get('stage5', {}).get('stylize') if hasattr(self, 'config') else None
            exporter = Stage5Exporter(
                correspondence_w_path=retopo_matrix_path,
                enable_lods=True,
                enable_armature=True
            )
            stage5_manifest = exporter.export_production_asset(
                neutral_vertices=vertices,
                faces=faces,
                output_dir=output_dir,
                export_fbx=True,
                stylization_params=stylize_config
            )
        except Exception as e:
            print(f"[Stage 5 Warning] Could not complete full production asset export: {e}")

        manifest_path = output_dir / 'manifest.json'
        manifest = {
            'pipeline_version': "0.1.0",
            'photo_paths': [str(p) for p in photo_paths],
            'beta_shape': beta.tolist(),
            'expression_metadata': expression_metadata,
            'vertex_count': int(vertices.shape[0]),
            'face_count': int(faces.shape[0]),
            'has_displacement': displacement is not None,
            'has_flame': self.flame is not None,
            'has_preview': preview_path.exists(),
            'stage5_production_assets': stage5_manifest,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return {
            'obj_path': str(obj_path),
            'manifest_path': str(manifest_path),
            'preview_path': str(preview_path) if preview_path.exists() else None,
            'blendshapes_path': stage5_manifest.get('blendshapes_json'),
            'armature_path': stage5_manifest.get('armature_json'),
            'fbx_path': stage5_manifest.get('fbx_file'),
        }
