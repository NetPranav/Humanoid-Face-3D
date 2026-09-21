import numpy as np
import cv2
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

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
    def __init__(
        self,
        config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
        model_dir: Union[str, Path] = "models_cache",
        detector: Optional[FaceDetector] = None,
        cfg: Optional[Dict[str, Any]] = None
    ):
        if cfg is not None:
            self.cfg = cfg
        elif isinstance(config_path, dict):
            self.cfg = config_path
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}

        self.model_dir = Path(model_dir) if model_dir is not None else Path("models_cache")
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
        self._stage4 = None

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
            self._stage2 = ExpressionEncoder(str(smirk_ckpt), allow_neutral=True)
        return self._stage2

    def _get_stage3(self):
        if self._stage3 is None:
            stage3_cfg = self.cfg.get('stage3', {}) if isinstance(self.cfg, dict) else {}
            if not stage3_cfg.get('enabled', True):
                return None
            try:
                from src.stage3_detail.inference import DetailSynthesizer
                ckpt_p = stage3_cfg.get('checkpoint')
                stats_p = stage3_cfg.get('stats')
                uv_template_p = stage3_cfg.get('uv_template')
                self._stage3 = DetailSynthesizer(
                    checkpoint_path=ckpt_p,
                    stats_path=stats_p,
                    uv_template_path=uv_template_p
                )
            except (FileNotFoundError, RuntimeError) as e:
                print(f"[Stage 3 Notice] Detail GAN not initialized: {e}")
                self._stage3 = None
        return self._stage3

    def _get_stage4(self):
        if self._stage4 is None:
            stage4_cfg = self.cfg.get('stage4', {}) if isinstance(self.cfg, dict) else {}
            if not stage4_cfg.get('enabled', True):
                return None
            try:
                from src.stage4_facial_hair.generator import FacialHairGenerator
                neck_collar = float(stage4_cfg.get('neck_collar_threshold', 0.20))
                self._stage4 = FacialHairGenerator(neck_collar_threshold=neck_collar)
            except Exception as e:
                print(f"[Stage 4 Notice] Facial hair generator not initialized: {e}")
                self._stage4 = None
        return self._stage4

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

        # Optional Pixel3DMM dense per-pixel fitting refinement
        stage1_cfg = self.cfg.get('stage1', {}) if isinstance(self.cfg, dict) else {}
        if stage1_cfg.get('enable_dense_fitting', False) and self.flame is not None:
            print("[Stage 1 Upgrade] Refining β with Pixel3DMM dense per-pixel fitting...")
            try:
                import torch
                from src.stage1_identity.pixel3dmm_fitter import DenseFLAMEFitter
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                fitter = DenseFLAMEFitter(
                    self.flame,
                    n_iterations=int(stage1_cfg.get('dense_iterations', 100)),
                    device=device
                )
                frontal_img = frontal_det.aligned_face if frontal_det is not None else None
                if frontal_img is not None:
                    fit_res = fitter.fit(frontal_img, initial_beta=beta)
                    beta = fit_res['beta']
            except Exception as e:
                print(f"[Stage 1 Warning] Pixel3DMM dense fitting skipped: {e}")

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

        # ── Stage 1.5: Macro-Shape Non-Linear Residual Correction ────────────
        stage1_5_cfg = self.cfg.get('stage1_5', {}) if isinstance(self.cfg, dict) else {}
        residual_ckpt = self.model_dir / 'stage1_5/checkpoint_latest.pt'
        if stage1_5_cfg.get('enabled', False) or residual_ckpt.exists():
            print("[Stage 1.5] Applying macro-shape non-linear residual correction...")
            try:
                import torch
                from src.stage1_5_residual.residual_net import MacroShapeResidualNet
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                res_net = MacroShapeResidualNet(flame_model=self.flame).to(device)
                if residual_ckpt.exists():
                    ckpt = torch.load(residual_ckpt, map_location=device)
                    res_net.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
                res_net.eval()
                with torch.no_grad():
                    v_in = torch.from_numpy(neutral_vertices).float().unsqueeze(0).to(device)
                    id_feats = [d.embedding for d in valid_dets if hasattr(d, 'embedding') and d.embedding is not None]
                    if id_feats:
                        id_feat = torch.from_numpy(np.mean(id_feats, axis=0)).float().unsqueeze(0).to(device)
                    else:
                        id_feat = torch.zeros(1, 512, device=device)
                    corrected_v, delta_v = res_net(v_in, id_feat)
                    neutral_vertices = corrected_v.squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"[Stage 1.5 Warning] Residual correction skipped: {e}")

        expression_metadata = {
            'expression_psi': expression_psi.tolist(),
            'pose_theta': pose_theta.tolist(),
        }

        # ── Stage 3: Micro-displacement detail synthesis ────────────────────
        detail_maps = None
        detail_displacement = None
        stage3 = self._get_stage3()
        if stage3 is not None:
            print("[Stage 3] Synthesizing high-frequency micro-displacement & normal maps...")
            try:
                stage3_cfg = self.cfg.get('stage3', {}) if isinstance(self.cfg, dict) else {}
                res = int(stage3_cfg.get('resolution', 512))
                id_feats = [d.embedding for d in valid_dets if hasattr(d, 'embedding') and d.embedding is not None]
                per_view_feats = np.stack(id_feats, axis=0) if id_feats else None

                synth_res = stage3.synthesize(
                    neutral_vertices=neutral_vertices,
                    faces=faces,
                    per_view_feats=per_view_feats,
                    beta=beta,
                    psi=expression_psi,
                    resolution=res
                )
                detail_maps = stage3.save_maps(synth_res, output_dir=output_dir, prefix="head")
                detail_displacement = synth_res['disp_mm']
                print(f"[Stage 3] Synthesized {res}x{res} 16-bit displacement and normal maps.")
            except Exception as e:
                print(f"[Stage 3 Warning] Micro-displacement synthesis failed: {e}")

        # ── Stage 4: Facial Hair & Stubble Geometry ──────────────────────────
        facial_hair_manifest = None
        stage4 = self._get_stage4()
        if stage4 is not None:
            stage4_cfg = self.cfg.get('stage4', {}) if isinstance(self.cfg, dict) else {}
            hair_preset = stage4_cfg.get('preset', 'stubble')
            hair_res = int(stage4_cfg.get('resolution', 512))
            try:
                hair_res_data = stage4.generate(
                    neutral_vertices=neutral_vertices,
                    faces=faces,
                    detail_displacement_mm=detail_displacement,
                    config=stage4_cfg.get('config', hair_preset),
                    output_dir=output_dir,
                    resolution=hair_res,
                )
                facial_hair_manifest = hair_res_data.get('manifest')
                if facial_hair_manifest and facial_hair_manifest.get('stubble', {}).get('generated', False):
                    detail_displacement = hair_res_data['stubble_displacement_mm']
                    stubble_maps = facial_hair_manifest['stubble'].get('maps', {})
                    if stubble_maps:
                        if detail_maps is None:
                            detail_maps = {}
                        detail_maps['stubble_displacement'] = stubble_maps.get('displacement_png')
                        detail_maps['stubble_normal'] = stubble_maps.get('normal_png')
            except Exception as e:
                print(f"[Stage 4 Warning] Facial hair generation failed: {e}")

        # ── Stage 5: Export ──────────────────────────────────────────────────
        print("[Stage 5] Exporting clean neutral OBJ and run manifest...")
        output_paths = self._export(
            vertices=neutral_vertices,
            faces=faces,
            displacement=detail_displacement,
            detail_maps=detail_maps,
            expression_metadata=expression_metadata,
            beta=beta,
            detections=detections,
            photo_paths=photo_paths,
            output_dir=output_dir,
            facial_hair_manifest=facial_hair_manifest,
        )

        print(f"[Done] Complete. Output generated at: {output_dir}")
        return output_paths

    def _export(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        displacement: Optional[np.ndarray],
        detail_maps: Optional[Dict[str, str]],
        expression_metadata: dict,
        beta: np.ndarray,
        detections: list,
        photo_paths: list,
        output_dir: Path,
        facial_hair_manifest: Optional[dict] = None,
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
            cfg_dict = self.cfg if isinstance(self.cfg, dict) else {}
            retopo_matrix_path = cfg_dict.get('stage5', {}).get('correspondence_w')
            stylize_config = cfg_dict.get('stage5', {}).get('stylize')
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
                stylization_params=stylize_config,
                detail_maps=detail_maps,
                facial_hair_manifest=facial_hair_manifest,
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
            'detail_maps': detail_maps,
            'has_flame': self.flame is not None,
            'has_preview': preview_path.exists(),
            'facial_hair': facial_hair_manifest,
            'stage5_production_assets': stage5_manifest,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        res_dict = {
            'obj_path': str(obj_path),
            'manifest_path': str(manifest_path),
            'preview_path': str(preview_path) if preview_path.exists() else None,
            'blendshapes_path': stage5_manifest.get('blendshapes_json'),
            'armature_path': stage5_manifest.get('armature_json'),
            'fbx_path': stage5_manifest.get('fbx_file'),
        }
        if detail_maps:
            res_dict.update(detail_maps)
        return res_dict
