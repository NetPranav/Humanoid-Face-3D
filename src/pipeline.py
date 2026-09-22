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

        # Lazy-loaded network models & detail engines
        self._stage1 = None
        self._stage2 = None
        self._stage3 = None
        self._stage4 = None
        self._stage6 = None
        self._stage7 = None
        self._stage8 = None
        self._meso_extractor = None
        self._micro_synthesizer = None
        self._detail_fusion = None

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

    def _get_stage6(self):
        if self._stage6 is None:
            stage6_cfg = self.cfg.get('stage6', {}) if isinstance(self.cfg, dict) else {}
            if not stage6_cfg.get('enabled', True):
                return None
            try:
                from src.stage6_texture.projector import MultiViewTextureProjector
                self._stage6 = MultiViewTextureProjector(
                    texture_resolution=int(stage6_cfg.get('texture_resolution', 2048)),
                    blend_gamma=float(stage6_cfg.get('blend_gamma', 2.0)),
                    visibility_epsilon=float(stage6_cfg.get('visibility_epsilon', 0.005)),
                )
            except Exception as e:
                print(f"[Stage 6 Notice] Texture projector not initialized: {e}")
                self._stage6 = None
        return self._stage6

    def _get_stage7(self):
        if self._stage7 is None:
            stage7_cfg = self.cfg.get('stage7', {}) if isinstance(self.cfg, dict) else {}
            if not stage7_cfg.get('enabled', True):
                return None
            try:
                from src.stage7_delight.delight_net import DelightingPipeline
                self._stage7 = DelightingPipeline(
                    delight_checkpoint=stage7_cfg.get('delight_checkpoint'),
                    inpaint_method=stage7_cfg.get('inpaint_method', 'procedural'),
                    inpaint_iterations=int(stage7_cfg.get('inpaint_iterations', 50)),
                    use_fp16=stage7_cfg.get('use_fp16', True),
                )
            except Exception as e:
                print(f"[Stage 7 Notice] Delighting pipeline not initialized: {e}")
                self._stage7 = None
        return self._stage7

    def _get_stage8(self):
        if self._stage8 is None:
            stage8_cfg = self.cfg.get('stage8', {}) if isinstance(self.cfg, dict) else {}
            if not stage8_cfg.get('enabled', True):
                return None
            try:
                from src.stage8_pbr.material_stack import PBRMaterialStack
                self._stage8 = PBRMaterialStack(
                    roughness_config={
                        'r_t_zone': float(stage8_cfg.get('roughness_t_zone', 0.33)),
                        'r_cheeks': float(stage8_cfg.get('roughness_cheeks', 0.57)),
                        'r_lips': float(stage8_cfg.get('roughness_lips', 0.22)),
                    },
                    cavity_strength=float(stage8_cfg.get('cavity_strength', 0.35)),
                    sss_ray_count=int(stage8_cfg.get('sss_ray_count', 32)),
                    sss_normalize=stage8_cfg.get('sss_normalize', True),
                )
            except Exception as e:
                print(f"[Stage 8 Notice] PBR material stack not initialized: {e}")
                self._stage8 = None
        return self._stage8

    def _get_meso_extractor(self):
        if self._meso_extractor is None:
            stage3_cfg = self.cfg.get('stage3', {}) if isinstance(self.cfg, dict) else {}
            res = int(stage3_cfg.get('resolution', 1024))
            max_wrinkle = float(stage3_cfg.get('max_wrinkle_depth_mm', 1.20))
            from src.stage3_detail.photometric_detail import PhotometricDetailExtractor
            self._meso_extractor = PhotometricDetailExtractor(
                resolution=res,
                max_wrinkle_depth_mm=max_wrinkle,
            )
        return self._meso_extractor

    def _get_micro_synthesizer(self):
        if self._micro_synthesizer is None:
            stage3_cfg = self.cfg.get('stage3', {}) if isinstance(self.cfg, dict) else {}
            res = int(stage3_cfg.get('resolution', 1024))
            density = float(stage3_cfg.get('pore_density_scale', 1.0))
            from src.stage3_detail.anatomical_pores import AnatomicalPoreSynthesizer
            self._micro_synthesizer = AnatomicalPoreSynthesizer(
                resolution=res,
                pore_density_scale=density,
            )
        return self._micro_synthesizer

    def _get_detail_fusion(self):
        if self._detail_fusion is None:
            stage3_cfg = self.cfg.get('stage3', {}) if isinstance(self.cfg, dict) else {}
            res = int(stage3_cfg.get('resolution', 1024))
            max_scale = float(stage3_cfg.get('max_scale_mm', 5.0))
            cavity_str = float(stage3_cfg.get('cavity_strength', 0.40))
            from src.stage3_detail.fusion import MultiTierDetailFusion
            self._detail_fusion = MultiTierDetailFusion(
                resolution=res,
                max_scale_mm=max_scale,
                cavity_strength=cavity_str,
            )
        return self._detail_fusion

    def run(self, photo_paths: List[str], output_dir: str) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Pre-flight input validation & detection ──────────────────────────
        print("[Stage 0] Validating input portraits and extracting alignments...")
        pipe_cfg = self.cfg.get('pipeline', {}) if isinstance(self.cfg, dict) else {}
        min_photos = pipe_cfg.get('input_min_photos', 3)
        if 'input_min_photos' not in pipe_cfg and len(photo_paths) < min_photos and len(photo_paths) >= 1:
            min_photos = len(photo_paths)
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

        # ── PBR TEXTURE ENGINE: Projection & Delighting (Stages 6 & 7) ───────
        # Runs first so clean diffuse albedo is available for photo-derived wrinkles
        texture_maps = {}
        albedo_diffuse_img = None
        try:
            texture_maps = self._run_texture_engine(
                photos=[cv2.imread(str(p)) for p in photo_paths],
                detections=detections,
                vertices=neutral_vertices,
                faces=faces,
                detail_maps=None,
                output_dir=output_dir,
            )
            for k in ['albedo_diffuse_png', 'projected_raw_png', 'albedo']:
                if k in texture_maps and Path(texture_maps[k]).exists():
                    albedo_diffuse_img = cv2.imread(str(texture_maps[k]))
                    break
        except Exception as e:
            print(f"[Texture Engine Warning] UV texture projection / delighting failed: {e}")

        # ── TIER 1 + 2 + 3 MULTI-TIER GEOMETRY & PBR COUPLING ────────────────
        stage3_cfg = self.cfg.get('stage3', {}) if isinstance(self.cfg, dict) else {}
        detail_res = int(stage3_cfg.get('resolution', 1024))
        detail_maps = {}

        # 1. Tier 1 Macro displacement (optional GAN checkpoint if available)
        disp_macro = None
        stage3 = self._get_stage3()
        if stage3 is not None:
            try:
                id_feats = [d.embedding for d in valid_dets if hasattr(d, 'embedding') and d.embedding is not None]
                per_view_feats = np.stack(id_feats, axis=0) if id_feats else None
                synth_res = stage3.synthesize(
                    neutral_vertices=neutral_vertices,
                    faces=faces,
                    per_view_feats=per_view_feats,
                    beta=beta,
                    psi=expression_psi,
                    resolution=detail_res
                )
                disp_macro = synth_res['disp_mm']
            except Exception as e:
                print(f"[Tier 1 Notice] Macro GAN detail synthesis skipped: {e}")

        # 2. Tier 2 Meso Wrinkles (Photo-Derived Shape-from-Shading)
        disp_meso = None
        if stage3_cfg.get('enable_meso', True) and albedo_diffuse_img is not None:
            print("[Tier 2 Meso] Extracting real photo-derived wrinkles (shape-from-shading)...")
            try:
                meso_ext = self._get_meso_extractor()
                meso_res = meso_ext.extract_from_albedo(
                    albedo_rgb=albedo_diffuse_img,
                    vertices=neutral_vertices,
                    faces=faces,
                )
                disp_meso = meso_res['displacement_mm']
            except Exception as e:
                print(f"[Tier 2 Warning] Photo-derived wrinkle extraction skipped: {e}")

        # 3. Tier 3 Micro Pores (4K Anatomical Zone Synthesis)
        disp_micro = None
        zone_masks = None
        if stage3_cfg.get('enable_micro', True):
            print("[Tier 3 Micro] Synthesizing anatomical cellular pores (50-micron follicles)...")
            try:
                micro_synth = self._get_micro_synthesizer()
                micro_res = micro_synth.synthesize(
                    resolution=detail_res,
                    vertices=neutral_vertices,
                    faces=faces,
                )
                disp_micro = micro_res['displacement_mm']
                zone_masks = micro_res['zone_masks']
            except Exception as e:
                print(f"[Tier 3 Warning] Anatomical pore synthesis skipped: {e}")

        # 4. Multi-Tier Detail Fusion & PBR Coupling
        fusion = self._get_detail_fusion()
        if zone_masks is None:
            zone_masks = {'valid': np.ones((detail_res, detail_res), dtype=np.float32)}

        coupled_pbr = fusion.couple_pbr_material_stack(
            meso_disp_mm=disp_meso if disp_meso is not None else np.zeros((detail_res, detail_res), dtype=np.float32),
            micro_disp_mm=disp_micro if disp_micro is not None else np.zeros((detail_res, detail_res), dtype=np.float32),
            zone_masks=zone_masks,
            macro_disp_mm=disp_macro,
            weight_meso=float(stage3_cfg.get('weight_meso', 1.0)),
            weight_micro=float(stage3_cfg.get('weight_micro', 1.0)),
            weight_macro=float(stage3_cfg.get('weight_macro', 1.0)),
        )
        saved_detail_maps = fusion.save_maps(coupled_pbr, output_dir=output_dir, prefix="film")
        detail_displacement = coupled_pbr['composite_displacement_mm']

        # Enforce System Invariant Rule 4: Neck Seam Contract (Bitwise Collar Pinning)
        bottom_20_rows = int(0.85 * detail_res)
        collar_slice = detail_displacement[bottom_20_rows:, :]
        collar_max = float(np.max(np.abs(collar_slice)))
        if collar_max > 1e-4:
            raise RuntimeError(
                f"Rule 4 Invariant Violated: Neck boundary collar has non-zero displacement (max={collar_max:.6f} mm). "
                "The lowest 20% of vertices must remain strictly pinned to 0.0 mm."
            )

        for k, v in saved_detail_maps.items():
            detail_maps[k] = str(v)
        detail_maps['displacement_16bit'] = str(saved_detail_maps['displacement_png'])
        detail_maps['displacement_png'] = str(saved_detail_maps['displacement_png'])
        detail_maps['normal_map'] = str(saved_detail_maps['normal_png'])
        detail_maps['normal_png'] = str(saved_detail_maps['normal_png'])
        detail_maps['cavity'] = str(saved_detail_maps['cavity_ao_png'])
        detail_maps['cavity_ao'] = str(saved_detail_maps['cavity_ao_png'])
        detail_maps['roughness'] = str(saved_detail_maps['roughness_base_png'])
        detail_maps['roughness_base'] = str(saved_detail_maps['roughness_base_png'])
        detail_maps['roughness_coat'] = str(saved_detail_maps['roughness_coat_png'])

        # Merge texture maps (projected and delighted albedo)
        if texture_maps:
            detail_maps.update(texture_maps)
            if 'albedo_diffuse_png' in texture_maps:
                detail_maps['albedo'] = str(texture_maps['albedo_diffuse_png'])
            elif 'projected_raw_png' in texture_maps:
                detail_maps['albedo'] = str(texture_maps['projected_raw_png'])

        # ── Stage 8: SSS Thickness Estimation ────────────────────────────────
        if 'sss_thickness' in texture_maps:
            detail_maps['sss_thickness'] = str(texture_maps['sss_thickness'])
            detail_maps['sss_thickness_png'] = str(texture_maps['sss_thickness'])
        elif 'sss_thickness_map' in texture_maps:
            detail_maps['sss_thickness'] = str(texture_maps['sss_thickness_map'])
            detail_maps['sss_thickness_png'] = str(texture_maps['sss_thickness_map'])
        else:
            stage8 = self._get_stage8()
            if stage8 is not None:
                try:
                    from src.stage3_detail.rasterizer import load_flame_uv_layout, compute_vertex_normals
                    uv_coords, uv_faces = load_flame_uv_layout()
                    vertex_normals = compute_vertex_normals(neutral_vertices, faces)
                    sss_map = stage8.sss_gen.generate(
                        neutral_vertices, faces, vertex_normals, uv_coords, uv_faces, faces,
                        resolution=detail_res
                    )
                    sss_path = output_dir / "sss_thickness_map.png"
                    cv2.imwrite(str(sss_path), np.round(sss_map * 255.0).astype(np.uint8))
                    detail_maps['sss_thickness'] = str(sss_path)
                    detail_maps['sss_thickness_png'] = str(sss_path)
                except Exception as e:
                    print(f"[Stage 8 SSS Notice] SSS thickness estimation skipped: {e}")

        # ── Stage 4: Facial Hair & Stubble Geometry ──────────────────────────
        facial_hair_manifest = None
        stage4 = self._get_stage4()
        if stage4 is not None:
            stage4_cfg = self.cfg.get('stage4', {}) if isinstance(self.cfg, dict) else {}
            hair_preset = stage4_cfg.get('preset', 'stubble')
            hair_res = int(stage4_cfg.get('resolution', detail_res))
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

        # ── Step 4: Automated Film-Grade Blender Cycles Studio Look-Dev ────
        render_cfg = self.cfg.get('render', {}) if isinstance(self.cfg, dict) else {}
        if render_cfg.get('enabled', True):
            print("[Film Look-Dev] Launching automated Blender Cycles studio engine...")
            try:
                from scripts.render_blender_film import parse_args, execute_film_render
                mesh_to_render = output_paths.get('obj_path')
                face_lod0_p = output_dir / "face_lod0.obj"
                if face_lod0_p.exists():
                    mesh_to_render = str(face_lod0_p)

                film_render_png = output_dir / "film_render_cycles.png"
                studio_blend = output_dir / "studio_scene.blend"

                samples_val = str(render_cfg.get('samples', 128))
                res_w = str(render_cfg.get('width', 2048))
                res_h = str(render_cfg.get('height', 2048))
                dev_val = str(render_cfg.get('device', 'AUTO'))

                render_args = parse_args([
                    "--mesh", str(mesh_to_render),
                    "--textures_dir", str(output_dir),
                    "--output", str(film_render_png),
                    "--save_blend", str(studio_blend),
                    "--samples", samples_val,
                    "--resolution", res_w, res_h,
                    "--device", dev_val,
                ])
                render_res = execute_film_render(render_args)
                output_paths['film_render'] = str(render_res['rendered_image'])
                output_paths['blend_file'] = str(render_res['blend_file'])
                print(f"[Film Look-Dev] Look-dev turnaround and studio scene ready.")
            except Exception as e:
                print(f"[Film Look-Dev Warning] Blender Cycles rendering skipped: {e}")

        print(f"[Done] Complete. Output generated at: {output_dir}")
        return output_paths

    def _run_texture_engine(
        self,
        photos: List[np.ndarray],
        detections: list,
        vertices: np.ndarray,
        faces: np.ndarray,
        detail_maps: Optional[Dict[str, str]],
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Runs the PBR Texture Engine: Stage 6 → Stage 7 → Stage 8.
        Returns a dictionary of output texture map file paths.
        """
        output_dir = Path(output_dir)
        texture_maps = {}

        # Load FLAME UV layout (shared across stages)
        from src.stage3_detail.rasterizer import load_flame_uv_layout, compute_vertex_normals
        uv_coords, uv_faces = load_flame_uv_layout()
        vertex_normals = compute_vertex_normals(vertices, faces)

        # ── Stage 6: Multi-View UV Texture Projection ────────────────────
        stage6 = self._get_stage6()
        projection_result = None
        if stage6 is not None:
            print("[Stage 6] Projecting input photographs onto UV texture space...")
            try:
                valid_photos = [p for p in photos if p is not None]
                valid_dets = [d for d in detections if d is not None]
                if valid_photos and valid_dets:
                    projection_result = stage6.project(
                        photos=valid_photos,
                        detections=valid_dets,
                        vertices=vertices,
                        faces=faces,
                        vertex_normals=vertex_normals,
                        uv_coords=uv_coords,
                        uv_faces=uv_faces,
                        flame_faces=faces,
                    )
                    proj_paths = stage6.save_maps(projection_result, output_dir)
                    texture_maps.update(proj_paths)
                    print(f"[Stage 6] Projected texture saved ({stage6.resolution}×{stage6.resolution}).")
                else:
                    print("[Stage 6 Warning] No valid photos/detections for texture projection.")
            except Exception as e:
                print(f"[Stage 6 Warning] UV texture projection failed: {e}")

        # ── Stage 7: AI Delighting + UV Inpainting ───────────────────────
        stage7 = self._get_stage7()
        delight_result = None
        if stage7 is not None and projection_result is not None:
            print("[Stage 7] Running delighting and UV inpainting...")
            try:
                # Load normal map for delighting conditioning
                normal_map = None
                if detail_maps and detail_maps.get('normal_png'):
                    nm_path = detail_maps['normal_png']
                    if Path(nm_path).exists():
                        normal_map = cv2.imread(nm_path, cv2.IMREAD_COLOR)
                        if normal_map is not None:
                            normal_map = normal_map.astype(np.float32) / 255.0

                delight_result = stage7.process(
                    projected_rgb=projection_result['projected_rgb'],
                    projection_mask=projection_result['projection_mask'],
                    normal_map=normal_map,
                )
                delight_paths = stage7.save_maps(delight_result, output_dir)
                texture_maps.update(delight_paths)
                print("[Stage 7] Clean albedo maps saved.")
            except Exception as e:
                print(f"[Stage 7 Warning] Delighting failed: {e}")

        # ── Stage 8: PBR Material Stack ──────────────────────────────────
        stage8 = self._get_stage8()
        if stage8 is not None:
            print("[Stage 8] Generating PBR material stack (roughness, cavity, SSS)...")
            try:
                # Load displacement map for roughness/cavity derivation
                disp_map = None
                if detail_maps:
                    disp_path = detail_maps.get('displacement_png') or detail_maps.get('stubble_displacement')
                    if disp_path and Path(disp_path).exists():
                        disp_raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
                        if disp_raw is not None:
                            if disp_raw.dtype == np.uint16:
                                disp_map = (disp_raw.astype(np.float32) / 65535.0 * 2.0 - 1.0)
                            else:
                                disp_map = disp_raw.astype(np.float32)
                            if len(disp_map.shape) > 2:
                                disp_map = disp_map[:, :, 0]

                stage8_cfg = self.cfg.get('stage8', {}) if isinstance(self.cfg, dict) else {}
                tex_res = int(stage8_cfg.get('resolution', self.cfg.get('stage6', {}).get('texture_resolution', 2048) if isinstance(self.cfg, dict) else 2048))

                pbr_result = stage8.generate(
                    vertices=vertices,
                    faces=faces,
                    vertex_normals=vertex_normals,
                    uv_coords=uv_coords,
                    uv_faces=uv_faces,
                    flame_faces=faces,
                    displacement_map=disp_map,
                    resolution=tex_res,
                )
                pbr_paths = stage8.save_maps(pbr_result, output_dir)
                texture_maps.update(pbr_paths)
                print("[Stage 8] PBR material stack saved (roughness, cavity/AO, SSS).")
            except Exception as e:
                print(f"[Stage 8 Warning] PBR material generation failed: {e}")

        return texture_maps

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
