"""
Face geometry + texture pipeline (Phase 0 of DOCS/04_roadmap.md).

Stage flow:
  0  detection (InsightFace) · EXIF intrinsics · skin parsing (MediaPipe)
  1  identity β: MICA with MICA's own ArcFace                           [fix R1]
  2  per-view camera + pose + expression + jaw fit to 68 landmarks        [fix R2/R3]
  6  per-texel backprojection onto the fitted posed mesh, z-buffer + parsing  [fix R4]
  7  provenance-aware UV fill (observed / mirrored / interpolated)        [fix R6]
  3  detail: micro pores only by default; Multiface GAN and luminance→height off  [fix R5]
  8  PBR maps (roughness / cavity / SSS)
  4  facial hair (v1 presets, unchanged in Phase 0)
  5  production export (v1, unchanged in Phase 0)
  eval  overlays, turntables, report.json (evaluation/phase0_eval.py)

Failure policy: a required model or asset that is missing raises. Optional stages that
fail are recorded in manifest['stage_status'] with the error (never silently skipped);
texture/detail stages never block geometry export (AGENTS.md Rule 6).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import yaml

from src.stage0_preprocess.detector import FaceDetector
from src.utils.flame_model import FLAMEModel
from src.utils.validation import validate_inputs

ROOT = Path(__file__).resolve().parent.parent


class FaceGeoPipeline:
    def __init__(
        self,
        config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
        model_dir: Union[str, Path] = "models_cache",
        detector: Optional[FaceDetector] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        if cfg is not None:
            self.cfg = cfg
        elif isinstance(config_path, dict):
            self.cfg = config_path
        elif config_path is not None:
            with open(config_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}

        self.model_dir = Path(model_dir) if model_dir is not None else ROOT / "models_cache"
        if not self.model_dir.is_absolute() and not self.model_dir.exists():
            self.model_dir = ROOT / self.model_dir

        flame_candidates = [self.model_dir / "flame" / "generic_model.pkl", self.model_dir / "generic_model.pkl"]
        if str(model_dir) in ("models_cache", str(ROOT / "models_cache")):
            # the repo default location (written by scripts/extract_flame_from_mica.py); never used
            # when the caller points model_dir somewhere else explicitly
            flame_candidates.append(ROOT / "data" / "flame_model" / "generic_model.pkl")
        flame_path = next((p for p in flame_candidates if p.exists()), None)
        self.flame = FLAMEModel(str(flame_path)) if flame_path is not None else None
        self.flame_path = flame_path

        if detector is not None:
            self.detector = detector
        else:
            self.detector = FaceDetector(allow_degraded=bool(self.cfg.get("allow_degraded", False)))

        self._stage1 = None
        self._fitter = None
        self._parser = None
        self._projector = None
        self._filler = None
        self._stage4 = None
        self._stage8 = None
        self._detail_fusion = None
        self._uv = None
        self._uv_regions = None

    # ------------------------------------------------------------------ helpers
    def _c(self, section: str) -> Dict[str, Any]:
        return self.cfg.get(section, {}) if isinstance(self.cfg, dict) else {}

    def _require_flame(self) -> FLAMEModel:
        if self.flame is None:
            raise RuntimeError(
                "FLAME model is not initialized. Create data/flame_model/generic_model.pkl with:\n"
                "  python3 scripts/extract_flame_from_mica.py --mica models_cache/mica/mica.tar"
            )
        return self.flame

    def uv_layout(self):
        if self._uv is None:
            from src.stage3_detail.rasterizer import load_flame_uv_layout
            uv, uvf = load_flame_uv_layout()
            if len(uvf) != len(self._require_flame().faces):
                raise RuntimeError("FLAME UV template face count does not match the FLAME model.")
            self._uv = (uv.astype(np.float64), uvf.astype(np.int64))
        return self._uv

    def uv_regions(self) -> Dict[str, np.ndarray]:
        """UV masks from official FLAME regions + landmark-drawn brows/eyes/lips (flame_regions.py)."""
        if self._uv_regions is None:
            from src.stage2_expression.landmark_fit import load_landmark_embedding
            from src.utils import flame_regions as fr
            flame = self._require_flame()
            faces = flame.faces.astype(np.int64)
            uv, uvf = self.uv_layout()
            ras = self._get_projector().uv_raster(uv, uvf)
            regions = fr.uv_region_masks(ras, fr.face_region_masks(faces, len(flame.v_template), fr.load_vertex_masks()))
            R = ras.mask.shape[0]
            regions.update(fr.feature_masks(fr.landmark_uv(load_landmark_embedding(), faces, uv, uvf), R))
            features = regions["brows"] | regions["eye_openings"] | regions["lip_border"] | regions["lips_landmark"]
            # plain skin for fitting lighting: face without eyes, lips, brows
            regions["plain_skin"] = regions["face"] & ~regions["eye_region"] & ~regions["lips"] & ~features
            # clean skin grain exemplar: forehead + cheeks (no nose: large pores / nostrils)
            # (upper face only: chin/jaw carry stubble and beard shadow on many subjects)
            lmk = fr.landmark_uv(load_landmark_embedding(), faces, uv, uvf)
            nose_tip_row = int((1 - lmk[33, 1]) * R)
            upper = np.zeros_like(regions["face"])
            upper[:nose_tip_row] = True
            regions["skin_exemplar"] = regions["plain_skin"] & ~regions["nose"] & upper
            regions["texel_mm"] = fr.texel_scale_mm(ras, flame.v_template, faces, uv, uvf)
            # delighting strength: full on plain skin, half where coarse normals are unreliable
            unreliable = (regions["eye_region"] | regions["lips"] | features).astype(np.float32)
            regions["delight_strength"] = 1.0 - 0.5 * cv2.GaussianBlur(unreliable, (0, 0), 6)
            self._uv_regions = regions
        return self._uv_regions

    # ------------------------------------------------------------------ stages
    def _get_stage1(self):
        if self._stage1 is None:
            from src.stage1_identity.inference import MICAIdentityEncoder
            c = self._c("stage1")
            candidates = [
                self.model_dir / "mica" / "mica.tar",
                self.model_dir / "mica" / "pretrained.tar",
                Path("/kaggle/input/mica-pretrained/mica.tar"),
                Path("/kaggle/input/mica-pretrained/pretrained.tar"),
            ]
            ckpt = next((p for p in candidates if p.exists() and p.stat().st_size > 0), candidates[0])
            self._stage1 = MICAIdentityEncoder(
                str(ckpt),
                fusion_temperature=float(c.get("fusion_temperature", 0.05)),
                yaw_weight_exponent=int(c.get("yaw_weight_exponent", 2)),
            )
        return self._stage1

    def _get_fitter(self):
        if self._fitter is None:
            from src.stage2_expression.landmark_fit import LandmarkFitter
            c = self._c("stage2")
            self._fitter = LandmarkFitter(
                self._require_flame(),
                n_expr=int(c.get("n_expr", 50)),
                iters=int(c.get("iters", 400)),
                fit_focal=bool(c.get("fit_focal", True)),
            )
        return self._fitter

    def _get_parser(self):
        if self._parser is None:
            from src.stage0_preprocess.parsing import FaceParser
            self._parser = FaceParser()
        return self._parser

    def _get_projector(self):
        if self._projector is None:
            from src.stage6_texture.projector import MultiViewTextureProjector
            c = self._c("stage6")
            self._projector = MultiViewTextureProjector(
                texture_resolution=int(c.get("texture_resolution", 2048)),
                blend_gamma=float(c.get("blend_gamma", 2.0)),
                min_cos=float(c.get("min_cos", 0.15)),
                depth_tolerance_m=float(c.get("depth_tolerance_m", 0.0015)),
            )
        return self._projector

    def _get_filler(self):
        if self._filler is None:
            from src.stage7_delight.uv_fill import ProvenanceUVFill
            self._filler = ProvenanceUVFill()
        return self._filler

    def _get_stage4(self):
        if self._stage4 is None:
            from src.stage4_facial_hair.generator import FacialHairGenerator
            c = self._c("stage4")
            self._stage4 = FacialHairGenerator(neck_collar_threshold=float(c.get("neck_collar_threshold", 0.20)))
        return self._stage4

    def _get_stage8(self):
        if self._stage8 is None:
            from src.stage8_pbr.material_stack import PBRMaterialStack
            c = self._c("stage8")
            self._stage8 = PBRMaterialStack(
                roughness_config={
                    "r_t_zone": float(c.get("roughness_t_zone", 0.33)),
                    "r_cheeks": float(c.get("roughness_cheeks", 0.57)),
                    "r_lips": float(c.get("roughness_lips", 0.22)),
                },
                cavity_strength=float(c.get("cavity_strength", 0.35)),
                sss_ray_count=int(c.get("sss_ray_count", 32)),
                sss_normalize=c.get("sss_normalize", True),
            )
        return self._stage8

    def _get_detail_fusion(self):
        if self._detail_fusion is None:
            from src.stage3_detail.fusion import MultiTierDetailFusion
            c = self._c("stage3")
            self._detail_fusion = MultiTierDetailFusion(
                resolution=int(c.get("resolution", 1024)),
                max_scale_mm=float(c.get("max_scale_mm", 1.0)),
                cavity_strength=float(c.get("cavity_strength", 0.40)),
            )
        return self._detail_fusion

    # ------------------------------------------------------------------ run
    def run(self, photo_paths: List[str], output_dir: str) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        status: Dict[str, Any] = {}

        def optional(name: str, fn):
            """Runs an optional stage; failures are recorded (and printed), never swallowed silently."""
            try:
                out = fn()
                status[name] = "ok"
                return out
            except Exception as e:  # noqa: BLE001 - recorded in the manifest
                status[name] = f"failed: {type(e).__name__}: {e}"
                print(f"[{name} FAILED] {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()
                return None

        flame = self._require_flame()
        faces = flame.faces.astype(np.int64)
        if self._c("stage1_5").get("enabled", False):
            raise ValueError(
                "stage1_5 (contour deformer) is disabled in Phase 0 (DOCS/01 R3); remove it from the config."
            )

        # ── Stage 0 ─────────────────────────────────────────────────────────
        print("[Stage 0] Detection, EXIF intrinsics, parsing...")
        c0 = self._c("stage0")
        pipe_cfg = self._c("pipeline")
        val = validate_inputs(
            photo_paths, detector=self.detector,
            min_photos=int(pipe_cfg.get("input_min_photos", 1)),
            min_face_confidence=float(c0.get("min_face_confidence", 0.5)),
        )
        if not val.is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(val.errors)}")
        pairs = [(p, d) for p, d in zip(photo_paths, val.detections) if d is not None]
        photos = [cv2.imread(str(p)) for p, _ in pairs]
        dets = [d for _, d in pairs]

        from src.stage0_preprocess.camera import intrinsics_from_exif, intrinsics_from_fov
        intrinsics = []
        for (p, _), im in zip(pairs, photos):
            K = intrinsics_from_exif(p, im.shape[:2])
            if K is None:
                K = intrinsics_from_fov(im.shape[:2], float(c0.get("default_focal_35mm", 50.0)))
                print(f"[Stage 0] No EXIF focal in {Path(p).name}; using {K.fx:.0f}px (35mm-equiv "
                      f"{c0.get('default_focal_35mm', 50.0)}) as prior, refined in Stage 2.")
            intrinsics.append(K)

        parser = self._get_parser()
        parsings = [parser.parse(im) for im in photos]
        erode = [max(1, int(float(c0.get("skin_erode_iod", 0.04)) * _iod(d))) for d in dets]
        skin_masks = [parser.skin_mask(c, e) for c, e in zip(parsings, erode)]
        # never sample the inside of the mouth (teeth/tongue would be painted onto closed lips)
        for m, d in zip(skin_masks, dets):
            if d.landmark_3d_68 is not None and c0.get("exclude_mouth_interior", True):
                poly = np.zeros(m.shape, np.uint8)
                cv2.fillPoly(poly, [np.round(d.landmark_3d_68[60:68, :2]).astype(np.int32)], 1)
                k = max(3, int(0.06 * _iod(d)))
                m &= ~cv2.dilate(poly, np.ones((k, k), np.uint8)).astype(bool)
                m &= ~_mouth_interior_pixels(photos[dets.index(d)], d, m)

        # ── Stage 1 ─────────────────────────────────────────────────────────
        print("[Stage 1] MICA identity (MICA ArcFace features)...")
        beta = self._get_stage1().encode_multiview(dets)

        # ── Stage 2 ─────────────────────────────────────────────────────────
        print("[Stage 2] Fitting camera, head pose, expression and jaw per view...")
        fitter = self._get_fitter()
        view_fits = [fitter.fit(beta, d.landmark_3d_68, K) for d, K in zip(dets, intrinsics)]
        for i, vf in enumerate(view_fits):
            m = vf.metrics
            print(f"  view {i}: landmark error {100 * m['lmk_err_iod_mean']:.1f}% IOD, "
                  f"f={vf.intrinsics.fx:.0f}px ({vf.intrinsics.source}), distance {m['subject_distance_m']:.2f} m")

        neutral_vertices, _ = flame.decode_neutral(beta)
        uv, uvf = self.uv_layout()

        # ── Stage 6 + 7 ─────────────────────────────────────────────────────
        texture_maps: Dict[str, str] = {}
        albedo = None
        fill_fractions = None
        fr = None
        if self._c("stage6").get("enabled", True):
            def _texture():
                print("[Stage 6] Backprojecting photos onto the fitted mesh (z-buffer + skin parsing)...")
                proj = self._get_projector()
                c7 = self._c("stage7")
                regions = self.uv_regions()
                delight = c7.get("delight", True)
                res = proj.project(photos, [vf.vertices_cam for vf in view_fits],
                                   [vf.intrinsics.K for vf in view_fits], skin_masks, faces, uv, uvf,
                                   delight_fit_region=regions["plain_skin"] if delight else None,
                                   delight_strength=(regions["delight_strength"] * float(c7.get("delight_strength", 1.0))
                                                     if delight else None),
                                   white_balance=bool(c7.get("white_balance", False)))
                texture_maps.update(proj.save_maps(res, output_dir))
                source = res.get("albedo_rgb", res["projected_rgb"])
                print("[Stage 7] Provenance-aware UV fill (observed / mirrored / interpolated)...")
                from src.stage7_delight.uv_fill import load_or_build_mirror_map
                mirror = load_or_build_mirror_map(ROOT / "models_cache" / "flame_uv",
                                                  proj.uv_raster(uv, uvf), flame.v_template, faces)
                fr = self._get_filler().fill(
                    source, res["projection_mask"] > 0, res["uv_valid"], mirror,
                    exemplar_region=regions["skin_exemplar"] if c7.get("synthesize_skin", True) else None,
                    mesh={"uv_raster": proj.uv_raster(uv, uvf), "faces": faces, "uv": uv, "uv_faces": uvf,
                          "n_verts": len(flame.v_template)},
                    closed_regions={"lips": regions["lips"] | regions["lips_landmark"]})
                fr["sh_coeffs"] = res.get("sh_coeffs")
                fr["delit"] = "albedo_rgb" in res
                texture_maps.update(self._get_filler().save_maps(fr, output_dir))
                return fr
            fr = optional("stage6_7_texture", _texture)
            if fr is not None:
                albedo = fr["albedo_srgb"]
                fill_fractions = fr["fractions"]

        # ── Stage 3: detail ─────────────────────────────────────────────────
        c3 = self._c("stage3")
        res3 = int(c3.get("resolution", 1024))
        detail_maps: Dict[str, str] = dict(texture_maps)
        zeros = np.zeros((res3, res3), np.float32)
        disp_micro = None
        zone_masks = None
        if float(c3.get("weight_macro", 0.0)) > 0 or c3.get("enable_meso", False) or c3.get("mode", "sculpt") != "sculpt":
            raise ValueError(
                "stage3: the v1 detail GAN (weight_macro), luminance meso (enable_meso) and legacy pores were "
                "removed on 2026-09-25 (older/REMOVED_WORK_REPORT_2026-09-25.md). Use stage3.mode: sculpt."
            )
        sculpt = None
        weight_micro = 1.0
        # Phase 4A: photo-derived wrinkles + synthesized pores/grooves, one 0-100 knob
        if fr is None:
            status["stage3_sculpt"] = "skipped: needs the Stage 6/7 texture"
        else:
            from src.stage3_detail.sculpt_detail import build_detail
            level = float(c3.get("detail_level", 75))
            sculpt = optional("stage3_sculpt", lambda: build_detail(
                fr["albedo_srgb"], fr["provenance"], self.uv_regions(), level,
                out_res=int(c3.get("detail_resolution", 4096))))
            if sculpt is not None:
                disp_micro = cv2.resize(sculpt["displacement_mm"], (res3, res3), interpolation=cv2.INTER_AREA)

        fusion = self._get_detail_fusion()
        coupled = fusion.couple_pbr_material_stack(
            meso_disp_mm=zeros,
            micro_disp_mm=disp_micro if disp_micro is not None else zeros,
            zone_masks=zone_masks or {"valid": np.ones((res3, res3), np.float32)},
            macro_disp_mm=None,
            weight_meso=float(c3.get("weight_meso", 0.5)),
            weight_micro=weight_micro,
            weight_macro=0.0,
        )
        saved = fusion.save_maps(coupled, output_dir=output_dir, prefix="film")
        sculpt_info = None
        if sculpt is not None:
            sculpt_info = self._save_sculpt(sculpt, fusion.max_scale_mm, saved, neutral_vertices, faces, output_dir, status)
        detail_displacement = coupled["composite_displacement_mm"]
        collar = detail_displacement[int(0.85 * res3):, :]
        if float(np.max(np.abs(collar))) > 1e-4:
            raise RuntimeError("Rule 4 invariant violated: neck collar displacement is non-zero.")
        for k, v in saved.items():
            detail_maps[k] = str(v)
        detail_maps.update({
            "displacement_16bit": str(saved["displacement_png"]), "normal_map": str(saved["normal_png"]),
            "cavity": str(saved["cavity_ao_png"]), "roughness": str(saved["roughness_base_png"]),
            "roughness_coat": str(saved["roughness_coat_png"]),
        })
        detail_maps.update(texture_maps)

        # ── Stage 8 ─────────────────────────────────────────────────────────
        if self._c("stage8").get("enabled", True):
            def _pbr():
                from src.stage3_detail.rasterizer import compute_vertex_normals
                s8 = self._get_stage8()
                out = s8.generate(
                    vertices=neutral_vertices, faces=faces,
                    vertex_normals=compute_vertex_normals(neutral_vertices, faces),
                    uv_coords=uv, uv_faces=uvf, flame_faces=faces, displacement_map=detail_displacement,
                    resolution=int(self._c("stage6").get("texture_resolution", 2048)),
                    projected_rgb=albedo.astype(np.float32) if albedo is not None else None,
                )
                return s8.save_maps(out, output_dir)
            pbr = optional("stage8_pbr", _pbr)
            if pbr:
                detail_maps.update(pbr)

        # ── Stage 4 ─────────────────────────────────────────────────────────
        facial_hair_manifest = None
        if self._c("stage4").get("enabled", True):
            c4 = self._c("stage4")
            hair = optional("stage4_facial_hair", lambda: self._get_stage4().generate(
                neutral_vertices=neutral_vertices, faces=faces, detail_displacement_mm=detail_displacement,
                config=c4.get("config", c4.get("preset", "stubble")), output_dir=output_dir,
                resolution=int(c4.get("resolution", res3))))
            if hair:
                facial_hair_manifest = hair.get("manifest")
                if facial_hair_manifest and facial_hair_manifest.get("stubble", {}).get("generated", False):
                    detail_displacement = hair["stubble_displacement_mm"]

        # ── Stage 5 ─────────────────────────────────────────────────────────
        print("[Stage 5] Exporting neutral mesh and production assets...")
        expression_metadata = {
            "per_view": [{"psi": vf.psi.tolist(), "pose_theta": vf.pose_theta.tolist()} for vf in view_fits],
            "note": "Export is neutral (psi=0, theta=0); per-view expression is metadata only.",
        }
        output_paths = self._export(
            vertices=neutral_vertices, faces=faces, detail_maps=detail_maps,
            expression_metadata=expression_metadata, beta=beta, photo_paths=[p for p, _ in pairs],
            output_dir=output_dir, facial_hair_manifest=facial_hair_manifest, status=status,
            extra={
                "view_fits": [vf.to_dict() for vf in view_fits],
                "texture_fractions": fill_fractions,
                "displacement_encoding": {"max_scale_mm": fusion.max_scale_mm,
                                          "png_full_range_m": 2 * fusion.max_scale_mm / 1000.0},
                "flame_source": str(self.flame_path),
                "sculpt_detail": sculpt_info,
            },
        )

        # ── Evaluation ──────────────────────────────────────────────────────
        if self._c("evaluation").get("enabled", True) and albedo is not None:
            def _eval():
                from evaluation.phase0_eval import IdentityJudge, evaluate_run
                judge = IdentityJudge(self.detector.app) if getattr(self.detector, "app", None) is not None else None
                sh = fr.get("sh_coeffs") if fr is not None else None
                return evaluate_run(output_dir, photos, dets, view_fits, neutral_vertices, faces, uv, uvf, albedo,
                                    judge, extra={"texture_fractions": fill_fractions},
                                    sh_lum=sh[0] if sh is not None and len(sh) else None)
            rep = optional("evaluation", _eval)
            if rep is not None:
                output_paths["report"] = str(output_dir / "report.json")
            detail_mesh = output_dir / "head_mesh_detail.obj"
            if self._c("evaluation").get("clay_renders", True) and detail_mesh.exists():
                def _clay():
                    from scripts.render_clay import render_clay
                    from src.stage2_expression.landmark_fit import load_landmark_embedding
                    emb = load_landmark_embedding()
                    idx = emb["full_lmk_faces_idx"].astype(np.int64).reshape(-1)
                    lm = (neutral_vertices[faces[idx]] * emb["full_lmk_bary_coords"].reshape(-1, 3)[..., None]).sum(1)
                    ev = output_dir / "eval"
                    out = {"clay_front": render_clay(str(detail_mesh), str(ev / "clay_front.png"), 1024, 20)}
                    out["clay_eye"] = render_clay(str(detail_mesh), str(ev / "clay_eye.png"), 1024, 25,
                                                  tuple(lm[36:42].mean(0)), 0.05)
                    out["clay_mouth"] = render_clay(str(detail_mesh), str(ev / "clay_mouth.png"), 1024, 15,
                                                    tuple(lm[48:68].mean(0) + [0, 0.005, 0]), 0.075)
                    return out
                clay = optional("evaluation_clay", _clay)
                if clay:
                    output_paths.update(clay)

        # ── Look-dev render (Blender Cycles) ────────────────────────────────
        rc = self._c("render")
        if rc.get("enabled", False):
            def _render():
                from scripts.render_blender_film import execute_film_render, parse_args
                mesh = output_dir / "face_lod0.obj"
                args = parse_args([
                    "--mesh", str(mesh if mesh.exists() else output_paths["obj_path"]),
                    "--textures_dir", str(output_dir), "--output", str(output_dir / "film_render_cycles.png"),
                    "--save_blend", str(output_dir / "studio_scene.blend"),
                    "--samples", str(rc.get("samples", 128)),
                    "--resolution", str(rc.get("width", 2048)), str(rc.get("height", 2048)),
                    "--device", str(rc.get("device", "AUTO")),
                ])
                return execute_film_render(args)
            r = optional("render", _render)
            if r:
                output_paths["film_render"] = str(r["rendered_image"])

        # status is final only now: rewrite it into the manifest
        with open(output_paths["manifest_path"]) as fh:
            man = json.load(fh)
        man["stage_status"] = status
        with open(output_paths["manifest_path"], "w") as fh:
            json.dump(man, fh, indent=2)
        failed = {k: v for k, v in status.items() if v != "ok"}
        print(f"[Done] Output at {output_dir}" + (f"  (FAILED stages: {list(failed)})" if failed else ""))
        return output_paths

    def _save_sculpt(self, sculpt, max_scale_mm, saved, neutral_vertices, faces, output_dir, status):
        """Writes 4K displacement/normal maps (replacing the 1K fusion ones) and the sculpt mesh."""
        from src.stage3_detail.sculpt_detail import apply_to_mesh
        from src.utils.flame_regions import load_vertex_masks
        from src.utils.subdivision import export_obj_with_uvs

        disp = sculpt["displacement_mm"]
        if float(np.abs(disp).max()) > max_scale_mm:
            raise RuntimeError(f"Sculpt displacement exceeds the ±{max_scale_mm} mm encoding range.")
        u16 = np.round(np.clip((disp / max_scale_mm + 1.0) * 0.5, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(saved["displacement_png"]), u16)
        cv2.imwrite(str(saved["normal_png"]), cv2.cvtColor(sculpt["normal_rgb"], cv2.COLOR_RGB2BGR))
        st = sculpt["settings"]
        info = {"detail_level": st.level, "subdivision": st.subdivision, "meso_gain": st.meso_gain,
                "micro_gain": st.micro_gain, "map_resolution": int(disp.shape[0]),
                "meso_source": "photo (delit texture, Hessian crease detection)",
                "micro_source": "synthesized (pores, micro-grooves, lip striations)"}

        def _mesh():
            uv, uvf = self.uv_layout()
            vm = load_vertex_masks()
            pinned = np.unique(np.concatenate([vm["boundary"], vm["left_eyeball"], vm["right_eyeball"]]))
            V, F, U, UF = apply_to_mesh(neutral_vertices, faces, uv, uvf, disp, st.subdivision, pinned)
            path = Path(output_dir) / "head_mesh_detail.obj"
            export_obj_with_uvs(path, V.astype(np.float32), F, U, UF)
            info.update({"mesh": str(path), "mesh_vertices": int(len(V)), "mesh_triangles": int(len(F))})
            return path
        self._run_optional(status, "stage3_sculpt_mesh", _mesh)
        return info

    # ------------------------------------------------------------------ export
    def _export(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        detail_maps: Dict[str, str],
        expression_metadata: dict,
        beta: np.ndarray,
        photo_paths: list,
        output_dir: Path,
        facial_hair_manifest: Optional[dict],
        status: Dict[str, Any],
        extra: Optional[dict] = None,
    ) -> Dict[str, Any]:
        uv, uvf = self.uv_layout()
        obj_path = output_dir / "head_mesh.obj"
        with open(obj_path, "w") as f:
            f.write("# Face Geometry Pipeline - canonical neutral base mesh with UVs\n")
            f.write("mtllib head_mesh.mtl\nusemtl skin\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for vt in uv:
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
            for fv, fvt in zip(faces + 1, uvf + 1):
                f.write(f"f {fv[0]}/{fvt[0]} {fv[1]}/{fvt[1]} {fv[2]}/{fvt[2]}\n")
        if detail_maps.get("albedo"):
            rel = Path(detail_maps["albedo"]).resolve().relative_to(output_dir.resolve())
            (output_dir / "head_mesh.mtl").write_text(f"newmtl skin\nKd 1 1 1\nmap_Kd {rel}\n")

        stage5_manifest: Dict[str, Any] = {}
        if self._c("stage5").get("enabled", True):
            def _stage5():
                from src.stage5_export.exporter import Stage5Exporter
                c5 = self._c("stage5")
                exporter = Stage5Exporter(correspondence_w_path=c5.get("correspondence_w"),
                                          enable_lods=True, enable_armature=True)
                return exporter.export_production_asset(
                    neutral_vertices=vertices, faces=faces, output_dir=output_dir,
                    export_fbx=bool(c5.get("export_fbx", True)), stylization_params=c5.get("stylize"),
                    detail_maps=detail_maps, facial_hair_manifest=facial_hair_manifest)
            stage5_manifest = self._run_optional(status, "stage5_export", _stage5) or {}

        manifest = {
            "pipeline_version": "0.2.0-phase0",
            "photo_paths": [str(p) for p in photo_paths],
            "beta_shape": beta.tolist(),
            "expression_metadata": expression_metadata,
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "detail_maps": detail_maps,
            "facial_hair": facial_hair_manifest,
            "stage5_production_assets": stage5_manifest,
        }
        if extra:
            manifest.update(extra)
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=_json_default)

        mh = stage5_manifest.get("metahuman_bridge", {}) or {}
        out = {
            "obj_path": str(obj_path),
            "manifest_path": str(manifest_path),
            "blendshapes_path": stage5_manifest.get("blendshapes_json"),
            "armature_path": stage5_manifest.get("armature_json"),
            "fbx_path": stage5_manifest.get("fbx_file"),
            "metahuman_obj": mh.get("obj_path"),
        }
        out.update(detail_maps)
        return out

    @staticmethod
    def _run_optional(status: Dict[str, Any], name: str, fn):
        try:
            out = fn()
            status[name] = "ok"
            return out
        except Exception as e:  # noqa: BLE001 - recorded in the manifest
            status[name] = f"failed: {type(e).__name__}: {e}"
            print(f"[{name} FAILED] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            return None


def _iod(det) -> float:
    lm = det.landmark_3d_68
    if lm is None:
        return float(np.linalg.norm(det.landmarks_5pt[0] - det.landmarks_5pt[1]))
    return float(np.linalg.norm(lm[36:42, :2].mean(0) - lm[42:48, :2].mean(0)))


def _mouth_interior_pixels(photo: np.ndarray, det, skin: np.ndarray) -> np.ndarray:
    """
    Teeth and mouth-cavity pixels inside the (dilated) outer lip contour: much less chromatic
    than skin (teeth) or much darker (cavity). Complements the inner-lip landmark polygon,
    which misses teeth on wide smiles.
    """
    lm = det.landmark_3d_68[:, :2]
    outer = np.zeros(photo.shape[:2], np.uint8)
    cv2.fillPoly(outer, [np.round(lm[48:60]).astype(np.int32)], 1)
    outer = cv2.dilate(outer, np.ones((max(3, int(0.05 * _iod(det))),) * 2, np.uint8)).astype(bool)
    lab = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ref = skin & ~outer
    if ref.sum() < 100:
        return np.zeros_like(outer)
    c_ref, l_ref = np.median(chroma[ref]), np.median(lab[..., 0][ref])
    teeth = chroma < 0.45 * c_ref
    cavity = lab[..., 0] < 0.45 * l_ref
    bad = outer & (teeth | cavity)
    return cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
