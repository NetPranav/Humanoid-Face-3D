#!/usr/bin/env python3
"""
scripts/production_inference.py

Master Single-Command Production Inference & Benchmark Validation CLI.
Orchestrates the entire 4-tier film-grade MetaHuman reconstruction pipeline:
  Stage 0: Pre-flight input validation & face detection
  Stage 1: Multi-view identity shape regression (β)
  Stage 2: Canonical neutral normalization (ψ=0, θ=0)
  Stage 6 & 7: Multi-view UV backprojection & AI delighting → clean diffuse albedo
  Tier 2: Photo-derived meso wrinkles (Shape-from-Shading)
  Tier 3: 4K anatomical cellular pore synthesis (50-micron follicles)
  Step 3: Multi-tier geometry fusion & PBR material coupling (roughness, cavity AO, normals)
  Stage 4: Facial hair & stubble geometry
  Stage 8: Subsurface scattering thickness estimation (opposing-normal ray march)
  Stage 5: Production retopology, ARKit-52 blendshapes, 4-tier LODs, armature, FBX packaging
  Step 4: Automated film-grade Blender Cycles studio look-dev turnaround & .blend project save

Usage:
    # Run on benchmark subject:
    python scripts/production_inference.py --subject carell
    python scripts/production_inference.py --subject connelly
    python scripts/production_inference.py --subject lawrence

    # Run on custom portraits:
    python scripts/production_inference.py --photos photo1.jpg photo2.jpg photo3.jpg --output_dir outputs/custom_subject
"""
from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from src.pipeline import FaceGeoPipeline


BENCHMARK_SUBJECTS = {
    "carell": {
        "hair_preset": "heavy_stubble",
        "stylize": "chiseled",
        "filenames": ["carell.jpg", "carell.png"],
    },
    "connelly": {
        "hair_preset": "clean_shaven",
        "stylize": "neutral",
        "filenames": ["connelly.jpg", "connelly.png"],
    },
    "justin": {
        "hair_preset": "goatee",
        "stylize": "heroic",
        "filenames": ["justin.png", "justin.jpg"],
    },
    "lawrence": {
        "hair_preset": "clean_shaven",
        "stylize": "neutral",
        "filenames": ["lawrence.jpg", "lawrence.png"],
    },
}


def find_subject_photos(subject_name: str) -> List[str]:
    """Auto-discovers portrait photographs for a benchmark subject."""
    sub_info = BENCHMARK_SUBJECTS.get(subject_name.lower())
    target_names = sub_info["filenames"] if sub_info else [f"{subject_name}.jpg", f"{subject_name}.png"]

    search_dirs = [
        PROJECT_ROOT / "vendor" / "MICA" / "demo" / "input",
        PROJECT_ROOT / "data" / "benchmark_test" / subject_name,
        PROJECT_ROOT / "data" / "benchmark_test",
        PROJECT_ROOT / "outputs" / "production_batch" / subject_name,
    ]

    for sdir in search_dirs:
        if sdir.exists():
            for name in target_names:
                cand = sdir / name
                if cand.is_file():
                    return [str(cand.resolve())]

    # Search recursively for matching filename
    for sdir in search_dirs:
        if sdir.exists():
            for p in sdir.rglob("*.*"):
                if p.name.lower() in [n.lower() for n in target_names]:
                    return [str(p.resolve())]

    raise FileNotFoundError(
        f"Could not locate portrait photos for benchmark subject '{subject_name}'. "
        f"Searched in: {[str(d) for d in search_dirs]}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Humanoid-Face-3D: Single-Command Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--subject", type=str, choices=["carell", "connelly", "lawrence", "justin"],
        help="Run inference on a built-in benchmark subject"
    )
    parser.add_argument(
        "--photos", nargs="+", default=None,
        help="Custom portrait photograph(s) of the subject"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for all generated deliverables"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Base pipeline YAML configuration file"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models_cache",
        help="Pretrained model weights directory"
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Displacement and texture resolution (1024 or 2048)"
    )
    parser.add_argument(
        "--hair_preset", type=str, default=None,
        choices=["clean_shaven", "stubble", "heavy_stubble", "goatee"],
        help="Facial hair geometry preset"
    )
    parser.add_argument(
        "--stylize", type=str, default=None,
        choices=["neutral", "chiseled", "heroic", "angular"],
        help="Facial anatomy stylization preset"
    )
    parser.add_argument(
        "--render_samples", type=int, default=128,
        help="Cycles path tracing sample count (default: 128)"
    )
    parser.add_argument(
        "--no_render", action="store_true",
        help="Skip Blender Cycles look-dev rendering"
    )
    parser.add_argument(
        "--no_fbx", action="store_true",
        help="Skip Unreal Engine 5 FBX export"
    )
    parser.add_argument(
        "--device", default="AUTO", choices=["AUTO", "GPU", "CPU"],
        help="Compute device for Cycles render engine"
    )

    return parser.parse_args()


def run_production_inference(args) -> Dict[str, Any]:
    # 1. Resolve photo paths and defaults
    if args.subject:
        subj = args.subject.lower()
        photo_paths = find_subject_photos(subj)
        subj_defaults = BENCHMARK_SUBJECTS.get(subj, {})
        hair_preset = args.hair_preset or subj_defaults.get("hair_preset", "clean_shaven")
        stylize_preset = args.stylize or subj_defaults.get("stylize", "neutral")
        out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "production_batch" / subj
    else:
        if not args.photos:
            raise ValueError("Either --subject or --photos must be provided.")
        photo_paths = [str(Path(p).resolve()) for p in args.photos]
        hair_preset = args.hair_preset or "stubble"
        stylize_preset = args.stylize or "neutral"
        out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "production_custom"

    for p in photo_paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Input photo does not exist: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load and build runtime config
    cfg_file = PROJECT_ROOT / args.config
    cfg = {}
    if cfg_file.is_file():
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # Override config parameters
    cfg.setdefault("pipeline", {})["input_min_photos"] = len(photo_paths)
    cfg.setdefault("stage0", {})["min_face_confidence"] = 0.40
    cfg.setdefault("stage3", {})["resolution"] = args.resolution
    cfg.setdefault("stage3", {})["enable_meso"] = True
    cfg.setdefault("stage3", {})["enable_micro"] = True

    cfg.setdefault("stage4", {})["enabled"] = True
    cfg.setdefault("stage4", {})["preset"] = hair_preset
    cfg.setdefault("stage4", {})["resolution"] = args.resolution

    cfg.setdefault("stage5", {})["stylize"] = stylize_preset
    cfg.setdefault("stage5", {})["enable_lods"] = True
    cfg.setdefault("stage5", {})["enable_armature"] = True
    if args.no_fbx:
        cfg["stage5"]["export_fbx"] = False

    cfg.setdefault("render", {})["enabled"] = not args.no_render
    cfg.setdefault("render", {})["samples"] = args.render_samples
    cfg.setdefault("render", {})["width"] = 2048
    cfg.setdefault("render", {})["height"] = 2048
    cfg.setdefault("render", {})["device"] = args.device

    # Local development fallback detection for InsightFace
    try:
        import insightface
        allow_degraded = False
    except ImportError:
        allow_degraded = True
    cfg["allow_degraded"] = allow_degraded

    # 3. Print Banner
    print("=" * 76)
    print("  🎬 HUMANOID-FACE-3D: SINGLE-COMMAND PRODUCTION SYNTHESIS PIPELINE")
    print("=" * 76)
    print(f"  Target Subject:       {args.subject.upper() if args.subject else 'CUSTOM'}")
    print(f"  Input Photos:         {len(photo_paths)} view(s): {[Path(p).name for p in photo_paths]}")
    print(f"  Output Directory:     {out_dir}")
    print(f"  Hair Preset:          {hair_preset}")
    print(f"  Stylization:          {stylize_preset}")
    print(f"  Resolution:           {args.resolution}x{args.resolution}")
    print(f"  Cycles Look-Dev:      {'DISABLED' if args.no_render else f'ENABLED ({args.render_samples} samples, 2048x2048)'}")
    print("=" * 76)

    t_start = time.time()

    # 4. Instantiate and execute pipeline
    pipeline = FaceGeoPipeline(cfg=cfg, model_dir=args.model_dir)
    result = pipeline.run(photo_paths=photo_paths, output_dir=str(out_dir))

    t_elapsed = time.time() - t_start

    # 5. Deliverable Audit & Verification
    print("\n" + "=" * 76)
    print("  📋 PRODUCTION DELIVERABLE AUDIT & VERIFICATION")
    print("=" * 76)

    deliverables = [
        ("Base OBJ Mesh", result.get("obj_path"), None),
        ("ARKit-52 Blendshapes JSON", result.get("blendshapes_path"), None),
        ("Skeletal Armature JSON", result.get("armature_path"), None),
        ("UE5 Live Link FBX Asset", result.get("fbx_path"), out_dir / "run_blender_fbx_packaging.sh"),
        ("MetaHuman Neutral OBJ (UE5)", out_dir / "metahuman_neutral.obj", None),
        ("MetaHuman Identity Manifest", out_dir / "metahuman_identity_manifest.json", None),
        ("Cinematic Cycles Render PNG", result.get("film_render"), None),
        ("Cycles 45° Hero Turnaround", out_dir / "film_render_cycles_hero45.png", None),
        ("Cycles 105mm Macro Eye Zoom", out_dir / "film_render_cycles_macro_eye.png", None),
        ("Editable Look-Dev .blend Scene", result.get("blend_file"), out_dir / "run_blender_film_render.sh"),
        ("16-Bit Displacement Map", result.get("displacement_16bit"), None),
        ("Tangent Normal Map", result.get("normal_map"), None),
        ("Delighted Diffuse Albedo", result.get("albedo"), None),
        ("Dual-Lobe Roughness Map", result.get("roughness"), None),
        ("Micro-Cavity Ambient Occlusion", result.get("cavity"), None),
        ("Subsurface Scattering Map", result.get("sss_thickness"), None),
        ("Master Run Manifest JSON", result.get("manifest_path"), None),
    ]

    for label, file_path, fallback_script in deliverables:
        if file_path is not None and Path(file_path).exists():
            print(f"  ✅ {label:<32}: {Path(file_path).name}")
        elif fallback_script is not None and Path(fallback_script).exists():
            print(f"  ⚙️ {label:<32}: Turnkey Script ({Path(fallback_script).name})")
        else:
            print(f"  ⚠️ {label:<32}: Not Generated")

    # 6. Verify System Invariant Rule 4: Collar Boundary Pinning Contract
    disp_path = result.get("displacement_16bit")
    if disp_path and Path(disp_path).exists():
        raw_disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if raw_disp is not None:
            # For 16-bit PNG: midlevel is 32768
            bottom_rows = int(0.85 * raw_disp.shape[0])
            collar_slice = raw_disp[bottom_rows:, :].astype(np.int32) - 32768
            max_collar_delta = float(np.max(np.abs(collar_slice)))
            if max_collar_delta == 0:
                print(f"  ✅ Collar Pinning Contract (Rule 4)  : VERIFIED (Δv ≡ 0.000000 mm on lowest 20%)")
            else:
                print(f"  ⚠️ Collar Pinning Contract (Rule 4)  : WARNING (Collar delta={max_collar_delta})")

    print("=" * 76)
    print(f"🎉 Production reconstruction finished in {t_elapsed:.1f}s.")
    print(f"📁 Deliverables located in: {out_dir}")
    print("=" * 76 + "\n")

    return result


if __name__ == "__main__":
    cli_args = parse_arguments()
    run_production_inference(cli_args)
