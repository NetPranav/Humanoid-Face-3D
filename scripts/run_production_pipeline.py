#!/usr/bin/env python3
"""
Master End-to-End Production Pipeline Script.

Executes the full Humanoid-Face-3D reconstruction pipeline sequentially:
  Stage 0: Input validation, face detection, pose estimation
  Stage 1: Multi-view identity shape regression (β)
  Stage 2: Expression/pose regression (ψ, θ)
  Stage 3: Micro-displacement synthesis (1024×1024 16-bit)
  Stage 6: Multi-view UV texture backprojection
  Stage 7: AI delighting + UV inpainting → clean diffuse albedo
  Stage 8: Procedural PBR material stack (roughness, cavity/AO, SSS)
  Stage 5: 320k Loop subdivision + production FBX packaging

Usage:
    python scripts/run_production_pipeline.py \\
        --photos path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg \\
        --output_dir outputs/production_run \\
        --config configs/default.yaml
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import FaceGeoPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Humanoid-Face-3D Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--photos", nargs="+", required=True,
        help="3-5 multi-view portrait photographs of the subject"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/production_run",
        help="Output directory for all generated assets"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Pipeline configuration YAML file"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models_cache",
        help="Directory containing pretrained model weights"
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Displacement map resolution (512 or 1024)"
    )
    parser.add_argument(
        "--no_texture", action="store_true",
        help="Skip texture engine (Stages 6, 7, 8)"
    )
    parser.add_argument(
        "--no_fbx", action="store_true",
        help="Skip FBX production packaging (Stage 5)"
    )
    parser.add_argument(
        "--no_detail", action="store_true",
        help="Skip micro-displacement synthesis (Stage 3)"
    )
    args = parser.parse_args()

    # Validate inputs exist
    photo_paths = []
    for p in args.photos:
        pp = Path(p)
        if not pp.exists():
            print(f"ERROR: Photo not found: {p}")
            sys.exit(1)
        photo_paths.append(str(pp.resolve()))

    if len(photo_paths) < 1:
        print("ERROR: At least 1 photo required (3-5 recommended for best results).")
        sys.exit(1)

    # Load or build config
    config_path = Path(args.config)
    import yaml
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(f"[INFO] Config file not found at {config_path}, using defaults.")
        cfg = {}

    # Apply CLI overrides to config
    if 'stage3' not in cfg:
        cfg['stage3'] = {}
    cfg['stage3']['resolution'] = args.resolution

    if args.no_texture:
        cfg.setdefault('stage6', {})['enabled'] = False
        cfg.setdefault('stage7', {})['enabled'] = False
        cfg.setdefault('stage8', {})['enabled'] = False

    if args.no_detail:
        cfg['stage3']['enabled'] = False

    if args.no_fbx:
        cfg.setdefault('stage5', {})['export_fbx'] = False

    # Run pipeline
    print("=" * 72)
    print("  HUMANOID-FACE-3D: Production Pipeline")
    print("=" * 72)
    print(f"  Photos:      {len(photo_paths)}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Config:      {args.config}")
    print(f"  Resolution:  {args.resolution}x{args.resolution}")
    print(f"  Textures:    {'OFF' if args.no_texture else 'ON'}")
    print(f"  Detail GAN:  {'OFF' if args.no_detail else 'ON'}")
    print(f"  FBX Export:  {'OFF' if args.no_fbx else 'ON'}")
    print("=" * 72)

    start = time.time()

    pipeline = FaceGeoPipeline(
        config_path=cfg,
        model_dir=args.model_dir,
    )

    result = pipeline.run(
        photo_paths=photo_paths,
        output_dir=args.output_dir,
    )

    elapsed = time.time() - start

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)
    print(f"  Total time:           {elapsed:.1f}s")
    print(f"  OBJ mesh:             {result.get('obj_path', 'N/A')}")
    print(f"  FBX package:          {result.get('fbx_path', 'N/A')}")
    print(f"  Displacement 16-bit:  {result.get('displacement_16bit', 'N/A')}")
    print(f"  Normal map:           {result.get('normal_map', 'N/A')}")
    print(f"  Albedo:               {result.get('albedo', 'N/A')}")
    print(f"  Roughness:            {result.get('roughness', 'N/A')}")
    print(f"  Cavity/AO:            {result.get('cavity', 'N/A')}")
    print(f"  SSS thickness:        {result.get('sss_thickness', 'N/A')}")
    print(f"  Manifest:             {result.get('manifest_path', 'N/A')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
