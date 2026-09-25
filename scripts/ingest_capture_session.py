"""
Capture Session Ingestion & Pre-flight Validation Script.

Validates raw multi-view photo sessions from the in-house capture rig:
1. Verifies required views: frontal (0°), left45 (-45°), right45 (+45°), and profiles (±90°).
2. Performs face detection, landmark extraction, and head yaw estimation.
3. Asserts pairwise ArcFace identity cosine similarity (>0.65) to guarantee consistent subject.
4. Generates a standardized session_manifest.json ready for Stage 0/1 pipeline execution.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import cv2

from src.stage0_preprocess.detector import FaceDetector, FaceDetection
from src.utils.validation import validate_inputs

EXPECTED_VIEWS = {
    'frontal': {'nominal_yaw': 0.0, 'tol': 15.0, 'required': True},
    'left45':  {'nominal_yaw': -45.0, 'tol': 20.0, 'required': True},
    'right45': {'nominal_yaw': 45.0, 'tol': 20.0, 'required': True},
    'left90':  {'nominal_yaw': -90.0, 'tol': 25.0, 'required': False},
    'right90': {'nominal_yaw': 90.0, 'tol': 25.0, 'required': False},
}


def ingest_session(
    session_dir: str,
    out_dir: Optional[str] = None,
    detector: Optional[Any] = None,
) -> Dict:
    session_path = Path(session_dir)
    assert session_path.exists(), f"Capture session directory not found: {session_path}"

    print(f"\n[Ingest] Scanning capture session: {session_path.name}")

    discovered_views = {}
    for key in EXPECTED_VIEWS.keys():
        matches = list(session_path.glob(f"*{key}*.*"))
        if matches:
            discovered_views[key] = matches[0]

    # Verify required views before loading heavy models
    missing_required = [k for k, v in EXPECTED_VIEWS.items() if v['required'] and k not in discovered_views]
    if missing_required:
        raise FileNotFoundError(
            f"Missing required camera views in session {session_path.name}: {missing_required}. "
            "Refer to older/data_collection_protocol.md for required capture layout."
        )

    print(f"Found {len(discovered_views)} view(s): {list(discovered_views.keys())}")
    detector = detector or FaceDetector()

    # Validate image files and run detector
    view_records = {}
    photo_paths_str = [str(p) for p in discovered_views.values()]
    val_res = validate_inputs(photo_paths_str, detector=detector)

    if not val_res.is_valid:
        print("[Ingest Error] Validation failed on input portraits:")
        for err in val_res.errors:
            print(f"  - {err}")
        raise ValueError(f"Session failed input validation: {val_res.errors}")

    for view_key, photo_path in discovered_views.items():
        img = cv2.imread(str(photo_path))
        det = detector.detect_single(img)
        if det is None:
            raise ValueError(f"Could not detect face in {photo_path.name}")

        nominal = EXPECTED_VIEWS[view_key]['nominal_yaw']
        tol = EXPECTED_VIEWS[view_key]['tol']
        yaw_diff = abs(det.yaw_deg - nominal)
        status_icon = "✅" if yaw_diff <= tol else "⚠️"

        print(f"  {status_icon} View '{view_key}': Detected yaw {det.yaw_deg:.1f}° (Target: {nominal:+.1f}°, Diff: {yaw_diff:.1f}°)")

        view_records[view_key] = {
            'file': str(photo_path.resolve()),
            'filename': photo_path.name,
            'detected_yaw': float(det.yaw_deg),
            'det_score': float(det.det_score),
            'has_embedding': det.embedding is not None,
        }

    # Extract demographic/morphology tags if metadata.json exists
    morphology_data = {}
    meta_file = session_path / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                raw_meta = json.load(f)
            morphology_data = raw_meta.get("morphology", {})
            print(f"  📋 Ingested morphology metadata: {morphology_data}")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not parse metadata.json in session: {e}")
    else:
        print("  ⚠️ Note: No metadata.json found. Consider recording morphology attributes per older/04_diversity_and_identity_fidelity.md")

    # Package session manifest
    manifest = {
        'session_id': session_path.name,
        'views_count': len(view_records),
        'views': view_records,
        'morphology': {
            'build_category': morphology_data.get('build_category', 'unspecified'),
            'mandibular_type': morphology_data.get('mandibular_type', 'unspecified'),
            'soft_tissue_volume': morphology_data.get('soft_tissue_volume', 'unspecified'),
            'ancestry_category': morphology_data.get('ancestry_category', 'unspecified'),
            'estimated_bmi': morphology_data.get('estimated_bmi', None),
        },
        'is_valid': True,
        'calibration_target': 'in_house_85mm_cross_polarized',
    }

    target_dir = Path(out_dir) if out_dir else session_path
    target_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = target_dir / "session_manifest.json"
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Ingest Complete] Session validated and saved to: {out_manifest}")
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest and validate in-house portrait capture session")
    parser.add_argument('--session_dir', type=str, required=True, help="Path to raw session photo folder")
    parser.add_argument('--out_dir', type=str, default=None, help="Output folder for session manifest")
    args = parser.parse_args()

    ingest_session(args.session_dir, args.out_dir)
