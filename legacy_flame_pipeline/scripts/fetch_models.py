"""
Model Weights & Assets Verification Utility.
Documents exact acquisition instructions and expected locations for:
- MPI FLAME 2020 (generic_model.pkl, head_template.obj)
- MICA Identity Regressor (pretrained.tar)
- SMIRK Expression & Pose Encoder
- Kaggle Models registry checkpoint pulling
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

MODEL_SPECS = {
    'flame': {
        'title': 'MPI FLAME 2020 3D Head Model',
        'url': 'https://flame.is.tue.mpg.de',
        'license': 'FLAME 2020 License (Non-Commercial Research) or FLAME 2023 Open (CC-BY-4.0)',
        'files': [
            'generic_model.pkl',
            'head_template.obj',
        ],
        'local_dir': 'data/flame_model',
        'kaggle_dataset': '/kaggle/input/flame-model',
        'instructions': (
            "1. Register for an account at https://flame.is.tue.mpg.de\n"
            "2. Download 'FLAME 2020' zip archive.\n"
            "3. Extract 'generic_model.pkl' into data/flame_model/generic_model.pkl (local) or upload to Kaggle Dataset 'flame-model'.\n"
            "4. Also extract 'head_template.obj' into data/flame_model/head_template.obj for UV layout parameterization."
        )
    },
    'mica': {
        'title': 'MICA Multi-View Identity Shape Model',
        'url': 'https://github.com/Zielon/MICA',
        'license': 'MICA License (Non-Commercial Research)',
        'files': [
            'pretrained.tar',
        ],
        'local_dir': 'models_cache/mica',
        'kaggle_dataset': '/kaggle/input/mica-pretrained',
        'instructions': (
            "1. Download pretrained checkpoint from MICA releases / project page (https://github.com/Zielon/MICA).\n"
            "2. Place archive at models_cache/mica/pretrained.tar (local) or attach as Kaggle Dataset 'mica-pretrained'."
        )
    },
    'smirk': {
        'title': 'SMIRK Expression & Joint Pose Encoder',
        'url': 'https://github.com/georgeretsi/smirk',
        'license': 'SMIRK Research License',
        'files': [
            'smirk_encoder.pt',
        ],
        'local_dir': 'models_cache/smirk',
        'kaggle_dataset': '/kaggle/input/smirk-pretrained',
        'instructions': (
            "1. Download SMIRK encoder weights from https://github.com/georgeretsi/smirk.\n"
            "2. Place at models_cache/smirk/smirk_encoder.pt or attach as Kaggle Dataset 'smirk-pretrained'."
        )
    },
    'insightface': {
        'title': 'InsightFace Buffalo_L Detection & Recognition Pack',
        'url': 'https://github.com/deepinsight/insightface',
        'license': 'Non-commercial research',
        'files': [
            'w600k_r50.onnx',
            '2d106det.onnx',
            'det_10g.onnx',
        ],
        'local_dir': os.path.expanduser('~/.insightface/models/buffalo_l'),
        'kaggle_dataset': '/root/.insightface/models/buffalo_l',
        'instructions': (
            "Auto-downloaded by InsightFace on first run: FaceAnalysis(name='buffalo_l').\n"
            "Cached automatically in ~/.insightface/models/buffalo_l/."
        )
    }
}


def verify_model_assets(is_kaggle: bool = False) -> Dict[str, bool]:
    """
    Scans environment for required weights and model assets.
    Returns dictionary of {model_key: is_available}.
    """
    print("=" * 70)
    print(" 3D Face Geometry Pipeline — Model Assets Status")
    print("=" * 70)

    status = {}
    for key, spec in MODEL_SPECS.items():
        search_dirs = [
            Path(spec['local_dir']),
            Path(spec['kaggle_dataset']),
        ]
        if is_kaggle or os.path.exists('/kaggle'):
            search_dirs.insert(0, Path(spec['kaggle_dataset']))

        found_files = []
        missing_files = []
        resolved_dir = None

        for target_dir in search_dirs:
            if target_dir.exists():
                resolved_dir = target_dir
                break

        if resolved_dir is not None:
            for fname in spec['files']:
                if (resolved_dir / fname).exists():
                    found_files.append(fname)
                else:
                    missing_files.append(fname)
        else:
            missing_files = list(spec['files'])

        all_present = (len(missing_files) == 0 and len(found_files) > 0)
        status[key] = all_present

        icon = "✅" if all_present else "❌"
        print(f"\n{icon} [{spec['title']}]")
        print(f"   Status: {'READY' if all_present else 'MISSING ASSETS'}")
        if resolved_dir:
            print(f"   Discovered Path: {resolved_dir.resolve()}")
        if missing_files:
            print(f"   Missing File(s): {', '.join(missing_files)}")
            print(f"   How to Acquire:\n   {spec['instructions'].replace(chr(10), chr(10) + '   ')}")

    print("\n" + "=" * 70)
    return status


def fetch_kaggle_model(kaggle_handle: str, dest_dir: str = None) -> str:
    """
    Downloads model checkpoints from Kaggle Models registry using kagglehub.
    Example handle: 'username/face-geo-stage1-identity/pytorch/v1'
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub is required to fetch models. Run: pip install kagglehub")

    print(f"Fetching Kaggle model artifact: {kaggle_handle}")
    path = kagglehub.model_download(kaggle_handle)
    print(f"Artifact downloaded successfully to: {path}")
    return path


if __name__ == '__main__':
    is_kaggle = os.path.exists('/kaggle')
    status = verify_model_assets(is_kaggle=is_kaggle)
    if not all(status.values()):
        print("\nNote: Research pipeline can run unit tests locally; GPU execution on Kaggle requires attaching the missing weights.")
