"""
Uploads trained model checkpoints to the Kaggle Models registry.
Enforces strict pre-upload gates:
- EMA generator weights must exist for Stage 3+ (raw weights prohibited)
- Normalization statistics (p99 displacement mm) must accompany checkpoints
"""
import os
import json
import argparse
from pathlib import Path

def upload_checkpoint(
    checkpoint_dir: str,
    handle: str,
    version_notes: str,
    stage: int,
):
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f"Checkpoint directory does not exist: {checkpoint_dir}"

    # Stage 3+ Detail Synthesis Gate: Enforce EMA weights
    if stage >= 3:
        ema_file = checkpoint_dir / 'ema_generator.pt'
        if not ema_file.exists():
            raise FileNotFoundError(
                f"\n[Validation Blocked] EMA checkpoint not found at: {ema_file}\n"
                "DO NOT upload raw generator.pt weights. EMA weights significantly reduce high-frequency artifacts.\n"
                "Ensure update_ema() has saved ema_generator.pt before uploading."
            )

        norm_stats = checkpoint_dir / 'normalization_stats.json'
        if not norm_stats.exists():
            raise FileNotFoundError(
                f"\n[Validation Blocked] normalization_stats.json not found at: {norm_stats}\n"
                "This file specifies the p99 metric required for denormalizing displacement maps back to millimeters."
            )

    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub is required for uploading to Kaggle Models. Run: pip install kagglehub")

    print(f"Uploading artifacts from {checkpoint_dir} to Kaggle Models: {handle}")
    print(f"Version notes: {version_notes}")

    kagglehub.model_upload(
        handle=handle,
        local_model_dir=str(checkpoint_dir),
        version_notes=version_notes,
    )
    print(f"Upload complete for model handle: {handle}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload checkpoint to Kaggle Models registry")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--handle', type=str, required=True, help="format: username/model-slug/framework/variation")
    parser.add_argument('--version_notes', type=str, required=True)
    parser.add_argument('--stage', type=int, required=True)
    args = parser.parse_args()

    upload_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        handle=args.handle,
        version_notes=args.version_notes,
        stage=args.stage
    )
