"""
Utility to download pretrained checkpoints and pull trained weights
from Kaggle Models registry.
"""
import os
from pathlib import Path

PRETRAINED_SOURCES = {
    'mica': {
        'url': 'https://github.com/Zielon/MICA',
        'dest': '/kaggle/working/models/mica/pretrained.tar',
        'note': 'Acquire from MICA releases / project page and attach as Kaggle Dataset input.'
    },
    'smirk': {
        'url': 'https://github.com/georgeretsi/smirk',
        'dest': '/kaggle/working/models/smirk/pretrained.tar',
        'note': 'Acquire from SMIRK releases and attach as Kaggle Dataset input.'
    },
    'flame': {
        'url': 'https://flame.is.tue.mpg.de',
        'dest': '/kaggle/working/models/flame/generic_model.pkl',
        'note': 'Register at MPI FLAME site and upload generic_model.pkl to Kaggle Dataset.'
    }
}

def print_fetch_instructions():
    print("Pretrained Weights Acquisition Guide:")
    for name, info in PRETRAINED_SOURCES.items():
        print(f"\n[{name.upper()}]")
        print(f" Source: {info['url']}")
        print(f" Target Path: {info['dest']}")
        print(f" Instructions: {info['note']}")

def fetch_kaggle_model(kaggle_handle: str, dest_dir: str = None) -> str:
    """
    Downloads fine-tuned model artifacts from Kaggle Models registry using kagglehub.
    Example handle: 'youruser/face-geo-stage1-identity/pytorch/v1'
    """
    import kagglehub
    print(f"Fetching Kaggle model artifact: {kaggle_handle}")
    path = kagglehub.model_download(kaggle_handle)
    print(f"Artifact downloaded successfully to: {path}")
    return path

if __name__ == '__main__':
    print_fetch_instructions()
