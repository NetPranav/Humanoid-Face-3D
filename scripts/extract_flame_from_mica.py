"""
Rebuild FLAME 2020 model files from the official MICA checkpoint.

MICA's `mica.tar` (downloaded by vendor/MICA/install.sh) stores the complete FLAME
generator as buffers: template, 300 identity + 100 expression bases, pose correctives,
joint regressor, skinning weights and the static/dynamic/full landmark embeddings.
FLAME's license terms still apply to the extracted arrays.

Writes:
  data/flame_model/generic_model.pkl       (keys expected by src/utils/flame_model.py)
  data/flame_model/landmark_embedding_mica.npz  (static 51 + dynamic 79x17 contour + full 68)

Usage:
  python3 scripts/extract_flame_from_mica.py --mica models_cache/mica/mica.tar
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

N_VERTS = 5023


def extract(mica_path: Path, out_dir: Path) -> None:
    if not mica_path.exists():
        raise FileNotFoundError(
            f"MICA checkpoint not found at {mica_path}. Download it with:\n"
            "  python3 -m gdown 1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt -O models_cache/mica/mica.tar"
        )
    ckpt = torch.load(mica_path, map_location="cpu", weights_only=False)
    g = {k[len("generator."):]: v.numpy() for k, v in ckpt["flameModel"].items() if k.startswith("generator.")}

    required = ["v_template", "shapedirs", "posedirs", "J_regressor", "lbs_weights", "faces_tensor", "parents"]
    missing = [k for k in required if k not in g]
    if missing:
        raise RuntimeError(f"MICA checkpoint lacks FLAME buffers {missing}; is this a MICA checkpoint?")

    # MICA stores posedirs as (36, V*3) = reshape(orig, [-1, 36]).T
    posedirs = g["posedirs"].T.reshape(N_VERTS, 3, -1)
    parents = g["parents"].astype(np.int64)
    kintree = np.stack([parents, np.arange(len(parents))]).astype(np.int64)

    model = {
        "v_template": g["v_template"].astype(np.float64),
        "shapedirs": g["shapedirs"].astype(np.float64),       # (5023, 3, 400): 300 id + 100 expr
        "posedirs": posedirs.astype(np.float64),              # (5023, 3, 36)
        "J_regressor": g["J_regressor"].astype(np.float64),   # dense (5, 5023)
        "weights": g["lbs_weights"].astype(np.float64),
        "f": g["faces_tensor"].astype(np.uint32),
        "kintree_table": kintree,
        "source": f"extracted from {mica_path.name}",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "generic_model.pkl", "wb") as fh:
        pickle.dump(model, fh, protocol=2)

    np.savez(
        out_dir / "landmark_embedding_mica.npz",
        static_lmk_faces_idx=g["lmk_faces_idx"],
        static_lmk_bary_coords=g["lmk_bary_coords"],
        dynamic_lmk_faces_idx=g["dynamic_lmk_faces_idx"],
        dynamic_lmk_bary_coords=g["dynamic_lmk_bary_coords"],
        full_lmk_faces_idx=g["full_lmk_faces_idx"].reshape(-1),
        full_lmk_bary_coords=g["full_lmk_bary_coords"].reshape(-1, 3),
    )
    print(f"Wrote {out_dir / 'generic_model.pkl'} and landmark_embedding_mica.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mica", default="models_cache/mica/mica.tar")
    ap.add_argument("--out", default="data/flame_model")
    a = ap.parse_args()
    extract(Path(a.mica), Path(a.out))
