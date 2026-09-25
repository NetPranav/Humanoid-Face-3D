"""
FLAME 68-landmark embedding as a sparse linear map (moved here from the removed Stage 1.5
contour deformer; used by the Stage 5 MetaHuman bridge).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_flame_landmark_matrix(
    faces: np.ndarray,
    n_verts: int,
    embedding_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Constructs the linear mapping matrix M_lmk in R^{68 x N_verts} from FLAME's
    landmark barycentric embedding, such that L_3d = M_lmk @ V.
    """
    candidates = []
    if embedding_path:
        candidates.append(Path(embedding_path))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend([
        root / 'data' / 'flame_model' / 'landmark_embedding.npy',
        root / 'vendor' / 'MICA' / 'data' / 'FLAME2020' / 'landmark_embedding.npy',
        root / 'data' / 'FLAME2020' / 'landmark_embedding.npy',
    ])

    for cand in candidates:
        if cand.exists():
            try:
                emb = np.load(cand, allow_pickle=True, encoding='latin1')
                if hasattr(emb, 'item'):
                    emb = emb.item()
                elif isinstance(emb, np.ndarray) and emb.dtype == object:
                    emb = emb[()]

                lmk_faces = emb['full_lmk_faces_idx']
                if lmk_faces.ndim > 1:
                    lmk_faces = lmk_faces[0]
                lmk_bary = emb['full_lmk_bary_coords']
                if lmk_bary.ndim > 2:
                    lmk_bary = lmk_bary[0]

                M = np.zeros((68, n_verts), dtype=np.float32)
                for i in range(min(68, len(lmk_faces))):
                    f_idx = lmk_faces[i]
                    if f_idx < len(faces):
                        f = faces[f_idx]
                        for k in range(3):
                            if f[k] < n_verts:
                                M[i, f[k]] += float(lmk_bary[i, k])
                    if np.sum(M[i]) == 0 and n_verts > 0:
                        M[i, i % n_verts] = 1.0
                return M
            except Exception as e:
                logger.warning("Failed loading landmark embedding %s: %s", cand, e)

    raise FileNotFoundError(
        "Could not find FLAME landmark_embedding.npy in data/flame_model/ or vendor/MICA/."
    )
