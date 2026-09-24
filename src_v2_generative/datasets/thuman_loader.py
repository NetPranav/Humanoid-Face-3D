"""
THuman 2.1 Volumetric Hair & Photogrammetry Scan Ingestion.

Handles 2,500 high-resolution scans captured across 120 DSLR cameras,
extracting head, hair silhouette, and neck/collar transitions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class THumanModel:
    """Represents a single THuman 2.1 high-resolution photogrammetry scan."""

    def __init__(self, subject_id: str, obj_path: Path, texture_path: Optional[Path] = None):
        self.subject_id = subject_id
        self.obj_path = obj_path
        self.texture_path = texture_path
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None

    def extract_head_crop(self, y_min_ratio: float = 0.70) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops the full-body scan to the cranial, facial, and neck region.
        In normalized standing pose, the head and neck occupy the top 30% (y > y_min_ratio).
        """
        if self.vertices is None or self.faces is None:
            self._load_obj()

        assert self.vertices is not None
        assert self.faces is not None

        y_vals = self.vertices[:, 1]
        y_min = np.min(y_vals)
        y_max = np.max(y_vals)
        height = y_max - y_min

        cutoff = y_min + height * y_min_ratio
        head_vert_mask = y_vals >= cutoff

        # Remap vertex indices for selected faces
        vert_indices = np.where(head_vert_mask)[0]
        index_map = {old: new for new, old in enumerate(vert_indices)}

        head_faces = []
        for face in self.faces:
            if all(v in index_map for v in face):
                head_faces.append([index_map[v] for v in face])

        cropped_verts = self.vertices[vert_indices]
        cropped_faces = np.array(head_faces, dtype=np.int32)

        return cropped_verts, cropped_faces

    def _load_obj(self) -> None:
        """Loads raw OBJ geometry."""
        if not self.obj_path.exists():
            raise FileNotFoundError(f"THuman scan not found at: {self.obj_path}")

        verts = []
        faces = []
        with open(self.obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    face_v = [int(p.split("/")[0]) - 1 for p in parts]
                    faces.append(face_v)

        self.vertices = np.array(verts, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
