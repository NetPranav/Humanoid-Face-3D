"""
FaceScape Topologically Uniform (TU) Dataset Ingestion & Preprocessing.

Supports 16,940 high-resolution models (847 subjects x 20 expressions),
extracting 26,000-vertex registered meshes, 4K displacement maps, and PBR textures.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2


class FaceScapeModel:
    """Represents a single FaceScape Topologically Uniform 3D head model."""

    def __init__(
        self,
        subject_id: int,
        expression_id: int,
        obj_path: Path,
        dpmap_path: Optional[Path] = None,
        texture_path: Optional[Path] = None,
    ):
        self.subject_id = subject_id
        self.expression_id = expression_id
        self.obj_path = obj_path
        self.dpmap_path = dpmap_path
        self.texture_path = texture_path
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.uv_coords: Optional[np.ndarray] = None
        self.uv_indices: Optional[np.ndarray] = None

    def load_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads vertices and faces from the base OBJ file."""
        if not self.obj_path.exists():
            raise FileNotFoundError(f"FaceScape OBJ not found at: {self.obj_path}")

        verts = []
        faces = []
        uvs = []
        uv_faces = []

        with open(self.obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vt "):
                    parts = line.strip().split()
                    uvs.append([float(parts[1]), float(parts[2])])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    face_v = []
                    face_vt = []
                    for p in parts:
                        vals = p.split("/")
                        face_v.append(int(vals[0]) - 1)
                        if len(vals) > 1 and vals[1]:
                            face_vt.append(int(vals[1]) - 1)
                    faces.append(face_v)
                    if len(face_vt) == len(face_v):
                        uv_faces.append(face_vt)

        self.vertices = np.array(verts, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        if uvs:
            self.uv_coords = np.array(uvs, dtype=np.float32)
        if uv_faces:
            self.uv_indices = np.array(uv_faces, dtype=np.int32)

        return self.vertices, self.faces

    def load_displacement(self, target_resolution: int = 1024) -> Optional[np.ndarray]:
        """
        Loads 4K displacement map, normalizes to millimeter range, and resizes if requested.
        """
        if self.dpmap_path is None or not self.dpmap_path.exists():
            return None

        # Load 16-bit or 8-bit displacement
        disp = cv2.imread(str(self.dpmap_path), cv2.IMREAD_UNCHANGED)
        if disp is None:
            return None

        if disp.ndim == 3:
            disp = disp[:, :, 0]

        if disp.shape[0] != target_resolution or disp.shape[1] != target_resolution:
            disp = cv2.resize(
                disp, (target_resolution, target_resolution), interpolation=cv2.INTER_AREA
            )

        # Convert to float [-1.0, 1.0] or millimeters
        if disp.dtype == np.uint16:
            disp_float = (disp.astype(np.float32) / 65535.0) * 2.0 - 1.0
        else:
            disp_float = (disp.astype(np.float32) / 255.0) * 2.0 - 1.0

        return disp_float

    def load_texture(self, target_resolution: int = 1024) -> Optional[np.ndarray]:
        """Loads and resizes 4K RGB albedo texture."""
        if self.texture_path is None or not self.texture_path.exists():
            return None

        tex = cv2.imread(str(self.texture_path))
        if tex is None:
            return None

        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        if tex.shape[0] != target_resolution or tex.shape[1] != target_resolution:
            tex = cv2.resize(
                tex, (target_resolution, target_resolution), interpolation=cv2.INTER_AREA
            )

        return tex


class FaceScapeDataset:
    """Manages the FaceScape TU models directory and indexing."""

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.models_dir = self.root_dir / "models_reg"
        self.dpmap_dir = self.root_dir / "dpmap"
        self.samples: List[Tuple[int, int]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Discovers all available subject-expression pairs."""
        if not self.models_dir.exists():
            return

        for obj_file in self.models_dir.glob("*.obj"):
            stem = obj_file.stem
            parts = stem.split("_")
            if len(parts) >= 2:
                try:
                    sub_id = int(parts[0])
                    exp_id = int(parts[1])
                    self.samples.append((sub_id, exp_id))
                except ValueError:
                    continue

        self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def get_model(self, index: int) -> FaceScapeModel:
        if index >= len(self.samples):
            raise IndexError(f"Index {index} out of range ({len(self.samples)} available).")

        sub_id, exp_id = self.samples[index]
        obj_path = self.models_dir / f"{sub_id}_{exp_id}.obj"
        dpmap_path = self.dpmap_dir / f"{sub_id}_{exp_id}.png"
        tex_path = self.models_dir / f"{sub_id}_{exp_id}.jpg"

        return FaceScapeModel(
            subject_id=sub_id,
            expression_id=exp_id,
            obj_path=obj_path,
            dpmap_path=dpmap_path if dpmap_path.exists() else None,
            texture_path=tex_path if tex_path.exists() else None,
        )
