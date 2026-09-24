"""
USC ICT-FaceKit Topology Loader & Semantic Submesh Partitioning.

Handles the 26,719-vertex Light Stage photogrammetry topology (MIT License),
including inner mouth cavity, teeth, gums, tongue, and eyeball sockets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class ICTFaceKitTopology:
    """
    Manages the 26,719-vertex USC ICT-FaceKit reference geometry and sub-meshes.
    """

    # Anatomical submesh index ranges according to USC ICT specification
    SUBMESH_RANGES = {
        "face": (0, 9409),
        "head_neck": (9409, 11248),
        "mouth_socket": (11248, 13294),
        "eye_socket_left": (13294, 13678),
        "eye_socket_right": (13678, 14062),
        "gums_tongue": (14062, 17039),
        "teeth": (17039, 21451),
        "eyeball_left": (21451, 23021),
        "eyeball_right": (23021, 24591),
        "lacrimal_fluid": (24591, 24999),
        "eyelashes": (25351, 26719),
    }

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.submesh_masks: Dict[str, np.ndarray] = {}

    def get_submesh_mask(self, name: str, total_vertices: int = 26719) -> np.ndarray:
        """Returns a boolean mask for the requested anatomical submesh."""
        if name not in self.SUBMESH_RANGES:
            raise KeyError(f"Unknown submesh name: {name}. Available: {list(self.SUBMESH_RANGES.keys())}")

        start, end = self.SUBMESH_RANGES[name]
        mask = np.zeros(total_vertices, dtype=bool)
        mask[start:end] = True
        return mask

    def build_oral_cavity_mask(self, total_vertices: int = 26719) -> np.ndarray:
        """Combines mouth socket, teeth, gums, and tongue into an oral cavity mask."""
        mask = np.zeros(total_vertices, dtype=bool)
        for part in ["mouth_socket", "gums_tongue", "teeth"]:
            mask |= self.get_submesh_mask(part, total_vertices)
        return mask

    def build_facial_skin_mask(self, total_vertices: int = 26719) -> np.ndarray:
        """Returns mask for outer facial skin, cranium, and neck (excluding inner teeth/tongue)."""
        mask = np.zeros(total_vertices, dtype=bool)
        for part in ["face", "head_neck"]:
            mask |= self.get_submesh_mask(part, total_vertices)
        return mask
