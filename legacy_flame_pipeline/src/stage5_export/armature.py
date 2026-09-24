"""
Skeletal Armature Rigging & Skinning Weights Engine.

Constructs the 5-joint skeletal armature hierarchy required by Unreal Engine 5 Live Link:
  neck (root) -> head -> jaw, eye_L, eye_R

Computes anatomical joint centers and Linear Blend Skinning (LBS) weights,
enforcing partition of unity (sum(w_j) = 1.0) and supporting weight transfer
across target retopology meshes via the barycentric correspondence matrix W.
"""
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np


JOINT_HIERARCHY = {
    "neck": {"parent": None, "children": ["head"]},
    "head": {"parent": "neck", "children": ["jaw", "eye_L", "eye_R"]},
    "jaw":  {"parent": "head", "children": []},
    "eye_L": {"parent": "head", "children": []},
    "eye_R": {"parent": "head", "children": []},
}

JOINT_NAMES = ["neck", "head", "jaw", "eye_L", "eye_R"]


def estimate_anatomical_joint_centers(vertices: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Estimates 3D coordinates for the 5 key facial joints from the neutral mesh vertices.
    Based on cranial anatomy:
      - neck: inferior base of skull / C1-C2 pivot
      - head: cranial center of mass / base of skull
      - jaw: temporomandibular joint (TMJ) rotation center
      - eye_L: center of left orbit
      - eye_R: center of right orbit
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    y_span = y_max - y_min
    z_span = z_max - z_min
    x_span = x_max - x_min

    # Neck base: bottom 10% Y, posterior 20% Z, centered X
    neck_pos = np.array([0.0, y_min + 0.08 * y_span, z_min + 0.30 * z_span], dtype=np.float32)

    # Head cranial center: 55% Y, posterior 40% Z, centered X
    head_pos = np.array([0.0, y_min + 0.55 * y_span, z_min + 0.40 * z_span], dtype=np.float32)

    # Jaw TMJ pivot: 28% Y, posterior 35% Z, centered X
    jaw_pos = np.array([0.0, y_min + 0.28 * y_span, z_min + 0.35 * z_span], dtype=np.float32)

    # Eye orbits: 62% Y, anterior 75% Z, lateral offset ±22% X
    eye_l_pos = np.array([x_min + 0.68 * x_span, y_min + 0.62 * y_span, z_min + 0.72 * z_span], dtype=np.float32)
    eye_r_pos = np.array([x_min + 0.32 * x_span, y_min + 0.62 * y_span, z_min + 0.72 * z_span], dtype=np.float32)

    return {
        "neck": neck_pos,
        "head": head_pos,
        "jaw": jaw_pos,
        "eye_L": eye_l_pos,
        "eye_R": eye_r_pos,
    }


def compute_linear_skinning_weights(
    vertices: np.ndarray,
    joint_centers: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Computes smooth Linear Blend Skinning (LBS) weights W in R^(N_verts x 5).
    Enforces strict partition of unity: sum_{j=1}^5 W_{i, j} = 1.0 for all vertices i.
    """
    n_verts = len(vertices)
    weights = np.zeros((n_verts, len(JOINT_NAMES)), dtype=np.float32)

    y = vertices[:, 1]
    z = vertices[:, 2]
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    y_span = max(y_max - y_min, 1e-4)
    z_span = max(z_max - z_min, 1e-4)

    y_norm = (y - y_min) / y_span
    z_norm = (z - z_min) / z_span

    # 1. Neck weights: dominant at bottom 20% Y
    neck_w = np.clip((0.25 - y_norm) / 0.20, 0.0, 1.0) ** 2

    # 2. Jaw weights: lower 35% Y and anterior 60% Z
    jaw_w = np.clip((0.38 - y_norm) / 0.30, 0.0, 1.0) * np.clip((z_norm - 0.30) / 0.50, 0.0, 1.0)
    jaw_w = jaw_w ** 1.5

    # 3. Eye weights: Gaussian radial falloff around eyeball centers
    dist_l = np.linalg.norm(vertices - joint_centers["eye_L"], axis=1)
    dist_r = np.linalg.norm(vertices - joint_centers["eye_R"], axis=1)
    eye_radius = 0.08 * y_span
    eye_l_w = np.exp(-0.5 * (dist_l / eye_radius) ** 2)
    eye_r_w = np.exp(-0.5 * (dist_r / eye_radius) ** 2)

    # 4. Head weights: cranial vault, forehead, cheeks, temples
    # Residual cranial weight
    head_w = np.maximum(0.1, 1.0 - (neck_w + jaw_w + eye_l_w + eye_r_w))

    # Assemble raw weight matrix
    weights[:, 0] = neck_w     # neck
    weights[:, 1] = head_w     # head
    weights[:, 2] = jaw_w      # jaw
    weights[:, 3] = eye_l_w    # eye_L
    weights[:, 4] = eye_r_w    # eye_R

    # Partition of unity: strictly normalize every row to sum to 1.0
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    weights = weights / row_sums

    return weights.astype(np.float32)


def export_armature_json(
    joint_centers: Dict[str, np.ndarray],
    skinning_weights: np.ndarray,
    output_path: Union[str, Path]
) -> str:
    """
    Exports armature joint hierarchy and per-vertex skinning weights to JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joints_dict = {}
    for name in JOINT_NAMES:
        pos = joint_centers[name].tolist()
        parent = JOINT_HIERARCHY[name]["parent"]
        joints_dict[name] = {
            "position": [round(p, 6) for p in pos],
            "parent": parent,
            "children": JOINT_HIERARCHY[name]["children"],
        }

    payload = {
        "version": "1.0",
        "joints_count": len(JOINT_NAMES),
        "joints": joints_dict,
        "num_vertices": int(len(skinning_weights)),
        "joint_names": JOINT_NAMES,
        # Round weights to 4 decimal places to keep file sizes lean
        "skinning_weights": np.round(skinning_weights, decimals=4).tolist(),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return str(path)


def load_armature_json(json_path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Loads joint centers and skinning weights from an exported armature JSON."""
    path = Path(json_path)
    assert path.exists(), f"Armature file not found: {path}"
    with open(path, "r") as f:
        data = json.load(f)

    joint_centers = {
        k: np.array(v["position"], dtype=np.float32)
        for k, v in data["joints"].items()
    }
    weights = np.array(data["skinning_weights"], dtype=np.float32)
    return joint_centers, weights
