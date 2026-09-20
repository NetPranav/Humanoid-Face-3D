"""
Stage 5 Parametric Facial Stylization Layer.

Enables continuous, identity-preserving aesthetic deformation of facial geometry
(e.g., jaw width, chin depth/squareness, gonial flare, cheekbone prominence)
while strictly adhering to the Unreal Engine 5 Neck Seam Contract:
- The lowest 20% of vertices (neck boundary collar) have displacements pinned
  strictly to zero (Delta v = 0).
- Scale-invariant deformation normalized by bounding-box height H.
- Bilateral sagittal symmetry (x -> -x).
- Manifold topology preservation via smooth Gaussian and Hermite falloffs.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Union, Tuple, Any
import numpy as np


@dataclass
class StylizationParameters:
    """
    Continuous deformation parameters for facial stylization.
    Sliders are nominally normalized in [-1.0, 1.0] where 0.0 represents
    the neutral reconstructed base mesh. chin_cleft is in [0.0, 1.0].
    """
    jaw_width: float = 0.0             # Lateral expansion (+)/contraction (-) of mandible
    jaw_squareness: float = 0.0        # Boxy/rectangular (+) vs tapered/V-shaped (-) jaw
    chin_depth: float = 0.0            # Anterior projection (+) vs retrognathic recession (-)
    chin_width: float = 0.0            # Lateral width of mental protuberance pad
    chin_cleft: float = 0.0            # Vertical cleft depression at mental midline [0, 1]
    gonial_flare: float = 0.0          # Lateral & angular flare of mandibular gonial angles
    cheekbone_prominence: float = 0.0  # Anterolateral projection of zygomatic arches
    brow_prominence: float = 0.0       # Anterior projection of supraorbital ridge

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def is_neutral(self) -> bool:
        """Returns True if all slider values are zero."""
        return all(abs(v) < 1e-6 for v in self.to_dict().values())

    def clamp(self) -> "StylizationParameters":
        """Clamps slider values to valid safe ranges."""
        return StylizationParameters(
            jaw_width=float(np.clip(self.jaw_width, -1.0, 1.0)),
            jaw_squareness=float(np.clip(self.jaw_squareness, -1.0, 1.0)),
            chin_depth=float(np.clip(self.chin_depth, -1.0, 1.0)),
            chin_width=float(np.clip(self.chin_width, -1.0, 1.0)),
            chin_cleft=float(np.clip(self.chin_cleft, 0.0, 1.0)),
            gonial_flare=float(np.clip(self.gonial_flare, -1.0, 1.0)),
            cheekbone_prominence=float(np.clip(self.cheekbone_prominence, -1.0, 1.0)),
            brow_prominence=float(np.clip(self.brow_prominence, -1.0, 1.0)),
        )


STYLIZATION_PRESETS: Dict[str, StylizationParameters] = {
    "neutral": StylizationParameters(),
    "chiseled": StylizationParameters(
        jaw_width=0.55,
        jaw_squareness=0.65,
        chin_depth=0.45,
        chin_width=0.40,
        chin_cleft=0.35,
        gonial_flare=0.75,
        cheekbone_prominence=0.60,
        brow_prominence=0.40,
    ),
    "gigachad": StylizationParameters(
        jaw_width=0.80,
        jaw_squareness=0.85,
        chin_depth=0.70,
        chin_width=0.60,
        chin_cleft=0.50,
        gonial_flare=0.95,
        cheekbone_prominence=0.75,
        brow_prominence=0.55,
    ),
    "heroic": StylizationParameters(
        jaw_width=0.40,
        jaw_squareness=0.50,
        chin_depth=0.35,
        chin_width=0.30,
        chin_cleft=0.0,
        gonial_flare=0.50,
        cheekbone_prominence=0.40,
        brow_prominence=0.30,
    ),
    "soft_oval": StylizationParameters(
        jaw_width=-0.30,
        jaw_squareness=-0.40,
        chin_depth=-0.10,
        chin_width=-0.20,
        chin_cleft=0.0,
        gonial_flare=-0.30,
        cheekbone_prominence=0.10,
        brow_prominence=-0.10,
    ),
    "round": StylizationParameters(
        jaw_width=0.20,
        jaw_squareness=-0.50,
        chin_depth=-0.25,
        chin_width=-0.20,
        chin_cleft=0.0,
        gonial_flare=-0.40,
        cheekbone_prominence=-0.20,
        brow_prominence=-0.20,
    ),
}


def resolve_stylization_params(
    params: Optional[Union[StylizationParameters, Dict[str, float], str]] = None,
    **kwargs
) -> StylizationParameters:
    """
    Resolves stylization configuration from presets, dicts, instances, or kwargs.
    """
    base = StylizationParameters()

    if isinstance(params, str):
        key = params.lower().strip()
        if key not in STYLIZATION_PRESETS:
            raise ValueError(
                f"Unknown stylization preset '{params}'. Available presets: {list(STYLIZATION_PRESETS.keys())}"
            )
        base = STYLIZATION_PRESETS[key]
    elif isinstance(params, StylizationParameters):
        base = params
    elif isinstance(params, dict):
        base_dict = base.to_dict()
        base_dict.update(params)
        base = StylizationParameters(**base_dict)
    elif params is not None:
        raise TypeError(f"Unsupported stylization params type: {type(params)}")

    if kwargs:
        merged = base.to_dict()
        merged.update(kwargs)
        base = StylizationParameters(**merged)

    return base.clamp()


def compute_hermite_smoothstep(val: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Computes C1 Hermite smoothstep interpolation."""
    t = np.clip((val - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class FaceStylizer:
    """
    Parametric deformation engine for humanoid facial geometry.
    Operates on 3D meshes in standard coordinate space:
      X: Right to Left (x > 0 is subject's left, x = 0 is sagittal midline)
      Y: Superior to Inferior (positive is upwards, y_min is neck base)
      Z: Anterior to Posterior (positive is forwards/anterior)
    """
    def __init__(self, neck_collar_threshold: float = 0.20, collar_transition: float = 0.12):
        """
        Args:
            neck_collar_threshold: Fraction of bounding box height from bottom
                                   where displacement is pinned strictly to 0.0.
            collar_transition: Smoothstep transition band above the pinned collar.
        """
        self.neck_collar_threshold = float(neck_collar_threshold)
        self.collar_transition = float(collar_transition)

    def compute_normalized_coordinates(
        self,
        vertices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """
        Computes normalized bounding coordinates and characteristic scales.
        Returns:
            (x_norm, y_norm, z_norm, x_center, height, width, depth)
        """
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        z_min, z_max = float(np.min(z)), float(np.max(z))

        height = max(y_max - y_min, 1e-4)
        width = max(x_max - x_min, 1e-4)
        depth = max(z_max - z_min, 1e-4)

        x_center = (x_min + x_max) * 0.5
        x_norm = (x - x_center) / (width * 0.5)
        y_norm = (y - y_min) / height
        z_norm = (z - z_min) / depth

        return x_norm, y_norm, z_norm, x_center, height, width, depth

    def compute_pinning_mask(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Computes the neck seam collar pinning mask.
        Strictly zero for y_norm <= neck_collar_threshold, then smoothly
        blends to 1.0 using Hermite interpolation.
        """
        edge0 = self.neck_collar_threshold
        edge1 = self.neck_collar_threshold + self.collar_transition
        mask = compute_hermite_smoothstep(y_norm, edge0, edge1)
        # Enforce exact bitwise zero below threshold
        mask[y_norm <= edge0] = 0.0
        return mask

    def apply_stylization(
        self,
        vertices: np.ndarray,
        params: Optional[Union[StylizationParameters, Dict[str, float], str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deforms input vertices according to stylization parameters.

        Args:
            vertices: (N, 3) float array of base neutral vertices.
            params: StylizationParameters, dict, or preset name.
            **kwargs: Individual parameter overrides (e.g. jaw_width=0.5).

        Returns:
            stylized_vertices: (N, 3) float array of deformed vertices.
            deltas: (N, 3) float array of displacement vectors (stylized - input).
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected vertices shape (N, 3), got {vertices.shape}")
        if np.isnan(vertices).any() or np.isinf(vertices).any():
            raise ValueError("Input vertices contain NaN or Infinite values.")

        p = resolve_stylization_params(params, **kwargs)
        if p.is_neutral():
            return vertices.copy(), np.zeros_like(vertices)

        n_verts = len(vertices)
        deltas = np.zeros((n_verts, 3), dtype=np.float32)

        x_norm, y_norm, z_norm, x_center, H, W, D = self.compute_normalized_coordinates(vertices)
        pinning_mask = self.compute_pinning_mask(y_norm)

        # Lateral sign and magnitude: symmetric around midline
        abs_x = np.abs(x_norm)
        sign_x = np.sign(x_norm)

        # ── 1. JAW WIDTH ──────────────────────────────────────────────────
        # Expands or contracts lower mandible laterally (+/- X)
        if abs(p.jaw_width) > 1e-5:
            # Mandible zone: y_norm in [0.20, 0.44], anterior/lateral z_norm in [0.25, 0.85]
            y_weight = np.exp(-((y_norm - 0.30) / 0.10) ** 2)
            z_weight = np.clip((z_norm - 0.20) / 0.40, 0.0, 1.0)
            x_weight = np.clip(abs_x / 0.70, 0.0, 1.0)
            w_jaw = y_weight * z_weight * x_weight * pinning_mask
            disp_x = sign_x * (p.jaw_width * 0.035 * H) * w_jaw
            deltas[:, 0] += disp_x

        # ── 2. JAW SQUARENESS ─────────────────────────────────────────────
        # Boxy gonial curve vs tapered V-line
        if abs(p.jaw_squareness) > 1e-5:
            # Squareness widens lateral chin corners and pushes anterior lower jaw slightly forward
            y_weight = np.exp(-((y_norm - 0.27) / 0.08) ** 2)
            # Lateral corner of chin/jawline (|x_norm| around 0.25 to 0.65)
            x_corner_weight = np.exp(-((abs_x - 0.45) / 0.20) ** 2)
            z_weight = np.clip((z_norm - 0.35) / 0.45, 0.0, 1.0)
            w_sq = y_weight * x_corner_weight * z_weight * pinning_mask

            deltas[:, 0] += sign_x * (p.jaw_squareness * 0.025 * H) * w_sq
            deltas[:, 2] += (p.jaw_squareness * 0.012 * H) * w_sq

        # ── 3. CHIN DEPTH ─────────────────────────────────────────────────
        # Anterior (+Z) projection of mental protuberance
        if abs(p.chin_depth) > 1e-5:
            y_weight = np.exp(-((y_norm - 0.25) / 0.06) ** 2)
            x_weight = np.exp(-(abs_x / 0.25) ** 2)
            z_weight = np.clip((z_norm - 0.50) / 0.40, 0.0, 1.0)
            w_chin = y_weight * x_weight * z_weight * pinning_mask

            deltas[:, 2] += (p.chin_depth * 0.035 * H) * w_chin

        # ── 4. CHIN WIDTH ─────────────────────────────────────────────────
        # Lateral width of chin pad (squared vs pointed chin)
        if abs(p.chin_width) > 1e-5:
            y_weight = np.exp(-((y_norm - 0.25) / 0.06) ** 2)
            x_flank_weight = np.exp(-((abs_x - 0.20) / 0.15) ** 2)
            z_weight = np.clip((z_norm - 0.60) / 0.35, 0.0, 1.0)
            w_cw = y_weight * x_flank_weight * z_weight * pinning_mask

            deltas[:, 0] += sign_x * (p.chin_width * 0.022 * H) * w_cw

        # ── 5. CHIN CLEFT ─────────────────────────────────────────────────
        # Midline vertical groove flanked by subtle bilateral prominences
        if p.chin_cleft > 1e-5:
            y_weight = np.exp(-((y_norm - 0.25) / 0.05) ** 2)
            z_weight = np.clip((z_norm - 0.65) / 0.30, 0.0, 1.0)
            # Central depression
            midline_weight = np.exp(-(abs_x / 0.06) ** 2)
            w_cleft = y_weight * z_weight * midline_weight * pinning_mask
            deltas[:, 2] -= (p.chin_cleft * 0.016 * H) * w_cleft

            # Complementary lateral tubercles
            bilateral_weight = np.exp(-((abs_x - 0.12) / 0.07) ** 2)
            w_tub = y_weight * z_weight * bilateral_weight * pinning_mask
            deltas[:, 2] += (p.chin_cleft * 0.008 * H) * w_tub

        # ── 6. GONIAL FLARE ───────────────────────────────────────────────
        # Sharp lateral/posterior expansion at the mandibular angle (jaw corners)
        if abs(p.gonial_flare) > 1e-5:
            y_weight = np.exp(-((y_norm - 0.29) / 0.07) ** 2)
            x_weight = np.exp(-((abs_x - 0.72) / 0.18) ** 2)
            # Posterior-to-mid anterior region
            z_weight = np.exp(-((z_norm - 0.40) / 0.20) ** 2)
            w_gf = y_weight * x_weight * z_weight * pinning_mask

            deltas[:, 0] += sign_x * (p.gonial_flare * 0.040 * H) * w_gf
            deltas[:, 1] -= (p.gonial_flare * 0.008 * H) * w_gf  # Crisp downward hook
            deltas[:, 2] -= (p.gonial_flare * 0.006 * H) * w_gf  # Posterior tuck

        # ── 7. CHEEKBONE PROMINENCE ───────────────────────────────────────
        # Anterolateral projection of the zygomatic arches
        if abs(p.cheekbone_prominence) > 1e-5:
            y_weight = np.exp(-((y_norm - 0.58) / 0.08) ** 2)
            x_weight = np.exp(-((abs_x - 0.62) / 0.18) ** 2)
            z_weight = np.clip((z_norm - 0.40) / 0.45, 0.0, 1.0)
            w_cb = y_weight * x_weight * z_weight * pinning_mask

            deltas[:, 0] += sign_x * (p.cheekbone_prominence * 0.028 * H) * w_cb
            deltas[:, 2] += (p.cheekbone_prominence * 0.024 * H) * w_cb

        # ── 8. BROW PROMINENCE ────────────────────────────────────────────
        # Anterior shelf projection of the supraorbital arches
        if abs(p.brow_prominence) > 1e-5:
            y_weight = np.exp(-((y_norm - 0.74) / 0.06) ** 2)
            x_weight = np.clip(1.0 - (abs_x / 0.65), 0.0, 1.0)
            z_weight = np.clip((z_norm - 0.55) / 0.40, 0.0, 1.0)
            w_bp = y_weight * x_weight * z_weight * pinning_mask

            deltas[:, 2] += (p.brow_prominence * 0.022 * H) * w_bp
            deltas[:, 1] -= (p.brow_prominence * 0.005 * H) * w_bp

        # Guarantee strict neck boundary pinning
        deltas[pinning_mask == 0.0] = 0.0

        stylized_vertices = (vertices + deltas).astype(vertices.dtype)
        return stylized_vertices, deltas
