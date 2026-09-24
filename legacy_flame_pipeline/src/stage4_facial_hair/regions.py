"""
Anatomical Facial Hair Region Segmentation & Style Presets.

Defines continuous parametric hair zone weighting across 3D head geometry:
  - mustache: philtrum & upper lip zone
  - chin: mental protuberance & soul patch
  - jawline: mandibular body & gonial flanks
  - sideburns: preauricular cheek zones
  - eyebrows: supraorbital arches

Strictly enforces the Unreal Engine 5 Neck Seam Contract:
  - Vertices in the lowest 20% collar region have strictly zero hair weight (weight = 0.0).
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np


@dataclass
class FacialHairConfig:
    """
    Parametric configuration for facial hair and stubble generation.
    Density sliders are normalized in [0.0, 1.0].
    """
    # Regional density weights
    mustache_density: float = 0.0
    chin_density: float = 0.0
    jawline_density: float = 0.0
    sideburns_density: float = 0.0
    eyebrows_density: float = 0.0

    # Stubble micro-displacement parameters
    stubble_length_mm: float = 0.6      # Hair stubble relief height in mm [0.1, 2.0]
    stubble_density: float = 0.8        # Follicle frequency factor [0.1, 1.0]
    generate_stubble: bool = True

    # Static hair card polygonal mesh parameters
    generate_cards: bool = False
    card_density: int = 120             # Approximate number of card quads to place
    card_length_mm: float = 2.5         # Extrusion length of cards in mm [0.5, 6.0]
    card_width_mm: float = 0.5          # Width of each card ribbon in mm [0.1, 1.2]
    card_curvature: float = 0.3         # Outward/downward bend factor [0.0, 1.0]

    # Material & Shader parameters
    card_segments: int = 2              # Segmented ribbon resolution (2 segments = 6 verts, 4 tris)
    alpha_softness: float = 0.5         # Texture alpha feathering

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_clean_shaven(self) -> bool:
        """Returns True if all hair densities are zero or near zero."""
        weights = [
            self.mustache_density,
            self.chin_density,
            self.jawline_density,
            self.sideburns_density,
            self.eyebrows_density,
        ]
        return all(w < 1e-4 for w in weights)

    def clamp(self) -> "FacialHairConfig":
        """Clamps all parameters to safe, physically realistic ranges."""
        return FacialHairConfig(
            mustache_density=float(np.clip(self.mustache_density, 0.0, 1.0)),
            chin_density=float(np.clip(self.chin_density, 0.0, 1.0)),
            jawline_density=float(np.clip(self.jawline_density, 0.0, 1.0)),
            sideburns_density=float(np.clip(self.sideburns_density, 0.0, 1.0)),
            eyebrows_density=float(np.clip(self.eyebrows_density, 0.0, 1.0)),
            stubble_length_mm=float(np.clip(self.stubble_length_mm, 0.05, 3.0)),
            stubble_density=float(np.clip(self.stubble_density, 0.1, 1.0)),
            generate_stubble=bool(self.generate_stubble),
            generate_cards=bool(self.generate_cards),
            card_density=int(np.clip(self.card_density, 10, 1000)),
            card_length_mm=float(np.clip(self.card_length_mm, 0.5, 6.0)),
            card_width_mm=float(np.clip(self.card_width_mm, 0.1, 1.5)),
            card_curvature=float(np.clip(self.card_curvature, 0.0, 1.0)),
            card_segments=int(np.clip(self.card_segments, 1, 4)),
            alpha_softness=float(np.clip(self.alpha_softness, 0.1, 1.0)),
        )


FACIAL_HAIR_PRESETS: Dict[str, FacialHairConfig] = {
    "clean_shaven": FacialHairConfig(
        mustache_density=0.0,
        chin_density=0.0,
        jawline_density=0.0,
        sideburns_density=0.0,
        eyebrows_density=0.0,
        generate_stubble=False,
        generate_cards=False,
    ),
    "stubble": FacialHairConfig(
        mustache_density=0.75,
        chin_density=0.85,
        jawline_density=0.80,
        sideburns_density=0.70,
        eyebrows_density=0.0,
        stubble_length_mm=0.55,
        stubble_density=0.85,
        generate_stubble=True,
        generate_cards=False,
    ),
    "heavy_stubble": FacialHairConfig(
        mustache_density=0.90,
        chin_density=0.95,
        jawline_density=0.90,
        sideburns_density=0.80,
        eyebrows_density=0.0,
        stubble_length_mm=0.90,
        stubble_density=0.95,
        generate_stubble=True,
        generate_cards=True,
        card_density=80,
        card_length_mm=2.2,
        card_width_mm=0.40,
    ),
    "mustache": FacialHairConfig(
        mustache_density=1.0,
        chin_density=0.0,
        jawline_density=0.0,
        sideburns_density=0.0,
        eyebrows_density=0.0,
        stubble_length_mm=0.8,
        generate_stubble=True,
        generate_cards=True,
        card_density=60,
        card_length_mm=3.2,
        card_width_mm=0.55,
    ),
    "goatee": FacialHairConfig(
        mustache_density=0.85,
        chin_density=1.0,
        jawline_density=0.15,
        sideburns_density=0.0,
        eyebrows_density=0.0,
        stubble_length_mm=0.85,
        generate_stubble=True,
        generate_cards=True,
        card_density=100,
        card_length_mm=3.5,
        card_width_mm=0.60,
    ),
    "full_beard": FacialHairConfig(
        mustache_density=0.90,
        chin_density=1.0,
        jawline_density=0.90,
        sideburns_density=0.85,
        eyebrows_density=0.0,
        stubble_length_mm=1.1,
        generate_stubble=True,
        generate_cards=True,
        card_density=180,
        card_length_mm=4.5,
        card_width_mm=0.70,
    ),
    "eyebrows_only": FacialHairConfig(
        mustache_density=0.0,
        chin_density=0.0,
        jawline_density=0.0,
        sideburns_density=0.0,
        eyebrows_density=1.0,
        generate_stubble=False,
        generate_cards=True,
        card_density=50,
        card_length_mm=2.5,
        card_width_mm=0.45,
    ),
}


def resolve_hair_config(
    config: Optional[Union[FacialHairConfig, Dict[str, Any], str]] = None,
    **kwargs
) -> FacialHairConfig:
    """
    Resolves a FacialHairConfig from a preset string, dict, or instance.
    """
    base = FacialHairConfig()

    if isinstance(config, str):
        key = config.lower().strip()
        if key not in FACIAL_HAIR_PRESETS:
            raise ValueError(
                f"Unknown facial hair preset '{config}'. Available presets: {list(FACIAL_HAIR_PRESETS.keys())}"
            )
        base = FACIAL_HAIR_PRESETS[key]
    elif isinstance(config, FacialHairConfig):
        base = config
    elif isinstance(config, dict):
        base_dict = base.to_dict()
        base_dict.update(config)
        base = FacialHairConfig(**base_dict)
    elif config is not None:
        raise TypeError(f"Unsupported config type: {type(config)}")

    if kwargs:
        merged = base.to_dict()
        merged.update(kwargs)
        base = FacialHairConfig(**merged)

    return base.clamp()


def compute_normalized_coordinates(
    vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Computes normalized coordinates relative to bounding box:
      x_norm in [-1, 1] (0 is sagittal midline)
      y_norm in [0, 1] (0 is neck base, 1 is skull crown)
      z_norm in [0, 1] (0 is posterior skull, 1 is anterior nose/chin)
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


def compute_collar_pinning_mask(y_norm: np.ndarray, threshold: float = 0.20, transition: float = 0.10) -> np.ndarray:
    """
    Computes smooth collar pinning mask enforcing the Neck Seam Contract:
    Strictly 0.0 for y_norm <= threshold, smoothstep blending to 1.0 above.
    """
    t = np.clip((y_norm - threshold) / max(transition, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    smooth[y_norm <= threshold] = 0.0
    return smooth


class FacialHairRegionSegmenter:
    """
    Segments 3D head vertices and surface areas into anatomical hair growth zones.
    """
    def __init__(self, neck_collar_threshold: float = 0.20):
        self.neck_collar_threshold = float(neck_collar_threshold)

    def extract_region_weights(
        self,
        vertices: np.ndarray,
        config: Optional[Union[FacialHairConfig, Dict[str, Any], str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Computes per-vertex continuous weights [0.0, 1.0] for all hair regions.

        Returns dict containing:
          - 'mustache': (N,) float array
          - 'chin': (N,) float array
          - 'jawline': (N,) float array
          - 'sideburns': (N,) float array
          - 'eyebrows': (N,) float array
          - 'composite': (N,) total weighted hair density mask
          - 'growth_directions': (N, 3) normalized tangent growth vectors
        """
        cfg = resolve_hair_config(config)
        n_verts = len(vertices)

        x_norm, y_norm, z_norm, _, H, W, D = compute_normalized_coordinates(vertices)
        abs_x = np.abs(x_norm)
        sign_x = np.sign(x_norm)
        sign_x[sign_x == 0.0] = 1.0

        collar_mask = compute_collar_pinning_mask(y_norm, threshold=self.neck_collar_threshold)

        # ── 1. Exclusion Zones (Lips & Eyes) ──────────────────────────────────
        # Inner lips exclusion: y_norm ~ 0.33, center x, anterior z
        lip_y = np.exp(-((y_norm - 0.33) / 0.04) ** 2)
        lip_x = np.exp(-(abs_x / 0.20) ** 2)
        lip_z = np.clip((z_norm - 0.70) / 0.25, 0.0, 1.0)
        lip_mask = np.clip(lip_y * lip_x * lip_z, 0.0, 1.0)

        # Eye orbit exclusion: y_norm ~ 0.60, lateral x ~ 0.32, anterior z
        eye_y = np.exp(-((y_norm - 0.60) / 0.05) ** 2)
        eye_x = np.exp(-((abs_x - 0.32) / 0.15) ** 2)
        eye_z = np.clip((z_norm - 0.65) / 0.25, 0.0, 1.0)
        eye_mask = np.clip(eye_y * eye_x * eye_z, 0.0, 1.0)

        # ── 2. Mustache (Philtrum & Upper Lip Zone) ──────────────────────────
        # y_norm ~ [0.36, 0.44], centered lateral |x| <= 0.35, anterior z >= 0.65
        mustache_y = np.exp(-((y_norm - 0.40) / 0.04) ** 2)
        mustache_x = np.exp(-(abs_x / 0.24) ** 2)
        mustache_z = np.clip((z_norm - 0.65) / 0.25, 0.0, 1.0)
        w_mustache = mustache_y * mustache_x * mustache_z * (1.0 - lip_mask * 1.5)
        w_mustache = np.clip(w_mustache, 0.0, 1.0) * collar_mask

        # ── 3. Chin / Mental Protuberance ─────────────────────────────────────
        # y_norm ~ [0.22, 0.31], centered lateral |x| <= 0.30, anterior z >= 0.55
        chin_y = np.exp(-((y_norm - 0.26) / 0.05) ** 2)
        chin_x = np.exp(-(abs_x / 0.22) ** 2)
        chin_z = np.clip((z_norm - 0.55) / 0.35, 0.0, 1.0)
        w_chin = chin_y * chin_x * chin_z * (1.0 - lip_mask)
        w_chin = np.clip(w_chin, 0.0, 1.0) * collar_mask

        # ── 4. Jawline / Mandible Flanks ──────────────────────────────────────
        # y_norm ~ [0.24, 0.38], lateral |x| in [0.25, 0.75], mid-to-anterior z >= 0.35
        jaw_y = np.exp(-((y_norm - 0.29) / 0.06) ** 2)
        jaw_x = np.exp(-((abs_x - 0.52) / 0.20) ** 2)
        jaw_z = np.clip((z_norm - 0.35) / 0.45, 0.0, 1.0)
        w_jaw = jaw_y * jaw_x * jaw_z
        w_jaw = np.clip(w_jaw, 0.0, 1.0) * collar_mask

        # ── 5. Sideburns (Preauricular Cheek) ─────────────────────────────────
        # y_norm ~ [0.42, 0.58], lateral |x| in [0.65, 0.90], z in [0.25, 0.60]
        sb_y = np.exp(-((y_norm - 0.49) / 0.08) ** 2)
        sb_x = np.exp(-((abs_x - 0.76) / 0.14) ** 2)
        sb_z = np.exp(-((z_norm - 0.42) / 0.18) ** 2)
        w_sb = sb_y * sb_x * sb_z
        w_sb = np.clip(w_sb, 0.0, 1.0) * collar_mask

        # ── 6. Eyebrows (Supraorbital Ridge) ──────────────────────────────────
        # y_norm ~ [0.62, 0.68], lateral |x| in [0.12, 0.55], anterior z >= 0.65
        brow_y = np.exp(-((y_norm - 0.64) / 0.035) ** 2)
        brow_x = np.exp(-((abs_x - 0.33) / 0.18) ** 2)
        brow_z = np.clip((z_norm - 0.65) / 0.25, 0.0, 1.0)
        w_brow = brow_y * brow_x * brow_z * (1.0 - eye_mask * 0.8)
        w_brow = np.clip(w_brow, 0.0, 1.0) * collar_mask

        # ── 7. Composite Weighted Mask ────────────────────────────────────────
        composite = (
            w_mustache * cfg.mustache_density +
            w_chin * cfg.chin_density +
            w_jaw * cfg.jawline_density +
            w_sb * cfg.sideburns_density +
            w_brow * cfg.eyebrows_density
        )
        composite = np.clip(composite, 0.0, 1.0)
        # Enforce exact bitwise zero on neck collar
        composite[y_norm <= self.neck_collar_threshold] = 0.0

        # ── 8. Growth Direction Field ─────────────────────────────────────────
        # Tangent growth vectors for hair cards and directional stubble tilt
        growth = np.zeros((n_verts, 3), dtype=np.float32)

        # Mustache: flows downward and outward from sagittal midline
        growth[:, 0] += sign_x * 0.35 * w_mustache
        growth[:, 1] -= 0.85 * w_mustache
        growth[:, 2] += 0.20 * w_mustache

        # Chin: flows downward with slight forward protrusion
        growth[:, 1] -= 0.90 * w_chin
        growth[:, 2] += 0.25 * w_chin

        # Jawline: flows downward and backwards along the mandibular contour
        growth[:, 0] += sign_x * 0.25 * w_jaw
        growth[:, 1] -= 0.80 * w_jaw
        growth[:, 2] -= 0.30 * w_jaw

        # Sideburns: flows downward
        growth[:, 1] -= 0.95 * w_sb

        # Eyebrows: flows laterally outwards from midline along supraorbital arch
        growth[:, 0] += sign_x * 0.90 * w_brow
        growth[:, 1] += 0.20 * w_brow
        growth[:, 2] += 0.10 * w_brow

        # Default downward vector for any non-zero region
        norm_growth = np.linalg.norm(growth, axis=1, keepdims=True)
        growth = np.where(norm_growth > 1e-4, growth / np.maximum(norm_growth, 1e-6), np.array([0.0, -1.0, 0.0], dtype=np.float32))

        return {
            "mustache": w_mustache.astype(np.float32),
            "chin": w_chin.astype(np.float32),
            "jawline": w_jaw.astype(np.float32),
            "sideburns": w_sb.astype(np.float32),
            "eyebrows": w_brow.astype(np.float32),
            "composite": composite.astype(np.float32),
            "growth_directions": growth.astype(np.float32),
            "collar_mask": collar_mask.astype(np.float32),
        }
