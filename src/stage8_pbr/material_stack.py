"""
Stage 8: PBR Material Stack Derivation.

Generates physically based rendering texture maps from existing geometry,
displacement, and albedo data. All operations are procedural (pure NumPy) —
no neural networks, no GPU training required.

Output maps:
  - Roughness:     Anatomical-zone-based with displacement-correlated micro-variation.
  - Cavity / AO:   Laplacian of displacement field for pore-coupled ambient occlusion.
  - SSS Thickness: Geometric ray-march through the 3D mesh for subsurface scattering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Anatomical Zone Masks (FLAME UV space)
# ---------------------------------------------------------------------------

def _build_anatomical_zone_masks(
    uv_coords: np.ndarray,
    uv_faces: np.ndarray,
    flame_faces: np.ndarray,
    vertices: np.ndarray,
    resolution: int,
) -> Dict[str, np.ndarray]:
    """
    Build per-pixel anatomical zone masks in UV space by rasterizing
    vertex positions and classifying by normalised 3D coordinates.

    Returns dict of (resolution, resolution) float32 masks in [0, 1].
    """
    # Normalise vertex positions to [0, 1] range
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_range = v_max - v_min
    v_range[v_range == 0] = 1.0
    v_norm = (vertices - v_min) / v_range  # (V, 3) in [0, 1]

    # Rasterize normalised positions into UV space
    pos_map = np.zeros((resolution, resolution, 3), dtype=np.float32)
    valid_map = np.zeros((resolution, resolution), dtype=np.float32)

    uv_px = uv_coords.copy().astype(np.float64)
    uv_px[:, 0] = np.clip(uv_px[:, 0] * (resolution - 1), 0, resolution - 1)
    uv_px[:, 1] = np.clip((1.0 - uv_px[:, 1]) * (resolution - 1), 0, resolution - 1)

    n_faces = min(len(uv_faces), len(flame_faces))
    for i in range(n_faces):
        tri_uv = uv_faces[i]
        tri_geom = flame_faces[i]

        p0 = uv_px[tri_uv[0]]
        p1 = uv_px[tri_uv[1]]
        p2 = uv_px[tri_uv[2]]

        xmin = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        xmax = min(resolution - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        ymin = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        ymax = min(resolution - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

        if xmax <= xmin or ymax <= ymin:
            continue

        area = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(area) < 1e-6:
            continue

        xs, ys = np.meshgrid(
            np.arange(xmin, xmax + 1, dtype=np.float64),
            np.arange(ymin, ymax + 1, dtype=np.float64)
        )
        b0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / area
        b1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / area
        b2 = 1.0 - b0 - b1

        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not np.any(inside):
            continue

        yc = ys[inside].astype(np.int32)
        xc = xs[inside].astype(np.int32)
        b0i, b1i, b2i = b0[inside], b1[inside], b2[inside]

        interp = (
            b0i[:, None] * v_norm[tri_geom[0]] +
            b1i[:, None] * v_norm[tri_geom[1]] +
            b2i[:, None] * v_norm[tri_geom[2]]
        )
        pos_map[yc, xc] = interp.astype(np.float32)
        valid_map[yc, xc] = 1.0

    x_n = pos_map[:, :, 0]  # lateral
    y_n = pos_map[:, :, 1]  # vertical (0=bottom, 1=top)
    z_n = pos_map[:, :, 2]  # depth (0=back, 1=front)

    # Build zone masks using normalised coordinates
    masks = {}

    # T-zone: forehead + nose bridge + chin tip (oily skin)
    forehead = (y_n > 0.62) & (np.abs(x_n - 0.5) < 0.20) & (z_n > 0.55) & (valid_map > 0)
    nose = (y_n > 0.38) & (y_n < 0.58) & (np.abs(x_n - 0.5) < 0.08) & (z_n > 0.70) & (valid_map > 0)
    chin_tip = (y_n > 0.22) & (y_n < 0.32) & (np.abs(x_n - 0.5) < 0.10) & (z_n > 0.60) & (valid_map > 0)
    masks['t_zone'] = (forehead | nose | chin_tip).astype(np.float32)

    # Cheeks (drier skin)
    masks['cheeks'] = (
        (y_n > 0.35) & (y_n < 0.58) &
        (np.abs(x_n - 0.5) > 0.10) & (np.abs(x_n - 0.5) < 0.40) &
        (z_n > 0.35) & (valid_map > 0)
    ).astype(np.float32)

    # Lips (glossy)
    masks['lips'] = (
        (y_n > 0.32) & (y_n < 0.42) &
        (np.abs(x_n - 0.5) < 0.12) &
        (z_n > 0.65) & (valid_map > 0)
    ).astype(np.float32)

    # Periorbital (thin skin around eyes)
    masks['periorbital'] = (
        (y_n > 0.52) & (y_n < 0.65) &
        (np.abs(x_n - 0.5) > 0.08) & (np.abs(x_n - 0.5) < 0.25) &
        (z_n > 0.50) & (valid_map > 0)
    ).astype(np.float32)

    # Ears (thin, high SSS)
    masks['ears'] = (
        (np.abs(x_n - 0.5) > 0.40) &
        (y_n > 0.40) & (y_n < 0.65) &
        (valid_map > 0)
    ).astype(np.float32)

    # Neck collar (bottom 20% — pinned, no texture modification)
    masks['neck_collar'] = (y_n < 0.20).astype(np.float32) * valid_map

    # Full face valid mask
    masks['valid'] = valid_map

    return masks


# ---------------------------------------------------------------------------
# 8A: Roughness Map Generator
# ---------------------------------------------------------------------------

class RoughnessMapGenerator:
    """
    Generates per-texel roughness from anatomical zone classification
    with displacement-correlated micro-variation.

    Roughness values (GGX metallic roughness model):
    - T-zone (forehead, nose, chin): 0.28–0.38 (oily, smooth)
    - Cheeks: 0.50–0.65 (drier)
    - Lips: 0.18–0.25 (glossy)
    - Periorbital: 0.42–0.52 (thin, delicate)
    - Default skin: 0.45
    """

    def __init__(
        self,
        r_t_zone: float = 0.33,
        r_cheeks: float = 0.57,
        r_lips: float = 0.22,
        r_periorbital: float = 0.47,
        r_default: float = 0.45,
        displacement_variation: float = 0.08,
    ):
        self.r_t_zone = r_t_zone
        self.r_cheeks = r_cheeks
        self.r_lips = r_lips
        self.r_periorbital = r_periorbital
        self.r_default = r_default
        self.disp_var = displacement_variation

    def generate(
        self,
        zone_masks: Dict[str, np.ndarray],
        displacement_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate roughness map.

        Parameters
        ----------
        zone_masks : dict of (H, W) float32 masks from _build_anatomical_zone_masks
        displacement_map : (H, W) float32 displacement in mm (optional)

        Returns
        -------
        roughness : (H, W) float32 in [0, 1]
        """
        valid = zone_masks.get('valid', np.ones_like(zone_masks.get('t_zone', np.zeros((1, 1)))))
        h, w = valid.shape

        roughness = np.full((h, w), self.r_default, dtype=np.float32)

        # Apply zone-specific values (with soft blending via mask weights)
        roughness = roughness * (1 - zone_masks.get('t_zone', np.zeros((h, w)))) + \
                    self.r_t_zone * zone_masks.get('t_zone', np.zeros((h, w)))

        roughness = roughness * (1 - zone_masks.get('cheeks', np.zeros((h, w)))) + \
                    self.r_cheeks * zone_masks.get('cheeks', np.zeros((h, w)))

        roughness = roughness * (1 - zone_masks.get('lips', np.zeros((h, w)))) + \
                    self.r_lips * zone_masks.get('lips', np.zeros((h, w)))

        roughness = roughness * (1 - zone_masks.get('periorbital', np.zeros((h, w)))) + \
                    self.r_periorbital * zone_masks.get('periorbital', np.zeros((h, w)))

        # Add displacement-correlated micro-variation
        if displacement_map is not None:
            disp_resized = displacement_map
            if disp_resized.shape != (h, w):
                disp_resized = cv2.resize(displacement_map, (w, h), interpolation=cv2.INTER_LINEAR)

            # Compute Laplacian of displacement (pore curvature)
            laplacian = cv2.Laplacian(disp_resized.astype(np.float32), cv2.CV_32F, ksize=3)

            # Normalize to [-1, 1]
            lap_max = np.abs(laplacian).max()
            if lap_max > 0:
                laplacian_norm = laplacian / lap_max
            else:
                laplacian_norm = laplacian

            roughness += self.disp_var * laplacian_norm

        # Mask to valid region and clamp
        roughness = np.clip(roughness, 0.0, 1.0) * valid

        return roughness


# ---------------------------------------------------------------------------
# 8B: Cavity / Ambient Occlusion Map Generator
# ---------------------------------------------------------------------------

class CavityMapGenerator:
    """
    Computes ambient occlusion / cavity darkening from displacement maps.

    Cavity(u,v) = clamp(0.5 + k * ∇²D(u,v), 0, 1)

    Where k controls the strength and ∇²D is the discrete Laplacian of the
    displacement field. Pore pits get darkened, wrinkle valleys get subtle
    ambient occlusion.
    """

    def __init__(self, strength: float = 0.35, blur_sigma: float = 1.0, mode: str = "curvature"):
        self.strength = strength
        self.blur_sigma = blur_sigma
        self.mode = mode

    def generate(
        self,
        displacement_map: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate cavity/AO map from displacement.

        Parameters
        ----------
        displacement_map : (H, W) float32
        valid_mask : (H, W) float32 or None

        Returns
        -------
        cavity : (H, W) float32 in [0, 1]
            mode='film_ao'   : 1.0 = flat skin, <1.0 = pore/crevasse shadow darkening
            mode='curvature' : 0.5 = flat, <0.5 = concave, >0.5 = convex
        """
        h, w = displacement_map.shape[:2]

        # Optionally smooth displacement before computing Laplacian
        if self.blur_sigma > 0:
            disp_smooth = cv2.GaussianBlur(
                displacement_map.astype(np.float32),
                (0, 0), self.blur_sigma
            )
        else:
            disp_smooth = displacement_map.astype(np.float32)

        # Discrete Laplacian (second-order derivative, measures curvature)
        laplacian = cv2.Laplacian(disp_smooth, cv2.CV_32F, ksize=5)

        if self.mode == "film_ao":
            # Film-grade pore-depth light trapping:
            # Cavity(u,v) = clip(1.0 - strength * max(0, ∇²D), 0.0, 1.0)
            concavity = np.maximum(0.0, laplacian)
            cavity = np.clip(1.0 - self.strength * concavity, 0.0, 1.0)
            if valid_mask is not None:
                cavity = cavity * valid_mask + 1.0 * (1.0 - valid_mask)
        else:
            # Centered curvature mode: depressions (∇²D > 0) darken below 0.5
            cavity = 0.5 - self.strength * laplacian
            cavity = np.clip(cavity, 0.0, 1.0)
            if valid_mask is not None:
                cavity = cavity * valid_mask + 0.5 * (1.0 - valid_mask)

        return cavity.astype(np.float32)


# ---------------------------------------------------------------------------
# 8C: SSS Thickness Generator
# ---------------------------------------------------------------------------

class SSSThicknessGenerator:
    """
    Computes subsurface scattering thickness by estimating how thin
    the mesh is at each vertex.

    Method: For each vertex, cast rays in multiple directions and measure
    the distance to the nearest exit point on the opposite side of the mesh.
    Thin areas (ears, nostrils, eyelids) → high SSS value.
    Thick areas (forehead, jaw) → low SSS value.

    Since full ray-mesh intersection is expensive, we use a simplified
    heuristic: measure the distance from each vertex to the nearest vertex
    on the "opposite side" (vertices with opposing normals).
    """

    def __init__(self, n_rays: int = 32, normalize: bool = True):
        self.n_rays = n_rays
        self.normalize = normalize

    def generate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_normals: np.ndarray,
        uv_coords: np.ndarray,
        uv_faces: np.ndarray,
        flame_faces: np.ndarray,
        resolution: int = 2048,
    ) -> np.ndarray:
        """
        Generate SSS thickness map.

        Parameters
        ----------
        vertices : (V, 3) float32
        faces : (F, 3) int
        vertex_normals : (V, 3) float32
        uv_coords, uv_faces, flame_faces : UV parameterisation arrays
        resolution : output resolution

        Returns
        -------
        sss_map : (resolution, resolution) float32 in [0, 1]
            Higher = thinner tissue (more light scattering)
        """
        n_verts = len(vertices)

        # Compute per-vertex thickness using the opposing-normal heuristic
        # For each vertex, find vertices whose normals point in roughly
        # the opposite direction and measure the minimum distance
        per_vertex_thickness = np.ones(n_verts, dtype=np.float32) * 1.0  # Default: thick

        # Build a simple spatial index: for each vertex, find the nearest
        # vertex on the "other side" (opposing normal direction)
        # This is O(V²) worst case but V=5023 for FLAME so it's ~25M operations — fine.

        # Subsample rays per vertex for efficiency
        for vi in range(n_verts):
            n_i = vertex_normals[vi]
            p_i = vertices[vi]

            # Find vertices with opposing normals (dot product < -0.3)
            dots = np.sum(vertex_normals * n_i[np.newaxis, :], axis=1)
            opposing = np.where(dots < -0.3)[0]

            if len(opposing) == 0:
                per_vertex_thickness[vi] = 1.0  # Thick (no opposing surface found)
                continue

            # Compute distances to opposing vertices
            diffs = vertices[opposing] - p_i[np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=1)

            # The minimum distance is the local thickness
            min_dist = dists.min()
            per_vertex_thickness[vi] = min_dist

        # Normalize to [0, 1] where 0 = thickest, 1 = thinnest
        if self.normalize:
            t_min = per_vertex_thickness.min()
            t_max = per_vertex_thickness.max()
            t_range = t_max - t_min
            if t_range > 1e-8:
                per_vertex_thickness = (per_vertex_thickness - t_min) / t_range
            else:
                per_vertex_thickness = np.zeros_like(per_vertex_thickness)

            # Invert: thin = high SSS value, thick = low
            per_vertex_sss = 1.0 - per_vertex_thickness
        else:
            per_vertex_sss = per_vertex_thickness

        # Rasterize per-vertex SSS into UV space
        sss_map = self._rasterize_scalar(
            per_vertex_sss, uv_coords, uv_faces, flame_faces, resolution
        )

        # Smooth slightly to reduce triangle-edge artifacts
        sss_map = cv2.GaussianBlur(sss_map, (5, 5), 1.0)

        return np.clip(sss_map, 0, 1).astype(np.float32)

    def _rasterize_scalar(
        self,
        per_vertex_scalar: np.ndarray,
        uv_coords: np.ndarray,
        uv_faces: np.ndarray,
        flame_faces: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        """Rasterize a per-vertex scalar field into UV space."""
        R = resolution
        result = np.zeros((R, R), dtype=np.float32)
        count = np.zeros((R, R), dtype=np.float32)

        uv_px = uv_coords.copy().astype(np.float64)
        uv_px[:, 0] = np.clip(uv_px[:, 0] * (R - 1), 0, R - 1)
        uv_px[:, 1] = np.clip((1.0 - uv_px[:, 1]) * (R - 1), 0, R - 1)

        n_faces = min(len(uv_faces), len(flame_faces))

        for i in range(n_faces):
            tri_uv = uv_faces[i]
            tri_geom = flame_faces[i]

            p0 = uv_px[tri_uv[0]]
            p1 = uv_px[tri_uv[1]]
            p2 = uv_px[tri_uv[2]]

            xmin = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
            xmax = min(R - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
            ymin = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
            ymax = min(R - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

            if xmax <= xmin or ymax <= ymin:
                continue

            area = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
            if abs(area) < 1e-6:
                continue

            xs, ys = np.meshgrid(
                np.arange(xmin, xmax + 1, dtype=np.float64),
                np.arange(ymin, ymax + 1, dtype=np.float64)
            )
            b0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / area
            b1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / area
            b2 = 1.0 - b0 - b1

            inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
            if not np.any(inside):
                continue

            yc = ys[inside].astype(np.int32)
            xc = xs[inside].astype(np.int32)
            val = (b0[inside] * per_vertex_scalar[tri_geom[0]] +
                   b1[inside] * per_vertex_scalar[tri_geom[1]] +
                   b2[inside] * per_vertex_scalar[tri_geom[2]])

            result[yc, xc] += val.astype(np.float32)
            count[yc, xc] += 1.0

        valid = count > 0
        result[valid] /= count[valid]
        return result


# ---------------------------------------------------------------------------
# PBR Material Stack Orchestrator
# ---------------------------------------------------------------------------

class PBRMaterialStack:
    """
    Orchestrates generation of the complete PBR material stack:
    roughness, cavity/AO, and SSS thickness maps.

    Parameters
    ----------
    roughness_config : dict
        Roughness zone parameters (r_t_zone, r_cheeks, r_lips, etc.)
    cavity_strength : float
        Laplacian cavity modulation strength.
    sss_ray_count : int
        Rays per vertex for SSS thickness estimation.
    sss_normalize : bool
        Whether to normalize SSS values to [0, 1].
    """

    def __init__(
        self,
        roughness_config: Optional[Dict] = None,
        cavity_strength: float = 0.35,
        sss_ray_count: int = 32,
        sss_normalize: bool = True,
    ):
        r_cfg = roughness_config or {}
        self.roughness_gen = RoughnessMapGenerator(
            r_t_zone=r_cfg.get('r_t_zone', 0.33),
            r_cheeks=r_cfg.get('r_cheeks', 0.57),
            r_lips=r_cfg.get('r_lips', 0.22),
            r_periorbital=r_cfg.get('r_periorbital', 0.47),
        )
        self.cavity_gen = CavityMapGenerator(strength=cavity_strength)
        self.sss_gen = SSSThicknessGenerator(
            n_rays=sss_ray_count, normalize=sss_normalize
        )

    def generate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_normals: np.ndarray,
        uv_coords: np.ndarray,
        uv_faces: np.ndarray,
        flame_faces: np.ndarray,
        displacement_map: Optional[np.ndarray] = None,
        resolution: int = 2048,
    ) -> Dict[str, np.ndarray]:
        """
        Generate the complete PBR material stack.

        Returns
        -------
        dict with keys:
            'roughness'     : (R, R) float32 in [0, 1]
            'cavity'        : (R, R) float32 in [0, 1]
            'sss_thickness' : (R, R) float32 in [0, 1]
            'zone_masks'    : dict of zone mask arrays
        """
        print("[Stage 8] Building anatomical zone masks...")
        zone_masks = _build_anatomical_zone_masks(
            uv_coords, uv_faces, flame_faces, vertices, resolution
        )

        # 8A: Roughness
        print("[Stage 8A] Generating roughness map (anatomical zones + displacement variation)...")
        roughness = self.roughness_gen.generate(zone_masks, displacement_map)

        # 8B: Cavity / AO
        print("[Stage 8B] Generating cavity/ambient occlusion map...")
        if displacement_map is not None:
            disp = displacement_map
            if disp.shape[0] != resolution:
                disp = cv2.resize(disp, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
            cavity = self.cavity_gen.generate(disp, zone_masks.get('valid'))
        else:
            cavity = np.full((resolution, resolution), 0.5, dtype=np.float32)
            print("[Stage 8B] No displacement map available — generating flat cavity map.")

        # 8C: SSS Thickness
        print("[Stage 8C] Computing SSS thickness map (geometric ray-march)...")
        sss = self.sss_gen.generate(
            vertices, faces, vertex_normals,
            uv_coords, uv_faces, flame_faces,
            resolution
        )

        return {
            'roughness': roughness,
            'cavity': cavity,
            'sss_thickness': sss,
            'zone_masks': zone_masks,
        }

    def save_maps(
        self,
        result: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "head",
    ) -> Dict[str, str]:
        """Save PBR material maps to disk."""
        out = Path(output_dir)
        textures_dir = out / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Roughness (8-bit grayscale)
        roughness_path = textures_dir / f"{prefix}_roughness_map.png"
        roughness_8bit = (np.clip(result['roughness'], 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(roughness_path), roughness_8bit)
        paths['roughness_map'] = str(roughness_path)

        # Cavity / AO (8-bit grayscale)
        cavity_path = textures_dir / f"{prefix}_cavity_ao_map.png"
        cavity_8bit = (np.clip(result['cavity'], 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(cavity_path), cavity_8bit)
        paths['cavity_ao_map'] = str(cavity_path)

        # SSS Thickness (8-bit grayscale)
        sss_path = textures_dir / f"{prefix}_sss_thickness_map.png"
        sss_8bit = (np.clip(result['sss_thickness'], 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(sss_path), sss_8bit)
        paths['sss_thickness_map'] = str(sss_path)

        return paths
