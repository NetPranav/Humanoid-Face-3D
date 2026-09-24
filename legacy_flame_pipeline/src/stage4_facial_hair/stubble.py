"""
Procedural Follicular Stubble Micro-Displacement Engine.

Synthesizes high-frequency follicular stubble micro-displacement and tangent normal maps
for short facial hair (5 o'clock shadow, light mustache, stubble, beard base).

Modulates directly with Stage 3 GAN micro-displacement maps while strictly
enforcing the Unreal Engine 5 Neck Seam Contract:
  - Displacements on the lowest 20% collar vertices/pixels are strictly Delta v = 0.0 mm.
"""
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import cv2

from src.stage4_facial_hair.regions import (
    FacialHairConfig,
    FacialHairRegionSegmenter,
    resolve_hair_config,
    compute_normalized_coordinates,
)
from src.stage3_detail.rasterizer import load_flame_uv_layout, rasterize_uv_maps


class ProceduralStubbleEngine:
    """
    Synthesizes organic, high-frequency follicular stubble displacement.
    """
    def __init__(self, neck_collar_threshold: float = 0.20):
        self.segmenter = FacialHairRegionSegmenter(neck_collar_threshold=neck_collar_threshold)
        self.neck_collar_threshold = float(neck_collar_threshold)

    def generate_stubble_displacement(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        base_displacement_mm: Optional[np.ndarray] = None,
        config: Optional[Union[FacialHairConfig, Dict[str, Any], str]] = None,
        resolution: int = 512,
        uv_coords: Optional[np.ndarray] = None,
        uv_faces: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes procedural follicular micro-displacement and updates normal maps.

        Args:
            vertices: (N, 3) float array of neutral head vertices.
            faces: (F, 3) int array of triangle faces.
            base_displacement_mm: (H, W) optional incoming Stage 3 displacement map in mm.
            config: FacialHairConfig, preset name, or dict.
            resolution: Output texture resolution (default 512x512).
            uv_coords: (N_uv, 2) optional explicit UV coordinates.
            uv_faces: (F_uv, 3) optional explicit UV triangle faces.

        Returns:
            dict containing:
              - 'displacement_mm': (H, W) combined displacement map in millimeters
              - 'stubble_only_mm': (H, W) isolated stubble displacement
              - 'normal_map': (H, W, 3) uint8 RGB tangent normal map
              - 'hair_mask_uv': (H, W) float32 regional hair density mask [0, 1]
              - 'collar_mask_uv': (H, W) float32 collar pinning mask
        """
        cfg = resolve_hair_config(config)

        # 1. Resolve canonical UV layout if not provided
        if uv_coords is None or uv_faces is None:
            try:
                uv_coords, uv_faces = load_flame_uv_layout()
            except Exception:
                # Fallback cylindrical unwrapping
                x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                u = (np.arctan2(x, -z) + np.pi) / (2.0 * np.pi)
                y_min, y_max = float(np.min(y)), float(np.max(y))
                v = (y - y_min) / max(y_max - y_min, 1e-6)
                uv_coords = np.stack([u, v], axis=1).astype(np.float32)
                uv_faces = faces.copy()

        # 2. Extract anatomical region weights on 3D vertices
        region_res = self.segmenter.extract_region_weights(vertices, cfg)
        vertex_mask = region_res['composite']  # (N,) in [0, 1]
        vertex_collar = region_res['collar_mask']

        # 3. Rasterize vertex masks into UV space
        uv_mask = self._rasterize_scalar_mask(vertex_mask, uv_coords, uv_faces, resolution)
        uv_collar = self._rasterize_scalar_mask(vertex_collar, uv_coords, uv_faces, resolution)

        # Enforce exact bitwise zero on collar in UV space
        uv_mask = uv_mask * uv_collar

        # If clean shaven or stubble disabled, return base or zero displacement
        if cfg.is_clean_shaven() or not cfg.generate_stubble or np.max(uv_mask) < 1e-4:
            if base_displacement_mm is not None:
                disp_total = base_displacement_mm.copy()
            else:
                disp_total = np.zeros((resolution, resolution), dtype=np.float32)
            normal_rgb = self.compute_normal_map_from_displacement(disp_total, resolution)
            return {
                'displacement_mm': disp_total,
                'stubble_only_mm': np.zeros((resolution, resolution), dtype=np.float32),
                'normal_map': normal_rgb,
                'hair_mask_uv': uv_mask,
                'collar_mask_uv': uv_collar,
            }

        # 4. Generate high-frequency follicular cellular noise pattern
        follicle_pattern = self._generate_follicular_noise(
            resolution=resolution,
            density=cfg.stubble_density
        )

        # 5. Modulate stubble displacement by regional mask and stubble length
        # Follicles are positive protrusions with small pores (valleys around shafts)
        stubble_relief = follicle_pattern * cfg.stubble_length_mm
        stubble_displacement = uv_mask * stubble_relief

        # Enforce collar contract
        stubble_displacement[uv_collar <= 1e-5] = 0.0

        # 6. Combine with base Stage 3 GAN displacement
        if base_displacement_mm is not None:
            if base_displacement_mm.shape != (resolution, resolution):
                base_resized = cv2.resize(
                    base_displacement_mm,
                    (resolution, resolution),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                base_resized = base_displacement_mm
            disp_total = base_resized + stubble_displacement
        else:
            disp_total = stubble_displacement

        # 7. Compute updated tangent-space normal map
        normal_rgb = self.compute_normal_map_from_displacement(disp_total, resolution)

        return {
            'displacement_mm': disp_total.astype(np.float32),
            'stubble_only_mm': stubble_displacement.astype(np.float32),
            'normal_map': normal_rgb,
            'hair_mask_uv': uv_mask.astype(np.float32),
            'collar_mask_uv': uv_collar.astype(np.float32),
        }

    def _generate_follicular_noise(self, resolution: int, density: float) -> np.ndarray:
        """
        Generates cellular follicle micro-texture at specified resolution.
        """
        rng = np.random.RandomState(42)

        # Follicle grid cells
        grid_cells = int(np.clip(128 * density, 64, 256))
        cell_size = resolution / grid_cells

        # Generate jittered follicle seed points in cell coordinates
        jitter_x = rng.uniform(0.15, 0.85, size=(grid_cells, grid_cells))
        jitter_y = rng.uniform(0.15, 0.85, size=(grid_cells, grid_cells))
        follicle_height = rng.uniform(0.6, 1.0, size=(grid_cells, grid_cells))

        # Pixel coordinates
        y_coords, x_coords = np.mgrid[0:resolution, 0:resolution]
        cell_i = np.clip((y_coords / cell_size).astype(np.int32), 0, grid_cells - 1)
        cell_j = np.clip((x_coords / cell_size).astype(np.int32), 0, grid_cells - 1)

        # Distance to center seed of current cell
        center_x = (cell_j + jitter_x[cell_i, cell_j]) * cell_size
        center_y = (cell_i + jitter_y[cell_i, cell_j]) * cell_size

        dx = x_coords - center_x
        dy = y_coords - center_y
        dist_sq = (dx * dx + dy * dy) / (cell_size * cell_size * 0.25)

        # Organic follicle shaft bump: sharp core with smooth dropoff
        follicle_bump = np.exp(-dist_sq * 2.5) * follicle_height[cell_i, cell_j]

        # Add subtle epidermal micro-grain
        grain = rng.normal(0.0, 0.05, size=(resolution, resolution))
        follicle_bump = np.clip(follicle_bump + grain * follicle_bump, 0.0, 1.0)

        return follicle_bump.astype(np.float32)

    def _rasterize_scalar_mask(
        self,
        vertex_values: np.ndarray,
        uv_coords: np.ndarray,
        uv_faces: np.ndarray,
        resolution: int
    ) -> np.ndarray:
        """
        Rasterizes per-vertex scalar values into UV space via barycentric triangle rasterization.
        """
        mask = np.zeros((resolution, resolution), dtype=np.float32)
        n_uv = len(uv_coords)

        # Map vertex values to UV vertices
        if len(vertex_values) == n_uv:
            uv_vals = vertex_values
        else:
            # Map by indexing or nearest if counts differ
            indices = np.linspace(0, len(vertex_values) - 1, n_uv).astype(np.int32)
            uv_vals = vertex_values[indices]

        # Rasterize faces using OpenCV fillConvexPoly with barycentric interpolation approximation
        for f_idx, face in enumerate(uv_faces):
            if np.max(face) >= n_uv:
                continue
            uvs = uv_coords[face]  # (3, 2)
            vals = uv_vals[face]   # (3,)

            # Convert UV [0, 1] to pixel coords [0, resolution - 1]
            pts = np.zeros((3, 2), dtype=np.int32)
            pts[:, 0] = np.clip(uvs[:, 0] * (resolution - 1), 0, resolution - 1).astype(np.int32)
            pts[:, 1] = np.clip((1.0 - uvs[:, 1]) * (resolution - 1), 0, resolution - 1).astype(np.int32)

            mean_val = float(np.mean(vals))
            if mean_val > 1e-4:
                cv2.fillConvexPoly(mask, pts, mean_val)

        # Smooth boundary transitions
        mask = cv2.GaussianBlur(mask, (5, 5), 1.2)
        return mask

    def compute_normal_map_from_displacement(
        self,
        disp_mm: np.ndarray,
        resolution: int,
        strength: float = 2.0
    ) -> np.ndarray:
        """
        Computes tangent-space normal map from 2D displacement using Sobel finite differences.
        """
        sobel_x = cv2.Sobel(disp_mm, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(disp_mm, cv2.CV_32F, 0, 1, ksize=3)

        # Tangent space normal: [-dx, -dy, 1.0] normalized
        nx = -sobel_x * strength
        ny = -sobel_y * strength
        nz = np.ones_like(disp_mm, dtype=np.float32)

        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        norm = np.maximum(norm, 1e-6)

        nx = nx / norm
        ny = ny / norm
        nz = nz / norm

        # Map [-1, 1] to [0, 255] RGB
        r = np.clip(((nx + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        g = np.clip(((ny + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        b = np.clip(((nz + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)

        return np.stack([r, g, b], axis=-1)

    def encode_16bit_displacement(self, disp_mm: np.ndarray, p99_scale: float = 1.5) -> np.ndarray:
        """
        Encodes displacement in millimeters into 16-bit unsigned integer array (0 to 65535).
        Midpoint 32768 represents exactly 0.0 mm displacement.
        """
        norm_disp = np.clip(disp_mm / max(p99_scale, 1e-4), -1.0, 1.0)
        uint16_vals = np.clip((norm_disp * 32767.0) + 32768.0, 0, 65535).astype(np.uint16)
        return uint16_vals

    def save_maps(
        self,
        stubble_data: Dict[str, Any],
        output_dir: Union[str, Path],
        prefix: str = "stubble",
        p99_scale: float = 1.5
    ) -> Dict[str, str]:
        """
        Saves 16-bit displacement PNG and normal map PNG to output directory.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        disp_mm = stubble_data['displacement_mm']
        normal_rgb = stubble_data['normal_map']

        # 16-bit PNG
        disp_16bit = self.encode_16bit_displacement(disp_mm, p99_scale=p99_scale)
        disp_file = out_path / f"{prefix}_displacement_16bit.png"
        cv2.imwrite(str(disp_file.resolve()), disp_16bit)

        # Normal map (OpenCV uses BGR)
        normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)
        normal_file = out_path / f"{prefix}_normal_map.png"
        cv2.imwrite(str(normal_file.resolve()), normal_bgr)

        return {
            "displacement_png": str(disp_file.resolve()),
            "normal_png": str(normal_file.resolve()),
            "p99_scale_mm": str(p99_scale),
        }
