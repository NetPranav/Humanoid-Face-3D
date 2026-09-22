"""
Stage 3 Tier 3: 4K Anatomical Zone Pore & Micro-Detail Synthesis Engine.

Synthesizes true 50-micron epidermal skin pores, sebaceous cellular follicles,
and directional skin tension lines calibrated strictly to human facial anatomy
(the Texturing.xyz / MetaHuman industry standard).

All operations are procedural (vectorized NumPy/OpenCV) — zero GPU required,
deterministic, and supports both real-time (1024²) and film-grade (4096²) synthesis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.stage3_detail.rasterizer import (
    compute_vertex_normals,
    load_flame_uv_layout,
    rasterize_uv_maps,
)
from src.stage8_pbr.material_stack import _build_anatomical_zone_masks


class AnatomicalPoreSynthesizer:
    """
    Tier 3 Micro-Detail Engine: Anatomical Skin Pore Synthesis.

    Synthesizes multi-octave cellular skin grain, sebaceous follicles,
    and directional tension lines modulated by facial anatomy zones.
    """

    def __init__(
        self,
        uv_template_path: Optional[Union[str, Path]] = None,
        resolution: int = 1024,
        pore_density_scale: float = 1.0,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        uv_template_path : Optional path to FLAME UV template (.obj or .npz)
        resolution : Base synthesis resolution (1024 or 4096)
        pore_density_scale : Density multiplier for pore count (higher = denser pores)
        random_seed : Deterministic RNG seed
        """
        self.resolution = int(resolution)
        self.density_scale = float(pore_density_scale)
        self.seed = int(random_seed)

        # Load canonical FLAME UV layout
        self.uv_coords, self.uv_faces = load_flame_uv_layout(
            str(uv_template_path) if uv_template_path else None
        )

    # -----------------------------------------------------------------------
    # 1. Cellular Basis Generators (Worley / Voronoi / Gabor)
    # -----------------------------------------------------------------------

    def generate_follicular_pores(
        self,
        resolution: int,
        grid_dim: int = 64,
        pore_scale: float = 4.0,
        rim_weight: float = 0.25,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Synthesizes circular sebaceous pore depressions with raised annular rims
        (typical of nose, chin, and central forehead T-zone follicles).

        Returns (R, R) float32 array in [-1.0, +0.3] range.
        """
        grid_dim = int(grid_dim * np.sqrt(self.density_scale))
        grid_dim = max(16, min(grid_dim, resolution // 4))
        cell_w = resolution / float(grid_dim)

        rng = np.random.RandomState(seed)
        seeds = rng.uniform(0.15, 0.85, (grid_dim, grid_dim, 2)).astype(np.float32)

        x = np.arange(resolution, dtype=np.float32) / cell_w
        y = np.arange(resolution, dtype=np.float32) / cell_w
        xs, ys = np.meshgrid(x, y)

        cx = np.clip(np.floor(xs).astype(int), 0, grid_dim - 1)
        cy = np.clip(np.floor(ys).astype(int), 0, grid_dim - 1)

        min_dist = np.full((resolution, resolution), 999.0, dtype=np.float32)

        # Query 3x3 neighboring cells
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = np.clip(cx + dx, 0, grid_dim - 1)
                ny = np.clip(cy + dy, 0, grid_dim - 1)
                sx = nx.astype(np.float32) + seeds[ny, nx, 0]
                sy = ny.astype(np.float32) + seeds[ny, nx, 1]
                dist = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
                min_dist = np.minimum(min_dist, dist)

        r = min_dist * pore_scale
        # Follicular pore profile: negative depression in center + subtle raised annular rim
        pit = -np.exp(-(r ** 2))
        rim = rim_weight * r * np.exp(-((r - 1.2) ** 2))
        pore_field = pit + rim

        return pore_field.astype(np.float32)

    def generate_anisotropic_micro_grain(
        self,
        resolution: int,
        grid_dim: int = 128,
        stretch_factor: float = 1.8,
        angle_deg: float = 35.0,
        seed: int = 101,
    ) -> np.ndarray:
        """
        Synthesizes fine cellular epidermal grain stretched along Langer's skin tension lines
        (typical of lateral cheeks and zygomatic arches).

        Returns (R, R) float32 array.
        """
        grid_dim = int(grid_dim * np.sqrt(self.density_scale))
        grid_dim = max(32, min(grid_dim, resolution // 2))
        cell_w = resolution / float(grid_dim)

        rng = np.random.RandomState(seed)
        seeds = rng.uniform(0.1, 0.9, (grid_dim, grid_dim, 2)).astype(np.float32)

        # Coordinate rotation for Langer's lines
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        x = (np.arange(resolution, dtype=np.float32) - resolution * 0.5) / cell_w
        y = (np.arange(resolution, dtype=np.float32) - resolution * 0.5) / cell_w
        xs, ys = np.meshgrid(x, y)

        xr = (xs * cos_a + ys * sin_a)
        yr = (-xs * sin_a + ys * cos_a) * (1.0 / stretch_factor)

        xr_norm = xr + (grid_dim * 0.5)
        yr_norm = yr + (grid_dim * 0.5)

        cx = np.clip(np.floor(xr_norm).astype(int), 0, grid_dim - 1)
        cy = np.clip(np.floor(yr_norm).astype(int), 0, grid_dim - 1)

        d1 = np.full((resolution, resolution), 999.0, dtype=np.float32)
        d2 = np.full((resolution, resolution), 999.0, dtype=np.float32)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = np.clip(cx + dx, 0, grid_dim - 1)
                ny = np.clip(cy + dy, 0, grid_dim - 1)
                sx = nx.astype(np.float32) + seeds[ny, nx, 0]
                sy = ny.astype(np.float32) + seeds[ny, nx, 1]
                dist = np.sqrt((xr_norm - sx) ** 2 + ((yr_norm - sy) * stretch_factor) ** 2)

                # Track 1st and 2nd closest distances (Voronoi cell boundaries)
                update_d1 = dist < d1
                d2 = np.where(update_d1, d1, np.minimum(d2, dist))
                d1 = np.where(update_d1, dist, d1)

        # Cellular boundary ridge: F2 - F1
        cellular = (d2 - d1) * 2.5 - 0.5
        # Soften and normalize to [-1, 1]
        cellular = np.clip(cellular, -1.0, 1.0)

        return cellular.astype(np.float32)

    def generate_lip_striations(
        self,
        resolution: int,
        frequency_x: int = 160,
        frequency_y: int = 24,
        seed: int = 202,
    ) -> np.ndarray:
        """
        Synthesizes vertical dermal papillary ridges and micro-folds on the vermilion lips.
        Strictly vertical orientation with multi-octave sinusoidal modulation.

        Returns (R, R) float32 array.
        """
        x = np.linspace(0, 2.0 * np.pi * frequency_x, resolution, dtype=np.float32)
        y = np.linspace(0, 2.0 * np.pi * frequency_y, resolution, dtype=np.float32)
        xs, ys = np.meshgrid(x, y)

        rng = np.random.RandomState(seed)
        noise_phase = rng.uniform(0, 2.0 * np.pi, (resolution, resolution)).astype(np.float32)
        noise_phase = cv2.GaussianBlur(noise_phase, (0, 0), 4.0)

        # Vertical grooves modulated by subtle lateral jitter
        octave1 = np.sin(xs + noise_phase * 1.5)
        octave2 = 0.5 * np.sin(xs * 2.0 + noise_phase * 2.0)
        octave3 = 0.25 * np.sin(xs * 4.0)

        striations = (octave1 + octave2 + octave3) / 1.75
        # Modulate vertically so ridges fade at outer lip margins
        v_profile = np.sin(np.linspace(0, np.pi, resolution, dtype=np.float32))[:, np.newaxis]
        striations = striations * v_profile

        return striations.astype(np.float32)

    def generate_transverse_micro_bands(
        self,
        resolution: int,
        frequency_y: int = 72,
        seed: int = 303,
    ) -> np.ndarray:
        """
        Synthesizes horizontal dermal tension micro-bands (forehead and neck tension lines).
        """
        y = np.linspace(0, 2.0 * np.pi * frequency_y, resolution, dtype=np.float32)
        x = np.linspace(0, 2.0 * np.pi * 12, resolution, dtype=np.float32)
        xs, ys = np.meshgrid(x, y)

        rng = np.random.RandomState(seed)
        noise_warp = cv2.GaussianBlur(
            rng.uniform(-1.0, 1.0, (resolution, resolution)).astype(np.float32),
            (0, 0),
            8.0,
        )

        bands = np.sin(ys + noise_warp * 3.0) * 0.7 + 0.3 * np.sin(ys * 2.5)
        return bands.astype(np.float32)

    # -----------------------------------------------------------------------
    # 2. Anatomical Zone Assembly & Synthesis
    # -----------------------------------------------------------------------

    def build_zone_masks(
        self,
        resolution: int,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Constructs normalized anatomical zone masks in UV space.
        """
        if vertices is not None and faces is not None:
            flame_faces = faces
            n_faces = min(len(self.uv_faces), len(flame_faces))
            masks = _build_anatomical_zone_masks(
                uv_coords=self.uv_coords,
                uv_faces=self.uv_faces[:n_faces],
                flame_faces=flame_faces[:n_faces],
                vertices=vertices,
                resolution=resolution,
            )
            # Add explicit neck_pinning key
            if 'neck_pinning' not in masks:
                masks['neck_pinning'] = masks.get('neck_collar', np.zeros((resolution, resolution), dtype=np.float32))
            return masks

        # Fallback UV geometric heuristics if 3D vertices are omitted
        masks = {}
        u = np.linspace(0, 1, resolution)
        v = np.linspace(1, 0, resolution)  # Row 0 = V 1.0, Row H-1 = V 0.0
        ug, vg = np.meshgrid(u, v)

        # Valid facial UV disc
        valid = (np.sqrt((ug - 0.5) ** 2 + (vg - 0.5) ** 2) < 0.45).astype(np.float32)
        masks['valid'] = valid

        # T-zone: central column (forehead + nose + chin)
        t_zone = ((np.abs(ug - 0.5) < 0.12) & (vg > 0.25) & (vg < 0.85) & (valid > 0)).astype(np.float32)
        masks['t_zone'] = cv2.GaussianBlur(t_zone, (0, 0), resolution * 0.015)

        # Cheeks: lateral to T-zone
        cheeks = (
            (np.abs(ug - 0.5) >= 0.12) & (np.abs(ug - 0.5) < 0.35) &
            (vg > 0.35) & (vg < 0.65) & (valid > 0)
        ).astype(np.float32)
        masks['cheeks'] = cv2.GaussianBlur(cheeks, (0, 0), resolution * 0.02)

        # Lips: center lower-middle
        lips = ((np.abs(ug - 0.5) < 0.14) & (vg >= 0.30) & (vg <= 0.42) & (valid > 0)).astype(np.float32)
        masks['lips'] = cv2.GaussianBlur(lips, (0, 0), resolution * 0.01)

        # Forehead: upper zone
        forehead = ((np.abs(ug - 0.5) < 0.35) & (vg >= 0.65) & (vg < 0.85) & (valid > 0)).astype(np.float32)
        masks['forehead'] = cv2.GaussianBlur(forehead, (0, 0), resolution * 0.02)

        # Neck collar: bottom 20% (strictly pinned)
        neck = ((vg <= 0.20) & (valid > 0)).astype(np.float32)
        masks['neck_collar'] = neck
        masks['neck_pinning'] = neck

        return masks

    def synthesize(
        self,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        resolution: Optional[int] = None,
        zone_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Executes end-to-end procedural pore synthesis.

        Parameters
        ----------
        vertices : Optional (V, 3) FLAME neutral mesh vertices in metres
        faces : Optional (F, 3) face indices
        resolution : Optional resolution override (e.g. 4096 or 1024)
        zone_masks : Optional precomputed zone masks dict

        Returns
        -------
        dict with keys:
            'displacement_mm'    : (R, R) float32 signed micro-displacement in mm
            'tangent_normal_rgb' : (R, R, 3) uint8 tangent-space normal map
            'pore_density_map'   : (R, R) float32 visual pore density mask
            'zone_masks'         : Dict of anatomical masks
        """
        res = int(resolution or self.resolution)

        # 1. Resolve anatomical masks
        if zone_masks is None:
            masks = self.build_zone_masks(res, vertices, faces)
        else:
            masks = zone_masks

        valid = masks.get('valid', np.ones((res, res), dtype=np.float32))
        m_tzone = masks.get('t_zone', np.zeros((res, res), dtype=np.float32))
        m_cheeks = masks.get('cheeks', np.zeros((res, res), dtype=np.float32))
        m_lips = masks.get('lips', np.zeros((res, res), dtype=np.float32))
        m_neck_pinning = masks.get('neck_pinning', masks.get('neck_collar', np.zeros((res, res), dtype=np.float32)))

        # 2. Synthesize individual anatomical micro-octaves
        # T-Zone / Nose: large sebaceous follicles (0.15 - 0.35 mm depth)
        t_pores = self.generate_follicular_pores(
            resolution=res,
            grid_dim=res // 16,
            pore_scale=3.5,
            rim_weight=0.30,
            seed=self.seed + 1,
        )
        d_tzone = t_pores * 0.28  # up to 0.28 mm depth

        # Cheeks: fine elliptical pores & cellular micro-grain (0.05 - 0.12 mm)
        cheek_grain = self.generate_anisotropic_micro_grain(
            resolution=res,
            grid_dim=res // 12,
            stretch_factor=1.75,
            angle_deg=35.0,
            seed=self.seed + 2,
        )
        d_cheeks = cheek_grain * 0.08  # ~0.08 mm depth

        # Lips: vertical dermal papillary ridges & micro-folds (0.10 - 0.25 mm)
        lip_ridges = self.generate_lip_striations(
            resolution=res,
            frequency_x=int(res * 0.16),
            frequency_y=int(res * 0.025),
            seed=self.seed + 3,
        )
        d_lips = lip_ridges * 0.18  # ~0.18 mm amplitude

        # Forehead & general background micro-texture (0.05 - 0.15 mm)
        bg_bands = self.generate_transverse_micro_bands(
            resolution=res,
            frequency_y=int(res * 0.07),
            seed=self.seed + 4,
        )
        d_bg = bg_bands * 0.06

        # 3. Composite layers modulated by anatomical zone masks
        # Background skin cellular grain everywhere on face
        total_disp = d_bg * valid

        # Modulate T-zone follicles
        total_disp = total_disp + (d_tzone * m_tzone)

        # Modulate Cheek directional pores
        total_disp = total_disp + (d_cheeks * m_cheeks)

        # Modulate Lip vertical striations
        total_disp = np.where(m_lips > 0.3, (total_disp * (1.0 - m_lips) + d_lips * m_lips), total_disp)

        # 4. Strictly Enforce Rule 4: Neck Collar Pinning Contract
        # Lowest 20% of vertices must have delta v strictly 0.000000 mm
        collar_pin = np.clip(m_neck_pinning, 0.0, 1.0)
        total_disp = total_disp * (1.0 - collar_pin)

        # Ensure outside valid boundary is zero
        total_disp = total_disp * (valid > 0.1)

        # 5. Compute Tangent Normal Map
        pixel_size_mm = 0.25 * (1024.0 / float(res))
        dz_dx = cv2.Sobel(total_disp, cv2.CV_32F, 1, 0, ksize=3) / (2.0 * pixel_size_mm)
        dz_dy = cv2.Sobel(total_disp, cv2.CV_32F, 0, 1, ksize=3) / (2.0 * pixel_size_mm)

        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones((res, res), dtype=np.float32)

        norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        norm[norm == 0] = 1.0

        r = np.clip((nx / norm + 1.0) * 127.5, 0, 255).astype(np.uint8)
        g = np.clip((ny / norm + 1.0) * 127.5, 0, 255).astype(np.uint8)
        b = np.clip((nz / norm + 1.0) * 127.5, 0, 255).astype(np.uint8)
        normals_rgb = np.stack([r, g, b], axis=-1)

        # 6. Compute visual pore density map
        pore_density = np.clip(m_tzone * 0.9 + m_cheeks * 0.6 + m_lips * 0.8, 0.0, 1.0) * valid

        return {
            'displacement_mm': total_disp.astype(np.float32),
            'tangent_normal_rgb': normals_rgb,
            'pore_density_map': pore_density.astype(np.float32),
            'zone_masks': masks,
        }

    # -----------------------------------------------------------------------
    # 3. Serialization
    # -----------------------------------------------------------------------

    def save_maps(
        self,
        synthesis_result: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "micro",
        max_scale_mm: float = 1.0,
    ) -> Dict[str, Path]:
        """
        Saves 16-bit unsigned PNG displacement and tangent normal PNG.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        disp_mm = synthesis_result['displacement_mm']
        normals_rgb = synthesis_result['tangent_normal_rgb']

        # Encode displacement to 16-bit uint PNG
        normalized = np.clip(
            (disp_mm / max_scale_mm + 1.0) * 0.5,
            0.0,
            1.0,
        )
        disp_u16 = np.round(normalized * 65535.0).astype(np.uint16)

        disp_path = out_dir / f"{prefix}_displacement_16bit.png"
        normal_path = out_dir / f"{prefix}_normal.png"

        cv2.imwrite(str(disp_path), disp_u16)
        cv2.imwrite(str(normal_path), cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))

        return {
            'displacement_png': disp_path,
            'normal_png': normal_path,
        }
