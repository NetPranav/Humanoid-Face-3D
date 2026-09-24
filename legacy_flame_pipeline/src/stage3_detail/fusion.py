"""
Stage 3 Tier 3.5: Multi-Tier Geometry & PBR Material Coupling Engine.

Fuses Macro, Meso (photo-derived wrinkles), and Micro (4K anatomical pores)
into a unified, film-grade 16-bit metric displacement heightfield and mathematically
couples light interaction (micro-cavity ambient occlusion and dual-lobe roughness)
to pore depth.

Pure NumPy/OpenCV execution — zero GPU required, deterministic, and preserves
the Rule 4 Neck Seam Contract (bitwise collar boundary pinning).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.stage3_detail.photometric_detail import PhotometricDetailExtractor
from src.stage3_detail.anatomical_pores import AnatomicalPoreSynthesizer


class MultiTierDetailFusion:
    """
    Composites multi-tier facial geometry and couples PBR material properties.
    """

    def __init__(
        self,
        resolution: int = 1024,
        max_scale_mm: float = 5.0,
        cavity_strength: float = 0.40,
        cavity_blur_sigma: float = 0.8,
    ):
        """
        Parameters
        ----------
        resolution : Base canvas resolution (1024 or 4096)
        max_scale_mm : Maximum physical displacement bound in millimeters (default: 5.0 mm)
        cavity_strength : Scaling factor for pore-depth ambient occlusion darkening
        cavity_blur_sigma : Smoothing radius for cavity Laplacian
        """
        self.resolution = int(resolution)
        self.max_scale_mm = float(max_scale_mm)
        self.cavity_strength = float(cavity_strength)
        self.cavity_blur_sigma = float(cavity_blur_sigma)

    # -----------------------------------------------------------------------
    # 1. Multi-Tier Geometry Compositing
    # -----------------------------------------------------------------------

    def fuse_displacements(
        self,
        meso_disp_mm: np.ndarray,
        micro_disp_mm: np.ndarray,
        macro_disp_mm: Optional[np.ndarray] = None,
        weight_macro: float = 1.0,
        weight_meso: float = 0.5,
        weight_micro: float = 0.35,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Composites multi-tier displacements into a unified metric heightfield:
        D_total = w_macro * D_macro + w_meso * D_meso + w_micro * D_micro

        Parameters
        ----------
        meso_disp_mm : (H, W) photo-derived wrinkle displacement (mm)
        micro_disp_mm : (H, W) anatomical pore displacement (mm)
        macro_disp_mm : Optional (H, W) coarse macro residual displacement (mm)
        weight_macro, weight_meso, weight_micro : Relative tier blending weights
        mask : Optional (H, W) valid facial skin mask

        Returns
        -------
        total_disp_mm : (H, W) float32 composite displacement in millimeters
        """
        h, w = self.resolution, self.resolution

        # Ensure all maps match target resolution
        def _match_res(arr: Optional[np.ndarray]) -> np.ndarray:
            if arr is None:
                return np.zeros((h, w), dtype=np.float32)
            if arr.shape[:2] != (h, w):
                return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            return arr.astype(np.float32)

        meso = _match_res(meso_disp_mm)
        micro = _match_res(micro_disp_mm)
        macro = _match_res(macro_disp_mm)

        total = (weight_macro * macro) + (weight_meso * meso) + (weight_micro * micro)

        if mask is not None:
            m = _match_res(mask)
            total = total * (m > 0.1).astype(np.float32)

        # Enforce Rule 4 Neck Seam Contract Invariant (smooth Hermite taper + bitwise boundary pinning)
        transition_start = int(0.70 * h)
        bottom_start = int(0.85 * h)
        ramp_len = max(bottom_start - transition_start, 1)
        row_indices = np.arange(h, dtype=np.float32)[:, None]
        t = np.clip((bottom_start - row_indices) / float(ramp_len), 0.0, 1.0)
        smooth_falloff = t * t * (3.0 - 2.0 * t)
        total = total * smooth_falloff
        total[bottom_start:, :] = 0.0

        # Clip strictly to metric bounds
        total = np.clip(total, -self.max_scale_mm, self.max_scale_mm)
        return total.astype(np.float32)

    # -----------------------------------------------------------------------
    # 2. Tangent Normal Map Derivation
    # -----------------------------------------------------------------------

    def compute_tangent_normal_map(
        self,
        displacement_mm: np.ndarray,
        pixel_size_mm: Optional[float] = None,
    ) -> np.ndarray:
        """
        Computes standard RGB tangent-space normal map from composite displacement.
        """
        h, w = displacement_mm.shape[:2]
        if pixel_size_mm is None:
            pixel_size_mm = 0.25 * (1024.0 / float(w))

        dz_dx = cv2.Sobel(displacement_mm, cv2.CV_32F, 1, 0, ksize=3) / (2.0 * pixel_size_mm)
        dz_dy = cv2.Sobel(displacement_mm, cv2.CV_32F, 0, 1, ksize=3) / (2.0 * pixel_size_mm)

        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones((h, w), dtype=np.float32)

        norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        norm[norm == 0] = 1.0

        r = np.clip((nx / norm + 1.0) * 127.5, 0, 255).astype(np.uint8)
        g = np.clip((ny / norm + 1.0) * 127.5, 0, 255).astype(np.uint8)
        b = np.clip((nz / norm + 1.0) * 127.5, 0, 255).astype(np.uint8)

        return np.stack([r, g, b], axis=-1)

    # -----------------------------------------------------------------------
    # 3. Micro-Cavity / Ambient Occlusion Derivation
    # -----------------------------------------------------------------------

    def compute_cavity_ao_map(
        self,
        displacement_mm: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Derives pore-depth ambient occlusion directly from the negative Laplacian
        of the composite displacement:

        Cavity(u, v) = clip(1.0 - beta_cavity * max(0, -Laplacian(D)), 0.0, 1.0)

        Flat skin = 1.0 (full ambient illumination)
        Deep pores & wrinkle crevasses < 1.0 (light physically trapped in valleys)
        """
        h, w = displacement_mm.shape[:2]

        if self.cavity_blur_sigma > 0:
            disp_smooth = cv2.GaussianBlur(displacement_mm, (0, 0), self.cavity_blur_sigma)
        else:
            disp_smooth = displacement_mm

        # Discrete Laplacian (curvature)
        laplacian = cv2.Laplacian(disp_smooth, cv2.CV_32F, ksize=5)

        # In a displacement heightfield, concave depressions (pore pits and wrinkle valleys)
        # have positive Laplacian: ∇²D > 0
        concavity = np.maximum(0.0, laplacian)

        # Scale by cavity strength
        cavity = 1.0 - (self.cavity_strength * concavity)
        cavity = np.clip(cavity, 0.0, 1.0)

        if mask is not None:
            m = mask
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h))
            cavity = cavity * (m > 0.1).astype(np.float32) + 1.0 * (m <= 0.1).astype(np.float32)

        return cavity.astype(np.float32)

    # -----------------------------------------------------------------------
    # 4. Dual-Lobe Roughness Derivation
    # -----------------------------------------------------------------------

    def compute_dual_lobe_roughness(
        self,
        zone_masks: Dict[str, np.ndarray],
        micro_disp_mm: Optional[np.ndarray] = None,
        base_t_zone: float = 0.52,
        base_cheeks: float = 0.65,
        base_lips: float = 0.26,
        base_periorbital: float = 0.55,
        base_default: float = 0.60,
        coat_sheen_t_zone: float = 0.20,
    ) -> Dict[str, np.ndarray]:
        """
        Generates dual-lobe specular roughness maps:
        1. Base Roughness: Matte lipid barrier of the stratum corneum (0.50 - 0.68)
        2. Coat / Micro-Roughness: Sharp glossy specular sheen from sebum oil (0.18 - 0.24)
           concentrated in the T-zone and eyelid margins.
        3. Coat Weight: Localized sebum layer strength (0.35 on T-zone, 0.0 on cheeks/neck).
        """
        h, w = self.resolution, self.resolution

        def _get_mask(name: str) -> np.ndarray:
            m = zone_masks.get(name)
            if m is None:
                return np.zeros((h, w), dtype=np.float32)
            if m.shape[:2] != (h, w):
                return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            return m.astype(np.float32)

        valid = _get_mask('valid')
        if not np.any(valid > 0):
            valid = np.ones((h, w), dtype=np.float32)

        m_tzone = _get_mask('t_zone')
        m_cheeks = _get_mask('cheeks')
        m_lips = _get_mask('lips')
        m_periorbital = _get_mask('periorbital')

        # --- 1. Base Layer Roughness ---
        base_rough = np.full((h, w), base_default, dtype=np.float32)
        base_rough = base_rough * (1.0 - m_tzone) + base_t_zone * m_tzone
        base_rough = base_rough * (1.0 - m_cheeks) + base_cheeks * m_cheeks
        base_rough = base_rough * (1.0 - m_lips) + base_lips * m_lips
        base_rough = base_rough * (1.0 - m_periorbital) + base_periorbital * m_periorbital

        # --- 2. Coat Sheen Layer (Sebum Oil) ---
        # Sharp micro-sheen on T-zone and eyelids
        coat_rough = np.full((h, w), 0.38, dtype=np.float32)
        coat_rough = coat_rough * (1.0 - m_tzone) + coat_sheen_t_zone * m_tzone
        coat_rough = coat_rough * (1.0 - m_periorbital) + 0.25 * m_periorbital

        # --- 3. Coat Weight (Strength of Sebum Oil Layer) ---
        coat_weight = np.clip(m_tzone * 0.40 + m_periorbital * 0.20, 0.0, 1.0)

        # --- 4. Displacement-Coupled Micro-Roughness Variation ---
        if micro_disp_mm is not None:
            m_disp = micro_disp_mm
            if m_disp.shape[:2] != (h, w):
                m_disp = cv2.resize(m_disp.astype(np.float32), (w, h))
            lap = cv2.Laplacian(m_disp, cv2.CV_32F, ksize=3)
            lap_norm = lap / (np.percentile(np.abs(lap), 99.0) + 1e-6)
            base_rough += 0.05 * np.clip(lap_norm, -1.0, 1.0)
            coat_rough += 0.03 * np.clip(lap_norm, -1.0, 1.0)

        base_rough = np.clip(base_rough, 0.0, 1.0) * valid
        coat_rough = np.clip(coat_rough, 0.0, 1.0) * valid
        coat_weight = np.clip(coat_weight, 0.0, 1.0) * valid

        return {
            'base_roughness': base_rough.astype(np.float32),
            'coat_roughness': coat_rough.astype(np.float32),
            'coat_weight': coat_weight.astype(np.float32),
        }

    # -----------------------------------------------------------------------
    # 5. End-to-End Coupling Workflow
    # -----------------------------------------------------------------------

    def couple_pbr_material_stack(
        self,
        meso_disp_mm: np.ndarray,
        micro_disp_mm: np.ndarray,
        zone_masks: Dict[str, np.ndarray],
        macro_disp_mm: Optional[np.ndarray] = None,
        weight_meso: float = 0.5,
        weight_micro: float = 0.35,
        weight_macro: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Executes complete multi-tier fusion and PBR material derivation.

        Returns
        -------
        dict with keys:
            'composite_displacement_mm': (H, W) float32 composite displacement
            'composite_normal_rgb'     : (H, W, 3) uint8 tangent normal map
            'cavity_ao_map'            : (H, W) float32 ambient occlusion
            'base_roughness'           : (H, W) float32 base stratum corneum roughness
            'coat_roughness'           : (H, W) float32 sebum oil coat sheen roughness
        """
        valid = zone_masks.get('valid')

        # 1. Composite displacement
        comp_disp = self.fuse_displacements(
            meso_disp_mm=meso_disp_mm,
            micro_disp_mm=micro_disp_mm,
            macro_disp_mm=macro_disp_mm,
            weight_macro=weight_macro,
            weight_meso=weight_meso,
            weight_micro=weight_micro,
            mask=valid,
        )
        # Enforce Rule 4 Neck Seam Contract Invariant (smooth Hermite taper + bitwise boundary pinning)
        transition_start = int(0.70 * self.resolution)
        bottom_start = int(0.85 * self.resolution)
        ramp_len = max(bottom_start - transition_start, 1)
        row_indices = np.arange(self.resolution, dtype=np.float32)[:, None]
        t = np.clip((bottom_start - row_indices) / float(ramp_len), 0.0, 1.0)
        smooth_falloff = t * t * (3.0 - 2.0 * t)
        comp_disp = comp_disp * smooth_falloff

        neck_pinning = zone_masks.get('neck_pinning', zone_masks.get('neck_collar'))
        if neck_pinning is not None:
            comp_disp = comp_disp * (1.0 - np.clip(neck_pinning, 0.0, 1.0))
        comp_disp[bottom_start:, :] = 0.0

        # 2. Tangent normal map
        normals_rgb = self.compute_tangent_normal_map(comp_disp)

        # 3. Micro-cavity ambient occlusion
        cavity_ao = self.compute_cavity_ao_map(comp_disp, mask=valid)

        # 4. Dual-lobe roughness maps
        roughness_dict = self.compute_dual_lobe_roughness(
            zone_masks=zone_masks,
            micro_disp_mm=micro_disp_mm,
        )

        return {
            'composite_displacement_mm': comp_disp,
            'composite_normal_rgb': normals_rgb,
            'cavity_ao_map': cavity_ao,
            'base_roughness': roughness_dict['base_roughness'],
            'coat_roughness': roughness_dict['coat_roughness'],
            'coat_weight': roughness_dict.get('coat_weight', np.zeros_like(comp_disp)),
        }

    # -----------------------------------------------------------------------
    # 6. Serialization
    # -----------------------------------------------------------------------

    def save_maps(
        self,
        coupled_result: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "film",
    ) -> Dict[str, Path]:
        """
        Saves all coupled PBR maps in lossless format to disk.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        disp_mm = coupled_result['composite_displacement_mm']
        normals_rgb = coupled_result['composite_normal_rgb']
        cavity_ao = coupled_result['cavity_ao_map']
        base_rough = coupled_result['base_roughness']
        coat_rough = coupled_result['coat_roughness']
        coat_weight = coupled_result.get('coat_weight', np.zeros_like(base_rough))

        # 16-bit uint displacement PNG (scale: self.max_scale_mm)
        norm_disp = np.clip((disp_mm / self.max_scale_mm + 1.0) * 0.5, 0.0, 1.0)
        disp_u16 = np.round(norm_disp * 65535.0).astype(np.uint16)

        disp_p = out_dir / f"{prefix}_displacement_16bit.png"
        normal_p = out_dir / f"{prefix}_normal.png"
        cavity_p = out_dir / f"{prefix}_cavity_ao.png"
        rough_base_p = out_dir / f"{prefix}_roughness_base.png"
        rough_coat_p = out_dir / f"{prefix}_roughness_coat.png"
        coat_weight_p = out_dir / f"{prefix}_coat_weight.png"

        cv2.imwrite(str(disp_p), disp_u16)
        cv2.imwrite(str(normal_p), cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(cavity_p), np.round(cavity_ao * 255.0).astype(np.uint8))
        cv2.imwrite(str(rough_base_p), np.round(base_rough * 255.0).astype(np.uint8))
        cv2.imwrite(str(rough_coat_p), np.round(coat_rough * 255.0).astype(np.uint8))
        cv2.imwrite(str(coat_weight_p), np.round(coat_weight * 255.0).astype(np.uint8))

        return {
            'displacement_png': disp_p,
            'normal_png': normal_p,
            'cavity_ao_png': cavity_p,
            'roughness_base_png': rough_base_p,
            'roughness_coat_png': rough_coat_p,
            'coat_weight_png': coat_weight_p,
        }
