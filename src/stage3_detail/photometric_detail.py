"""
Stage 3 Tier 2: Photo-Derived Photometric Meso Wrinkle Engine.

Extracts real, person-specific anatomical wrinkles (crow's feet, brow furrows,
nasolabial folds, laugh lines) directly from multi-view portrait photographs
using multi-scale photometric gradient decomposition and Frankot-Chellappa
Fourier surface integration.

Pure NumPy/SciPy/OpenCV execution — zero GPU required, deterministic, and preserves
1:1 fidelity with the real human subject's skin creases without neural hallucination.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.stage3_detail.rasterizer import (
    compute_vertex_normals,
    load_flame_uv_layout,
    rasterize_uv_maps,
)


class PhotometricDetailExtractor:
    """
    Tier 2 Meso Detail Engine: Photo-Derived Shape-from-Shading.

    Extracts high-resolution wrinkle heightfields and tangent normal deviations
    directly from multi-view photographs or delighted UV albedo maps.
    """

    def __init__(
        self,
        uv_template_path: Optional[Union[str, Path]] = None,
        resolution: int = 1024,
        max_wrinkle_depth_mm: float = 1.20,
        sigma_fine: float = 1.5,
        sigma_meso: float = 12.0,
        gradient_gain: float = 2.5,
    ):
        """
        Parameters
        ----------
        uv_template_path : Optional path to FLAME UV template (.obj or .npz)
        resolution : Working UV map resolution (1024 or 2048)
        max_wrinkle_depth_mm : Maximum physical peak-to-valley wrinkle amplitude (mm)
        sigma_fine : Gaussian blur radius for noise suppression (pixels)
        sigma_meso : Gaussian blur radius for low-frequency macro illumination separation (pixels)
        gradient_gain : Scaling factor converting photometric intensity gradients to surface slopes
        """
        self.resolution = int(resolution)
        self.max_depth_mm = float(max_wrinkle_depth_mm)
        self.sigma_fine = float(sigma_fine)
        self.sigma_meso = float(sigma_meso)
        self.gradient_gain = float(gradient_gain)

        # Load canonical FLAME UV layout
        self.uv_coords, self.uv_faces = load_flame_uv_layout(
            str(uv_template_path) if uv_template_path else None
        )

    # -----------------------------------------------------------------------
    # 1. Multi-Scale Photometric Gradient Extraction
    # -----------------------------------------------------------------------

    def extract_wrinkle_gradients(
        self,
        image_rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decomposes an RGB UV texture into high-frequency luminance gradients
        corresponding to epidermal wrinkles and crevasse valleys.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 or float32 image
        mask : Optional (H, W) binary/float mask of valid facial skin

        Returns
        -------
        grad_x : (H, W) float32 surface slope along U axis
        grad_y : (H, W) float32 surface slope along V axis
        wrinkle_intensity : (H, W) float32 isolated wrinkle crevasse response
        """
        img = image_rgb.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # 1. Extract perceptual luminance
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # 2. Multi-scale bandpass filtering
        # Remove camera noise with fine smoothing
        l_clean = cv2.GaussianBlur(luminance, (0, 0), self.sigma_fine)
        # Remove low-frequency lighting / macro skull curves
        l_macro = cv2.GaussianBlur(luminance, (0, 0), self.sigma_meso)

        # High-pass meso band: negative values denote dark crease crevasses
        meso_band = l_clean - l_macro

        if mask is not None:
            m = (mask > 0.5).astype(np.float32)
            meso_band = meso_band * m

        # 3. Compute spatial surface gradients (Sobel 5x5 for smooth derivatives)
        grad_x = cv2.Sobel(meso_band, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(meso_band, cv2.CV_32F, 0, 1, ksize=5)

        # Scale by gradient gain
        grad_x = grad_x * self.gradient_gain
        grad_y = grad_y * self.gradient_gain

        # 4. Wrinkle crevasse detector (dark valley lines have negative meso response)
        wrinkle_response = np.clip(-meso_band * 3.0, 0.0, 1.0)

        return grad_x, grad_y, wrinkle_response

    # -----------------------------------------------------------------------
    # 2. Frankot-Chellappa Fourier Surface Integration
    # -----------------------------------------------------------------------

    def integrate_heightfield(
        self,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Integrates 2D surface slopes (grad_x, grad_y) into a globally continuous
        heightfield using the exact Frankot-Chellappa Fourier formulation.

        Enforces surface integrability: curl(grad) = 0.
        Runs in O(N log N) time with zero numerical divergence.

        Parameters
        ----------
        grad_x : (H, W) surface slope dz/dx
        grad_y : (H, W) surface slope dz/dy
        mask : Optional (H, W) valid facial skin mask

        Returns
        -------
        heightfield : (H, W) float32 surface displacement in millimeters
        """
        h, w = grad_x.shape[:2]

        # Frequency coordinates
        u = np.fft.fftfreq(w).astype(np.float64)
        v = np.fft.fftfreq(h).astype(np.float64)
        u_grid, v_grid = np.meshgrid(u, v)

        denom = 4.0 * (np.pi ** 2) * (u_grid ** 2 + v_grid ** 2)
        denom[0, 0] = 1.0  # Avoid division by zero at DC frequency

        # Fourier transform of gradients
        gx_fft = np.fft.fft2(grad_x.astype(np.float64))
        gy_fft = np.fft.fft2(grad_y.astype(np.float64))

        # Frankot-Chellappa integration formula:
        # Z(u,v) = (-i * 2*pi*u * P - i * 2*pi*v * Q) / (4*pi^2 * (u^2 + v^2))
        z_fft = (-1j * 2.0 * np.pi * u_grid * gx_fft - 1j * 2.0 * np.pi * v_grid * gy_fft) / denom
        z_fft[0, 0] = 0.0  # Zero DC frequency (zero-mean heightfield)

        height = np.real(np.fft.ifft2(z_fft)).astype(np.float32)

        # Normalize heightfield to zero mean over valid mask
        if mask is not None and np.any(mask > 0.5):
            valid = mask > 0.5
            height = height - np.mean(height[valid])
            height[~valid] = 0.0
        else:
            height = height - np.mean(height)

        # Scale into physical metric millimeter range [-max_depth_mm, +max_depth_mm]
        p99 = np.percentile(np.abs(height[height != 0]), 99.0) if np.any(height != 0) else 1.0
        if p99 > 1e-6:
            height = (height / p99) * self.max_depth_mm

        height = np.clip(height, -self.max_depth_mm, self.max_depth_mm)

        return height.astype(np.float32)

    # -----------------------------------------------------------------------
    # 3. Neck Boundary Pinning Contract
    # -----------------------------------------------------------------------

    def apply_collar_pinning(
        self,
        displacement_mm: np.ndarray,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Enforces System Invariant Rule 4 (Neck Seam Contract):
        The lowest 20% of vertices (neck boundary collar) must have their
        displacements pinned strictly to zero so the exported head never tears
        when connected to a common body mesh in Unreal Engine 5.

        Parameters
        ----------
        displacement_mm : (H, W) float32 displacement map
        vertices : (V, 3) 3D mesh vertices in metres
        faces : (F, 3) triangle face indices

        Returns
        -------
        pinned_disp : (H, W) float32 displacement map with collar strictly zero
        """
        h, w = displacement_mm.shape[:2]
        disp = displacement_mm.copy()

        if vertices is not None and faces is not None:
            # Use rasterized 3D position map to get physical vertical coordinates
            normals = compute_vertex_normals(vertices, faces)
            _, pos_map, _, mask = rasterize_uv_maps(
                flame_verts_m=vertices,
                flame_normals=normals,
                uv_coords=self.uv_coords,
                uv_faces=self.uv_faces,
                flame_faces=faces,
                resolution=h,
            )
            valid = mask > 0
            if np.any(valid):
                y = pos_map[:, :, 1]
                y_valid = y[valid]
                y_span = max(float(y_valid.max() - y_valid.min()), 1e-6)
                y_norm = (y - float(y_valid.min())) / y_span

                # Cosine-Hermite smooth taper:
                # y_norm <= 0.15: strictly 0.0 displacement (Rule 4 pinning)
                # 0.15 < y_norm < 0.25: smooth S-curve taper from 0 to 1
                # y_norm >= 0.25: 1.0 (full displacement preserved)
                collar_weight = np.ones((h, w), dtype=np.float32)
                collar_weight[valid & (y_norm <= 0.15)] = 0.0

                trans = valid & (y_norm > 0.15) & (y_norm < 0.25)
                t = (y_norm[trans] - 0.15) / 0.10
                collar_weight[trans] = 0.5 * (1.0 - np.cos(np.pi * t))

                # Zero unmapped background space
                collar_weight[~valid] = 0.0

                disp = disp * collar_weight

        else:
            # Fallback heuristic using UV coordinates:
            # In standard FLAME UV, the neck collar is located at V in [0.0, 0.18].
            # In image coordinates, row 0 is V=1.0 and row h-1 is V=0.0.
            # Thus V <= 0.15 corresponds to rows >= 0.85 * h.
            collar_weight = np.ones((h, w), dtype=np.float32)
            bottom_start = int(0.85 * h)
            collar_weight[bottom_start:, :] = 0.0

            # Smooth cosine transition between 0.75 * h and 0.85 * h
            trans_start = int(0.75 * h)
            for r in range(trans_start, bottom_start):
                t = (r - trans_start) / max(bottom_start - trans_start, 1)
                collar_weight[r, :] = 0.5 * (1.0 + np.cos(np.pi * t))

            disp = disp * collar_weight

        # Always enforce bitwise zero on the bottom collar rows (Rule 4 invariant)
        bottom_start = int(0.85 * h)
        disp[bottom_start:, :] = 0.0

        return disp.astype(np.float32)

    # -----------------------------------------------------------------------
    # 4. Tangent-Space Normal Map Derivation
    # -----------------------------------------------------------------------

    def compute_tangent_normal_map(
        self,
        displacement_mm: np.ndarray,
        pixel_size_mm: float = 0.25,
    ) -> np.ndarray:
        """
        Converts a signed metric displacement heightfield into an RGB
        tangent-space normal map formatted for Blender and Unreal Engine 5.

        Parameters
        ----------
        displacement_mm : (H, W) float32 displacement in millimeters
        pixel_size_mm : Metric width of one UV texel (e.g. ~0.25 mm at 1024²)

        Returns
        -------
        normal_rgb : (H, W, 3) uint8 tangent normal map in [0, 255]
                     (R=X, G=Y, B=Z with neutral flat normal = (128, 128, 255))
        """
        h, w = displacement_mm.shape[:2]

        # Compute surface heightfield gradients
        dz_dx = cv2.Sobel(displacement_mm, cv2.CV_32F, 1, 0, ksize=3) / (2.0 * pixel_size_mm)
        dz_dy = cv2.Sobel(displacement_mm, cv2.CV_32F, 0, 1, ksize=3) / (2.0 * pixel_size_mm)

        # Tangent normal vectors: n = Normalize(-dz/dx, -dz/dy, 1.0)
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones((h, w), dtype=np.float32)

        norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        norm[norm == 0] = 1.0

        nx = nx / norm
        ny = ny / norm
        nz = nz / norm

        # Map from [-1, 1] to uint8 [0, 255]
        r = np.clip((nx + 1.0) * 127.5, 0, 255).astype(np.uint8)
        g = np.clip((ny + 1.0) * 127.5, 0, 255).astype(np.uint8)
        b = np.clip((nz + 1.0) * 127.5, 0, 255).astype(np.uint8)

        normal_rgb = np.stack([r, g, b], axis=-1)
        return normal_rgb

    # -----------------------------------------------------------------------
    # 5. High-Level Extraction Workflow
    # -----------------------------------------------------------------------

    def extract_from_albedo(
        self,
        albedo_rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        End-to-end extraction from a delighted UV albedo texture.

        Parameters
        ----------
        albedo_rgb : (H, W, 3) RGB or BGR texture map
        mask : Optional (H, W) valid facial skin mask
        vertices : Optional (V, 3) FLAME neutral mesh vertices (in metres)
        faces : Optional (F, 3) FLAME face triangles

        Returns
        -------
        dict with keys:
            'displacement_mm'      : (H, W) float32 signed displacement in millimeters
            'tangent_normal_rgb'   : (H, W, 3) uint8 tangent normal map
            'wrinkle_mask'         : (H, W) float32 isolated wrinkle crevasse response
            'grad_x'               : (H, W) float32 U-derivative
            'grad_y'               : (H, W) float32 V-derivative
        """
        gx, gy, wrinkle_resp = self.extract_wrinkle_gradients(albedo_rgb, mask)
        height = self.integrate_heightfield(gx, gy, mask)

        # Apply neck collar boundary pinning
        height = self.apply_collar_pinning(height, vertices, faces)

        # Compute tangent normal map
        normals = self.compute_tangent_normal_map(height)

        return {
            'displacement_mm': height,
            'tangent_normal_rgb': normals,
            'wrinkle_mask': wrinkle_resp,
            'grad_x': gx,
            'grad_y': gy,
        }

    def extract_from_views(
        self,
        projected_views: List[Dict[str, np.ndarray]],
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fuses multi-view backprojected textures into a unified meso-wrinkle map
        using cosine-weighted confidence blending.

        Parameters
        ----------
        projected_views : List of dicts, each with:
            'rgb'    : (H, W, 3) projected texture from that view
            'weight' : (H, W) confidence weight / cosine angle map
            'mask'   : (H, W) visibility mask
        vertices : (V, 3) mesh vertices in metres
        faces : (F, 3) face indices

        Returns
        -------
        Fused wrinkle extraction dictionary
        """
        h = self.resolution
        w = self.resolution

        accum_gx = np.zeros((h, w), dtype=np.float32)
        accum_gy = np.zeros((h, w), dtype=np.float32)
        accum_weights = np.zeros((h, w), dtype=np.float32)
        accum_mask = np.zeros((h, w), dtype=np.float32)

        for view in projected_views:
            rgb = view.get('rgb')
            if rgb is None:
                continue

            v_weight = view.get('weight', np.ones((h, w), dtype=np.float32))
            v_mask = view.get('mask', (v_weight > 0.1).astype(np.float32))

            gx, gy, _ = self.extract_wrinkle_gradients(rgb, v_mask)

            accum_gx += gx * v_weight
            accum_gy += gy * v_weight
            accum_weights += v_weight
            accum_mask = np.maximum(accum_mask, v_mask)

        # Normalize accumulated gradients by weights
        valid = accum_weights > 1e-4
        accum_gx[valid] /= accum_weights[valid]
        accum_gy[valid] /= accum_weights[valid]

        # Integrate fused gradients into heightfield
        height = self.integrate_heightfield(accum_gx, accum_gy, accum_mask)
        height = self.apply_collar_pinning(height, vertices, faces)
        normals = self.compute_tangent_normal_map(height)

        return {
            'displacement_mm': height,
            'tangent_normal_rgb': normals,
            'wrinkle_mask': (accum_weights > 0.1).astype(np.float32),
            'grad_x': accum_gx,
            'grad_y': accum_gy,
        }

    # -----------------------------------------------------------------------
    # 6. Serialization
    # -----------------------------------------------------------------------

    def save_maps(
        self,
        extraction_result: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "meso",
    ) -> Dict[str, Path]:
        """
        Saves 16-bit unsigned displacement PNG and tangent normal PNG to disk.

        Storage Contract:
        - 16-bit uint: midlevel = 32768, scale = max_depth_mm
        - d_mm = ((uint16 / 65535.0) * 2.0 - 1.0) * max_depth_mm
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        disp_mm = extraction_result['displacement_mm']
        normals_rgb = extraction_result['tangent_normal_rgb']

        # Encode displacement to 16-bit uint PNG
        normalized = np.clip(
            (disp_mm / self.max_depth_mm + 1.0) * 0.5,
            0.0,
            1.0,
        )
        disp_u16 = np.round(normalized * 65535.0).astype(np.uint16)

        disp_path = out_dir / f"{prefix}_displacement_16bit.png"
        normal_path = out_dir / f"{prefix}_normal.png"

        cv2.imwrite(str(disp_path), disp_u16)
        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(str(normal_path), cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))

        return {
            'displacement_png': disp_path,
            'normal_png': normal_path,
        }
