from __future__ import annotations
"""
Stage 3 High-Frequency Detail GAN Inference Module.

Provides end-to-end inference using the trained U-Net Generator with multi-scale
skip connections and cross-attention bottleneck, synthesizing metric displacement
maps (16-bit uint PNG) and tangent-space normal maps for Unreal Engine 5.
"""
from typing import Dict, Optional, Union, Any, Tuple
from pathlib import Path
import json
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.rasterizer import (
    load_flame_uv_layout,
    compute_vertex_normals,
    rasterize_uv_maps
)


class DetailSynthesizer:
    """
    Production inference engine for Stage 3 High-Frequency Micro-Displacement.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        stats_path: Optional[Union[str, Path]] = None,
        uv_template_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for DetailSynthesizer inference.")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Resolve and validate model checkpoint
        ckpt_p = self._resolve_checkpoint(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(
                f"Stage 3 Detail GAN weights not found at: {ckpt_p}. "
                "Please ensure models_cache/stage3_detail/ema_generator.pt is present."
            )
        self.checkpoint_path = ckpt_p

        # 2. Resolve normalization statistics (metric conversion contract)
        stats_p = self._resolve_stats(stats_path)
        if not stats_p.exists():
            raise FileNotFoundError(
                f"Stage 3 normalization statistics not found at: {stats_p}. "
                "Required to convert normalized [-1, 1] output to metric millimeters."
            )
        with open(stats_p, 'r') as f:
            stats = json.load(f)
        self.p99_mm = float(stats.get('p99_mm', stats.get('scale', 1.1465)))
        self.stats = stats

        # 3. Load FLAME UV layout parameterization
        self.uv_coords, self.uv_faces = load_flame_uv_layout(
            str(uv_template_path) if uv_template_path else None
        )

        # 4. Initialize and load generator
        self.generator = DetailGenerator().to(self.device)
        self._load_weights(ckpt_p)
        self.generator.eval()

    def _resolve_checkpoint(self, path: Optional[Union[str, Path]]) -> Path:
        if path:
            return Path(path)
        root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            root / 'models_cache' / 'stage3_detail' / 'ema_generator.pt',
            root / 'models_cache' / 'stage3_detail' / 'generator_latest.pt',
            root / 'checkpoints' / 'stage3_detail' / 'ema_generator.pt',
            root / 'outputs' / 'kaggle_phase3_gan' / 'extracted' / 'ema_generator.pt',
            Path('/kaggle/working/checkpoints/stage3_detail/ema_generator.pt'),
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _resolve_stats(self, path: Optional[Union[str, Path]]) -> Path:
        if path:
            return Path(path)
        root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            root / 'models_cache' / 'stage3_detail' / 'normalization_stats.json',
            root / 'outputs' / 'kaggle_phase3_gan' / 'extracted' / 'normalization_stats.json',
            root / 'outputs' / 'uv_displacement_dataset_1024' / 'normalization_stats.json',
            Path('/kaggle/working/checkpoints/stage3_detail/normalization_stats.json'),
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _load_weights(self, path: Path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict) and 'ema_generator' in ckpt:
            state_dict = ckpt['ema_generator']
        elif isinstance(ckpt, dict) and 'generator' in ckpt:
            state_dict = ckpt['generator']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError(f"Unrecognized checkpoint format at {path}")

        # Strip module. prefix if trained under DistributedDataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            clean_state_dict[key] = v

        self.generator.load_state_dict(clean_state_dict, strict=False)

    def synthesize(
        self,
        neutral_vertices: np.ndarray,
        faces: np.ndarray,
        per_view_feats: Optional[Union[np.ndarray, torch.Tensor]] = None,
        beta: Optional[Union[np.ndarray, torch.Tensor]] = None,
        psi: Optional[Union[np.ndarray, torch.Tensor]] = None,
        resolution: int = 512
    ) -> Dict[str, Any]:
        """
        Synthesizes 16-bit metric displacement and tangent-space normal maps
        from canonical neutral geometry and multi-view perceptual features.
        """
        # 1. Compute vertex unit normals and rasterize UV spatial conditions
        vert_normals = compute_vertex_normals(neutral_vertices, faces)
        _, pos_map, norm_map, mask_map = rasterize_uv_maps(
            flame_verts_m=neutral_vertices,
            flame_normals=vert_normals,
            uv_coords=self.uv_coords,
            uv_faces=self.uv_faces,
            flame_faces=faces,
            resolution=resolution
        )

        # 2. Normalize position field to zero-mean unit-variance
        pos_norm = (pos_map - pos_map.mean(axis=(0, 1), keepdims=True)) / (
            pos_map.std(axis=(0, 1), keepdims=True) + 1e-8
        )
        pos_tensor = torch.from_numpy(pos_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        norm_tensor = torch.from_numpy(norm_map).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 3. Format conditioning tensors
        if per_view_feats is None:
            feats_tensor = torch.zeros(1, 1, 512, device=self.device)
        elif isinstance(per_view_feats, np.ndarray):
            f = per_view_feats if per_view_feats.ndim == 3 else per_view_feats[np.newaxis, ...]
            if f.ndim == 2:
                f = f[np.newaxis, ...]
            feats_tensor = torch.from_numpy(f).float().to(self.device)
        else:
            feats_tensor = per_view_feats.to(self.device)
            if feats_tensor.ndim == 2:
                feats_tensor = feats_tensor.unsqueeze(0)

        if beta is None:
            beta_tensor = torch.zeros(1, 300, device=self.device)
        elif isinstance(beta, np.ndarray):
            b = beta[np.newaxis, :] if beta.ndim == 1 else beta
            beta_tensor = torch.from_numpy(b).float().to(self.device)
        else:
            beta_tensor = beta.to(self.device)
            if beta_tensor.ndim == 1:
                beta_tensor = beta_tensor.unsqueeze(0)

        if psi is None:
            psi_tensor = torch.zeros(1, 100, device=self.device)
        elif isinstance(psi, np.ndarray):
            p = psi[np.newaxis, :] if psi.ndim == 1 else psi
            psi_tensor = torch.from_numpy(p).float().to(self.device)
        else:
            psi_tensor = psi.to(self.device)
            if psi_tensor.ndim == 1:
                psi_tensor = psi_tensor.unsqueeze(0)

        # 4. Forward execution
        with torch.no_grad():
            disp_pred = self.generator(pos_tensor, norm_tensor, feats_tensor, beta_tensor, psi_tensor)
            disp_norm = disp_pred.squeeze().cpu().numpy()

        # 5. Mask non-facial UV boundary pixels
        valid_mask = (mask_map > 0)
        disp_norm = disp_norm * valid_mask

        # 6. De-normalize to metric millimeters (Invariant: p99 scaling)
        disp_mm = disp_norm * self.p99_mm

        # 7. Encode 16-bit uint PNG (32768 corresponds to 0.0mm displacement)
        disp_uint16 = np.clip((disp_norm + 1.0) * 32767.5, 0, 65535).astype(np.uint16)

        # 8. Compute tangent-space normal map via spatial gradients
        normal_map_rgb = self._compute_tangent_normals(disp_mm, valid_mask)

        return {
            'disp_mm': disp_mm,
            'disp_norm': disp_norm,
            'disp_uint16': disp_uint16,
            'normal_map_rgb': normal_map_rgb,
            'mask': mask_map,
            'p99_mm': self.p99_mm,
            'resolution': resolution,
        }

    def _compute_tangent_normals(self, disp_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Derives tangent-space surface normal map from metric displacement gradients.
        Standard normal map color convention: flat surface is RGB(128, 128, 255).
        """
        grad_x = cv2.Sobel(disp_mm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disp_mm, cv2.CV_32F, 0, 1, ksize=3)

        # Scale factor mapping millimeter slope to tangent angle
        slope_scale = 1.5
        nx = -grad_x * slope_scale
        ny = -grad_y * slope_scale
        nz = np.ones_like(disp_mm)

        length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx, ny, nz = nx / length, ny / length, nz / length

        # Map [-1, 1] to [0, 255] RGB
        r = np.clip((nx + 1.0) * 127.5, 0, 255).astype(np.uint8)
        g = np.clip((ny + 1.0) * 127.5, 0, 255).astype(np.uint8)
        b = np.clip((nz + 1.0) * 127.5, 0, 255).astype(np.uint8)

        normal_rgb = np.stack([r, g, b], axis=-1)
        # Set invalid UV regions to neutral flat normal (128, 128, 255)
        normal_rgb[~mask] = [128, 128, 255]
        return normal_rgb

    @staticmethod
    def save_maps(
        synth_result: Dict[str, Any],
        output_dir: Union[str, Path],
        prefix: str = "head"
    ) -> Dict[str, str]:
        """
        Writes 16-bit displacement PNG, tangent normal PNG, and preview color images.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        disp_16_path = out_p / f"{prefix}_displacement_16bit.png"
        cv2.imwrite(str(disp_16_path), synth_result['disp_uint16'])

        norm_path = out_p / f"{prefix}_normal_map.png"
        cv2.imwrite(str(norm_path), cv2.cvtColor(synth_result['normal_map_rgb'], cv2.COLOR_RGB2BGR))

        # 8-bit preview colormap
        preview_path = out_p / f"{prefix}_displacement_preview.png"
        disp_norm = synth_result['disp_norm']
        disp_vis = ((disp_norm + 1.0) * 127.5).astype(np.uint8)
        color_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA)
        color_vis[synth_result['mask'] == 0] = 0
        cv2.imwrite(str(preview_path), color_vis)

        return {
            'displacement_16bit': str(disp_16_path.resolve()),
            'normal_map': str(norm_path.resolve()),
            'displacement_preview': str(preview_path.resolve()),
        }
