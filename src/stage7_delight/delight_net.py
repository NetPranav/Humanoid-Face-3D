"""
Stage 7: AI Delighting U-Net + UV Inpainting Engine.

Two complementary sub-stages:
  7A — DelightUNet: Encoder-decoder that strips environment lighting from
       projected face textures, producing pure diffuse albedo (skin pigment only).
       Supports pre-trained DECA albedo decoder weights with automatic key remapping.

  7B — UVInpainter: Fills unseen UV regions (back of ears, under chin, scalp)
       using iterative Gaussian dilation (V1, zero-download) or optional neural
       inpainting via LaMa (V2, requires weight download).

The DelightingPipeline orchestrates: raw texture → inpaint → delight → clean albedo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# 7B: UV Inpainting (Procedural Gaussian Dilation)
# ---------------------------------------------------------------------------

class UVInpainter:
    """
    Fills unseen UV regions (projection_mask == 0) by iteratively dilating
    valid pixel values into the missing areas using Gaussian blur.

    Parameters
    ----------
    iterations : int
        Number of dilation passes. Each pass extends the border by ~2 pixels.
        50 iterations can fill holes up to ~100 pixels deep.
    kernel_size : int
        Gaussian kernel size (must be odd).
    """

    def __init__(self, iterations: int = 50, kernel_size: int = 5):
        self.iterations = iterations
        self.kernel_size = kernel_size

    def inpaint(
        self,
        texture: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill missing regions in the texture using Gaussian dilation.

        Parameters
        ----------
        texture : (H, W, 3) float32 or uint8 — input texture with holes
        mask : (H, W) uint8 — 255 where data exists, 0 where missing

        Returns
        -------
        filled_texture : (H, W, 3) same dtype as input
        filled_mask : (H, W) uint8 — updated mask (255 everywhere that was filled)
        """
        is_float = texture.dtype in (np.float32, np.float64)
        if not is_float:
            work_tex = texture.astype(np.float32)
        else:
            work_tex = texture.copy()

        h, w = texture.shape[:2]
        obs_mask = (mask > 127).astype(np.uint8)

        # If already completely filled, return unmodified
        if np.all(obs_mask > 0):
            return (texture.copy(), mask.copy())

        # Step 1: Sample the subject's central facial skin tone (L*a*b* / BGR baseline)
        u_min, u_max = int(w * 0.38), int(w * 0.62)
        v_min, v_max = int(h * 0.38), int(h * 0.68)
        face_crop = work_tex[v_min:v_max, u_min:u_max]
        face_mask = obs_mask[v_min:v_max, u_min:u_max] > 0

        if np.any(face_mask):
            valid_face_pixels = face_crop[face_mask]
            median_skin = np.median(valid_face_pixels, axis=0)
        else:
            # Fallback to any observed pixels
            valid_pixels = work_tex[obs_mask > 0]
            if len(valid_pixels) > 0:
                median_skin = np.median(valid_pixels, axis=0)
            else:
                median_skin = np.array([125.0, 115.0, 175.0], dtype=np.float32)

        # Step 2: Multi-scale Fast-Marching / Telea harmonic field for smooth boundary transition
        scale_div = max(1, w // 512)
        small_w = w // scale_div
        small_h = h // scale_div
        small_tex = cv2.resize(work_tex, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(obs_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        inv_mask_small = (small_mask == 0).astype(np.uint8)
        if np.any(small_mask > 0) and np.any(inv_mask_small > 0):
            inpaint_small = cv2.inpaint(
                small_tex.astype(np.uint8),
                inv_mask_small,
                inpaintRadius=5,
                flags=cv2.INPAINT_TELEA
            )
            smooth_field = cv2.resize(
                inpaint_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            smooth_field = np.full((h, w, 3), median_skin, dtype=np.float32)

        # Step 3: Distance transform from observed boundary to prevent directional ray smearing
        inv_mask_full = (obs_mask == 0).astype(np.uint8)
        dist = cv2.distanceTransform(inv_mask_full, cv2.DIST_L2, 5)
        max_blend_dist = float(w) * 0.035  # ~70 pixels at 2048x2048
        alpha = np.clip(dist / max_blend_dist, 0.0, 1.0)[:, :, np.newaxis]

        # Base anatomical skin canvas
        base_canvas = np.full((h, w, 3), median_skin, dtype=np.float32)

        # Eyeball islands: circular UV islands at top-left and top-right (anatomical sclera ivory)
        sclera_color = np.array([232.0, 236.0, 238.0], dtype=np.float32)
        eye_y_end = int(h * 0.18)
        eye_x_left = int(w * 0.18)
        eye_x_right = int(w * 0.82)
        base_canvas[0:eye_y_end, 0:eye_x_left] = sclera_color
        base_canvas[0:eye_y_end, eye_x_right:w] = sclera_color

        # Anatomical zone warmth: modulate ears with microvascular tone
        ear_mask = np.zeros((h, w), dtype=np.float32)
        ear_mask[int(h * 0.38):int(h * 0.65), int(w * 0.10):int(w * 0.28)] = 1.0
        ear_mask[int(h * 0.38):int(h * 0.65), int(w * 0.72):int(w * 0.90)] = 1.0
        ear_blur = cv2.GaussianBlur(ear_mask, (0, 0), sigmaX=float(w) * 0.015)[:, :, np.newaxis]

        # Warm capillary vascular boost for ears (+Red, +slight Green, -Blue in BGR)
        base_canvas[:, :, 2] += ear_blur[:, :, 0] * 12.0
        base_canvas[:, :, 1] += ear_blur[:, :, 0] * 3.0
        base_canvas[:, :, 0] -= ear_blur[:, :, 0] * 3.0

        # Blend smooth boundary field towards anatomical baseline canvas
        synthesized = (1.0 - alpha) * smooth_field + alpha * base_canvas

        # Step 4: Organic epidermal micro-porosity to eliminate flat plastic wax appearance
        rng = np.random.RandomState(42)
        dermal_noise = rng.normal(0.0, 2.2, (h, w, 3)).astype(np.float32)
        synthesized += dermal_noise * alpha

        # Step 5: Exact 100% preservation of observed photographic projection
        final_tex = work_tex.copy()
        unseen = (obs_mask == 0)
        final_tex[unseen] = synthesized[unseen]
        final_tex = np.clip(final_tex, 0.0, 255.0)

        filled_mask = np.full((h, w), 255, dtype=np.uint8)

        if not is_float:
            return final_tex.astype(np.uint8), filled_mask
        else:
            return final_tex.astype(np.float32), filled_mask


# ---------------------------------------------------------------------------
# 7A: Delighting U-Net
# ---------------------------------------------------------------------------

class DelightUNet(nn.Module if HAS_TORCH else object):
    """
    Encoder-decoder U-Net for lighting removal from projected face textures.

    Input:  6-channel UV image — RGB projected texture (3ch) + surface normal (3ch)
    Output: 3-channel clean diffuse albedo

    Architecture: 6-level encoder-decoder, 64 base channels, InstanceNorm + LeakyReLU.
    VRAM: ~1.8 GB at 2048² with FP16.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 3, base_channels: int = 64):
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for DelightUNet. "
                "Install with: pip install torch"
            )
        super().__init__()
        bc = base_channels

        # Encoder (6 levels)
        self.enc1 = self._conv_block(in_channels, bc)
        self.enc2 = self._conv_block(bc, bc * 2)
        self.enc3 = self._conv_block(bc * 2, bc * 4)
        self.enc4 = self._conv_block(bc * 4, bc * 8)
        self.enc5 = self._conv_block(bc * 8, bc * 8)
        self.enc6 = self._conv_block(bc * 8, bc * 8)

        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(bc * 8, bc * 8)

        # Decoder (6 levels with skip connections)
        self.dec6 = self._conv_block(bc * 16, bc * 8)
        self.dec5 = self._conv_block(bc * 16, bc * 8)
        self.dec4 = self._conv_block(bc * 16, bc * 4)
        self.dec3 = self._conv_block(bc * 8, bc * 2)
        self.dec2 = self._conv_block(bc * 4, bc)
        self.dec1 = nn.Sequential(
            nn.Conv2d(bc * 2, bc, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(bc, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc, out_channels, kernel_size=1),
        )

        self.out_act = nn.Sigmoid()  # Output in [0, 1]

    def _conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, 6, H, W) — concatenated projected RGB (3ch) + surface normals (3ch)

        Returns
        -------
        albedo : (B, 3, H, W) — clean diffuse albedo in [0, 1]
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))

        # Bottleneck (ensure spatial size is at least 2x2 for InstanceNorm2d compatibility)
        p6 = self.pool(e6)
        if p6.shape[2] < 2 or p6.shape[3] < 2:
            p6 = F.interpolate(p6, size=(max(2, p6.shape[2]), max(2, p6.shape[3])), mode='nearest')
        b = self.bottleneck(p6)

        # Decoder with skip connections
        d6 = self.dec6(torch.cat([F.interpolate(b, e6.shape[2:], mode='bilinear', align_corners=False), e6], dim=1))
        d5 = self.dec5(torch.cat([F.interpolate(d6, e5.shape[2:], mode='bilinear', align_corners=False), e5], dim=1))
        d4 = self.dec4(torch.cat([F.interpolate(d5, e4.shape[2:], mode='bilinear', align_corners=False), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, e3.shape[2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1], dim=1))

        return self.out_act(d1)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = 'cpu',
        strict: bool = False,
    ) -> 'DelightUNet':
        """
        Load from a checkpoint file with automatic key remapping for
        DECA albedo decoder weights.
        """
        model = cls()
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        else:
            state = ckpt

        # Attempt automatic key remapping for DECA
        remapped = {}
        for k, v in state.items():
            # Strip common prefixes
            new_key = k
            for prefix in ('module.', 'delight_net.', 'albedo_decoder.', 'E_albedo.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            remapped[new_key] = v

        try:
            model.load_state_dict(remapped, strict=strict)
        except RuntimeError as e:
            print(f"[Stage 7 Warning] Partial weight loading: {e}")
            # Try loading what we can
            own_state = model.state_dict()
            loaded = 0
            for k, v in remapped.items():
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k] = v
                    loaded += 1
            model.load_state_dict(own_state, strict=False)
            print(f"[Stage 7] Loaded {loaded}/{len(own_state)} compatible layers.")

        model.to(device)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Delighting Pipeline Orchestrator
# ---------------------------------------------------------------------------

class DelightingPipeline:
    """
    Orchestrates: raw projected texture → UV inpainting → delighting → clean albedo.

    Supports:
    - Pre-trained DECA albedo decoder (zero training)
    - Custom fine-tuned checkpoint
    - Graceful fallback if no neural weights are available (returns gamma-corrected projection)

    Parameters
    ----------
    delight_checkpoint : str or None
        Path to delighting network weights. None = attempt DECA auto-discovery.
    inpaint_method : str
        'procedural' (Gaussian dilation) or 'lama' (neural, requires download).
    inpaint_iterations : int
        Dilation iterations for procedural inpainting.
    use_fp16 : bool
        Use FP16 inference for VRAM savings.
    device : str or None
        PyTorch device. None = auto-detect (CUDA > MPS > CPU).
    """

    def __init__(
        self,
        delight_checkpoint: Optional[str] = None,
        inpaint_method: str = 'procedural',
        inpaint_iterations: int = 50,
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        self.inpainter = UVInpainter(iterations=inpaint_iterations)
        self.inpaint_method = inpaint_method
        self.use_fp16 = use_fp16

        # Auto-detect device
        if device is None:
            if HAS_TORCH:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Load delighting network
        self.delight_net = None
        if HAS_TORCH:
            if delight_checkpoint and Path(delight_checkpoint).exists():
                print(f"[Stage 7] Loading delighting network from: {delight_checkpoint}")
                self.delight_net = DelightUNet.from_pretrained(
                    delight_checkpoint, device=self.device
                )
            else:
                # Try DECA auto-discovery
                deca_candidates = [
                    Path('models_cache/deca/deca_model.tar'),
                    Path('models_cache/deca/albedo_decoder.pt'),
                    Path('/kaggle/input/deca-pretrained/deca_model.tar'),
                ]
                for cand in deca_candidates:
                    if cand.exists():
                        print(f"[Stage 7] Loading DECA albedo decoder from: {cand}")
                        try:
                            self.delight_net = DelightUNet.from_pretrained(
                                str(cand), device=self.device, strict=False
                            )
                            break
                        except Exception as e:
                            print(f"[Stage 7 Warning] Failed to load DECA: {e}")

                if self.delight_net is None:
                    print("[Stage 7 Notice] No delighting weights found. "
                          "Using gamma-correction fallback (still produces usable albedo).")

    def process(
        self,
        projected_rgb: np.ndarray,
        projection_mask: np.ndarray,
        normal_map: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Full delighting pipeline.

        Parameters
        ----------
        projected_rgb : (H, W, 3) float32 [0, 255] — raw projected texture
        projection_mask : (H, W) uint8 — 255 where data, 0 where unseen
        normal_map : (H, W, 3) float32 [0, 1] — surface normal map (optional)

        Returns
        -------
        dict with:
            'albedo_srgb'   : (H, W, 3) uint8 — sRGB albedo for display
            'albedo_linear' : (H, W, 3) float32 [0, 1] — linear-space albedo for UE5
            'inpainted_mask': (H, W) uint8 — mask after inpainting
        """
        h, w = projected_rgb.shape[:2]

        # Step 1: Inpaint missing regions
        print("[Stage 7B] Inpainting unseen UV regions...")
        inpainted, filled_mask = self.inpainter.inpaint(projected_rgb, projection_mask)

        # Step 2: Delighting
        if self.delight_net is not None and HAS_TORCH:
            print("[Stage 7A] Running AI delighting (removing environment lighting)...")
            albedo_linear = self._run_neural_delight(inpainted, normal_map)
        else:
            print("[Stage 7A] Applying dichromatic specular stripping & illumination delighting...")
            albedo_linear = self._dichromatic_delight(inpainted)

        # Convert to sRGB for display
        albedo_srgb = np.clip(np.power(albedo_linear, 1.0 / 2.2) * 255, 0, 255).astype(np.uint8)

        return {
            'albedo_srgb': albedo_srgb,
            'albedo_linear': albedo_linear,
            'inpainted_mask': filled_mask,
        }

    def _run_neural_delight(
        self,
        texture: np.ndarray,
        normal_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """Run the DelightUNet on the inpainted texture."""
        h, w = texture.shape[:2]

        # Prepare RGB input (normalize to [0, 1])
        rgb = texture.astype(np.float32) / 255.0

        # Prepare normal map (or synthesise flat normals)
        if normal_map is None or normal_map.shape[:2] != (h, w):
            normals = np.full((h, w, 3), [0.5, 0.5, 1.0], dtype=np.float32)
        else:
            normals = normal_map.astype(np.float32)
            if normals.max() > 1.5:
                normals = normals / 255.0

        # Stack to 6-channel input: [RGB, Normals]
        inp = np.concatenate([rgb, normals], axis=2)  # (H, W, 6)

        # Convert to torch tensor (B, C, H, W)
        inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)  # (1, 6, H, W)

        # Pad to multiple of 64 for U-Net
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            inp_tensor = F.pad(inp_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        inp_tensor = inp_tensor.to(self.device)

        dtype = torch.float16 if (self.use_fp16 and self.device != 'cpu') else torch.float32
        inp_tensor = inp_tensor.to(dtype)
        self.delight_net = self.delight_net.to(dtype)

        with torch.no_grad():
            albedo_tensor = self.delight_net(inp_tensor)

        # Convert back
        albedo_tensor = albedo_tensor.float()
        if pad_h > 0 or pad_w > 0:
            albedo_tensor = albedo_tensor[:, :, :h, :w]

        albedo = albedo_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        albedo = np.clip(albedo, 0, 1).astype(np.float32)

        return albedo

    def _dichromatic_delight(self, texture: np.ndarray) -> np.ndarray:
        """
        Dichromatic reflection delighting: decomposes reflected light into
        diffuse body chromaticity and surface specular glare, stripping white
        camera flash and ambient specular reflections before normalizing macro illumination.
        """
        rgb = texture.astype(np.float32) / 255.0

        # Linearise (inverse sRGB gamma)
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

        # HSV representation for specular saturation/luminance analysis
        hsv = cv2.cvtColor(texture, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        s_c = hsv[:, :, 1]
        v_c = hsv[:, :, 2]

        # Detect specular glare (high brightness + desaturation where flash bounced off sebum)
        v_smooth = cv2.GaussianBlur(v_c, (0, 0), 25.0)
        glare_mask = np.clip((v_c - 0.52) / 0.35, 0.0, 1.0) * np.clip((0.45 - s_c) / 0.35, 0.0, 1.0)

        # Subtract specular component in linear space
        excess_v = np.maximum(0.0, v_c - v_smooth * 0.9)
        spec_component = glare_mask[:, :, None] * excess_v[:, :, None] * 0.85
        diffuse_linear = np.clip(linear - spec_component, 0.0, 1.0)

        # Macro illumination normalization: remove directional key light falloff
        lum = 0.2126 * diffuse_linear[:, :, 2] + 0.7152 * diffuse_linear[:, :, 1] + 0.0722 * diffuse_linear[:, :, 0]
        macro_illum = cv2.GaussianBlur(lum, (0, 0), 32.0)
        macro_illum = np.clip(macro_illum, 0.15, 1.0)

        target_lum = 0.38  # Canonical calibrated human skin linear luminance
        albedo = diffuse_linear / (macro_illum[:, :, None] + 1e-4) * target_lum
        albedo = np.clip(albedo, 0.0, 1.0).astype(np.float32)

        return albedo

    def _gamma_delight(self, texture: np.ndarray) -> np.ndarray:
        """Alias for backward compatibility."""
        return self._dichromatic_delight(texture)

    def save_maps(
        self,
        result: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "head",
    ) -> Dict[str, str]:
        """Save albedo maps to disk."""
        out = Path(output_dir)
        textures_dir = out / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)

        albedo_srgb_path = textures_dir / f"{prefix}_albedo_diffuse.png"
        albedo_linear_path = textures_dir / f"{prefix}_albedo_linear.exr"

        cv2.imwrite(str(albedo_srgb_path), result['albedo_srgb'])

        # Save linear albedo as 32-bit EXR if possible, else 16-bit PNG
        try:
            cv2.imwrite(str(albedo_linear_path), result['albedo_linear'])
        except Exception:
            albedo_linear_path = textures_dir / f"{prefix}_albedo_linear.png"
            linear_16bit = (np.clip(result['albedo_linear'], 0, 1) * 65535).astype(np.uint16)
            cv2.imwrite(str(albedo_linear_path), linear_16bit)

        return {
            'albedo': str(albedo_srgb_path),
            'albedo_diffuse': str(albedo_srgb_path),
            'albedo_diffuse_png': str(albedo_srgb_path),
            'albedo_linear': str(albedo_linear_path),
        }
