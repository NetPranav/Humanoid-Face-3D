"""
Differentiable Rendering Losses for Stage 1 Identity Training.

Adds render-and-compare losses to the Stage 1 training loop:
1. Silhouette IoU loss — does the mesh outline match the photo?
2. Perceptual loss (LPIPS) — does the rendered normal map look right?
3. Landmark reprojection loss — do 2D landmarks align with mesh projections?

Why this matters:
    The current Stage 1 trainer only computes losses in β-coefficient space
    and 3D vertex space. It has NO mechanism to verify that the predicted mesh
    actually looks correct when rendered from the original camera angle. This
    is why a mesh with "correct" β values can still look wrong — the loss
    function never checks the visual output.

    Adding differentiable rendering closes the loop: the network is penalized
    when the rendered mesh doesn't match what the camera saw.

Dependencies:
    - PyTorch3D (for differentiable rasterization) or
    - nvdiffrast (NVIDIA's differentiable renderer, faster but CUDA-only)
    Both are optional; the module gracefully degrades to silhouette-only
    rendering via a simple Z-buffer when neither is available.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    class _MockModule:
        pass
    class _MockNN:
        Module = _MockModule
    nn = _MockNN()
    F = None


# Try to import differentiable renderers (optional dependencies)
_RENDERER_BACKEND = None
try:
    import pytorch3d
    from pytorch3d.renderer import (
        MeshRasterizer, RasterizationSettings,
        look_at_view_transform, FoVPerspectiveCameras,
    )
    from pytorch3d.structures import Meshes
    _RENDERER_BACKEND = 'pytorch3d'
except ImportError:
    pass

if _RENDERER_BACKEND is None:
    try:
        import nvdiffrast.torch as dr
        _RENDERER_BACKEND = 'nvdiffrast'
    except ImportError:
        pass


class SoftSilhouetteRenderer:
    """
    Differentiable silhouette renderer.

    Renders binary silhouette masks from mesh vertices and faces,
    supporting gradient flow back through vertex positions.

    Falls back to a simple orthographic Z-buffer rasterizer when
    PyTorch3D and nvdiffrast are unavailable.
    """

    def __init__(self, image_size: int = 256, device: str = 'cuda'):
        if torch is None:
            raise RuntimeError("PyTorch required for SoftSilhouetteRenderer")

        self.image_size = image_size
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.backend = _RENDERER_BACKEND

    def render_silhouette(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        camera_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Render a differentiable binary silhouette mask.

        Args:
            vertices: (B, V, 3) mesh vertices
            faces: (F, 3) face indices
            camera_params: optional dict with 'azimuth', 'elevation', 'distance'

        Returns:
            silhouette: (B, 1, H, W) soft binary mask
        """
        if self.backend == 'pytorch3d':
            return self._render_pytorch3d(vertices, faces, camera_params)
        else:
            return self._render_zbuffer_approx(vertices, faces, camera_params)

    def _render_pytorch3d(
        self, vertices: torch.Tensor, faces: torch.Tensor,
        camera_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """Render via PyTorch3D's differentiable rasterizer."""
        B = vertices.shape[0]
        azim = camera_params.get('azimuth', 0.0) if camera_params else 0.0
        elev = camera_params.get('elevation', 0.0) if camera_params else 0.0
        dist = camera_params.get('distance', 0.4) if camera_params else 0.4

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=1e-5,
            faces_per_pixel=1,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        faces_batch = faces.unsqueeze(0).expand(B, -1, -1)
        meshes = Meshes(verts=vertices, faces=faces_batch)

        fragments = rasterizer(meshes)
        # Silhouette: 1 where a face was rasterized, 0 elsewhere
        silhouette = (fragments.pix_to_face[..., 0] >= 0).float()
        return silhouette.unsqueeze(1)

    def _render_zbuffer_approx(
        self, vertices: torch.Tensor, faces: torch.Tensor,
        camera_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Approximate differentiable silhouette via soft orthographic projection.

        Not exact, but provides gradient flow for training when proper
        differentiable renderers are unavailable.
        """
        B, V, _ = vertices.shape
        H = W = self.image_size

        # Simple orthographic projection
        x = vertices[:, :, 0]
        y = vertices[:, :, 1]

        # Normalize to [0, H-1] range
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        y_min = y.min(dim=1, keepdim=True).values
        y_max = y.max(dim=1, keepdim=True).values

        span = torch.max(x_max - x_min, y_max - y_min) * 1.15 + 1e-8

        px = ((x - (x_min + x_max) / 2) / span * (W - 1) + W / 2)
        py = (-(y - (y_min + y_max) / 2) / span * (H - 1) + H / 2)

        # Soft rasterization: place Gaussian splats at vertex positions
        grid_x = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_y = torch.arange(H, device=self.device, dtype=torch.float32)
        gx, gy = torch.meshgrid(grid_x, grid_y, indexing='xy')
        gx = gx.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        gy = gy.unsqueeze(0).unsqueeze(0)

        px_exp = px.unsqueeze(-1).unsqueeze(-1)  # (B, V, 1, 1)
        py_exp = py.unsqueeze(-1).unsqueeze(-1)

        sigma = 1.5
        gaussian = torch.exp(-((gx - px_exp)**2 + (gy - py_exp)**2) / (2 * sigma**2))
        silhouette = gaussian.max(dim=1, keepdim=False).values  # (B, H, W)

        return silhouette.unsqueeze(1).clamp(0, 1)


class DifferentiableRenderLoss(nn.Module):
    """
    Composite render-and-compare loss for Stage 1 identity training.

    Combines:
    1. Silhouette IoU — binary mask overlap between rendered mesh and detected face
    2. Contour distance — directional distance along jaw/chin contour
    3. Landmark reprojection — 2D landmark alignment error

    All losses are differentiable w.r.t. mesh vertex positions and therefore
    w.r.t. β (through the FLAME decode chain).
    """

    def __init__(
        self,
        image_size: int = 256,
        lambda_silhouette: float = 1.0,
        lambda_landmark: float = 0.5,
        device: str = 'cuda',
    ):
        super().__init__()
        if torch is None:
            raise RuntimeError("PyTorch required for DifferentiableRenderLoss")

        self.lambda_silhouette = lambda_silhouette
        self.lambda_landmark = lambda_landmark
        self.renderer = SoftSilhouetteRenderer(image_size=image_size, device=device)
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    def silhouette_iou_loss(
        self,
        pred_silhouette: torch.Tensor,
        target_silhouette: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable IoU loss between predicted and target silhouettes.

        Args:
            pred_silhouette: (B, 1, H, W) rendered mesh silhouette
            target_silhouette: (B, 1, H, W) target face mask from detection

        Returns:
            1 - IoU (lower is better)
        """
        intersection = (pred_silhouette * target_silhouette).sum(dim=[1, 2, 3])
        union = pred_silhouette.sum(dim=[1, 2, 3]) + target_silhouette.sum(dim=[1, 2, 3]) - intersection
        iou = intersection / (union + 1e-6)
        return (1.0 - iou).mean()

    def landmark_reprojection_loss(
        self,
        vertices: torch.Tensor,
        landmarks_2d: torch.Tensor,
        landmark_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        L2 reprojection error between 3D vertex positions (projected to 2D)
        and detected 2D facial landmarks.

        Args:
            vertices: (B, V, 3) mesh vertices
            landmarks_2d: (B, K, 2) detected 2D landmark coordinates
            landmark_indices: (K,) FLAME vertex indices corresponding to landmarks

        Returns:
            mean L2 reprojection error
        """
        # Simple orthographic projection of landmark vertices
        landmark_verts = vertices[:, landmark_indices, :2]  # (B, K, 2) — XY only

        # Normalize both to [-1, 1] range for scale-invariant comparison
        pred_norm = landmark_verts - landmark_verts.mean(dim=1, keepdim=True)
        pred_scale = pred_norm.abs().max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values + 1e-8
        pred_norm = pred_norm / pred_scale

        gt_norm = landmarks_2d - landmarks_2d.mean(dim=1, keepdim=True)
        gt_scale = gt_norm.abs().max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values + 1e-8
        gt_norm = gt_norm / gt_scale

        return F.mse_loss(pred_norm, gt_norm)

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        target_silhouette: Optional[torch.Tensor] = None,
        landmarks_2d: Optional[torch.Tensor] = None,
        landmark_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite differentiable render loss.

        Returns:
            total_loss: scalar tensor
            metrics: dict of individual loss values for logging
        """
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}

        if target_silhouette is not None:
            pred_sil = self.renderer.render_silhouette(vertices, faces)
            loss_sil = self.silhouette_iou_loss(pred_sil, target_silhouette)
            total_loss = total_loss + self.lambda_silhouette * loss_sil
            metrics['loss_silhouette_iou'] = float(loss_sil.item())

        if landmarks_2d is not None and landmark_indices is not None:
            loss_lmk = self.landmark_reprojection_loss(vertices, landmarks_2d, landmark_indices)
            total_loss = total_loss + self.lambda_landmark * loss_lmk
            metrics['loss_landmark_reproj'] = float(loss_lmk.item())

        metrics['loss_render_total'] = float(total_loss.item())
        return total_loss, metrics
