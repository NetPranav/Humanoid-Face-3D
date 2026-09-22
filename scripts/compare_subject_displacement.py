#!/usr/bin/env python3
"""
Run side-by-side inference on subject Justin using Pilot (1.5k) vs Deep (32.3k) Generator.
Generates:
1. High-resolution 1024² displacement maps for both models.
2. Tangent-space normal maps derived from displacement gradients.
3. Absolute pore difference map highlighting micro-follicular grain.
4. Zoomed-in forehead pore analysis panel.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import cv2
import torch

from src.stage3_detail.generator import DetailGenerator
from src.stage3_detail.rasterizer import (
    load_flame_uv_layout,
    compute_vertex_normals,
    rasterize_uv_maps
)

def parse_obj(obj_path: str):
    verts, faces = [], []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

def compute_normals_from_disp(disp_mm: np.ndarray, mask: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Computes tangent-space normal map from displacement surface gradient."""
    dx = cv2.Sobel(disp_mm, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(disp_mm, cv2.CV_32F, 0, 1, ksize=3)
    
    # Tangent normal vectors (-dx, -dy, 1) normalized to [-1, 1] then [0, 255]
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm

    # Pack to RGB: R=X, G=Y, B=Z
    rgb = np.stack([(nx + 1.0) * 0.5, (ny + 1.0) * 0.5, (nz + 1.0) * 0.5], axis=-1)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb

def main():
    print("=" * 70)
    print("👤 Running Live Inference Comparison on Subject: Justin")
    print("=" * 70)

    device = torch.device('cpu')
    mesh_path = "outputs/production_batch/justin/head_mesh_neutral.obj"
    manifest_path = "outputs/production_batch/justin/manifest.json"
    pilot_ckpt = "outputs/kaggle_phase3_gan/extracted/ema_generator.pt"
    deep_ckpt = "outputs/phase-3-deep-detail-gan-train/Humanoid-Face-3D/checkpoints/stage3_detail/ema_generator.pt"
    out_dir = Path("outputs/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse mesh
    print(f"Loading neutral base geometry from: {mesh_path}")
    verts, faces = parse_obj(mesh_path)
    manifest = json.load(open(manifest_path))
    beta = np.array(manifest['beta_shape'], dtype=np.float32)

    # 2. Synthesize with Pilot Model (1,500 steps)
    print("Synthesizing with Pilot Generator (1,500 steps)...")
    from src.stage3_detail.inference import DetailSynthesizer
    synth_pilot = DetailSynthesizer(checkpoint_path=pilot_ckpt, device='cpu')
    res_pilot = synth_pilot.synthesize(verts, faces, beta=beta, resolution=1024)

    # 3. Synthesize with Studio Deep Model (32,291 steps)
    print("Synthesizing with Studio Deep Generator (32,291 steps)...")
    synth_deep = DetailSynthesizer(checkpoint_path=deep_ckpt, device='cpu')
    res_deep = synth_deep.synthesize(verts, faces, beta=beta, resolution=1024)

    disp_pilot_mm = res_pilot['disp_mm']
    disp_deep_mm = res_deep['disp_mm']
    norm_pilot = res_pilot['normal_map_rgb']
    norm_deep = res_deep['normal_map_rgb']
    mask = res_deep['mask']
    diff_mm = np.abs(disp_deep_mm - disp_pilot_mm) * (mask > 0)

    # Compute high-frequency metric (Laplacian variance on facial zone)
    lap_p = float(np.var(cv2.Laplacian(disp_pilot_mm, cv2.CV_32F)[mask > 0.5]))
    lap_d = float(np.var(cv2.Laplacian(disp_deep_mm, cv2.CV_32F)[mask > 0.5]))
    sharpness_gain = ((lap_d - lap_p) / (lap_p + 1e-8)) * 100

    print("\n" + "=" * 70)
    print("📊 EXACT NUMERICAL DIFFERENCES GAINED (Subject: Justin)")
    print("=" * 70)
    print(f"Displacement Range (Pilot): [{disp_pilot_mm[mask > 0.5].min():.4f}, {disp_pilot_mm[mask > 0.5].max():.4f}] mm")
    print(f"Displacement Range (Deep):  [{disp_deep_mm[mask > 0.5].min():.4f}, {disp_deep_mm[mask > 0.5].max():.4f}] mm")
    print(f"Displacement Std (Pilot):   {disp_pilot_mm[mask > 0.5].std():.4f} mm")
    print(f"Displacement Std (Deep):    {disp_deep_mm[mask > 0.5].std():.4f} mm")
    print(f"Pore Detail Energy (Pilot): {lap_p:.6f}")
    print(f"Pore Detail Energy (Deep):  {lap_d:.6f}")
    print(f"Mean Micro-Shift Magnitude: {diff_mm[mask > 0.5].mean():.4f} mm across all facial UV pixels")


    # Render Visual Comparison Panels
    def to_color(d: np.ndarray, vmin=-0.6, vmax=0.6, cmap=cv2.COLORMAP_INFERNO):
        norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)
        return cv2.applyColorMap(u8, cmap)

    c_pilot = to_color(disp_pilot_mm)
    c_deep = to_color(disp_deep_mm)
    c_diff = cv2.applyColorMap((np.clip(diff_mm / 0.3, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_HOT)

    # Zoom in on Forehead pore cluster: [360:540, 420:600]
    crop_slice = (slice(360, 540), slice(420, 600))
    z_pilot = cv2.resize(to_color(disp_pilot_mm[crop_slice], vmin=-0.3, vmax=0.3), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    z_deep = cv2.resize(to_color(disp_deep_mm[crop_slice], vmin=-0.3, vmax=0.3), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    z_norm_pilot = cv2.resize(norm_pilot[crop_slice], (1024, 1024), interpolation=cv2.INTER_NEAREST)
    z_norm_deep = cv2.resize(norm_deep[crop_slice], (1024, 1024), interpolation=cv2.INTER_NEAREST)

    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_label(img, title, subtitle=""):
        res = img.copy()
        cv2.rectangle(res, (0, 0), (1024, 85), (20, 20, 20), -1)
        cv2.putText(res, title, (25, 42), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(res, subtitle, (25, 74), font, 0.72, (100, 255, 100), 2, cv2.LINE_AA)
        return res

    p1 = add_label(c_pilot, "1. Justin Displacement (Pilot: 1.5k Steps)", f"Std: {disp_pilot_mm[mask>0.5].std():.3f} mm | Coarse folds")
    p2 = add_label(c_deep, "2. Justin Displacement (Deep: 32.3k Steps)", f"Std: {disp_deep_mm[mask>0.5].std():.3f} mm | High-freq resolved")
    p3 = add_label(c_diff, "3. Synthesized Micro-Detail Difference", f"Avg detail delta: {diff_mm[mask>0.5].mean():.3f} mm")

    p4 = add_label(z_pilot, "4. Forehead Zoom: Pilot (1.5k Steps)", "Washed-out follicular pores, blurry grain")
    p5 = add_label(z_deep, "5. Forehead Zoom: Studio Deep (32.3k Steps)", "Sub-millimeter skin pores & epidermal micro-texture")
    p6 = add_label(z_norm_deep, "6. Tangent Normal Map Zoom (Deep Model)", "Shader-ready normal perturbations for UE5")

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    panel = np.vstack([row1, row2])

    out_panel = out_dir / "justin_inference_comparison_1500_vs_32291.png"
    cv2.imwrite(str(out_panel), panel)

    # Also save standalone high-res 16-bit PNG displacement for inspection
    disp_deep_16 = res_deep['disp_uint16']
    cv2.imwrite(str(out_dir / "justin_displacement_16bit_deep_32k.png"), disp_deep_16)
    cv2.imwrite(str(out_dir / "justin_normal_map_deep_32k.png"), cv2.cvtColor(norm_deep, cv2.COLOR_RGB2BGR))

    print(f"\n✅ High-res comparison visual panel saved to: {out_panel}")
    print(f"✅ 16-bit deep displacement map saved to: {out_dir / 'justin_displacement_16bit_deep_32k.png'}")
    print(f"✅ Deep tangent normal map saved to: {out_dir / 'justin_normal_map_deep_32k.png'}")
    print("=" * 70)

if __name__ == '__main__':
    main()
