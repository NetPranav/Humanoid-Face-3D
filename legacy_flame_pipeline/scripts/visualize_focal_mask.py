"""
Vertex Mask Visualization Script.

Renders the anatomical focal weight mask used in Stage 1 training
directly onto the FLAME template mesh to visually verify that the
mandibular border, chin pad, and zygomatic arch selections are correct.

Usage:
    python3 scripts/visualize_focal_mask.py \
        --flame_path data/flame_model/generic_model.pkl \
        --output outputs/focal_mask_overlay.png
"""
import argparse
import numpy as np
import cv2
from pathlib import Path

from src.utils.flame_model import FLAMEModel


def compute_focal_mask(flame: FLAMEModel):
    """Reproduce the exact same mask computation from trainer.py."""
    v = flame.v_template  # (5023, 3) in metres
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    y_norm = (v[:, 1] - y_min) / (y_max - y_min + 1e-8)
    z_norm = (v[:, 2] - z_min) / (z_max - z_min + 1e-8)

    focal_mask = (y_norm >= 0.15) & (y_norm <= 0.60) & (z_norm >= 0.20)
    return focal_mask, y_norm, z_norm


def render_mask_overlay(flame: FLAMEModel, focal_mask: np.ndarray, output_path: str,
                        width: int = 1024, height: int = 1024):
    """Render 3/4 perspective view with focal vertices highlighted in red."""
    verts = flame.v_template.copy()
    faces = flame.faces

    # Camera: 3/4 perspective (30° yaw rotation)
    angle_y = np.radians(30)
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    # Slight downward tilt (10°)
    angle_x = np.radians(-10)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    verts_rot = verts @ (Ry @ Rx).T

    # Project to 2D
    x_min, x_max = verts_rot[:, 0].min(), verts_rot[:, 0].max()
    y_min, y_max = verts_rot[:, 1].min(), verts_rot[:, 1].max()
    span = max(x_max - x_min, y_max - y_min) * 1.15

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    scale = min(width, height) / span

    px = ((verts_rot[:, 0] - cx) * scale + width / 2).astype(np.int32)
    py = (-(verts_rot[:, 1] - cy) * scale + height / 2).astype(np.int32)  # flip Y

    # Z-buffer for face sorting
    face_z = np.mean(verts_rot[faces, 2], axis=1)
    face_order = np.argsort(face_z)  # painter's algorithm: far to near

    canvas = np.full((height, width, 3), 30, dtype=np.uint8)  # dark background

    for fi in face_order:
        tri = faces[fi]
        pts = np.array([[px[tri[0]], py[tri[0]]],
                        [px[tri[1]], py[tri[1]]],
                        [px[tri[2]], py[tri[2]]]]).reshape((-1, 1, 2))

        # Check if any vertex in this face is in the focal mask
        any_focal = focal_mask[tri].any()
        all_focal = focal_mask[tri].all()

        if all_focal:
            color = (50, 50, 220)   # Red: fully in focal region
        elif any_focal:
            color = (60, 120, 200)  # Orange: partially in focal region
        else:
            color = (160, 160, 160) # Light grey: normal weight (1.0x)

        cv2.fillPoly(canvas, [pts], color)

    # Draw wireframe edges
    for fi in face_order:
        tri = faces[fi]
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            cv2.line(canvas, (px[a], py[a]), (px[b], py[b]), (40, 40, 40), 1, cv2.LINE_AA)

    # Legend
    legend_y = 40
    cv2.rectangle(canvas, (width - 350, 20), (width - 20, 140), (20, 20, 20), -1)
    cv2.putText(canvas, "Anatomical Focal Mask", (width - 340, legend_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 340, legend_y + 20), (width - 320, legend_y + 35), (50, 50, 220), -1)
    cv2.putText(canvas, "Focal (2.5x weight)", (width - 310, legend_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 340, legend_y + 42), (width - 320, legend_y + 57), (60, 120, 200), -1)
    cv2.putText(canvas, "Partial overlap", (width - 310, legend_y + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 340, legend_y + 64), (width - 320, legend_y + 79), (160, 160, 160), -1)
    cv2.putText(canvas, "Normal (1.0x weight)", (width - 310, legend_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Stats
    n_focal = int(focal_mask.sum())
    n_total = len(focal_mask)
    pct = 100.0 * n_focal / n_total
    cv2.putText(canvas, f"Focal vertices: {n_focal}/{n_total} ({pct:.1f}%)",
                (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "y_norm in [0.15, 0.60] AND z_norm >= 0.20",
                (20, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"[Mask Visualization] Saved to: {output_path}")
    print(f"  Focal region: {n_focal}/{n_total} vertices ({pct:.1f}%)")
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize anatomical focal weight mask on FLAME template")
    parser.add_argument('--flame_path', type=str, default='data/flame_model/generic_model.pkl')
    parser.add_argument('--output', type=str, default='outputs/focal_mask_overlay.png')
    args = parser.parse_args()

    flame = FLAMEModel(args.flame_path, scale_to_mm=False)
    focal_mask, y_norm, z_norm = compute_focal_mask(flame)
    render_mask_overlay(flame, focal_mask, args.output)


if __name__ == '__main__':
    main()
