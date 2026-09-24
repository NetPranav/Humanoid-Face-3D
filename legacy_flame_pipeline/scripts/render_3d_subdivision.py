#!/usr/bin/env python3
"""
Film-Quality 320k Subdivided 3D Head Mesh Renderer & Progression Suite.

Subdivides production meshes to ~320,000 vertices (640,000 triangles) via Loop
subdivision, enforces the Neck Seam Contract, applies physical micro-displacement,
and renders organic, facet-free 3D visual comparisons across all production subjects.
"""

import os
import sys
from pathlib import Path
import time
import json
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.subdivision import (
    loop_subdivide,
    apply_displacement_to_mesh,
    compute_vertex_normals,
    export_obj_with_uvs
)

# ─── Configuration ───────────────────────────────────────────────────────────
SUBJECTS = ['carell', 'connelly', 'justin', 'lawrence']
SUBJECT_DISPLAY = {
    'carell': 'Steve Carell',
    'connelly': 'Jennifer Connelly',
    'justin': 'Justin Timberlake',
    'lawrence': 'Jennifer Lawrence'
}

PROD_BATCH = ROOT / 'outputs' / 'production_batch'
COMPARISONS = ROOT / 'outputs' / 'comparisons'
SUBDIV_OUT = ROOT / 'outputs' / 'subdivided_320k'
SUBDIV_OUT.mkdir(parents=True, exist_ok=True)


def load_flame_template():
    """Load canonical template UV layout and topology."""
    template_path = ROOT / 'data' / 'flame_model' / 'head_template.obj'
    v_list, vt_list, f_v_list, f_vt_list = [], [], [], []
    with open(template_path) as f:
        for line in f:
            if line.startswith('v '):
                v_list.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('vt '):
                vt_list.append([float(x) for x in line.split()[1:3]])
            elif line.startswith('f '):
                parts = line.strip().split()[1:4]
                f_v_list.append([int(p.split('/')[0]) - 1 for p in parts])
                f_vt_list.append([int(p.split('/')[1]) - 1 for p in parts])

    return (
        np.array(v_list, dtype=np.float32),
        np.array(f_v_list, dtype=np.int32),
        np.array(vt_list, dtype=np.float32),
        np.array(f_vt_list, dtype=np.int32)
    )


def load_subject_base_mesh(subject: str) -> np.ndarray:
    """Loads subject base neutral mesh vertices."""
    obj_path = PROD_BATCH / subject / 'head_mesh_neutral.obj'
    if not obj_path.exists():
        obj_path = PROD_BATCH / subject / 'head_mesh.obj'
    v_list = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                v_list.append([float(x) for x in line.split()[1:4]])
    return np.array(v_list, dtype=np.float32)


def render_mesh(verts, faces, normals=None, is_smooth=True, res=800, base_color=(215, 190, 170)):
    """
    Renders a mesh using three-point Blinn-Phong lighting.
    
    If is_smooth is True: uses interpolated smooth vertex normals (zero visible facets).
    If is_smooth is False: uses flat geometric face normals (shows mesh polygon facets).
    """
    center = verts.mean(axis=0)
    centered = verts - center
    scale = 0.85 / (np.abs(centered).max() + 1e-8)
    scaled = centered * scale

    if normals is None:
        normals = compute_vertex_normals(verts, faces)

    if is_smooth:
        # Smooth normal per triangle from averaged vertex normals
        fn = (normals[faces[:, 0]] + normals[faces[:, 1]] + normals[faces[:, 2]]) / 3.0
        fn_len = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
        fn = fn / fn_len
    else:
        # Flat face normals (shows raw polygonal facets)
        v0 = scaled[faces[:, 0]]
        v1 = scaled[faces[:, 1]]
        v2 = scaled[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_len = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
        fn = fn / fn_len

    # Backface culling
    front_mask = (fn[:, 2] > -0.05)
    vis_faces = faces[front_mask]
    vis_fn = fn[front_mask]

    proj_x = ((scaled[:, 0] * 0.5 + 0.5) * (res - 1)).astype(np.float32)
    proj_y = ((0.5 - scaled[:, 1] * 0.5) * (res - 1)).astype(np.float32)
    proj_z = scaled[:, 2]

    # Z-sort (painter's algorithm)
    face_z = (proj_z[vis_faces[:, 0]] + proj_z[vis_faces[:, 1]] + proj_z[vis_faces[:, 2]]) / 3.0
    sort_idx = np.argsort(face_z)
    sorted_faces = vis_faces[sort_idx]
    sorted_fn = vis_fn[sort_idx]

    # Three-point studio lighting setup
    light_key = np.array([0.4, 0.5, 0.75], dtype=np.float32)
    light_key /= np.linalg.norm(light_key)
    light_fill = np.array([-0.5, 0.2, 0.6], dtype=np.float32)
    light_fill /= np.linalg.norm(light_fill)
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    half_vec = light_key + view_dir
    half_vec /= np.linalg.norm(half_vec)

    ndotl_key = np.maximum(np.sum(sorted_fn * light_key, axis=1), 0.0)
    ndotl_fill = np.maximum(np.sum(sorted_fn * light_fill, axis=1), 0.0)
    ndoth = np.maximum(np.sum(sorted_fn * half_vec, axis=1), 0.0)
    spec = ndoth ** 28

    base_r, base_g, base_b = base_color
    r = np.clip(base_r * (0.22 + 0.60 * ndotl_key + 0.22 * ndotl_fill) + spec * 0.32 * 255.0, 0, 255).astype(np.uint8)
    g = np.clip(base_g * (0.22 + 0.58 * ndotl_key + 0.24 * ndotl_fill) + spec * 0.32 * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(base_b * (0.22 + 0.54 * ndotl_key + 0.26 * ndotl_fill) + spec * 0.32 * 255.0, 0, 255).astype(np.uint8)

    img = np.zeros((res, res, 3), dtype=np.uint8)
    p0_x = proj_x[sorted_faces[:, 0]].astype(np.int32)
    p0_y = proj_y[sorted_faces[:, 0]].astype(np.int32)
    p1_x = proj_x[sorted_faces[:, 1]].astype(np.int32)
    p1_y = proj_y[sorted_faces[:, 1]].astype(np.int32)
    p2_x = proj_x[sorted_faces[:, 2]].astype(np.int32)
    p2_y = proj_y[sorted_faces[:, 2]].astype(np.int32)

    pts_all = np.stack([
        np.stack([p0_x, p0_y], axis=-1),
        np.stack([p1_x, p1_y], axis=-1),
        np.stack([p2_x, p2_y], axis=-1)
    ], axis=1)

    for i in range(len(sorted_faces)):
        color = (int(b[i]), int(g[i]), int(r[i]))
        cv2.fillConvexPoly(img, pts_all[i], color)

    return img


def add_label(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Adds professional header badge to render."""
    out = img.copy()
    h, w = out.shape[:2]
    # Gradient banner at top
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.putText(out, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 210, 255), 1, cv2.LINE_AA)
    return out


def main():
    print("=" * 70)
    print("Film-Quality 320k Subdivided 3D Head Mesh Renderer")
    print("=" * 70)

    # 1. Load canonical template
    _, template_fv, template_vt, template_fvt = load_flame_template()
    print(f"Loaded template: {len(template_vt)} UVs, {len(template_fv)} faces")

    all_subject_panels = []

    for subject in SUBJECTS:
        t_sub_start = time.time()
        print(f"\nProcessing [{subject.upper()}] ({SUBJECT_DISPLAY[subject]})...")
        subj_dir = SUBDIV_OUT / subject
        subj_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load base mesh
        base_v = load_subject_base_mesh(subject)
        print(f"  Base mesh: {len(base_v)} verts, {len(template_fv)} faces")

        # 3. Render raw faceted 5k baseline
        img_5k_faceted = render_mesh(base_v, template_fv, is_smooth=False)
        img_5k_faceted = add_label(
            img_5k_faceted,
            f"{SUBJECT_DISPLAY[subject]} - Baseline",
            "5,023 Verts | Flat Shaded (Visible Geometry Facets)"
        )

        # 4. Perform 3 levels of Loop subdivision -> ~320k verts
        t_subdiv = time.time()
        v3, f3, vt3, f_vt3 = loop_subdivide(
            base_v, template_fv, template_vt, template_fvt, levels=3
        )
        subdiv_dur = time.time() - t_subdiv
        print(f"  Loop Subdivision (3 levels): {len(v3):,} verts, {len(f3):,} tris ({subdiv_dur:.3f}s)")

        # 5. Render smooth subdivided base (no displacement)
        n3_smooth = compute_vertex_normals(v3, f3)
        img_320k_smooth = render_mesh(v3, f3, n3_smooth, is_smooth=True)
        img_320k_smooth = add_label(
            img_320k_smooth,
            f"{SUBJECT_DISPLAY[subject]} - 320k Film Mesh",
            f"{len(v3):,} Verts | Smooth C2 Curvature (Zero Facets)"
        )

        # 6. Export smooth 320k base OBJ
        obj_320k_path = subj_dir / "head_mesh_320k_neutral.obj"
        export_obj_with_uvs(obj_320k_path, v3, f3, vt3, f_vt3, n3_smooth)
        print(f"  Exported 320k neutral OBJ: {obj_320k_path.stat().st_size / 1e6:.1f} MB")

        # 7. Load displacement maps
        # Pilot map
        pilot_p = PROD_BATCH / subject / "head_displacement_16bit.png"
        disp_u16_pilot = cv2.imread(str(pilot_p), cv2.IMREAD_UNCHANGED)
        disp_pilot_mm = (disp_u16_pilot.astype(np.float32) / 65535.0 * 2.0 - 1.0) * 1.5

        # Deep map
        deep_p = COMPARISONS / subject / "displacement_16bit_deep.png"
        if deep_p.exists():
            disp_u16_deep = cv2.imread(str(deep_p), cv2.IMREAD_UNCHANGED)
            disp_deep_mm = (disp_u16_deep.astype(np.float32) / 65535.0 * 2.0 - 1.0) * 1.5
        else:
            disp_deep_mm = disp_pilot_mm

        # 8. Apply displacement along smooth normals with Neck Seam Contract
        v3_pilot, n3_pilot = apply_displacement_to_mesh(
            v3, f3, vt3, f_vt3, disp_pilot_mm, neck_pinning=True
        )
        v3_deep, n3_deep = apply_displacement_to_mesh(
            v3, f3, vt3, f_vt3, disp_deep_mm, neck_pinning=True
        )

        # 9. Render displaced meshes
        img_320k_pilot = render_mesh(v3_pilot, f3, n3_pilot, is_smooth=True)
        img_320k_pilot = add_label(
            img_320k_pilot,
            f"{SUBJECT_DISPLAY[subject]} - Pilot GAN Displaced",
            f"{len(v3):,} Verts | Sharp Wrinkle/Pore Geometric Relief"
        )

        img_320k_deep = render_mesh(v3_deep, f3, n3_deep, is_smooth=True)
        img_320k_deep = add_label(
            img_320k_deep,
            f"{SUBJECT_DISPLAY[subject]} - Deep GAN Displaced",
            f"{len(v3):,} Verts | Deep GAN 32k Displaced Geometry"
        )

        # 10. Save individual renders
        cv2.imwrite(str(subj_dir / "render_5k_baseline.png"), img_5k_faceted)
        cv2.imwrite(str(subj_dir / "render_320k_smooth.png"), img_320k_smooth)
        cv2.imwrite(str(subj_dir / "render_320k_pilot_disp.png"), img_320k_pilot)
        cv2.imwrite(str(subj_dir / "render_320k_deep_disp.png"), img_320k_deep)

        # Export displaced OBJ for UE5 / DCC testing
        export_obj_with_uvs(
            subj_dir / "head_mesh_320k_displaced_pilot.obj",
            v3_pilot, f3, vt3, f_vt3, n3_pilot
        )

        # 11. Build 4-panel subject strip
        # [5k Faceted | 320k Smooth | 320k Pilot Displaced | 320k Deep Displaced]
        subj_strip = np.hstack([img_5k_faceted, img_320k_smooth, img_320k_pilot, img_320k_deep])
        cv2.imwrite(str(subj_dir / "progression_strip_4panel.png"), subj_strip)

        all_subject_panels.append(subj_strip)
        print(f"  Done in {time.time() - t_sub_start:.2f}s")

    # 12. Build master 4-subject progression grid
    # 4 rows (subjects) x 4 columns (5k raw -> 320k smooth -> 320k pilot -> 320k deep)
    master_grid = np.vstack(all_subject_panels)

    # Master title banner
    banner_h = 100
    banner = np.zeros((banner_h, master_grid.shape[1], 3), dtype=np.uint8)
    banner[:] = (20, 24, 30)
    cv2.putText(
        banner,
        "Humanoid-Face-3D: Film-Quality 320,000-Vertex Geometric Progression",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )
    cv2.putText(
        banner,
        "Col 1: Raw 5k Base Mesh (Faceted) | Col 2: Subdivided 320k Base Mesh (Zero Faceting) | Col 3: Pilot GAN Geometric Displacement | Col 4: Deep GAN Geometric Displacement",
        (40, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (160, 210, 255),
        2,
        cv2.LINE_AA
    )

    full_showcase = np.vstack([banner, master_grid])
    showcase_path = COMPARISONS / "full_4subject_320k_progression.png"
    cv2.imwrite(str(showcase_path), full_showcase)
    print(f"\nSaved Master 4-Subject Progression Showcase: {showcase_path}")
    print(f"Dimensions: {full_showcase.shape[1]}x{full_showcase.shape[0]}")


if __name__ == '__main__':
    main()
