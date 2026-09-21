"""
Static 3D Hair Card Polygonal Mesh Generator & Procedural Hair Textures.

Synthesizes game-ready hair card ribbon geometry (quad strips) placed on the facial mesh
at anatomical hair zones (mustache, chin/goatee, jawline, sideburns, eyebrows).

Generates:
  1. 3D polygonal hair card mesh with UV unwrapping ('facial_hair_cards.obj')
  2. Procedural alpha opacity mask ('hair_card_alpha.png')
  3. Procedural hair strand normal map ('hair_card_normal.png')
  4. Armature skinning weights linking hair cards to skeletal bones (jaw, head).
"""
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np
import cv2

from src.stage4_facial_hair.regions import (
    FacialHairConfig,
    FacialHairRegionSegmenter,
    resolve_hair_config,
    compute_normalized_coordinates,
)


class HairCardGenerator:
    """
    Generates oriented 3D hair card strips rooted on facial geometry.
    """
    def __init__(self, neck_collar_threshold: float = 0.20):
        self.segmenter = FacialHairRegionSegmenter(neck_collar_threshold=neck_collar_threshold)
        self.neck_collar_threshold = float(neck_collar_threshold)

    def generate_hair_cards(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        config: Optional[Union[FacialHairConfig, Dict[str, Any], str]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes 3D hair card quads rooted on facial mesh triangles.

        Args:
            vertices: (N, 3) float array of neutral head mesh.
            faces: (F, 3) int array of triangle face indices.
            config: FacialHairConfig, preset name, or dict.

        Returns:
            dict containing:
              - 'card_vertices': (M, 3) float array of card vertices
              - 'card_faces': (K, 3) int array of triangle face indices (0-indexed)
              - 'card_uvs': (M, 2) float array of UV texture coordinates
              - 'card_normals': (M, 3) float array of vertex normal vectors
              - 'skinning_weights': (M, 5) float array of LBS weights for [neck, head, jaw, eye_L, eye_R]
              - 'num_cards': int total number of card ribbons generated
        """
        cfg = resolve_hair_config(config)

        if cfg.is_clean_shaven() or not cfg.generate_cards:
            return {
                'card_vertices': np.zeros((0, 3), dtype=np.float32),
                'card_faces': np.zeros((0, 3), dtype=np.int32),
                'card_uvs': np.zeros((0, 2), dtype=np.float32),
                'card_normals': np.zeros((0, 3), dtype=np.float32),
                'skinning_weights': np.zeros((0, 5), dtype=np.float32),
                'num_cards': 0,
            }

        # 1. Compute vertex normals and region weights
        v_normals = self._compute_vertex_normals(vertices, faces)
        region_res = self.segmenter.extract_region_weights(vertices, cfg)
        composite_weights = region_res['composite']
        growth_dirs = region_res['growth_directions']

        # 2. Compute face centers, face normals, and face weights
        face_verts = vertices[faces]  # (F, 3, 3)
        face_centers = np.mean(face_verts, axis=1)  # (F, 3)

        face_normals_raw = np.cross(
            face_verts[:, 1] - face_verts[:, 0],
            face_verts[:, 2] - face_verts[:, 0]
        )
        face_norm_lens = np.linalg.norm(face_normals_raw, axis=1, keepdims=True)
        face_normals = np.where(face_norm_lens > 1e-6, face_normals_raw / np.maximum(face_norm_lens, 1e-6), np.array([0, 0, 1.0]))

        face_weights = np.mean(composite_weights[faces], axis=1)  # (F,)
        face_growths = np.mean(growth_dirs[faces], axis=1)        # (F, 3)

        # 3. Filter candidates: faces with hair weight > 0.05
        candidate_indices = np.where(face_weights > 0.05)[0]
        if len(candidate_indices) == 0:
            return {
                'card_vertices': np.zeros((0, 3), dtype=np.float32),
                'card_faces': np.zeros((0, 3), dtype=np.int32),
                'card_uvs': np.zeros((0, 2), dtype=np.float32),
                'card_normals': np.zeros((0, 3), dtype=np.float32),
                'skinning_weights': np.zeros((0, 5), dtype=np.float32),
                'num_cards': 0,
            }

        candidate_weights = face_weights[candidate_indices]
        prob = candidate_weights / np.sum(candidate_weights)

        # Sample root locations
        rng = np.random.RandomState(42)
        n_samples = min(cfg.card_density, len(candidate_indices) * 2)
        chosen_faces = rng.choice(candidate_indices, size=n_samples, p=prob, replace=True)

        card_verts_list = []
        card_faces_list = []
        card_uvs_list = []
        card_normals_list = []
        card_skinning_list = []

        # Check mesh coordinate scale:
        # FLAME canonical mesh coordinates are in meters (extents ~0.15 - 0.25 m).
        # Synthetic test meshes or metric millimeter meshes have extents ~100 - 200 mm.
        mesh_extent = float(np.max(np.abs(vertices)))
        unit_scale = 0.001 if mesh_extent < 5.0 else 1.0

        # Geometry parameters scaled to match mesh units
        card_len = cfg.card_length_mm * unit_scale
        card_w = cfg.card_width_mm * unit_scale
        curvature = cfg.card_curvature

        vertex_offset = 0
        cards_count = 0

        for f_idx in chosen_faces:
            # Jitter root within face
            tri = vertices[faces[f_idx]]
            r1, r2 = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            r3 = 1.0 - r1 - r2
            root_pos = r1 * tri[0] + r2 * tri[1] + r3 * tri[2]

            surface_norm = face_normals[f_idx]
            growth_dir = face_growths[f_idx]

            # Orthogonalize growth direction to surface normal
            proj = np.dot(growth_dir, surface_norm)
            tangent_dir = growth_dir - proj * surface_norm
            t_len = np.linalg.norm(tangent_dir)
            if t_len > 1e-4:
                tangent_dir = tangent_dir / t_len
            else:
                tangent_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            # Binormal across the card width
            binormal = np.cross(surface_norm, tangent_dir)
            b_len = np.linalg.norm(binormal)
            if b_len > 1e-4:
                binormal = binormal / b_len
            else:
                binormal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            # Randomize length and width slightly for natural clump variation
            len_jitter = card_len * rng.uniform(0.85, 1.15)
            w_jitter = card_w * rng.uniform(0.85, 1.15)

            # Build 2-segment quad strip: 3 pairs of vertices (Root, Mid, Tip)
            # Level 0 (Root): sits flush on surface
            half_w_root = w_jitter * 0.5
            p0_l = root_pos - binormal * half_w_root
            p0_r = root_pos + binormal * half_w_root

            # Level 1 (Mid): elevated along normal, moved along tangent
            half_w_mid = w_jitter * 0.45
            mid_center = root_pos + tangent_dir * (len_jitter * 0.5) + surface_norm * (len_jitter * 0.25 * curvature)
            p1_l = mid_center - binormal * half_w_mid
            p1_r = mid_center + binormal * half_w_mid

            # Level 2 (Tip): curls further along tangent, tapers in width
            half_w_tip = w_jitter * 0.25
            tip_center = root_pos + tangent_dir * len_jitter + surface_norm * (len_jitter * 0.15 * curvature)
            p2_l = tip_center - binormal * half_w_tip
            p2_r = tip_center + binormal * half_w_tip

            # 6 vertices per card
            c_verts = np.array([p0_l, p0_r, p1_l, p1_r, p2_l, p2_r], dtype=np.float32)

            # UVs: (u, v)
            c_uvs = np.array([
                [0.0, 0.0], [1.0, 0.0],  # Root
                [0.0, 0.5], [1.0, 0.5],  # Mid
                [0.0, 1.0], [1.0, 1.0],  # Tip
            ], dtype=np.float32)

            # Normals: oriented along surface normal with slight tilt
            c_norm = surface_norm + tangent_dir * 0.2
            c_norm = c_norm / np.maximum(np.linalg.norm(c_norm), 1e-6)
            c_normals = np.tile(c_norm, (6, 1)).astype(np.float32)

            # Triangles (4 triangles forming 2 quads):
            # Quad 1: [0, 1, 3] and [0, 3, 2]
            # Quad 2: [2, 3, 5] and [2, 5, 4]
            c_faces = np.array([
                [vertex_offset + 0, vertex_offset + 1, vertex_offset + 3],
                [vertex_offset + 0, vertex_offset + 3, vertex_offset + 2],
                [vertex_offset + 2, vertex_offset + 3, vertex_offset + 5],
                [vertex_offset + 2, vertex_offset + 5, vertex_offset + 4],
            ], dtype=np.int32)

            # Skinning weights relative to 5 facial joints: [neck, head, jaw, eye_L, eye_R]
            # If root y_norm < 0.45: predominantly jaw (moves when talking/mouth opens)
            # If root y_norm >= 0.45: predominantly head bone
            # Normalized coordinates of root
            y_span = vertices[:, 1].max() - vertices[:, 1].min()
            root_y_norm = (root_pos[1] - vertices[:, 1].min()) / max(y_span, 1e-4)

            skin_w = np.zeros((6, 5), dtype=np.float32)
            if root_y_norm < 0.44:
                # Lower face (mustache, chin, jaw): 85% jaw, 15% head
                skin_w[:, 2] = 0.85  # jaw
                skin_w[:, 1] = 0.15  # head
            else:
                # Upper face (eyebrows, sideburns): 95% head, 5% jaw
                skin_w[:, 1] = 0.95  # head
                skin_w[:, 2] = 0.05  # jaw

            card_verts_list.append(c_verts)
            card_faces_list.append(c_faces)
            card_uvs_list.append(c_uvs)
            card_normals_list.append(c_normals)
            card_skinning_list.append(skin_w)

            vertex_offset += 6
            cards_count += 1

        all_verts = np.vstack(card_verts_list).astype(np.float32)
        all_faces = np.vstack(card_faces_list).astype(np.int32)
        all_uvs = np.vstack(card_uvs_list).astype(np.float32)
        all_normals = np.vstack(card_normals_list).astype(np.float32)
        all_skinning = np.vstack(card_skinning_list).astype(np.float32)

        return {
            'card_vertices': all_verts,
            'card_faces': all_faces,
            'card_uvs': all_uvs,
            'card_normals': all_normals,
            'skinning_weights': all_skinning,
            'num_cards': cards_count,
        }

    def _compute_vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Computes smooth area-weighted vertex normals."""
        normals = np.zeros_like(vertices, dtype=np.float32)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)

        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)

        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(lens > 1e-6, normals / np.maximum(lens, 1e-6), np.array([0, 1.0, 0]))
        return normals

    def generate_hair_card_textures(self, resolution: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes procedural hair card alpha opacity mask and tangent normal map.

        Returns:
            (alpha_mask, normal_map):
              - alpha_mask: (H, W) uint8 grayscale image [0, 255]
              - normal_map: (H, W, 3) uint8 RGB tangent normal map
        """
        rng = np.random.RandomState(42)
        alpha = np.zeros((resolution, resolution), dtype=np.float32)

        # Generate 4-7 overlapping hair strands per card ribbon
        n_strands = 6
        v_coords = np.linspace(0.0, 1.0, resolution)

        for s_idx in range(n_strands):
            center_u = (s_idx + 0.5) / n_strands + rng.uniform(-0.04, 0.04)
            width = rng.uniform(0.12, 0.18)
            strand_len = rng.uniform(0.88, 1.0)

            # Strand profile across height (v)
            for v_pix in range(resolution):
                v_norm = v_pix / (resolution - 1)
                if v_norm > strand_len:
                    continue

                # Root feathering: v in [0.0, 0.15]
                root_fade = min(1.0, v_norm / 0.15) if v_norm < 0.15 else 1.0
                # Tip taper: v in [strand_len - 0.25, strand_len]
                tip_fade = max(0.0, (strand_len - v_norm) / 0.25) if v_norm > (strand_len - 0.25) else 1.0
                intensity = root_fade * tip_fade

                # Slight curl/sway across width
                u_offset = 0.03 * np.sin(v_norm * np.pi * 2.0 + s_idx)
                u_pos = center_u + u_offset

                # Parabolic cross-section across width
                u_coords = np.linspace(0.0, 1.0, resolution)
                u_dist = np.abs(u_coords - u_pos) / width
                strand_cross = np.maximum(0.0, 1.0 - u_dist ** 2) * intensity

                alpha[v_pix, :] = np.maximum(alpha[v_pix, :], strand_cross)

        # Normalize alpha to [0, 255]
        alpha_uint8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

        # Synthesize tangent normal map from the strands
        sobel_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)

        nx = -sobel_x * 1.5
        ny = -sobel_y * 1.5
        nz = np.ones_like(alpha, dtype=np.float32)

        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        norm = np.maximum(norm, 1e-6)

        nx = nx / norm
        ny = ny / norm
        nz = nz / norm

        r = np.clip(((nx + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        g = np.clip(((ny + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        b = np.clip(((nz + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)

        normal_uint8 = np.stack([r, g, b], axis=-1)

        return alpha_uint8, normal_uint8

    def export_cards_obj(
        self,
        cards_data: Dict[str, Any],
        output_obj_path: Union[str, Path]
    ) -> str:
        """
        Exports static hair card geometry to clean Wavefront OBJ format.
        Includes vertices (v), texture coordinates (vt), normals (vn), and faces (f v/vt/vn).
        """
        out_path = Path(output_obj_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        verts = cards_data['card_vertices']
        uvs = cards_data['card_uvs']
        normals = cards_data['card_normals']
        faces = cards_data['card_faces']

        with open(out_path, 'w') as fp:
            fp.write("# Humanoid-Face-3D Facial Hair Cards Mesh\n")
            fp.write(f"# Cards: {cards_data['num_cards']}, Vertices: {len(verts)}, Triangles: {len(faces)}\n")

            # 1. Vertices
            for v in verts:
                fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # 2. UVs
            for uv in uvs:
                fp.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # 3. Normals
            for n in normals:
                fp.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # 4. Faces (1-indexed in OBJ format)
            for f in faces + 1:
                fp.write(f"f {f[0]}/{f[0]}/{f[0]} {f[1]}/{f[1]}/{f[1]} {f[2]}/{f[2]}/{f[2]}\n")

        return str(out_path.resolve())

    def export_card_textures(
        self,
        output_dir: Union[str, Path],
        resolution: int = 512,
        prefix: str = "hair_card"
    ) -> Dict[str, str]:
        """
        Saves procedural alpha opacity mask and tangent normal map.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        alpha_img, normal_rgb = self.generate_hair_card_textures(resolution=resolution)

        alpha_path = out_path / f"{prefix}_alpha.png"
        cv2.imwrite(str(alpha_path.resolve()), alpha_img)

        normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)
        normal_path = out_path / f"{prefix}_normal.png"
        cv2.imwrite(str(normal_path.resolve()), normal_bgr)

        return {
            "alpha_png": str(alpha_path.resolve()),
            "normal_png": str(normal_path.resolve()),
        }
