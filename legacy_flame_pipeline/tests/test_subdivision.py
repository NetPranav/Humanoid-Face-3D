"""
Unit tests for Loop subdivision and displacement mapping engine.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile

from src.utils.subdivision import (
    compute_vertex_normals,
    loop_subdivide_step,
    loop_subdivide,
    sample_bilinear_2d,
    apply_displacement_to_mesh,
    export_obj_with_uvs
)


class TestSubdivision(unittest.TestCase):

    def setUp(self):
        # Create a simple tetrahedron with boundary (open cone) or use a sample mesh
        self.root = Path(__file__).resolve().parent.parent
        self.template_path = self.root / 'data' / 'flame_model' / 'head_template.obj'

    def _load_flame_template(self):
        v_list, vt_list, f_v_list, f_vt_list = [], [], [], []
        with open(self.template_path) as f:
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

    def test_single_triangle_subdivision(self):
        # A single triangle has 3 vertices and 3 boundary edges
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        v1, f1, _, _ = loop_subdivide_step(verts, faces)
        # 3 original + 3 edge vertices = 6 vertices, 4 faces
        self.assertEqual(len(v1), 6)
        self.assertEqual(len(f1), 4)

    def test_flame_level_1_subdivision(self):
        v, f, vt, f_vt = self._load_flame_template()
        self.assertEqual(len(v), 5023)
        self.assertEqual(len(f), 9976)
        
        v1, f1, vt1, f_vt1 = loop_subdivide_step(v, f, vt, f_vt)
        self.assertEqual(len(v1), 20018)
        self.assertEqual(len(f1), 39904)
        self.assertEqual(len(f1), len(f_vt1))
        self.assertTrue(np.all(vt1 >= 0.0) and np.all(vt1 <= 1.0))

    def test_flame_film_quality_subdivision_320k(self):
        v, f, vt, f_vt = self._load_flame_template()
        
        v3, f3, vt3, f_vt3 = loop_subdivide(v, f, vt, f_vt, levels=3)
        # Film quality target: ~320k vertices, ~640k triangles
        self.assertEqual(len(v3), 319484)
        self.assertEqual(len(f3), 638464)
        self.assertEqual(len(f3), len(f_vt3))
        self.assertFalse(np.isnan(v3).any())

    def test_neck_seam_contract_pinning(self):
        v, f, vt, f_vt = self._load_flame_template()
        # Subdivide 1 level for speed
        v1, f1, vt1, f_vt1 = loop_subdivide(v, f, vt, f_vt, levels=1)
        
        # Create a uniform positive displacement of 2.0 mm everywhere
        disp_map = np.full((128, 128), 2.0, dtype=np.float32)
        
        v_disp, normals = apply_displacement_to_mesh(
            v1, f1, vt1, f_vt1, disp_map, neck_pinning=True
        )
        
        # Verify delta displacement
        delta = np.linalg.norm(v_disp - v1, axis=1)
        
        # Lowest 20% Y vertices must have strictly delta == 0.0
        y = v1[:, 1]
        y_norm = (y - y.min()) / (y.max() - y.min())
        neck_verts = (y_norm <= 0.20)
        
        self.assertGreater(neck_verts.sum(), 0)
        np.testing.assert_allclose(delta[neck_verts], 0.0, atol=1e-7)
        
        # Upper face vertices must have received displacement
        upper_face = (y_norm > 0.30)
        self.assertTrue(np.all(delta[upper_face] > 0.001))

    def test_obj_export(self):
        v, f, vt, f_vt = self._load_flame_template()
        v1, f1, vt1, f_vt1 = loop_subdivide(v, f, vt, f_vt, levels=1)
        normals = compute_vertex_normals(v1, f1)
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            export_obj_with_uvs(tmp_path, v1, f1, vt1, f_vt1, normals)
            self.assertTrue(tmp_path.exists())
            self.assertGreater(tmp_path.stat().st_size, 1000)
            
            # Read first few lines to verify OBJ format
            with open(tmp_path) as f_obj:
                lines = [f_obj.readline() for _ in range(20)]
            has_v = any(l.startswith('v ') for l in lines)
            self.assertTrue(has_v)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == '__main__':
    unittest.main()
