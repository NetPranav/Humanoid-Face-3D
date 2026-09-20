"""
Comprehensive Unit Tests for Stage 5 Production Retopology, Rigging, and Export.

Tests:
1. Retopology: SparseMatrixCSR, correspondence matrix W, barycentric weights, and I/O.
2. Blendshapes: ARKit-52 coverage, neck boundary condition pinning, and JSON schema.
3. LODs: Multi-resolution decimation and shape-key re-projection across all 4 tiers.
4. Armature: Anatomical joint estimation and LBS skinning weight partition of unity.
5. End-to-end Stage5Exporter orchestrator.
"""
import unittest
import tempfile
import json
import numpy as np
from pathlib import Path

from src.stage5_export.retopology import (
    SparseMatrixCSR,
    compute_barycentric_weights_triangle,
    build_correspondence_matrix,
    save_correspondence_matrix,
    load_correspondence_matrix,
    transfer_retopology,
    compute_retopology_error
)
from src.stage5_export.blendshapes import (
    ARKIT_52_NAMES,
    synthesize_canonical_arkit_deltas,
    transfer_blendshapes_deformation,
    export_blendshapes_json,
    load_blendshapes_json,
    get_anatomical_region_weights
)
from src.stage5_export.lod import (
    decimate_mesh_spatial_clustering,
    reproject_blendshapes_to_lod,
    generate_lod_chain,
    export_lod_chain_assets
)
from src.stage5_export.armature import (
    JOINT_NAMES,
    estimate_anatomical_joint_centers,
    compute_linear_skinning_weights,
    export_armature_json,
    load_armature_json
)
from src.stage5_export.exporter import Stage5Exporter


def create_synthetic_face_mesh(n_rings=20, n_slices=20):
    """Generates a synthetic hemisphere face mesh with 500-1000 vertices."""
    verts = []
    # Base neck vertex
    verts.append([0.0, -100.0, 0.0])

    for i in range(1, n_rings):
        phi = (np.pi / 2.0) * (i / n_rings)
        y = -100.0 + 200.0 * (i / n_rings)
        r = 80.0 * np.sin(phi)
        for j in range(n_slices):
            theta = 2.0 * np.pi * (j / n_slices)
            x = r * np.cos(theta)
            z = r * np.sin(theta) + 20.0 * (1.0 - i / n_rings)
            verts.append([x, y, z])

    # Top skull vertex
    verts.append([0.0, 100.0, 0.0])
    verts = np.array(verts, dtype=np.float32)

    # Build faces
    faces = []
    # Bottom cap
    for j in range(n_slices):
        next_j = (j + 1) % n_slices
        faces.append([0, 1 + j, 1 + next_j])

    # Middle quads split into triangles
    for i in range(n_rings - 2):
        row_start = 1 + i * n_slices
        next_row_start = 1 + (i + 1) * n_slices
        for j in range(n_slices):
            next_j = (j + 1) % n_slices
            v0 = row_start + j
            v1 = row_start + next_j
            v2 = next_row_start + next_j
            v3 = next_row_start + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Top cap
    top_idx = len(verts) - 1
    last_row_start = 1 + (n_rings - 2) * n_slices
    for j in range(n_slices):
        next_j = (j + 1) % n_slices
        faces.append([last_row_start + j, top_idx, last_row_start + next_j])

    faces = np.array(faces, dtype=np.int32)
    return verts, faces


class TestStage5Retopology(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verts, self.faces = create_synthetic_face_mesh(15, 15)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sparse_matrix_csr_operations(self):
        # 3x4 test matrix
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        indices = np.array([0, 2, 1, 3], dtype=np.int32)
        indptr = np.array([0, 2, 3, 4], dtype=np.int32)
        mat = SparseMatrixCSR(data, indices, indptr, shape=(3, 4))

        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        res = mat.dot(vec)
        # row 0: 1*1 + 2*3 = 7
        # row 1: 3*2 = 6
        # row 2: 4*4 = 16
        np.testing.assert_allclose(res, [7.0, 6.0, 16.0], rtol=1e-5)

        # 2D matrix multiplication
        mat2d = np.ones((4, 2), dtype=np.float32)
        res2d = mat.dot(mat2d)
        self.assertEqual(res2d.shape, (3, 2))
        np.testing.assert_allclose(res2d[:, 0], [3.0, 3.0, 4.0], rtol=1e-5)

    def test_barycentric_weights_partition_of_unity(self):
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        pt = np.array([0.2, 0.3, 0.0])

        u, v, w = compute_barycentric_weights_triangle(pt, v0, v1, v2)
        self.assertAlmostEqual(u + v + w, 1.0, places=5)
        self.assertGreaterEqual(u, 0.0)
        self.assertGreaterEqual(v, 0.0)
        self.assertGreaterEqual(w, 0.0)

    def test_build_and_save_correspondence_matrix(self):
        # Build correspondence mapping from mesh to itself
        w_mat = build_correspondence_matrix(self.verts, self.faces, self.verts)
        self.assertEqual(w_mat.shape, (len(self.verts), len(self.verts)))

        # Transfer should reconstruct original mesh with near-zero error
        reconstructed = transfer_retopology(self.verts, w_mat)
        errors = compute_retopology_error(self.verts, reconstructed)
        self.assertLess(errors['mean_error_mm'], 1.0)

        # Save and load round-trip
        save_path = Path(self.temp_dir.name) / "w_test.npz"
        save_correspondence_matrix(w_mat, save_path)
        self.assertTrue(save_path.exists())

        loaded_w = load_correspondence_matrix(save_path)
        self.assertEqual(loaded_w.shape, w_mat.shape)
        np.testing.assert_allclose(loaded_w.data, w_mat.data, rtol=1e-5)


class TestStage5ARKitBlendshapes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verts, self.faces = create_synthetic_face_mesh(15, 15)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_arkit_52_catalogue_completeness(self):
        self.assertEqual(len(ARKIT_52_NAMES), 52)
        self.assertIn("jawOpen", ARKIT_52_NAMES)
        self.assertIn("mouthSmileLeft", ARKIT_52_NAMES)
        self.assertIn("eyeBlinkLeft", ARKIT_52_NAMES)
        self.assertIn("browInnerUp", ARKIT_52_NAMES)

    def test_synthesize_canonical_arkit_deltas(self):
        deltas = synthesize_canonical_arkit_deltas(self.verts)
        self.assertEqual(len(deltas), 52)

        # Verify jawOpen has movement
        jaw_delta = deltas['jawOpen']
        self.assertEqual(jaw_delta.shape, (len(self.verts), 3))
        max_jaw_disp = float(np.max(np.linalg.norm(jaw_delta, axis=1)))
        self.assertGreater(max_jaw_disp, 0.5)

        # Verify neck boundary pinning (lowest vertex should be 0.0)
        lowest_vert_idx = int(np.argmin(self.verts[:, 1]))
        for name, d in deltas.items():
            disp_norm = float(np.linalg.norm(d[lowest_vert_idx]))
            self.assertAlmostEqual(disp_norm, 0.0, places=4, msg=f"Neck vertex not pinned for {name}")

    def test_blendshapes_json_roundtrip(self):
        deltas = synthesize_canonical_arkit_deltas(self.verts)
        out_json = Path(self.temp_dir.name) / "test_bs.json"
        export_blendshapes_json(deltas, out_json)
        self.assertTrue(out_json.exists())

        loaded = load_blendshapes_json(out_json)
        self.assertEqual(len(loaded), 52)
        np.testing.assert_allclose(loaded['jawOpen'], deltas['jawOpen'], atol=1e-5)


class TestStage5LODDecimation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verts, self.faces = create_synthetic_face_mesh(20, 20)
        self.blendshapes = synthesize_canonical_arkit_deltas(self.verts)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mesh_decimation_reduces_faces(self):
        target_tri = 100
        new_v, new_f = decimate_mesh_spatial_clustering(self.verts, self.faces, target_tri)
        self.assertLess(len(new_f), len(self.faces))
        self.assertGreater(len(new_f), 0)
        self.assertGreater(len(new_v), 0)

    def test_reproject_blendshapes_preserves_all_52(self):
        target_tri = 100
        new_v, new_f = decimate_mesh_spatial_clustering(self.verts, self.faces, target_tri)
        lod_bs = reproject_blendshapes_to_lod(
            lod0_vertices=self.verts,
            lod0_faces=self.faces,
            lod0_blendshapes=self.blendshapes,
            lod_vertices=new_v
        )
        self.assertEqual(len(lod_bs), 52)
        for name, d in lod_bs.items():
            self.assertEqual(d.shape, (len(new_v), 3))

    def test_generate_and_export_lod_chain(self):
        targets = {'LOD0': None, 'LOD1': 300, 'LOD2': 150, 'LOD3': 50}
        chain = generate_lod_chain(
            lod0_vertices=self.verts,
            lod0_faces=self.faces,
            lod0_blendshapes=self.blendshapes,
            targets=targets
        )
        self.assertIn('LOD0', chain)
        self.assertIn('LOD1', chain)
        self.assertIn('LOD2', chain)
        self.assertIn('LOD3', chain)

        # Verify all LODs have 52 shape keys
        for lod_tier in ['LOD0', 'LOD1', 'LOD2', 'LOD3']:
            self.assertEqual(len(chain[lod_tier]['blendshapes']), 52)

        manifest = export_lod_chain_assets(chain, self.temp_dir.name)
        self.assertEqual(len(manifest), 4)
        for lod_tier in ['LOD0', 'LOD1', 'LOD2', 'LOD3']:
            self.assertTrue(Path(manifest[lod_tier]['mesh_obj']).exists())
            self.assertTrue(Path(manifest[lod_tier]['blendshapes_json']).exists())


class TestStage5Armature(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verts, self.faces = create_synthetic_face_mesh(15, 15)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_joint_center_estimation(self):
        joints = estimate_anatomical_joint_centers(self.verts)
        self.assertEqual(len(joints), 5)
        for j_name in JOINT_NAMES:
            self.assertIn(j_name, joints)
            self.assertEqual(joints[j_name].shape, (3,))

    def test_skinning_weights_partition_of_unity(self):
        joints = estimate_anatomical_joint_centers(self.verts)
        weights = compute_linear_skinning_weights(self.verts, joints)
        self.assertEqual(weights.shape, (len(self.verts), 5))

        # Check partition of unity (sum to 1.0 for every vertex)
        sums = np.sum(weights, axis=1)
        np.testing.assert_allclose(sums, np.ones(len(self.verts)), rtol=1e-5)

        # All weights non-negative
        self.assertTrue(np.all(weights >= 0.0))

    def test_export_armature_json(self):
        joints = estimate_anatomical_joint_centers(self.verts)
        weights = compute_linear_skinning_weights(self.verts, joints)
        arm_file = Path(self.temp_dir.name) / "armature.json"
        export_armature_json(joints, weights, arm_file)
        self.assertTrue(arm_file.exists())

        loaded_j, loaded_w = load_armature_json(arm_file)
        self.assertEqual(len(loaded_j), 5)
        self.assertEqual(loaded_w.shape, weights.shape)


class TestStage5ExporterMaster(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verts, self.faces = create_synthetic_face_mesh(15, 15)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_production_asset_complete(self):
        exporter = Stage5Exporter(enable_lods=True, enable_armature=True)
        manifest = exporter.export_production_asset(
            neutral_vertices=self.verts,
            faces=self.faces,
            output_dir=self.temp_dir.name,
            export_fbx=False  # Local tests skip external blender invocation
        )

        self.assertEqual(manifest['status'], "success")
        self.assertEqual(manifest['arkit_blendshapes_count'], 52)
        self.assertTrue(Path(manifest['neutral_base_obj']).exists())
        self.assertTrue(Path(manifest['blendshapes_json']).exists())
        self.assertTrue(Path(manifest['armature_json']).exists())
        self.assertEqual(len(manifest['lods']), 4)


if __name__ == '__main__':
    unittest.main()
