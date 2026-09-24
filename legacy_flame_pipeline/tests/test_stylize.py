"""
Comprehensive Unit Tests for Stage 5 Parametric Facial Stylization.

Validates:
1. Parameter resolution, preset catalogue, clamping, and error guards.
2. Bilateral sagittal symmetry across the midline (x = 0).
3. Critical System Invariant 4: Strict Neck Seam Contract pinning (Delta v = 0 on lowest 20% collar).
4. Scale-invariance across metric systems (meters vs. millimeters).
5. Anatomical slider responsiveness (jaw, chin, zygoma, brow, gonion).
6. End-to-end integration with Stage5Exporter and manifest packaging.
"""
import unittest
import tempfile
import json
import numpy as np
from pathlib import Path

from src.stage5_export.stylize import (
    StylizationParameters,
    STYLIZATION_PRESETS,
    resolve_stylization_params,
    FaceStylizer,
    compute_hermite_smoothstep
)
from src.stage5_export.exporter import Stage5Exporter


def create_test_face_mesh(n_rings=24, n_slices=24, scale=1.0):
    """
    Generates a synthetic hemisphere face mesh with bilateral symmetry.
    Scale = 1.0 produces ~200mm tall mesh; scale = 0.001 produces ~0.2m tall mesh.
    """
    verts = []
    # Base neck vertex at bottom center
    verts.append([0.0, -100.0, 0.0])

    for i in range(1, n_rings):
        phi = (np.pi / 2.0) * (i / n_rings)
        y = -100.0 + 200.0 * (i / n_rings)
        r = 80.0 * np.sin(phi)
        for j in range(n_slices):
            theta = 2.0 * np.pi * (j / n_slices)
            x = r * np.cos(theta)
            # Anterior face profile projection
            z = r * np.sin(theta) + 30.0 * (1.0 - i / n_rings)
            verts.append([x, y, z])

    # Top skull vertex
    verts.append([0.0, 100.0, 0.0])
    verts = (np.array(verts, dtype=np.float32) * scale).astype(np.float32)

    # Build faces
    faces = []
    for j in range(n_slices):
        next_j = (j + 1) % n_slices
        faces.append([0, 1 + j, 1 + next_j])

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

    top_idx = len(verts) - 1
    last_row_start = 1 + (n_rings - 2) * n_slices
    for j in range(n_slices):
        next_j = (j + 1) % n_slices
        faces.append([last_row_start + j, top_idx, last_row_start + next_j])

    faces = np.array(faces, dtype=np.int32)
    return verts, faces


class TestStylizationParameters(unittest.TestCase):
    def test_default_is_neutral(self):
        p = StylizationParameters()
        self.assertTrue(p.is_neutral())
        for val in p.to_dict().values():
            self.assertEqual(val, 0.0)

    def test_clamping(self):
        p = StylizationParameters(
            jaw_width=2.5,
            jaw_squareness=-3.0,
            chin_cleft=1.5,
            brow_prominence=-0.5
        )
        clamped = p.clamp()
        self.assertEqual(clamped.jaw_width, 1.0)
        self.assertEqual(clamped.jaw_squareness, -1.0)
        self.assertEqual(clamped.chin_cleft, 1.0)
        self.assertEqual(clamped.brow_prominence, -0.5)

    def test_resolve_presets(self):
        self.assertIn("chiseled", STYLIZATION_PRESETS)
        self.assertIn("gigachad", STYLIZATION_PRESETS)
        self.assertIn("heroic", STYLIZATION_PRESETS)
        self.assertIn("soft_oval", STYLIZATION_PRESETS)
        self.assertIn("round", STYLIZATION_PRESETS)

        p = resolve_stylization_params("chiseled")
        self.assertFalse(p.is_neutral())
        self.assertGreater(p.jaw_width, 0.0)
        self.assertGreater(p.chin_depth, 0.0)

    def test_resolve_dict_and_kwargs(self):
        p = resolve_stylization_params({"jaw_width": 0.5}, chin_depth=0.8)
        self.assertAlmostEqual(p.jaw_width, 0.5)
        self.assertAlmostEqual(p.chin_depth, 0.8)
        self.assertEqual(p.chin_width, 0.0)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            resolve_stylization_params("non_existent_preset")

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_stylization_params(12345)


class TestFaceStylizer(unittest.TestCase):
    def setUp(self):
        self.verts, self.faces = create_test_face_mesh(24, 24, scale=1.0)
        self.stylizer = FaceStylizer(neck_collar_threshold=0.20, collar_transition=0.12)

    def test_neutral_preserves_geometry(self):
        stylized, deltas = self.stylizer.apply_stylization(self.verts, "neutral")
        np.testing.assert_allclose(stylized, self.verts)
        np.testing.assert_allclose(deltas, 0.0)

    def test_neck_seam_contract_strict_pinning(self):
        """
        Critical System Invariant 4: Lowest 20% collar vertices must have
        Delta v == 0.000000 (no seam tearing with UE5 torso).
        """
        y = self.verts[:, 1]
        y_min, y_max = float(np.min(y)), float(np.max(y))
        y_norm = (y - y_min) / (y_max - y_min)
        collar_mask = y_norm <= 0.20

        self.assertGreater(np.sum(collar_mask), 0, "Collar test vertices must exist")

        # Test against aggressive presets
        for preset_name in ["chiseled", "gigachad", "heroic"]:
            _, deltas = self.stylizer.apply_stylization(self.verts, preset_name)
            collar_deltas = deltas[collar_mask]
            np.testing.assert_allclose(
                collar_deltas,
                0.0,
                atol=1e-7,
                err_msg=f"Neck boundary collar violated contract for preset '{preset_name}'!"
            )

    def test_bilateral_sagittal_symmetry(self):
        """
        Deformations on symmetric input points must be symmetric across x = 0:
        Delta x(-x) = -Delta x(x), Delta y(-x) = Delta y(x), Delta z(-x) = Delta z(x).
        """
        # Create a perfectly symmetric pair of points in the jaw region
        test_pts = np.array([
            [30.0, -40.0, 40.0],   # left jaw point
            [-30.0, -40.0, 40.0],  # right jaw point
            [0.0, -100.0, 0.0],    # neck bottom anchor
            [0.0, 100.0, 0.0],     # skull top anchor
        ], dtype=np.float32)

        _, deltas = self.stylizer.apply_stylization(test_pts, "chiseled")

        # Left point (idx 0) vs Right point (idx 1)
        # Delta X should be anti-symmetric
        self.assertAlmostEqual(deltas[0, 0], -deltas[1, 0], places=5)
        # Delta Y and Z should be symmetric
        self.assertAlmostEqual(deltas[0, 1], deltas[1, 1], places=5)
        self.assertAlmostEqual(deltas[0, 2], deltas[1, 2], places=5)

    def test_scale_invariance(self):
        """
        Normalized deformation (Delta v / H) must be identical whether
        mesh is in millimeters (~200 mm) or meters (~0.2 m).
        """
        verts_mm, _ = create_test_face_mesh(20, 20, scale=1.0)      # H ~ 200 mm
        verts_m, _ = create_test_face_mesh(20, 20, scale=0.001)     # H ~ 0.20 m

        _, deltas_mm = self.stylizer.apply_stylization(verts_mm, "heroic")
        _, deltas_m = self.stylizer.apply_stylization(verts_m, "heroic")

        H_mm = float(np.max(verts_mm[:, 1]) - np.min(verts_mm[:, 1]))
        H_m = float(np.max(verts_m[:, 1]) - np.min(verts_m[:, 1]))

        norm_deltas_mm = deltas_mm / H_mm
        norm_deltas_m = deltas_m / H_m

        np.testing.assert_allclose(norm_deltas_mm, norm_deltas_m, atol=1e-5)

    def test_anatomical_slider_responsiveness(self):
        """Verify each individual slider induces expected localized deformation."""
        # 1. Jaw width: increases lateral displacement on lower jaw
        _, d_jaw = self.stylizer.apply_stylization(self.verts, jaw_width=0.8)
        self.assertGreater(float(np.max(np.abs(d_jaw[:, 0]))), 1.0)

        # 2. Chin depth: increases anterior Z displacement on chin
        _, d_chin = self.stylizer.apply_stylization(self.verts, chin_depth=0.8)
        self.assertGreater(float(np.max(d_chin[:, 2])), 1.0)

        # 3. Chin cleft: creates central depression with negative delta Z
        _, d_cleft = self.stylizer.apply_stylization(self.verts, chin_cleft=0.8)
        self.assertLess(float(np.min(d_cleft[:, 2])), -0.2)

        # 4. Cheekbone prominence: anterolateral displacement in upper midface
        _, d_cb = self.stylizer.apply_stylization(self.verts, cheekbone_prominence=0.8)
        self.assertGreater(float(np.max(np.abs(d_cb[:, 0]))), 0.5)
        self.assertGreater(float(np.max(d_cb[:, 2])), 0.5)

        # 5. Brow prominence: anterior displacement on forehead/brow
        _, d_brow = self.stylizer.apply_stylization(self.verts, brow_prominence=0.8)
        self.assertGreater(float(np.max(d_brow[:, 2])), 0.5)

    def test_input_validation_and_safety(self):
        # NaN detection
        bad_verts = self.verts.copy()
        bad_verts[5, 0] = np.nan
        with self.assertRaises(ValueError):
            self.stylizer.apply_stylization(bad_verts, "chiseled")

        # Wrong shape detection
        with self.assertRaises(ValueError):
            self.stylizer.apply_stylization(self.verts[:, :2], "chiseled")


class TestStage5ExporterStylizeIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verts, self.faces = create_test_face_mesh(16, 16, scale=1.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exporter_with_chiseled_preset(self):
        exporter = Stage5Exporter(enable_lods=True, enable_armature=True)
        manifest = exporter.export_production_asset(
            neutral_vertices=self.verts,
            faces=self.faces,
            output_dir=self.temp_dir.name,
            export_fbx=False,
            stylization_params="chiseled"
        )

        self.assertEqual(manifest['status'], "success")
        self.assertIn("stylization", manifest)
        self.assertIsNotNone(manifest["stylization"])
        self.assertGreater(manifest["stylization"]["max_displacement"], 0.0)

        # Check that manifest file on disk contains the stylization data
        manifest_file = Path(self.temp_dir.name) / "stage5_export_manifest.json"
        self.assertTrue(manifest_file.exists())
        with open(manifest_file, "r") as fp:
            data = json.load(fp)
            self.assertIn("stylization", data)
            self.assertAlmostEqual(data["stylization"]["parameters"]["jaw_width"], 0.55)

        # Verify the exported neutral OBJ has strict collar preservation
        exported_obj = Path(manifest["neutral_base_obj"])
        self.assertTrue(exported_obj.exists())

        # Parse vertices from OBJ
        exported_verts = []
        with open(exported_obj, "r") as fp:
            for line in fp:
                if line.startswith("v "):
                    parts = [float(p) for p in line.strip().split()[1:4]]
                    exported_verts.append(parts)
        exported_verts = np.array(exported_verts, dtype=np.float32)

        # Verify neck collar vertices are identical
        y = self.verts[:, 1]
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        collar_idx = np.where(y_norm <= 0.20)[0]

        np.testing.assert_allclose(
            exported_verts[collar_idx],
            self.verts[collar_idx],
            atol=1e-5,
            err_msg="Exported OBJ neck vertices did not preserve collar pinning!"
        )


if __name__ == '__main__':
    unittest.main()
