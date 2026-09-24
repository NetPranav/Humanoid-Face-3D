"""
Unit tests for Stage 5 FBXPackager and headless Blender integration.
"""
from __future__ import annotations
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.stage5_export.fbx_packager import FBXPackager


class TestFBXPackager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mesh_file = self.temp_dir / "head_mesh.obj"
        with open(self.mesh_file, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

        self.bs_file = self.temp_dir / "blendshapes.json"
        with open(self.bs_file, "w") as f:
            json.dump({"blendshapes": {"mouthOpen": [[0, -1, 0], [0, -1, 0], [0, -1, 0]]}}, f)

        self.out_fbx = self.temp_dir / "head_mesh_ue5_livelink.fbx"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_blender_binary(self):
        packager = FBXPackager()
        # Should return None or a valid string path
        bin_path = packager.find_blender_binary()
        if bin_path is not None:
            self.assertIsInstance(bin_path, str)

    def test_input_validation(self):
        packager = FBXPackager()
        nonexistent = self.temp_dir / "does_not_exist.obj"
        with self.assertRaises(FileNotFoundError):
            packager.package_fbx(
                mesh_obj=nonexistent,
                blendshapes_json=self.bs_file,
                output_fbx=self.out_fbx,
            )

        with self.assertRaises(FileNotFoundError):
            packager.package_fbx(
                mesh_obj=self.mesh_file,
                blendshapes_json=nonexistent,
                output_fbx=self.out_fbx,
            )

    def test_dry_run_script_generation_when_blender_missing(self):
        # Force blender_bin = None
        packager = FBXPackager(blender_path=None)
        packager.blender_bin = None

        res = packager.package_fbx(
            mesh_obj=self.mesh_file,
            blendshapes_json=self.bs_file,
            output_fbx=self.out_fbx,
            armature_json=self.mesh_file,  # dummy exists
        )

        self.assertEqual(res["status"], "dry_run_script_generated")
        self.assertIsNone(res["fbx_file"])
        sh_path = Path(res["shell_script"])
        self.assertTrue(sh_path.exists())

        content = sh_path.read_text()
        self.assertIn("--mesh", content)
        self.assertIn("--blendshapes", content)
        self.assertIn("--output", content)
        self.assertIn("--armature", content)

    @patch("subprocess.run")
    def test_mock_blender_success(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "[Blender Export] Clean FBX export completed: head_mesh_ue5_livelink.fbx"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        packager = FBXPackager(blender_path="/mock/blender")
        res = packager.package_fbx(
            mesh_obj=self.mesh_file,
            blendshapes_json=self.bs_file,
            output_fbx=self.out_fbx,
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fbx_file"], str(self.out_fbx.resolve()))
        self.assertTrue(mock_run.called)

    @patch("subprocess.run")
    def test_mock_blender_failure(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Error: Out of memory"
        mock_run.return_value = mock_proc

        packager = FBXPackager(blender_path="/mock/blender")
        res = packager.package_fbx(
            mesh_obj=self.mesh_file,
            blendshapes_json=self.bs_file,
            output_fbx=self.out_fbx,
        )

        self.assertEqual(res["status"], "blender_error")
        self.assertIsNone(res["fbx_file"])
        self.assertIn("Error: Out of memory", res["error_message"])


if __name__ == '__main__':
    unittest.main()
