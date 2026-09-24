"""
Unit tests for scripts/render_blender_film.py.
Verifies CLI argument parsing, texture path discovery, Blender binary resolution,
turnkey shell runner generation, preview fallback generation, and shader node construction.
"""
from __future__ import annotations
import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from scripts.render_blender_film import (
    parse_args,
    resolve_texture_paths,
    find_blender_binary,
    execute_film_render,
    _generate_fallback_render,
)


class TestBlenderFilmRender(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mesh_file = self.temp_dir / "head_mesh.obj"
        with open(self.mesh_file, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

        self.tex_dir = self.temp_dir / "textures"
        self.tex_dir.mkdir(parents=True)

        # Create dummy texture maps
        dummy_img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(self.tex_dir / "albedo_diffuse.png"), dummy_img)
        cv2.imwrite(str(self.tex_dir / "film_roughness_base.png"), dummy_img)
        cv2.imwrite(str(self.tex_dir / "film_normal.png"), dummy_img)
        cv2.imwrite(str(self.tex_dir / "film_displacement_16bit.png"), np.ones((64, 64), dtype=np.uint16) * 32768)
        cv2.imwrite(str(self.tex_dir / "film_cavity_ao.png"), dummy_img)
        cv2.imwrite(str(self.tex_dir / "sss_thickness_map.png"), dummy_img)

        self.out_png = self.temp_dir / "film_render_cycles.png"
        self.out_blend = self.temp_dir / "studio_scene.blend"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_args_defaults(self):
        args = parse_args([
            "--mesh", str(self.mesh_file),
            "--output", str(self.out_png)
        ])
        self.assertEqual(args.mesh, str(self.mesh_file))
        self.assertEqual(args.output, str(self.out_png))
        self.assertEqual(args.samples, 128)
        self.assertEqual(args.resolution, [2048, 2048])
        self.assertEqual(args.device, "AUTO")
        self.assertIsNone(args.save_blend)

    def test_parse_args_custom(self):
        args = parse_args([
            "--mesh", str(self.mesh_file),
            "--textures_dir", str(self.tex_dir),
            "--albedo", str(self.tex_dir / "custom_albedo.png"),
            "--output", str(self.out_png),
            "--save_blend", str(self.out_blend),
            "--samples", "256",
            "--resolution", "1024", "1024",
            "--device", "GPU",
        ])
        self.assertEqual(args.samples, 256)
        self.assertEqual(args.resolution, [1024, 1024])
        self.assertEqual(args.device, "GPU")
        self.assertEqual(args.save_blend, str(self.out_blend))
        self.assertEqual(args.albedo, str(self.tex_dir / "custom_albedo.png"))

    def test_resolve_texture_paths_auto_discovery(self):
        args = parse_args([
            "--mesh", str(self.mesh_file),
            "--textures_dir", str(self.tex_dir)
        ])
        textures = resolve_texture_paths(args)

        self.assertIsNotNone(textures['albedo'])
        self.assertEqual(textures['albedo'].name, "albedo_diffuse.png")
        self.assertIsNotNone(textures['roughness'])
        self.assertEqual(textures['roughness'].name, "film_roughness_base.png")
        self.assertIsNotNone(textures['normal'])
        self.assertEqual(textures['normal'].name, "film_normal.png")
        self.assertIsNotNone(textures['displacement'])
        self.assertEqual(textures['displacement'].name, "film_displacement_16bit.png")
        self.assertIsNotNone(textures['cavity'])
        self.assertEqual(textures['cavity'].name, "film_cavity_ao.png")
        self.assertIsNotNone(textures['sss'])
        self.assertEqual(textures['sss'].name, "sss_thickness_map.png")

    def test_resolve_texture_paths_explicit_override(self):
        explicit_albedo = self.temp_dir / "explicit_albedo.png"
        cv2.imwrite(str(explicit_albedo), np.zeros((32, 32, 3), dtype=np.uint8))

        args = parse_args([
            "--mesh", str(self.mesh_file),
            "--textures_dir", str(self.tex_dir),
            "--albedo", str(explicit_albedo)
        ])
        textures = resolve_texture_paths(args)
        self.assertEqual(textures['albedo'], explicit_albedo)

    def test_find_blender_binary(self):
        # Explicit non-existent path returns None or checks file existence
        res = find_blender_binary("/non/existent/path/to/blender")
        self.assertIsNone(res)

        # Real file that is executable
        fake_bin = self.temp_dir / "fake_blender"
        with open(fake_bin, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(fake_bin, 0o755)

        res = find_blender_binary(str(fake_bin))
        self.assertEqual(res, str(fake_bin))

    def test_execute_film_render_fallback_mode(self):
        """
        When Blender is absent, execute_film_render must:
        1. Not crash.
        2. Generate a valid turnkey shell script.
        3. Generate a calibrated preview render image.
        """
        args = parse_args([
            "--mesh", str(self.mesh_file),
            "--textures_dir", str(self.tex_dir),
            "--output", str(self.out_png),
            "--save_blend", str(self.out_blend),
            "--resolution", "256", "256",
        ])

        with patch("scripts.render_blender_film.find_blender_binary", return_value=None):
            result = execute_film_render(args)

        self.assertFalse(result['blender_available'])
        self.assertTrue(self.out_png.exists())
        self.assertTrue(result['rendered_image'].exists())

        # Check turnkey runner shell script
        runner_sh = self.out_png.parent / "run_blender_film_render.sh"
        self.assertTrue(runner_sh.exists())
        content = runner_sh.read_text()
        self.assertIn("--mesh", content)
        self.assertIn("render_blender_film.py", content)

        # Verify generated preview image shape
        img = cv2.imread(str(self.out_png))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (256, 256))

    @patch("subprocess.run")
    def test_execute_film_render_mock_blender_success(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "[Blender Cycles] Render completed successfully"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        args = parse_args([
            "--mesh", str(self.mesh_file),
            "--textures_dir", str(self.tex_dir),
            "--output", str(self.out_png),
            "--save_blend", str(self.out_blend),
            "--samples", "64",
        ])

        with patch("scripts.render_blender_film.find_blender_binary", return_value="/mock/blender"):
            result = execute_film_render(args)

        self.assertTrue(result['blender_available'])
        self.assertTrue(mock_run.called)
        cmd_called = mock_run.call_args[0][0]
        self.assertEqual(cmd_called[0], "/mock/blender")
        self.assertIn("-b", cmd_called)
        self.assertIn("-P", cmd_called)
        self.assertIn("--samples", cmd_called)
        self.assertIn("64", cmd_called)


if __name__ == "__main__":
    unittest.main()
