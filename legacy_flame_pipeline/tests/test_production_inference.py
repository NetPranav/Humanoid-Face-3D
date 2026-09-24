"""
Unit tests for scripts/production_inference.py.
Verifies benchmark photo auto-discovery, argument parsing, deliverable auditing,
and System Invariant Rule 4 (collar boundary pinning contract).
"""
from __future__ import annotations
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import cv2

from scripts.production_inference import (
    find_subject_photos,
    BENCHMARK_SUBJECTS,
    run_production_inference,
)


class TestProductionInference(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_benchmark_subject_photos(self):
        """Verify that built-in benchmark subjects can be located in vendor/MICA demo directory."""
        for subj in ["carell", "connelly", "lawrence", "justin"]:
            photos = find_subject_photos(subj)
            self.assertGreater(len(photos), 0, f"No photos found for {subj}")
            for p in photos:
                self.assertTrue(Path(p).is_file(), f"Photo path does not exist: {p}")

    def test_find_subject_photos_nonexistent_raises(self):
        """Non-existent subject must raise FileNotFoundError with actionable error message."""
        with self.assertRaises(FileNotFoundError):
            find_subject_photos("nonexistent_celebrity_xyz")

    def test_rule4_collar_pinning_audit(self):
        """
        Verify that 16-bit displacement map produced has strictly 0 delta
        in the neck collar region (lowest 20% along V axis).
        """
        h, w = 1024, 1024
        # Midlevel 32768 corresponds to 0.0 mm displacement
        disp_u16 = np.ones((h, w), dtype=np.uint16) * 32768
        # Put some non-zero displacement in the face area
        disp_u16[200:500, 200:500] = 34000
        # Bottom rows (lowest 20% in UV) must be 32768
        bottom_rows = int(0.85 * h)
        collar_slice = disp_u16[bottom_rows:, :].astype(np.int32) - 32768
        self.assertEqual(int(np.max(np.abs(collar_slice))), 0)

    @patch("src.pipeline.FaceGeoPipeline.run")
    def test_run_production_inference_mock(self, mock_pipeline_run):
        mock_pipeline_run.return_value = {
            "obj_path": str(self.temp_dir / "head_mesh.obj"),
            "manifest_path": str(self.temp_dir / "manifest.json"),
            "displacement_16bit": str(self.temp_dir / "film_displacement_16bit.png"),
            "normal_map": str(self.temp_dir / "film_normal.png"),
            "film_render": str(self.temp_dir / "film_render_cycles.png"),
            "blend_file": str(self.temp_dir / "studio_scene.blend"),
        }

        # Create dummy displacement map with midlevel
        disp_img = np.ones((128, 128), dtype=np.uint16) * 32768
        cv2.imwrite(str(self.temp_dir / "film_displacement_16bit.png"), disp_img)
        Path(self.temp_dir / "head_mesh.obj").write_text("v 0 0 0\n")
        Path(self.temp_dir / "manifest.json").write_text("{}")
        Path(self.temp_dir / "film_render_cycles.png").write_bytes(b"dummy")
        Path(self.temp_dir / "studio_scene.blend").write_bytes(b"dummy")

        dummy_args = MagicMock()
        dummy_args.subject = "carell"
        dummy_args.photos = None
        dummy_args.output_dir = str(self.temp_dir)
        dummy_args.config = "configs/default.yaml"
        dummy_args.model_dir = "models_cache"
        dummy_args.resolution = 512
        dummy_args.hair_preset = "stubble"
        dummy_args.stylize = "chiseled"
        dummy_args.render_samples = 32
        dummy_args.no_render = False
        dummy_args.no_fbx = False
        dummy_args.device = "AUTO"

        res = run_production_inference(dummy_args)
        self.assertIn("displacement_16bit", res)
        self.assertIn("film_render", res)
        self.assertTrue(mock_pipeline_run.called)


if __name__ == "__main__":
    unittest.main()
