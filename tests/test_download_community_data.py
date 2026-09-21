"""
Unit tests for scripts/download_community_data.py.
Verifies dataset inventory auditing, license template generation, and sample scan creation.
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.download_community_data import (
    check_dataset_inventory,
    generate_license_templates,
    download_sample_scans,
    ACADEMIC_REQUEST_TEMPLATES,
)


class TestDownloadCommunityData(unittest.TestCase):
    def test_check_dataset_inventory(self):
        """Audits that inventory check returns a valid structured dictionary."""
        report = check_dataset_inventory()
        self.assertIn("flame_model", report)
        self.assertIn("external_scans", report)
        self.assertIn("synthetic_1024", report)
        self.assertIn("models_cache", report)
        self.assertIsInstance(report["flame_model"], bool)
        self.assertIsInstance(report["external_scans"], int)

    def test_generate_license_templates(self):
        """Verifies academic license template creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("scripts.download_community_data.LICENSES_DIR", tmp_path):
                generated = generate_license_templates()
                self.assertEqual(len(generated), len(ACADEMIC_REQUEST_TEMPLATES))
                for file_path in generated:
                    self.assertTrue(file_path.exists())
                    content = file_path.read_text()
                    self.assertIn("TO:", content)
                    self.assertIn("SUBJECT:", content)

    def test_download_sample_scans(self):
        """Verifies sample scan generation without corrupting workspace."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("scripts.download_community_data.SCANS_DIR", tmp_path):
                scans = download_sample_scans()
                # If FLAME model is present, one calibrated reference scan is produced
                for scan in scans:
                    self.assertTrue(scan.exists())
                    self.assertTrue(scan.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
