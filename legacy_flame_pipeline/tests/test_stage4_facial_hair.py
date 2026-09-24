"""
Unit tests for Stage 4: Facial Hair & Stubble Engine.
Tests anatomical region weighting, procedural stubble micro-displacement,
static 3D hair card polygonal mesh generation, textures, and pipeline integration.
"""
from __future__ import annotations
import unittest
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np

from src.stage4_facial_hair.regions import (
    FacialHairConfig,
    FacialHairRegionSegmenter,
    resolve_hair_config,
    FACIAL_HAIR_PRESETS,
    compute_collar_pinning_mask,
    compute_normalized_coordinates,
)
from src.stage4_facial_hair.stubble import ProceduralStubbleEngine
from src.stage4_facial_hair.cards import HairCardGenerator
from src.stage4_facial_hair.generator import FacialHairGenerator


def create_synthetic_head_mesh(n_rings: int = 25, n_pts_per_ring: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates an ellipsoid head mesh roughly spanning [-70, 70] in X,
    [-110, 110] in Y (neck base to cranium), and [-80, 80] in Z (posterior to anterior).
    """
    verts = []
    for i in range(n_rings):
        phi = np.pi * i / (n_rings - 1)  # 0 to pi
        y = 110.0 * np.cos(phi)
        r = 75.0 * np.sin(phi)
        for j in range(n_pts_per_ring):
            theta = 2.0 * np.pi * j / n_pts_per_ring
            x = r * np.sin(theta)
            # Flatten back, project front
            z = r * np.cos(theta)
            if z > 0:
                z = z * 1.15  # Anterior protrusion (face)
            verts.append([x, y, z])

    vertices = np.array(verts, dtype=np.float32)

    # Generate faces
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_pts_per_ring):
            next_j = (j + 1) % n_pts_per_ring
            v00 = i * n_pts_per_ring + j
            v01 = i * n_pts_per_ring + next_j
            v10 = (i + 1) * n_pts_per_ring + j
            v11 = (i + 1) * n_pts_per_ring + next_j
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])

    faces = np.array(faces, dtype=np.int32)
    return vertices, faces


class TestFacialHairRegions(unittest.TestCase):
    """Tests for anatomical region segmentation and presets."""

    def test_presets_and_clamping(self):
        cfg = resolve_hair_config('stubble')
        self.assertAlmostEqual(cfg.stubble_length_mm, 0.55)
        self.assertTrue(cfg.generate_stubble)
        self.assertFalse(cfg.generate_cards)

        cfg_clean = resolve_hair_config('clean_shaven')
        self.assertTrue(cfg_clean.is_clean_shaven())
        self.assertFalse(cfg_clean.generate_cards)

        # Clamping
        wild = FacialHairConfig(
            mustache_density=10.0,
            chin_density=-5.0,
            stubble_length_mm=100.0,
            card_density=50000,
        ).clamp()
        self.assertEqual(wild.mustache_density, 1.0)
        self.assertEqual(wild.chin_density, 0.0)
        self.assertEqual(wild.stubble_length_mm, 3.0)
        self.assertEqual(wild.card_density, 1000)

    def test_collar_pinning_contract(self):
        """CRITICAL: Neck Seam Contract — lower 20% collar vertices MUST have strictly 0.0 weight."""
        verts, faces = create_synthetic_head_mesh()
        segmenter = FacialHairRegionSegmenter(neck_collar_threshold=0.20)
        res = segmenter.extract_region_weights(verts, 'full_beard')

        x_norm, y_norm, z_norm, _, _, _, _ = compute_normalized_coordinates(verts)
        collar_mask = y_norm <= 0.20
        self.assertTrue(np.any(collar_mask))

        # Composite hair mask on collar vertices must be strictly 0.0
        np.testing.assert_allclose(res['composite'][collar_mask], 0.0, atol=1e-7)

    def test_anatomical_localization(self):
        """Tests that mustache, chin, and eyebrows activate at their anatomical locations."""
        verts, faces = create_synthetic_head_mesh(n_rings=35, n_pts_per_ring=35)
        segmenter = FacialHairRegionSegmenter()
        res = segmenter.extract_region_weights(verts, 'full_beard')

        x_norm, y_norm, z_norm, _, _, _, _ = compute_normalized_coordinates(verts)

        # Mustache should be near y_norm ~ 0.40, center x, anterior z
        mustache_idx = np.argmax(res['mustache'])
        self.assertTrue(0.32 <= y_norm[mustache_idx] <= 0.48)
        self.assertLess(abs(x_norm[mustache_idx]), 0.35)
        self.assertGreater(z_norm[mustache_idx], 0.60)

        # Chin should be near y_norm ~ 0.26, center x, anterior z
        chin_idx = np.argmax(res['chin'])
        self.assertTrue(0.20 <= y_norm[chin_idx] <= 0.35)
        self.assertLess(abs(x_norm[chin_idx]), 0.35)

        # Eyebrows should be near y_norm ~ 0.64
        res_brow = segmenter.extract_region_weights(verts, 'eyebrows_only')
        brow_idx = np.argmax(res_brow['eyebrows'])
        self.assertTrue(0.58 <= y_norm[brow_idx] <= 0.70)


class TestProceduralStubbleEngine(unittest.TestCase):
    """Tests for follicular stubble micro-displacement engine."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.verts, self.faces = create_synthetic_head_mesh()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stubble_displacement_generation(self):
        engine = ProceduralStubbleEngine()
        res = engine.generate_stubble_displacement(
            vertices=self.verts,
            faces=self.faces,
            config='stubble',
            resolution=128
        )

        disp = res['displacement_mm']
        normal_rgb = res['normal_map']

        self.assertEqual(disp.shape, (128, 128))
        self.assertEqual(normal_rgb.shape, (128, 128, 3))
        self.assertGreater(float(np.max(disp)), 0.0)

        # Tangent normal map dominant blue (facing out)
        # Blue channel should have high average value
        self.assertGreater(float(np.mean(normal_rgb[:, :, 2])), 150.0)

    def test_collar_pinning_in_stubble(self):
        """CRITICAL: Neck collar in UV space must have zero stubble displacement."""
        engine = ProceduralStubbleEngine(neck_collar_threshold=0.20)
        res = engine.generate_stubble_displacement(
            vertices=self.verts,
            faces=self.faces,
            config='stubble',
            resolution=128
        )
        collar_mask = res['collar_mask_uv']
        stubble_only = res['stubble_only_mm']

        zero_collar_pixels = stubble_only[collar_mask <= 1e-5]
        if len(zero_collar_pixels) > 0:
            np.testing.assert_allclose(zero_collar_pixels, 0.0, atol=1e-7)

    def test_16bit_encoding_and_save(self):
        engine = ProceduralStubbleEngine()
        disp_test = np.array([[-0.5, 0.0], [0.5, 1.0]], dtype=np.float32)
        encoded = engine.encode_16bit_displacement(disp_test, p99_scale=1.0)
        self.assertEqual(encoded.dtype, np.uint16)
        # 0.0 mm must encode to exactly 32768
        self.assertEqual(encoded[0, 1], 32768)
        self.assertLess(encoded[0, 0], 32768)
        self.assertGreater(encoded[1, 0], 32768)

        # Test save_maps
        stubble_data = {
            'displacement_mm': np.zeros((64, 64), dtype=np.float32),
            'normal_map': np.full((64, 64, 3), 128, dtype=np.uint8),
        }
        paths = engine.save_maps(stubble_data, self.temp_dir, prefix="test")
        self.assertTrue(Path(paths['displacement_png']).exists())
        self.assertTrue(Path(paths['normal_png']).exists())


class TestHairCardGenerator(unittest.TestCase):
    """Tests for static 3D hair card polygonal mesh and textures."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.verts, self.faces = create_synthetic_head_mesh()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hair_cards_topology_and_skinning(self):
        generator = HairCardGenerator()
        cfg = FacialHairConfig(
            mustache_density=1.0,
            chin_density=1.0,
            generate_cards=True,
            card_density=20,
            card_length_mm=8.0,
        )
        cards = generator.generate_hair_cards(self.verts, self.faces, config=cfg)

        num_cards = cards['num_cards']
        self.assertGreater(num_cards, 0)
        # 2-segment quad ribbon: 6 vertices, 4 triangles per card
        self.assertEqual(len(cards['card_vertices']), num_cards * 6)
        self.assertEqual(len(cards['card_faces']), num_cards * 4)

        # UV bounds
        uvs = cards['card_uvs']
        self.assertTrue(np.all(uvs >= 0.0) and np.all(uvs <= 1.0))

        # Skinning weights partition of unity
        skin = cards['skinning_weights']
        self.assertEqual(skin.shape, (num_cards * 6, 5))
        sums = np.sum(skin, axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

        # Lower face cards should have high jaw weight
        self.assertTrue(np.any(skin[:, 2] > 0.5))

    def test_procedural_textures_and_obj_export(self):
        generator = HairCardGenerator()
        alpha, normal_rgb = generator.generate_hair_card_textures(resolution=128)
        self.assertEqual(alpha.shape, (128, 128))
        self.assertEqual(normal_rgb.shape, (128, 128, 3))
        self.assertGreater(np.max(alpha), 200)

        # Export OBJ
        cfg = FacialHairConfig(
            mustache_density=1.0,
            generate_cards=True,
            card_density=10,
        )
        cards = generator.generate_hair_cards(self.verts, self.faces, config=cfg)
        obj_file = self.temp_dir / "cards.obj"
        out_p = generator.export_cards_obj(cards, obj_file)
        self.assertTrue(Path(out_p).exists())

        # Verify OBJ structure
        with open(obj_file, 'r') as f:
            content = f.read()
            self.assertIn("v ", content)
            self.assertIn("vt ", content)
            self.assertIn("vn ", content)
            self.assertIn("f ", content)


class TestFacialHairMasterGenerator(unittest.TestCase):
    """Tests for master FacialHairGenerator orchestrator."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.verts, self.faces = create_synthetic_head_mesh()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_subsystem_execution(self):
        generator = FacialHairGenerator()
        res = generator.generate(
            neutral_vertices=self.verts,
            faces=self.faces,
            config='full_beard',
            output_dir=self.temp_dir,
            resolution=128,
        )

        manifest = res['manifest']
        self.assertEqual(manifest['status'], 'success')
        self.assertFalse(manifest['is_clean_shaven'])
        self.assertTrue(manifest['stubble']['generated'])
        self.assertTrue(manifest['hair_cards']['generated'])

        # Verify created files in facial_hair/
        hair_dir = self.temp_dir / "facial_hair"
        self.assertTrue((hair_dir / "facial_hair_manifest.json").exists())
        self.assertTrue((hair_dir / "facial_hair_cards.obj").exists())
        self.assertTrue((hair_dir / "hair_card_alpha.png").exists())
        self.assertTrue((hair_dir / "hair_card_normal.png").exists())
        self.assertTrue((hair_dir / "hair_cards_skinning.json").exists())
        self.assertTrue((hair_dir / "head_stubble_displacement_16bit.png").exists())
        self.assertTrue((hair_dir / "head_stubble_normal_map.png").exists())

    def test_exporter_incorporates_facial_hair_manifest(self):
        from src.stage5_export.exporter import Stage5Exporter
        exporter = Stage5Exporter(enable_lods=False, enable_armature=True)
        fake_hair_manifest = {
            "status": "success",
            "is_clean_shaven": False,
            "hair_cards": {"num_cards": 120}
        }
        manifest = exporter.export_production_asset(
            neutral_vertices=self.verts,
            faces=self.faces,
            output_dir=self.temp_dir,
            export_fbx=False,
            facial_hair_manifest=fake_hair_manifest,
        )
        self.assertEqual(manifest['facial_hair'], fake_hair_manifest)
        with open(self.temp_dir / "stage5_export_manifest.json") as f:
            saved_manifest = json.load(f)
            self.assertEqual(saved_manifest['facial_hair'], fake_hair_manifest)


if __name__ == '__main__':
    unittest.main()
