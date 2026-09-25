"""
Phase 0 regression tests (DOCS/04_roadmap.md §Phase 0).
Tests needing weights skip when models_cache/mica/mica.tar or the extracted FLAME are absent.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MICA = ROOT / "models_cache" / "mica" / "mica.tar"
FLAME = ROOT / "data" / "flame_model" / "generic_model.pkl"
LMK = ROOT / "data" / "flame_model" / "landmark_embedding_mica.npz"


class TestExifIntrinsics(unittest.TestCase):
    def _jpeg(self, d, exif_fields):
        from PIL import Image
        im = Image.new("RGB", (600, 800), (120, 120, 120))
        exif = Image.Exif()
        if exif_fields:
            exif[0x8769] = dict(exif_fields)
        p = Path(d) / "x.jpg"
        im.save(p, exif=exif)
        return p

    def test_focal_plane_resolution(self):
        from src.stage0_preprocess.camera import intrinsics_from_exif
        with tempfile.TemporaryDirectory() as d:
            p = self._jpeg(d, {0x920A: 100.0, 0xA20E: 200.0, 0xA210: 4})  # 100 mm, 200 px/mm
            K = intrinsics_from_exif(p, (800, 600))
            self.assertAlmostEqual(K.fx, 20000.0, places=3)
            self.assertEqual((K.cx, K.cy), (300.0, 400.0))
            self.assertEqual(K.source, "exif_focal_plane")

    def test_no_exif_returns_none(self):
        from src.stage0_preprocess.camera import intrinsics_from_exif
        with tempfile.TemporaryDirectory() as d:
            p = self._jpeg(d, {})
            self.assertIsNone(intrinsics_from_exif(p, (800, 600)))


@unittest.skipUnless(FLAME.exists() and LMK.exists(), "FLAME not extracted (scripts/extract_flame_from_mica.py)")
class TestLandmarkFitter(unittest.TestCase):
    def test_recovers_synthetic_pose_and_distance(self):
        import torch
        from src.stage0_preprocess.camera import Intrinsics
        from src.stage2_expression.landmark_fit import F_FLIP, LandmarkFitter
        from src.utils.flame_torch import batch_rodrigues, load_flame

        fl = load_flame()
        fitter = LandmarkFitter(fl, iters=400)
        rng = np.random.default_rng(1)
        beta = (rng.normal(size=300) * 0.3).astype(np.float32)
        psi = np.zeros(100, np.float32)
        psi[:10] = rng.normal(size=10) * 0.5
        pose = np.zeros(15, np.float32)
        pose[6] = 0.1
        with torch.no_grad():
            v = fitter.layer(torch.tensor(beta), torch.tensor(psi), torch.tensor(pose))
            R = F_FLIP @ batch_rodrigues(torch.tensor([0.05, 0.3, 0.02])).numpy()
            t = np.array([0.02, -0.01, 2.5])
            l3 = fitter.landmarks_3d(v).numpy() @ R.T + t
        K = Intrinsics(8000.0, 8000.0, 1000.0, 1250.0, "test")
        l2 = l3[:, :2] / l3[:, 2:] * 8000.0 + [1000.0, 1250.0]
        vf = fitter.fit(beta, l2, K)
        # The expression prior (lambda_expr) deliberately biases strong expressions slightly in
        # exchange for robustness to noisy detector landmarks; unregularized it reaches ~0.4%.
        self.assertLess(vf.metrics["lmk_err_iod_mean"], 0.02)
        self.assertAlmostEqual(vf.metrics["subject_distance_m"], 2.5, delta=0.35)


@unittest.skipUnless(MICA.exists(), "MICA checkpoint not downloaded")
class TestMICAParity(unittest.TestCase):
    def test_encoder_matches_vendor_mica(self):
        """β must equal vendor MICA (its ArcFace + regressor) on the same crop, not buffalo_l features."""
        import sys
        import torch
        import torch.nn.functional as F
        from src.stage1_identity.inference import MICAIdentityEncoder, mica_arcface_blob

        enc = MICAIdentityEncoder(str(MICA), device="cpu")
        rng = np.random.default_rng(0)
        crop = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        beta = enc.encode_single(crop)

        sys.path.insert(0, str(ROOT / "vendor" / "MICA"))
        from models.arcface import Arcface
        from models.generator import MappingNetwork as VendorMapping
        ck = torch.load(MICA, map_location="cpu", weights_only=False)
        arc = Arcface()
        arc.load_state_dict(ck["arcface"])
        arc.eval()
        reg = VendorMapping(512, 300, 300, 3)
        reg.load_state_dict({k[10:]: v for k, v in ck["flameModel"].items() if k.startswith("regressor.")})
        reg.eval()
        with torch.no_grad():
            feat = F.normalize(arc(torch.from_numpy(mica_arcface_blob(crop))[None]))
            ref = reg(feat)[0].numpy()
        np.testing.assert_allclose(beta, ref, atol=1e-4)


class TestProvenanceFill(unittest.TestCase):
    def test_observed_kept_and_holes_filled(self):
        from src.stage7_delight.uv_fill import INFERRED, OBSERVED, ProvenanceUVFill
        R = 64
        tex = np.zeros((R, R, 3), np.float32)
        tex[:, : R // 2] = 150.0
        tex[10, 10] = 220.0                          # a detail on the observed side
        obs = np.zeros((R, R), bool)
        obs[:, : R // 2] = True
        valid = np.ones((R, R), bool)
        mirror = np.full((R, R, 2), -1, np.int32)
        yy, xx = np.mgrid[0:R, 0:R]
        mirror[..., 0], mirror[..., 1] = yy, R - 1 - xx
        out = ProvenanceUVFill(detail_sigma_px=2.0).fill(tex, obs, valid, mirror)
        a = out["albedo_srgb"].astype(np.float32)
        self.assertTrue(np.all(a[obs] == np.clip(tex[obs], 0, 255).astype(np.uint8)))
        right = a[:, R // 2:].copy()
        right[10, R - 11 - R // 2] = 150                              # the mirrored detail itself
        self.assertLess(float(np.median(np.abs(right - 150))), 5)    # smooth continuation elsewhere
        self.assertGreater(a[10, R - 11, 0], a[20, R - 11, 0] + 20)  # mirrored detail transferred
        self.assertTrue(np.all(out["provenance"][obs] == OBSERVED))
        self.assertTrue(np.all(out["provenance"][:, R // 2:] == INFERRED))

    def test_empty_observation_raises(self):
        from src.stage7_delight.uv_fill import ProvenanceUVFill
        R = 8
        with self.assertRaises(ValueError):
            ProvenanceUVFill().fill(np.zeros((R, R, 3)), np.zeros((R, R), bool), np.ones((R, R), bool),
                                    np.full((R, R, 2), -1, np.int32))


class TestPhase0ConfigGuards(unittest.TestCase):
    """The deprecated v1 components must stay off by default."""

    def test_default_config(self):
        cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
        self.assertEqual(cfg["stage3"]["mode"], "sculpt")
        self.assertNotIn("weight_macro", cfg["stage3"])      # v1 GAN removed
        self.assertFalse(cfg["stage3"].get("enable_meso", False))
        self.assertFalse(cfg["stage1_5"]["enabled"])

    def test_pipeline_refuses_contour_deformer(self):
        from src.pipeline import FaceGeoPipeline

        class _Det:
            app = None
        pipe = FaceGeoPipeline(cfg={"stage1_5": {"enabled": True}}, detector=_Det())
        if pipe.flame is None:
            self.skipTest("FLAME not extracted")
        with self.assertRaises(ValueError):
            pipe.run(["unused.jpg"], tempfile.mkdtemp())


if __name__ == "__main__":
    unittest.main()
