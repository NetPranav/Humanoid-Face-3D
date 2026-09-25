"""
Phase 3A (texture completion + delighting) and Phase 4A (sculpt detail) tests.
"""
import unittest

import cv2
import numpy as np


class TestDelighting(unittest.TestCase):
    def test_recovers_constant_albedo_under_directional_light(self):
        from src.stage6_texture.projector import fit_sh_shading
        rng = np.random.default_rng(0)
        n = rng.normal(size=(5000, 3))
        n[:, 2] = -np.abs(n[:, 2])                      # facing the camera (-z)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        light = np.array([0.4, -0.3, -0.85])
        light /= np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
        albedo = np.array([0.45, 0.30, 0.25])
        lin = shade[:, None] * albedo[None]
        out = fit_sh_shading(lin, n, np.ones(len(n), bool))    # default clamp (0.6, 1.6)
        delit = lin / out["shading"]
        # after delighting the albedo is (nearly) constant: spread shrinks a lot
        self.assertLess(delit[:, 1].std() / delit[:, 1].mean(), 0.35 * (lin[:, 1].std() / lin[:, 1].mean()))
        # hue is untouched: channel ratios identical to the true albedo
        np.testing.assert_allclose(delit[:, 0] / delit[:, 1], albedo[0] / albedo[1], rtol=1e-6)

    def test_correction_is_clamped(self):
        from src.stage6_texture.projector import fit_sh_shading
        rng = np.random.default_rng(1)
        n = rng.normal(size=(2000, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        lin = np.clip(n[:, :1] + 1.2, 0.01, None).repeat(3, 1) * 0.3
        s = fit_sh_shading(lin, n, np.ones(len(n), bool))["shading"]
        self.assertGreaterEqual(s.min(), 0.6 - 1e-9)
        self.assertLessEqual(s.max(), 1.6 + 1e-9)


class TestSkinSynthesis(unittest.TestCase):
    def test_quilt_fills_target_and_keeps_grain_contrast(self):
        from src.stage7_delight.skin_synthesis import synthesize_skin_detail
        rng = np.random.default_rng(0)
        R = 256
        tex = np.full((R, R, 3), 150, np.float32)
        tex[:, :128] += cv2.GaussianBlur(rng.normal(0, 12, (R, 128, 3)).astype(np.float32), (0, 0), 1.0)
        observed = np.zeros((R, R), bool)
        observed[:, :128] = True
        target = ~observed
        grain = synthesize_skin_detail(tex, observed, np.ones((R, R), bool), target, sigma_px=4, patch=32)
        self.assertTrue(np.all(grain[observed] == 0))
        from src.stage7_delight.skin_synthesis import high_pass
        ref = high_pass(tex, observed, 4)[observed].std()
        got = grain[target].std()
        self.assertGreater(got, 0.5 * ref)
        self.assertLess(got, 1.5 * ref)


class TestMeshHarmonicFill(unittest.TestCase):
    def test_uv_seam_gets_one_colour(self):
        """Two UV islands of the same surface (a seam) must receive the same low-frequency colour."""
        from src.stage6_texture.projector import rasterize_uv
        from src.stage7_delight.uv_fill import mesh_harmonic_low
        # a strip of 4 vertices / 2 triangles, UV-split into two islands along the shared edge 1-2
        faces = np.array([[0, 1, 2], [2, 1, 3]])
        uv = np.array([[0.05, 0.05], [0.45, 0.05], [0.05, 0.45],      # island A (face 0)
                       [0.55, 0.55], [0.95, 0.55], [0.55, 0.95]])     # island B (face 1): v2, v1, v3
        uv_faces = np.array([[0, 1, 2], [3, 4, 5]])
        R = 64
        ras = rasterize_uv(uv, uv_faces, R)
        tex = np.zeros((R, R, 3), np.float32)
        obs = np.zeros((R, R), bool)
        a = (ras.face_id == 0)
        tex[a] = 200.0
        obs[a] = True
        low = mesh_harmonic_low(tex, obs, ras, faces, uv, uv_faces, n_verts=4, sigma_px=2)
        b = ras.face_id == 1
        # island B touches island A only through shared vertices 1 and 2; those carry A's colour
        self.assertGreater(float(low[b].max()), 150.0)


class TestSculptDetail(unittest.TestCase):
    def test_knob_mapping(self):
        from src.stage3_detail.sculpt_detail import DetailSettings
        self.assertEqual(DetailSettings.from_level(0).subdivision, 1)
        self.assertEqual(DetailSettings.from_level(0).meso_gain, 0.0)
        self.assertEqual(DetailSettings.from_level(50).subdivision, 2)
        self.assertEqual(DetailSettings.from_level(75).subdivision, 3)
        s100 = DetailSettings.from_level(100)
        self.assertEqual((s100.subdivision, s100.meso_gain, s100.micro_gain), (4, 1.0, 1.0))
        self.assertEqual(DetailSettings.from_level(250).level, 100.0)

    def test_crease_becomes_groove_but_round_spot_does_not(self):
        from src.stage3_detail.sculpt_detail import photo_meso
        R = 256
        img = np.full((R, R, 3), 170, np.uint8)
        cv2.line(img, (40, 60), (220, 70), (120, 120, 120), 3, cv2.LINE_AA)   # wrinkle: long dark line
        cv2.circle(img, (128, 190), 5, (110, 110, 110), -1, cv2.LINE_AA)     # mole: round dark spot
        img = cv2.GaussianBlur(img, (0, 0), 1.0)
        m = np.ones((R, R), bool)
        d = photo_meso(img, m, m, texel_mm=0.15)
        line_depth = float(d[55:75, 60:200].min())
        spot_depth = float(d[180:200, 118:138].min())
        self.assertLess(line_depth, -0.05)
        self.assertGreater(spot_depth, 0.5 * line_depth)

    def test_normal_map_is_flat_for_flat_displacement(self):
        from src.stage3_detail.sculpt_detail import normal_map
        n = normal_map(np.zeros((32, 32), np.float32), np.full((32, 32), 0.1, np.float32))
        self.assertTrue(np.all(np.abs(n.astype(int) - [128, 128, 255]) <= 1))


if __name__ == "__main__":
    unittest.main()
