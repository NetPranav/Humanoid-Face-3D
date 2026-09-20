from __future__ import annotations
import unittest
import tempfile
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from src.stage3_detail.trainer import get_recon_lambda, update_ema, HYPERPARAMS
from src.stage3_detail.losses import (
    adversarial_loss_g, adversarial_loss_d,
    reconstruction_loss_masked, r1_gradient_penalty,
    identity_preservation_loss, photometric_consistency_loss,
)
from src.stage3_detail.data import UVDisplacementDataset
from scripts.upload_to_kaggle_models import upload_checkpoint


class TestStage3DetailGAN(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # ── 1. Pure Math & Schedule Tests (No GPU / torch required) ───────
    def test_recon_lambda_annealing_schedule(self):
        """Verify lambda_recon anneals linearly from 100.0 down to 10.0 over 50,000 steps."""
        cfg = {
            'recon_lambda_start': 100.0,
            'recon_lambda_end': 10.0,
            'recon_anneal_steps': 50000,
        }
        # Start of training
        self.assertAlmostEqual(get_recon_lambda(0, cfg), 100.0)
        # Mid-training (25k steps)
        self.assertAlmostEqual(get_recon_lambda(25000, cfg), 55.0)
        # End of annealing (50k steps)
        self.assertAlmostEqual(get_recon_lambda(50000, cfg), 10.0)
        # Post-annealing clamp (75k steps)
        self.assertAlmostEqual(get_recon_lambda(75000, cfg), 10.0)

    # ── 2. Dataset Error Handling & Stratification Tests ──────────────
    def test_dataset_raises_on_missing_displacement_maps(self):
        """Verify UVDisplacementDataset raises RuntimeError when no *_disp.png files exist."""
        with self.assertRaises(RuntimeError) as ctx:
            UVDisplacementDataset(str(self.temp_path))
        self.assertIn("No *_disp.png files found", str(ctx.exception))

    def test_dataset_subject_stratified_split(self):
        """Verify subject stratification: no subject can exist in both train and val splits."""
        import cv2
        # Generate dummy 16-bit displacement files for 10 distinct subjects
        for i in range(10):
            subj_id = f"sub{i:03d}"
            # 2 expressions per subject: neutral and smile
            for expr in ["neutral", "smile"]:
                stem = f"{subj_id}_{expr}"
                disp_file = self.temp_path / f"{stem}_disp.png"
                # Write minimal 16-bit uint PNG
                dummy_u16 = np.zeros((32, 32), dtype=np.uint16)
                cv2.imwrite(str(disp_file), dummy_u16)

        train_ds = UVDisplacementDataset(str(self.temp_path), is_train=True)
        val_ds = UVDisplacementDataset(str(self.temp_path), is_train=False)

        train_subjects = {p.stem.split('_')[0] for p in train_ds.files}
        val_subjects = {p.stem.split('_')[0] for p in val_ds.files}

        # Assert no subject leakage
        leakage = train_subjects.intersection(val_subjects)
        self.assertEqual(len(leakage), 0, f"Subject leakage detected in Phase 3 dataset: {leakage}")
        self.assertGreater(len(train_subjects), 0)
        self.assertGreater(len(val_subjects), 0)
        self.assertEqual(len(train_subjects) + len(val_subjects), 10)

    @unittest.skipIf(torch is None, "PyTorch required for dataset getitem test")
    def test_dataset_getitem_per_view_feats_fallback(self):
        """Verify UVDisplacementDataset.__getitem__ safely falls back when per_view_feats is missing."""
        import cv2
        stem = "sub001_neutral"
        # Write paired mock images
        cv2.imwrite(str(self.temp_path / f"{stem}_disp.png"), np.zeros((32, 32), dtype=np.uint16))
        cv2.imwrite(str(self.temp_path / f"{stem}_pos.png"), np.zeros((32, 32, 3), dtype=np.uint8))
        cv2.imwrite(str(self.temp_path / f"{stem}_norm.png"), np.zeros((32, 32, 3), dtype=np.uint8))
        cv2.imwrite(str(self.temp_path / f"{stem}_mask.png"), np.ones((32, 32), dtype=np.uint8) * 255)

        # 1. Meta file with only beta and psi (missing per_view_feats)
        np.savez(
            str(self.temp_path / f"{stem}_meta.npz"),
            beta=np.zeros(300, dtype=np.float32),
            psi=np.zeros(100, dtype=np.float32)
        )

        ds = UVDisplacementDataset(str(self.temp_path), is_train=False)
        item = ds[0]

        self.assertIn('per_view_feats', item)
        self.assertEqual(item['per_view_feats'].shape, (1, 512))
        self.assertEqual(item['beta'].shape, (300,))
        self.assertEqual(item['psi'].shape, (100,))


    # ── 3. Checkpoint Upload Gate Tests ────────────────────────────────
    def test_upload_gate_blocks_missing_ema_checkpoint(self):
        """Verify upload_checkpoint raises FileNotFoundError if ema_generator.pt is missing for stage 3."""
        # Create dummy folder without ema_generator.pt
        with self.assertRaises(FileNotFoundError) as ctx:
            upload_checkpoint(
                checkpoint_dir=str(self.temp_path),
                handle="test/model/pytorch/v1",
                version_notes="Test note",
                stage=3,
            )
        self.assertIn("EMA checkpoint not found", str(ctx.exception))

    def test_upload_gate_blocks_missing_normalization_stats(self):
        """Verify upload_checkpoint raises FileNotFoundError if normalization_stats.json is missing for stage 3."""
        # Create dummy ema_generator.pt
        (self.temp_path / "ema_generator.pt").touch()

        with self.assertRaises(FileNotFoundError) as ctx:
            upload_checkpoint(
                checkpoint_dir=str(self.temp_path),
                handle="test/model/pytorch/v1",
                version_notes="Test note",
                stage=3,
            )
        self.assertIn("normalization_stats.json not found", str(ctx.exception))

    # ── 4. PyTorch Architecture & Forward Pass Tests (Skipped if torch missing)
    @unittest.skipIf(torch is None, "PyTorch required for neural network module tests")
    def test_generator_architecture_and_norm_layers(self):
        """Verify DetailGenerator uses InstanceNorm2d (affine=True) and NO BatchNorm2d."""
        from src.stage3_detail.generator import DetailGenerator

        gen = DetailGenerator(base_channels=16)

        # Inspect all submodules
        has_instance_norm = False
        for name, module in gen.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.fail(f"Found forbidden BatchNorm2d layer in generator: {name}")
            if isinstance(module, nn.InstanceNorm2d):
                has_instance_norm = True
                self.assertTrue(module.affine, f"InstanceNorm2d in {name} must have affine=True")

        self.assertTrue(has_instance_norm, "DetailGenerator must contain InstanceNorm2d layers")

        # Verify LayerNorm on cross-attention bottleneck
        self.assertTrue(hasattr(gen, 'attn_norm'), "Generator must have attn_norm")
        self.assertIsInstance(gen.attn_norm, nn.LayerNorm)

    @unittest.skipIf(torch is None, "PyTorch required for neural network module tests")
    def test_generator_forward_and_output_bounds(self):
        """Verify DetailGenerator forward pass produces valid (B, 1, 512, 512) tensor bounded in [-1, 1]."""
        from src.stage3_detail.generator import DetailGenerator

        # Use small base channels for fast unit test
        gen = DetailGenerator(base_channels=16)
        gen.eval()

        B = 2
        pos = torch.randn(B, 3, 512, 512)
        norm = torch.randn(B, 3, 512, 512)
        feats = torch.randn(B, 3, 512)
        beta = torch.randn(B, 300)
        psi = torch.randn(B, 100)

        with torch.no_grad():
            out = gen(pos, norm, feats, beta, psi)

        self.assertEqual(out.shape, (B, 1, 512, 512))
        self.assertTrue(torch.all(out >= -1.0))
        self.assertTrue(torch.all(out <= 1.0))
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipIf(torch is None, "PyTorch required for neural network module tests")
    def test_discriminator_patchgan_structure(self):
        """Verify DetailDiscriminator outputs 2D patch grid and optional auxiliary classification heads."""
        from src.stage3_detail.discriminator import DetailDiscriminator

        disc = DetailDiscriminator(base_channels=16, num_identities=50, num_expressions=20)
        disc.eval()

        B = 2
        x = torch.randn(B, 1, 512, 512)
        patch_out, id_out, expr_out = disc(x)

        # PatchGAN output should have spatial dims (grid of local patches)
        self.assertEqual(patch_out.dim(), 4)
        self.assertEqual(patch_out.shape[0], B)
        self.assertEqual(patch_out.shape[1], 1)
        self.assertGreater(patch_out.shape[2], 1)
        self.assertGreater(patch_out.shape[3], 1)

        # Auxiliary heads
        self.assertIsNotNone(id_out)
        self.assertEqual(id_out.shape, (B, 50))
        self.assertIsNotNone(expr_out)
        self.assertEqual(expr_out.shape, (B, 20))

    @unittest.skipIf(torch is None, "PyTorch required for loss tests")
    def test_masked_reconstruction_loss_ignores_invalid_uv(self):
        """Verify reconstruction_loss_masked calculates L1 only over valid mask pixels."""
        B, C, H, W = 1, 1, 64, 64
        pred = torch.zeros((B, C, H, W))
        target = torch.ones((B, C, H, W))

        # Mask only top half (H/2) as valid (1.0), bottom half as invalid (0.0)
        mask = torch.zeros((B, C, H, W))
        mask[:, :, :32, :] = 1.0

        # Change pred on invalid region (bottom half) to something massive
        pred[:, :, 32:, :] = 999.0

        loss = reconstruction_loss_masked(pred, target, mask)
        # Since valid region has pred=0 and target=1, error on valid region is 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=4)

        # Assert raising when mask is entirely zero
        empty_mask = torch.zeros_like(mask)
        with self.assertRaises(AssertionError):
            reconstruction_loss_masked(pred, target, empty_mask)

    @unittest.skipIf(torch is None, "PyTorch required for loss tests")
    def test_adversarial_losses(self):
        """Verify adversarial loss functions return finite non-negative scalars."""
        d_real = torch.tensor([1.5, 2.0])
        d_fake = torch.tensor([-1.5, -2.0])

        d_loss = adversarial_loss_d(d_real, d_fake)
        g_loss = adversarial_loss_g(d_fake)

        self.assertGreater(d_loss.item(), 0.0)
        self.assertGreater(g_loss.item(), 0.0)
        self.assertFalse(torch.isnan(d_loss))
        self.assertFalse(torch.isnan(g_loss))

    @unittest.skipIf(torch is None, "PyTorch required for EMA tests")
    def test_update_ema_interpolates_params_and_copies_buffers(self):
        """Verify update_ema updates parameters by decay and copies buffers exactly."""
        class MockNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([10.0]))
                self.register_buffer('running_count', torch.tensor([5]))

        model = MockNet()
        ema_model = MockNet()
        # Initialize EMA with different values
        ema_model.weight.data = torch.tensor([0.0])
        ema_model.running_count.data = torch.tensor([0])

        # Step EMA with decay = 0.9
        update_ema(ema_model, model, decay=0.9)

        # Expected weight: 0.9 * 0.0 + 0.1 * 10.0 = 1.0
        self.assertAlmostEqual(ema_model.weight.item(), 1.0, places=4)
        # Expected buffer: exactly copied -> 5
        self.assertEqual(ema_model.running_count.item(), 5)

    def test_hyperparams_structure(self):
        """Verify HYPERPARAMS contains required Stage 3 GAN training settings."""
        self.assertIn('id_lambda', HYPERPARAMS)
        self.assertGreaterEqual(HYPERPARAMS['id_lambda'], 0.0)
        self.assertIn('recon_lambda_start', HYPERPARAMS)
        self.assertIn('r1_gamma', HYPERPARAMS)
        self.assertIn('ema_decay', HYPERPARAMS)

    @unittest.skipIf(torch is None, "PyTorch required for loss tests")
    def test_identity_preservation_loss_computation(self):
        """Verify identity_preservation_loss computes cosine distance correctly."""
        # Identical vectors: distance should be 0.0
        emb1 = torch.tensor([[1.0, 0.0, 0.0]])
        emb2 = torch.tensor([[1.0, 0.0, 0.0]])
        loss_identical = identity_preservation_loss(emb1, emb2)
        self.assertAlmostEqual(loss_identical.item(), 0.0, places=5)

        # Orthogonal vectors: distance should be 1.0
        emb_ortho = torch.tensor([[0.0, 1.0, 0.0]])
        loss_ortho = identity_preservation_loss(emb1, emb_ortho)
        self.assertAlmostEqual(loss_ortho.item(), 1.0, places=5)

    @unittest.skipIf(torch is None, "PyTorch required for loss tests")
    def test_photometric_consistency_loss_computation(self):
        """Verify photometric_consistency_loss computes L1 difference."""
        img1 = torch.zeros((1, 3, 32, 32))
        img2 = torch.ones((1, 3, 32, 32))
        loss = photometric_consistency_loss(img1, img2)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
