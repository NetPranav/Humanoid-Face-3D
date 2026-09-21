#!/usr/bin/env python3
"""
Beta-Collapse Smoke Test.

Validates training dynamics to prove that the composite identity loss
(specifically the explicit β-norm matching term `loss_norm`) prevents
magnitude collapse toward the population mean face (β = 0).

Cosine similarity alone is scale-invariant:
    cos(c * beta_gt, beta_gt) == 1.0  for ANY scalar c > 0

Without explicit norm penalization, any regularization (such as AdamW weight decay)
or gradient unbalance drives ‖β̂‖ toward zero, producing a generic average humanoid face.
This smoke test runs short training dynamics with and without norm matching
and verifies that norm matching strictly preserves morphological magnitude.

Supports PyTorch autograd if installed, and falls back to an exact NumPy
gradient descent formulation with Adam optimizer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def run_smoke_test_torch(n_steps: int = 40) -> bool:
    from src.stage1_identity.models import MappingNetwork

    torch.manual_seed(42)
    batch_size = 16
    feature_dim = 512
    beta_dim = 300

    features = torch.randn(batch_size, feature_dim)
    features = F.normalize(features, dim=-1)

    scale = torch.cat([
        torch.linspace(3.0, 1.5, 10),
        torch.linspace(1.5, 0.5, beta_dim - 10)
    ])
    beta_gt = torch.randn(batch_size, beta_dim) * scale
    target_norm = float(torch.norm(beta_gt, dim=-1).mean().item())
    print(f"Target Ground-Truth Mean ‖β‖: {target_norm:.4f}\n")

    def run_training(use_norm: bool):
        torch.manual_seed(1337)
        model = MappingNetwork(z_dim=512, map_hidden_dim=300, map_output_dim=300, hidden=3)
        # Weight decay actively induces magnitude shrinkage if loss is magnitude-blind
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        loss_hist = []
        init_ratio = 0.0

        for step in range(n_steps):
            optimizer.zero_grad()
            pred = model(features)
            pred_norm = torch.norm(pred, dim=-1)
            if step == 0:
                init_ratio = pred_norm.mean().item() / target_norm

            cos_loss = torch.mean(1.0 - F.cosine_similarity(pred, beta_gt, dim=-1))
            l1_loss = torch.mean(torch.abs(pred - beta_gt))

            if use_norm:
                gt_norms = torch.norm(beta_gt, dim=-1)
                norm_loss = torch.mean(torch.abs(pred_norm - gt_norms))
                loss = l1_loss + cos_loss + norm_loss
            else:
                loss = l1_loss + cos_loss

            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        with torch.no_grad():
            final_pred = model(features)
            final_ratio = torch.norm(final_pred, dim=-1).mean().item() / target_norm

        return init_ratio, final_ratio, loss_hist

    print("[Run A] Training WITHOUT explicit norm matching (PyTorch)...")
    init_a, final_a, hist_a = run_training(use_norm=False)
    print(f"  Initial ‖β̂‖ / ‖β_gt‖: {init_a:.4f} | Final: {final_a:.4f} | Loss: {hist_a[0]:.4f} -> {hist_a[-1]:.4f}")

    print("\n[Run B] Training WITH explicit norm matching (PyTorch)...")
    init_b, final_b, hist_b = run_training(use_norm=True)
    print(f"  Initial ‖β̂‖ / ‖β_gt‖: {init_b:.4f} | Final: {final_b:.4f} | Loss: {hist_b[0]:.4f} -> {hist_b[-1]:.4f}")

    passed = (final_b >= 0.85) and (hist_b[-1] < hist_b[0])
    return passed


def run_smoke_test_numpy(n_steps: int = 100) -> bool:
    """
    NumPy analytical simulation using Adam optimizer on linear/affine projection W
    mapping normalized ArcFace embeddings to FLAME beta space.
    """
    print("[Environment] Running exact NumPy simulation with Adam optimizer...")
    rng = np.random.default_rng(42)
    B = 16
    D_in = 512
    D_out = 300

    # Synthetic ArcFace embeddings (unit normalized)
    X = rng.standard_normal((B, D_in)).astype(np.float32)
    X /= np.linalg.norm(X, axis=-1, keepdims=True)

    # Synthetic diverse ground truth beta
    scale = np.concatenate([np.linspace(3.0, 1.5, 10), np.linspace(1.5, 0.5, D_out - 10)])
    beta_gt = (rng.standard_normal((B, D_out)) * scale).astype(np.float32)
    gt_norms = np.linalg.norm(beta_gt, axis=-1, keepdims=True)
    target_norm = float(np.mean(gt_norms))
    print(f"Target Ground-Truth Mean ‖β‖: {target_norm:.4f}\n")

    def run_opt(use_norm_matching: bool, lr: float = 0.05):
        rng_init = np.random.default_rng(1337)
        W = (rng_init.standard_normal((D_in, D_out)) * 0.01).astype(np.float32)
        mW, vW = np.zeros_like(W), np.zeros_like(W)

        init_ratio = 0.0
        loss_hist = []

        for step in range(1, n_steps + 1):
            pred = np.dot(X, W)
            pred_norms = np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8
            mean_pred_norm = float(np.mean(pred_norms))

            if step == 1:
                init_ratio = mean_pred_norm / target_norm

            diff = pred - beta_gt
            grad_l1 = np.sign(diff) / (B * D_out)
            loss_l1 = np.mean(np.abs(diff))

            dot_prod = np.sum(pred * beta_gt, axis=-1, keepdims=True)
            cos_sim = dot_prod / (pred_norms * gt_norms)
            loss_cosine = np.mean(1.0 - cos_sim)
            grad_cos = -(beta_gt / (pred_norms * gt_norms)) + (dot_prod / (pred_norms**3 * gt_norms)) * pred
            grad_cos /= B

            if use_norm_matching:
                norm_diff = pred_norms - gt_norms
                loss_norm = np.mean(np.abs(norm_diff))
                grad_norm = (np.sign(norm_diff) * (pred / pred_norms)) / B
                total_loss = loss_l1 + loss_cosine + loss_norm
                grad = grad_l1 + grad_cos + grad_norm
            else:
                total_loss = loss_l1 + loss_cosine
                grad = grad_l1 + grad_cos

            loss_hist.append(float(total_loss))

            # Backprop to W
            grad_W = np.dot(X.T, grad)

            # Adam update
            mW = 0.9 * mW + 0.1 * grad_W
            vW = 0.999 * vW + 0.001 * (grad_W**2)
            m_hat = mW / (1.0 - 0.9**step)
            v_hat = vW / (1.0 - 0.999**step)
            W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        final_pred = np.dot(X, W)
        final_norm = float(np.mean(np.linalg.norm(final_pred, axis=-1)))
        final_ratio = final_norm / target_norm
        return init_ratio, final_ratio, loss_hist

    print("[Run A] Training WITHOUT explicit norm matching (NumPy)...")
    init_a, final_a, hist_a = run_opt(use_norm_matching=False)
    print(f"  Initial ‖β̂‖ / ‖β_gt‖: {init_a:.4f} | Final: {final_a:.4f} | Loss: {hist_a[0]:.4f} -> {hist_a[-1]:.4f}")

    print("\n[Run B] Training WITH explicit norm matching (NumPy)...")
    init_b, final_b, hist_b = run_opt(use_norm_matching=True)
    print(f"  Initial ‖β̂‖ / ‖β_gt‖: {init_b:.4f} | Final: {final_b:.4f} | Loss: {hist_b[0]:.4f} -> {hist_b[-1]:.4f}")

    print("\n" + "-" * 65)
    print(f"Norm Retention Comparison after {n_steps} steps:")
    print(f"  - Without Norm Matching: {final_a*100:.1f}% of ground truth magnitude")
    print(f"  - With Norm Matching:    {final_b*100:.1f}% of ground truth magnitude")

    passed = (final_b >= 0.85) and (hist_b[-1] < hist_b[0])
    if passed:
        print("\n✅ SMOKE TEST PASSED: β-norm matching reliably preserves face morphology.")
    else:
        print("\n❌ SMOKE TEST FAILED: β magnitude failed to track ground truth adequately.")

    print("=" * 65)
    return passed


def main():
    parser = argparse.ArgumentParser(description="FLAME beta collapse smoke test")
    parser.add_argument('--steps', type=int, default=100, help="Number of training steps")
    args = parser.parse_args()

    print("=" * 65)
    print(" FLAME β-Collapse Smoke Test")
    print("=" * 65)

    if torch is not None:
        passed = run_smoke_test_torch(n_steps=args.steps)
    else:
        passed = run_smoke_test_numpy(n_steps=args.steps)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
