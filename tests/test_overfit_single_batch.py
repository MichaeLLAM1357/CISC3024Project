"""Test 3: the training loop really learns — overfit one tiny batch.

A generative model that never moves its weights, or a loop with a detached graph / wrong label
plumbing, still produces valid-looking [0,1] tensors.  This test trains the actual
generator/discriminator pair on 16 manifest images for a few dozen steps and requires the
discriminator to separate that fixed batch, which is only possible if gradients flow end to end.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

from model import Discriminator, Generator
from train import build_dataset

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "submission" / "manifest.csv"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(),
                                reason="submission/manifest.csv not built yet")

STEPS = 160
N = 16


def test_overfit_single_batch() -> None:
    torch.manual_seed(0)
    x, y = build_dataset(MANIFEST, ROOT / "data", limit=N)
    assert x.shape[0] == N
    real = (x * 2 - 1)

    g = Generator(width=64)
    d = Discriminator(width=32)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-3, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    history = []
    for _ in range(STEPS):
        noise = torch.randn(N, g.z_dim)
        fake = g(noise, y)

        opt_d.zero_grad(set_to_none=True)
        loss_d = bce(d(real, y), torch.ones(N)) + bce(d(fake.detach(), y), torch.zeros(N))
        loss_d.backward()
        opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        loss_g = bce(d(fake, y), torch.ones(N))
        loss_g.backward()
        opt_g.step()

        assert torch.isfinite(loss_d) and torch.isfinite(loss_g)
        history.append(float(loss_d))

    early = sum(history[:20]) / 20
    late = sum(history[-20:]) / 20
    assert late < early, f"discriminator loss did not decrease: {early:.3f} -> {late:.3f}"

    with torch.no_grad():
        acc_real = (d(real, y) > 0).float().mean().item()
        acc_fake = (d(fake, y) < 0).float().mean().item()
    accuracy = 0.5 * (acc_real + acc_fake)
    assert accuracy >= 0.7, f"discriminator cannot separate a fixed batch (acc={accuracy:.2f})"
