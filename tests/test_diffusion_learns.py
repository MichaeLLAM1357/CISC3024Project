"""Extra check: the diffusion training objective actually decreases (model B's version of test 3).

`test_overfit_single_batch.py` proves the GAN loop learns.  Model B is a different loop -- a
timestep sampler, a schedule buffer, an epsilon-prediction loss -- and it can silently fail in ways
the GAN test cannot see (a schedule that never leaves abar ~ 1, a time embedding that is not used,
a `t` that is always 0).  Here the same eight manifest images are trained for a few dozen steps and
the epsilon loss must fall well below its initial value.
"""

from pathlib import Path

import pytest
import torch

from diffusion import DiffusionModel
from train import build_dataset

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "submission" / "manifest.csv"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(),
                                reason="submission/manifest.csv not built yet")

STEPS = 120
N = 8


def test_diffusion_loss_decreases_on_fixed_batch() -> None:
    torch.manual_seed(0)
    x, y = build_dataset(MANIFEST, ROOT / "data", limit=N)
    real = x * 2 - 1

    model = DiffusionModel(base=16, emb=32, n_steps=1000, ddim_steps=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    history = []
    for _ in range(STEPS):
        loss = model.loss(real, y, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
        history.append(float(loss))

    early = sum(history[:20]) / 20
    late = sum(history[-20:]) / 20
    assert late < early * 0.8, f"epsilon loss did not fall: {early:.4f} -> {late:.4f}"


def test_schedule_is_monotone_and_sampling_is_in_range() -> None:
    model = DiffusionModel(base=16, emb=32, n_steps=1000, ddim_steps=4).eval()
    assert torch.all(model.abar[:-1] >= model.abar[1:]), "abar must be non-increasing"
    assert model.abar[-1] < 0.01 and model.abar[0] > 0.99

    out = model.sample(torch.tensor([0, 1, 0, 1]), seed=3, steps=4)
    assert out.shape == (4, 3, 32, 32)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.001 and out.max() <= 1.001, "unclamped sampler output"
