"""Evaluator interface: load_model(device) and generate(model, labels, seed).

Contract (LIMITS.md §5, brief §10):
  * load_model returns ONE nn.Module in .eval() holding every generation parameter.
  * generate is a deterministic function of (labels, seed); each image depends only on its own
    label and the seed, never on the rest of the batch.
  * Output is [N, 3, 32, 32] float in [0, 1] with no NaN.
  * No network access, no downloads, no classifier.pt, no Inception.

Two model families are supported, both trained from scratch inside this project:
  * model A -- conditional DCGAN (`model.py`), stores a `state_dict` payload;
  * model B -- class-conditional DDPM (`diffusion.py`), stores `kind="ddpm"`.
The graded checkpoint decides which is used; `generate` dispatches on the loaded module itself.
"""

from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights.pt"

# Antithetic guidance weight for model B (see diffusion.py).  1.0 = plain conditional sampling.
# 2.5 was selected with the official evaluator only (`scan_posttrain.py`, 250/class, seed 0) and
# confirmed at the official 1000/class in results/legacy_1000row/eval_final.json.  It costs one extra UNet
# forward per DDIM step; determinism and batch independence are unaffected.
GUIDANCE = 2.5

_MASK = (1 << 63) - 1


def load_model(device: torch.device) -> nn.Module:
    import sys

    sys.path.insert(0, str(HERE))          # sibling modules, resolvable from any cwd
    ckpt = torch.load(WEIGHTS, map_location="cpu")

    if isinstance(ckpt, dict) and ckpt.get("kind") == "ddpm":
        from diffusion import load_payload
        model = load_payload(ckpt)
    else:
        from model import Generator
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model = Generator(z_dim=int(ckpt.get("z_dim", 96)),
                              width=int(ckpt.get("width", 256)),
                              norm=str(ckpt.get("norm", "bn")))
            ckpt = ckpt["state_dict"]
        else:
            model = Generator()
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def _noise(n: int, z_dim: int, seed: int) -> torch.Tensor:
    """Per-sample deterministic noise for model A.

    Sample i uses its own generator seeded from (seed, i), so the noise for index i does not
    depend on how many samples are in the batch or what the other labels are.  Drawing from a
    CPU generator keeps CPU and GPU results identical and leaves the global RNG untouched.

    The distribution is N(0, 1), exactly what train.py draws (`torch.randn`) and what the
    training sample grids use.  An earlier version used `torch.rand` (U[0,1), std 0.289):
    the generator was therefore evaluated far off the noise distribution it was trained on,
    which collapsed all samples towards one average face -- precision stayed ~0.72 while
    recall fell to 0.008.  Keep this in lockstep with the training sampler.
    """
    rows = []
    for i in range(n):
        g = torch.Generator().manual_seed((int(seed) * 1_000_003 + i) & _MASK)
        rows.append(torch.randn(z_dim, generator=g))
    return torch.stack(rows)


@torch.no_grad()
def generate(model: nn.Module, labels: torch.Tensor, seed: int) -> torch.Tensor:
    # Take the device from the model, not from `labels`: the evaluator may hand us CPU labels
    # while the weights live on the GPU, and noise built on the wrong device fails in the concat.
    device = next(model.parameters()).device
    labels = labels.to(device=device, dtype=torch.long)
    if hasattr(model, "sample"):                 # model B: DDPM, deterministic DDIM
        x = model.sample(labels, seed, guidance=GUIDANCE)
    else:                                        # model A: conditional DCGAN
        z = _noise(len(labels), model.z_dim, seed).to(device)
        x = model(z, labels)
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0).float()
