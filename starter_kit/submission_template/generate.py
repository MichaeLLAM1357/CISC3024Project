"""Submission interface -- copy this file into your submission directory and fill it in.

The evaluator imports this file and calls exactly two functions:

    model  = load_model(device)
    images = generate(model, labels, seed)      # called many times, under torch.no_grad()

Rules (checked automatically by evaluate.py):
  * load_model(device) returns ONE torch.nn.Module that owns every trainable
    parameter used for generation (wrap several networks in a single Module).
    Load weights from files inside this directory, e.g. Path(__file__).parent / "weights.pt".
  * generate(model, labels, seed):
      labels  LongTensor [N] with values 0 (cat) or 1 (dog), already on `device`
      seed    int; the output must be a deterministic function of (labels, seed),
              and different seeds must give different images
      returns float Tensor [N, 3, 32, 32] (RGB) with values in [0, 1]
      Each image may depend only on its own label and the seed, not on the other
      labels in the batch: the evaluator also asks for batches of a single class.
      (A BatchNorm layer left in training mode breaks exactly this.)
  * No network access, no downloads, and no pretrained weights you did not train
    yourself; the evaluation classifier and Inception network must not be used.
  * manifest.csv in this directory lists every training image (see the README):
    pool_id,x0,y0,size,angle,label. The evaluator rebuilds your training set from it.

This placeholder returns seeded random noise, so it "works" but scores terribly.
From the project root, run
`python starter_kit/evaluate.py --submission starter_kit/submission_template --n-per-class 100`
to see the evaluator end to end.
"""

from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent


class PlaceholderModel(nn.Module):
    """Replace with your generative model."""

    def __init__(self) -> None:
        super().__init__()
        self.brightness = nn.Parameter(torch.tensor(0.5))


def load_model(device: torch.device) -> nn.Module:
    model = PlaceholderModel()
    # model.load_state_dict(torch.load(HERE / "weights.pt", map_location="cpu"))
    return model.to(device).eval()  # eval() matters for BatchNorm / Dropout


@torch.no_grad()
def generate(model: nn.Module, labels: torch.Tensor, seed: int) -> torch.Tensor:
    # Draw noise from a CPU generator seeded with `seed`, then move it to the
    # device: this gives identical results on CPU and GPU and never touches the
    # global RNG state.
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(len(labels), 3, 32, 32, generator=g).to(labels.device)
    return (noise * model.brightness).clamp(0, 1)
