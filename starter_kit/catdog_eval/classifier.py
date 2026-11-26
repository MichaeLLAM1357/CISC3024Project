"""The fixed cat/dog classifier used by the evaluator to measure whether a generated image
shows the requested class.

It is a linear layer on the Inception-v3 pool3 features that FID already uses (so the
evaluator needs only one network). It is fixed by the teaching team and must not be used
for training or generation.
"""

import hashlib
from pathlib import Path

import torch
from torch import nn

from .features import FEATURE_DIM

WEIGHTS_PATH = Path(__file__).with_name("classifier.pt")


class CatDogProbe(nn.Module):
    """Standardise Inception features, then a linear layer: logits [N, 2] (0 = cat, 1 = dog)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(FEATURE_DIM, dtype=torch.float64))
        self.register_buffer("std", torch.ones(FEATURE_DIM, dtype=torch.float64))
        self.linear = nn.Linear(FEATURE_DIM, 2).double()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.linear((feats.double() - self.mean) / self.std)


def weights_sha256(path: Path = WEIGHTS_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else ""


def load_classifier(path: Path = WEIGHTS_PATH) -> CatDogProbe:
    model = CatDogProbe()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
