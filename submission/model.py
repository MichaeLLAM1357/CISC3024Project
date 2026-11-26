"""Conditional DCGAN for 32x32 cat/dog heads (trained from scratch, no pretrained weights).

Three design decisions, each tied to a failure seen in this project:

1. **Projection discriminator** (Miyato & Koyama, 2018).  The first version fed the label to the
   discriminator as constant image planes; the discriminator then scored real/fake without using
   the class at all, the generator had no incentive to condition, and the official evaluator
   measured cat 0.996 / dog 0.032 accuracy on the 500-sample check.  Here the class embedding is
   multiplied with the penultimate feature vector, so a wrong class lowers the logit directly.
2. **Normalisation that keeps generation batch-independent.**  `norm="bn"` uses BatchNorm, which
   gives sharper 32x32 texture, and `load_model` calls `.eval()` so running statistics are used
   and the output cannot depend on the rest of the batch.  `norm="gn"` uses GroupNorm, which is
   batch-independent even in training mode.  Both are supported; pytest measures the difference.
3. **Spectral normalisation on the discriminator**, for stable training with small batches.

Class conditioning follows Mirza & Osindero (2014); the convolutional stack follows the DCGAN
rules of Radford et al. (2016): strided convolutions, no pooling, LeakyReLU in D, tanh output.
"""

import torch
from torch import nn

Z_DIM = 96
N_CLASS = 2


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        return nn.GroupNorm(8, channels)
    raise ValueError(f"unknown norm {kind!r}")


def _up(cin: int, cout: int, kind: str) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
        _norm(kind, cout),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    """z [N, Z_DIM] ~ N(0, 1), y [N] -> images [N, 3, 32, 32] in [-1, 1]."""

    def __init__(self, z_dim: int = Z_DIM, n_class: int = N_CLASS, width: int = 256,
                 norm: str = "bn") -> None:
        super().__init__()
        self.z_dim = z_dim
        self.n_class = n_class
        self.width = width
        self.norm = norm
        self.embed = nn.Embedding(n_class, z_dim)
        self.fc = nn.Linear(z_dim * 2, width * 4 * 4)
        self.blocks = nn.Sequential(
            _up(width, width // 2, norm),          # 4 -> 8
            _up(width // 2, width // 4, norm),     # 8 -> 16
            _up(width // 4, width // 8, norm),     # 16 -> 32
            nn.Conv2d(width // 8, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.embed(y)], dim=1)
        h = self.fc(h).view(-1, self.width, 4, 4)
        return self.blocks(h)


class Discriminator(nn.Module):
    """Projection discriminator: logit = out(v) + <v, e_y>."""

    def __init__(self, n_class: int = N_CLASS, width: int = 64, feat: int = 512) -> None:
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        self.blocks = nn.Sequential(
            sn(nn.Conv2d(3, width, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),           # 16
            sn(nn.Conv2d(width, width * 2, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),   # 8
            sn(nn.Conv2d(width * 2, width * 4, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
        )                                                                                # 4x4
        self.fc = sn(nn.Linear(width * 4 * 4 * 4, feat))
        self.out = sn(nn.Linear(feat, 1))
        self.embed = nn.Embedding(n_class, feat)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x).flatten(1)
        v = torch.nn.functional.leaky_relu(self.fc(h), 0.2)
        return self.out(v).squeeze(1) + (v * self.embed(y)).sum(1)
