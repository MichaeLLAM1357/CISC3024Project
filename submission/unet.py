"""Small class-conditional UNet for 32x32 images (trained from scratch).

Design notes
------------
* **GroupNorm everywhere, no BatchNorm.**  Batch statistics would make the output depend on the
  other samples in the batch, which the interface contract forbids (LIMITS.md §5).  GroupNorm is
  batch-independent even in training mode, so nothing has to be "fixed up" at `.eval()` time.
* **Time and class conditioning: additive (`cond="add"`) or AdaGN/FiLM (`cond="film"`).**
  The diffusion step `t` and the class `y` each get a sinusoidal/embedding vector, which are
  summed into one conditioning vector `e`.  In `add` mode `e` is projected to a per-channel bias
  that is added after the first convolution (Ho et al. 2020).  In `film` mode the same projection
  emits *two* per-channel vectors, and the first GroupNorm is modulated as
  `norm(x) * (1 + scale) + shift` (Dhariwal & Nichol 2021 "AdaGN"; Peebles & Xie 2023 "adaLN").
  FiLM is per-block, so the class identity reaches every resolution instead of only the block
  input.  The modulation linear layer is zero-initialised, so training starts from the unmodulated
  network and the conditioning is learnt gradually — the standard stabiliser for adaLN blocks.
* Two downsamplings (32 -> 16 -> 8) with a 8x8 bottleneck, and self-attention at 16x16 only: at
  32x32 a single attention layer already gives global coherence without dominating the cost.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None] * 1000.0
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=1))


class ResBlock(nn.Module):
    """Residual block; `cond` selects how the conditioning vector `e` enters the block.

    Both modes keep the parameter *name* `emb`, so a checkpoint stores `emb.weight` either as
    [cout, emb] (add, a bias added after conv1) or as [2*cin, emb] (film, a scale/shift pair
    applied to the normalised input, which has `cin` channels).  That is what makes the two
    variants loadable by the same code path.
    """

    def __init__(self, cin: int, cout: int, emb: int, dropout: float = 0.1,
                 cond: str = "add") -> None:
        super().__init__()
        if cond not in ("add", "film"):
            raise ValueError(f"unknown conditioning mode {cond!r}")
        self.cond = cond
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.emb = nn.Linear(emb, cin * 2 if cond == "film" else cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        if cond == "film":
            nn.init.zeros_(self.emb.weight)      # start as the unmodulated network
            nn.init.zeros_(self.emb.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        proj = self.emb(F.silu(e))
        if self.cond == "film":
            scale, shift = proj[:, :, None, None].chunk(2, dim=1)
            h = self.conv1(self.norm1(x) * (1.0 + scale) + shift)
            h = self.conv2(self.drop(F.silu(self.norm2(h))))
        else:
            h = self.conv1(F.silu(self.norm1(x))) + proj[:, :, None, None]
            h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    """Single-head self-attention over the spatial grid."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        att = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, att.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, cond: str = "add") -> None:
        super().__init__()
        self.block = ResBlock(cin, cout, emb, cond=cond)
        self.down = nn.Conv2d(cout, cout, 3, 2, 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.block(x, e)
        return self.down(h)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, cond: str = "add") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 4, 2, 1)
        self.block = ResBlock(cout * 2, cout, emb, cond=cond)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([self.up(x), skip], dim=1), e)


class UNet(nn.Module):
    """eps_theta(x_t, t, y) -> predicted noise, same shape as x_t."""

    def __init__(self, base: int = 64, emb: int = 128, n_class: int = 2,
                 cond: str = "add") -> None:
        super().__init__()
        self.emb_dim = emb
        self.cond_mode = cond
        # TimeEmbedding emits emb*4; the class embedding is sized to match so the two can be
        # summed, and every ResBlock receives an emb*4 conditioning vector.
        # `n_class` is the number of embedding *rows*: for classifier-free guidance this is
        # n_class + 1, the last row being the null/unconditional token.
        self.time = TimeEmbedding(emb)
        self.label = nn.Embedding(n_class, emb * 4)
        self.in_conv = nn.Conv2d(3, base, 3, 1, 1)
        self.d1 = Down(base, base * 2, emb * 4, cond)              # 32 -> 16
        self.attn = Attn(base * 2)
        self.d2 = Down(base * 2, base * 4, emb * 4, cond)          # 16 -> 8
        self.mid = ResBlock(base * 4, base * 4, emb * 4, cond=cond)
        self.u2 = Up(base * 4, base * 2, emb * 4, cond)            # 8 -> 16
        self.u1 = Up(base * 2, base, emb * 4, cond)                # 16 -> 32
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = self.time(t) + self.label(y)
        h0 = self.in_conv(x)
        h1 = self.attn(self.d1(h0, e))
        h2 = self.d2(h1, e)
        m = self.mid(h2, e)
        u2 = self.u2(m, h1, e)
        u1 = self.u1(u2, h0, e)
        return self.out_conv(F.silu(self.out_norm(u1)))
