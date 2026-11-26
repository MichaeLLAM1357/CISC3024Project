"""Arm `aux_head`: the shipped 5.15M UNet plus an optional 514-parameter class head.

Only one thing is added to `submission/unet.py`:

    aux=True  ->  self.aux_head = Sequential(AdaptiveAvgPool2d(1), Flatten(), Linear(base*4, 2))

It reads the **bottleneck** feature `m` (the output of `self.mid`, 8x8 x base*4) through global
average pooling, so it costs 256*2 + 2 = 514 parameters.  `forward()` is untouched and returns
exactly the old eps, which is what makes this arm a single-factor change: the *sampler* and the
graded interface cannot tell the two networks apart.

The head is a training-time regulariser only.  It never runs inside `generate()`, so sampling cost
is unchanged; the parameters do sit in the checkpoint (514 of 5,146,245, i.e. +0.010%).
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
    """Residual block; `cond` selects how the conditioning vector `e` enters the block."""

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
    """eps_theta(x_t, t, y) -> predicted noise; `aux=True` adds the bottleneck class head."""

    def __init__(self, base: int = 64, emb: int = 128, n_class: int = 2,
                 cond: str = "add", aux: bool = False, aux_classes: int = 2,
                 aux_src: str = "mid") -> None:
        super().__init__()
        if aux_src not in ("mid", "stem"):
            raise ValueError(f"unknown aux_src {aux_src!r}")
        self.aux_src = aux_src
        self.emb_dim = emb
        self.cond_mode = cond
        self.has_aux = bool(aux)
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
        # 514 parameters at the bottleneck (`mid`); 130 at the input stem (`stem`, 64 channels).
        # `mid` is the arm the brief asked for; `stem` is the control for the conditioning-leak
        # question -- see pipeline/probe_aux_head.py.
        ch = base * 4 if aux_src == "mid" else base
        self.aux_head = (nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(ch, aux_classes))
                         if self.has_aux else None)

    def _run(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        e = self.time(t) + self.label(y)
        h0 = self.in_conv(x)
        h1 = self.attn(self.d1(h0, e))
        h2 = self.d2(h1, e)
        m = self.mid(h2, e)
        u2 = self.u2(m, h1, e)
        u1 = self.u1(u2, h0, e)
        return self.out_conv(F.silu(self.out_norm(u1))), m, h0

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._run(x, t, y)[0]

    def forward_aux(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """(eps, logits).  Identical eps to `forward`; the head only reads one feature map."""
        eps, m, h0 = self._run(x, t, y)
        return eps, self.aux_head(m if self.aux_src == "mid" else h0)
