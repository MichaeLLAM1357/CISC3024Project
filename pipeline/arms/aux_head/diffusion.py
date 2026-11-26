"""Arm `aux_head`: the shipped diffusion objective + a bottleneck class loss.

Everything in `submission/diffusion.py` is kept verbatim; the only additions are

    aux=True              -- the UNet grows the 514-parameter linear head (see unet.py)
    loss_with_aux(...)    -- L_simple + aux_weight * CE(head(bottleneck), label)

Why a bottleneck head and not a pixel-level classifier: the hypothesis is that a class boundary
enforced on the *deepest* feature forces every resolution below it to carry class information,
which is what the epsilon objective alone leaves implicit in a 2-class problem.

The CE term is restricted to low-noise timesteps (`t / T <= aux_tmax`), because at high noise the
bottleneck is dominated by epsilon and the class is not recoverable from it -- a CE there is a
noise-fitting term with expected value ln 2 and a gradient that only fights the denoiser.  The
per-batch scale factor `b / n_masked` keeps the *expected* contribution equal to `aux_weight`
independent of the duty cycle, so `aux_weight` means the same thing at every `aux_tmax`.

Nothing here touches the sampler: `sample()` still runs the plain eps-prediction DDIM of the
shipped model with the same antithetic guidance, and `forward()` returns the same eps.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from unet import UNet

N_CLASS = 2
_MASK = (1 << 63) - 1


def cosine_abar(n_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    abar = f / f[0]
    return abar.clamp(1e-8, 1.0).float()


class DiffusionModel(nn.Module):
    """Holds the UNet plus the schedule buffers, so `load_model` can return one module."""

    def __init__(self, base: int = 64, emb: int = 128, n_steps: int = 1000,
                 n_class: int = N_CLASS, ddim_steps: int = 50,
                 cond: str = "add", null_token: bool = False,
                 aux: bool = False, aux_weight: float = 0.0,
                 aux_tmax: float = 0.25, aux_src: str = "mid") -> None:
        super().__init__()
        n_rows = n_class + (1 if null_token else 0)
        self.unet = UNet(base=base, emb=emb, n_class=n_rows, cond=cond,
                         aux=aux, aux_classes=n_class, aux_src=aux_src)
        self.aux_src = aux_src
        self.n_steps = n_steps
        self.n_class = n_class
        self.ddim_steps = ddim_steps
        self.z_dim = 0
        self.cond = cond
        self.null_token = bool(null_token)
        self.null_index = n_class if null_token else None
        self.has_aux = bool(aux)
        self.aux_weight = float(aux_weight)
        self.aux_tmax = float(aux_tmax)
        self.register_buffer("abar", cosine_abar(n_steps), persistent=True)

    # -- training ---------------------------------------------------------
    def loss(self, x0: torch.Tensor, labels: torch.Tensor,
             generator: torch.Generator | None = None,
             p_uncond: float = 0.0, snr_gamma: float = 0.0) -> torch.Tensor:
        return self.loss_with_aux(x0, labels, generator, p_uncond, snr_gamma)[0]

    def loss_with_aux(self, x0: torch.Tensor, labels: torch.Tensor,
                      generator: torch.Generator | None = None,
                      p_uncond: float = 0.0,
                      snr_gamma: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """(total_loss, diagnostics) where diagnostics = (mse, ce, aux_acc, n_masked)."""
        b = x0.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=x0.device, generator=generator)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        a = self.abar.to(x0.device)[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        if p_uncond > 0.0 and self.null_token:
            drop = torch.rand(b, device=x0.device, generator=generator) < p_uncond
            if bool(drop.any()):
                labels = torch.where(drop, torch.full_like(labels, self.null_index), labels)

        if not self.has_aux:
            pred = self.unet(xt, t, labels)
            mse = F.mse_loss(pred, eps)
            return mse, (mse.detach(), xt.new_zeros(()), xt.new_zeros(()), 0)

        pred, feat = self.unet.forward_aux(xt, t, labels)
        if snr_gamma <= 0.0:
            mse = F.mse_loss(pred, eps)
        else:
            snr = a / (1 - a)
            w = torch.where(snr > snr_gamma, snr_gamma / snr, torch.ones_like(snr))
            per = F.mse_loss(pred, eps, reduction="none").mean(dim=(1, 2, 3), keepdim=True)
            mse = (w * per).mean()

        keep = (t.float() / self.n_steps) <= self.aux_tmax
        n_keep = int(keep.sum())
        if n_keep == 0:
            return mse, (mse.detach(), xt.new_zeros(()), xt.new_zeros(()), 0)
        logits = feat[keep]
        ce = F.cross_entropy(logits, labels[keep])
        acc = (logits.argmax(dim=1) == labels[keep]).float().mean()
        total = mse + self.aux_weight * (b / n_keep) * ce
        return total, (mse.detach(), ce.detach(), acc.detach(), n_keep)

    # -- sampling ---------------------------------------------------------
    def _initial_noise(self, n: int, seed: int, device: torch.device) -> torch.Tensor:
        rows = []
        for i in range(n):
            g = torch.Generator().manual_seed((int(seed) * 1_000_003 + i) & _MASK)
            rows.append(torch.randn(3, 32, 32, generator=g))
        return torch.stack(rows).to(device)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, seed: int, steps: int | None = None,
               guidance: float = 1.0) -> torch.Tensor:
        steps = steps or self.ddim_steps
        device = labels.device
        abar = self.abar.to(device)
        x = self._initial_noise(len(labels), seed, device)
        neg_labels = (torch.full_like(labels, self.null_index) if self.null_token
                      else 1 - labels)
        ts = torch.linspace(self.n_steps - 1, 0, steps + 1).round().long()
        for i in range(steps):
            t = ts[i]
            t_prev = ts[i + 1]
            a = abar[t]
            a_prev = abar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            tt = torch.full((len(labels),), int(t), device=device, dtype=torch.long)
            eps = self.unet(x, tt, labels)
            if guidance != 1.0:
                neg = self.unet(x, tt, neg_labels)
                eps = neg + guidance * (eps - neg)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x


def save(model: DiffusionModel, path: str | Path, extra: dict | None = None) -> None:
    payload = {"kind": "ddpm", "state_dict": model.state_dict(),
               "base": model.unet.in_conv.out_channels, "emb": model.unet.emb_dim,
               "n_steps": model.n_steps, "ddim_steps": model.ddim_steps,
               "cond": model.cond, "null_token": model.null_token,
               "aux": model.has_aux, "aux_weight": model.aux_weight,
               "aux_tmax": model.aux_tmax, "aux_src": model.aux_src}
    payload.update(extra or {})
    torch.save(payload, path)


def load_payload(payload: dict) -> DiffusionModel:
    model = DiffusionModel(base=int(payload.get("base", 64)), emb=int(payload.get("emb", 128)),
                           n_steps=int(payload.get("n_steps", 1000)),
                           ddim_steps=int(payload.get("ddim_steps", 50)),
                           cond=str(payload.get("cond", "add")),
                           null_token=bool(payload.get("null_token", False)),
                           aux=bool(payload.get("aux", False)),
                           aux_weight=float(payload.get("aux_weight", 0.0)),
                           aux_tmax=float(payload.get("aux_tmax", 0.25)),
                           aux_src=str(payload.get("aux_src", "mid")))
    model.load_state_dict(payload["state_dict"])
    return model
