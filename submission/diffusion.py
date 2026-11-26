"""Gaussian diffusion with class conditioning (model B), trained from scratch.

Forward process (Ho, Jain & Abbeel 2020):

    x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps,     eps ~ N(0, I)

Training objective actually implemented (epsilon prediction, simple loss):

    L_simple = E_{t, x0, eps, y} || eps - eps_theta(sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, t, y) ||^2

`abar` follows the cosine schedule of Nichol & Dhariwal (2021), which keeps more signal at
small t than the linear schedule and works better at 32x32.

The shipped checkpoint uses the plain objective above.  A Min-SNR weighted variant of the *same*
objective (identical eps-prediction, identical sampler) is available for ablation:

    L_minSNR = E_t [ w(t) || eps - eps_theta(.) ||^2 ],  w(t) = min(SNR(t), gamma) / SNR(t)

with `snr_gamma` = gamma; `snr_gamma = 0` reproduces `L_simple` bit-for-bit.

Sampling is deterministic DDIM (Song, Meng & Ermon 2021) with eta = 0: the trajectory is fully
determined by the initial noise, so `generate()` stays deterministic and each image depends only
on its own label and the seed.  Only the initial noise is random, which is also why the batch
composition cannot affect a single sample.

Guidance (inference only)
-------------------------
Two forms are implemented, selected by the checkpoint:

1. **Classifier-free guidance with a null token** (Ho & Salimans 2022), when the model was trained
   with label dropout (`null_token=True`, dropout `p_uncond`).  Row `n_class` of the embedding is
   the null/unconditional token `∅`, seen during training with probability `p_uncond`:

       eps_tilde(x_t, t, y) = eps_theta(x_t, t, ∅) + w * ( eps_theta(x_t, t, y) - eps_theta(x_t, t, ∅) )

   `∅` is a true "no class" baseline, so extrapolating away from it sharpens the class without
   pushing the sample off the data manifold the way an adversarial negative does.

2. **Antithetic guidance**, for checkpoints trained *without* label dropout (no null row exists).
   The other class plays the role of the negative prompt:

       eps_tilde(x_t, t, y) = eps_theta(x_t, t, 1 - y)
                              + w * ( eps_theta(x_t, t, y) - eps_theta(x_t, t, 1 - y) )

Both terms are per-sample, so batch-composition independence is untouched.  `w` was chosen with
the official evaluator (seed 0), never with a dev image statistic: see `scan_posttrain.py` and the
sweep tables in README.md.
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
                 cond: str = "add", null_token: bool = False) -> None:
        super().__init__()
        # With label dropout the UNet carries one extra embedding row: the null token.
        n_rows = n_class + (1 if null_token else 0)
        self.unet = UNet(base=base, emb=emb, n_class=n_rows, cond=cond)
        self.n_steps = n_steps
        self.n_class = n_class
        self.ddim_steps = ddim_steps
        self.z_dim = 0                      # keep the generate() contract uniform with model A
        self.cond = cond
        self.null_token = bool(null_token)
        self.null_index = n_class if null_token else None
        self.register_buffer("abar", cosine_abar(n_steps), persistent=True)

    # -- training ---------------------------------------------------------
    def loss(self, x0: torch.Tensor, labels: torch.Tensor,
             generator: torch.Generator | None = None,
             p_uncond: float = 0.0, snr_gamma: float = 0.0) -> torch.Tensor:
        """Denoising loss; `p_uncond` drops the label to the null token with that probability.

        `snr_gamma > 0` switches on Min-SNR loss weighting (Hang et al. 2023), gamma = snr_gamma:

            w(t) = min( SNR(t), gamma ) / SNR(t),       SNR(t) = abar_t / (1 - abar_t)
            L    = E_t [ w(t) * || eps - eps_theta(x_t, t, y) ||^2 ]

        The weight is per sample, so batch composition is still irrelevant.  Nothing about the
        sampler changes: the model still predicts eps and DDIM is untouched -- only the scalar
        that the training loss is multiplied by differs.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=x0.device, generator=generator)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        a = self.abar.to(x0.device)[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        if p_uncond > 0.0 and self.null_token:
            drop = torch.rand(b, device=x0.device, generator=generator) < p_uncond
            if bool(drop.any()):
                labels = torch.where(drop, torch.full_like(labels, self.null_index), labels)
        pred = self.unet(xt, t, labels)
        if snr_gamma <= 0.0:
            return F.mse_loss(pred, eps)
        snr = a / (1 - a)                                   # [b,1,1,1]; +inf at abar == 1
        w = torch.where(snr > snr_gamma, snr_gamma / snr, torch.ones_like(snr))
        per_sample = F.mse_loss(pred, eps, reduction="none").mean(dim=(1, 2, 3), keepdim=True)
        return (w * per_sample).mean()

    # -- sampling ---------------------------------------------------------
    def _initial_noise(self, n: int, seed: int, device: torch.device) -> torch.Tensor:
        """Sample i depends only on (seed, i) -- never on n or on the other labels."""
        rows = []
        for i in range(n):
            g = torch.Generator().manual_seed((int(seed) * 1_000_003 + i) & _MASK)
            rows.append(torch.randn(3, 32, 32, generator=g))
        return torch.stack(rows).to(device)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, seed: int, steps: int | None = None,
               guidance: float = 1.0) -> torch.Tensor:
        """DDIM eta=0 trajectory -> images in [-1, 1].

        `guidance` > 1 applies classifier-free guidance.  The negative is the null token when the
        checkpoint was trained with label dropout, otherwise the other class (antithetic form).
        Either way the negative is a function of the label alone, so each sample still depends
        only on (seed, i, label).  Costs one extra forward pass per step.
        """
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
               "cond": model.cond, "null_token": model.null_token}
    payload.update(extra or {})
    torch.save(payload, path)


def load_payload(payload: dict) -> DiffusionModel:
    model = DiffusionModel(base=int(payload.get("base", 64)), emb=int(payload.get("emb", 128)),
                           n_steps=int(payload.get("n_steps", 1000)),
                           ddim_steps=int(payload.get("ddim_steps", 50)),
                           cond=str(payload.get("cond", "add")),
                           null_token=bool(payload.get("null_token", False)))
    model.load_state_dict(payload["state_dict"])
    return model
