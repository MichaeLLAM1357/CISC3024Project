"""Arm `edm_heun`: same 5.15M GroupNorm UNet, EDM noise levels + preconditioning + Heun.

Reference: Karras, Aittala, Aila & Laine, "Elucidating the Design Space of Diffusion-Based
Generative Models", NeurIPS 2022.  Only the parts that do not touch the backbone are changed.

What changes relative to `submission/diffusion.py`
--------------------------------------------------
1. **Noise levels, not timesteps.**  Training draws sigma ~ log-normal(P_mean=-1.2, P_std=1.2) and
   forms x = x0 + sigma * eps directly.  There is no `abar` table and no cosine schedule; the
   schedule only reappears at sampling time as the sigma grid.
2. **Preconditioning (parameter-free).**  With sigma_data = 0.5 (the paper's value for images in
   [-1, 1]):

       c_skip = sigma_d^2 / (sigma^2 + sigma_d^2)
       c_out  = sigma * sigma_d / sqrt(sigma^2 + sigma_d^2)
       c_in   = 1 / sqrt(sigma^2 + sigma_d^2)
       c_noise = ln(sigma) / 4

       D(x; sigma) = c_skip * x + c_out * F(c_in * x, c_noise, y)

   F is the *same* UNet.  These coefficients keep the input to F and its target at unit variance
   across all noise levels, which is the original claim being tested: the eps-parameterisation asks
   one network to regress eps whose scale varies by orders of magnitude over the trajectory.
3. **Loss weighting.**  lambda(sigma) = (sigma^2 + sigma_d^2) / (sigma * sigma_d)^2 applied to
   ||D - x0||^2, so no noise level dominates the gradient budget.
4. **Heun sampler (2nd order, deterministic).**  sigma grid with rho = 7 from sigma_max = 80 to
   sigma_min = 0.002, then a final step to 0.  N steps cost 2N-1 network evaluations, so 25 Heun
   steps (49 evals) match the shipped 50-step DDIM (50 evals).  Written with the same per-sample
   deterministic initial noise and the same antithetic guidance as the shipped sampler:

       D_tilde = D(x; sigma, 1-y) + w * ( D(x; sigma, y) - D(x; sigma, 1-y) )

   Guidance is applied to F's output, which is equivalent because c_skip and c_out do not depend
   on the label.

The UNet file is copied byte for byte from `submission/unet.py`; only the time input changes from
an integer timestep to `c_noise` mapped into [0, n_steps - 1] (the same range the shipped model's
`TimeEmbedding` was scaled for), so the embedding magnitudes stay comparable.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from unet import UNet

N_CLASS = 2
_MASK = (1 << 63) - 1

SIGMA_DATA = 0.5
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
P_MEAN = -1.2
P_STD = 1.2


class DiffusionModel(nn.Module):
    """D(x; sigma, y) with EDM preconditioning, plus the Heun sampler.

    The class keeps the same public surface as the shipped model (`loss`, `sample`,
    `_initial_noise`) so `generate.py` is byte-identical between the two arms.
    """

    def __init__(self, base: int = 64, emb: int = 128, n_steps: int = 1000,
                 n_class: int = N_CLASS, ddim_steps: int = 25,
                 cond: str = "add", null_token: bool = False,
                 sigma_data: float = SIGMA_DATA) -> None:
        super().__init__()
        n_rows = n_class + (1 if null_token else 0)
        self.unet = UNet(base=base, emb=emb, n_class=n_rows, cond=cond)
        self.n_steps = n_steps
        self.n_class = n_class
        self.ddim_steps = ddim_steps
        self.z_dim = 0
        self.cond = cond
        self.null_token = bool(null_token)
        self.null_index = n_class if null_token else None
        self.scheme = "edm"
        self.sigma_data = float(sigma_data)

    # -- sigma helpers ----------------------------------------------------
    def _t_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map sigma to the pseudo-timestep in [0, n_steps-1] the TimeEmbedding expects."""
        lo = torch.log(torch.tensor(SIGMA_MIN, device=sigma.device, dtype=sigma.dtype))
        hi = torch.log(torch.tensor(SIGMA_MAX, device=sigma.device, dtype=sigma.dtype))
        u = ((torch.log(sigma) - lo) / (hi - lo)).clamp(0.0, 1.0)
        return u * (self.n_steps - 1)

    def _precondition(self, sigma: torch.Tensor):
        s2 = sigma ** 2
        sd2 = self.sigma_data ** 2
        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * self.sigma_data / (s2 + sd2).sqrt()
        c_in = 1.0 / (s2 + sd2).sqrt()
        return c_skip, c_out, c_in

    def _denoise(self, x: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor,
                 neg_labels: torch.Tensor | None = None, guidance: float = 1.0) -> torch.Tensor:
        c_skip, c_out, c_in = self._precondition(sigma)
        t = self._t_from_sigma(sigma).reshape(sigma.shape[0])
        out = self.unet(x * c_in, t, labels)
        if guidance != 1.0 and neg_labels is not None:
            neg = self.unet(x * c_in, t, neg_labels)
            out = neg + guidance * (out - neg)
        return c_skip * x + c_out * out

    # -- training ---------------------------------------------------------
    def loss(self, x0: torch.Tensor, labels: torch.Tensor,
             generator: torch.Generator | None = None,
             p_uncond: float = 0.0, snr_gamma: float = 0.0) -> torch.Tensor:
        """EDM denoising loss with the lambda(sigma) weight (p_uncond/snr_gamma are inert here)."""
        b = x0.shape[0]
        sigma = torch.exp(torch.randn(b, device=x0.device, generator=generator) * P_STD
                          + P_MEAN)[:, None, None, None]
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        x = x0 + sigma * eps
        d = self._denoise(x, sigma, labels)
        lam = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (lam * (d - x0) ** 2).flatten(1).mean(dim=1).mean()

    # -- sampling ---------------------------------------------------------
    def _initial_noise(self, n: int, seed: int, device: torch.device) -> torch.Tensor:
        rows = []
        for i in range(n):
            g = torch.Generator().manual_seed((int(seed) * 1_000_003 + i) & _MASK)
            rows.append(torch.randn(3, 32, 32, generator=g))
        return torch.stack(rows).to(device)

    def _sigma_grid(self, steps: int) -> list[float]:
        inv = 1.0 / RHO
        i = torch.arange(steps, dtype=torch.float64) / max(steps - 1, 1)
        grid = (SIGMA_MAX ** inv + i * (SIGMA_MIN ** inv - SIGMA_MAX ** inv)) ** RHO
        return [float(v) for v in grid] + [0.0]

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, seed: int, steps: int | None = None,
               guidance: float = 1.0) -> torch.Tensor:
        """Deterministic Heun trajectory from sigma_max to 0; returns pixels in [-1, 1]."""
        steps = steps or self.ddim_steps
        device = labels.device
        sig = self._sigma_grid(steps)
        neg_labels = (torch.full_like(labels, self.null_index) if self.null_token
                      else 1 - labels)
        x = self._initial_noise(len(labels), seed, device) * SIGMA_MAX
        for i in range(steps):
            s_cur, s_next = sig[i], sig[i + 1]
            sc = torch.full((len(labels), 1, 1, 1), s_cur, device=device)
            d = (x - self._denoise(x, sc, labels, neg_labels, guidance)) / s_cur
            x_next = x + d * (s_next - s_cur)
            if s_next > 0.0:
                sn = torch.full((len(labels), 1, 1, 1), s_next, device=device)
                d2 = (x_next - self._denoise(x_next, sn, labels, neg_labels, guidance)) / s_next
                x = x + ((d + d2) / 2.0) * (s_next - s_cur)
            else:
                x = x_next
        return x


def save(model: DiffusionModel, path: str | Path, extra: dict | None = None) -> None:
    payload = {"kind": "ddpm", "scheme": "edm", "state_dict": model.state_dict(),
               "base": model.unet.in_conv.out_channels, "emb": model.unet.emb_dim,
               "n_steps": model.n_steps, "ddim_steps": model.ddim_steps,
               "cond": model.cond, "null_token": model.null_token,
               "sigma_data": model.sigma_data}
    payload.update(extra or {})
    torch.save(payload, path)


def load_payload(payload: dict) -> DiffusionModel:
    model = DiffusionModel(base=int(payload.get("base", 64)), emb=int(payload.get("emb", 128)),
                           n_steps=int(payload.get("n_steps", 1000)),
                           ddim_steps=int(payload.get("ddim_steps", 25)),
                           cond=str(payload.get("cond", "add")),
                           null_token=bool(payload.get("null_token", False)),
                           sigma_data=float(payload.get("sigma_data", SIGMA_DATA)))
    model.load_state_dict(payload["state_dict"])
    return model
