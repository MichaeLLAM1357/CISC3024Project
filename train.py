"""Train model A: a class-conditional DCGAN on the images described by submission/manifest.csv.

Only images rebuilt from the manifest are used (LIMITS.md §1.1).  The dev set is never read
here.  No pretrained weights are loaded.

Examples
--------
    python train.py --manifest submission/manifest.csv --out submission/weights.pt
    python train.py --manifest submission/manifest.csv --limit 64 --steps 200 --out /tmp/smoke.pt
"""

# LEGACY TEST-ONLY: imported by tests/test_overfit_single_batch.py.
# Not a training entry point for PROJECT3; use train_diffusion.py.

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter_kit"))
sys.path.insert(0, str(ROOT / "submission"))

from catdog_eval.data import Pool, crop, pool_dir, to_tensor  # noqa: E402

from model import Discriminator, Generator  # noqa: E402


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({"pool_id": r["pool_id"], "x0": int(r["x0"]), "y0": int(r["y0"]),
                         "size": int(r["size"]), "angle": float(r["angle"]),
                         "label": int(r["label"])})
    return rows


def build_dataset(manifest_path: str | Path, data_root: str | Path = "data",
                  limit: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(images [N,3,32,32] in [0,1], labels [N]) rebuilt from the manifest with the official crop."""
    rows = read_manifest(manifest_path)
    if limit:
        rows = rows[:limit]
    pool = Pool(pool_dir(data_root))
    tiles = [np.asarray(crop(pool.image(r["pool_id"]), r["x0"], r["y0"], r["size"], r["angle"]))
             for r in rows]
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    return to_tensor(np.stack(tiles)), labels


def sample_grid(g: nn.Module, device: torch.device, path: Path, n: int = 8) -> None:
    from torchvision.utils import save_image
    labels = torch.arange(2).repeat_interleave(n).to(device)
    z = torch.randn(len(labels), g.z_dim, generator=torch.Generator().manual_seed(1234)).to(device)
    with torch.no_grad():
        x = ((g(z, labels) + 1) / 2).clamp(0, 1)
    save_image(x, path, nrow=n, padding=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="submission/manifest.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="submission/weights.pt")
    ap.add_argument("--log-dir", default="results/experiments/A_dcgan")
    ap.add_argument("--epochs", type=int, default=0, help="0 = derive from --steps")
    ap.add_argument("--steps", type=int, default=12000, help="generator steps (overrides epochs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=1e-4, help="two-timescale: D learns slower")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay for the saved generator")
    ap.add_argument("--z-dim", type=int, default=96)
    ap.add_argument("--width", type=int, default=256, help="generator base width")
    ap.add_argument("--norm", default="bn", choices=("bn", "gn"), help="generator normalisation")
    ap.add_argument("--d-width", type=int, default=64, help="discriminator base width")
    ap.add_argument("--r1", type=float, default=10.0, help="R1 gradient penalty gamma (0 = off)")
    ap.add_argument("--r1-every", type=int, default=16, help="apply R1 every N discriminator steps")
    ap.add_argument("--noise", type=float, default=0.1,
                    help="instance noise sigma on D inputs, decayed to --noise-floor")
    ap.add_argument("--noise-floor", type=float, default=0.03,
                    help="keep a little noise so D cannot lock onto high-frequency artefacts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N manifest rows")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample-every", type=int, default=200)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    x, y = build_dataset(args.manifest, args.data_root, args.limit)
    print(f"dataset: {tuple(x.shape)} cats={int((y == 0).sum())} dogs={int((y == 1).sum())}")
    x = x * 2 - 1  # [0,1] -> [-1,1]
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True,
                        drop_last=True, num_workers=args.num_workers)

    g = Generator(z_dim=args.z_dim, width=args.width, norm=args.norm).to(device)
    d = Discriminator(width=args.d_width).to(device)
    n_params = sum(p.numel() for p in g.parameters())
    print(f"generator params: {n_params/1e6:.2f}M (discriminator is not submitted)")
    assert n_params <= 20_000_000, "generator exceeds the 20M parameter limit"

    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    # EMA of the generator: the averaged weights sample noticeably better than the last
    # iterate and cost one extra copy of a 1.5M-parameter model.
    ema = {k: v.detach().clone() for k, v in g.state_dict().items()}

    @torch.no_grad()
    def update_ema(decay: float) -> None:
        for k, v in g.state_dict().items():
            e = ema[k]
            if e.dtype.is_floating_point:
                e.mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                e.copy_(v.detach())

    steps_total = args.steps or args.epochs * max(1, len(loader))
    epochs_total = args.epochs or math.ceil(steps_total / max(1, len(loader)))
    step = 0
    t0 = time.perf_counter()
    losses = []
    g.train()
    d.train()
    stop = False
    for epoch in range(epochs_total):
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            noise = torch.randn(real.shape[0], args.z_dim, device=device)
            fake = g(noise, labels)

            # discriminator: real (one-sided label smoothing) vs fake
            # Instance noise decays to zero: early on it stops the discriminator from
            # separating real from fake by high-frequency statistics alone, which is the
            # failure mode that produced blobs in runs A2/A3/A4.
            sigma = max(args.noise_floor, args.noise * (1.0 - step / steps_total)) \
                if args.noise > 0 else 0.0
            real_in = real + sigma * torch.randn_like(real) if sigma else real
            fake_in = fake.detach() + sigma * torch.randn_like(fake) if sigma else fake.detach()
            opt_d.zero_grad(set_to_none=True)
            d_real = d(real_in, labels)
            d_fake = d(fake_in, labels)
            loss_d = bce(d_real, torch.full_like(d_real, 0.9)) + bce(d_fake, torch.zeros_like(d_fake))
            if args.r1 > 0 and step % args.r1_every == 0:
                # R1: penalise the discriminator gradient on real data.  Without it D locks onto
                # high-frequency cues, G answers with noise, and the run collapses to blobs.
                real_g = real.detach().clone().requires_grad_(True)
                out_real = d(real_g, labels)
                grad = torch.autograd.grad(out_real.sum(), real_g, create_graph=True)[0]
                loss_d = loss_d + 0.5 * args.r1 * grad.flatten(1).pow(2).sum(1).mean()
            loss_d.backward()
            opt_d.step()

            # generator: non-saturating
            opt_g.zero_grad(set_to_none=True)
            d_fake = d(fake, labels)
            loss_g = bce(d_fake, torch.ones_like(d_fake))
            loss_g.backward()
            opt_g.step()
            update_ema(args.ema)

            step += 1
            losses.append((step, float(loss_d), float(loss_g)))
            if step % 50 == 0:
                print(f"step {step}/{steps_total} epoch {epoch} d={float(loss_d):.3f} "
                      f"g={float(loss_g):.3f} {(time.perf_counter()-t0)/step*1000:.0f}ms/step",
                      flush=True)
            if step % args.sample_every == 0:
                sample_grid(g, device, log_dir / f"samples_step{step:06d}.png")
            if step >= steps_total:
                stop = True
                break
        if stop:
            break

    elapsed = time.perf_counter() - t0
    g.eval()
    g.load_state_dict(ema)          # sample and ship the averaged generator
    sample_grid(g, device, log_dir / "samples_final.png")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": g.state_dict(), "z_dim": args.z_dim, "width": args.width,
                "norm": args.norm}, out)

    with open(log_dir / "losses.csv", "w", encoding="utf-8") as f:
        f.write("step,loss_d,loss_g\n")
        for row in losses:
            f.write(f"{row[0]},{row[1]:.5f},{row[2]:.5f}\n")

    config = {
        "model": f"conditional DCGAN ({args.norm}, projection D, spectral norm)",
        "manifest": str(args.manifest),
        "rows": int(x.shape[0]),
        "steps": step,
        "epochs_run": epoch + 1,
        "batch": args.batch,
        "lr": args.lr,
        "z_dim": args.z_dim,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "generator_params": n_params,
        "train_seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / max(step, 1), 4),
        "loss_first": losses[0][1:] if losses else None,
        "loss_last": losses[-1][1:] if losses else None,
        "ema_decay": args.ema,
        "norm": args.norm,
        "d_width": args.d_width,
        "instance_noise": args.noise,
        "lr_d": args.lr_d,
        "weights": str(out),
        "weights_bytes": out.stat().st_size,
    }
    (log_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB), log {log_dir}")


if __name__ == "__main__":
    main()
