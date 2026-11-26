"""Train model B: class-conditional DDPM on the images described by submission/manifest.csv.

Only images rebuilt from the manifest are used; the dev set is never read; no pretrained
weights are loaded anywhere (LIMITS.md §1.1, §4).

    python train_diffusion.py --manifest submission/manifest.csv --out submission/weights.pt \
        --log-dir results/experiments/B_ddpm --steps 40000 --batch 64
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter_kit"))
sys.path.insert(0, str(ROOT / "submission"))

from train import build_dataset  # noqa: E402  (manifest -> [N,3,32,32] via the official crop)

# `diffusion` is imported inside main(), after `--code-dir` has been put on sys.path.  The shipped
# tree (`submission/`) stays the default, so every existing command behaves exactly as before; an
# experimental arm passes `--code-dir pipeline/arms/<arm>` and gets that tree's `diffusion`/`unet`
# instead.  Keeping the arm code out of `submission/` is deliberate: the graded package is frozen
# and must not grow an unused sampler while an experiment is running.


@torch.no_grad()
def sample_grid(model: DiffusionModel, device: torch.device, path: Path, n: int = 8,
                seed: int = 1234, guidance: float = 1.0) -> None:
    from torchvision.utils import save_image
    labels = torch.arange(2).repeat_interleave(n).to(device)
    x = model.sample(labels, seed, guidance=guidance)
    save_image(((x + 1) / 2).clamp(0, 1), path, nrow=n, padding=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="submission/manifest.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--code-dir", default="submission",
                    help="directory holding unet.py/diffusion.py/generate.py to train with "
                         "(default: the shipped tree; arms pass pipeline/arms/<arm>)")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight of the bottleneck cross-entropy term (0 = off; requires a "
                         "code dir whose DiffusionModel has loss_with_aux)")
    ap.add_argument("--aux-tmax", type=float, default=0.25,
                    help="apply the auxiliary CE only to t/T <= aux_tmax")
    ap.add_argument("--aux-src", choices=("mid", "stem"), default="mid",
                    help="feature the auxiliary head reads: mid = 8x8 bottleneck (conditioned), "
                         "stem = 32x32 in_conv output (unconditioned control)")
    ap.add_argument("--out", default="submission/weights.pt")
    ap.add_argument("--log-dir", default="results/experiments/B_ddpm")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200,
                    help="linear warmup steps (AdamW is unstable at full lr from step 0)")
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--lr-floor", type=float, default=0.01,
                    help="cosine decays to lr_floor * lr")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ema-list", default="",
                    help="extra EMA decays to track in parallel, comma separated "
                         "(e.g. 0.995,0.999,0.9995). Each is written as weights_ema<decay>.pt so "
                         "'same moment, different decay' is measurable without a retrain.")
    ap.add_argument("--loss-weight", choices=("none", "minsnr"), default="none",
                    help="none = L_simple; minsnr = Min-SNR weighting min(SNR,g)/SNR on the SAME "
                         "eps-prediction objective (no v-prediction, sampler untouched)")
    ap.add_argument("--snr-gamma", type=float, default=5.0,
                    help="gamma for --loss-weight minsnr")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--cond", choices=("add", "film"), default="add",
                    help="how the conditioning vector enters each residual block")
    ap.add_argument("--p-uncond", type=float, default=0.0,
                    help="label dropout probability for classifier-free guidance")
    ap.add_argument("--null-token", action="store_true",
                    help="add a null embedding row and enable CFG sampling")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save an EMA snapshot every N steps into log_dir (0 disables)")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="guidance used for the preview grids only")
    args = ap.parse_args()

    if args.p_uncond > 0.0 and not args.null_token:
        ap.error("--p-uncond > 0 requires --null-token (there is no row to drop the label to)")

    code_dir = (ROOT / args.code_dir).resolve()
    if not (code_dir / "diffusion.py").exists():
        ap.error(f"--code-dir {code_dir} has no diffusion.py")
    sys.path.insert(0, str(code_dir))
    from diffusion import DiffusionModel, save as save_ckpt  # noqa: E402
    print(f"code dir: {code_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    x, y = build_dataset(args.manifest, args.data_root, args.limit)
    print(f"dataset: {tuple(x.shape)} cats={int((y == 0).sum())} dogs={int((y == 1).sum())}")
    x = x * 2 - 1
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True, drop_last=True)

    model_kw = dict(base=args.base, emb=args.emb, n_steps=args.timesteps,
                    ddim_steps=args.ddim_steps, cond=args.cond, null_token=args.null_token)
    if args.aux_weight > 0.0:
        # Only the aux arm's DiffusionModel accepts these; the shipped tree would raise TypeError,
        # which is the correct outcome for "ask for a head the code cannot build".
        model_kw.update(aux=True, aux_weight=args.aux_weight, aux_tmax=args.aux_tmax,
                        aux_src=args.aux_src)
    model = DiffusionModel(**model_kw).to(device)
    n_unet = sum(p.numel() for p in model.unet.parameters())
    print(f"unet params: {n_unet/1e6:.2f}M  cond={args.cond} null_token={args.null_token} "
          f"p_uncond={args.p_uncond} code_dir={args.code_dir}")
    assert n_unet <= 20_000_000, "model exceeds the 20M parameter limit"

    use_aux = args.aux_weight > 0.0
    if use_aux and not hasattr(model, "loss_with_aux"):
        ap.error(f"--aux-weight > 0 but {args.code_dir}/diffusion.py has no loss_with_aux")
    if use_aux:
        print(f"aux head: weight {args.aux_weight} on t/T <= {args.aux_tmax}, src={args.aux_src}, "
              f"{sum(p.numel() for p in model.unet.aux_head.parameters())} parameters")

    # Also count the schedule buffer, so the reported total matches what the evaluator sees.
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_total/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    # Optional extra decays.  They are sampling-side only: the gradient trajectory is identical,
    # so "same step, different decay" is free to measure, whereas "different step, same decay"
    # needs the --ckpt-every snapshots.  Only `--ema` is used for the snapshots and weights.pt.
    ema_extra: dict[float, dict] = {}
    if args.ema_list:
        for d in [float(x) for x in args.ema_list.split(",") if x.strip()]:
            if abs(d - args.ema) < 1e-12:
                continue
            ema_extra[d] = {k: v.clone() for k, v in ema.items()}
        print(f"extra EMA decays tracked: {sorted(ema_extra)}")

    gamma = args.snr_gamma if args.loss_weight == "minsnr" else 0.0
    if gamma > 0.0:
        # Report the expected scale of the weighted loss so the log cannot be misread later.
        abar = model.abar.to(device)
        snr_all = abar / (1 - abar)
        w_all = torch.where(snr_all > gamma, gamma / snr_all, torch.ones_like(snr_all))
        print(f"loss weighting: Min-SNR gamma={gamma}  mean w(t) over t=0..{args.timesteps - 1}"
              f" = {float(w_all.mean()):.4f}  (expect ~0.82; loss values are NOT comparable "
              f"with unweighted runs)")

    def set_lr(step: int) -> float:
        """Linear warmup then (optionally) cosine decay to `lr_floor * lr`."""
        if args.warmup > 0 and step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        elif args.lr_schedule == "cosine":
            p = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * (args.lr_floor + (1 - args.lr_floor)
                            * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
        else:
            lr = args.lr
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    @torch.no_grad()
    def update_ema(decay: float) -> None:
        for k, v in model.state_dict().items():
            if k in ema:
                ema[k].mul_(decay).add_(v.detach(), alpha=1 - decay)

    step = 0
    losses: list[tuple[int, float]] = []
    aux_rows: list[tuple[int, float, float, float, int]] = []
    t0 = time.perf_counter()
    epoch = 0

    def ema_payload() -> dict:
        """A checkpoint carrying the EMA weights (that is what we ship and evaluate)."""
        live = model.state_dict()
        merged = {k: (ema[k] if k in ema else v) for k, v in live.items()}
        return {"kind": "ddpm", "state_dict": merged,
                "base": args.base, "emb": args.emb, "n_steps": args.timesteps,
                "ddim_steps": args.ddim_steps, "cond": args.cond,
                "null_token": args.null_token, "train_steps": step,
                "scheme": getattr(model, "scheme", "ddpm"),
                "aux": bool(getattr(model, "has_aux", False)),
                "aux_weight": getattr(model, "aux_weight", 0.0),
                "aux_tmax": getattr(model, "aux_tmax", 0.25)}

    while step < args.steps:
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            lr_now = set_lr(step)
            if use_aux:
                loss, diag = model.loss_with_aux(real, labels, p_uncond=args.p_uncond,
                                                 snr_gamma=gamma)
            else:
                loss = model.loss(real, labels, p_uncond=args.p_uncond, snr_gamma=gamma)
                diag = None
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(args.ema)
            if ema_extra:
                live = model.state_dict()
                for d, store in ema_extra.items():
                    for k, v in live.items():
                        if k in store:
                            store[k].mul_(d).add_(v.detach(), alpha=1 - d)
            step += 1
            losses.append((step, float(loss)))
            if step % 100 == 0:
                print(f"step {step}/{args.steps} loss={float(loss):.4f} lr={lr_now:.2e} "
                      f"{(time.perf_counter()-t0)/step*1000:.0f}ms/step", flush=True)
                if diag is not None:
                    mse, ce, acc, n_keep = diag
                    aux_rows.append((step, float(mse), float(ce), float(acc), int(n_keep)))
                    print(f"    mse={float(mse):.4f} ce={float(ce):.4f} "
                          f"aux_acc={float(acc):.4f} n_low_noise={int(n_keep)}", flush=True)
            if args.ckpt_every and step % args.ckpt_every == 0 and step < args.steps:
                snap = log_dir / f"snap_step{step:06d}.pt"
                torch.save(ema_payload(), snap)
                print(f"  snapshot -> {snap.name} ({snap.stat().st_size/1e6:.1f} MB)", flush=True)
            if step % args.sample_every == 0:
                model.eval()
                sample_grid(model, device, log_dir / f"samples_step{step:06d}.png",
                            guidance=args.guidance)
                model.train()
            if step >= args.steps:
                break
        epoch += 1

    elapsed = time.perf_counter() - t0
    model.eval()
    state = model.state_dict()
    state.update({k: v for k, v in ema.items()})
    model.load_state_dict(state)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(model, out, {"train_steps": args.steps, "batch": args.batch, "lr": args.lr,
                           "lr_schedule": args.lr_schedule, "warmup": args.warmup,
                           "p_uncond": args.p_uncond, "ema": args.ema,
                           "loss_weight": args.loss_weight, "snr_gamma": gamma,
                           "code_dir": args.code_dir,
                           "scheme": getattr(model, "scheme", "ddpm")})
    # Extra decays: same step, different decay.  Experiment directory only -- never submission/.
    ema_files = []
    for d, store in sorted(ema_extra.items()):
        state = {k: (store[k] if k in store else v) for k, v in model.state_dict().items()}
        p = out.parent / f"{out.stem}_ema{d:g}.pt"
        torch.save({"kind": "ddpm", "state_dict": state, "base": args.base, "emb": args.emb,
                    "n_steps": args.timesteps, "ddim_steps": args.ddim_steps,
                    "cond": args.cond, "null_token": args.null_token,
                    "train_steps": args.steps, "ema": d, "loss_weight": args.loss_weight,
                    "snr_gamma": gamma}, p)
        ema_files.append(str(p))
        print(f"  extra EMA {d:g} -> {p.name} ({p.stat().st_size/1e6:.1f} MB)")

    sample_grid(model, device, log_dir / "samples_final.png", guidance=args.guidance)
    with open(log_dir / "losses.csv", "w", encoding="utf-8") as f:
        f.write("step,loss\n")
        for s, v in losses:
            f.write(f"{s},{v:.5f}\n")
    if aux_rows:
        with open(log_dir / "auxloss.csv", "w", encoding="utf-8") as f:
            f.write("step,mse,ce,aux_acc,n_low_noise\n")
            for s, m, c, a, n in aux_rows:
                f.write(f"{s},{m:.5f},{c:.5f},{a:.5f},{n}\n")

    config = {
        "model": "class-conditional DDPM (cosine schedule, eps-prediction, GroupNorm UNet)",
        "manifest": str(args.manifest), "rows": int(x.shape[0]), "steps": step,
        "epochs_run": epoch, "batch": args.batch, "lr": args.lr, "ema_decay": args.ema,
        "lr_schedule": args.lr_schedule, "warmup": args.warmup, "lr_floor": args.lr_floor,
        "cond": args.cond, "null_token": args.null_token, "p_uncond": args.p_uncond,
        "loss_weight": args.loss_weight, "snr_gamma": gamma,
        "code_dir": args.code_dir, "scheme": getattr(model, "scheme", "ddpm"),
        "aux_weight": args.aux_weight, "aux_tmax": args.aux_tmax, "aux_src": args.aux_src,
        "ema_extra_files": ema_files,
        "base": args.base, "emb": args.emb, "timesteps": args.timesteps,
        "ddim_steps": args.ddim_steps, "seed": args.seed, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "unet_params": n_unet, "train_seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / max(step, 1), 4),
        "loss_first": losses[0][1] if losses else None,
        "loss_last": losses[-1][1] if losses else None,
        "weights": str(out), "weights_bytes": out.stat().st_size,
    }
    (log_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB), log {log_dir}")


if __name__ == "__main__":
    main()
