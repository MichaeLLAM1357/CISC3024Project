"""Zero-training-cost scan: pick the sampler signature, not new weights.

Why this exists
---------------
LIMITS.md §3 allows using dev metrics to choose the model / hyperparameters / checkpoint.
Training is the expensive part; *how* a trained DDPM is run is free.  This script sweeps the
sampler only, and every variant is graded by the official evaluator
(`starter_kit/evaluate.py`), not by a metric we invented.

Protocol (deliberately fixed so the numbers are comparable):
  * the graded checkpoint is copied unchanged; only `generate.py` differs between variants;
  * the manifest is never touched; no dev image or dev feature is used for anything except the
    evaluator's own reference statistics;
  * round 1 screened at 250/class, round 2 runs at the official 1000/class so the marks are real;
  * seed is always 0 -- never search over seeds, the hidden test uses a different one;
  * only the 15-mark total decides; FID alone is never a reason to keep a variant.

Discipline (see AGENTS.md): at most 6 variants per round, one factor at a time, and any variant
whose recall drops below 0.40 is discarded regardless of the other numbers.

Output naming (fixed 2026-11-24)
--------------------------------
Every artefact carries the checkpoint label, because the per-variant file used to be
`results/scan/<variant>_n<N>.json` with no checkpoint in the name -- screening a second checkpoint
with the same variant silently overwrote the first one's numbers.
Now: `results/scan/[<exp>/]<variant>__<ckpt>_n<N>.json` and `scan__<ckpt>_n<N>[__<tag>].json`,
where `<ckpt>` is `--ckpt-tag` or the checkpoint's parent directory name. An existing per-variant
file is never overwritten without `--force`; the run is skipped and reported instead.

Sampler knobs
-------------
  steps    DDIM trajectory length (eta = 0).
  w_max    guidance weight reached at the *last* DDIM step.
  w_from   first step index (0-based) where guidance turns on; 0 = from the start.
  ramp     True  -> linear 1.0 -> w_max between w_from and the last step
           False -> constant w_max from w_from onwards (0 before)
  w_cat / w_dog   per-class weights at the last step (None = same as w_max); they follow the same
                  ramp, and each sample only ever uses its own label.
  post     "" | "unsharp": deterministic per-sample unsharp mask (3x3 binomial blur, amount 0.35).

Guidance form, for the record:

    eps_tilde = eps_theta(x_t, t, neg) + w(t) * ( eps_theta(x_t, t, y) - eps_theta(x_t, t, neg) )

`neg` is chosen by `mode`:
  * "contrast" -- neg = 1 - y (antithetic).  Works with any 2-class checkpoint and is what the
    11/15 model used.
  * "nulltok"  -- neg = the null token (row n_class of the embedding).  Only valid for a
    checkpoint trained with label dropout, where that row was actually seen during training.
  * "null"     -- legacy: zero the embedding output through a forward hook.  Kept only because it
    is the documented failure (off-distribution negative, KID 49.0 vs 35.3); do not use it.
  * "none"     -- no guidance at all (w = 1).

Usage
-----
    python scan_posttrain.py                                  # the shipped config (DEFAULT_SET)
    python scan_posttrain.py --all --n-per-class 250           # every catalogued variant
    python scan_posttrain.py --only ramp15 --n-per-class 1000  # a specific variant, official n
    python scan_posttrain.py --weights results/experiments/B_ddpm3/weights.pt --only nt250
                                                              # screen a candidate checkpoint
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SCAN = ROOT / "results" / "scan"


def V(steps=50, w_max=1.0, w_from=0, ramp=False, mode="contrast", w_cat=None, w_dog=None,
      post=""):
    return dict(steps=steps, w_max=w_max, w_from=w_from, ramp=ramp, mode=mode,
                w_cat=w_cat, w_dog=w_dog, post=post)


VARIANTS = {
    # -- round 1: does guidance exist, and how many DDIM steps (screened at 250/class) --------
    "base_s50":     V(50, 1.00, 0, False, "none"),
    "steps35":      V(35, 1.00, 0, False, "none"),
    "steps75":      V(75, 1.00, 0, False, "none"),
    "steps100":     V(100, 1.00, 0, False, "none"),
    "null150":      V(50, 1.50, 0, False, "null"),
    "contrast115":  V(50, 1.15),
    "contrast130":  V(50, 1.30),
    "contrast150":  V(50, 1.50),
    "contrast175":  V(50, 1.75),
    "contrast200":  V(50, 2.00),
    "contrast250":  V(50, 2.50),          # shipped in round 1: 0.8425 / 89.71 / 0.5420 -> 11/15
    "contrast300":  V(50, 3.00),
    "contrast150s75": V(75, 1.50),
    # -- round 2: does the *schedule* matter (run at 1000/class) ------------------------------
    "ramp15":       V(50, 2.50, 35, True),    # w: 1 -> 2.5 over the last 15 steps
    "late15":       V(50, 2.50, 35, False),   # w = 2.5 only in the last 15 steps
    "ramp25":       V(50, 2.50, 25, True),    # w: 1 -> 2.5 over the last 25 steps
    "perclass":     V(50, 2.50, 0, False, w_cat=2.75, w_dog=2.25),
    "combo":        V(50, 2.50, 35, True, w_cat=2.75, w_dog=2.25),
    "post":         V(50, 2.50, 35, True, w_cat=2.75, w_dog=2.25, post="unsharp"),
    # -- round 3: model B3 (film conditioning + null token), true classifier-free guidance ------
    "nt100":        V(50, 1.00, 0, False, "nulltok"),   # plain conditional, no guidance
    "nt150":        V(50, 1.50, 0, False, "nulltok"),
    "nt200":        V(50, 2.00, 0, False, "nulltok"),
    "nt250":        V(50, 2.50, 0, False, "nulltok"),
    "nt300":        V(50, 3.00, 0, False, "nulltok"),
    "nt400":        V(50, 4.00, 0, False, "nulltok"),
    "ct250_B3":     V(50, 2.50, 0, False, "contrast"),  # same checkpoint, antithetic negative
    # -- round 4: does the 11/15 checkpoint keep gaining accuracy above w=3? --------------------
    "contrast350":  V(50, 3.50, 0, False, "contrast"),
    "contrast400":  V(50, 4.00, 0, False, "contrast"),
    "contrast450":  V(50, 4.50, 0, False, "contrast"),
}

DEFAULT_SET = ["contrast250"]   # the shipped configuration: reproduces the headline

GENERATE = '''"""Sampler variant {name}: steps={steps}, w_max={w_max}, w_from={w_from}, ramp={ramp}, post={post}.

Generated by scan_posttrain.py -- do not edit by hand.  Identical to submission/generate.py
except for the constants below and the guided denoising loop.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights.pt"

STEPS = {steps}
W_MAX = {w_max}
W_FROM = {w_from}
W_RAMP = {ramp}
MODE = "{mode}"
W_CAT = {w_cat}
W_DOG = {w_dog}
POST = "{post}"

_MASK = (1 << 63) - 1


def load_model(device: torch.device) -> nn.Module:
    import sys

    sys.path.insert(0, str(HERE))
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(ckpt, dict) and ckpt.get("kind") == "ddpm":
        from diffusion import load_payload
        model = load_payload(ckpt)
    else:
        from model import Generator
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model = Generator(z_dim=int(ckpt.get("z_dim", 96)),
                              width=int(ckpt.get("width", 256)),
                              norm=str(ckpt.get("norm", "bn")))
            ckpt = ckpt["state_dict"]
        else:
            model = Generator()
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def _noise(n: int, z_dim: int, seed: int) -> torch.Tensor:
    rows = []
    for i in range(n):
        g = torch.Generator().manual_seed((int(seed) * 1_000_003 + i) & _MASK)
        rows.append(torch.randn(z_dim, generator=g))
    return torch.stack(rows)


@contextmanager
def _zeroed_label(embedding):
    def hook(_module, _inputs, out):
        return torch.zeros_like(out)

    handle = embedding.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _progress(i: int) -> float:
    """0 -> no guidance at this step, 1 -> full weight.  Linear in the ramp window."""
    if i < W_FROM:
        return 0.0
    if not W_RAMP or W_FROM >= STEPS - 1:
        return 1.0
    return (i - W_FROM) / (STEPS - 1 - W_FROM)


@torch.no_grad()
def _sample(model: nn.Module, labels: torch.Tensor, seed: int) -> torch.Tensor:
    device = labels.device
    abar = model.abar.to(device)
    x = model._initial_noise(len(labels), seed, device)
    ts = torch.linspace(model.n_steps - 1, 0, STEPS + 1).round().long()
    for i in range(STEPS):
        t, t_prev = ts[i], ts[i + 1]
        a = abar[t]
        a_prev = abar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
        tt = torch.full((len(labels),), int(t), device=device, dtype=torch.long)
        eps = model.unet(x, tt, labels)
        p = _progress(i)
        if p > 0.0 and W_MAX != 1.0:
            if MODE == "contrast":
                eps_neg = model.unet(x, tt, 1 - labels)
            elif MODE == "nulltok":
                # The real unconditional token learnt during training (label dropout).
                if getattr(model, "null_index", None) is None:
                    raise RuntimeError("nulltok needs a checkpoint trained with --null-token")
                eps_neg = model.unet(x, tt, torch.full_like(labels, model.null_index))
            else:
                with _zeroed_label(model.unet.label):
                    eps_neg = model.unet(x, tt, labels)
            if W_CAT is None:
                w = 1.0 + p * (W_MAX - 1.0)      # python float: bit-identical to diffusion.py
            else:
                wc = 1.0 + p * (W_CAT - 1.0)
                wd = 1.0 + p * (W_DOG - 1.0)
                w = torch.where((labels == 0)[:, None, None, None],
                                torch.tensor(wc, device=device),
                                torch.tensor(wd, device=device))
            eps = eps_neg + w * (eps - eps_neg)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x


@torch.no_grad()
def generate(model: nn.Module, labels: torch.Tensor, seed: int) -> torch.Tensor:
    device = next(model.parameters()).device
    labels = labels.to(device=device, dtype=torch.long)
    if hasattr(model, "sample"):
        x = _sample(model, labels, seed)
    else:
        z = _noise(len(labels), model.z_dim, seed).to(device)
        x = model(z, labels)
    out = ((x + 1.0) / 2.0).clamp(0.0, 1.0).float()
    if POST == "unsharp":
        k = torch.tensor([1.0, 2.0, 1.0], device=out.device, dtype=out.dtype)
        k = k[:, None] * k[None, :]
        k = (k / k.sum()).expand(3, 1, 3, 3)
        blur = F.conv2d(out, k, padding=1, groups=3)
        out = (out + 0.35 * (out - blur)).clamp(0.0, 1.0)
    return out
'''


def build(name: str, weights: Path | None = None) -> Path:
    spec = VARIANTS[name]
    d = SCAN / f"sub_{name}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in ("manifest.csv", "diffusion.py", "unet.py"):
        shutil.copy2(SUB / f, d / f)
    # `weights` lets a candidate checkpoint be screened without overwriting submission/weights.pt.
    shutil.copy2(weights or (SUB / "weights.pt"), d / "weights.pt")
    (d / "generate.py").write_text(GENERATE.format(name=name, **spec), encoding="utf-8")
    return d


def run(name: str, n_per_class: int, seed: int, weights: Path | None = None,
        out_dir: Path | None = None, ckpt: str = "submission", force: bool = False) -> dict:
    d = build(name, weights)
    # The checkpoint label is part of the filename: two checkpoints screened with the same
    # variant name used to write the same file and the second run silently replaced the first.
    out = (out_dir or SCAN) / f"{name}__{ckpt}_n{n_per_class}.json"
    log = out.with_suffix(".log")
    if out.exists() and not force:
        print(f"  !! {out} already exists -- refusing to overwrite (pass --force to replace). "
              f"Change --ckpt-tag / --tag, or delete the file deliberately.")
        return {"name": name, "ckpt": ckpt, "failed": True, "exit": "refused-to-overwrite"}
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call(
            [sys.executable, "-u", str(ROOT / "starter_kit" / "evaluate.py"),
             "--submission", str(d), "--n-per-class", str(n_per_class), "--seed", str(seed),
             "--out", str(out)], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    wall = time.perf_counter() - t0
    if code != 0 or not out.exists():
        print(f"  !! {name} failed (exit {code}), see {log}")
        return {"name": name, "failed": True, "exit": code}
    r = json.loads(out.read_text(encoding="utf-8"))
    m = r["metrics"]
    row = {"name": name, "ckpt": ckpt, "n_per_class": n_per_class, **VARIANTS[name], "failed": False,
           "json": str(out).replace("\\", "/"),
           "accuracy": m["accuracy"], "fid": m["fid"], "kid": m["kid"],
           "precision": m["precision"], "recall": m["recall"],
           "memorization": m["memorization_rate"], "marks": r["marks"]["total"],
           "tiers": {k: v["tier"] for k, v in r["marks"]["per_metric"].items()},
           "sec_per_1000": r["resources"]["seconds_per_1000"],
           "deterministic": r["checks"]["deterministic"],
           "single_vs_mixed": [r["checks"]["single_class_batch_accuracy"],
                               r["checks"]["mixed_batch_accuracy"]],
           "violations": r["violations"], "wall_s": round(wall, 1)}
    if row["recall"] < 0.40:
        row["discard"] = "recall < 0.40"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every catalogued variant")
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-dirs", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for the summary file, so a new round never overwrites an old one")
    ap.add_argument("--ckpt-tag", default="",
                    help="checkpoint label used in every output filename (default: derived from "
                         "--weights, e.g. B_ddpm4, or 'submission')")
    ap.add_argument("--exp", default="",
                    help="subdirectory of results/scan for this experiment (e.g. E2)")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing per-variant json")
    ap.add_argument("--weights", default=None,
                    help="checkpoint to screen (default: submission/weights.pt)")
    args = ap.parse_args()

    names = list(VARIANTS) if args.all else (args.only or DEFAULT_SET)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}; choose from {list(VARIANTS)}")

    SCAN.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights) if args.weights else None
    if weights and not weights.exists():
        raise SystemExit(f"no such checkpoint: {weights}")
    # Label the files with the checkpoint, always -- that is the fix for the silent overwrite.
    ckpt = args.ckpt_tag or (weights.parent.name if weights else "submission")
    out_dir = SCAN / args.exp if args.exp else SCAN
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint label: {ckpt}   output dir: {out_dir}")
    rows = []
    for i, name in enumerate(names, 1):
        spec = VARIANTS[name]
        print(f"[{i}/{len(names)}] {name} steps={spec['steps']} w_max={spec['w_max']} "
              f"w_from={spec['w_from']} ramp={spec['ramp']} "
              f"w_cat={spec['w_cat']} w_dog={spec['w_dog']} post={spec['post'] or '-'}",
              flush=True)
        row = run(name, args.n_per_class, args.seed, weights, out_dir, ckpt, args.force)
        rows.append(row)
        if not row.get("failed"):
            print(f"      acc {row['accuracy']:.4f}  KID {row['kid']*1000:.2f}e-3  "
                  f"FID {row['fid']:.2f}  prec {row['precision']:.3f}  "
                  f"rec {row['recall']:.4f}  {row['marks']}/15  {row['wall_s']}s"
                  f"{'  DISCARD (' + row['discard'] + ')' if row.get('discard') else ''}",
                  flush=True)
        if not args.keep_dirs:
            shutil.rmtree(SCAN / f"sub_{name}", ignore_errors=True)

    ok = [r for r in rows if not r.get("failed")]
    ok.sort(key=lambda r: (-r["marks"], -r["recall"], r["kid"]))
    print("\n%-12s %5s %8s %8s %8s %7s %7s %6s %5s" %
          ("variant", "steps", "w", "acc", "KID*1e3", "prec", "recall", "marks", "s/1k"))
    for r in ok:
        w = f"{r['w_max']:.2f}" + (f"@{r['w_from']}" if r["w_from"] else "")
        print("%-12s %5d %8s %8.4f %8.2f %7.3f %7.4f %6d %5.1f %s" %
              (r["name"], r["steps"], w, r["accuracy"], r["kid"] * 1000, r["precision"],
               r["recall"], r["marks"], r["sec_per_1000"],
               ("DISCARD: " + r["discard"]) if r.get("discard") else ""))

    tag = f"__{args.tag}" if args.tag else ""
    summary = out_dir / f"scan__{ckpt}_n{args.n_per_class}{tag}.json"
    summary.write_text(json.dumps({"protocol": {"n_per_class": args.n_per_class,
                                                "seed": args.seed, "batch_size": 500,
                                                "checkpoint": str(weights) if weights
                                                else "submission/weights.pt",
                                                "ckpt_label": ckpt},
                                   "variants": rows}, indent=2), encoding="utf-8")
    print(f"\nsummary -> {summary}")


if __name__ == "__main__":
    main()
