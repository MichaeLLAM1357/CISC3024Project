"""Why does the aux-head CE collapse to ~1e-4 within a few hundred steps?

`pipeline/arms/aux_head` puts a linear head on the bottleneck `m` of a UNet whose every block also
receives the class embedding (`e = time(t) + label(y)`).  There are two possible reasons the head
solves its task instantly, and they mean different things for the report:

  (a) **it reads the conditioning.**  `m` already contains a label-dependent component that the
      mid-block adds as a bias, so a linear map can recover the label from `m` without looking at
      the image at all.  Then the CE term constrains nothing about the pixels and adds ~0 gradient
      after the first ~100 steps.
  (b) **it reads the pixels.**  At low noise `x_t` is a nearly clean image, so a linear read of the
      pooled bottleneck can genuinely classify it.  Then the CE is a real signal -- but it is a
      *discriminative read on x_t*, which helps classification and not synthesis: `generate()`
      has to produce the class from noise, and `load_model` throws the head away.

Test: run the same bottleneck through the same head three times, changing only the label that the
UNet is conditioned on:
    y_true  --  the real label
    y_zero  --  every sample conditioned on class 0 (no class information in the input)
    y_flip  --  1 - y
If CE(y_zero) is also ~0, the head is reading pixels (b).  If CE(y_zero) ~ ln 2 = 0.693, the head
was reading the conditioning (a).  The label-shift norm ||m(y) - m(1-y)|| / ||m|| says how big that
injected component is, independent of any head.

    python pipeline/probe_aux_head.py --weights results/ab/aux_head/weights.pt --limit 512
"""

from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # train.py (build_dataset) lives at the project root
sys.path.insert(0, str(ROOT / "starter_kit"))
sys.path.insert(0, str(ROOT / "submission"))

from train import build_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/ab/aux_head/weights.pt")
    ap.add_argument("--manifest", default="submission/manifest.csv")
    ap.add_argument("--data-root", default=os.environ.get("CATDOG_DATA", str(ROOT / "data")))
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--out", default="results/ab/probe_aux_head.json")
    ap.add_argument("--times", default="50,250,600,900")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    if not ckpt.get("aux", False):
        raise SystemExit(f"{args.weights} carries no aux head -- probe a trained aux arm")
    base, emb = int(ckpt["base"]), int(ckpt["emb"])
    n_steps = int(ckpt["n_steps"])

    sys.path.insert(0, str(ROOT / "pipeline" / "arms" / "aux_head"))
    for m in ("unet",):
        sys.modules.pop(m, None)
    from unet import UNet  # noqa: E402  (this is the aux-capable copy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(base=base, emb=emb, n_class=2, cond=str(ckpt.get("cond", "add")),
               aux=True, aux_classes=2, aux_src=str(ckpt.get("aux_src", "mid"))
               ).to(device).eval()
    net.load_state_dict({k[len("unet."):]: v for k, v in ckpt["state_dict"].items()
                         if k.startswith("unet.")})

    x, y = build_dataset(args.manifest, args.data_root, args.limit)
    x, y = x.to(device) * 2 - 1, y.to(device)
    abar = ckpt["state_dict"]["abar"].to(device)

    g = torch.Generator().manual_seed(0)
    eps = torch.randn(x.shape, generator=g).to(device)
    rows = []
    with torch.no_grad():
        for t in [int(v) for v in args.times.split(",")]:
            a = abar[t]
            xt = a.sqrt() * x + (1 - a).sqrt() * eps
            tt = torch.full((len(y),), t, device=device, dtype=torch.long)
            _, m_true, _ = net._run(xt, tt, y)
            _, m_zero, _ = net._run(xt, tt, torch.zeros_like(y))
            _, m_flip, _ = net._run(xt, tt, 1 - y)
            feats = {"y_true": m_true, "y_zero": m_zero, "y_flip": m_flip}
            rec = {"t": t, "abar": float(a)}
            for name, m in feats.items():
                logits = net.aux_head(m)
                rec[f"ce_{name}"] = float(F.cross_entropy(logits, y))
                rec[f"acc_{name}"] = float((logits.argmax(1) == y).float().mean())
            shift = (m_true - m_flip).flatten(1).norm(dim=1)
            scale = m_true.flatten(1).norm(dim=1).clamp_min(1e-8)
            rec["label_shift_rel"] = float((shift / scale).mean())
            rec["m_absmean"] = float(m_true.abs().mean())
            rows.append(rec)
            print(f"t={t:4d}  CE true {rec['ce_y_true']:.4f}  zero {rec['ce_y_zero']:.4f}  "
                  f"flip {rec['ce_y_flip']:.4f}   acc true {rec['acc_y_true']:.3f} "
                  f"zero {rec['acc_y_zero']:.3f}   |dm|/|m| {rec['label_shift_rel']:.3f}")

    verdict = ("reads conditioning: CE(y_zero) ~ ln2, so the head never has to look at pixels"
               if sum(r["ce_y_zero"] for r in rows) / len(rows) > 0.5 else
               "reads pixels: CE(y_zero) is low too, so the CE is a discriminative read on x_t")
    out = {"weights": args.weights, "md5_note": "see results/ab/eval_*", "limit": int(len(y)),
           "rows": rows, "verdict": verdict}
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nverdict: {verdict}\nwrote {args.out}")


if __name__ == "__main__":
    main()
