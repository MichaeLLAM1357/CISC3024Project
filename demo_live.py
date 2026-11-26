#!/usr/bin/env python3
"""LIVE DEMO for the CISC3024 cat/dog head generator.

    python demo_live.py --submission CISC3024_UC325629/submission --seed 1 --per-class 4 --open

Loads the graded model straight from a submission folder (only torch, numpy and pillow
required -- no starter kit, no Inception, no classifier.pt, no network), draws one row of
cats and one row of dogs for the requested seed, writes a PNG grid next to this script, and
optionally opens it in the default image viewer.

The images come from the exact same `load_model` / `generate` pair the evaluator calls, so
what you see here is what the marker measures.  `generate` is deterministic in (labels, seed)
and each image depends only on its own label, so the grid is reproducible and the two rows
are independent of their order.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_submission(submission_dir: Path):
    """Import generate.py from an arbitrary submission folder under a private module name."""
    gen_py = submission_dir / "generate.py"
    if not gen_py.is_file():
        raise SystemExit(f"generate.py not found under {submission_dir}")
    spec = importlib.util.spec_from_file_location("_live_demo_generate", gen_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_grid(images: torch.Tensor, per_class: int, tile: int, label_h: int, gap: int) -> Image.Image:
    """[2*per_class,3,32,32] float in [0,1] -> labelled PNG grid (cats top, dogs bottom)."""
    arr = (images.detach().cpu().numpy() * 255.0).round().astype(np.uint8)  # [N,3,32,32]
    n = arr.shape[0]
    w = per_class * tile + (per_class + 1) * gap
    h = 2 * (tile + label_h) + 3 * gap
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        row, col = divmod(i, per_class)
        x = gap + col * (tile + gap)
        y = gap + row * (label_h + tile + gap)
        draw.text((x + 2, y + 2), "CAT (label 0)" if row == 0 else "DOG (label 1)",
                  fill=(30, 30, 30))
        pil = Image.fromarray(arr[i].transpose(1, 2, 0), "RGB").resize(
            (tile, tile), Image.Resampling.NEAREST)
        canvas.paste(pil, (x, y + label_h))
    return canvas


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, default=here / "submission",
                    help="folder holding generate.py and weights.pt")
    ap.add_argument("--seed", type=int, default=0, help="generation seed (default: 0)")
    ap.add_argument("--per-class", type=int, default=4,
                    help="images per class; 2*per-class total (default: 4)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG path (default: demo_live_seed<seed>.png)")
    ap.add_argument("--tile", type=int, default=128, help="pixels per 32x32 tile (default: 128)")
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    ap.add_argument("--open", action="store_true", help="open the PNG in the default viewer")
    args = ap.parse_args()

    if args.per_class < 1:
        raise SystemExit("--per-class must be >= 1")

    cwd = Path.cwd()
    # Resolve --submission from wherever the script is run.  The documented command passes
    # "CISC3024_UC325629/submission"; that is correct from the project root but wrong if you
    # have `cd`'d into the unzipped folder, so try cwd, the script folder and its parent.
    candidates = []
    if args.submission.is_absolute():
        candidates.append(args.submission)
    else:
        candidates += [cwd / args.submission, here / args.submission,
                       here.parent / args.submission]
        if args.submission.name == "submission":           # also accept a bare "submission"
            candidates += [cwd / "submission", here / "submission"]
    sub = next((p for p in candidates if (p / "generate.py").is_file()), candidates[0])
    sub = sub.resolve()
    gen = load_submission(sub)

    device = torch.device(args.device)
    model = gen.load_model(device)                       # exactly the evaluator's entry point
    labels = torch.tensor([0] * args.per_class + [1] * args.per_class, dtype=torch.long)
    images = gen.generate(model, labels, args.seed)      # [2*per_class,3,32,32] in [0,1]
    if not torch.isfinite(images).all():
        raise SystemExit("generate() returned NaN/Inf")

    out = args.out or Path(f"demo_live_seed{args.seed}.png")
    out = out if out.is_absolute() else (cwd / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = build_grid(images, args.per_class, args.tile, label_h=14, gap=8)
    grid.save(out)
    print(f"seed={args.seed}  per_class={args.per_class}  images={len(labels)}")
    print(f"saved grid -> {out}")

    if args.open:
        try:
            if sys.platform.startswith("win"):
                import os
                os.startfile(str(out))                    # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", str(out)], check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", str(out)], check=False)
            print("opened in default image viewer")
        except Exception as exc:                          # headless machine: keep the PNG
            print(f"could not open viewer ({exc}); open {out} manually")


if __name__ == "__main__":
    main()
