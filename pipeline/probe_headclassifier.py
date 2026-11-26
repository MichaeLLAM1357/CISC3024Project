"""Probe the pretrained cat/dog head classifier on dev images and pool crops."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, load_dev, pool_dir  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "gsyllas/vit-cats-dogs-head"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
    print(model.config.id2label)

    images, labels = load_dev("data")
    idx = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))[:100]
    probs = []
    for i in range(0, len(idx), 32):
        batch = images[idx[i:i + 32]]
        pil = [Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
               for x in batch]
        inp = proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            p = model(**inp).logits.softmax(-1).cpu()
        probs.append(p)
    p = torch.cat(probs)
    pred = p.argmax(1)
    print(f"dev top-1 {100*(pred == labels[idx]).float().mean():.1f}%")
    for c in (0, 1):
        m = labels[idx] == c
        print(f"  class {c}: n={int(m.sum())} acc={100*(pred[m] == c).float().mean():.1f}% "
              f"mean max prob={p[m].max(1).values.mean():.3f}")

    pool = Pool(pool_dir("data"))
    ids = list(pool.index)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ids), 60, replace=False)
    pils = [pool.image(ids[i]).convert("RGB") for i in sel]
    inp = proc(images=pils, return_tensors="pt").to(device)
    with torch.no_grad():
        q = model(**inp).logits.softmax(-1).cpu()
    print("random whole pool images: mean max prob", q.max(1).values.mean().item(),
          "frac >0.9", (q.max(1).values > 0.9).float().mean().item())


if __name__ == "__main__":
    main()
