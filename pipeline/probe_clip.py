"""Exploratory probe: CLIP image-level scores on a random pool sample.

Not part of the final pipeline; used to calibrate prompts and thresholds before
committing to a full 76k scan.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

import open_clip  # noqa: E402

PROMPTS = {
    "cat": [
        "a photo of a cat", "a photo of a cat's head", "a close-up photo of a cat",
        "a cat looking at the camera", "a photo of a kitten",
    ],
    "dog": [
        "a photo of a dog", "a photo of a dog's head", "a close-up photo of a dog",
        "a dog looking at the camera", "a photo of a puppy",
    ],
    "other_animal": [
        "a photo of another animal", "a photo of a bird", "a photo of a horse",
        "a photo of a wild animal",
    ],
    "cartoon": [
        "a cartoon of a cat or dog", "a drawing of a cat or dog",
        "a painting of a cat or dog", "an illustration of a cat or dog",
        "a 3d render of a cat or dog", "a toy cat or dog", "a statue of a cat or dog",
    ],
    "product": [
        "a product photo", "a cat or dog printed on a shirt", "a cat or dog on a mug",
        "a cat or dog logo", "a screenshot", "a meme", "text",
    ],
    "no_animal": [
        "a photo of a person", "a photo of a landscape", "a photo of a building",
        "a photo of food", "a photo of furniture", "a photo of a car",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/probe_clip")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    names, texts = [], []
    for group, prompts in PROMPTS.items():
        for p in prompts:
            names.append(group)
            texts.append(p)
    with torch.no_grad():
        text = model.encode_text(tokenizer(texts).to(device))
        text /= text.norm(dim=-1, keepdim=True)

    pool = Pool(pool_dir("data"))
    ids = list(pool.index)
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.n]

    rows = []
    thumbs = []
    batch_size = 64
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        imgs, kept = [], []
        for pid in chunk:
            try:
                im = pool.image(pid).convert("RGB")
            except Exception:
                continue
            imgs.append(preprocess(im))
            kept.append(pid)
        if not imgs:
            continue
        with torch.no_grad():
            x = torch.stack(imgs).to(device)
            feat = model.encode_image(x)
            feat /= feat.norm(dim=-1, keepdim=True)
            sim = (100.0 * feat @ text.T).softmax(dim=-1).cpu()
        for pid, s in zip(kept, sim):
            scores = {}
            for group in PROMPTS:
                idx = [i for i, n in enumerate(names) if n == group]
                scores[group] = float(s[idx].sum())
            rows.append({"pool_id": pid, **scores})

    rows.sort(key=lambda r: max(r["cat"], r["dog"]), reverse=True)
    with open(out / "scores.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Montage: top 40 by cat+dog score, with scores printed.
    cell = 150
    cols = 8
    sel = rows[:40]
    canvas = Image.new("RGB", (cols * cell, ((len(sel) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, r in enumerate(sel):
        im = pool.image(r["pool_id"]).convert("RGB")
        im.thumbnail((cell - 4, cell - 30))
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im, (x + 2, y + 2))
        draw.text((x + 2, y + cell - 28),
                  f"{r['pool_id']} c{r['cat']:.2f} d{r['dog']:.2f}", fill=(0, 0, 0))
        draw.text((x + 2, y + cell - 14),
                  f"art{r['cartoon']:.2f} pr{r['product']:.2f}", fill=(80, 80, 80))
    canvas.save(out / "top_catdog.png")
    print(f"scanned {len(rows)} -> {out}/scores.json, {out}/top_catdog.png")


if __name__ == "__main__":
    main()
