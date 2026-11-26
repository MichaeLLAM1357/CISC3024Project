"""Probe: ViT attention rollout of the cat/dog head classifier as a head localizer."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402


def rollout(attentions):
    # attentions: tuple of [B, heads, tokens, tokens]
    result = torch.eye(attentions[0].shape[-1], device=attentions[0].device)
    for attn in attentions:
        a = attn.mean(dim=1)[0]  # average heads
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="results/probe_attention")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "gsyllas/vit-cats-dogs-head"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(
        name, attn_implementation="eager").to(device).eval()

    pool = Pool(pool_dir("data"))
    recs = [r for r in json.load(open("results/probe_yolo/detections.json"))
            if r["detections"]][:args.n]
    cell = 260
    cols = 4
    canvas = Image.new("RGB", (cols * cell, ((len(recs) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, rec in enumerate(recs):
        im = pool.image(rec["pool_id"]).convert("RGB")
        inp = proc(images=im, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inp, output_attentions=True)
        r = rollout(o.attentions)[0, 1:]  # CLS -> patches
        n = int(r.shape[0] ** 0.5)
        heat = r.reshape(n, n).float().cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        heat_im = Image.fromarray((heat * 255).astype(np.uint8)).resize(im.size, Image.Resampling.BICUBIC)
        # overlay red heat
        arr = np.asarray(im).astype(np.float32)
        h = np.asarray(heat_im).astype(np.float32) / 255.0
        arr[..., 0] = np.clip(arr[..., 0] + 120 * h, 0, 255)
        arr[..., 1:] = np.clip(arr[..., 1:] * (1 - 0.4 * h[..., None]), 0, 255)
        over = Image.fromarray(arr.astype(np.uint8))
        scale = min((cell - 4) / im.width, (cell - 24) / im.height)
        over = over.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                           Image.Resampling.BILINEAR)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(over, (x + 2, y + 2))
        draw.text((x + 2, y + cell - 14), rec["pool_id"], fill=(0, 0, 0))
    canvas.save(out / "attention.png")
    print(f"-> {out}/attention.png")


if __name__ == "__main__":
    main()
