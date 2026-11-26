"""Probe: localize heads by sliding a pretrained cat/dog head classifier over YOLO boxes."""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

AX = (-0.35, -0.2, -0.05, 0.1, 0.25, 0.4)
AY = (0.10, 0.22, 0.34, 0.46, 0.58, 0.70)
SCALES = (0.30, 0.40, 0.50, 0.65, 0.80, 1.0)


def square_crop(im, cx, cy, side):
    w, h = im.size
    side = int(round(side))
    if side < 40:
        return None
    side = min(side, min(w, h))
    x0 = max(0, min(int(round(cx - side / 2)), w - side))
    y0 = max(0, min(int(round(cy - side / 2)), h - side))
    return x0, y0, side


def candidates(box, im):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    out = []
    for ax in AX:
        for ay in AY:
            for s in SCALES:
                c = square_crop(im, x1 + ax * w, y1 + ay * h, s * max(w, h))
                if c is not None:
                    out.append(c)
    c = square_crop(im, (x1 + x2) / 2, (y1 + y2) / 2, max(w, h))
    if c is not None:
        out.append(c)
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="results/probe_yolo/detections.json")
    ap.add_argument("--out", default="results/probe_headcrop")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--top", type=int, default=48)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "gsyllas/vit-cats-dogs-head"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()

    records = [r for r in json.load(open(args.detections)) if r["detections"]][:args.limit]
    pool = Pool(pool_dir("data"))
    results = []
    for rec in records:
        pid = rec["pool_id"]
        im = pool.image(pid).convert("RGB")
        for det in rec["detections"]:
            cls = 0 if det["cls"] == 15 else 1
            boxes = candidates(det["xyxy"], im)
            if not boxes:
                continue
            crops = [im.crop((x, y, x + s, y + s)) for x, y, s in boxes]
            probs = []
            for i in range(0, len(crops), 128):
                inp = proc(images=crops[i:i + 128], return_tensors="pt").to(device)
                with torch.no_grad():
                    probs.append(model(**inp).logits.softmax(-1).cpu())
            probs = torch.cat(probs)
            score = probs[:, cls]
            j = int(score.argmax())
            best = {"box": list(boxes[j]), "p": float(score[j]),
                    "p_other": float(probs[j, 1 - cls]), "cls": int(cls)}
            results.append({"pool_id": pid, "det_conf": det["conf"], "best": best})
            print(f"{pid} cls={cls} det={det['conf']:.2f} p={best['p']:.3f} "
                  f"box={best['box']}")

    with open(out / "headcrops.json", "w") as f:
        json.dump(results, f, indent=2)

    results.sort(key=lambda r: r["best"]["p"], reverse=True)
    sel = results[:args.top]
    cell = 200
    cols = 8
    canvas = Image.new("RGB", (cols * cell, ((len(sel) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, rec in enumerate(sel):
        im = pool.image(rec["pool_id"]).convert("RGB")
        x, y, s = rec["best"]["box"]
        crop = im.crop((x, y, x + s, y + s)).resize((96, 96), Image.Resampling.LANCZOS)
        scale = min((cell - 4) / im.width, (cell - 20) / im.height)
        im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.Resampling.BILINEAR)
        cx, cy = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im2, (cx + 2, cy + 2))
        draw.rectangle([cx + 2 + x * scale, cy + 2 + y * scale,
                        cx + 2 + (x + s) * scale, cy + 2 + (y + s) * scale],
                       outline=(255, 0, 0), width=2)
        canvas.paste(crop, (cx + 2, cy + cell - 100))
        draw.text((cx + 102, cy + cell - 98),
                  f"{rec['pool_id']} {'cat' if rec['best']['cls'] == 0 else 'dog'}",
                  fill=(0, 0, 0))
        draw.text((cx + 102, cy + cell - 84), f"p{rec['best']['p']:.3f}", fill=(0, 0, 0))
    canvas.save(out / "best_headcrops.png")
    print(f"-> {out}/headcrops.json, {out}/best_headcrops.png")


if __name__ == "__main__":
    main()
