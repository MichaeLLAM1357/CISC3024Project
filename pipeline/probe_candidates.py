"""Exploratory probe: generate head-crop candidates from YOLO boxes and rank them with CLIP.

Used to calibrate candidate geometry and prompts before the full pool scan.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

import open_clip  # noqa: E402

PROMPTS = {
    "cat_head": [
        "a close-up photo of a cat's head", "a photo of a cat's face",
        "a photo of a cat's head with ears", "a portrait of a cat",
    ],
    "cat_body": [
        "a photo of a cat's body", "a photo of a cat's tail", "a photo of a cat's paws",
        "a photo of a cat in the distance", "a photo of a cat lying down",
        "a photo of a cat's back",
    ],
    "dog_head": [
        "a close-up photo of a dog's head", "a photo of a dog's face",
        "a photo of a dog's head with ears", "a portrait of a dog",
    ],
    "dog_body": [
        "a photo of a dog's body", "a photo of a dog's tail", "a photo of a dog's paws",
        "a photo of a dog in the distance", "a photo of a dog lying down",
        "a photo of a dog's back",
    ],
    "photo": [
        "a photograph of a real animal", "a real photo", "a close-up photograph of a pet",
    ],
    "art": [
        "a drawing of an animal", "a painting of an animal", "a cartoon animal",
        "an illustration of an animal", "a 3D render of an animal",
        "a sculpture of an animal", "a logo", "an animal printed on a product",
        "a product photo", "a screenshot", "a meme", "text",
    ],
}

# Candidate geometry: (center dx in box widths, center dy in box heights, side / max(w,h))
AX = (-0.12, 0.0, 0.12)
AY = (0.12, 0.25, 0.38, 0.52, 0.66)
SCALES = (0.35, 0.5, 0.65, 0.8, 1.0)


def square_crop(im, cx, cy, side):
    w, h = im.size
    side = int(round(side))
    if side < 40:
        return None
    side = min(side, min(w, h))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, w - side))
    y0 = max(0, min(y0, h - side))
    return x0, y0, side


def candidates(box, im):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    out = []
    for ax in AX:
        for ay in AY:
            for s in SCALES:
                side = s * max(w, h)
                c = square_crop(im, cx + ax * w, y1 + ay * h, side)
                if c is not None:
                    out.append(c)
    # whole box as a square (for portrait detections)
    side = max(w, h)
    c = square_crop(im, cx, (y1 + y2) / 2, side)
    if c is not None:
        out.append(c)
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="results/probe_yolo/detections.json")
    ap.add_argument("--out", default="results/probe_candidates")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--top", type=int, default=48)
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

    records = json.load(open(args.detections))
    records = [r for r in records if r["detections"]][:args.limit]
    pool = Pool(pool_dir("data"))

    results = []
    for rec in records:
        pid = rec["pool_id"]
        im = pool.image(pid).convert("RGB")
        for det in rec["detections"]:
            cls = "cat" if det["cls"] == 15 else "dog"
            boxes = candidates(det["xyxy"], im)
            if not boxes:
                continue
            crops = [preprocess(im.crop((x, y, x + s, y + s))) for x, y, s in boxes]
            with torch.no_grad():
                feats = []
                for i in range(0, len(crops), 128):
                    x = torch.stack(crops[i:i + 128]).to(device)
                    f = model.encode_image(x)
                    feats.append(f / f.norm(dim=-1, keepdim=True))
                feat = torch.cat(feats)
                sim = (100.0 * feat @ text.T).softmax(dim=-1).cpu()
            best = None
            for j, (x, y, s) in enumerate(boxes):
                sc = {g: float(sim[j][[i for i, n in enumerate(names) if n == g]].sum())
                      for g in PROMPTS}
                head = sc[f"{cls}_head"]
                body = sc[f"{cls}_body"]
                style = sc["photo"] / (sc["photo"] + sc["art"] + 1e-9)
                score = head * style
                cand = {"box": [x, y, s], "head": head, "body": body,
                        "style": style, "score": score, **sc}
                if best is None or cand["score"] > best["score"]:
                    best = cand
            results.append({"pool_id": pid, "cls": cls, "det_conf": det["conf"],
                            "best": best})
            print(f"{pid} {cls} det={det['conf']:.2f} head={best['head']:.3f} "
                  f"body={best['body']:.3f} style={best['style']:.3f} box={best['box']}")

    with open(out / "candidates.json", "w") as f:
        json.dump(results, f, indent=2)

    # Montage of the best candidates, sorted by score.
    results.sort(key=lambda r: r["best"]["score"], reverse=True)
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
                  f"{rec['pool_id']} {rec['cls']}", fill=(0, 0, 0))
        draw.text((cx + 102, cy + cell - 84),
                  f"h{rec['best']['head']:.2f} b{rec['best']['body']:.2f}", fill=(0, 0, 0))
        draw.text((cx + 102, cy + cell - 70),
                  f"s{rec['best']['style']:.2f}", fill=(0, 0, 0))
    canvas.save(out / "best_candidates.png")
    print(f"-> {out}/candidates.json, {out}/best_candidates.png")


if __name__ == "__main__":
    main()
