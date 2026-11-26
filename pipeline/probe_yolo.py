"""Exploratory probe: YOLO cat/dog detections on a random pool sample."""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

COCO_CAT, COCO_DOG = 15, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/probe_yolo")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    pool = Pool(pool_dir("data"))
    ids = list(pool.index)
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.n]

    records = []
    batch = 32
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        imgs = [pool.image(pid).convert("RGB") for pid in chunk]
        results = model.predict(imgs, imgsz=args.imgsz, conf=args.conf, verbose=False,
                                device=device, classes=[COCO_CAT, COCO_DOG])
        for pid, r in zip(chunk, results):
            dets = []
            for box in r.boxes:
                dets.append({
                    "cls": int(box.cls.item()),
                    "conf": float(box.conf.item()),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                })
            records.append({"pool_id": pid, "detections": dets})

    with open(out / "detections.json", "w") as f:
        json.dump(records, f, indent=2)

    # Montage of images with the highest-confidence detection.
    def best(r):
        return max((d["conf"] for d in r["detections"]), default=0.0)

    records.sort(key=best, reverse=True)
    sel = [r for r in records if r["detections"]][:48]
    cell = 180
    cols = 8
    canvas = Image.new("RGB", (cols * cell, ((len(sel) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, rec in enumerate(sel):
        im = pool.image(rec["pool_id"]).convert("RGB")
        scale = min((cell - 4) / im.width, (cell - 20) / im.height)
        im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.Resampling.BILINEAR)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im2, (x + 2, y + 2))
        for d in rec["detections"]:
            x1, y1, x2, y2 = [v * scale for v in d["xyxy"]]
            color = (0, 200, 0) if d["cls"] == COCO_CAT else (0, 80, 255)
            draw.rectangle([x + 2 + x1, y + 2 + y1, x + 2 + x2, y + 2 + y2], outline=color, width=2)
            draw.text((x + 2 + x1, y + 2 + y1), f"{d['conf']:.2f}", fill=color)
        draw.text((x + 2, y + cell - 14), rec["pool_id"], fill=(0, 0, 0))
    canvas.save(out / "top_detections.png")

    n_any = sum(bool(r["detections"]) for r in records)
    n_cat = sum(any(d["cls"] == COCO_CAT for d in r["detections"]) for r in records)
    n_dog = sum(any(d["cls"] == COCO_DOG for d in r["detections"]) for r in records)
    print(f"{len(records)} images: any={n_any} ({n_any/len(records):.1%}), "
          f"cat={n_cat}, dog={n_dog}, both={sum(any(d['cls']==15 for d in r['detections']) and any(d['cls']==16 for d in r['detections']) for r in records)}")
    print(f"-> {out}/detections.json, {out}/top_detections.png")


if __name__ == "__main__":
    main()
