"""Probe Grounding DINO tiny for direct 'cat head' / 'dog head' detection."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/probe_gdino")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    pool = Pool(pool_dir("data"))
    recs = [r for r in json.load(open("results/probe_yolo/detections.json"))
            if r["detections"]][:args.n]

    t0 = time.perf_counter()
    results = []
    for rec in recs:
        im = pool.image(rec["pool_id"]).convert("RGB")
        inputs = processor(images=im, text="cat head. dog head.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=args.box_threshold,
            text_threshold=args.text_threshold, target_sizes=[im.size[::-1]])[0]
        dets = []
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            dets.append({"box": [float(v) for v in box.tolist()], "score": float(score),
                         "label": label})
        results.append({"pool_id": rec["pool_id"], "dets": dets})
        print(rec["pool_id"], [(d["label"], round(d["score"], 3),
                                [round(v) for v in d["box"]]) for d in dets])
    dt = time.perf_counter() - t0
    print(f"{len(recs)} images in {dt:.1f}s = {dt/len(recs):.2f}s/image")
    json.dump(results, open(out / "gdino.json", "w"), indent=2)

    cell = 220
    cols = 5
    canvas = Image.new("RGB", (cols * cell, ((len(results) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, rec in enumerate(results):
        im = pool.image(rec["pool_id"]).convert("RGB")
        scale = min((cell - 4) / im.width, (cell - 24) / im.height)
        im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.Resampling.BILINEAR)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im2, (x + 2, y + 2))
        for d in rec["dets"]:
            x1, y1, x2, y2 = [v * scale for v in d["box"]]
            color = (0, 200, 0) if "cat" in d["label"] else (0, 80, 255)
            draw.rectangle([x + 2 + x1, y + 2 + y1, x + 2 + x2, y + 2 + y2], outline=color, width=2)
            draw.text((x + 2 + x1, y + 2 + y1), f"{d['label']} {d['score']:.2f}", fill=color)
        draw.text((x + 2, y + cell - 14), rec["pool_id"], fill=(0, 0, 0))
    canvas.save(out / "gdino_boxes.png")
    print(f"-> {out}/gdino.json, {out}/gdino_boxes.png")


if __name__ == "__main__":
    main()
