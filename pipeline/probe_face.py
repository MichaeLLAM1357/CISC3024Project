"""Probe: cat/dog face localizer TFLite models on pool images."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

from ai_edge_litert.interpreter import Interpreter  # noqa: E402


def load_model(repo, filename):
    path = hf_hub_download(repo, filename)
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


def letterbox(im, size=224):
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im2, ((size - nw) // 2, (size - nh) // 2))
    return canvas, scale, (size - nw) // 2, (size - nh) // 2


def predict(interp, inp, outp, im):
    canvas, scale, px, py = letterbox(im)
    x = np.asarray(canvas, dtype=np.float32)[None] / 255.0
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    box = interp.get_tensor(outp["index"])[0]
    # The exported localizer returns the x coordinates in reverse order; sort them.
    x1, x2 = sorted((float(box[0]), float(box[2])))
    y1, y2 = sorted((float(box[1]), float(box[3])))
    # normalized xyxy in letterbox -> original image coords
    x1 = (x1 * 224 - px) / scale
    y1 = (y1 * 224 - py) / scale
    x2 = (x2 * 224 - px) / scale
    y2 = (y2 * 224 - py) / scale
    w, h = im.size
    x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
    y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
    return [float(v) for v in (x1, y1, x2, y2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="results/probe_face")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cat = load_model("hugocornellier/cat-face-landmarks", "cat_face_localizer.tflite")
    dog = load_model("hugocornellier/dog-face-landmarks", "dog_face_localizer.tflite")
    pool = Pool(pool_dir("data"))
    recs = [r for r in json.load(open("results/probe_yolo/detections.json"))
            if r["detections"]][:args.n]
    results = []
    t0 = time.perf_counter()
    for rec in recs:
        im = pool.image(rec["pool_id"]).convert("RGB")
        cb = predict(*cat, im)
        db = predict(*dog, im)
        results.append({"pool_id": rec["pool_id"], "cat": cb, "dog": db})
        print(rec["pool_id"], "cat", [round(v) for v in cb], "dog", [round(v) for v in db])
    dt = time.perf_counter() - t0
    print(f"{len(recs)} in {dt:.1f}s = {dt/len(recs):.3f}s/image")
    json.dump(results, open(out / "faces.json", "w"), indent=2)

    cell = 240
    cols = 4
    canvas = Image.new("RGB", (cols * cell, ((len(results) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, rec in enumerate(results):
        im = pool.image(rec["pool_id"]).convert("RGB")
        scale = min((cell - 4) / im.width, (cell - 24) / im.height)
        im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.Resampling.BILINEAR)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im2, (x + 2, y + 2))
        for key, color in (("cat", (0, 200, 0)), ("dog", (0, 80, 255))):
            x1, y1, x2, y2 = [v * scale for v in rec[key]]
            draw.rectangle([x + 2 + x1, y + 2 + y1, x + 2 + x2, y + 2 + y2], outline=color, width=2)
        draw.text((x + 2, y + cell - 14), rec["pool_id"], fill=(0, 0, 0))
    canvas.save(out / "faces.png")
    print(f"-> {out}/faces.json, {out}/faces.png")


if __name__ == "__main__":
    main()
