"""Visual sanity check for the fixed face-box -> head-crop scale.

This is NOT a dev-fitted calibration.  It runs the class-specific face localizer on a
few pool candidates and lays out several fixed crop scales side by side so a human can
pick the scale that puts head + ears at roughly 60-80% of the square.

Usage:
    python pipeline/probe_crop_scale.py --scan results/scan/yolo_detections.jsonl --n 24
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

SCALES = (1.4, 1.6, 1.8, 2.0)
CONTEXT = 0.15


def load_localizer(repo, fn):
    interp = Interpreter(model_path=hf_hub_download(repo, fn))
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


def predict(interp, inp, outp, im):
    size = 224
    w, h = im.size
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.Resampling.BILINEAR),
                 ((size - nw) // 2, (size - nh) // 2))
    x = np.asarray(canvas, dtype=np.float32)[None] / 255.0
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    b = interp.get_tensor(outp["index"])[0]
    x1, x2 = sorted((float(b[0]), float(b[2])))
    y1, y2 = sorted((float(b[1]), float(b[3])))
    px, py = (size - nw) // 2, (size - nh) // 2
    return [(x1 * size - px) / s, (y1 * size - py) / s,
            (x2 * size - px) / s, (y2 * size - py) / s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="results/scan/yolo_detections.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="results/probe_crop_scale")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cat = load_localizer("hugocornellier/cat-face-landmarks", "cat_face_localizer.tflite")
    dog = load_localizer("hugocornellier/dog-face-landmarks", "dog_face_localizer.tflite")
    models = {0: cat, 1: dog}
    pool = Pool(pool_dir("data"))

    records = []
    with open(args.scan, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            dets = rec.get("detections", [])
            if len(dets) == 1:
                records.append(rec)
                if len(records) >= args.n:
                    break
    print(f"using {len(records)} records with exactly one detection")

    cell = 180
    cols = 1 + len(SCALES)
    canvas = Image.new("RGB", (cols * cell, len(records) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    ratios = []
    for k, rec in enumerate(records):
        im = pool.image(rec["pool_id"]).convert("RGB")
        w, h = im.size
        det = rec["detections"][0]
        cls = 0 if det["cls"] == 15 else 1
        x1, y1, x2, y2 = det["xyxy"]
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - CONTEXT * bw))
        cy1 = max(0, int(y1 - CONTEXT * bh))
        cx2 = min(w, int(x2 + CONTEXT * bw))
        cy2 = min(h, int(y2 + CONTEXT * bh))
        sub = im.crop((cx1, cy1, cx2, cy2))
        fb = predict(*models[cls], sub)
        fb = [fb[0] + cx1, fb[1] + cy1, fb[2] + cx1, fb[3] + cy1]
        fw, fh = fb[2] - fb[0], fb[3] - fb[1]
        m = max(fw, fh)
        ratios.append(fh / fw if fw > 0 else 0.0)

        y = k * cell
        # original with face box
        scale = min((cell - 4) / im.width, (cell - 24) / im.height)
        im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.Resampling.BILINEAR)
        canvas.paste(im2, (2, y + 2))
        draw.rectangle([2 + fb[0] * scale, y + 2 + fb[1] * scale,
                        2 + fb[2] * scale, y + 2 + fb[3] * scale],
                       outline=(255, 0, 0), width=2)
        draw.text((2, y + cell - 14), f"{rec['pool_id']} {'cat' if cls == 0 else 'dog'}",
                  fill=(0, 0, 0))
        draw.text((2, y + cell - 28), f"face {fw:.0f}x{fh:.0f}", fill=(0, 0, 0))

        for j, s in enumerate(SCALES):
            side = min(s * m, min(w, h))
            ccx, ccy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            x0 = max(0, min(int(round(ccx - side / 2)), w - int(side)))
            y0 = max(0, min(int(round(ccy - side / 2)), h - int(side)))
            crop = im.crop((x0, y0, x0 + int(side), y0 + int(side)))
            crop = crop.resize((cell - 4, cell - 4), Image.Resampling.LANCZOS)
            canvas.paste(crop, ((j + 1) * cell + 2, y + 2))
            draw.text(((j + 1) * cell + 2, y + cell - 14), f"K={s:.1f}", fill=(0, 0, 0))

    canvas.save(out / "scale_grid.png")
    r = np.array(ratios)
    print(f"face h/w: mean={r.mean():.3f} p50={np.percentile(r,50):.3f} "
          f"p10={np.percentile(r,10):.3f} p90={np.percentile(r,90):.3f}")
    print(f"-> {out}/scale_grid.png")


if __name__ == "__main__":
    main()
