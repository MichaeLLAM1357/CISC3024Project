"""Probe: YOLO box -> face localizer -> calibrated head crop."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

# Calibrated on the dev set (aggregate only): crop side = K * max(face_w, face_h),
# crop centre = face centre + (OX, OY) * max(face_w, face_h).
K, OX, OY = 1.81, -0.01, 0.10
CONTEXT = 0.15  # extra margin around the YOLO box before running the localizer


def load(repo, fn):
    i = Interpreter(model_path=hf_hub_download(repo, fn)); i.allocate_tensors()
    return i, i.get_input_details()[0], i.get_output_details()[0]


def predict(interp, inp, outp, im):
    size = 224
    w, h = im.size
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    c = Image.new("RGB", (size, size), (114, 114, 114))
    c.paste(im.resize((nw, nh), Image.Resampling.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    x = np.asarray(c, dtype=np.float32)[None] / 255.0
    interp.set_tensor(inp["index"], x); interp.invoke()
    b = interp.get_tensor(outp["index"])[0]
    x1, x2 = sorted((float(b[0]), float(b[2])))
    y1, y2 = sorted((float(b[1]), float(b[3])))
    px, py = (size - nw) // 2, (size - nh) // 2
    return [(x1 * size - px) / s, (y1 * size - py) / s,
            (x2 * size - px) / s, (y2 * size - py) / s]


def clamp_box(b, w, h):
    return [max(0.0, min(w, b[0])), max(0.0, min(h, b[1])),
            max(0.0, min(w, b[2])), max(0.0, min(h, b[3]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="results/probe_yolo/detections.json")
    ap.add_argument("--out", default="results/probe_headcrop2")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--top", type=int, default=48)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cat = load("hugocornellier/cat-face-landmarks", "cat_face_localizer.tflite")
    pool = Pool(pool_dir("data"))
    records = [r for r in json.load(open(args.detections)) if r["detections"]][:args.limit]

    results = []
    for rec in records:
        pid = rec["pool_id"]
        im = pool.image(pid).convert("RGB")
        w, h = im.size
        # Reject images with more than one cat/dog detection: too likely to contain two heads.
        if len(rec["detections"]) != 1:
            continue
        det = rec["detections"][0]
        x1, y1, x2, y2 = det["xyxy"]
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - CONTEXT * bw)); cy1 = max(0, int(y1 - CONTEXT * bh))
        cx2 = min(w, int(x2 + CONTEXT * bw)); cy2 = min(h, int(y2 + CONTEXT * bh))
        sub = im.crop((cx1, cy1, cx2, cy2))
        fb = predict(*cat, sub)
        # map face box back to the original image
        fb = [fb[0] + cx1, fb[1] + cy1, fb[2] + cx1, fb[3] + cy1]
        fw, fh = fb[2] - fb[0], fb[3] - fb[1]
        m = max(fw, fh)
        if m < 16:
            continue
        fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
        side = K * m
        ccx, ccy = fcx + OX * m, fcy + OY * m
        # square crop, shifted inside the image if necessary
        side = min(side, min(w, h))
        x0 = int(round(ccx - side / 2)); y0 = int(round(ccy - side / 2))
        x0 = max(0, min(x0, w - int(side))); y0 = max(0, min(y0, h - int(side)))
        crop = im.crop((x0, y0, x0 + int(side), y0 + int(side)))
        # sharpness: variance of Laplacian on the crop
        g = np.asarray(crop.convert("L"), dtype=np.float32)
        lap = np.asarray(crop.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        sharp = float(lap.var())
        results.append({"pool_id": pid, "cls": int(det["cls"]), "det_conf": det["conf"],
                        "face": fb, "crop": [x0, y0, int(side)], "face_size": m,
                        "sharp": sharp})
        print(f"{pid} cls={det['cls']} face={[round(v) for v in fb]} m={m:.0f} "
              f"crop={x0},{y0},{int(side)} sharp={sharp:.0f}")

    json.dump(results, open(out / "headcrops.json", "w"), indent=2)
    results.sort(key=lambda r: r["sharp"], reverse=True)
    sel = results[:args.top]
    cell = 200; cols = 8
    canvas = Image.new("RGB", (cols * cell, ((len(sel) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, rec in enumerate(sel):
        im = pool.image(rec["pool_id"]).convert("RGB")
        x0, y0, s = rec["crop"]
        crop = im.crop((x0, y0, x0 + s, y0 + s)).resize((96, 96), Image.Resampling.LANCZOS)
        scale = min((cell - 4) / im.width, (cell - 20) / im.height)
        im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.Resampling.BILINEAR)
        cx, cy = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im2, (cx + 2, cy + 2))
        draw.rectangle([cx + 2 + x0 * scale, cy + 2 + y0 * scale,
                        cx + 2 + (x0 + s) * scale, cy + 2 + (y0 + s) * scale],
                       outline=(255, 0, 0), width=2)
        canvas.paste(crop, (cx + 2, cy + cell - 100))
        draw.text((cx + 102, cy + cell - 98), rec["pool_id"], fill=(0, 0, 0))
        draw.text((cx + 102, cy + cell - 84), f"m{rec['face_size']:.0f}", fill=(0, 0, 0))
        draw.text((cx + 102, cy + cell - 70), f"s{rec['sharp']:.0f}", fill=(0, 0, 0))
    canvas.save(out / "headcrops.png")
    print(f"{len(results)} crops -> {out}/headcrops.json, {out}/headcrops.png")


if __name__ == "__main__":
    main()
