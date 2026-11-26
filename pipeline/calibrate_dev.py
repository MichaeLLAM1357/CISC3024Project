"""Calibrate the face-box -> head-crop expansion on the dev set (aggregate statistics only)."""
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import load_dev  # noqa: E402

from ai_edge_litert.interpreter import Interpreter  # noqa: E402


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


def main():
    cat = load("hugocornellier/cat-face-landmarks", "cat_face_localizer.tflite")
    images, labels = load_dev("data")
    ks, ox, oy, ars = [], [], [], []
    for i in range(len(images)):
        im = Image.fromarray((images[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        im = im.resize((224, 224), Image.Resampling.LANCZOS)
        b = predict(*cat, im)
        fw, fh = b[2] - b[0], b[3] - b[1]
        if fw <= 0 or fh <= 0:
            continue
        m = max(fw, fh)
        ks.append(224 / m)
        ox.append((112 - (b[0] + b[2]) / 2) / m)
        oy.append((112 - (b[1] + b[3]) / 2) / m)
        ars.append(fw / fh)
    for name, v in (("k", ks), ("offset_x", ox), ("offset_y", oy), ("aspect", ars)):
        v = np.array(v)
        print(f"{name}: n={len(v)} mean={v.mean():.3f} sd={v.std():.3f} "
              f"p10={np.percentile(v,10):.3f} p25={np.percentile(v,25):.3f} "
              f"p50={np.percentile(v,50):.3f} p75={np.percentile(v,75):.3f} "
              f"p90={np.percentile(v,90):.3f}")


if __name__ == "__main__":
    main()
