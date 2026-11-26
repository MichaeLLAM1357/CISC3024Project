"""Probe: run the face localizer on dev images (upscaled) to calibrate head expansion."""
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

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
    dog = load("hugocornellier/dog-face-landmarks", "dog_face_localizer.tflite")
    images, labels = load_dev("data")
    idx = list(range(0, 10)) + list(torch.where(labels == 1)[0][:10].tolist())
    cell = 200
    cols = 5
    canvas = Image.new("RGB", (cols * cell, ((len(idx) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, i in enumerate(idx):
        im = Image.fromarray((images[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        im = im.resize((224, 224), Image.Resampling.LANCZOS)
        cb = predict(*cat, im)
        db = predict(*dog, im)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im.resize((cell - 4, cell - 4)), (x + 2, y + 2))
        for b, color in ((cb, (0, 200, 0)), (db, (0, 80, 255))):
            draw.rectangle([x + 2 + b[0], y + 2 + b[1], x + 2 + b[2], y + 2 + b[3]],
                           outline=color, width=2)
        draw.text((x + 2, y + cell - 14), f"#{i} y{int(labels[i])} "
                  f"cat {cb[2]-cb[0]:.0f}x{cb[3]-cb[1]:.0f}", fill=(0, 0, 0))
    canvas.save("results/probe_face/dev_faces.png")
    print("saved results/probe_face/dev_faces.png")


if __name__ == "__main__":
    main()
