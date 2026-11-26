"""Probe: CLIP style/head scores on the face-localizer crops."""
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

import open_clip  # noqa: E402

PROMPTS = {
    "cat_head": ["a close-up photo of a cat's head", "a photo of a cat's face",
                 "a photo of a cat's head with ears", "a portrait of a cat"],
    "dog_head": ["a close-up photo of a dog's head", "a photo of a dog's face",
                 "a photo of a dog's head with ears", "a portrait of a dog"],
    "cat_body": ["a photo of a cat's body", "a photo of a cat's tail",
                 "a photo of a cat in the distance", "a photo of a cat lying down"],
    "dog_body": ["a photo of a dog's body", "a photo of a dog's tail",
                 "a photo of a dog in the distance", "a photo of a dog lying down"],
    "photo": ["a photograph of a real animal", "a real photo of a pet",
              "a close-up photograph of an animal"],
    "art": ["a drawing of an animal", "a painting of an animal", "a cartoon animal",
            "an illustration of an animal", "a 3D render of an animal",
            "a sculpture of an animal", "a logo", "an animal printed on a product",
            "a product photo", "a screenshot", "a meme", "text"],
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-L-14")
    names, texts = [], []
    for g, ps in PROMPTS.items():
        for p in ps:
            names.append(g); texts.append(p)
    with torch.no_grad():
        t = model.encode_text(tok(texts).to(device)); t /= t.norm(dim=-1, keepdim=True)

    recs = json.load(open("results/probe_headcrop2/headcrops.json"))
    pool = Pool(pool_dir("data"))
    crops = []
    for r in recs:
        im = pool.image(r["pool_id"]).convert("RGB")
        x, y, s = r["crop"]
        crops.append(im.crop((x, y, x + s, y + s)))
    with torch.no_grad():
        feats = []
        for i in range(0, len(crops), 64):
            x = torch.stack([preprocess(c) for c in crops[i:i + 64]]).to(device)
            f = model.encode_image(x); feats.append(f / f.norm(dim=-1, keepdim=True))
        sim = (100.0 * torch.cat(feats) @ t.T).softmax(-1).cpu()
    for r, s in zip(recs, sim):
        sc = {g: float(s[[i for i, n in enumerate(names) if n == g]].sum()) for g in PROMPTS}
        r.update(sc)
        r["style"] = sc["photo"] / (sc["photo"] + sc["art"] + 1e-9)
        r["head"] = sc["cat_head"] if r["cls"] == 15 else sc["dog_head"]
    json.dump(recs, open("results/probe_headcrop2/clip_scores.json", "w"), indent=2)

    for r in sorted(recs, key=lambda r: r["style"]):
        print(f"{r['pool_id']} cls={r['cls']} style={r['style']:.3f} head={r['head']:.3f} "
              f"art={r['art']:.3f} photo={r['photo']:.3f}")

    for tag, sel in (("low_style", sorted(recs, key=lambda r: r["style"])[:24]),
                     ("high_style", sorted(recs, key=lambda r: r["style"], reverse=True)[:24])):
        cell = 180; cols = 8
        canvas = Image.new("RGB", (cols * cell, ((len(sel) + cols - 1) // cols) * cell), "white")
        draw = ImageDraw.Draw(canvas)
        for k, r in enumerate(sel):
            x, y, s = r["crop"]
            im = pool.image(r["pool_id"]).convert("RGB").crop((x, y, x + s, y + s))
            im = im.resize((96, 96), Image.Resampling.LANCZOS)
            cx, cy = (k % cols) * cell, (k // cols) * cell
            canvas.paste(im, (cx + 2, cy + 2))
            draw.text((cx + 2, cy + 100), r["pool_id"], fill=(0, 0, 0))
            draw.text((cx + 2, cy + 114), f"style {r['style']:.2f}", fill=(0, 0, 0))
            draw.text((cx + 2, cy + 128), f"head {r['head']:.2f}", fill=(0, 0, 0))
            draw.text((cx + 2, cy + 142), f"cls {r['cls']}", fill=(0, 0, 0))
        canvas.save(f"results/probe_headcrop2/{tag}.png")
    print("-> results/probe_headcrop2/{low,high}_style.png")


if __name__ == "__main__":
    main()
