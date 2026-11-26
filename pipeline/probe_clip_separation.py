"""Probe: can CLIP separate known-good dev heads from random pool crops?"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, load_dev, pool_dir  # noqa: E402

import open_clip  # noqa: E402

PROMPTS = {
    "cat_head": ["a close-up photo of a cat's head", "a photo of a cat's face",
                 "a photo of a cat's head with ears", "a portrait of a cat"],
    "dog_head": ["a close-up photo of a dog's head", "a photo of a dog's face",
                 "a photo of a dog's head with ears", "a portrait of a dog"],
    "cat_body": ["a photo of a cat's body", "a photo of a cat's tail", "a photo of a cat's paws",
                 "a photo of a cat in the distance", "a photo of a cat lying down"],
    "dog_body": ["a photo of a dog's body", "a photo of a dog's tail", "a photo of a dog's paws",
                 "a photo of a dog in the distance", "a photo of a dog lying down"],
    "photo": ["a photograph of a real animal", "a real photo"],
    "art": ["a drawing of an animal", "a painting of an animal", "a cartoon animal",
            "an illustration of an animal", "a 3D render of an animal", "a sculpture of an animal",
            "a logo", "an animal printed on a product", "a product photo", "a screenshot",
            "a meme", "text"],
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

    def score(pils):
        out = []
        for i in range(0, len(pils), 64):
            x = torch.stack([preprocess(im) for im in pils[i:i + 64]]).to(device)
            with torch.no_grad():
                f = model.encode_image(x); f /= f.norm(dim=-1, keepdim=True)
                s = (100.0 * f @ t.T).softmax(-1).cpu()
            for row in s:
                out.append({g: float(row[[j for j, n in enumerate(names) if n == g]].sum())
                            for g in PROMPTS})
        return out

    dev, labels = load_dev("data")
    dev_pils = [Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                for x in dev]
    ds = score(dev_pils)
    for c, g in ((0, "cat_head"), (1, "dog_head")):
        v = np.array([r[g] for r, y in zip(ds, labels.tolist()) if y == c])
        st = np.array([r["photo"] / (r["photo"] + r["art"] + 1e-9) for r, y in zip(ds, labels.tolist()) if y == c])
        print(f"dev class {c}: head mean={v.mean():.3f} p10={np.percentile(v,10):.3f} "
              f"p50={np.percentile(v,50):.3f} p90={np.percentile(v,90):.3f} style mean={st.mean():.3f}")

    pool = Pool(pool_dir("data"))
    ids = list(pool.index)
    rng = np.random.default_rng(1)
    sel = rng.choice(len(ids), 200, replace=False)
    pils = [pool.image(ids[i]).convert("RGB") for i in sel]
    ps = score(pils)
    for g in ("cat_head", "dog_head"):
        v = np.array([r[g] for r in ps])
        print(f"random whole pool {g}: mean={v.mean():.3f} p50={np.percentile(v,50):.3f} "
              f"p90={np.percentile(v,90):.3f} max={v.max():.3f}")


if __name__ == "__main__":
    main()
