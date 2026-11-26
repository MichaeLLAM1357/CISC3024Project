"""Probe: do stronger negative prompts separate prints/line art from real photos?

Scores a fixed set of pool crops that a human eye rejects (product boxes, line art, screenshots,
silhouettes) against random manifest rows, with the current prompt set and an extended one.
No dev images are used; thresholds are not fitted here, the probe only shows the margin.

    python pipeline/probe_clip_prompts.py
"""

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

import open_clip  # noqa: E402

BAD = ["s001665", "s001743", "s002020", "s002263", "s003296", "s003036", "s003861",
       "s001374", "s003296", "s000241", "s000379", "s003475"]
POS = ["a photograph of a real cat head", "a photo of a real dog's head"]
NEG_BASE = ["a cartoon of an animal", "a drawing of an animal", "a painting of an animal",
            "a 3D render of an animal", "a toy animal", "a statue of an animal",
            "a sketch of an animal", "an embroidery of an animal"]
NEG_EXTRA = ["line art of an animal", "clipart of an animal", "a vector illustration of an animal",
             "a silhouette of an animal", "a logo with an animal", "a sticker of an animal",
             "a tattoo of an animal", "product packaging with an animal",
             "a screenshot of a website", "a photo of a screen or monitor",
             "an animal made of plastic", "a poster hanging on a wall"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-L-14")

    def enc(prompts):
        with torch.no_grad():
            f = model.encode_text(tok(prompts).to(device))
            return f / f.norm(dim=-1, keepdim=True)

    pos, neg_base, neg_ext = enc(POS), enc(NEG_BASE), enc(NEG_BASE + NEG_EXTRA)

    pool = Pool(pool_dir("data"))
    geom = {}
    for line in open("results/manifest_v1/candidates_raw.jsonl", encoding="utf-8"):
        d = json.loads(line)
        geom[d["pool_id"]] = d

    rows = list(csv.DictReader(open("submission/manifest.csv", encoding="utf-8")))
    good_ids = [r["pool_id"] for r in rows[::70]][:30]
    ids = [p for p in dict.fromkeys(BAD + good_ids)]

    def crop_of(pid):
        g = geom.get(pid)
        if g is None:
            return None
        im = pool.image(pid).convert("RGB")
        return im.crop((g["x0"], g["y0"], g["x0"] + g["size"], g["y0"] + g["size"]))

    crops, kept_ids = [], []
    for pid in ids:
        c = crop_of(pid)
        if c is not None:
            crops.append(preprocess(c))
            kept_ids.append(pid)

    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 16):
            batch = torch.stack(crops[i:i + 16]).to(device)
            f = model.encode_image(batch)
            f /= f.norm(dim=-1, keepdim=True)
            for k in range(f.shape[0]):
                ps = (f[k] @ pos.T).mean()
                for name, neg in (("base", neg_base), ("ext", neg_ext)):
                    ns = (f[k] @ neg.T).mean()
                    s = torch.softmax(torch.stack([ps, ns]) * 100.0, 0)[0].item()
                    out.append((kept_ids[i + k], name, s))

    by_id = {}
    for pid, name, s in out:
        by_id.setdefault(pid, {})[name] = s
    for pid in kept_ids:
        tag = "BAD " if pid in BAD else "good"
        print(f"{tag} {pid}  base={by_id[pid]['base']:.3f}  ext={by_id[pid]['ext']:.3f}")


if __name__ == "__main__":
    main()
