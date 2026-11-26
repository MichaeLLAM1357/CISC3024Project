import os
"""Measure orientation signals for the Phase 0 'not upright' errors -- BEFORE touching the pipeline.

    python pipeline/pose_check.py --index results/probe/probe_index_HUMAN.csv \
        --data-root data --yolo $CATDOG_YOLO

The Phase 0 sheet said 14 of 100 kept crops are `not upright`, and no geometric check can see
pose (the tflite face localizer returns 4 numbers, no landmarks). This script asks a narrower
question than "build a pose model": **on the human's own 14 positives and 57 'good' negatives,
does any zero-download signal separate them well enough to be worth a pipeline change?**

Two candidate signals, both from curation tools already registered in README.md:

    yolo_rot   rotate the crop by k*45 deg (k=0..7) and re-detect with YOLO11x. An upright head
               peaks at 0 deg; a lying/inverted head peaks at another angle. Signal = best other
               rotation's confidence minus the 0 deg confidence.
    clip_orient  CLIP ViT-L-14 cosine margin between "upright head" prompts and
               "upside-down / lying on its side / rotated" prompts.

It also checks EXIF orientation (a rotated jpg loaded without `exif_transpose` would look
identical to a rotated animal) -- that hypothesis was checked and is already dead (all 170 cells
are orientation=1), but the column stays so a future build cannot reintroduce it silently.

Nothing is written into the pipeline. The output JSON is the evidence for whether option B is
worth doing at all: with 14 pose errors out of 43 bad cells, a *perfect* pose filter moves
57/100 to 57/86 = 0.663, so the decision rule here is pre-registered as
    recall(not_upright) >= 0.70  AND  false-positive rate on good <= 0.07,
and the printed projection says what precision that would actually buy.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))
from catdog_eval.data import Pool, crop, pool_dir  # noqa: E402

CLIP_POS = ("a photo of an upright {a} head", "a photo of a {a} facing the camera")
CLIP_NEG = ("an upside-down photo of a {a}", "a photo of a {a} lying on its side",
            "a rotated photo of a {a}", "a photo of a {a} on its back")


def rot(im: Image.Image, deg: int) -> Image.Image:
    if deg % 90 == 0:
        return im.transpose({0: None, 90: Image.Transpose.ROTATE_90,
                             180: Image.Transpose.ROTATE_180,
                             270: Image.Transpose.ROTATE_270}[deg]) if deg else im
    r = im.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True)
    side = max(r.size)
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    canvas.paste(r, ((side - r.width) // 2, (side - r.height) // 2))
    return canvas


def box(c: dict) -> tuple[int, int, int]:
    """The human sheet stores the crop as three columns, not a tuple."""
    return int(c["x0"]), int(c["y0"]), int(c["size"])


def exif_orientation(pool: Pool, pid: str) -> int:
    where = pool._where.get(pid)  # noqa: SLF001
    if where is None:
        return 0
    if where.suffix == ".zip":
        with ZipFile(where) as z:
            im = Image.open(BytesIO(z.read(f"images/{pid}.jpg")))
    else:
        im = Image.open(where)
    try:
        return int(im.getexif().get(274, 1))
    except Exception:  # noqa: BLE001
        return 1


def yolo_rot_scores(cells, pool, weights, steps, imgsz, batch, device):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    angles = [int(360 * k / steps) for k in range(steps)]
    imgs, keys = [], []
    for i, c in enumerate(cells):
        x0, y0, size = box(c)
        base = crop(pool.image(c["pool_id"]), x0, y0, size, 0, res=size)
        for a in angles:
            imgs.append(np.asarray(rot(base, a).convert("RGB")))
            keys.append((i, a))
    conf = {}
    for s in range(0, len(imgs), batch):
        res = model.predict(imgs[s:s + batch], imgsz=imgsz, conf=0.01, verbose=False, device=device)
        for (i, a), r in zip(keys[s:s + batch], res):
            best = 0.0
            if r.boxes is not None and len(r.boxes):
                for cls, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    if int(cls) in (15, 16):
                        best = max(best, float(cf))
            conf[(i, a)] = best
    out = []
    for i in range(len(cells)):
        c0 = conf[(i, 0)]
        other = max(conf[(i, a)] for a in angles if a != 0)
        arg = max((a for a in angles if a != 0), key=lambda a: conf[(i, a)])
        out.append({"conf0": c0, "conf_best_rot": other, "best_rot": arg,
                    "margin": other - c0,
                    "conf_by_rot": {str(a): conf[(i, a)] for a in angles}})
    return out


def clip_scores(cells, pool, model_name, pretrained, device):
    """Two CLIP signals on the same 224 px crop.

    clip_orient : margin between "upright head" and "upside-down / lying" prompts (text-side).
    clip_rot    : encode the crop and its 90/180/270 deg twins, compare each to the *class* text
                  embedding, and take best-other minus 0 deg. An upright head should already be
                  the closest to "a photo of a cat"; an inverted one should improve after rotation.
    """
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(model_name)

    def text_emb(strings):
        with torch.no_grad():
            f = model.encode_text(tok(strings).to(device))
        return f / f.norm(dim=-1, keepdim=True)

    cls_emb = {"cat": text_emb(["a photo of a cat head"]),
               "dog": text_emb(["a photo of a dog head"])}
    pos_emb = {a: text_emb([s_.format(a=a) for s_ in CLIP_POS]) for a in ("cat", "dog")}
    neg_emb = {a: text_emb([s_.format(a=a) for s_ in CLIP_NEG]) for a in ("cat", "dog")}

    out = []
    for c in cells:
        x0, y0, size = box(c)
        base = crop(pool.image(c["pool_id"]), x0, y0, size, 0, res=224)
        a = "cat" if c["label"] == "0" else "dog"
        sims = {}
        with torch.no_grad():
            for ang in (0, 90, 180, 270):
                t = preprocess(rot(base, ang).convert("RGB")).unsqueeze(0).to(device)
                f = model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)
                sims[ang] = (float((f @ pos_emb[a].T).max()),
                             float((f @ neg_emb[a].T).max()),
                             float((f @ cls_emb[a].T).max()))
        out.append({"clip_pos": sims[0][0], "clip_neg": sims[0][1],
                    "orient_margin": sims[0][0] - sims[0][1],
                    "rot_margin": max(sims[k][2] for k in (90, 180, 270)) - sims[0][2],
                    "sim_by_rot": {str(k): round(v[2], 5) for k, v in sims.items()}})
    return out


def face_rot_scores(cells, pool):
    """The face localizer we already ship, used as an orientation probe.

    It regresses a face box and returns no score, so "confidence" has to be proxied by the box
    it commits to: run it on the crop and on its 90/180/270 deg twins and record the box area.
    A head the model reads at 0 deg should give the largest box there; if another rotation wins
    by a wide margin, the head is probably not upright. Weak proxy, but it is the only model in
    the repo that was ever trained on cat/dog faces.
    """
    sys.path.insert(0, str(ROOT / "pipeline"))
    from build_manifest import FACE_MODELS, load_localizer, predict_face
    localizers = {c: load_localizer(*FACE_MODELS[c]) for c in (0, 1)}
    out = []
    for c in cells:
        x0, y0, size = box(c)
        base = crop(pool.image(c["pool_id"]), x0, y0, size, 0, res=96)
        lab = 0 if c["label"] == "0" else 1
        areas = {}
        for ang in (0, 90, 180, 270):
            fb = predict_face(localizers[lab], rot(base, ang).convert("RGB"))
            areas[ang] = (fb[2] - fb[0]) * (fb[3] - fb[1]) if fb else 0.0
        a0 = areas[0]
        best = max(areas[90], areas[180], areas[270])
        out.append({"area0": a0, "area_best_rot": best, "best_rot": max(areas, key=areas.get),
                    "gain": (best / a0 - 1.0) if a0 > 0 else (9.9 if best > 0 else 0.0),
                    "area_by_rot": {str(k): round(v, 1) for k, v in areas.items()}})
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score of a positive > score of a negative), ties counted 0.5."""
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


def sweep(values, flags, grid):
    """For each threshold, count flagged good / flagged bad over the 100 pass cells."""
    rows = []
    for t in grid:
        hit = [v >= t for v in values]
        g = sum(1 for h, f in zip(hit, flags) if h and f == "good")
        b = sum(1 for h, f in zip(hit, flags) if h and f == "bad")
        nu = sum(1 for h, f in zip(hit, flags) if h and f == "not_upright")
        rows.append({"threshold": round(t, 4), "flagged_good": g, "flagged_bad": b,
                     "flagged_not_upright": nu,
                     "recall_pu": nu / 14 if nu else 0.0,
                     "fpr_good": g / 57, "projected_precision": (57 - g) / (100 - g - b) if (100 - g - b) > 0 else 1.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", default="results/probe/probe_index_HUMAN.csv")
    ap.add_argument("--data-root", default=os.environ.get("CATDOG_DATA", str(ROOT / "data")))
    ap.add_argument("--yolo", default=os.environ.get("CATDOG_YOLO", str(ROOT / "yolo11x.pt")))
    ap.add_argument("--clip-model", default="ViT-L-14")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--steps", type=int, default=8, help="rotations around the circle")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/probe/pose_check.json")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-yolo", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.index, encoding="utf-8")) if r["board"] == "A_pass"]
    pool = Pool(pool_dir(args.data_root))
    dev_yolo = 0 if torch.cuda.is_available() else "cpu"
    dev_torch = "cuda" if torch.cuda.is_available() else "cpu"
    flags = ["good" if r["verdict"] == "good"
             else "not_upright" if r["verdict_reason"] == "not upright" else "bad" for r in rows]
    print("A_pass cells:", Counter(flags))

    orients = [exif_orientation(pool, r["pool_id"]) for r in rows]
    print("EXIF orientation:", Counter(orients))

    res = {"n": len(rows), "counts": dict(Counter(flags)),
           "exif_orientation": {str(k): v for k, v in Counter(orients).items()},
           "ceiling_if_pose_perfect": 57 / 86, "signals": {}}

    if not args.skip_yolo:
        print("running YOLO rotation check ...", flush=True)
        yr = yolo_rot_scores(rows, pool, args.yolo, args.steps, args.imgsz, args.batch, dev_yolo)
    else:
        yr = [{"margin": float("nan"), "conf0": float("nan"), "best_rot": -1,
               "conf_best_rot": float("nan"), "conf_by_rot": {}} for _ in rows]
    print("running face-localizer rotation check ...", flush=True)
    fr = face_rot_scores(rows, pool)
    if not args.skip_clip:
        print("running CLIP checks ...", flush=True)
        cr = clip_scores(rows, pool, args.clip_model, args.clip_pretrained, dev_torch)
    else:
        cr = [{"orient_margin": float("nan"), "rot_margin": float("nan"),
               "clip_pos": float("nan"), "clip_neg": float("nan")} for _ in rows]

    for name, vals in (("yolo_rot_margin", [d["margin"] for d in yr]),
                       ("clip_orient_margin", [d["orient_margin"] for d in cr]),
                       ("clip_rot_margin", [d["rot_margin"] for d in cr]),
                       ("face_rot_gain", [d["gain"] for d in fr])):
        pos = [v for v, f in zip(vals, flags) if f == "not_upright"]
        neg = [v for v, f in zip(vals, flags) if f == "good"]
        grid = np.linspace(min(vals) - 1e-6, max(vals) + 1e-6, 61)
        sw = sweep(vals, flags, grid)
        ok = [r for r in sw if r["recall_pu"] >= 0.70 and r["fpr_good"] <= 0.07]
        best = max(sw, key=lambda r: (r["recall_pu"] - 3 * r["fpr_good"]))
        res["signals"][name] = {"auc": auc(pos, neg), "sweep": sw,
                                "meets_prereg": bool(ok),
                                "prereg_rows": ok[:3], "best_by_scalar": best,
                                "values": [round(v, 5) for v in vals]}
        print(f"{name}: AUC={res['signals'][name]['auc']:.3f}  meets prereg={bool(ok)}  "
              f"best {best}")

    res["face_rot_detail"] = [{"pool_id": r["pool_id"], "flag": f, **d}
                              for r, f, d in zip(rows, flags, fr)]
    res["yolo_rot_detail"] = [{"pool_id": r["pool_id"], "flag": f, **{k: v for k, v in d.items()}}
                              for r, f, d in zip(rows, flags, yr)]
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("->", args.out)


if __name__ == "__main__":
    main()
