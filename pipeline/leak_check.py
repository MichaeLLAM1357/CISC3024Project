import os
"""Which of the Phase 0 leak classes is actually detectable with the tools on hand?

    python pipeline/leak_check.py --index results/probe/probe_index_HUMAN.csv \
        --data-root data --yolo $CATDOG_YOLO

`pose_check.py` showed that orientation is invisible to EXIF, YOLO rotation and CLIP rotation.
This script measures the three leak classes that *do* have a cheap signal, on the same 100
human-labelled A_pass cells, so the choice between "build a landmark model" and "fix what is
measurable" rests on numbers instead of on hope:

    second_head  YOLO11x at conf=0.02 on the whole pool image; the signal is the best
                 confidence of a *second* cat/dog box whose centre falls inside the crop square.
                 The current gate only rejects when YOLO returns 2+ boxes at its own conf, so a
                 weak second head is exactly what leaks.
    ocr_area     from the cached stage-4 record (already computed on every crop).
    style        from the cached stage-3 CLIP record (photo-vs-art margin).

Each signal is scored against the human's matching group (positives) and the 57 `good` cells
(negatives): AUC, plus the recall/FPR at the best band, plus what precision the projection buys.
Nothing here is wired into the pipeline; it only answers "would this gate have helped?".
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

COCO = (15, 16)


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


def second_head_signals(rows, pool, weights, conf, imgsz, device):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out = []
    for r in rows:
        im = np.asarray(pool.image(r["pool_id"]).convert("RGB"))
        res = model.predict([im], imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        x0, y0, size = int(r["x0"]), int(r["y0"]), int(r["size"])
        inside = []
        if res.boxes is not None and len(res.boxes):
            for cls, cf, xy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                   res.boxes.xyxy.tolist()):
                if int(cls) not in COCO:
                    continue
                cx, cy = (xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2
                if x0 <= cx <= x0 + size and y0 <= cy <= y0 + size:
                    inside.append(float(cf))
        inside.sort(reverse=True)
        out.append({"n_in_crop": len(inside),
                    "best": inside[0] if inside else 0.0,
                    "second": inside[1] if len(inside) > 1 else 0.0})
    return out


def sweep(pos_vals, neg_vals, pos_ids, neg_ids, all_vals, all_flags, grid, direction=1):
    rows = []
    for t in grid:
        def flag(v):
            return (v >= t) if direction > 0 else (v <= t)
        recall = sum(1 for v in pos_vals if flag(v)) / len(pos_vals) if pos_vals else 0.0
        fpr = sum(1 for v in neg_vals if flag(v)) / len(neg_vals) if neg_vals else 0.0
        g = sum(1 for v, f in zip(all_vals, all_flags) if flag(v) and f == "good")
        b = sum(1 for v, f in zip(all_vals, all_flags) if flag(v) and f != "good")
        rows.append({"threshold": round(float(t), 4), "recall": recall, "fpr": fpr,
                     "flagged_good": g, "flagged_bad": b,
                     "projected_precision": (57 - g) / (100 - g - b) if (100 - g - b) > 0 else 1.0})
    best = max(rows, key=lambda r: r["recall"] - 3 * r["fpr"])
    return rows, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", default="results/probe/probe_index_HUMAN.csv")
    ap.add_argument("--data-root", default=os.environ.get("CATDOG_DATA", str(ROOT / "data")))
    ap.add_argument("--yolo", default=os.environ.get("CATDOG_YOLO", str(ROOT / "yolo11x.pt")))
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "manifest_v1"))
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results/probe/leak_check.json")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.index, encoding="utf-8")) if r["board"] == "A_pass"]
    qual = {}
    for line in open(Path(args.cache_dir) / "quality_raw.jsonl", encoding="utf-8"):
        q = json.loads(line)
        qual[q["pool_id"]] = q

    flag = ["good" if r["verdict"] == "good"
            else r["verdict_reason"].strip() if r["verdict"] == "bad" else "?" for r in rows]
    print("A_pass:", Counter(flag))

    pool = Pool(pool_dir(args.data_root))
    device = 0 if torch.cuda.is_available() else "cpu"
    print("running YOLO at conf", args.conf, "...", flush=True)
    sh = second_head_signals(rows, pool, args.yolo, args.conf, args.imgsz, device)

    sig = {
        "second_head_conf": ([d["second"] for d in sh], 1,
                             ["two heads"], "second cat/dog box inside the crop square"),
        "ocr_area": ([qual[r["pool_id"]].get("ocr_area", 0.0) for r in rows], 1,
                     ["text/watermark"], "OCR text area fraction on the crop"),
        "style_low": ([qual[r["pool_id"]].get("style", 1.0) for r in rows], -1,
                      ["product/print", "cartoon/drawing", "statue/toy"], "CLIP photo-vs-art score (low = art)"),
    }

    res = {"index": args.index, "conf": args.conf, "counts": dict(Counter(flag)), "signals": {}}
    for name, (vals, direction, pos_groups, what) in sig.items():
        pos = [v for v, f in zip(vals, flag) if f in pos_groups]
        neg = [v for v, f in zip(vals, flag) if f == "good"]
        grid = np.linspace(min(vals), max(vals) + 1e-9, 41)
        sw, best = sweep(pos, neg, [], [], vals, flag, grid, direction)
        a = auc(pos, neg) if direction > 0 else auc([-v for v in pos], [-v for v in neg])
        ok = [r for r in sw if r["recall"] >= 0.70 and r["fpr"] <= 0.07]
        res["signals"][name] = {"what": what, "positives": pos_groups, "n_pos": len(pos),
                                "n_neg": len(neg), "auc": a, "sweep": sw, "best": best,
                                "meets_prereg": bool(ok), "values": [round(float(v), 5) for v in vals]}
        print(f"{name:18s} n_pos={len(pos):2d} AUC={a:.3f} meets_prereg={bool(ok)} "
              f"best recall={best['recall']:.2f} fpr={best['fpr']:.2f} "
              f"proj_prec={best['projected_precision']:.3f}")

    res["second_head_detail"] = [{"pool_id": r["pool_id"], "flag": f, **d}
                                 for r, f, d in zip(rows, flag, sh)]
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("->", args.out)


if __name__ == "__main__":
    main()
