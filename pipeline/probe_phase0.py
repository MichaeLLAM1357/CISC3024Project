"""Phase 0: build the 150-cell human-eye probe (plus a 20-cell boundary strip).

    python pipeline/probe_phase0.py \
        --scan results/scan/yolo_detections.jsonl \
        --data-root data \
        --cache-dir results/manifest_v1 \
        --out results/probe

What it draws (phase-0 probe):

    probe_pass100.png      100 cells: 50 cat + 50 dog that the current rules KEEP
    probe_reject50.png      50 cells: what the rules THROW AWAY, stratified over the four
                            stages that can reject -- YOLO (0 dets / 2+ dets), geometry,
                            CLIP style, OCR text
    probe_boundary20.png    20 cells: the same kept candidate re-cropped so the head fills
                            r = 0.575 (10) or r = 0.825 (10) of the side, i.e. just outside
                            the 0.60-0.80 specification band

The pool is read read-only through the starter-kit `Pool` (zip shards); nothing is copied.
The geometry stage is recomputed with the builder's own `predict_face` / `fixed_head_box` /
`photo_stats`, so the thresholds are the ones Phase 1 will actually use -- no second copy of
the rules to drift. Legitimately rejected geometry candidates are not cached with a crop box,
so their cell is drawn with the crop the same formula *would* have produced (clamped into the
image); the cell says so via `crop_source` in probe_index.csv.

Sampling is even across the whole pool id span (not a random draw), one cell per pool image.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # curation models are cached; no downloads here

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "starter_kit"))

from build_manifest import (  # noqa: E402
    COCO_CAT, COCO_DOG, CONTEXT, FACE_MODELS, MIN_SIZE, TARGET_R,
    fixed_head_box, load_localizer, photo_stats, predict_face,
)
from catdog_eval.data import Pool, crop, pool_dir  # noqa: E402

CELL, COLS = 140, 10
HEADER = 30
REASONS = ("ratio", "border", "head_too_small", "aspect")
PRED_R = 0.70  # the target ratio the builder uses; only for the "would-be" display crop
FONT_PATHS = ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf")


def font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def even_pick(ids: list[str], k: int) -> list[str]:
    """k ids spread evenly over the sorted list (deterministic, whole-span, no RNG)."""
    ids = sorted(ids)
    n = len(ids)
    if k > n:
        raise SystemExit(f"even_pick: asked {k} of {n} candidates")
    if k == n:
        return ids
    taken: set[int] = set()
    out = []
    for t in np.linspace(0, n - 1, k):
        j = int(round(float(t)))
        while j in taken and j < n - 1:
            j += 1
        while j in taken and j > 0:
            j -= 1
        taken.add(j)
        out.append(ids[j])
    assert len(set(out)) == k, "even_pick produced duplicates"
    return out


def load_scan(path: Path) -> dict[str, list[dict]]:
    dets = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            dets[r["pool_id"]] = r["detections"]
    return dets


def load_cache(folder: Path):
    """Join the stage-1 candidate cache with the stage-3/4 quality cache by pool_id."""
    cands, qual = {}, {}
    for line in open(folder / "candidates_raw.jsonl", encoding="utf-8"):
        r = json.loads(line)
        cands[r["pool_id"]] = r
    for line in open(folder / "quality_raw.jsonl", encoding="utf-8"):
        r = json.loads(line)
        qual[r["pool_id"]] = r
    if set(cands) != set(qual):
        raise SystemExit("candidate/quality cache disagree on pool ids; rebuild instead")
    return cands, qual


def letterbox(im: Image.Image, box: int, boxes=None, scale_ref=None) -> Image.Image:
    """Whole pool image inside a `box` square, with detection boxes drawn on top."""
    im = im.convert("RGB")
    w, h = im.size
    s = box / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    canvas = Image.new("RGB", (box, box), (232, 232, 232))
    ox, oy = (box - nw) // 2, (box - nh) // 2
    canvas.paste(im.resize((nw, nh), Image.Resampling.LANCZOS), (ox, oy))
    if boxes:
        d = ImageDraw.Draw(canvas)
        for b in boxes:
            x1, y1, x2, y2 = (v * s for v in b)
            d.rectangle([ox + x1, oy + y1, ox + x2 - 1, oy + y2 - 1], outline=(0, 190, 0), width=2)
    return canvas


def would_be_crop(face, image_size):
    """The crop `fixed_head_box` would have produced, with every gate skipped (display only)."""
    fx1, fy1, fx2, fy2 = face
    fw, fh = fx2 - fx1, fy2 - fy1
    hx1, hx2 = fx1 - 0.10 * fw, fx2 + 0.10 * fw
    hy1, hy2 = fy1 - 0.25 * fh, fy2 + 0.05 * fh
    w, box_h = hx2 - hx1, hy2 - hy1
    h = max(w, box_h)
    side = int(round(max(h / PRED_R, MIN_SIZE)))
    img_w, img_h = image_size
    side = min(side, min(img_w, img_h))
    if side < MIN_SIZE:
        return None
    x0 = int(round((hx1 + hx2) / 2.0 - side / 2.0))
    y0 = int(round((hy1 + hy2) / 2.0 - side / 2.0))
    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))
    return x0, y0, side, h


def harvest_geom(scan_single, sample_ids, pool, localizers, min_sharp, max_flat, want):
    """Replay stage 1 on a bounded sample and keep the real geometry/stats rejects."""
    buckets: dict[tuple[int, str], list[dict]] = {}
    scanned = 0
    for pid in sample_ids:
        if all(len(v) >= want for v in buckets.values()) and len(buckets) >= 8:
            break
        det = scan_single[pid]
        label = 0 if det["cls"] == COCO_CAT else 1
        try:
            im = pool.image(pid).convert("RGB")
        except Exception:  # noqa: BLE001 - a missing image is not a probe case
            continue
        scanned += 1
        w, h = im.size
        x1, y1, x2, y2 = det["xyxy"]
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - CONTEXT * bw)), max(0, int(y1 - CONTEXT * bh))
        cx2, cy2 = min(w, int(x2 + CONTEXT * bw)), min(h, int(y2 + CONTEXT * bh))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue
        fb = predict_face(localizers[label], im.crop((cx1, cy1, cx2, cy2)))
        if fb is None:
            continue
        fb = [fb[0] + cx1, fb[1] + cy1, fb[2] + cx1, fb[3] + cy1]
        geom, reason = fixed_head_box(fb, (w, h))
        rec = {"pool_id": pid, "label": label, "det_conf": float(det["conf"])}
        if geom is None:
            wb = would_be_crop(fb, (w, h))
            if wb is None:
                continue
            rec.update({"reason": reason, "crop": wb[:3], "h": wb[3], "crop_source": "would-be"})
        else:
            x0, y0, side = geom["x0"], geom["y0"], geom["size"]
            stats = photo_stats(im.crop((x0, y0, x0 + side, y0 + side)))
            rec.update({"sharp": stats["sharp"], "flat": stats["flat"], "h": geom["h"]})
            if stats["sharp"] < min_sharp:
                rec.update({"reason": "blurry", "crop": (x0, y0, side), "crop_source": "cache"})
            elif stats["flat"] > max_flat:
                rec.update({"reason": "flat", "crop": (x0, y0, side), "crop_source": "cache"})
            else:
                continue
        buckets.setdefault((label, rec["reason"]), []).append(rec)
    return buckets, scanned


def draw_sheet(cells: list[dict], title: str, path: Path, pool: Pool) -> None:
    rows = (len(cells) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * CELL, HEADER + rows * CELL), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 8), title, fill=(0, 0, 0), font=font(13))
    f1, f2 = font(12), font(10)
    for k, c in enumerate(cells):
        x, y = (k % COLS) * CELL, HEADER + (k // COLS) * CELL
        im = pool.image(c["pool_id"])
        if c["crop"] is None:  # whole image: YOLO stage cells
            sheet.paste(letterbox(im, 96, c.get("boxes")), (x + 2, y + 2))
            sheet.paste(letterbox(im, 32), (x + 102, y + 2))
        else:
            x0, y0, size = c["crop"]
            sheet.paste(crop(im, x0, y0, size, 0, res=96), (x + 2, y + 2))
            sheet.paste(crop(im, x0, y0, size, 0, res=32), (x + 102, y + 2))
        d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(205, 205, 205))
        name = "cat" if c["label"] == 0 else "dog" if c["label"] == 1 else "?"
        d.text((x + 3, y + 102), f"#{k:03d} {name}", fill=(0, 0, 0), font=f1)
        d.text((x + 3, y + 117), c["pool_id"], fill=(90, 90, 90), font=f2)
        if c.get("note"):
            d.text((x + 3, y + 128), c["note"][:22], fill=(150, 60, 0), font=f2)
    sheet.save(path)
    print(f"wrote {path}  ({len(cells)} cells, {sheet.size[0]}x{sheet.size[1]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan", default="results/scan/yolo_detections.jsonl")
    ap.add_argument("--data-root", default=os.environ.get("CATDOG_DATA", str(ROOT / "data")))
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "manifest_v1"),
                    help="stage-1/stage-3 cache of the pool (read-only; nothing is written)")
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--geom-sample", type=int, default=900,
                    help="how many single-detection images to replay stage 1 on")
    ap.add_argument("--min-sharpness", type=float, default=20.0)
    ap.add_argument("--max-flat", type=float, default=0.75)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    targets = [out / n for n in ("probe_pass100.png", "probe_reject50.png",
                                 "probe_boundary20.png", "probe_index.csv")]
    if any(t.exists() for t in targets) and not args.force:
        raise SystemExit(f"{out} already has a probe; pass --force to rebuild")
    out.mkdir(parents=True, exist_ok=True)

    scan = load_scan(Path(args.scan))
    cands, qual = load_cache(Path(args.cache_dir))
    if not set(cands) <= set(scan):
        raise SystemExit("cache has pool ids the scan does not; wrong cache dir?")

    miss = [p for p, ds in scan.items() if not ds]
    multi = [p for p, ds in scan.items() if len(ds) >= 2]
    single = {p: ds[0] for p, ds in scan.items()
              if len(ds) == 1 and ds[0]["cls"] in (COCO_CAT, COCO_DOG)}
    kept = [p for p in qual if qual[p]["keep"]]
    clip = [p for p in qual if qual[p]["reason"] == "clip_style"]
    ocr = [p for p in qual if qual[p]["reason"] == "ocr_text"]
    print(f"pool {len(scan)}  miss {len(miss)}  multi {len(multi)}  single {len(single)}  "
          f"kept {len(kept)}  clip {len(clip)}  ocr {len(ocr)}")

    pool = Pool(pool_dir(args.data_root))
    index = pool._where  # noqa: SLF001 - only used to assert membership, never to edit
    used: set[str] = set()
    cells_pass, cells_rej, cells_bnd = [], [], []

    # --- A: 100 kept candidates, 50 per class, interleaved so the sheet sweeps the pool -----
    per_cls = {}
    for lab in (0, 1):
        ids = [p for p in kept if cands[p]["label"] == lab]
        per_cls[lab] = even_pick(ids, 50)
        assert len({p for p in per_cls[lab]}) == 50
    for i in range(50):
        for lab in (0, 1):
            pid = per_cls[lab][i]
            used.add(pid)
            x0, y0, size = cands[pid]["crop"]
            cells_pass.append({"pool_id": pid, "label": lab, "crop": (x0, y0, size),
                               "note": f"r={cands[pid]['r']:.2f}"})

    # --- B: 50 rejects, stratified over the four stages ------------------------------------
    localizers = {c: load_localizer(*FACE_MODELS[c]) for c in (0, 1)}
    sample = [p for p in even_pick(list(single), args.geom_sample) if p not in used]
    buckets, scanned = harvest_geom(single, sample, pool, localizers,
                                    args.min_sharpness, args.max_flat, want=6)
    print(f"geometry replay: {scanned} images scanned, "
          f"{sum(len(v) for v in buckets.values())} rejects in "
          f"{sorted({r for _, r in buckets})}")

    geom_cells, geom_short = [], []
    rec_by_pid = {g["pool_id"]: g for v in buckets.values() for g in v}
    reason_of = {pid: r for (_, r), v in buckets.items() for pid in (g["pool_id"] for g in v)}
    for lab in (0, 1):
        picked: list[tuple[str, str]] = []
        for reason in REASONS:
            got = [g["pool_id"] for g in buckets.get((lab, reason), []) if g["pool_id"] not in used]
            ids = even_pick(got, min(2, len(got)))
            if len(ids) < 2:  # e.g. cats almost never fail h/w > 1.4: 1 of 895
                geom_short.append(f"{'cat' if lab == 0 else 'dog'}:{reason} found {len(got)} want 2")
            picked += [(p, reason) for p in ids]
        if len(picked) < 8:  # top up from whichever reason still has spares, same class
            chosen = {p for p, _ in picked}
            spare = [g["pool_id"] for reason in REASONS + ("flat", "blurry")
                     for g in buckets.get((lab, reason), [])
                     if g["pool_id"] not in used and g["pool_id"] not in chosen]
            picked += [(p, reason_of[p]) for p in even_pick(spare, 8 - len(picked))]
        for pid, reason in picked:
            used.add(pid)
            geom_cells.append({"pool_id": pid, "label": lab, "crop": rec_by_pid[pid]["crop"],
                               "note": reason.replace("_", " "),
                               "source": rec_by_pid[pid]["crop_source"], "reason": reason})
    print("geometry quotas short, filled from the same class:",
          geom_short if geom_short else "none")
    # explicit quotas: clip 4 per class, ocr 3 per class
    for reason, quota in (("clip_style", 4), ("ocr_text", 3)):
        src = clip if reason == "clip_style" else ocr
        for lab in (0, 1):
            ids = [p for p in src if cands[p]["label"] == lab and p not in used]
            for pid in even_pick(ids, min(quota, len(ids))):
                used.add(pid)
                st = qual[pid].get("style")
                note = reason.split("_")[0] + (f" {st:.2f}" if st is not None else "")
                cells_rej.append({"pool_id": pid, "label": lab, "crop": tuple(cands[pid]["crop"]),
                                  "note": note, "source": "cache", "reason": reason})
    for group, quota, src in (("yolo_multi", 10, multi), ("yolo_miss", 10, miss)):
        pick = []
        if group == "yolo_multi":
            lab_of = {}
            for p in src:
                cls = max(scan[p], key=lambda d: d["conf"])["cls"]
                lab_of[p] = 0 if cls == COCO_CAT else 1 if cls == COCO_DOG else None
            for lab in (0, 1):
                ids = [p for p in src if lab_of.get(p) == lab and p not in used]
                pick += [(p, lab) for p in even_pick(ids, quota // 2)]
        else:
            pick = [(p, None) for p in even_pick([p for p in src if p not in used], quota)]
        for pid, lab in pick:
            used.add(pid)
            cells_rej.append({"pool_id": pid, "label": lab, "crop": None,
                              "note": f"{len(scan[pid])} dets" if group == "yolo_multi" else "no det",
                              "source": "full", "reason": group,
                              "boxes": [d["xyxy"] for d in scan[pid]],
                              "group_label": lab})
    cells_rej = geom_cells + cells_rej

    # --- C: 20 boundary cells, the same candidate re-cropped at r = 0.575 / 0.825 ----------
    for r_t in (0.575, 0.825):
        for lab in (0, 1):
            made = 0
            ids = [p for p in kept if cands[p]["label"] == lab and p not in used]
            for pid in even_pick(ids, 80):
                if made == 5:
                    break
                im = pool.image(pid)
                W, H = im.size
                side = int(round(cands[pid]["h"] / r_t))
                if side < 40 or side > min(W, H):
                    continue
                hx1, hy1, hx2, hy2 = cands[pid]["head"]
                x0 = int(round((hx1 + hx2) / 2 - side / 2))
                y0 = int(round((hy1 + hy2) / 2 - side / 2))
                x0, y0 = max(0, min(x0, W - side)), max(0, min(y0, H - side))
                used.add(pid)
                made += 1
                cells_bnd.append({"pool_id": pid, "label": lab, "crop": (x0, y0, side),
                                  "note": f"r={cands[pid]['h'] / side:.2f}", "source": "boundary"})

    # --- self-checks before anything is written -------------------------------------------
    from collections import Counter
    print("reject groups:", dict(Counter(c.get("reason") for c in cells_rej)))
    print("geom buckets:", {f"{k[0]}:{k[1]}": len(v) for k, v in sorted(buckets.items())})
    all_pids = [c["pool_id"] for c in cells_pass + cells_rej + cells_bnd]
    assert len(cells_pass) == 100, len(cells_pass)
    assert len(cells_rej) == 50, len(cells_rej)
    assert len(cells_bnd) == 20, len(cells_bnd)
    assert len(set(all_pids)) == 170, "a pool image appears twice"
    assert all(p in index for p in all_pids), "pool id not in the pool index"
    for c in cells_pass:
        assert c["label"] in (0, 1) and qual[c["pool_id"]]["keep"] is True
    for c in cells_pass + cells_rej + cells_bnd:
        if c["crop"] is None:
            continue
        im = pool.image(c["pool_id"])
        x0, y0, size = c["crop"]
        assert size >= 40 and x0 >= 0 and y0 >= 0, c
        assert x0 + size <= im.size[0] and y0 + size <= im.size[1], c

    draw_sheet(cells_pass,
               "A | 100 KEPT candidates (50 cat / 50 dog, even over the pool) - 96 px left / "
               "32 px right = what the model sees", out / "probe_pass100.png", pool)
    draw_sheet(cells_rej,
               "B | 50 REJECTED (16 geometry - 8 CLIP - 6 OCR - 10 two+ dets - 10 no det) - "
               "FULL = whole pool image, green = YOLO boxes", out / "probe_reject50.png", pool)
    draw_sheet(cells_bnd,
               "C | 20 BOUNDARY: same crop rebuilt at r = 0.575 (rows 1-5) and r = 0.825 "
               "(rows 6-10)", out / "probe_boundary20.png", pool)

    with open(out / "probe_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["board", "cell", "pool_id", "label", "reason", "crop_source",
                    "x0", "y0", "size", "note"])
        for board, cells in (("A_pass", cells_pass), ("B_reject", cells_rej), ("C_boundary", cells_bnd)):
            for k, c in enumerate(cells):
                x0, y0, size = c["crop"] if c["crop"] else ("", "", "")
                w.writerow([board, k, c["pool_id"], c["label"], c.get("reason", ""),
                            c.get("source", ""), x0, y0, size, c.get("note", "")])

    summary = {
        "pool_records": len(scan),
        "strata": {"no_detection": len(miss), "multiple": len(multi), "single_catdog": len(single),
                   "kept_after_clip_ocr": len(kept), "clip_style": len(clip), "ocr_text": len(ocr)},
        "geometry_replay": {"images_scanned": scanned,
                            "rejects_found": {f"{k[0]}:{k[1]}": len(v) for k, v in sorted(buckets.items())}},
        "cells": {"pass": len(cells_pass), "reject": len(cells_rej), "boundary": len(cells_bnd)},
        "geometry_quota_shortfalls": geom_short,
        "sampling": "even across the sorted pool-id span; one cell per pool image; seed not used",
        "geometry_sample_requested": args.geom_sample,
        "thresholds": {"min_sharpness": args.min_sharpness, "max_flat": args.max_flat,
                       "target_r": TARGET_R},
    }
    (out / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["cells"]), "-> ", out)


if __name__ == "__main__":
    main()
