"""Build submission/manifest.csv from the full-pool YOLO scan.

Pipeline (all curation models are pretrained and used only for data curation):

    YOLO scan cache
      -> exactly one cat/dog detection
      -> class-specific face localizer
      -> fixed face-box -> head-box geometry (NOT fitted to dev)
      -> three crop invariants
      -> CLIP photo-vs-art / photo-vs-print exclusion (no top-k)
      -> fixed OCR / sharpness / flatness checks
      -> perceptual-hash deduplication
      -> stratified class balance
      -> manifest.csv

The fixed geometry is derived from the specification, not from per-image dev fitting:
    head width   w = face_width  * (1 + 2 * 0.10)
    head height  v = face_height * (1 + 0.25 + 0.05)          (ears above, chin below)
    long side    h = max(w, v)                                (brief §7 definition)
    target       h / size = 0.70                              (middle of 60-80%)
    size = max(h / 0.70, 40)
The 1.10/1.25 margins are fixed anatomical margins; the 0.70 target is the middle of the
handout's 60-80% range.  Dev is used only for an aggregate sanity check, never to fit these
constants.

Usage:
    python pipeline/build_manifest.py --scan results/scan/yolo_detections.jsonl \
        --out submission/manifest.csv
    python pipeline/build_manifest.py --debug-limit 500 --out-dir results/manifest_debug
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

COCO_CAT, COCO_DOG = 15, 16

# Fixed geometry (spec-derived; not dev-fitted).  The face localizer returns the face box
# (roughly brow to chin).  The head box adds a fixed side margin, a fixed ear margin above
# and a small chin margin below.  These are anatomical framing constants checked visually
# on pool candidates, not fitted to dev.
HEAD_SIDE = 0.10
HEAD_TOP = 0.25
HEAD_BOTTOM = 0.05
TARGET_R = 0.70
MAX_W_RATIO = 0.85
MIN_HEAD_W = 32.0
MIN_SIZE = 40
MAX_HEAD_ASPECT = 1.4
CONTEXT = 0.15

FACE_MODELS = {
    0: ("hugocornellier/cat-face-landmarks", "cat_face_localizer.tflite"),
    1: ("hugocornellier/dog-face-landmarks", "dog_face_localizer.tflite"),
}

# Brief §8: only these two positive prompts. Words such as breed, coat colour, lighting,
# frontal, close-up, studio or white background narrow the appearance and hurt recall, so
# they must not appear here.
POS_PROMPTS = [
    "a photograph of a real cat head",
    "a photo of a real dog's head",
]
ART_NEG_PROMPTS = [
    "a cartoon of an animal",
    "a drawing of an animal",
    "a painting of an animal",
    "a 3D render of an animal",
    "a toy animal",
    "a statue of an animal",
    "a sketch of an animal",
    "an embroidery of an animal",
]
PRINT_NEG_PROMPTS = [
    "an animal printed on a product",
    "a book cover with an animal",
    "a poster of an animal",
    "a framed picture of an animal",
    "a magazine cover with an animal",
    "a calendar with an animal",
    "a screenshot of an animal",
    "text or a watermark",
]


def sha_key(seed: int, pool_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pool_id}".encode()).hexdigest()


def load_localizer(repo: str, filename: str):
    path = hf_hub_download(repo, filename)
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


def predict_face(model, im: Image.Image):
    """Return (x1, y1, x2, y2) in the coordinates of `im`, or None."""
    interp, inp, outp = model
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
    box = [(x1 * size - px) / s, (y1 * size - py) / s,
           (x2 * size - px) / s, (y2 * size - py) / s]
    if box[2] - box[0] <= 1 or box[3] - box[1] <= 1:
        return None
    return box


def fixed_head_box(face, image_size):
    """Map a face box to a fixed head box and a square crop.

    Returns (dict, None) on success or (None, reason) on rejection.
    """
    fx1, fy1, fx2, fy2 = face
    fw, fh = fx2 - fx1, fy2 - fy1
    if fw <= 0 or fh <= 0:
        return None, "face_failed"

    hx1 = fx1 - HEAD_SIDE * fw
    hx2 = fx2 + HEAD_SIDE * fw
    hy1 = fy1 - HEAD_TOP * fh
    hy2 = fy2 + HEAD_BOTTOM * fh
    w, box_h = hx2 - hx1, hy2 - hy1
    # Brief §7: `h` is the head bounding-box LONG side (ears included), not the vertical
    # extent.  Wide, front-facing faces (many dogs, some cats) have w > vertical height;
    # using the vertical extent would under-size the square and then fail the 0.65-0.75
    # ratio check, silently throwing away legal crops.  See results/manifest_debug stats:
    # 41/600 rows were rejected as "ratio" before this fix.
    h = max(w, box_h)
    if w < MIN_HEAD_W:
        return None, "head_too_small"
    if h / w > MAX_HEAD_ASPECT:
        return None, "aspect"

    size = max(h / TARGET_R, MIN_SIZE)
    img_w, img_h = image_size
    side = int(round(size))
    if side > min(img_w, img_h):
        side = min(img_w, img_h)
    if side < MIN_SIZE:
        return None, "image_too_small"

    r = h / side
    if not (0.65 <= r <= 0.75):
        return None, "ratio"
    if w / side > MAX_W_RATIO:
        return None, "width"

    cx = (hx1 + hx2) / 2.0
    cy = (hy1 + hy2) / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    sx = 0
    sy = 0
    if x0 < 0:
        sx = -x0
        x0 = 0
    elif x0 + side > img_w:
        sx = x0 + side - img_w
        x0 = img_w - side
    if y0 < 0:
        sy = -y0
        y0 = 0
    elif y0 + side > img_h:
        sy = y0 + side - img_h
        y0 = img_h - side
    if max(sx, sy) > max(8, int(0.10 * side)):
        return None, "border"

    return {
        "x0": x0, "y0": y0, "size": side,
        "w": w, "h": h, "box_h": box_h, "r": r,
        "face_w": fw, "face_h": fh,
        "face_aspect": fh / fw,
        "face": [fx1, fy1, fx2, fy2],
        "head": [hx1, hy1, hx2, hy2],
    }, None


def photo_stats(crop: Image.Image) -> dict:
    im = crop.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
    a = np.asarray(im).astype(np.uint8)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    sharp = float(lap.var())
    edges = float(cv2.Canny(g, 50, 150).mean() / 255.0)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean() / 255.0)
    sat_std = float(hsv[..., 1].std() / 255.0)
    med = cv2.medianBlur(g, 5).astype(np.float32)
    noise = float(np.abs(g.astype(np.float32) - med).mean())
    mu = cv2.blur(g.astype(np.float32), (5, 5))
    mu2 = cv2.blur(g.astype(np.float32) ** 2, (5, 5))
    std = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    flat = float((std < 3).mean())
    return {"sharp": sharp, "edge": edges, "sat": sat, "sat_std": sat_std,
            "noise": noise, "flat": flat, "brightness": float(g.mean() / 255.0)}


def phash_int(im: Image.Image) -> int:
    return int(str(imagehash.phash(im)), 16)


class PhashIndex:
    """Exact Hamming-radius index over 64-bit perceptual hashes.

    Splitting 64 bits into 5 chunks means that any pair at distance <= 4 shares at least
    one chunk (pigeonhole), so only same-chunk pairs need a full popcount.
    """

    CHUNKS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))

    def __init__(self, max_dist: int):
        self.max_dist = max_dist
        self.buckets = [defaultdict(list) for _ in self.CHUNKS]

    def add_if_new(self, h: int) -> bool:
        for (shift, width), bucket in zip(self.CHUNKS, self.buckets):
            key = (h >> shift) & ((1 << width) - 1)
            for other in bucket.get(key, ()):
                if (h ^ other).bit_count() <= self.max_dist:
                    return False
        for (shift, width), bucket in zip(self.CHUNKS, self.buckets):
            key = (h >> shift) & ((1 << width) - 1)
            bucket[key].append(h)
        return True


class ClipStyle:
    """Two independent photo-vs-art and photo-vs-print binary prompts.

    Each group compares the mean positive similarity with the mean negative similarity,
    so adding print prompts does not dilute the art decision.  The reported style score is
    the minimum of the two; a crop must look like a photo against both groups.
    """

    def __init__(self, device: str):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device)
        self.model.eval()
        tok = open_clip.get_tokenizer("ViT-L-14")
        with torch.no_grad():
            pos = self.model.encode_text(tok(POS_PROMPTS).to(device))
            pos /= pos.norm(dim=-1, keepdim=True)
            self.groups = {}
            for name, prompts in (("art", ART_NEG_PROMPTS), ("print", PRINT_NEG_PROMPTS)):
                neg = self.model.encode_text(tok(prompts).to(device))
                neg /= neg.norm(dim=-1, keepdim=True)
                self.groups[name] = (pos, neg)

    @torch.no_grad()
    def score(self, crops: list[Image.Image]) -> list[dict]:
        out = []
        for i in range(0, len(crops), 32):
            batch = torch.stack([self.preprocess(c) for c in crops[i:i + 32]]).to(self.device)
            f = self.model.encode_image(batch)
            f /= f.norm(dim=-1, keepdim=True)
            scores = {}
            for name, (pos, neg) in self.groups.items():
                ps = (f @ pos.T).mean(1)
                ns = (f @ neg.T).mean(1)
                scores[name] = torch.softmax(
                    torch.stack([ps, ns], dim=-1) * 100.0, dim=-1)[:, 0]
            for k in range(f.shape[0]):
                out.append({name: float(v[k].cpu()) for name, v in scores.items()})
        return out


class TextFilter:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()

    def score(self, crop: Image.Image) -> tuple[float, int, str]:
        res, _ = self.ocr(np.asarray(crop.convert("RGB")))
        if not res:
            return 0.0, 0, ""
        area = 0.0
        chars = 0
        texts = []
        for box, text, conf in res:
            if conf is not None and conf < 0.5:
                continue
            pts = np.asarray(box, dtype=np.float32)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
            chars += len(str(text))
            texts.append(str(text))
        frac = float(area) / float(crop.width * crop.height)
        return frac, int(chars), " ".join(texts)[:120]


def stratified_sample(records: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Deterministic stratified sample over brightness and face aspect.

    This is not a top-k score ranking; it spreads the selection over generic pool-candidate
    statistics so that different breeds, coats and lighting survive.
    """
    if len(records) <= n:
        return list(records)
    bright = np.array([r["brightness"] for r in records], dtype=np.float64)
    aspect = np.array([r["face_aspect"] for r in records], dtype=np.float64)
    b_edges = np.quantile(bright, [0.0, 0.25, 0.5, 0.75, 1.0])
    a_edges = np.quantile(aspect, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r, b, a in zip(records, bright, aspect):
        bi = int(np.searchsorted(b_edges, b, side="right") - 1)
        ai = int(np.searchsorted(a_edges, a, side="right") - 1)
        bi = min(max(bi, 0), 3)
        ai = min(max(ai, 0), 3)
        bins[(bi, ai)].append(r)
    for key in bins:
        bins[key].sort(key=lambda r: sha_key(seed, r["pool_id"]))

    total = len(records)
    alloc = {k: int(math.floor(n * len(v) / total)) for k, v in bins.items()}
    left = n - sum(alloc.values())
    order = sorted(bins, key=lambda k: (-len(bins[k]), k))
    for k in order:
        if left <= 0:
            break
        if alloc[k] < len(bins[k]):
            alloc[k] += 1
            left -= 1
    chosen = []
    for k in sorted(bins):
        chosen.extend(bins[k][:alloc[k]])
    if len(chosen) < n:
        used = {r["pool_id"] for r in chosen}
        rest = [r for r in records if r["pool_id"] not in used]
        rest.sort(key=lambda r: sha_key(seed + 1, r["pool_id"]))
        chosen.extend(rest[:n - len(chosen)])
    return chosen[:n]


def montage(records: list[dict], pool: Pool, path: Path, title_key: str = "reason",
            limit: int = 48) -> None:
    if not records:
        return
    cell, cols = 170, 8
    rows = (min(len(records), limit) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for k, r in enumerate(records[:limit]):
        im = pool.image(r["pool_id"]).convert("RGB")
        if "crop" in r:
            x, y, s = r["crop"]
            im = im.crop((x, y, x + s, y + s))
        im = im.resize((96, 96), Image.Resampling.LANCZOS)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(im, (x + 2, y + 2))
        draw.text((x + 2, y + 100), r["pool_id"], fill=(0, 0, 0))
        # Reject records carry no label (stage-1 rejects have no class yet, stage-2 rejects are
        # dropped before the label matters), so print the class only when it is actually known.
        # An earlier version read `r.get('label') == 0` and defaulted every unknown row to
        # "dog", which made the whole reject grid look like a pile of mislabelled dogs.
        lab = r.get("label")
        prefix = "" if lab is None else ("cat " if lab == 0 else "dog ")
        line = f"{prefix}{r.get(title_key, '')}".strip()
        draw.text((x + 2, y + 114), line[:24], fill=(0, 0, 0))
        if "style" in r:
            draw.text((x + 2, y + 128), f"s{r['style']:.2f} sh{r.get('sharp', 0):.0f}",
                      fill=(0, 0, 0))
        if "ocr_text" in r and r["ocr_text"]:
            draw.text((x + 2, y + 142), r["ocr_text"][:22], fill=(0, 0, 0))
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="results/scan/yolo_detections.jsonl")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="submission/manifest.csv")
    ap.add_argument("--out-dir", default="results/manifest_debug")
    ap.add_argument("--debug-limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-style", type=float, default=0.50)
    ap.add_argument("--min-sharpness", type=float, default=20.0)
    ap.add_argument("--max-flat", type=float, default=0.75)
    ap.add_argument("--ocr-max-area", type=float, default=0.005)
    ap.add_argument("--ocr-min-chars", type=int, default=3)
    ap.add_argument("--ocr-workers", type=int, default=6)
    ap.add_argument("--ocr-max-style", type=float, default=0.75,
                    help="run OCR only when the CLIP style score is below this")
    ap.add_argument("--phash-distance", type=int, default=4)
    ap.add_argument("--keep-per-class", type=int, default=1000)
    ap.add_argument("--max-candidates", type=int, default=1300)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="reuse results/<out-dir>/candidates_raw.jsonl and stage1_cursor.json")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--stage1-only", action="store_true",
                    help="stop after the geometry pass; run again without it to filter and write")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = Pool(pool_dir(args.data_root))

    records = []
    with open(args.scan, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
            if args.debug_limit and len(records) >= args.debug_limit:
                break
    print(f"scan records: {len(records)}")

    # Resumable stage 1: candidates and the record cursor are appended as we go, so a hang,
    # a crash or a reboot costs at most the current 200-record block.
    cache = out_dir / "candidates_raw.jsonl"
    cursor_file = out_dir / "stage1_cursor.json"
    candidates = []
    reject_counts = Counter()
    start = 0
    if args.resume and cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        if cursor_file.exists():
            state = json.loads(cursor_file.read_text())
            start = int(state.get("next_index", 0))
            reject_counts.update(state.get("reject_counts", {}))
        start = min(start, len(records))
        print(f"resume: {start} records done, {len(candidates)} candidates cached", flush=True)
    elif cache.exists():
        cache.unlink()
    cache_f = open(cache, "a", encoding="utf-8")

    localizers = {c: load_localizer(*FACE_MODELS[c]) for c in (0, 1)}
    clip = None if args.skip_clip else ClipStyle(args.device)
    text = None if args.skip_ocr else TextFilter()

    rejects = []
    t0 = time.perf_counter()

    def reject(pool_id, reason, **extra):
        reject_counts[reason] += 1
        if len(rejects) < 2000:
            rejects.append({"pool_id": pool_id, "reason": reason, **extra})

    for i, rec in enumerate(records):
        if i < start:
            continue
        pid = rec["pool_id"]
        try:
            dets = rec.get("detections", [])
            if len(dets) == 0:
                reject(pid, "no_detection")
                continue
            if len(dets) != 1:
                reject(pid, "multiple_detections")
                continue
            det = dets[0]
            if det["cls"] not in (COCO_CAT, COCO_DOG):
                reject(pid, "bad_class")
                continue
            label = 0 if det["cls"] == COCO_CAT else 1
            try:
                im = pool.image(pid).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                reject(pid, "load_error", detail=str(exc)[:80])
                continue
            w, h = im.size
            x1, y1, x2, y2 = det["xyxy"]
            bw, bh = x2 - x1, y2 - y1
            cx1 = max(0, int(x1 - CONTEXT * bw))
            cy1 = max(0, int(y1 - CONTEXT * bh))
            cx2 = min(w, int(x2 + CONTEXT * bw))
            cy2 = min(h, int(y2 + CONTEXT * bh))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                reject(pid, "face_failed")
                continue
            sub = im.crop((cx1, cy1, cx2, cy2))
            fb = predict_face(localizers[label], sub)
            if fb is None:
                reject(pid, "face_failed")
                continue
            fb = [fb[0] + cx1, fb[1] + cy1, fb[2] + cx1, fb[3] + cy1]
            geom, reason = fixed_head_box(fb, (w, h))
            if geom is None:
                reject(pid, reason)
                continue

            x0, y0, side = geom["x0"], geom["y0"], geom["size"]
            crop = im.crop((x0, y0, x0 + side, y0 + side))
            stats = photo_stats(crop)
            if stats["sharp"] < args.min_sharpness:
                reject(pid, "blurry", **stats, crop=(x0, y0, side))
                continue
            if stats["flat"] > args.max_flat:
                reject(pid, "flat", **stats, crop=(x0, y0, side))
                continue

            rec_out = {
                "pool_id": pid, "label": label, "crop": (x0, y0, side),
                "x0": x0, "y0": y0, "size": side, "angle": 0,
                "det_conf": float(det["conf"]), **geom, **stats,
            }
            candidates.append(rec_out)
            cache_f.write(json.dumps(rec_out) + "\n")
        except Exception as exc:  # noqa: BLE001 - one bad image must not kill the run
            reject(pid, "exception", detail=f"{type(exc).__name__}: {exc}"[:120])

        if (i + 1) % 200 == 0:
            cache_f.flush()
            cursor_file.write_text(json.dumps({"next_index": i + 1}))
            dt = time.perf_counter() - t0
            print(f"{i+1}/{len(records)} candidates={len(candidates)} "
                  f"rejects={sum(reject_counts.values())} rate={(i+1-start)/dt:.1f}/s",
                  flush=True)
    cache_f.close()
    cursor_file.write_text(json.dumps({"next_index": len(records),
                                       "reject_counts": dict(reject_counts)}))
    if args.stage1_only:
        print(f"stage 1 only: {len(candidates)} candidates cached in {cache}", flush=True)
        return

    # Quality pass: one disk read per candidate.  CLIP is batched; OCR runs per crop.  Every
    # decision is appended to quality_raw.jsonl, so a restart resumes instead of re-running OCR.
    def crop_of(r):
        im = pool.image(r["pool_id"]).convert("RGB")
        return im.crop((r["x0"], r["y0"], r["x0"] + r["size"], r["y0"] + r["size"]))

    qcache = out_dir / "quality_raw.jsonl"
    decisions: dict[str, dict] = {}
    if args.resume and qcache.exists():
        with open(qcache, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    decisions[d["pool_id"]] = d
        print(f"resume quality: {len(decisions)} crops already scored", flush=True)
    elif qcache.exists():
        qcache.unlink()
    qf = open(qcache, "a", encoding="utf-8")

    hashes = PhashIndex(args.phash_distance) if args.phash_distance >= 0 else None
    kept: list[dict] = []
    q_counts: Counter = Counter()

    def record(r, keep, reason, **extra):
        d = {"pool_id": r["pool_id"], "keep": keep, "reason": reason,
             "crop": [r["x0"], r["y0"], r["size"]], **extra}
        qf.write(json.dumps(d) + "\n")
        decisions[r["pool_id"]] = d
        return d

    def apply(r, d):
        if d["keep"]:
            for key in ("style", "style_art", "style_print", "ocr_area", "ocr_chars", "ocr_text"):
                if key in d:
                    r[key] = d[key]
            if hashes is not None and "ph" in d:
                hashes.add_if_new(d["ph"])
            kept.append(r)
        else:
            q_counts[d["reason"]] += 1
            if len(rejects) < 2000:
                rejects.append({"pool_id": r["pool_id"], "label": r["label"], "reason": d["reason"],
                                "crop": tuple(d["crop"])})

    pending = []
    for r in candidates:
        d = decisions.get(r["pool_id"])
        if d is not None:
            apply(r, d)
        else:
            pending.append(r)

    if candidates:
        print(f"quality pass: {len(pending)} new of {len(candidates)} crops", flush=True)
    ocr_pool = ThreadPoolExecutor(max_workers=args.ocr_workers)
    for start_i in range(0, len(pending), args.batch):
        batch = pending[start_i:start_i + args.batch]
        crops = [crop_of(r) for r in batch]
        scores = clip.score(crops) if clip is not None else None

        provisional = []  # (index, record_fields, keep, reason)
        for i, (r, c) in enumerate(zip(batch, crops)):
            extra: dict = {}
            keep, reason = True, "kept"
            if scores is not None:
                s = scores[i]
                extra.update(style=min(s["art"], s["print"]), style_art=s["art"],
                             style_print=s["print"])
                if extra["style"] < args.min_style:
                    keep, reason = False, "clip_style"
            if keep and hashes is not None:
                ph = phash_int(c)
                if hashes.add_if_new(ph):
                    extra["ph"] = ph
                else:
                    keep, reason = False, "duplicate"
            provisional.append((i, extra, keep, reason))

        # OCR is the slowest filter: cascade it only on crops CLIP was unsure about, and run
        # those in parallel (onnxruntime releases the GIL).
        ocr_jobs = [(i, crops[i]) for i, extra, keep, _ in provisional
                    if keep and text is not None and extra.get("style", 0.0) < args.ocr_max_style]
        ocr_out = dict(zip([i for i, _ in ocr_jobs],
                           ocr_pool.map(lambda job: text.score(job[1]), ocr_jobs))) if ocr_jobs else {}

        for i, extra, keep, reason in provisional:
            if keep and i in ocr_out:
                frac, chars, txt = ocr_out[i]
                extra.update(ocr_area=frac, ocr_chars=chars, ocr_text=txt)
                if chars >= args.ocr_min_chars and frac > args.ocr_max_area:
                    keep, reason = False, "ocr_text"
            apply(batch[i], record(batch[i], keep, reason, **extra))
        qf.flush()
        print(f"quality {start_i + len(batch)}/{len(pending)} kept={len(kept)} "
              f"ocr={len(ocr_jobs)}", flush=True)
    ocr_pool.shutdown()
    qf.close()
    reject_counts.update(q_counts)
    candidates = kept

    by_class = defaultdict(list)
    for r in candidates:
        by_class[r["label"]].append(r)
    print("candidates/class:", {("cat" if k == 0 else "dog"): len(v)
                                for k, v in sorted(by_class.items())})

    selected = []
    for label in (0, 1):
        group = by_class[label]
        if not args.no_trim:
            if len(group) > args.max_candidates:
                group = stratified_sample(group, args.max_candidates, seed=100 + label)
            if len(group) > args.keep_per_class:
                group = stratified_sample(group, args.keep_per_class, seed=200 + label)
        selected.extend(group)
    selected.sort(key=lambda r: (r["label"], r["pool_id"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("pool_id,x0,y0,size,angle,label\n")
        for r in selected:
            f.write(f"{r['pool_id']},{r['x0']},{r['y0']},{r['size']},0,{r['label']}\n")
    print(f"wrote {len(selected)} rows -> {out}")

    with open(out_dir / "candidates_all.csv", "w", newline="", encoding="utf-8") as f:
        f.write("pool_id,x0,y0,size,angle,label,style,style_art,style_print,sharp,flat\n")
        for r in sorted(candidates, key=lambda r: (r["label"], r["pool_id"])):
            f.write(f"{r['pool_id']},{r['x0']},{r['y0']},{r['size']},0,{r['label']},"
                    f"{r.get('style', -1):.4f},{r.get('style_art', -1):.4f},"
                    f"{r.get('style_print', -1):.4f},{r['sharp']:.1f},{r['flat']:.3f}\n")

    stats = {
        "scan_records": len(records),
        "candidates_before_trim": {str(k): len(v) for k, v in by_class.items()},
        "selected": {str(k): sum(1 for r in selected if r["label"] == k) for k in (0, 1)},
        "reject_counts": dict(reject_counts),
        "args": vars(args),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (out_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    # Diagnostic grids sample across the whole run, not the first 48 in insertion order: the
    # candidate grid used to be `selected[:48]`, which is 48 cats with the smallest pool_ids and
    # says nothing about the dog half, and the reject grid only showed the earliest rejects.
    def spread(rows: list[dict], k: int = 48) -> list[dict]:
        if len(rows) <= k:
            return rows
        step = len(rows) / k
        return [rows[min(int(i * step), len(rows) - 1)] for i in range(k)]

    demo = spread([r for r in selected if r["label"] == 0], 24) + \
        spread([r for r in selected if r["label"] == 1], 24)
    montage(demo, pool, out_dir / "candidate_grid.png", title_key="label")
    montage(spread(rejects), pool, out_dir / "reject_grid.png", title_key="reason")
    print(f"stats -> {out_dir/'manifest_stats.json'}")
    print(f"grids -> {out_dir/'candidate_grid.png'}, {out_dir/'reject_grid.png'}")


if __name__ == "__main__":
    main()
