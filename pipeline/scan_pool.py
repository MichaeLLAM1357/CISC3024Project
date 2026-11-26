"""Full-pool YOLO scan with a resumable JSONL cache.

This is the first (cheapest) stage of the curation pipeline.  It runs the
pretrained YOLO11x detector on every pool image, keeps only COCO cat (15) and
dog (16) boxes, and appends one JSON object per image to a JSONL file.  Re-running
the same command resumes after the last completed pool_id.

The detector is used for data curation only; it is never part of the generative
model and never enters submission/.

Usage:
    python -u pipeline/scan_pool.py
    python -u pipeline/scan_pool.py --limit 64 --out results/scan_test
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from catdog_eval.data import Pool, pool_dir  # noqa: E402

COCO_CAT, COCO_DOG = 15, 16


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave one partial line; ignore it and it will be redone.
                continue
            if "pool_id" in rec:
                done.add(rec["pool_id"])
    return done


def predict_one(model: YOLO, images: list, args) -> list:
    kwargs = dict(
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=[COCO_CAT, COCO_DOG],
        max_det=args.max_det,
        verbose=False,
        device=args.device,
    )
    if args.half:
        kwargs["half"] = True
    return model.predict(images, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--out", default="results/scan")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--half", action="store_true", help="FP16 inference on CUDA")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N ids")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "yolo_detections.jsonl"
    log_path = out / "yolo_scan.log"
    meta_path = out / "yolo_scan_meta.json"

    def log(msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    pool = Pool(pool_dir(args.data_root))
    ids = list(pool.index)
    if args.limit:
        ids = ids[: args.limit]
    done = load_done(jsonl)
    todo = [pid for pid in ids if pid not in done]
    log(f"pool={len(ids)} already_done={len(done)} todo={len(todo)}")
    log(f"model={args.model} imgsz={args.imgsz} conf={args.conf} batch={args.batch} "
        f"device={args.device} half={args.half}")

    if not todo:
        log("nothing to do")
        return

    meta = {
        "started_utc": utc_now(),
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": __import__("ultralytics").__version__,
        "pool_size": len(ids),
        "already_done": len(done),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    model = YOLO(args.model)
    n_written = 0
    n_error = 0
    n_any = 0
    n_cat = 0
    n_dog = 0
    n_both = 0
    t0 = time.perf_counter()

    with open(jsonl, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            images, valid_ids = [], []
            for pid in chunk:
                try:
                    images.append(pool.image(pid).convert("RGB"))
                    valid_ids.append(pid)
                except Exception as exc:  # noqa: BLE001 - a single bad pool image must not stop the scan
                    rec = {"pool_id": pid, "error": f"{type(exc).__name__}: {exc}"}
                    f.write(json.dumps(rec) + "\n")
                    n_error += 1
                    n_written += 1
            if not images:
                f.flush()
                continue

            try:
                results = predict_one(model, images, args)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                log(f"CUDA OOM on batch at {start}; retrying one image at a time")
                torch.cuda.empty_cache()
                results = []
                for img in images:
                    results.extend(predict_one(model, [img], args))

            for pid, r in zip(valid_ids, results):
                dets = []
                for box in r.boxes:
                    dets.append({
                        "cls": int(box.cls.item()),
                        "conf": float(box.conf.item()),
                        "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                    })
                rec = {"pool_id": pid, "detections": dets}
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                if dets:
                    n_any += 1
                    has_cat = any(d["cls"] == COCO_CAT for d in dets)
                    has_dog = any(d["cls"] == COCO_DOG for d in dets)
                    n_cat += has_cat
                    n_dog += has_dog
                    n_both += has_cat and has_dog
            f.flush()

            elapsed = time.perf_counter() - t0
            rate = n_written / elapsed if elapsed > 0 else 0.0
            remaining = len(todo) - (start + len(chunk))
            eta_s = remaining / rate if rate > 0 else 0.0
            log(f"{min(start + len(chunk), len(todo))}/{len(todo)} "
                f"written={n_written} any={n_any} cat={n_cat} dog={n_dog} both={n_both} "
                f"errors={n_error} rate={rate:.1f}/s eta={eta_s/3600:.2f}h")

    elapsed = time.perf_counter() - t0
    meta.update({
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "written_this_run": n_written,
        "errors_this_run": n_error,
        "any_this_run": n_any,
        "cat_this_run": n_cat,
        "dog_this_run": n_dog,
        "both_this_run": n_both,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"DONE written={n_written} errors={n_error} any={n_any} cat={n_cat} dog={n_dog} "
        f"both={n_both} elapsed={elapsed/3600:.2f}h -> {jsonl}")


if __name__ == "__main__":
    main()
