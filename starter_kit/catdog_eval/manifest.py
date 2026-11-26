"""Check a training-data manifest against the raw pool.

manifest.csv has one row per training image:

    pool_id,x0,y0,size,angle,label
    s000123,41,17,96,0,0

  pool_id  image in the raw pool
  x0, y0   top-left corner of the square, in pixels of the pool image (integers)
  size     side of the square in pixels (integer)
  angle    rotation in degrees, counter-clockwise, about the square's centre (0 = none)
  label    0 = cat, 1 = dog

The training image is the square cut from the (rotated) pool image and resized to 32 x 32
(catdog_eval.data.crop). Errors make the manifest invalid; warnings are reported for the
data part of the grade.
"""

import csv
import math
from collections import Counter
from pathlib import Path

from .data import Pool

COLUMNS = ("pool_id", "x0", "y0", "size", "angle", "label")
# A head of at least 32 px taking at most 80% of the square needs a square of 40 px.
MIN_SIZE = 40
MAX_ANGLE = 45.0
MIN_PER_CLASS = 500
MAX_CLASS_RATIO = 2.0  # larger class / smaller class
DUPLICATE_IOU = 0.7


def _square_corners(x0, y0, size, angle):
    cx, cy, h = x0 + size / 2, y0 + size / 2, size / 2
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    # PIL rotates the image counter-clockwise; the square cut afterwards corresponds to
    # the square rotated clockwise in the original image (y points down).
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c)
            for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]


def _iou(a, b) -> float:
    ax, ay, asz = a
    bx, by, bsz = b
    ix = max(0, min(ax + asz, bx + bsz) - max(ax, bx))
    iy = max(0, min(ay + asz, by + bsz) - max(ay, by))
    inter = ix * iy
    return inter / (asz * asz + bsz * bsz - inter)


def check_manifest(path: str | Path, pool: Pool, max_errors: int = 20) -> dict:
    """Parse and check a manifest. Returns {"rows", "errors", "warnings", "stats"}."""
    path = Path(path)
    errors, warnings, rows = [], [], []
    if not path.exists():
        return {"rows": [], "errors": [f"{path} not found"], "warnings": [], "stats": {}}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            return {"rows": [], "errors": [f"missing columns {missing}; expected {list(COLUMNS)}"],
                    "warnings": [], "stats": {}}
        raw = list(reader)

    def bad(i: int, msg: str) -> None:
        if len(errors) < max_errors:
            errors.append(f"line {i + 2}: {msg}")
        elif len(errors) == max_errors:
            errors.append("... (further errors not shown)")

    for i, r in enumerate(raw):
        try:
            row = {"pool_id": r["pool_id"].strip(), "x0": int(r["x0"]), "y0": int(r["y0"]),
                   "size": int(r["size"]), "angle": float(r["angle"] or 0),
                   "label": int(r["label"])}
        except (TypeError, ValueError):
            bad(i, f"cannot parse {dict(r)} (x0, y0, size, label must be integers)")
            continue
        if row["pool_id"] not in pool:
            bad(i, f"pool_id {row['pool_id']!r} is not in the raw pool")
            continue
        if row["label"] not in (0, 1):
            bad(i, f"label must be 0 (cat) or 1 (dog), got {row['label']}")
            continue
        if row["size"] < MIN_SIZE:
            bad(i, f"size {row['size']} < {MIN_SIZE}: the head would be smaller than 32 px")
            continue
        if not -MAX_ANGLE <= row["angle"] <= MAX_ANGLE:
            bad(i, f"angle {row['angle']} outside [-{MAX_ANGLE:g}, {MAX_ANGLE:g}]")
            continue
        w, h = pool.size(row["pool_id"])
        corners = _square_corners(row["x0"], row["y0"], row["size"], row["angle"])
        if any(x < -0.5 or y < -0.5 or x > w + 0.5 or y > h + 0.5 for x, y in corners):
            bad(i, f"square ({row['x0']}, {row['y0']}, {row['size']}, angle {row['angle']:g}) "
                   f"is not inside the {w}x{h} image {row['pool_id']}")
            continue
        rows.append(row)

    exact = Counter((r["pool_id"], r["x0"], r["y0"], r["size"], r["angle"]) for r in rows)
    repeated = sum(n - 1 for n in exact.values() if n > 1)
    if repeated:
        errors.append(f"{repeated} rows repeat another row exactly")
    by_image: dict = {}
    for r in rows:
        by_image.setdefault(r["pool_id"], []).append(r)
    overlapping = 0
    for crops in by_image.values():
        for a in range(len(crops)):
            for b in range(a + 1, len(crops)):
                boxes = [(c["x0"], c["y0"], c["size"]) for c in (crops[a], crops[b])]
                overlapping += _iou(*boxes) > DUPLICATE_IOU
    if overlapping:
        warnings.append(f"{overlapping} pairs of crops from the same image overlap by more than "
                        f"IoU {DUPLICATE_IOU} (near-duplicate training images)")

    per_class = Counter(r["label"] for r in rows)
    for label, name in ((0, "cat"), (1, "dog")):
        if per_class[label] < MIN_PER_CLASS:
            warnings.append(f"only {per_class[label]} {name} images (expected at least "
                            f"{MIN_PER_CLASS})")
    if min(per_class[0], per_class[1]) > 0 and \
            max(per_class.values()) / min(per_class[0], per_class[1]) > MAX_CLASS_RATIO:
        warnings.append(f"classes are unbalanced: {per_class[0]} cats vs {per_class[1]} dogs")
    stats = {"rows": len(raw), "valid_rows": len(rows), "cat": per_class[0],
             "dog": per_class[1], "distinct_pool_images": len(by_image),
             "rotated": sum(1 for r in rows if r["angle"])}
    return {"rows": rows, "errors": errors, "warnings": warnings, "stats": stats}
