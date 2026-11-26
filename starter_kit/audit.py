"""Draw the random sample of your training data for the manual audit.

    python starter_kit/audit.py --manifest submission/manifest.csv --student-id 1234567

(from the project root, see README.md). The random draw depends only on your student ID and
the rows of your manifest: every row gets a score computed from your ID and the row itself, and the 200 rows (by default) with the lowest scores are drawn. So the
teaching team, running this on your submitted manifest, draws exactly the same rows and
re-labels some of them. Run it on your final manifest.
Writes into --out (default ./audit):

    audit.csv        one row per sampled image: fill in `verdict` (good / bad) and, for
                     bad ones, `reason` (e.g. cartoon, two heads, head too small, not upright)
    sheet_NN.png     the sampled images at 32x32 (what the model sees) and 96x96, numbered

Judge every image against the data specification in the handout, not against what your
pipeline intended.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catdog_eval.data import DEFAULT_DATA_ROOT, Pool, crop, pool_dir  # noqa: E402
from catdog_eval.manifest import check_manifest  # noqa: E402

FIELDS = ["index", "pool_id", "x0", "y0", "size", "angle", "label", "verdict", "reason"]
PER_SHEET = 50


def _score(student_id: str, row: dict) -> str:
    """Pseudo-random score of a row that depends only on the student ID and the row."""
    key = f"{row['pool_id']},{row['x0']},{row['y0']},{row['size']},{row['angle']},{row['label']}"
    return hashlib.sha256(f"cisc3024-audit:{student_id}:{key}".encode()).hexdigest()


def sample_rows(rows: list[dict], student_id: str, n: int) -> list[dict]:
    """The n rows with the lowest scores for this student ID (independent of row order)."""
    key = lambda r: (r["pool_id"], r["x0"], r["y0"], r["size"], r["angle"], r["label"])  # noqa: E731
    chosen = sorted(rows, key=lambda r: _score(student_id, r))[:n]
    return sorted(chosen, key=key)


def sheet(pool: Pool, rows: list[dict], start: int) -> Image.Image:
    cols, cell = 10, 140
    out = Image.new("RGB", (cols * cell, ((len(rows) + cols - 1) // cols) * cell), "white")
    draw = ImageDraw.Draw(out)
    for k, r in enumerate(rows):
        img = pool.image(r["pool_id"])
        big = crop(img, r["x0"], r["y0"], r["size"], r["angle"], res=96)
        small = crop(img, r["x0"], r["y0"], r["size"], r["angle"], res=32)
        x, y = (k % cols) * cell, (k // cols) * cell
        out.paste(big, (x + 2, y + 2))
        out.paste(small, (x + 102, y + 2))
        draw.text((x + 2, y + 100), f"#{start + k}  {'cat' if r['label'] == 0 else 'dog'}",
                  fill=(0, 0, 0))
        draw.text((x + 2, y + 114), r["pool_id"], fill=(90, 90, 90))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--student-id", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", default="audit")
    args = p.parse_args()

    pool = Pool(pool_dir(args.data_root))
    report = check_manifest(args.manifest, pool)
    if report["errors"]:
        raise SystemExit("fix the manifest first:\n  " + "\n  ".join(report["errors"]))
    rows = sample_rows(report["rows"], args.student_id, args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"index": i, **r, "verdict": "", "reason": ""})
    for s in range(0, len(rows), PER_SHEET):
        sheet(pool, rows[s:s + PER_SHEET], s).save(out / f"sheet_{s // PER_SHEET:02d}.png")
    print(f"{len(rows)} rows -> {out / 'audit.csv'} and {out}/sheet_*.png")


if __name__ == "__main__":
    main()
