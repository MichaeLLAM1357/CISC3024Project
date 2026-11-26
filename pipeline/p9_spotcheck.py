"""100-cell spot check of the machine arm -- for the human to calibrate their own ruler.

Same layout, same crop convention and same deterministic draw rule as `starter_kit/audit.py`
(imported, not re-implemented): 50 cells per sheet, 96 px crop + the 32 px view the model gets.

The draw is `_score(salt, row)` with salt = "<student-id>-spotcheck", so it is fixed before the
first verdict and cannot be re-rolled after seeing results. Rows are the 2,000 machine-gated ones
that never appeared on an admission sheet; 50 per class.

    python pipeline/p9_spotcheck.py --student-id UC325629

Writes results/mixed_v1/spotcheck_machine.csv + spotcheck_machine_0{0,1}.png
Fill `verdict` with good / bad and `reason` for bad ones, same vocabulary as the audit.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_kit"))

from audit import FIELDS, PER_SHEET, _score, sheet  # noqa: E402
from catdog_eval.data import DEFAULT_DATA_ROOT, Pool, pool_dir  # noqa: E402

OUT = ROOT / "results" / "mixed_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", default="UC325629")
    ap.add_argument("--manifest", default="results/mixed_v1/manifest_mixed.csv")
    ap.add_argument("--human", default="results/mixed_v1/manifest_human_1000.csv")
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = ap.parse_args()

    with open(ROOT / args.manifest, newline="", encoding="utf-8") as f:
        rows = [{"pool_id": r["pool_id"], "x0": int(r["x0"]), "y0": int(r["y0"]),
                 "size": int(r["size"]), "angle": float(r["angle"]), "label": int(r["label"])}
                for r in csv.DictReader(f)]
    with open(ROOT / args.human, newline="", encoding="utf-8") as f:
        human = {r["pool_id"] for r in csv.DictReader(f)}

    salt = f"{args.student_id}-spotcheck"
    picked: list[dict] = []
    for cls in (0, 1):
        pool = [r for r in rows if r["label"] == cls and r["pool_id"] not in human]
        pool.sort(key=lambda r: _score(salt, r))
        picked += pool[: args.per_class]
    picked.sort(key=lambda r: (int(r["label"]), r["pool_id"]))

    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "spotcheck_machine.csv"
    with open(csv_path, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(picked):
            w.writerow({"index": i, "pool_id": r["pool_id"], "x0": r["x0"], "y0": r["y0"],
                        "size": r["size"], "angle": r["angle"], "label": r["label"],
                        "verdict": "", "reason": ""})

    pool = Pool(pool_dir(args.data_root))
    for s in range(0, len(picked), PER_SHEET):
        img = sheet(pool, picked[s:s + PER_SHEET], s)
        p = out_dir / f"spotcheck_machine_{s // PER_SHEET:02d}.png"
        img.save(p)
        print(f"{p.relative_to(ROOT)}  {img.size[0]}x{img.size[1]}  cells {s}-{s + len(picked[s:s + PER_SHEET]) - 1}")
    print(f"{len(picked)} rows -> {csv_path.relative_to(ROOT)}  (verdict column empty, 50/class)")
    print("vocabulary: not_upright, two_heads, text, blurry, product_print, cartoon, ears_cut, "
          "wrong_animal, statue_toy, other")


if __name__ == "__main__":
    main()
