"""Build the mixed manifest for PROJECT4: 500 human-admitted + 1,000 machine-gated rows per class.

Provenance, so the file can be argued about later:

    human arm   1,000 rows, `human_ok=Y` on a phase-1.5 sheet (unchanged copy of
                PROJECT3/submission/manifest.csv, md5 66f60cfb)
    auto arm    2,000 rows, never shown to a human, drawn from `candidates_all.csv` -- the 4,821
                rows that already passed every machine gate (YOLO single detection, head geometry,
                CLIP photo-vs-art, OCR text, phash dedup). No threshold is moved here; this is the
                pool the 1,000-row quota had no room for.

Both arms are real crops of real pool images, coordinates copied verbatim from the pipeline that
produced `candidates_all.csv`. Nothing is generated, edited or downloaded.

Rows the human looked at are excluded from the auto arm even when they were left blank (blank on
these sheets means "looked at, not admitted"). The only exception is `admission_cat_w2_02.csv`,
which no one ever opened -- those 100 rows are unseen by construction and are allowed in.

Queue order for the auto arm is `sha256("0:" + pool_id)`, the same key the admission queue used, so
this is not a score ranking. The cat side has only 5 rows of slack (1,005 available, 1,000 needed);
the script refuses to write if either class is short.
"""

from __future__ import annotations

import os

import argparse
import csv
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "mixed_v1"
UNOPENED = "admission_cat_w2_02.csv"
MANIFEST_COLUMNS = ("pool_id", "x0", "y0", "size", "angle", "label")
HUMAN_MD5 = "66f60cfbfaf7f33550e2cfac11bb7d82"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("pool_id")]


def sha_key(pool_id: str, seed: int = 0) -> str:
    return hashlib.sha256(f"{seed}:{pool_id}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="results/mixed_v1/manifest_human_1000.csv")
    ap.add_argument("--candidates", default="results/manifest_v1/candidates_all.csv")
    ap.add_argument("--review-dir", default="results/review")
    ap.add_argument("--out", default="results/mixed_v1/manifest_mixed.csv")
    ap.add_argument("--per-class-human", type=int, default=500)
    ap.add_argument("--per-class-auto", type=int, default=1000)
    ap.add_argument("--data-root", default=os.environ.get("CATDOG_DATA", str(ROOT / "data")))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    human_path = ROOT / args.human
    if md5(human_path) != HUMAN_MD5:
        raise SystemExit(f"{human_path} md5 {md5(human_path)} != {HUMAN_MD5} -- wrong manifest")
    human = read_csv(human_path)
    for r in human:
        r["label"] = int(r["label"])
    human_ids = {r["pool_id"] for r in human}

    cand = read_csv(ROOT / args.candidates)
    cand_by_id = {r["pool_id"]: r for r in cand}

    # pool_ids the human looked at = everything on a sheet except the sheet nobody opened
    shown: Counter = Counter()
    looked: set[str] = set()
    for f in sorted(glob.glob(str(ROOT / args.review_dir / "admission_*.csv"))):
        for r in read_csv(Path(f)):
            shown[r["pool_id"]] += 1
            if Path(f).name != UNOPENED:
                looked.add(r["pool_id"])

    stats = {
        "human_manifest": {"path": args.human, "md5": md5(human_path), "rows": len(human)},
        "candidates_all": {"path": args.candidates, "rows": len(cand),
                           "per_class": dict(Counter(int(r["label"]) for r in cand))},
        "sheets": {"shown_unique": len(shown), "looked_at": len(looked),
                   "unopened_sheet_rows": len(read_csv(ROOT / args.review_dir / UNOPENED))},
        "auto_pool": {}, "auto_taken": {}, "headroom": {},
    }

    auto_rows: list[dict] = []
    for cls in (0, 1):
        pool = [r for r in cand
                if int(r["label"]) == cls
                and r["pool_id"] not in looked
                and r["pool_id"] not in human_ids]
        pool.sort(key=lambda r: sha_key(r["pool_id"]))
        stats["auto_pool"][cls] = len(pool)
        if len(pool) < args.per_class_auto:
            raise SystemExit(f"class {cls}: auto pool {len(pool)} < {args.per_class_auto} needed")
        stats["headroom"][cls] = len(pool) - args.per_class_auto
        take = pool[: args.per_class_auto]
        stats["auto_taken"][cls] = len(take)
        auto_rows += take
        n_human = sum(1 for r in human if r["label"] == cls)
        if n_human != args.per_class_human:
            raise SystemExit(f"class {cls}: human rows {n_human} != {args.per_class_human}")

    rows = human + [{c: r[c] for c in MANIFEST_COLUMNS} for r in auto_rows]
    rows.sort(key=lambda r: (int(r["label"]), r["pool_id"]))

    # invariants before anything touches disk
    ids = [r["pool_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate pool_id"
    assert len(rows) == 2 * (args.per_class_human + args.per_class_auto), "row count"
    assert Counter(int(r["label"]) for r in rows) == {0: args.per_class_human + args.per_class_auto,
                                                     1: args.per_class_human + args.per_class_auto}
    for r in rows:
        c = cand_by_id.get(r["pool_id"])
        assert c is not None, f"{r['pool_id']} not in candidates_all"
        assert tuple(str(c[k]) for k in MANIFEST_COLUMNS) == tuple(str(r[k]) for k in MANIFEST_COLUMNS), \
            f"{r['pool_id']} differs from candidates_all"
    for r in auto_rows:
        assert r["pool_id"] not in looked, f"{r['pool_id']} was looked at"
        assert r["pool_id"] not in human_ids, f"{r['pool_id']} already human-admitted"

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in MANIFEST_COLUMNS})
    stats["manifest"] = {"path": args.out, "md5": md5(out), "rows": len(rows),
                         "per_class": dict(Counter(int(r["label"]) for r in rows)),
                         "byte_size": out.stat().st_size}
    print(f"wrote {args.out}  {len(rows)} rows  md5 {stats['manifest']['md5']}")
    print(f"  per class {stats['manifest']['per_class']}  "
          f"(human {args.per_class_human} + auto {args.per_class_auto})")
    print(f"  auto pool {stats['auto_pool']}  headroom {stats['headroom']}")

    if not args.no_check:
        import sys
        sys.path.insert(0, str(ROOT / "starter_kit"))
        from catdog_eval.data import Pool
        from catdog_eval.manifest import check_manifest
        pool = Pool(Path(args.data_root) / "student")
        rep = check_manifest(out, pool)
        stats["check_manifest"] = {"rows": len(rep["rows"]), "errors": rep["errors"],
                                   "warnings": rep["warnings"],
                                   "stats": rep.get("stats", {})}
        print(f"  check_manifest: {len(rep['rows'])} rows, {len(rep['errors'])} errors, "
              f"{len(rep['warnings'])} warnings")
        for e in rep["errors"][:5]:
            print(f"    ERROR {e}")
        for w_ in rep["warnings"][:5]:
            print(f"    WARN  {w_}")
        if rep["errors"]:
            raise SystemExit("manifest has errors -- not usable")

    (OUT / "mixed_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"  stats -> results/mixed_v1/mixed_stats.json")


if __name__ == "__main__":
    main()
