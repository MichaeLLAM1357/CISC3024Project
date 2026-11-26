"""Self-consistency audit of the two labelling sittings, plus an adjudication sheet.

Three label files exist and none of them is a ground truth:

    audit_legacy_1000row/audit.csv   200 cells drawn from the 1,000-row human manifest
    audit/audit.csv                  200 cells drawn from the 3,000-row mixed manifest
    results/mixed_v1/spotcheck_machine.csv   100 machine cells, drawn with a different salt

59 pool_ids appear in both audit files, because the legacy draw only ever sampled human rows and
the new draw put 59 of them back. Those 59 are the only place in the project where the same crop
was judged twice, so they are the only handle on how stable the standard is. This script reports
that agreement, prices both arms with Wilson intervals, and renders the disagreements as a
20-cell adjudication sheet with **no verdicts printed** -- a third measurement, not a reminder of
the first two.

    python pipeline/p10_consistency.py

Writes results/mixed_v1/label_stats.json, consistency_17.png, consistency_17.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_kit"))

from audit import FIELDS, sheet  # noqa: E402
from catdog_eval.data import DEFAULT_DATA_ROOT, Pool, pool_dir  # noqa: E402

OUT = ROOT / "results" / "mixed_v1"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tally(rows: list[dict]) -> dict:
    g = sum(1 for r in rows if r["verdict"].strip().lower() == "good")
    lo, hi = wilson(g, len(rows))
    return {"n": len(rows), "good": g, "bad": len(rows) - g,
            "good_rate": round(g / len(rows), 4) if rows else None,
            "wilson95": [round(lo, 4), round(hi, 4)] if rows else None}


def main() -> None:
    new = load(ROOT / "audit" / "audit.csv")
    old = load(ROOT / "audit_legacy_1000row" / "audit.csv")
    spot = load(OUT / "spotcheck_machine.csv")
    human = {r["pool_id"] for r in load(OUT / "manifest_human_1000.csv")}

    by_arm = {"human": [r for r in new if r["pool_id"] in human],
              "machine": [r for r in new if r["pool_id"] not in human]}
    stats: dict = {
        "audit_mixed": tally(new),
        "audit_mixed_by_arm": {k: tally(v) for k, v in by_arm.items()},
        "audit_mixed_by_class": {c: tally([r for r in new if r["label"] == c])
                                 for c in ("0", "1")},
        "audit_legacy_1000row": tally(old),
        "spotcheck_machine": tally(spot),
        "reason_counts_new": dict(Counter(r["reason"] for r in new if r["reason"].strip())
                                  .most_common()),
        "reason_counts_machine_arm": dict(Counter(r["reason"] for r in by_arm["machine"]
                                                  if r["reason"].strip()).most_common()),
        "bad_rows_without_reason": [r["pool_id"] for r in new
                                    if r["verdict"].strip().lower() == "bad"
                                    and not r["reason"].strip()],
        "spotcheck_reason_filled": sum(1 for r in spot if r["reason"].strip()),
        "spotcheck_machine_by_class": {c: tally([r for r in spot if r["label"] == c])
                                       for c in ("0", "1")},
        "spotcheck_reason_counts": dict(Counter(r["reason"].strip() for r in spot
                                                if r["reason"].strip()).most_common()),
        "spotcheck_bad_without_reason": [r["pool_id"] for r in spot
                                         if r["verdict"].strip().lower() == "bad"
                                         and not r["reason"].strip()],
        "spotcheck_blank_verdict": [r["pool_id"] for r in spot if not r["verdict"].strip()],
    }

    # ---- the 59 crops judged twice ----
    o = {r["pool_id"]: r for r in old}
    n = {r["pool_id"]: r for r in new}
    both = [p for p in n if p in o]
    agree = sum(1 for p in both if o[p]["verdict"] == n[p]["verdict"])
    stats["repeat_frame"] = {
        "n": len(both),
        "agree": agree,
        "agreement": round(agree / len(both), 4),
        "cohen_kappa": None,
        "flips": {"bad_to_good": sum(1 for p in both if o[p]["verdict"] == "bad"
                                     and n[p]["verdict"] == "good"),
                  "good_to_bad": sum(1 for p in both if o[p]["verdict"] == "good"
                                     and n[p]["verdict"] == "bad")},
        "pass1_good_rate": round(sum(1 for p in both if o[p]["verdict"] == "good")
                                 / len(both), 4),
        "pass2_good_rate": round(sum(1 for p in both if n[p]["verdict"] == "good")
                                 / len(both), 4),
    }
    # Cohen's kappa on the repeat frame
    po = agree / len(both)
    pe = 0.0
    for v in ("good", "bad"):
        pe += (sum(1 for p in both if o[p]["verdict"] == v) / len(both)) * \
              (sum(1 for p in both if n[p]["verdict"] == v) / len(both))
    stats["repeat_frame"]["cohen_kappa"] = round((po - pe) / (1 - pe), 4) if pe < 1 else None

    # ---- adjudication sheet: disagreements only, no verdicts printed ----
    bad = sorted(p for p in both if o[p]["verdict"] != n[p]["verdict"])
    rows = []
    for i, p in enumerate(bad):
        r = n[p]
        rows.append({"pool_id": p, "x0": int(r["x0"]), "y0": int(r["y0"]),
                     "size": int(r["size"]), "angle": float(r["angle"]),
                     "label": int(r["label"])})
    adj_csv = OUT / "consistency_17.csv"
    adj_png = OUT / "consistency_17.png"
    if adj_png.exists() is False or not adj_csv.exists():
        pool = Pool(pool_dir(DEFAULT_DATA_ROOT))
        sheet(pool, rows, 0).save(adj_png)
    # never overwrite a sheet that already carries the third pass
    filled = False
    if adj_csv.exists():
        with open(adj_csv, newline="", encoding="utf-8") as f:
            filled = any(r.get("verdict", "").strip() for r in csv.DictReader(f))
    if not filled:
        with open(adj_csv, "w", newline="\n", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS, lineterminator="\n")
            w.writeheader()
            for i, r in enumerate(rows):
                w.writerow({"index": i, **r, "verdict": "", "reason": ""})
    stats["adjudication"] = {"cells": len(rows), "png": "consistency_17.png",
                             "csv": "consistency_17.csv", "verdicts_printed": False,
                             "filled": filled}

    # ---- third pass, if the human has filled the adjudication sheet ----
    if filled:
        p3 = {r["pool_id"]: r for r in load(adj_csv)}
        n3 = len(p3)
        a1 = sum(1 for p in p3 if o[p]["verdict"] == p3[p]["verdict"])
        a2 = sum(1 for p in p3 if n[p]["verdict"] == p3[p]["verdict"])
        stats["adjudication"].update({
            "n": n3,
            "pass3_good": sum(1 for r in p3.values() if r["verdict"] == "good"),
            "sides_with_pass1": a1,
            "sides_with_pass2": a2,
            "reason_filled": sum(1 for r in p3.values() if r["reason"].strip()),
            "pass3_reasons": dict(Counter(r["reason"] for r in p3.values()
                                          if r["reason"].strip()).most_common()),
        })

    # ---- how much of the manifest a human actually saw ----
    mixed = load(OUT / "manifest_mixed.csv")
    hum_rows = [r for r in mixed if r["pool_id"] in human]
    stats["composition"] = {
        "manifest_rows": len(mixed),
        "human_looked_at": len(hum_rows),
        "machine_only": len(mixed) - len(hum_rows),
        "human_share": round(len(hum_rows) / len(mixed), 4),
        "machine_share": round((len(mixed) - len(hum_rows)) / len(mixed), 4),
    }

    # ---- the machine frame, measured twice: two salts, two sittings ----
    mach = by_arm["machine"]
    amap = {r["pool_id"]: r for r in mach}
    shared = [r for r in spot if r["pool_id"] in amap]
    n1, g1 = len(mach), sum(1 for r in mach if r["verdict"].strip().lower() == "good")
    n2 = len(spot)
    g2 = stats["spotcheck_machine"]["good"]
    se = math.sqrt(g1 / n1 * (1 - g1 / n1) / n1 + g2 / n2 * (1 - g2 / n2) / n2)
    lo, hi = wilson(g1 + g2, n1 + n2)
    stats["machine_frame_two_draws"] = {
        "audit_machine": {"n": n1, "good": g1, "good_rate": round(g1 / n1, 4)},
        "spotcheck_machine": {"n": n2, "good": g2, "good_rate": round(g2 / n2, 4)},
        "pooled": {"n": n1 + n2, "good": g1 + g2, "good_rate": round((g1 + g2) / (n1 + n2), 4),
                   "wilson95": [round(lo, 4), round(hi, 4)]},
        "difference": round(g2 / n2 - g1 / n1, 4),
        "se_difference": round(se, 4),
        "z": round((g2 / n2 - g1 / n1) / se, 2),
        "shared_pool_ids": len(shared),
        "shared_agreement": sum(1 for r in shared
                                if r["verdict"].strip() == amap[r["pool_id"]]["verdict"]),
    }

    with open(OUT / "label_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in stats.items() if k.startswith(("audit_mixed", "spotcheck",
                                                                     "repeat", "machine_frame"))},
                     indent=2))
    print(f"\nadjudication: {len(rows)} cells -> {OUT / 'consistency_17.png'}")


if __name__ == "__main__":
    main()
