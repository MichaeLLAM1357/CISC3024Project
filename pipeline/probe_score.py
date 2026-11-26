"""Score the Phase 0 human review sheet (results/probe/probe_index_HUMAN.csv).

    python pipeline/probe_score.py --index results/probe/probe_index_HUMAN.csv

This is arithmetic on the human's own labels -- it does not add, remove or reinterpret a
single verdict. It prints the k / k_cat / k_dog gate, the error-type table for §1, the
saturation check for §2, and the reject / boundary cross-checks, and writes the numbers to
results/probe/phase0_review.json so every figure in PHASE0_REVIEW.md has one command behind it.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", default="results/probe/probe_index_HUMAN.csv")
    ap.add_argument("--out", default="results/probe/phase0_review.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.index, encoding="utf-8")))
    A = [r for r in rows if r["board"] == "A_pass"]
    B = [r for r in rows if r["board"] == "B_reject"]
    C = [r for r in rows if r["board"] == "C_boundary"]

    # ---- §3 the gate: k over the 100 pass cells --------------------------------------
    good = [r for r in A if r["verdict"] == "good"]
    bad = [r for r in A if r["verdict"] == "bad"]
    k = len(good)
    n = len(A)
    lo, hi = wilson(k, n)
    per_class = {}
    for lab, name in (("0", "cat"), ("1", "dog")):
        cls = [r for r in A if r["label"] == lab]
        kc = sum(1 for r in cls if r["verdict"] == "good")
        per_class[name] = {"n": len(cls), "good": kc, "rate": kc / len(cls) if cls else 0.0}

    # ---- §1 error types over the pass cells ------------------------------------------
    reasons = defaultdict(list)
    for r in bad:
        reasons[r["verdict_reason"].strip()].append(int(r["cell"]))
    types = [{"type": t, "first_cell": min(v), "cells": len(v), "where": sorted(v)}
             for t, v in reasons.items()]
    types.sort(key=lambda d: -d["cells"])

    # ---- §2 saturation: a type that only ever shows up after cell 49 -------------------
    early, late = set(), set()
    for t, cells in reasons.items():
        (early if min(cells) < 50 else late).add(t)
    new_after_50 = sorted(late - early)
    late_types = {t: sorted(v) for t, v in reasons.items() if min(v) >= 50}

    # ---- B_reject: was throwing it away the right call? -------------------------------
    # reject_ok=yes -> the image really was unusable (reason says why)
    # reject_ok=no  -> nothing wrong with it: a false reject
    # the yolo_miss rows use verdict=bad + verdict_reason instead of reject_ok
    rej = defaultdict(Counter)
    for r in B:
        stage = r["reason"]
        if r["reject_ok"] == "yes":
            rej[stage]["correct"] += 1
        elif r["reject_ok"] == "no":
            rej[stage]["false_reject"] += 1
            rej[stage + ":false"][r["cell"]] += 1
        elif r["verdict"] == "bad":
            rej[stage]["correct"] += 1
        else:
            rej[stage]["unreadable"] += 1

    # ---- C_boundary: does the 0.575 / 0.825 edge still look acceptable? ---------------
    bnd = defaultdict(Counter)
    for r in C:
        band = "r=0.575" if float(r["note"].split("=")[1]) < 0.70 else "r=0.825"
        bnd[band][r["threshold_call"].split(",")[0].strip()] += 1

    summary = {
        "index": args.index,
        "gate": {"k": k, "n": n, "rate": k / n, "wilson95": [lo, hi],
                 "passes_0.70": k / n >= 0.70, "per_class": per_class},
        "error_types": types,
        "saturation": {"new_types_after_cell_50": new_after_50, "late_only": late_types},
        "reject_calls": {s: dict(c) for s, c in rej.items() if not s.endswith(":false")},
        "reject_false_cells": {s[: -len(":false")]: sorted(int(c) for c in c2)
                               for s, c2 in rej.items() if s.endswith(":false")},
        "boundary_calls": {b: dict(c) for b, c in bnd.items()},
        "rows": {"A_pass": len(A), "B_reject": len(B), "C_boundary": len(C)},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"§3  k={k}/{n} = {k / n:.2f}   Wilson95 [{lo:.3f}, {hi:.3f}]   "
          f"gate {'PASS' if k / n >= 0.70 else 'FAIL'} (>=0.70)")
    for name, d in per_class.items():
        print(f"    {name:3s} {d['good']:2d}/{d['n']} = {d['rate']:.2f}")
    print("§1  error types (count desc):")
    for d in types:
        print(f"    {d['type']:24s} first #{d['first_cell']:03d}  {d['cells']:2d} cells  {d['where']}")
    print("§2  types that appear only after cell 49:",
          new_after_50 if new_after_50 else "none", late_types)
    print("B   reject calls:", {s: dict(c) for s, c in rej.items() if not s.endswith(":false")})
    print("B   false-reject cells:", summary["reject_false_cells"])
    print("C   boundary calls:", summary["boundary_calls"])
    print("->", args.out)


if __name__ == "__main__":
    main()
