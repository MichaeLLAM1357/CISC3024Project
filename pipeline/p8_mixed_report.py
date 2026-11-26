"""Collect the mixed-manifest arms into one table. Reads only evaluator JSON; nothing estimated."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AB = ROOT / "results" / "ab"
OUT = ROOT / "results" / "mixed_v1"

ARMS = [
    ("admitted_e20k", "1,000 rows, 500/class human", "eval_admitted_e20k_n1000.json", 20000),
    ("mixed20k", "3,000 rows, 1,500/class", "eval_mixed20k_n1000.json", 20000),
    ("mixed60k", "3,000 rows, 1,500/class", "eval_mixed60k_n1000.json", 60000),
]

HEAD = ("| arm | rows | steps | epochs | acc | FID | KID | prec | recall | mem | marks |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n")


def main() -> None:
    rows, table = [], HEAD
    for name, what, fname, steps in ARMS:
        p = AB / fname
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        m, n = r["metrics"], r["training_data"]["stats"]["rows"]
        epochs = steps * 64 // n
        table += (f"| {name} | {n:,} | {steps:,} | {epochs:,} | {m['accuracy']:.4f} | "
                  f"{m['fid']:.2f} | {m['kid']:.4f} | {m['precision']:.4f} | "
                  f"{m['recall']:.4f} | {m['memorization_rate']:.3f} | "
                  f"{r['marks']['total']}/15 |\n")
        rows.append({"arm": name, "what": what, "rows": n, "steps": steps, "epochs": epochs,
                     "metrics": m, "marks": r["marks"], "violations": r["violations"],
                     "nn_ratio_median": m.get("nn_ratio_median"),
                     "training_images_deduplicated": m.get("training_images_deduplicated"),
                     "source": fname})
    print(table)
    (OUT / "summary_mixed_n1000.json").write_text(
        json.dumps({"arms": rows}, indent=2) + "\n", encoding="utf-8")
    (OUT / "TABLE.md").write_text(table, encoding="utf-8")
    print(f"wrote results/mixed_v1/summary_mixed_n1000.json and TABLE.md ({len(rows)} arms)")


if __name__ == "__main__":
    main()
