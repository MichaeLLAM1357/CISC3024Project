"""Build the Phase 1.5 admission queue: the human reviews these sheets and marks human_ok=Y/N.

    python pipeline/review_sheet.py --manifest submission/manifest.csv \
        --stats results/manifest_v1/candidates_raw.jsonl --out results/review

Why this is not the audit. `starter_kit/audit.py` estimates a number from 200 rows; this tool
decides which rows exist. The layout is deliberately the same as audit.py (96 px crop + the 32 px
view the model gets, 10 cells per row, one cell per pool image) so the two image sets look alike,
but the two CSV formats are different on purpose and must not be confused:

    admission_NNN.csv : rank, pool_id, label, stratum, brightness, face_aspect, x0, y0, size, angle,
                        human_ok, reason        <- human_ok starts empty; the human writes "good"
                                                   (= admitted) and leaves the rest blank

Reading the blank cells (settled 2026-11-25, after 25 sheets): the human marks only the rows that
pass and leaves the rejects empty, in every wave -- marks are scattered over the whole 0..99 range
of each sheet, not a truncated prefix, and no sheet contains a single "N". So a blank row means
"reviewed, not admitted" and counts in the denominator; it never counts as a Y. A sheet with *no*
marks at all is the one exception: it carries no information, so it is excluded from the
denominator and named in the report instead of being scored as 100 rejects. Only a non-empty Y
cell can ever enter the manifest -- this reading changes a reported rate, never a row.

Queue order (fixed before the first verdict, never re-shuffled afterwards):

  1. nine strata per class = brightness tercile x face_aspect tercile, terciles computed from
     that class's own draft rows (pool-candidate statistics only -- nothing from data/dev/);
  2. inside a stratum, ascending sha256(f"{seed}:{pool_id}") with seed 0;
  3. strata are interleaved round-robin, so stopping early still leaves a balanced sample.

Ranking by det_conf, CLIP style, sharpness or any other score is forbidden: that is top-k
selection with extra steps. The only things that order the queue are a hash and two coarse bins.

Stopping rule lives in `--build-manifest`: it takes the first `--per-class` Y rows of each class
and refuses to write if a class is short, so a row the human never saw cannot enter the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))

from catdog_eval.data import DEFAULT_DATA_ROOT, Pool, crop, pool_dir  # noqa: E402

COLUMNS = ["rank", "pool_id", "label", "stratum", "brightness", "face_aspect",
           "x0", "y0", "size", "angle", "human_ok", "reason"]
PER_SHEET = 100
CELL = 140
COLS = 10


def hash_key(seed: int, pool_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pool_id}".encode()).hexdigest()


def terciles(values: list[float]) -> tuple[float, float]:
    """Two cut points -> three bins. Deterministic, no randomness."""
    q = statistics.quantiles(values, n=3)
    return q[0], q[1]


def bin_of(v: float, cuts: tuple[float, float]) -> int:
    return 0 if v <= cuts[0] else (1 if v <= cuts[1] else 2)


def load_stats(path: Path) -> dict[str, dict]:
    """pool_id -> {brightness, face_aspect, ...} from the candidate cache (no images needed)."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["pool_id"]] = r
    return out


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f)]


def read_candidates(path: Path) -> list[dict]:
    """The untrimmed candidates that passed *every* rule (default candidates_all.csv, 4,821 rows).

    Deliberately NOT candidates_raw.jsonl: that file holds the 8,230 rows that reached the quality
    pass, including the 3,409 that CLIP / OCR / dedup then rejected. Drawing the extension queue
    from it would quietly re-admit rows the pipeline already refused, which is the one thing the
    contract forbids ("widen the candidate set, never relax a rule").
    """
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append({"pool_id": r["pool_id"], "label": str(r["label"]),
                        "x0": r["x0"], "y0": r["y0"], "size": r["size"],
                        "angle": r.get("angle", 0)})
    return out


def wave_of(path: Path) -> int:
    m = re.search(r"_w(\d+)_", path.name)
    return int(m.group(1)) if m else 1


def seen_ids(review_dir: Path, before_wave: int) -> set[str]:
    """pool_ids on sheets of an earlier wave only.

    Excluding the current wave is what makes `--extend` idempotent: the queue is "candidates minus
    the draft minus everything already shown in previous waves", so re-running the same wave
    rewrites the same rows instead of sliding the window forward.
    """
    out: set[str] = set()
    for p in sorted(review_dir.glob("admission_*.csv")):
        if wave_of(p) >= before_wave:
            continue
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("pool_id") or "").strip():
                    out.add(r["pool_id"].strip())
    return out


def verdict_of(cell: str) -> str:
    """Wave 1 wrote 'good' for pass and left the rest blank; later waves write Y/N. Accept both."""
    v = (cell or "").strip().lower()
    if v in ("y", "good", "yes"):
        return "Y"
    if v in ("n", "bad", "no"):
        return "N"
    return ""


def build_queue_ext(draft: list[dict], all_rows: list[dict], stats: dict[str, dict],
                    label: int, seed: int, seen: set[str]) -> list[dict]:
    """Wave 2+: the untrimmed candidates, in the same bins and the same hash order.

    The tercile cuts come from the *draft* rows of this class (the same 1,000 the wave-1 queue was
    built on), so the bins do not move and the extension is a continuation of one queue rather than
    a new one. Rows already shown on any sheet are dropped; nothing else is re-ordered.
    """
    cuts_src = [r for r in draft if int(r["label"]) == label]
    b_cuts = terciles([stats[r["pool_id"]]["brightness"] for r in cuts_src])
    a_cuts = terciles([stats[r["pool_id"]]["face_aspect"] for r in cuts_src])
    shown = {r["pool_id"] for r in cuts_src} | seen

    ext = [r for r in all_rows if int(r["label"]) == label and r["pool_id"] not in shown]
    missing = [r["pool_id"] for r in ext if r["pool_id"] not in stats]
    if missing:
        raise SystemExit(f"{len(missing)} candidates have no stats (first: {missing[:3]})")

    strata: dict[int, list[dict]] = OrderedDict((s, []) for s in range(9))
    for r in ext:
        st = stats[r["pool_id"]]
        s = bin_of(st["brightness"], b_cuts) * 3 + bin_of(st["face_aspect"], a_cuts)
        strata[s].append({**r, "stratum": s,
                          "brightness": round(st["brightness"], 4),
                          "face_aspect": round(st["face_aspect"], 4)})
    for s in strata:
        strata[s].sort(key=lambda r: hash_key(seed, r["pool_id"]))

    order: list[dict] = []
    cursors = {s: 0 for s in strata}
    while True:
        progressed = False
        for s in strata:
            i = cursors[s]
            if i < len(strata[s]):
                order.append(strata[s][i])
                cursors[s] = i + 1
                progressed = True
        if not progressed:
            break
    for i, r in enumerate(order, 1 + len(cuts_src)):
        r["rank"] = i
    return order


def build_queue(rows: list[dict], stats: dict[str, dict], label: int, seed: int) -> list[dict]:
    """One class -> interleaved 9-stratum queue sorted by sha256(f'{seed}:{pool_id}')."""
    mine = [r for r in rows if int(r["label"]) == label]
    missing = [r["pool_id"] for r in mine if r["pool_id"] not in stats]
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest rows have no candidate stats (first: {missing[:3]}); "
            f"point --stats at the candidates_raw.jsonl that produced this manifest")

    b_cuts = terciles([stats[r["pool_id"]]["brightness"] for r in mine])
    a_cuts = terciles([stats[r["pool_id"]]["face_aspect"] for r in mine])

    strata: dict[int, list[dict]] = OrderedDict((s, []) for s in range(9))
    for r in mine:
        st = stats[r["pool_id"]]
        s = bin_of(st["brightness"], b_cuts) * 3 + bin_of(st["face_aspect"], a_cuts)
        strata[s].append({**r, "stratum": s,
                          "brightness": round(st["brightness"], 4),
                          "face_aspect": round(st["face_aspect"], 4)})
    for s in strata:
        strata[s].sort(key=lambda r: hash_key(seed, r["pool_id"]))

    order: list[dict] = []
    cursors = {s: 0 for s in strata}
    while True:
        progressed = False
        for s in strata:
            i = cursors[s]
            if i < len(strata[s]):
                order.append(strata[s][i])
                cursors[s] = i + 1
                progressed = True
        if not progressed:
            break
    for i, r in enumerate(order, 1):
        r["rank"] = i
    return order


def sheet(pool: Pool, rows: list[dict], start: int) -> Image.Image:
    """Same geometry as starter_kit/audit.py: big at +2, the model's view at +102."""
    n_rows = (len(rows) + COLS - 1) // COLS
    out = Image.new("RGB", (COLS * CELL, n_rows * CELL), "white")
    draw = ImageDraw.Draw(out)
    for k, r in enumerate(rows):
        img = pool.image(r["pool_id"])
        x0, y0, size = int(r["x0"]), int(r["y0"]), int(r["size"])
        big = crop(img, x0, y0, size, int(r["angle"]), res=96)
        small = crop(img, x0, y0, size, int(r["angle"]), res=32)
        x, y = (k % COLS) * CELL, (k // COLS) * CELL
        out.paste(big, (x + 2, y + 2))
        out.paste(small, (x + 102, y + 2))
        draw.text((x + 2, y + 100), f"{start + k}  {'cat' if int(r['label']) == 0 else 'dog'}",
                  fill=(0, 0, 0))
        draw.text((x + 2, y + 114), r["pool_id"], fill=(90, 90, 90))
    return out


def write_sheet(rows: list[dict], path_csv: Path) -> None:
    """Never overwrite a sheet that already carries a verdict."""
    if path_csv.exists():
        with open(path_csv, encoding="utf-8", newline="") as f:
            filled = any((r.get("human_ok") or "").strip() for r in csv.DictReader(f))
        if filled:
            raise SystemExit(f"{path_csv} already has human_ok values; refusing to overwrite")
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.setdefault("human_ok", "")
        r.setdefault("reason", "")
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="submission/manifest.csv",
                    help="the draft manifest: the candidate pool, not the dataset")
    ap.add_argument("--draft", default="results/draft_pre_review.csv",
                    help="frozen 2,000-row candidate pool the queue bins are computed from. Kept "
                         "separate from --manifest: pointing both at the admitted manifest makes the "
                         "tercile cuts and the 'already shown' set move, so the queue silently "
                         "re-shuffles and a second --merge admits a different 500 per class")
    ap.add_argument("--stats", default="results/manifest_v1/candidates_raw.jsonl")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default="results/review")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=500, help="admission target per class")
    ap.add_argument("--sheet-size", type=int, default=PER_SHEET)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="write submission/manifest.csv from the Y rows instead of making sheets")
    ap.add_argument("--extend", action="store_true",
                    help="wave 2+: draw the queue from the untrimmed candidates (candidates_all.csv, "
                         "everything that passed every rule) and skip rows already shown")
    ap.add_argument("--candidates", default="results/manifest_v1/candidates_all.csv")
    ap.add_argument("--sheets", default="",
                    help="with --extend: how many sheets per class, e.g. cat:2,dog:3")
    ap.add_argument("--wave", type=int, default=1, help="filename tag for wave >= 2")
    ap.add_argument("--report", action="store_true", help="print the verdict tallies and exit")
    ap.add_argument("--json", default="", help="with --report: also write the tallies here")
    args = ap.parse_args()

    out = Path(args.out)
    if args.report:
        report(args.per_class, out, Path(args.json) if args.json else None)
        return

    rows = read_manifest(Path(args.draft))
    if Path(args.draft).resolve() == Path(args.manifest).resolve() and args.merge:
        raise SystemExit("--draft and --manifest are the same file: --merge would rebuild the queue "
                         "from its own output. Point --draft at the frozen pre-review draft.")
    stats = load_stats(Path(args.stats))

    if args.merge:
        # Both waves in one queue order: draft ranks 1..1000, then the extension rows. A Y from
        # either wave can be admitted, and the order is still hash + bins, never a score.
        all_rows = read_candidates(Path(args.candidates))
        queues = {lab: build_queue(rows, stats, lab, args.seed)
                       + build_queue_ext(rows, all_rows, stats, lab, args.seed, set())
                  for lab in (0, 1)}
        merge(queues, args.per_class, Path(args.manifest), out)
        return

    if args.extend:
        want = {}
        for part in args.sheets.split(","):
            if ":" in part:
                k, v = part.split(":")
                want[k.strip()] = int(v)
        if not want:
            raise SystemExit("--extend needs --sheets cat:N,dog:M")
        all_rows = read_candidates(Path(args.candidates))
        seen = seen_ids(out, args.wave)
        queues = {lab: build_queue_ext(rows, all_rows, stats, lab, args.seed, seen)
                  for lab in (0, 1)}
        print(f"seen so far {len(seen)}; extension queue "
              f"cat {len(queues[0])} / dog {len(queues[1])}", flush=True)
        pool = None if args.no_png else Pool(pool_dir(args.data_root))
        for label in (0, 1):
            name = "cat" if label == 0 else "dog"
            n = want.get(name, 0)
            if not n:
                continue
            chunk_rows = queues[label][:n * args.sheet_size]
            for s in range(n):
                chunk = chunk_rows[s * args.sheet_size:(s + 1) * args.sheet_size]
                if not chunk:
                    print(f"{name}: queue exhausted after {s} sheets", flush=True)
                    break
                path_csv = out / f"admission_{name}_w{args.wave}_{s + 1:02d}.csv"
                write_sheet([dict(r) for r in chunk], path_csv)
                if pool is not None:
                    sheet(pool, chunk, chunk[0]["rank"]).save(
                        out / f"admission_{name}_w{args.wave}_{s + 1:02d}.png")
                print(f"  wrote {path_csv.name}  ranks {chunk[0]['rank']}-{chunk[-1]['rank']}",
                      flush=True)
        return

    queues = {0: build_queue(rows, stats, 0, args.seed), 1: build_queue(rows, stats, 1, args.seed)}

    pool = None if args.no_png else Pool(pool_dir(args.data_root))
    for label in (0, 1):
        q = queues[label]
        name = "cat" if label == 0 else "dog"
        n_sheets = (len(q) + args.sheet_size - 1) // args.sheet_size
        print(f"{name}: {len(q)} queued, {n_sheets} sheets of {args.sheet_size}, "
              f"target {args.per_class} Y", flush=True)
        for s in range(n_sheets):
            chunk = q[s * args.sheet_size:(s + 1) * args.sheet_size]
            path_csv = out / f"admission_{name}_{s + 1:02d}.csv"
            path_png = None if args.no_png else out / f"admission_{name}_{s + 1:02d}.png"
            write_sheet([dict(r) for r in chunk], path_csv)
            if path_png is not None:
                sheet(pool, chunk, s * args.sheet_size).save(path_png)
            print(f"  wrote {path_csv.name}"
                  f"{'' if path_png is None else ' + ' + path_png.name}"
                  f"  ranks {s * args.sheet_size}-{s * args.sheet_size + len(chunk) - 1}",
                  flush=True)


def merge(queues: dict[int, list[dict]], per_class: int, out_manifest: Path,
          review_dir: Path) -> None:
    """Turn the human's Y cells into the manifest, in fixed queue order, or refuse."""
    admitted: dict[int, list[dict]] = {}
    for label in (0, 1):
        name = "cat" if label == 0 else "dog"
        verdicts = load_verdicts(review_dir, name)
        shown = seen_ids(review_dir, before_wave=99)   # every sheet, every wave
        yes = [r for r in queues[label] if verdicts.get(r["pool_id"]) == "Y"]
        seen = sum(1 for r in queues[label] if r["pool_id"] in shown)
        admitted[label] = yes[:per_class]
        print(f"{name}: on sheets {seen}/{len(queues[label])}, Y {len(yes)}, "
              f"admitted {len(admitted[label])}/{per_class}", flush=True)
        if len(admitted[label]) < per_class:
            raise SystemExit(
                f"{name} short by {per_class - len(admitted[label])}: keep reviewing, or widen the "
                f"candidate set from the 18,640 single-detection images (never top up with unseen "
                f"rows, never relax a threshold)")

    fields = ["pool_id", "x0", "y0", "size", "angle", "label"]
    order = [r for label in (0, 1) for r in sorted(admitted[label], key=lambda r: r["rank"])]
    # lineterminator="\\n" on purpose: .gitattributes pins eol=lf, so a CRLF file would have a
    # different md5 in the working tree than in the blob and the documented hash would depend on
    # the platform. The md5 recorded everywhere is the LF one.
    with open(out_manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in order:
            w.writerow({k: r[k] for k in fields})
    print(f"wrote {out_manifest} with {len(order)} rows "
          f"(cat {len(admitted[0])}, dog {len(admitted[1])}) "
          f"md5 {hashlib.md5(out_manifest.read_bytes()).hexdigest()}", flush=True)


def load_verdicts(review_dir: Path, name: str) -> dict[str, str]:
    """pool_id -> 'Y' / 'N'. Only explicit marks are returned; blank rows are simply absent.

    This is the function `merge` uses, so an empty cell can never become an admitted row no matter
    how `report` reads the denominator. Duplicate pool_ids across waves keep the last sheet's value.
    """
    out: dict[str, str] = {}
    for p in sorted(review_dir.glob(f"admission_{name}_*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                pid = (r.get("pool_id") or "").strip()
                if not pid:
                    continue                 # the human's per-sheet tally row
                v = verdict_of(r.get("human_ok"))
                if v:
                    out[pid] = v
    return out


def report(per_class: int, review_dir: Path, json_path: Path | None = None) -> dict:
    """Count what has been marked so far, per class, without writing any manifest row.

    Denominator = every row on a sheet that carries at least one mark, because the sheets use
    "mark the good ones, leave the rejects empty" (see the module docstring). A fully empty sheet
    is excluded and listed -- scoring it as 100 rejects would be inventing 100 verdicts.
    `coverage` reports the last marked row index per sheet, so a half-finished sheet is visible
    rather than silently counted as reviewed.
    """
    stats: dict = {"per_class_target": per_class, "classes": {}, "unmarked_sheets": []}
    for label in (0, 1):
        name = "cat" if label == 0 else "dog"
        files = sorted(review_dir.glob(f"admission_{name}_*.csv"))
        good = total = 0
        coverage: list[dict] = []
        for p in files:
            with open(p, encoding="utf-8-sig", newline="") as f:
                rows = [r for r in csv.DictReader(f) if (r.get("pool_id") or "").strip()]
            marks = [i for i, r in enumerate(rows) if verdict_of(r.get("human_ok"))]
            if not marks:
                stats["unmarked_sheets"].append(p.name)
                continue
            good += sum(1 for i in marks if verdict_of(rows[i].get("human_ok")) == "Y")
            total += len(rows)          # blank row = reviewed, not admitted
            coverage.append({"sheet": p.name, "rows": len(rows),
                             "marks": len(marks), "last_mark_row": marks[-1] + 1})
        rate = good / total if total else 0.0
        need = max(0, per_class - good)
        est = int(need / rate) + 1 if rate else 0
        short = [c["sheet"] for c in coverage if c["last_mark_row"] < c["rows"]]
        stats["classes"][name] = {"sheets": len(files), "reviewed_rows": total, "admitted_Y": good,
                                  "admission_rate": round(rate, 4),
                                  "target": per_class, "need_more": need,
                                  "est_rows_needed": est, "coverage": coverage,
                                  "sheets_with_trailing_blank": short}
        print(f"{name}: sheets {len(files)}  reviewed {total}  good {good}  rate {rate:.3f}  "
              f"need {need} more  (~{est} rows at this rate)", flush=True)
        if short:
            print(f"  note: {len(short)} sheet(s) have blank rows after the last mark "
                  f"({', '.join(short[:3])}) -- counted as reviewed, check they were finished",
                  flush=True)
    if stats["unmarked_sheets"]:
        print(f"excluded {len(stats['unmarked_sheets'])} unmarked sheet(s): "
              f"{', '.join(stats['unmarked_sheets'])}", flush=True)
    if json_path is not None:
        json_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {json_path}", flush=True)
    return stats


if __name__ == "__main__":
    main()
