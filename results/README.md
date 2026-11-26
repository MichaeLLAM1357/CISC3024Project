# results/ — which file is the shipped model

**The graded numbers are `ab/eval_mixed60k_n1000.json` (seed 0) and
`ab/eval_mixed60k_n1000_seed1.json` (seed 1).** Everything else in this folder is the record of a
run that did not ship, kept so the report's comparisons can be re-read.

| file | what it measures | marks |
|---|---|---|
| `ab/eval_mixed60k_n1000.json` | **the submission**: `submission/weights.pt` (md5 `1e5c9b87`), official evaluator, 1,000 per class, seed 0 | **13/15** |
| `ab/eval_mixed60k_n1000_seed1.json` | same weights, seed 1 — the stability check behind the report's "12–13" | 12/15 |
| `ab/eval_mixed20k_n1000.json` | 20,000-step arm of the same 3,000-row tree | 11/15 |
| `ab/eval_admitted_e20k_n1000.json` | 1,000-row tree, 20,000 steps — the baseline the mixed arm was bought against | 11/15 |
| `legacy_1000row/eval_final.json` | the 1,000-row tree, 20,000 steps, checkpoint `32a70a7e` (superseded 2026-11-25); its `submission` field still points at `D:\CISC3024\PROJECT3\submission` | 11/15 |

**Two things to know before quoting `legacy_1000row/eval_final.json`.** (a) It is not the
submission: 0.8400 accuracy (cat 0.776 / dog 0.904), FID 92.55, precision 0.4580, recall 0.4646,
11/15, on a 2,000-image run — against 0.9065 / 77.80 / 0.5044 in `ab/eval_mixed60k_n1000.json`.
(b) `AGENTS.md`'s historical section also quotes **0.8425 / FID 89.71 / recall 0.5409** for the
earlier `db6fbc69` checkpoint of the same tree; that run's JSON lived in `PROJECT3/results/scan/`
and is not in this repo. Match on the numbers, not on the filename.

## Where the other numbers live

- `mixed_v1/audit.csv` — the 200 cells, 59 human-arm and 141 machine-arm rows.
- `mixed_v1/spotcheck_machine.csv` — the second, independent 100-cell machine draw.
- `mixed_v1/label_stats.json` — every rate in `DATA_CARD.md`, recomputed from the two sheets above.
- `mixed_v1/LABEL_AUDIT.md` — why 0.574 and 0.740 are the same frame, and how they pool to 0.643.
- `review/admission_*.csv` — the 25 admission sheets, one `human_ok` verdict per row.
- `manifest_v1/` — the 1,000-row admitted manifest and its stats, before the mixed arm existed.
- `draft_pre_review.csv` — the frozen 2,000-row draft the admission queue was drawn from.
- `scan/*_n1000.json` — the sampler scan: 16 unmodified `evaluate.py` outputs and 8
  `scan_posttrain.py` run summaries, all at 1,000 per class on the 2,000-row tree.
  `scan/README.md` says which schema each file has.
