# results/scan/ — the sampler scan, as the tools wrote it

Two kinds of file, both written by a script in the working tree (`D:\CISC3024\PROJECT`, which holds
the pool). This repository ships the outputs, not the pool. Nothing here was edited by hand: each
JSON is byte-identical to what the script wrote, md5 and line endings included, and `.gitattributes`
marks these files `-text` so that a checkout cannot rewrite them.

## 1. Evaluator output — 16 files

```bash
python starter_kit/evaluate.py --submission <dir> --n-per-class 1000 --seed 0 \
    --out results/scan/<name>_n1000.json
```

These are the official evaluator's own JSON, unmodified, so the schema is exactly the one
`evaluate.py` writes:

| key | what it holds |
|---|---|
| `submission` | the directory that was evaluated (an absolute path into the working tree) |
| `protocol` | `n_per_class`, `seed`, `batch_size`, `device`, `grading` |
| `checks` | `deterministic`, `seed_sensitive`, mixed and single-class batch accuracy |
| `training_data` | `check_manifest` errors and warnings, and the row statistics |
| `metrics` | accuracy, per-class accuracy, FID, KID and its std, precision, recall, memorization |
| `resources` | parameters, submission size, seconds per 1,000 images |
| `limits` | the fixed budgets |
| `violations`, `warnings`, `marks` | the evaluator's verdict, per metric and out of 15 |
| `environment` | torch, python, classifier sha256, reference sha256 |

`marks` is the evaluator's `assign_marks`, not a hand computation. Read one with
`python -c "import json,sys;print(json.load(open(sys.argv[1]))['marks'])" results/scan/<name>_n1000.json`.

## 2. Run summary — 8 files

`scan_posttrain.py` writes one summary per checkpoint: `protocol` (seed, batch size, checkpoint,
`ckpt_label`) plus `variants[]`, one entry per sampler variant with its settings, the path of the
evaluator JSON it produced, and its metrics. These are **not** evaluator output and do not share
its schema; they exist so that a variant name can be traced to the JSON behind it, and they are what
records `refused-to-overwrite` (see `AGENT_LOG.md` Check 4).

## What the numbers are

All 24 ran at `n_per_class = 1000`, seed 0, on 2,000 samples, against `data/dev/dev.npz`, with torch
2.6.0+cu124. Their `training_data.stats` says 2,000 rows, 1,000 per class: these scans predate the
3,000-row mixed tree, whose runs are in `results/ab/`. Every `submission` path points into the
working tree, not at this repository.
