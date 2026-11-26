# Class-conditional 32×32 cat/dog head generation

**CISC3024 — LAM KA WAI (UC325629)**

A denoising diffusion model trained from scratch that draws a 32×32 colour photograph of a cat's
head or a dog's head from a single label (0 = cat, 1 = dog) and a seed. The training set was built
from a 76,250-image unlabelled web pool with no labels at all: a detector, a face localiser, a CLIP
style filter and an OCR text filter reduce it to 4,821 candidates, a person reads 2,400 of them and
admits 500 per class, and a further 1,000 per class are selected by those same rules with no human
in the loop. The shipped manifest has 3,000 rows.

Every number in the report and in these documents is read from a JSON or log file under `results/`,
and the two files the marker rebuilds from — `submission/manifest.csv` and `audit/audit.csv` — are
byte-identical to the ones that were graded.

## Headline numbers

Official evaluator, 1,000 images per class, seed 0, against the local dev reference
(`results/ab/eval_mixed60k_n1000.json`):

| Metric | Value | Tier | Marks | Resource | Value |
|---|---|---|---|---|---|
| Accuracy | **0.9065** | Excellent | 5 | Parameters | 5,145,731 |
| FID | **77.80** | Pass | 3 | Submission size | 19.74 MB |
| KID (×10³) | 20.90 | — | — | Hardware | 1 × RTX 2060 (6 GB) |
| Precision | 0.5243 | — | — | Training time | 5,507 s (0.0918 s/step) |
| Recall | **0.5044** | Excellent | 5 | Sampling speed | 61.17 s / 1,000 images |
| Memorization | 0.0000 | — | — | Steps | 60,000 (1,280 passes) |
| | | | **13 / 15** | | |

Accuracy sits 0.0065 above the 0.90 line, so a second seed is shipped as evidence rather than a
footnote: at seed 1 the same recipe gives accuracy 0.8955, FID 76.53 and recall 0.4812, which is
**12/15**. The honest statement of the result is **12–13 of 15**; FID and recall move by less than
the measurement noise between seeds, accuracy is the one metric on a boundary.

## Data in one paragraph

Stages 0–5 are rules (YOLO11x single-detection scan → head geometry → CLIP photo-vs-print →
OCR text/watermark → pHash dedup), stage 6 is one human verdict per row, stage 7 is the rule stack
run again on the candidates the human quota had no room for. That gives 1,000 human-admitted rows
and 2,000 rule-selected rows. The 200-cell audit drawn from the shipped manifest measures quality
at **0.670** [0.602, 0.731], but the two arms are not the same population: **0.898** on rows a person
approved against **0.574** on rows a rule selected (0.643 once a second 100-cell draw is pooled).
Two thirds of the training set has never been seen by a human, and roughly a third of it would fail
the specification. That trade is why the model scores 13/15, and it is stated rather than hidden.

## Layout

```
submission/                 the graded model: generate.py, model.py, unet.py, diffusion.py,
                            manifest.csv (3,000 rows) and weights.pt (EMA at step 60,000)
audit/audit.csv             200 human verdicts on crops drawn by the unmodified audit.py
audit_legacy_1000row/       the earlier 200-cell audit on the 1,000-row manifest (0.740)
pipeline/                   curation (scan → manifest), mixed-arm builder (p6), arm runner (p7),
                            audit and spot-check tooling (p9, p10), probes kept as evidence
train.py, train_diffusion.py  training entry points (DCGAN baseline, DDPM delivered model)
tests/                      11 tests: crop parity with the starter kit, determinism, batch
                            independence, overfit-single-batch, generate() contract
starter_kit/                course evaluator and audit tool, unmodified, used for every number
results/                    evaluator JSON per arm, label statistics, review sheets;
                            results/README.md says which JSON is the submission (13/15)
PDF/UC325629_REPORT.pdf     the report: a cover page, six pages of body, references and two appendices;
                            the cover carries the link to this repository
DATA_CARD.md                one page: pipeline counts, compliance with its interval, error kinds
LIMITS.md, AGENTS.md,        the rules the agent worked under (every "LIMITS.md §n" in the code
AGENT_LOG.md                points here), and the two defects that created the rules
docs/                       method, data, experiments
```

## Documents

| File | What is in it |
|---|---|
| [`docs/METHOD.md`](docs/METHOD.md) | the model as implemented: schedule, loss, UNet, sampler, guidance, determinism, training recipe |
| [`docs/DATA.md`](docs/DATA.md) | the funnel with counts, crop geometry, admission design, audit statistics, how to read 0.670 vs 0.898 vs 0.574 |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | the six arms at matched passes, the eight failures with the measurement that settled each |
| [`DATA_CARD.md`](DATA_CARD.md) | the required one-pager |
| [`results/mixed_v1/LABEL_AUDIT.md`](results/mixed_v1/LABEL_AUDIT.md) | the audit's own appendix: vocabulary, κ = 0.196, the 17 adjudicated cells |

## Reproduce

The data is not redistributable, so this repository contains no pool, no dev set and no images. Put
the course pool at `data/student/` and the dev set at `data/dev/dev.npz`, or set `CATDOG_DATA` to
another root. Every script defaults to a path inside this checkout (never a machine-specific one);
`CATDOG_KIT` can point at a course starter kit that still has `classifier.pt`, and `CATDOG_YOLO`
at a YOLO11x weight if you re-run the curation probes. All three are optional and relative by
default, so a fresh copy of this folder works on another computer without edits.

```bash
pip install -r requirements.txt

# 1. what the marker runs: interface + manifest + all six metrics.
#    starter_kit/ here is the course kit with classifier.pt removed, because that file is a
#    forbidden artifact (handout p.4). Everything else in it is byte-identical, so point this
#    command at the course copy instead: <kit> = the starter_kit folder from UMMoodle.
python <kit>/evaluate.py --submission submission --data-root data --n-per-class 1000 --seed 0

# 2. the 200 audit cells, redrawn bit-identically from the shipped manifest
python starter_kit/audit.py --manifest submission/manifest.csv --student-id UC325629

# 3. the label statistics quoted in the data card (Wilson intervals, kappa, per-arm split)
python pipeline/p10_consistency.py

# 4. tests (run the two suites separately: the starter kit's own tests import bare module
#    names, so a single process with both on sys.path trips a known collision)
python -m pytest -q tests
python -m pytest -q starter_kit/tests
```

Checked on 2026-11-26: command 1 at 24 per class returns `deterministic=True  seed_sensitive=True
single/mixed batch 0.96/0.87` and `manifest 3000/3000 valid rows`; commands 2 and 4 run from this
copy as written; 11 and 26 tests pass. Three of the eleven rebuild crops from the pool: with `data/`
unpacked (or `CATDOG_DATA` set) all 11 pass, and without it those three stop on
`data/student/index.csv not found` while the other eight pass.

Training the delivered checkpoint, and the arms that did not ship:

```bash
python -u pipeline/p6_mixed.py --data-root data                 # rebuild the 3,000-row manifest
python -u pipeline/p7_mixed.py --stage train --arms mixed60k    # 60,000 steps, ~92 min on a 2060
python -u pipeline/p7_mixed.py --stage eval  --arms mixed20k mixed60k
```

## LIVE DEMO

`demo_live.py` draws a seed-controlled cat row and dog row straight from the shipped
`weights.pt`. It needs only torch, numpy and pillow — no starter kit, no data and no network:

```bash
unzip CISC3024_UC325629.zip -d CISC3024_UC325629
pip install torch numpy pillow --index-url https://download.pytorch.org/whl/cpu
python demo_live.py --submission CISC3024_UC325629/submission --seed 1 --per-class 4 --open
```

The command writes `demo_live_seed1.png` in the current folder and opens it in the default
viewer (`--open` can be omitted on a headless machine). Drop `--open`, or pass `--seed 0`,
`--per-class 8`, `--tile 192` or `--out path.png` to change the display. The images come from
the same `load_model` / `generate` pair the evaluator calls, so the grid is what the marker
measures. `demo_live.py` is also included at the root of the project zip.

Run the three lines from the **project root** (the folder that contains `submission/`), not
from inside the unzipped folder. `--submission` is resolved from the current folder, the
script folder and its parent, so `CISC3024_UC325629/submission` works from either place.

On Windows, if `python` is not on your PATH, or is a stub that cannot import torch, use the
launcher. It tries `.venv`, then `python`, then `py -3`, keeps the first one that can import
torch, prints which interpreter it chose, and names each candidate that failed and why:

```powershell
.\run_demo.ps1 -Submission CISC3024_UC325629/submission -Seed 1 -PerClass 4 -Open
```

`demo_live.py` and `run_demo.ps1` are both in the zip, and neither holds an absolute path, so this
works on any Windows machine with Python 3 and torch. The zip contains no virtualenv to find; if no
interpreter qualifies, the launcher prints the pip line to install one.

## What the zip leaves out, and why

- **Images, the pool, `data/`** — course data, not redistributable, and excluded by the brief.
- **Checkpoints other than the graded one** — 14 intermediate `.pt` files are not shipped.
- **The slides are in the zip** — the deck sits at the repo root as
  `Blue And White Futuristic Minimalist Artificial Intelligence Informative Presentation.pptx`
  (3.7 MB, 8 slides). Its PDF export is left out on purpose: the same 8 slides as 44 MB of page
  images.

## Provenance

Curation uses pretrained detectors (YOLO11x, cat/dog face landmarks, OpenCLIP ViT-L-14, RapidOCR);
none of their weights is loaded by `submission/`, which imports only `torch` and its own sibling
modules. The generative model is trained from scratch on the 3,000 manifest rows, and the evaluator's
classifier and Inception network were never used, not even for curation.
