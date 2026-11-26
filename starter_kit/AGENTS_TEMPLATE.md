# AGENTS.md — project context for coding agents

<!--
Copy this file to the root of your repository as AGENTS.md (or CLAUDE.md,
depending on your agent) and keep it up to date. It is graded: it should let a
fresh agent session work on your project correctly without you re-explaining
anything. Delete these comments.
-->

## Project

Class-conditional generation of 32×32 cat and dog heads (CISC3024 course project). The
training set is curated by us from a raw pool of LAION web images and described by
`submission/manifest.csv`. Graded by `starter_kit/evaluate.py`; the interface is fixed
(see `submission/generate.py`).

## Hard constraints (never violate)

- Training images come only from the raw pool in `data/student/`, and every one is a row of
  a manifest (`pool_id,x0,y0,size,angle,label`, cropped with `catdog_eval.data.crop`).
- Never train on `data/dev/` or anything derived from it; never use other datasets or images
  made or edited by generative models.
- Pretrained models (detectors, CLIP, ...) only for curating data. The generative model is
  trained from scratch; no pretrained weights, no downloads at generation time.
- Never use the evaluation classifier or the evaluator's Inception network in training or
  generation.
- `generate(model, labels, seed)` must be deterministic in `(labels, seed)` and return
  `[N, 3, 32, 32]` floats in `[0, 1]`; labels 0 = cat, 1 = dog.
- Budget: <= 20M parameters, submission directory <= 100 MB, memorization rate <= 5%.
- Do not modify anything under `starter_kit/`.

## Data specification

<!-- Copy the specification from the handout, and add the decisions you made to meet it
     (detector, thresholds, crop margin, filters), with the reason for each. -->

## Layout

<!-- Where things live: data pipeline, manifests, models, training
     scripts, configs, experiment logs, report. -->

## Commands

<!-- Exact commands, e.g.
- Tests:     `pytest -q`
- Data:      `python data/build_manifest.py --out submission/manifest.csv`
- Audit:     `python starter_kit/audit.py --manifest submission/manifest.csv --student-id ...`
- Train:     `python train.py --manifest submission/manifest.csv --config configs/cgan.yaml`
- Evaluate:  `python starter_kit/evaluate.py --submission submission --n-per-class 250`
-->

## Conventions

<!-- Code style, how experiments are named and logged, where results go,
     commit message format, what must be true before committing (tests pass, ...). -->

## Verification checklist for every change

<!-- What the agent must check before claiming something works, e.g.
- unit tests pass (including: my crop code == catdog_eval.data.crop)
- `check_manifest` reports no errors
- a grid of 100 random crops from a new manifest was saved and looked at
- a short training run shows the losses behaving and a sample grid was looked at
-->

## Known pitfalls in this project

<!-- Things that went wrong before and how to avoid them. -->
