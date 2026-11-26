# CISC3024 cat/dog head generation — project context for coding agents

This file is the working contract for any fresh agent session. Read it before touching code.
The authoritative rules are, in order: the course handout (not redistributed here) >
`starter_kit/README.md` > this file.
The current consolidated brief (v3) overrides older prompts; where this file and the brief differ,
follow the brief.

## Project

Class-conditional generation of 32x32 cat and dog heads for CISC3024. The training set is
curated from the raw LAION pool in `data/student/` and described by
`submission/manifest.csv`. The official evaluator is `starter_kit/evaluate.py`; the submission
interface is fixed by `starter_kit/submission_template/generate.py`.

Human owner: LAM KA WAI, UC325629. The human is responsible for the 200-image manual audit,
the slides, the oral explanation, and the "how I verified it" sentences in `AGENT_LOG.md`.

## Hard constraints (never violate)

1. Training images come only from the raw pool in `data/student/`, and every one is a row of
   `submission/manifest.csv` (`pool_id,x0,y0,size,angle,label`), rebuilt with
   `catdog_eval.data.crop`. **A row may exist only if the human marked `human_ok=Y` for that
   pool image** in `results/review/admission_*.csv` (Phase 1.5 quota admission). Unreviewed
   candidates never enter, however well they score; a blank cell means "not seen", not "fine".
   **Amendment (2026-11-25): the shipped manifest has two arms, and the paragraph above governs
   one of them.** After the 200-cell audit priced both populations, the decision was to keep the
   2,000 rule-selected rows as a second arm rather than ship the 1,000 human rows alone: 1,000
   rows carry `human_ok=Y`, and 2,000 were selected by the stage-7 rule stack with **no human
   ever seeing them**. Their measured cost is 0.574 good against 0.898 on the human arm (0.643
   once both machine draws are pooled), and that trade is what moved the evaluator from 11/15 to
   13/15. It is stated in `DATA_CARD.md` §3-§4 and in the report, not hidden. Read the paragraph
   above as the rule the human arm was built under; it is not a description of the shipped tree.
2. `data/dev/` may be used only for aggregate metrics / visual specification checks to choose
   the pipeline, hyperparameters, checkpoints and sanity checks. Never train on dev images or
   augmented/transformed dev images. Never use per-image dev features, per-image dev labels,
   or similarity to individual dev images to select pool images or fit a filter.
3. Pretrained models (YOLO11x detector, face localizer, CLIP) are for data curation only.
   The generative model is trained from scratch with no pretrained weights, including no
   pretrained autoencoder.
4. Never use `classifier.pt` or the evaluator's Inception network, not even for curation.
   Never train a model to imitate them.
5. Never download or add images outside the pool. Training images must not be generated or
   edited by a generative model.
6. At most 20M parameters and 100 MB in `submission/`. Store one `state_dict` only; no
   optimizer state and no extra checkpoints in `submission/`.
7. Memorization rate must be at most 5%. Deduplicate the manifest, avoid overfitting, and
   never return a training image or its mirror image. The evaluator counts horizontal flips
   as copies and deduplicates near-duplicates before computing d1/d2.
8. `generate` must be deterministic in `(labels, seed)`. Each image may depend only on its own
   label and the seed, never on the rest of the batch.
9. `load_model` returns exactly one `nn.Module` that owns every parameter, and it must be in
   `.eval()`. Single-class batch accuracy may not be more than 0.2 below mixed-batch accuracy.
10. Output must be `[N, 3, 32, 32]` float, values in `[0, 1]`, no NaN. No network access and
    no downloads at generation time.
11. Do not modify anything under `starter_kit/`. The unmodified copy is kept for reproducibility;
    the graders use their own copy.
12. Do not make slides, do not label the audit, do not answer the oral exam, and do not write
    "I verified" sentences that the human did not actually perform.
13. `yolo11x.pt` stays at the project root as a curation tool. It must not enter `submission/`
    or the project zip. The README must document how to obtain it (Ultralytics version, URL,
    automatic-download behaviour) so the raw-pool -> manifest path is reproducible on a clean
    machine.
14. `gsyllas/vit-cats-dogs-head` is not used for final labels or final features (training
    provenance unknown; if it shares data with dev it would violate constraint 2). It belongs
    only in "What did not work".
15. Do not use constants fitted to individual dev images. The old `K=1.81`, `OX=-0.01`,
    `OY=0.10` came from per-image measurements on 904 dev images (medians), so they are retired
    for final manifest generation. Re-derive the crop geometry as a fixed constant from the
    60-80% specification plus visual sanity checks; dev may only sanity-check aggregate ratios.
16. Do not fit CLIP thresholds, OCR thresholds, edge thresholds or saturation thresholds on
    dev. Use fixed heuristics, or statistics computed only from pool candidates.

## Data specification

Use separated measurements:

- `w` = head bounding-box width.
- `h` = head bounding-box long side, including ears.
- `size` = square side, derived from `max(w, h) / r`.
- "at least 32 px wide" is checked on `w`.
- "head + ears takes 60-80% of the side" is checked on `h / size`.
- Cats often have `h > w`; using `w` alone to set `size` cuts the ears, which the audit marks bad.

Every row must pass these asserts before it enters the manifest:

```python
assert w >= 32
assert 0.65 <= h / size <= 0.75   # pipeline target, 5% buffer inside the spec
assert 0.60 <= h / size <= 0.80   # specification bounds
assert size >= 40                 # defensive redundancy
```

`check_manifest` only checks `size >= 40`, angle, image bounds and repeated rows. It does not
check the 60-80% ratio. Do not rely on it to catch a loose crop.

Reject `h / w > 1.4` as an abnormal pose, or send it to human boundary review.

Audit rubric (the human labels, not the agent):

- good: one real photo, exactly one cat/dog head, upright, head + ears 60-80% of the square,
  ears complete, no obvious occlusion/blur, `w >= 32`.
- boundary: 55-60% or 80-85% -> human decides; lean bad and write the reason.
- bad: ears cut, an eye or contour occluded by more than about 15%, profile/lying/inverted head,
  two heads (including a visible corner or a mirror reflection), cartoon/drawing, product/print,
  statue/toy, wrong animal, blurry/upscaled, text/watermark.
- fixed reason vocabulary: `cartoon/drawing`, `product/print`, `statue/toy`, `two heads`,
  `head too small`, `head too large/ears cut`, `not upright`, `blurry/upscaled`, `wrong animal`,
  `text/watermark`.

Other decisions:

- Exactly one head: strict. Reject any image with more than one cat/dog detection.
- `angle = 0`; no rotation. There is no reliable roll estimate, rotation brings background in,
  and it complicates the in-image square check.
- Final target: **500 human-verified good images per class** (the handout lower bound), i.e.
  1,000 rows total, admitted one row at a time by the human (Phase 1.5). 1,000/class is a
  **stretch goal**, not the target, and is only pursued after 500/class is met with time to spare.
  Do not relax filters to hit a number. If a class cannot reach 500 from the existing candidates,
  widen the candidate set from the 18,640 single-detection pool images -- never admit a row the
  human has not seen, and never top up from a score-ranked list.
- One crop per pool image.
- Labels: 0 = cat, 1 = dog.
- `check_manifest` must report `errors=[]`; explain every warning.

## Pipeline

1. **Graded cache.** Run YOLO11x on the full 76,250-image pool first and cache detections in
   `results/scan/yolo_detections.jsonl`. The scan is resumable and appends one JSON object per
   image. This is the cheapest stage and can run overnight.
2. **Candidate crops.** Run the face localizer and CLIP only on images that passed YOLO. Build
   the head crop with fixed geometry (not dev-fitted), then apply the three invariants.
3. **Non-photo exclusion.** CLIP is used only as an exclusion filter, never as a top-k ranker.
   Positive prompts: `a photograph of a real cat head`, `a photo of a real dog's head`.
   Negative prompts: cartoon, illustration, painting, drawing, render, toy, statue, printed on a
   product, book cover, text or watermark. Do not use prompts containing breed, coat colour,
   lighting, frontal, close-up, studio or white background; those narrow appearance and hurt recall.
   Add fixed OCR/text, edge and saturation heuristics. The human audit is the final check.
4. **Deduplication.** One crop per pool image; drop exact repeated rows; if two crops from the
   same image overlap by IoU > 0.7, keep only one.
5. **Ordering first, human admission second.** Never pick the dataset by score. Order the passing
   candidates by `sha256(f"{seed}:{pool_id}")` (seed 0) inside 9 strata of `brightness` tercile x
   `face_aspect` tercile, interleaved round-robin, one queue per class. The human reviews that
   fixed order and marks `human_ok=Y/N`; only `Y` rows enter the human arm (the queue stopped at
   500 admitted rows per class, then was extended to 1,000; the other 2,000 shipped rows come from
   the rules-only arm described in constraint 1's amendment). Ranking by `det_conf`, CLIP
   style, sharpness or any other score is forbidden: that is top-k selection wearing a quota
   costume, and "it was not fitted on dev" does not make it legitimate. The order is fixed before
   the first verdict is seen and is never re-shuffled afterwards.
6. **Final label sources:** YOLO class + face localizer + CLIP style exclusion + human admission
   (Phase 1.5) + the independent 200-row audit (Phase 2). The HF `gsyllas/vit-cats-dogs-head`
   classifier is not used.

## Layout

```
data/student/                 raw pool (index.csv + shards); gitignored, never packaged
data/dev/                     dev set and Inception cache; gitignored, never packaged
pipeline/scan_pool.py         full-pool YOLO scan, resumable JSONL cache
pipeline/build_manifest.py    formal manifest builder (resumable)
pipeline/probe_*.py           exploratory probes, kept for the agent log / "what did not work"
train.py                      model training entry point (TTUR + EMA)
submission/generate.py        fixed evaluator interface
submission/model.py           generator/discriminator definitions
submission/weights.pt         one final state_dict
submission/manifest.csv       every training image
tests/                        four project tests
audit/audit.csv               human labels, 200 rows (human-owned)
results/                      scan caches, eval.json, sample grids, experiment logs
AGENTS.md, AGENT_LOG.md, README.md, DATA_CARD.md, COMMIT_PLAN.md, .gitignore
```

`data/` is reserved for the dataset and is gitignored. The formal pipeline entry point therefore
lives in `pipeline/`, not in `data/`; this avoids ignoring the source code along with the pool.

## Commands

```bash
# Environment / starter-kit sanity
python -m pytest -q starter_kit/tests

# Full-pool YOLO scan (resumable; run from the project root)
python -u pipeline/scan_pool.py --out results/scan --batch 8

# Build manifest (resumable; rerun the same command to continue after a stall)
python pipeline/build_manifest.py --scan results/scan/yolo_detections.jsonl \
    --out submission/manifest.csv --out-dir results/manifest_v1 --device cuda

# Manifest check
python -c "import sys; sys.path.insert(0, 'starter_kit'); from catdog_eval import check_manifest, default_pool; r = check_manifest('submission/manifest.csv', default_pool()); print(r['errors']); print(r['warnings']); print(r['stats'])"

# Manual audit: ONLY after the final manifest is frozen. Human labels the 200 rows.
python starter_kit/audit.py --manifest submission/manifest.csv --student-id UC325629

# Train model A (writes config, losses and sample grids under results/experiments/)
python train.py --manifest submission/manifest.csv --out submission/weights.pt \
    --steps 25000 --batch 128 --width 384

# Official evaluation protocol.  The starter_kit/ in this repo has no classifier.pt (forbidden
# artifact), so run the course copy of the same file; everything else is byte-identical.
python <kit>/evaluate.py --submission submission --data-root data --n-per-class 1000 --out results/eval.json
```

## Conventions

- English for `README.md`, `DATA_CARD.md`, `AGENTS.md`, `AGENT_LOG.md`, report text and plot labels.
- Keep experiment logs under `results/experiments/<name>/` with config, seed, hardware, loss curve,
  training time and sampling speed.
- Never put optimizer state or extra checkpoints in `submission/`.
- Before claiming a result, give the exact command, output path, and a number or assertion that
  another person can reproduce.
- Git is **not** deferred (LIMITS.md §6, re-confirmed 2026-11-23: the handout's "use git and commit
  often" applies). Commit real progress with real timestamps; never backdate.

## Verification checklist for every change

- [ ] `check_manifest` reports `errors=[]`; every warning is explained; **1,500 rows per class
      (3,000 total; 500 per class is the handout's floor)**, ratio < 2, no overlapping crops.
- [ ] Every row **of the human arm** has a matching `human_ok=Y` in `results/review/admission_*.csv`,
      and the count equals the number of `Y` cells per class. The 2,000 machine-arm rows carry no
      `human_ok` by design (constraint 1's amendment); they are covered by the stage-7 rules and
      priced in `DATA_CARD.md` §3.
- [ ] Every manifest row passes the three invariants; `w`/`h` are separated; `h/w > 1.4` handled.
- [ ] The four project pytest tests pass: crop equality, generate shape/range/NaN, single-batch
      overfit, fixed-seed reproducibility + different-seed difference + single-class/mixed-batch
      consistency.
- [ ] `evaluate.py --submission submission --n-per-class 1000` gives six metrics and a sample grid;
      the human has looked at the grid.
- [ ] Parameters <= 20M, `submission/` <= 100MB, memorization <= 5%.
- [ ] `submission/` has no downloads, no `classifier.pt`/Inception, no pretrained generator weights,
      one checkpoint only.
- [ ] README gives one command per reported number, including how to obtain `yolo11x.pt`;
      DATA_CARD has the Wilson 95% CI and the `32 / 0.8 = 40` derivation.
- [ ] AGENTS.md is current; AGENT_LOG.md has three human-written checks, at least one about data,
      with no invented mistakes.
- [ ] report.pdf is <= 6 pages, English, cover says LAM KA WAI / UC325629, six sections, Results
      table, sample grid, references.
- [ ] Git history has real commits with real timestamps (no backdating); `.gitignore` excludes
      `data/`, zips, `yolo11x.pt`, venv and non-graded checkpoints.
- [ ] The project zip excludes `data/`, zips, `yolo11x.pt`, `student/student/`,
      `Course Project Resources/`, venv, non-graded checkpoints, `audit/sheet_*.png`, and pool crops
      under `results/probe_*`; the slides are human-made.

## Known pitfalls in this project

- **Face localizer x coordinates were reversed** in the TFLite export; sort `(x1, x2)` and
  `(y1, y2)` before mapping back. This was found by checking boxes on dev/pool images.
- **CLIP alone is not a photo detector.** `ViT-B-32` could not separate cartoons/products.
  `ViT-L-14` still accepted cartoon dog `s030905` at style 0.927 and book cover `s042992` at
  0.644. Use negative prompts plus fixed heuristics plus human audit; never a single threshold.
- **YOLO recall is limited.** A 300-image random probe found cat/dog detections in 83 images
  (27.7%): 34 cat, 52 dog, 3 both. This implies roughly 21k candidates, enough for 1000/class,
  but not a reason to lower `imgsz` or sample the pool.
- **Template evaluator** gives accuracy 0.5, FID 440.81 and 1/15 marks; that is the expected
  placeholder result, not a model result.
- **BatchNorm in training mode** breaks the single-class/mixed-batch check. Prefer GroupNorm or
  spectral norm; if BatchNorm is kept, `load_model` must call `.eval()` and the pytest must measure
  the accuracy difference.
- **`h/w > 1.4`** often means the ears are cut or the pose is abnormal; reject or send to human
  boundary review.
- **Audit boundary cases** at 55-60% or 80-85% should lean bad; the graders relabel 50 images and
  compare agreement, not the good rate.
- **`generate()` must draw from the same noise distribution the training loop used.** The first
  shipped generator trained on `torch.randn` but was evaluated with `torch.rand` (std 0.289 vs
  0.997). Mean pairwise L1 among 128 samples fell from 0.342 to 0.158, precision looked fine at
  0.72 while recall collapsed to 0.008. Keep `_noise` in lockstep with `train.py`; the training
  sample grids and the evaluator must see the same distribution or the grids lie.
- **`generate()` must take the device from the model, not from `labels`.** Noise built on
  `labels.device` raises a device-mismatch error when the caller passes CPU labels and the weights
  are on the GPU -- which is exactly what a hand-written check does.
- **The label can be live in G and still look broken.** `embed.weight.grad` being non-zero and the
  two class rows being far apart (L2 9.25) only prove the wiring. Whether the condition *matters*
  has to be measured: fix z, flip the label, and look at the pixel difference (0.143 mean abs diff,
  50% of pixels > 0.1 for the DCGAN).
- **Do not trust a 250/class FID, but it is a usable A/B signal.** On this evaluator the same
  checkpoint gave FID 179.09 at 250/class and 159.03 at 1000/class. Only compare runs at equal
  `--n-per-class`, and quote only the 1000/class number.
- **`DETERMINISM_TOLERANCE` is 1e-4, not bitwise.** cuDNN picks a different kernel per call, so
  even an eval-mode DDPM repeats to ~5e-6. Tests must use `torch.allclose(atol=1e-5)`; `torch.equal`
  produces false failures that look like a broken RNG.
- **Duplicate folders** `student/student/` and `Course Project Resources/` are Windows
  "Extract All" backups. Ignore, do not package, do not delete. `data/student/` is the only pool.
- **`data/dev/dev.inception.pt`** is an evaluator feature cache, not training data; it is
  gitignored with `data/`.
- **A human-vetted manifest cannot be audited as if it were untouched.** Once admission requires
  `human_ok=Y` per row, the Phase 2 audit is drawn from rows the human already approved, so the
  number rises towards 1.0 *by construction* and is no longer an estimate of curation quality.
  Call it **post-admission re-label consistency** (never "precision"; the Wilson interval is a
  consistency interval, not a precision CI), and keep it away from the legacy 0.345 audit and the
  0.57 Phase 0 probe -- those two live in the methodology / "what did not work" section only.
  The 200 cells must be labelled **one session later, with the old verdict hidden**: never fill
  `good` in bulk from `human_ok=Y`. `audit.py` and the `audit.csv` columns stay untouched (handout
  §7 / §10 require that exact artefact).
- **`data/` is a read-only pool in the redo workspace.** `PROJECT3` carries the scan cache
  (`results/scan/yolo_detections.jsonl`, 76,250 rows) but not the images; every command that needs
  pixels must point `--data-root` / `CATDOG_DATA` at `D:/CISC3024/PROJECT/data`. A "pool not found"
  warning from `evaluate.py` silently turns `memorization_rate` into `null` instead of failing.
- **`--merge` reused its own output as the input draft.** `review_sheet.py` took the queue's tercile
  cuts and its "already shown" set from `--manifest`, which is also where `--merge` writes. The first
  run was correct (2,000-row draft -> 1,000-row manifest); the second run rebuilt the queue from the
  admitted 1,000 rows, so the strata moved, the extension set became "everything except the admitted
  rows", and the same command admitted a **different** 500 per class (md5 `a1191d15` -> `2d5970bc` (both are CRLF-era hashes from before the writer was pinned to LF; the canonical file is `66f60cfb`)).
  Nothing errored and the row count stayed 1,000, so only an md5 comparison caught it. **Fixed
  2026-11-25**: `--draft` defaults to the frozen `results/draft_pre_review.csv`, `--manifest` is
  output only, and the two resolving to one path is a hard refusal. Re-running is now idempotent
  (two runs, same md5). Lesson: any command that both reads a queue and writes the file that defines
  it needs the two paths named separately *and* an idempotency check in the run.
- **A blank `human_ok` cell is a verdict in this project, and the two waves were once read
  differently.** All 25 admission sheets mark the passes with `good` and leave the rejects blank --
  zero `N` cells anywhere, marks scattered over the whole 0–99 range, a hand-typed good tally under
  each sheet. Wave 1 was read that way (468/1,000), but `--report` was written to treat wave-2 blanks
  as "not seen yet", which reported cat 0.488 / dog 0.462 instead of 0.461 / 0.385 -- same data, two
  denominators, and the flattering one would have gone into the data card. **Fixed 2026-11-25**: one
  reading for every wave, a fully unmarked sheet excluded from the denominator and named in the
  output, and the last marked row per sheet printed so a half-finished sheet is visible. The manifest
  is unaffected either way because `load_verdicts` only accepts explicit `Y`/`N`. Lesson: when the
  human invents a shorthand, fix its meaning **once, in writing, before counting**, and make the
  admission path depend on the strictest possible reading.
- **A quoted md5 that depends on the line ending is not a hash, it is a coin flip.** `.gitattributes`
  pins `eol=lf`, but `csv.DictWriter` writes CRLF on Windows by default, so the same 1,000-row
  manifest had two identities: `a1191d15` in the working tree and `66f60cfb` in the blob, with a
  fresh checkout producing the second. `merge()` and `write_sheet()` now pass
  `lineterminator="\n"` and only the LF hash is documented. Check `git ls-files --eol <file>` shows
  `i/lf w/lf` before quoting any hash in `DATA_CARD.md`, the README or the report.
- **The UNet is stochastic in train mode, so "same loss twice" is not a test of anything.**
  `unet.py` carries `nn.Dropout(0.1)`. Two `model.loss(...)` calls with the *same* generator seed
  (identical `t` and identical `eps`, verified) still return values that differ by ~6e-3, and two
  `generate(...)` calls on an untrained model differ by ~1e-2. Both looked like bugs when the
  Phase 3 cross-assertions were written: the `snr_gamma=0 == L_simple` check reported "not
  bit-identical" and the device check reported a 1e-2 cpu/cuda gap. Neither was real. Any check
  that compares two loss values, or two noisy forward passes, must call `.eval()` first --
  `load_model` already returns `.eval()`, which is why the same checks pass on real checkpoints.
  Training itself runs in train mode, where the dropout is intended.
- **A fixed-step A/B between two manifests is not a fixed-epoch A/B.** Shot #1 pinned
  `--steps 40000 --batch 64` for both arms, which holds *samples* fixed (2.56M) but not *epochs*:
  1,280 passes over the 2,000-row draft versus 2,560 over the 1,000-row admitted set. The
  treatment arm lost 0.178 recall and one mark, and the obvious reading -- "human admission hurt
  the data" -- was wrong by more than half. Re-running the same manifest at 20,000 steps (the same
  1,280 epochs as the control) recovered 0.104 of that recall and the lost mark (11/15), and beat
  the 40k run of the *same* 1,000 rows on recall (+0.104), FID (-2.46) and total marks (11 vs 10):
  at 1,000 images the 40k arm had been overfitting for half its run. Lesson: when the two arms
  differ in dataset size, report the epoch count next to the step count, and if the question is
  "did the data change", hold epochs fixed or the answer is unidentifiable. `pipeline/p4_shot1.py`
  now carries a per-arm step budget for exactly this.
- **Starter kit integrity:** `README.md`, `audit.py`, `evaluate.py` and `AGENTS_TEMPLATE.md`
  match the original `starter_kit.zip` SHA256; do not modify them.
- **`h` means the long side.** The first version of `fixed_head_box` used the vertical extent,
  which under-sized wide faces and rejected 41 of 600 debug rows as `ratio`; `h = max(w, v)` is
  the specification-correct definition (brief §7) and removed those false rejects.
- **The face localizer returns four numbers, not landmarks.** Both `.tflite` models output
  `[1, 4]`, so pose (lying, inverted) cannot be checked geometrically; it is the human audit's job.
  Do not pretend a landmark check exists.
- **CLIP cannot separate prints from photos.** A probe on 12 hand-picked rejects and 30 manifest
  rows gave style 0.94-0.997 for product boxes, screenshots and line art, versus 0.75-0.99 for
  real photos (`pipeline/probe_clip_prompts.py`). Adding line-art/clipart/silhouette/packaging
  negatives moved scores by < 0.03. CLIP stays a coarse photo-vs-art filter at 0.50.
- **OCR must run on every crop.** Running it only when CLIP style was below 0.75 skipped exactly
  the prints that scored 0.99 and left "Fiera" boxes and web screenshots in the manifest. Text
  area on those crops is 9-38%, far above the 0.5% threshold, so pixel-level OCR is the layer
  that works.
- **Long runs stall in this environment; the builder is resumable.** `candidates_raw.jsonl`,
  `stage1_cursor.json` and `quality_raw.jsonl` let `build_manifest.py` restart at the last
  200-record block. Never restart it from scratch to "be safe"; that costs another full pass.
- **BatchNorm + `.eval()` still varies by ~1e-7 across batch shapes** (cuDNN picks a different
  convolution kernel), so batch-composition tests must use `allclose(atol=1e-5)`, not
  `torch.equal`. Noise must also be indexed by `(seed, position)`: comparing the same label at
  two different positions gives a 0.5 pixel difference and looks like a batch-dependence bug.
- **`--epochs` used to truncate `--steps`.** The training loop stopped at the epoch cap (6,200
  steps instead of 12,000). Epochs are now derived from steps; check `config.json: steps` against
  `losses.csv` line count after every run.
- **The montage label column defaults to "dog".** `montage()` read `r.get('label') == 0`, but
  reject records never carried a `label`, so every cell of `reject_grid.png` printed "dog" while
  the image was often a cat. The human spotted it and read it as a mislabelling bug. Fixed:
  rejects now store `label`, and the prefix is omitted when the label is unknown. Lesson: a
  diagnostic image whose caption can lie is worse than no caption.
- **More parameters + more steps is not monotone.** 11.18M params / 60k steps gave accuracy
  0.6795 (up from 0.636) but FID 111.96 (down from 100.44), KID 0.0526 (from 0.0339) and recall
  0.3695 (from 0.4325): total 7/15, one mark *below* the 5.15M/40k model. 1,936 epochs on 2,000
  images overfits the classifier and narrows coverage. Always compare the summed marks, never a
  single metric.
- **Threshold tightening cannot fix an audit.** Re-filtering the cached candidates with
  `style >= 0.60`, `sharp >= 100`, `flat <= 0.60`, `size >= 80`, `det_conf >= 0.50`, `h/w <= 1.20`
  moved audit precision from 0.332 to 0.372 with overlapping Wilson intervals while cutting the
  pool from 4,821 to 2,516. The residual error is semantic (pose, second head, "photograph of a
  print"), which no scalar threshold sees.
- **A checkpoint that keeps only the EMA weights cannot be post-selected.** `train_diffusion.py`
  overwrites the live weights with the EMA copy before saving and never writes the raw weights or
  any intermediate snapshot, so "try another EMA decay / an earlier snapshot" is *impossible*
  after the run — it needs a retrain. `--ckpt-every` was declared in `argparse` but never used in
  the loop, which is exactly what makes the dead end invisible. Check that every flag in the
  parser has a consumer before promising a sweep that depends on it. **Fixed 2026-11-24**: the flag
  now writes an EMA snapshot every N steps, and the first measurement it enabled (B3 at 10k/20k/
  30k/40k) showed the snapshots are *monotone in steps* — 7, 9, 9, 10 marks — so for this project
  the missing snapshots never cost anything. Keep the flag on for every future run anyway: it is
  the only way to answer this question instead of arguing about it.
- **A null token is not free when there are only two classes.** Training with 10% label dropout
  forces the network to predict the eps of the *mixture* `0.5 p(x|cat) + 0.5 p(x|dog)` on those
  batches, and that mixture is sharply bimodal in a 2-class problem — unlike ImageNet-1000, where
  the unconditional target is a broad continuum. Measured on the same data, same 40k steps, same
  seed: film conditioning + null token + cosine lr gave KID 38.0e-3 and recall 0.518 at 250/class,
  against **28.4e-3 and 0.594** for the additive-conditioning model the null token was supposed to
  improve. The class signal was demonstrably *stronger* (flipping the label at fixed noise moved
  30.8% of pixels vs 17.0%), so this is not a wiring failure: the interference from the dropped
  batches costs more than the guidance gains. Reverting is the correct outcome; the retrain is
  recorded in README.md "Round 3" and `results/experiments/B_ddpm3/`.
- **Guidance weight is capped by the determinism check, not by taste.** The evaluator's
  `DETERMINISM_TOLERANCE` is 1e-4 and guidance multiplies per-step drift by `(2w - 1)`. At w=4.5
  the official evaluator reported `generate() is not deterministic for a fixed seed`, set
  `valid: false` and scored **0/15** — with accuracy 0.870 and recall 0.530, i.e. the metric
  numbers looked fine and only the contract broke. The shipped w=2.5 sits at 8.7e-6, roughly a
  12x margin. Never raise the weight without re-running the evaluator's own determinism check.
- **`scan_posttrain.py` variant JSONs are not unique per checkpoint.** The per-variant output was
  `results/scan/<variant>_n<N>.json`, with no checkpoint tag, so screening a second checkpoint with
  the same variant name silently overwrote the first one's numbers (this happened twice: `nt250`
  lost the 40k row to a 30k snapshot, and `contrast200_n1000.json` was written three times in one
  afternoon by two different checkpoints). **Fixed 2026-11-24, not just recorded**: every output now
  carries the checkpoint label -- `results/scan/[<exp>/]<variant>__<ckpt>_n<N>.json` and
  `scan__<ckpt>_n<N>[__<tag>].json`, with `<ckpt>` = `--ckpt-tag` or the checkpoint's parent
  directory -- and an existing per-variant file is never overwritten without `--force` (the run is
  skipped and the refusal is written into the summary as `refused-to-overwrite`). Self-test: the
  same command twice leaves the first file's md5 unchanged and the second summary empty.
- **Inference-time guidance amplifies batch-shape numeric drift.** Antithetic guidance multiplies
  the per-step float difference by ~(2w-1); at w=2.5 the spread between batch shapes grew from
  ~1e-6 to 2.0e-05, which broke the old `atol=1e-5` batch-size test. Same-shape batches are still
  bitwise identical (0.0), and no pixel moved by 1/255, so the fix was to measure and re-pin the
  test at the evaluator's own 1e-4, not to weaken it to `isclose`.
- **The strongest free lever was the sampler, not the weights.** The 40k DDPM had a weak class
  signal, not a bad distribution: extrapolating away from `1 - y` (w=2.5, `diffusion.py`) raised
  accuracy 0.6365 -> 0.8425, FID 100.44 -> 89.705741 and recall 0.4325 -> 0.540929
  *simultaneously*, and
  memorization stayed 0. An untrained null token (zeroed label embedding) made everything worse
  (KID 49.0 vs 35.3), so only embeddings the model actually learned are usable as a negative.
  Costs one extra forward pass per step. **Sampling time in this project is not a stable number
  and must not be quoted from a contended run**: the same shipped submission measured 101.1, 79.9
  and 40.1 s/1000 on the same RTX 2060 depending on whether another job held the GPU. Quote only
  the value from the run that produced the reported numbers, and say what else was running.
- **Min-SNR loss weighting loses marks at a fixed step budget.** `w(t) = min(SNR(t), gamma)/SNR(t)`
  with `gamma = 5` on the *same* eps-prediction objective, same manifest, same 40k steps, same
  seed, same architecture (5,145,731 params) as the 11/15 model: accuracy 0.8425 -> 0.772-0.788,
  FID 89.706 -> 103.1-107.5, KID 28.3 -> 47.6-55.9e-3, recall 0.5409 -> 0.3883-0.4170, best total
  9/15 against 11/15. The mean weight is 0.8232, so the weighted loss is ~0.82x the unweighted one
  *by construction* -- its "lower" loss values are not evidence of anything and must never be
  compared with an unweighted run. Two of the five variants had to be discarded for recall < 0.40.
  Do not read this as "min-snr is wrong in general": the honest statement is that it costs marks
  under this budget, and the step budget is the first thing to change before trying it again.
  Round 6 in README.md; `PROJECT2/results/experiments/B_ddpm5/`.
- **`--ema-list` answers a different question from `--ckpt-every`.** Parallel decays compare the
  *same moment* at 0.995 / 0.999 / 0.9995; snapshots compare *different moments* of one decay.
  Measured at w=2.0 on the E2 weights, the decay axis is small but not zero (accuracy 0.7700 /
  0.7720 / 0.7765, FID 106.084 / 104.679 / 103.693, recall 0.4170 / 0.4170 / 0.3993), and 0.9995
  loses a mark by crossing recall 0.40. Keep both flags on: they are the only way to separate
  "the decay is wrong" from "the checkpoint is early".
- **Sampler scans have a budget, and the budget is a rule, not a preference.** 13 variants in
  round 1 plus 6 in round 2 is already the edge of fitting dev: the hidden test uses a different
  seed *and* a different reference set, so a 0.01 difference in accuracy at n=500 is noise, and
  every extra variant narrows the margin for no certain gain. Standing rules, enforced in
  `scan_posttrain.py`: at most **6 variants per round**; **one factor per variant**; seed fixed at
  **0** (never search seeds -- that is fitting the grader, not the model); only the **15-mark
  total** decides, so a variant may win FID and still be rejected; any variant with **recall
  < 0.40** is discarded outright; and every decision uses an **aggregate** metric only -- no
  per-image dev feature, no eyeballing a dev image to pick a variant.

- **An auxiliary classifier head on a class-conditioned bottleneck is solved by reading the
  conditioning, not the pixels.** Shot #2 (`pipeline/arms/aux_head`, `results/ab/SHOT2.md`) put
  `Linear(GAP(bottleneck), 2)` (514 params) on `mid` with `0.1 * CE` on `t/T <= 0.25`. The CE was
  already **5.5e-05 at step 100** and stayed there for 19,900 steps: `e = time(t) + label(y)` enters
  every block including `mid`, so ~50% of the bottleneck's norm is label-driven (measured
  `||dm||/||m|| = 0.42-0.60`) and a linear map recovers the label from that component alone. Give
  the same head a batch conditioned entirely on class 0 and it falls to CE 0.65-0.68 ~ ln 2. The
  control (`--aux-src stem`, head on the unconditioned `in_conv(x)`, 130 params) cannot be solved
  that way and is never solved at all -- CE stays at chance, 0.66-0.75, for all 20,000 steps.
  Both arms scored 11/15 (acc 0.8410 / 0.8510). Before adding an auxiliary objective, check that
  the feature it reads does not already contain the answer; the cheapest check is the CE at
  step 100, not the final metric.
- **Changing the noise parameterisation moves the precision/recall balance, not the marks.** EDM
  (Karras et al. 2022: `sigma` levels, `c_skip/c_out/c_in`, `lambda(sigma)` weight, Heun-25) on the
  same 5.15M UNet, manifest, 1,280 epochs and seed gave FID 92.55 -> **89.6-90.1** and precision
  0.458 -> 0.55-0.58 at three guidance weights -- a real ~2.6 improvement, 120x the FID noise
  floor -- and simultaneously recall 0.4646 -> **0.36**, i.e. below the 0.40 cliff that carries the
  Excellent tier. 10/15 against 11/15. FID 89.6 is still Marginal, so the 12th mark needed 10 more
  FID points that this scheme did not deliver. Two traps: the `lambda(sigma)`-weighted loss is ~7x
  the MSE scale and must never be compared with `losses.csv` of an eps-prediction run; and the Heun
  sampler's wall clock (131 s/1000 vs 75 s/1000 here) is not evidence about cost -- 25 Heun steps
  are 49 denoiser calls, 98 forwards with guidance, against DDIM-50's 100.
- **Arm code lives outside `submission/`.** `train_diffusion.py --code-dir pipeline/arms/<arm>`
  trains with that tree's `unet.py` / `diffusion.py` while the graded package stays frozen; the arm
  is rebuilt into `results/ab/<arm>_sub/` (code + its own manifest + its weights) only for
  evaluation. The default is `submission/`, and a 6-step CPU run with the default was checked after
  the flag was added (`5,145,731` params, 84 keys, ships unchanged). Do not grow `submission/` with
  a sampler that is not being graded.
- **`aux` is a reserved Windows device name, so git cannot index it.** The aux arm was written to
  `pipeline/arms/aux/`; `git add` failed with `open("pipeline/arms/aux/diffusion.py"): No such file
  or directory` while `ls` and Python could read the files, and even `rd`/`trash` refused with
  "the specified device name is invalid". Renaming the directory away (`mv aux _old`, then delete)
  and re-copying it as `pipeline/arms/aux_head/` fixed both. Same trap applies to `con`, `prn`,
  `nul`, `com1`-`com9`, `lpt1`-`lpt9` as any path component. Symptom to recognise: a file that
  exists for every tool except git.

## Historical status (2026-11-24 -- legacy 2,000-row tree, superseded)

**Superseded 2026-11-25.** The graded checkpoint is now `32a70a7e` (1,000 admitted rows, 20,000
steps = 1,280 epochs): accuracy 0.8400, FID 92.5499, KID 0.03310, precision 0.4580, recall 0.4646,
11/15, `results/legacy_1000row/eval_final.json`.

> **Which run is which file.** This tree was evaluated more than once and only one JSON survives.
> `results/legacy_1000row/eval_final.json` holds the `32a70a7e` run: 0.8400, FID 92.55, precision
> 0.4580, recall 0.4646. The **0.8425, FID 89.71, precision 0.4934, recall 0.5409** set quoted below
> is the earlier `db6fbc69` checkpoint of the same tree; its JSON lived in `PROJECT3/results/scan/`
> and is not in this repo, so no surviving file supports those four numbers. Match the numbers
> against the file before quoting either set. Neither is the submission. Everything in this section describes the previous tree
(`PROJECT/`) and is kept as the record of what was measured there; its md5 `db6fbc69` is archived
at `PROJECT/submission/weights.pt`.



- **The legacy tree's result (md5 `db6fbc69`): 11/15.** 5,145,731 params, 19.72 MB, `GUIDANCE = 2.5`,
  50 DDIM steps:
  accuracy 0.8425, FID 89.705741, KID 0.0283, precision 0.493363, recall 0.540929,
  memorization 0.000
  (JSON not kept; see the note above). Everything below is a negative result that did not beat it.
- **Min-SNR loss weighting (round 6, E2) also lost: best 9/15.** Same architecture, same manifest,
  same 40k steps, same seed, only `w(t) = min(SNR,5)/SNR` added: accuracy 0.772-0.788, FID
  103.1-107.5, KID 47.6-55.9e-3, recall 0.388-0.417 -> 8-9/15 across five variants (three guidance
  weights plus two EMA decays). Weighted loss scale is 0.82x by construction, so its loss values
  are not comparable to the shipped run's. Reverted by doing nothing: `submission/` untouched.
  Numbers in README "Round 6"; artefact `PROJECT2/results/experiments/B_ddpm5/`.
- **Retraining the conditioning path (round 3) lost, and the loss is in the weights.** A 40k-step
  retrain with AdaGN/FiLM in every block, a null embedding row, 10% label dropout and cosine lr
  (5,573,059 params, 79.8 min) reached only KID 38.0e-3 / recall 0.518 at 250/class against
  **28.4e-3 / 0.594** for the shipped 40k model at the same n. The FiLM layers did learn and the
  class signal got stronger (label flip at fixed noise: 30.8% of pixels vs 17.0%), so the cause is
  the mixture-interference of the dropped labels, not the architecture. Null-token CFG is however
  a better *negative* than antithetic on its own weights (KID 38.0 vs 44.2). Numbers in README.md
  "Round 3"; checkpoint at `results/experiments/B_ddpm3/weights.pt` (archived, not shipped).
- **Snapshot selection is now measured, not assumed**: B3's EMA snapshots at 10k/20k/30k/40k gave
  7/9/9/10 marks and KID 42.6/37.8/38.9/38.0 at 250/class — monotone in steps, so no mid-training
  checkpoint was ever being wasted.
- **11/15 is confirmed to be the ceiling of the shipped checkpoint (round 4, 1000/class).**
  w=3.5 -> accuracy 0.8635, FID 91.95, recall 0.4646, 11 marks. w=4.0 -> 0.8820, FID 97.02,
  recall 0.4613, 11 marks. Accuracy saturates at 0.882, short of the 0.90 tier, while FID, KID and
  recall all get worse than w=2.5. **w=4.5 scores 0/15**: the evaluator reports a determinism
  violation (`valid: false`), which is a contract failure rather than a quality one.
- **The code now supports both guidance forms and the shipped path is provably unaffected.**
  `unet.py` takes `cond="add"|"film"` and `diffusion.py` takes `null_token`; both are recorded in
  the checkpoint, so the 11/15 payload still loads with `strict=True`. Verified: old code vs new
  code on the shipped weights is **bit-identical** (`torch.equal` true) and the 11 tests pass.
- **Audit done and frozen.** 200 crops labelled by the human: 69 good / 131 bad,
  precision **0.345**, Wilson 95% **[0.283, 0.413]** (cat 0.413, dog 0.271). Taxonomy and the
  sharpness cross-check that rules out judging artefacts are in `DATA_CARD.md` §3. The manifest
  was not modified afterwards. Decision (human, 2026-11-23): keep the manifest, report the number
  honestly, spend the remaining time on eval + report rather than a rebuild that the evidence
  says would buy little.
- **Sampler sweep raised the score from 8/15 to 11/15 with zero training.** Weights unchanged
  (`submission/weights.pt`, 5.15M params, 19.72 MB); only `diffusion.py` / `generate.py` changed.
  `scan_posttrain.py` (8 variants, official evaluator, 250/class, seed 0) found that antithetic
  guidance `eps = eps(1-y) + w*(eps(y) - eps(1-y))` is monotone up to w=2.5 and then turns over.
  Shipped: **w=2.5, DDIM 50** -> accuracy **0.8425** (cat 0.804, dog 0.881), FID **89.705741**, KID 0.0283, precision 0.493363, recall **0.540929**,
  memorization 0.000,
  **11/15** (Good / Marginal / Excellent), 40.1 s per 1000 images on an idle GPU, violations `[]`.
  Table + reasoning: README.md "Reproducing each number"; artefacts: `results/scan/`.
- **Sampler sweep round 2 (1000/class, seed 0): no variant beat the constant weight.** Dropping
  guidance from the high-noise steps -- the "early steps decide layout, so only guide late"
  hypothesis -- costs 0.08-0.12 accuracy and a whole tier (`late15` 0.7615, `ramp25` 0.7470,
  `ramp15` 0.7215 vs 0.8425), and makes FID *worse* (94.4-96.9 vs 89.7). Class identity is fixed
  while the image is still noise. Per-class weights (cat 2.75 / dog 2.25) move every metric by less
  than its own spread and keep 11 marks, so the shipped `GUIDANCE = 2.5` stays a single constant.
  Conclusion: **11/15 is this checkpoint's ceiling under sampler-only changes** -- the next mark
  needs accuracy >= 0.90 or FID <= 80, both of which are data/capacity limits, not sampling ones.
- **The 60k experiment (7/15) stays archived** at `results/experiments/B_ddpm2/weights.pt` and
  belongs in "What did not work".

- Starter-kit tests: 26 passed. Template evaluator: accuracy 0.5, FID 440.81, 1/15 (placeholder).
- Full-pool YOLO scan finished: 76,250 images, 0 errors, 23,794 with a cat/dog detection
  (cat 10,432 / dog 14,142 / both 780), 18,640 with exactly one detection.
- `submission/manifest.csv` is final: 2,000 rows (1,000 cat / 1,000 dog), `check_manifest`
  `errors=[]`, `warnings=[]`, 2,000 distinct pool images, no rotation, every row passes the three
  invariants and `h/w <= 1.4`. Candidate pool kept: 2,105 cat / 2,716 dog. **Frozen** -- the audit
  can be drawn from it.
- `pipeline/build_manifest.py` is resumable and now uses `h = max(w, vertical extent)`; OCR runs on
  every crop (12 threads), which removed the prints/screenshots CLIP could not see.
- **Model A** (conditional DCGAN, projection D, 1.48M params): 1000/class accuracy 0.600,
  FID 159.03, KID 0.1078, precision 0.541, recall 0.0642, memorization 0% -> **2/15**
  (`results/eval_baseline.json`). It is a valid fallback: every red line passes
  (1.48M params, 5.7MB, deterministic, single/mixed difference 0.09 < 0.2).
- **Model B** (class-conditional DDPM, `submission/unet.py` + `submission/diffusion.py`, 5.15M
  params, 40k steps) with antithetic guidance w=2.5 at 50 DDIM steps: 1000/class accuracy 0.8425,
  FID 89.705741, KID 0.0283, precision 0.493363, **recall 0.540929**, memorization 0%,
  40.1 s/1000 ->
  **11/15**. Without guidance the same weights give 0.6365 / 100.44 /
  0.0339 / 0.459 / 0.4325 -> 8/15, so the sampler is worth 3 marks.
- DDIM step count was measured, not assumed: at fixed 250/class, 35 -> KID 35.6, 50 -> 35.3,
  75 -> 35.3, 100 -> 37.0 with recall falling 0.534 -> 0.394 (a mark lost). Ship 50.
- `tests/`: 11 passed (four required plus crop/contract extras and a diffusion-loop test).
- `results/diagnosis_recall.md` records why the first generator scored recall 0.008, with numbers.
- `README.md`, `DATA_CARD.md`, `AGENT_LOG.md` (human fills the three checks), `COMMIT_PLAN.md`,
  `.gitignore` exist.
- Old crop constants `K=1.81`, `OX=-0.01`, `OY=0.10` were medians of per-image dev measurements
  and are retired under brief §5.15; the manifest uses fixed geometry instead.

## PROJECT3 redo workspace (started 2026-11-24)

`D:\CISC3024\PROJECT3` is a separate git repo that redoes the data line from the raw pool. Its
contract is a phased redo plan kept outside this repository. Short version:

1. **Phase 1** rebuilds a *draft* manifest: a candidate pool, not a dataset. It is never submitted,
   evaluated, or trained on, except as the frozen `results/draft_pre_review.csv` control.
2. **Phase 1.5** admits **500 human-verified good rows per class** (1,000 total) out of that pool
   through a `sha256(f"{seed}:{pool_id}")` queue stratified over brightness x face_aspect --
   never a score ranking, never "sample 500 and let the rest through". 1,000/class is a stretch
   goal and is not promised anywhere. **Done 2026-11-25 04:10**: 25 sheets, 2,500 rows shown,
   `Y` = cat 507 / dog 501 (admission rate 0.461 / 0.385), first 500 per class admitted ->
   `submission/manifest.csv` 1,000 rows, md5 `66f60cfbfaf7f33550e2cfac11bb7d82`,
   `check_manifest` 0 errors / 0 warnings. Numbers in `results/manifest_v1/admission_stats.json`.
   Eight `Y` rows fell past the 500 line and are not in the manifest; that is the pre-declared
   `yes[:per_class]` rule, not a loss.
3. **Phase 2** freezes the admitted manifest and runs `audit.py`'s own 200-row draw, labelled one
   session later with the old verdict hidden. That number is **post-admission re-label
   consistency** (not precision; the Wilson band is a consistency interval) and is never compared
   to the legacy 0.345 or the Phase 0 0.57 probe, which stay in the methodology section.
4. **Phase 4**'s first shot is still the "swap only the manifest" A/B (draft vs admitted); it is
   not optional and it is not replaced by a model-side ablation.

`submission/weights.pt` **was** a fixed compatibility reference (md5
`db6fbc698d9907d5abc9e5204f2f86a1`, mode 0444) until 2026-11-25 14:51, when it was replaced by the
admitted-arm checkpoint **md5 `32a70a7e12a3afd30c4fcb3ed2eca2a9`** (20,000 steps, 1,280 epochs, seed 0)
so that the graded weights and `submission/manifest.csv` (md5 `66f60cfb`) describe the same 1,000 rows.
Symptom of the old state: the shipped weights had been trained on 2,000 rows, so 1,115 training images
were not manifest rows — a §2/"every training image is a row" inconsistency, not a quality one.
The legacy checkpoint is still readable at `PROJECT/submission/weights.pt` (0444); do not delete it.
In-place official evaluation of that package: `results/legacy_1000row/eval_final.json` (11/15, violations []).

**Shot #2 (2026-11-25, `pipeline/p5_shot2.py`, record in `results/ab/SHOT2.md`)**: two proposals,
four arms, all at 20,000 steps / 1,280 epochs / seed 0 on `submission/manifest.csv`. Neither reached
12 marks and the package was not changed.

| arm | acc | FID | KID | prec | recall | marks |
|---|---|---|---|---|---|---|
| `admitted_e20k` (superseded) | 0.8400 | 92.55 | 0.0331 | 0.4580 | 0.4646 | 11 |
| `aux_head` (bottleneck CE, 514 p) | 0.8410 | 95.02 | 0.0344 | 0.4491 | 0.4060 | 11 |
| `aux_stem` (stem CE, 130 p, diagnostic) | 0.8510 | 93.11 | 0.0342 | 0.4978 | 0.4281 | 11 |
| `edm_heun` (EDM + Heun-25, w=2.5) | 0.8495 | 90.14 | 0.0457 | 0.5763 | 0.3617 | 10 |
| `edm_heun` (w=2.0) | 0.8270 | 89.61 | 0.0454 | 0.5675 | 0.3628 | 10 |
| `edm_heun` (w=1.5) | 0.7725 | 93.45 | 0.0492 | 0.5520 | 0.3385 | 9 |

The auxiliary CE collapses to 5.5e-05 by step 100 (the bottleneck is class-conditioned, so the head
reads the label instead of the pixels); EDM buys ~2.6 FID and +0.11 precision but drops recall
below the 0.40 tier. Both go to "What did not work".

## Current status (2026-11-26) -- the shipped model

`submission/weights.pt`, md5 `1e5c9b87`, 5,145,731 parameters, 19.75 MiB, 60,000 steps on the
3,000-row manifest (`submission/manifest.csv`, md5 `a0022d84`, 1,500 per class).

- Official evaluator, 1,000 images per class, seed 0 (`results/ab/eval_mixed60k_n1000.json`):
  accuracy **0.9065**, FID **77.80**, KID 0.0209, precision 0.5243, recall **0.5044**,
  memorization 0.00%, **13/15**.
- Seed 1 (`..._seed1.json`): 0.8955 / 76.53 / 0.4812, **12/15**. The honest headline is 12-13.
- Checks on the shipped weights: `deterministic=True`, `seed_sensitive=True`, single-class 0.96
  against mixed 0.87 (limit 0.2), manifest 3,000/3,000 valid rows.
- `DATA_CARD.md` is one page: 0.670 [0.602, 0.731] on the 200-cell audit, 0.898 on the human arm,
  0.643 on the rule-selected arm. Report body is six pages.

The two blocks above (the 2026-11-24 status and the `PROJECT3` workspace) describe the 1,000-row
line and a separate working copy. They are kept as the record of what was measured there; neither
is the shipped model.
