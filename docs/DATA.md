# Data: from 76,250 unlabelled pool images to 3,000 training rows

Every count below is copied from `results/manifest_v1/manifest_stats.json`,
`results/manifest_v1/admission_stats.json` and `results/mixed_v1/mixed_stats.json`; the audit
statistics come from `results/mixed_v1/label_stats.json`. The pool itself is course data and is not
redistributed.

## 1. The funnel

| Stage | In | Out | Dropped | Rule |
|---|---|---|---|---|
| 0 pool scan, YOLO11x | 76,250 | 23,794 | 52,456 | cat or dog box at confidence ≥ 0.25 |
| 1 exactly one detection | 23,794 | 18,640 | 5,154 | two or more boxes rejected |
| 2 head geometry | 18,640 | 8,230 | 10,410 | `w ≥ 32 px`, `h/size ∈ [0.60, 0.80]`, `h/w ≤ 1.4` |
| 3 CLIP photo vs art or print | 8,230 | 5,517 | 2,713 | style score < 0.50 rejected |
| 4 OCR text and watermark | 5,517 | 4,840 | 677 | text area ≥ 0.5% of the crop |
| 5 pHash deduplication | 4,840 | 4,821 | 19 | hamming distance ≤ 4, sharper kept |
| 6 human quota admission | 2,500 | 1,000 | 1,500 | only `human_ok=Y`, first 500 per class in queue order |
| 7 rule-selected arm | 2,421 | 2,000 | 421 | 1,000 per class, `sha256(0:pool_id)` order, no verdict |

The arithmetic closes at every boundary:

```
76,250 − 52,456 = 23,794
23,794 −  5,154 = 18,640
18,640 − 10,410 =  8,230
 8,230 −  2,713 =  5,517
 5,517 −    677 =  4,840
 4,840 −     19 =  4,821
 4,821 = 1,000 admitted + 1,400 looked at and rejected + 2,421 never judged
```

`check_manifest` on the shipped file: **3,000 rows, 0 errors, 0 warnings, 3,000 distinct pool
images, 0 rotated**.

Stage 6 is the only stage a machine cannot do. Stage 7 is the only stage where a machine replaced
it, and it is the one this project had to argue about.

## 2. Crop geometry is derived, not fitted

The square side comes from a head box measured by a face localiser, widened to leave room for ears.
With `w` the box width and `v` its height:

```
w' = 1.20 w ,  v' = 1.30 v ,  h = max(w', v') ,  size = max(h / 0.70, 40)
```

The constants were fixed in code before any label existed, as were the two rejection thresholds
(`min_style = 0.50`, `min_sharpness = 20`). The 70% head-to-side target sits inside the handout's
60–80% band with 5% of margin on each side. Because the crop is a pure function of the detected box,
the evaluator can rebuild every training image from `manifest.csv` alone.

## 3. Admission was a queue, not a ranking

Two per-class queues were ordered by `sha256(seed:pool_id)` inside nine brightness × face-aspect
strata and drawn round-robin, so what was reviewed was whatever the queue served — never a model's
ranking. That matters for the audit: if the human had admitted the rows a score liked best, the
human arm would not be a sample of anything.

- 2,500 rows drawn onto 25 sheets, 24 opened: denominators 1,100 cat and 1,300 dog.
- 507 cat and 501 dog marked `Y` → admission rates **0.461** and **0.385**.
- Eight `Y` rows fell outside the 500/class quota and are not in the manifest.
- **A blank cell means "looked at, not admitted" and stays in the denominator.** Only an explicit
  `Y` can enter the manifest. This one convention is the difference between 0.461 and 0.488 on the
  cat side; the agent got it wrong once and the report says so.

The 1,500 rows the human rejected are not noise. They are the measure of what the rule stack could
not see, and they are why the machine arm is worth pricing.

## 4. The machine arm: the same rules, no verdict

The 2,421 candidates no human ever judged became stage 7 — the 2,000 needed, plus 421 unused. Order
is the same `sha256("0:" + pool_id)` key the admission queue used, so this is not a score ranking
either. No threshold moved: these are crops that had already passed every machine gate and simply
had no quota left.

| | cat | dog |
|---|---|---|
| candidates never judged | 1,005 | 1,416 |
| taken | 1,000 | 1,000 |
| spare | 5 | 416 |

The cat side had five rows of slack, which is worth knowing: the machine arm for cats is effectively
the whole remaining pool, not a draw from a large one.

Composition of the shipped manifest: **1,000 human-admitted (33%) + 2,000 rule-selected (67%)**.

## 5. The audit: 200 cells, drawn from the shipped manifest

`starter_kit/audit.py` was run unmodified on `submission/manifest.csv`, so the 200 cells describe
the training set that actually ships. 141 of them land on rule-selected rows and 59 on
human-admitted rows — that 2:1 split is the manifest's composition, not a design choice, and it is
what lets the audit price the automation.

| | n | good | bad | good rate | Wilson 95% |
|---|---|---|---|---|---|
| all 200 cells | 200 | 134 | 66 | **0.670** | [0.602, 0.731] |
| rows a human admitted | 59 | 53 | 6 | 0.898 | [0.795, 0.953] |
| rows a rule selected | 141 | 81 | 60 | 0.574 | [0.492, 0.653] |
| cat | 107 | 73 | 34 | 0.682 | [0.589, 0.763] |
| dog | 93 | 61 | 32 | 0.656 | [0.555, 0.745] |

### How to read the three numbers

The three rates answer three different questions and must not be quoted interchangeably:

- **0.670** is manifest-wide compliance: what fraction of *all* shipped rows meet the specification.
  This is the number the data card is asked for, and it is measured directly on 200 cells, not
  back-computed.
- **0.574** is a *precision estimate* for the 2,000 rule-selected rows. That arm is a uniform draw
  from rows selected by rules alone, which is exactly the population a precision estimate needs.
- **0.898** is *re-label consistency* on rows already approved, not a precision of the pipeline. The
  sample is post-admission, so it cannot be.

### The machine frame was measured twice

A second, independent 100-cell draw under its own salt (fixed before any verdict was written) gave
74/100 on the same 2,000-row frame. The two draws share 6 cells and agree on all 6, so they are two
samples of one population, not two looks at one sample:

| draw | n | good | rate | Wilson 95% |
|---|---|---|---|---|
| inside the 200-cell audit | 141 | 81 | 0.574 | [0.492, 0.653] |
| second draw (own salt) | 100 | 74 | 0.740 | [0.646, 0.816] |
| pooled | 241 | 155 | **0.643** | [0.581, 0.701] |

The gap is 0.166 with SE 0.061, z = 2.74 — larger than sampling noise. The second draw was labelled
after the third blind sitting (see below), i.e. with the looser ruler, so part of the gap is the
ruler and part may be the crops; 100 cells cannot separate the two. Both numbers are reported, the
pooled 0.643 is used for the arm comparison, and the honest cost statement is that a human rejects
**about 36%** of the rule-selected rows.

### The ruler moved: κ = 0.196

An earlier audit drew from the 1,000-row manifest and reported 0.740. Fifty-nine of its cells recur
in the shipped-manifest audit, so 59 crops now carry two verdicts from the same person:

- they agree 42/59 (0.712), but **Cohen's κ = 0.196** — slight agreement, not good agreement;
- the flips are one-sided: 15 bad→good, 2 good→bad;
- on identical pixels the good rate moves from 0.678 to 0.898;
- judged a third time with no verdict printed, **16 of the 17 disagreements** sided with the second
  sitting.

So the standard moved towards leniency between sittings, and 0.740 on the old manifest is not what
the same person says on a slower look. This is a measurement about the labeller, not about the
data, and it is the reason `AGENTS.md` now says a label is not a measurement until it is repeated.

### What the failures are

| token | n (200) | n (machine arm) | what it means |
|---|---|---|---|
| `product/print` | 22 | 21 | a print, mug, cushion, sticker |
| `cartoon/drawing` | 11 | 11 | art, render, painting |
| `not upright` | 7 | 6 | head rotated or lying down |
| `head too small` | 6 | 4 | below the 60–80% head-to-side band |
| `humanhead` | 6 | 6 | a person, not an animal |
| `text/watermark` | 6 | 5 | overlaid text or logo |
| `statue/toy` | 3 | 2 | sculpture, figurine, plush |
| `blurry/upscaled` | 2 | 2 | under 32 px, upscaled |
| `two_head` | 2 | 2 | two animals in the crop |
| `wrong_animal` | 1 | 1 | label is the other class |

**The errors are semantic, not geometric**: 48 of the 60 machine-arm failures are semantic against
3 of 6 on the human arm. A per-pixel noise regression does not care whether the cat is a photograph
or a print; the specification does. Human review was the semantic layer of this pipeline, and the
rule stack has none.

## 6. What I would do next

1. **Stop adding rule-selected rows.** Every one of them is rejected by a human about 36% of the
   time, so the marginal row buys metric points and spends specification compliance.
2. **Build the semantic filter this stack never had.** The seven scalar pose/style signals tried
   for curation top out at AUC 0.634 (see `docs/EXPERIMENTS.md`), so it has to be a classifier, not
   another threshold.
3. **Label to a written rubric, then re-label a subsample.** κ = 0.196 says the current ruler is not
   stable enough to grade a 0.02 difference.

## 7. Files

| File | Contents |
|---|---|
| `submission/manifest.csv` | the shipped 3,000 rows (`pool_id, x0, y0, size, angle, label`) |
| `results/mixed_v1/manifest_human_1000.csv` | frozen copy of the 1,000 human-admitted rows (provenance) |
| `results/review/admission_*.csv` | the 25 review sheets as marked — `Y`, blank, nothing else |
| `results/manifest_v1/candidates_all.csv` | the 4,821 rows that passed stages 0–5 |
| `results/mixed_v1/spotcheck_machine.csv` | the second 100-cell machine draw, with reasons |
| `results/mixed_v1/consistency_17.csv` | the 17 adjudicated cells, verdicts printed only after the third sitting |
| `results/mixed_v1/LABEL_AUDIT.md` | the audit's appendix: vocabulary, provenance of every token, open items |
