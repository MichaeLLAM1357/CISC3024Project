# Experiments: what moved the number, and what did not

All arms run the same program with the same flags and the same seed; only `--manifest` and the step
budget differ. Every row is an official evaluator run at 1,000 images per class, seed 0, and the
JSON it comes from is named in the table.

## 1. The six arms that ship in the report

Rows is the data; passes is `60,000 × 64 / rows`, i.e. how many times each image is seen. The two
move together when the step budget is held fixed, so the arms are arranged to separate them.

| Arm | Rows | Steps | Passes | Acc | FID | KID | Prec | Recall | Marks |
|---|---|---|---|---|---|---|---|---|---|
| draft, 2,000 rows unreviewed | 2,000 | 40k | 1,280 | 0.8435 | 89.69 | 0.0283 | 0.4967 | 0.5387 | 11 |
| admitted 1,000, same passes | 1,000 | 40k | 2,560 | 0.8430 | 95.01 | 0.0386 | 0.4956 | 0.3606 | 10 |
| admitted 1,000, previous delivery | 1,000 | 20k | 1,280 | 0.8400 | 92.55 | 0.0331 | 0.4580 | 0.4646 | 11 |
| mixed 3,000, a third of the passes | 3,000 | 20k | 427 | 0.8375 | 98.77 | 0.0300 | 0.4181 | 0.4812 | 11 |
| **mixed 3,000, passes matched (delivered)** | **3,000** | **60k** | **1,280** | **0.9065** | **77.80** | **0.0209** | **0.5243** | **0.5044** | **13** |
| the same recipe at seed 1 | 3,000 | 60k | 1,280 | 0.8955 | 76.53 | 0.0203 | 0.5575 | 0.4812 | 12 |

Sources: `results/ab/eval_mixed60k_n1000.json`, `..._seed1.json`, `eval_mixed20k_n1000.json`,
`eval_admitted_e20k_n1000.json`, `results/legacy_1000row/eval_final.json`, and the 2,000-row draft from the earlier
project tree (`results/mixed_v1/summary_mixed_n1000.json`).

### Read the third and fourth rows together

The 1,000-row recipe was tuned at 1,280 passes. Three times the rows at 20,000 steps is **427 passes**,
a third of the training the smaller set received, and it scores 0.8375 / 98.77 — more rows at a third
of the passes bought nothing. Hold the passes at 1,280 (60,000 steps) and the same 3,000 rows give
0.9065 / 77.80 / 0.5044.

The gain comes from **more data at the epoch count the recipe needs**, and the earlier 2,000-row
comparison never tested that. That is why this project's own earlier conclusion — "data volume is not
the lever" — was wrong: it was measured on 1,000 ↔ 2,000 rows, and the 2,000-row arm did not have its
passes matched.

### The 13 is not a stable 13

0.9065 sits 0.0065 above the Excellent line at 0.90. One measurement cannot separate a tier from a
lucky draw, so the same recipe was run at seed 1: accuracy 0.8955, FID 76.53, recall 0.4812 → 12
marks. FID lands in the same tier at both seeds (77.80 and 76.53, both Pass) and recall clears the
same 0.40 line at both (0.5044 and 0.4812, both 5 marks), so the one-mark difference is entirely
accuracy crossing 0.90 — which is exactly the number that cannot be trusted to a thousandth.

**The honest headline is 12–13 / 15**, and the report says so in the abstract.

## 2. What did not work

Eight attempts failed or were rejected after measurement. Rows 1–4 ran on the 2,000-row draft
manifest, an earlier data line, so they are the starting point this project inherited rather than
results for the delivered model; row 5 ran on the candidate pool, rows 6–7 on the 1,000-row manifest
at its step budget, and row 8 on the shipped 3,000 rows at a third of the passes.

| # | Attempt | Measured outcome | What it taught |
|---|---|---|---|
| 1 | Conditional DCGAN baseline, 1.48M params | accuracy 0.600, FID 159.03, recall 0.0642 → 2/15 | precision 0.541 with recall 0.064 is mode collapse: a few heads, drawn well |
| 2 | Sampler sweep, 19 variants (DDIM 35–100 steps, late guidance, per-class weights) | no variant beat 11/15; 100 steps cost a mark (recall 0.394) | step count is not the lever; accuracy and FID are set by data and capacity |
| 3 | Guidance `w = 4.5` | accuracy 0.870, recall 0.530, then **not deterministic** → 0/15 | a broken contract zeroes a good metric; drift scales with `2w − 1` |
| 4 | Min-SNR loss weighting, γ = 5 | accuracy 0.772–0.788, FID 103–107, best 9/15 | down-weighting low-noise steps costs appearance, which is what FID measures |
| 5 | Pose filter for curation | seven signals, AUC 0.27–0.63; perfect-filter ceiling 0.663 < 0.70 | pose is not separable in this stack; it stays a human judgement |
| 6 | Auxiliary class head: bottleneck (514 p), and as a control the input stem (130 p) | 0.8410 / 0.8510 accuracy, FID 95.02 / 93.11, both 11/15; bottleneck CE 5.5×10⁻⁵ by step 100, stem CE at chance for 20,000 steps | where the feature is class-conditioned the head reads the label and learns nothing; where it is not, the head cannot learn |
| 7 | EDM noise levels, preconditioning, Heun-25 | FID 90.14, precision 0.5763, recall 0.3617 → 10/15 | a real 2.4 FID and +0.12 precision in the wrong currency: recall fell through the 0.40 tier worth five marks |
| 8 | 3,000 rows at 20,000 steps (427 passes) | accuracy 0.8375, FID 98.77, recall 0.4812 → 11/15 | rows without passes is a smaller budget, not more data |

### Three of these are worth more than their row

**Row 3 is the only attempt that produced better-looking numbers than the delivered model and would
have scored zero.** At `w = 4.5` the accuracy was 0.870 and recall 0.530 on the draft manifest — better
looking than the delivered model's — and the evaluator's repeat check failed, so the run returns 0/15 rather than a lower score.
Guidance multiplies per-step numerical drift by `2w − 1`; at 2.5 the same-seed difference is
8.7×10⁻⁶, which passes.

**Row 6 did what it claimed and moved nothing.** Every residual block is conditioned by
`e = time(t) + label(y)`, so the 8×8 bottleneck already carries the label in 42–60% of its norm; a
514-parameter linear map solves the cross-entropy by reading the conditioning, not the pixels (zero
the label and it is back near ln 2). Move the same objective onto the unconditioned input stem and
the head never learns at all. There is no setting in between, because the conditioning is added
everywhere or nowhere.

**Row 7 is the mirror case**: a real improvement (FID 90.14 against 92.55, precision 0.576 against
0.458) in the wrong currency. Recall fell to 0.3617, below the 0.40 tier worth five marks, and the
total dropped from 11 to 10. Two of the three marked metrics are thresholds, so an arm can improve
two of them and lose marks.

## 3. What the extra rows cost

The delivered manifest is 67% rule-selected. On the audit that means 0.574 against 0.898, with 60 of
the 66 bad cells coming from the machine arm. The metric gain is real — accuracy +0.067, FID −14.8,
precision +0.066 — and so is the specification cost. `DATA_CARD.md` §3 and `docs/DATA.md` §5 price
both; the report states them side by side rather than quoting the 13 alone.

## 4. Reproducing an arm

```bash
# the arm builder writes results/mixed_v1/manifest_mixed.csv (md5 a0022d84)
python pipeline/p6_mixed.py --data-root data

# smoke: 300 steps + the determinism gate
python -u pipeline/p7_mixed.py --stage smoke

# train one or both arms (31 min / 92 min on one RTX 2060)
python -u pipeline/p7_mixed.py --stage train --arms mixed20k
python -u pipeline/p7_mixed.py --stage train --arms mixed60k

# official evaluation, 1,000 per class
python -u pipeline/p7_mixed.py --stage eval --arms mixed20k mixed60k

# the comparison table
python pipeline/p8_mixed_report.py
```

Determinism is part of every run: `p7_mixed.py --stage train` refuses to report a checkpoint whose
same-seed repeat differs, and the evaluator's `checks` block is stored in each JSON.
