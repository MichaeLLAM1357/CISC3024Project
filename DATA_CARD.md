# DATA_CARD

`submission/manifest.csv`, md5 `a0022d84`, 3,000 rows, 1,500 per class. Every number here comes from
`results/*.json` or `audit/audit.csv`; the long version is `docs/DATA.md`.

## 1. What the rows are

1,000 rows (33%) were admitted by a person; 2,000 (67%) were picked by rules and never looked at.

## 2. Pipeline and counts

| Stage | In → out | Dropped | Rule |
|---|---|---|---|
| 0 YOLO11x scan | 76,250 → 23,794 | 52,456 | no cat/dog box at conf ≥ 0.25 |
| 1 exactly one detection | 23,794 → 18,640 | 5,154 | two or more boxes |
| 2 head geometry | 18,640 → 8,230 | 10,410 | ratio 4,231 · border 4,347 · head too small 1,058 · aspect 700 · flat 50 · blurry 23 |
| 3 CLIP photo vs art | 8,230 → 5,517 | 2,713 | style score < 0.50 |
| 4 OCR text | 5,517 → 4,840 | 677 | text ≥ 0.5% of the crop, ≥ 3 chars |
| 5 pHash dedup | 4,840 → 4,821 | 19 | hamming ≤ 4 |
| 6 human admission | 2,400 read → 1,000 | 1,400 | `Y` only, first 500 per class in a sha256 queue |
| 7 machine arm | 2,421 → 2,000 | 421 | same stack, same queue order, no human |
| `check_manifest` | 3,000 | 0 | 0 errors, 0 warnings, 0 duplicates, 0 rotated |

Geometry is derived, not fitted: `size = max(face_w × 1.20, face_h × 1.30) / 0.70`, floor 40 px. Only
`min_style 0.50` and `min_sharpness 20` were tuned.

## 3. Audit: 200 cells from the shipped manifest

| frame | n | good | rate | Wilson 95% |
|---|---|---|---|---|
| all 200 cells | 200 | 134 | **0.670** | [0.602, 0.731] |
| human-admitted rows | 59 | 53 | 0.898 | [0.795, 0.953] |
| rule-selected rows, two draws pooled | 241 | 155 | 0.643 | [0.581, 0.701] |

Drawn by the unmodified `starter_kit/audit.py`, so it describes the submitted training set.
Rule-selected rows were sampled twice (141 cells 0.574; a separate 100-cell sheet 0.740, z = 2.74);
0.643 pools them. The human-arm rate is re-label consistency: the same 59 crops agree at
κ = 0.196, and the later sitting was the lenient one, so these rates are upper bounds.

## 4. What the failures are (66 bad)

`print` 22 · `cartoon` 11 · `not upright` 7 · `head too small` 6 · `human head` 6 · `text` 6 ·
`statue` 3 · `blurry` 2 · `two heads` 2 · `wrong animal` 1. That is 51 semantic against 15 geometric,
and 48 of the 60 machine-arm failures are semantic, because the human review was the semantic layer
and the rules have none. CLIP cannot tell a photograph of a print from a photograph (36 of 66); pose
is unmeasurable (AUC 0.27 to 0.63), so `not upright` stays human.

## 5. What I would do next

1. No rows past 1,000 without a human pass. A rule-selected row is rejected about 36% of the time and
   bought +0.067 accuracy and −14.8 FID.
2. A semantic filter trained on the ~2,400 judgements in `results/review/`, validated
   leave-sheet-out (51 of the 66).
3. Re-derive the box on the visible head rather than shrinking the margin (6 of the 66).
4. Require 48 px of head in the pool image and measure sharpness after the 32×32 resample.

## 6. Files and provenance

`manifest.csv` `a0022d84` · `weights.pt` `1e5c9b87` (5,145,731 parameters, 60,000 steps) ·
`audit/audit.csv` `bafda409` · `manifest_human_1000.csv` `66f60cfb`. Images come only from the course
pool; no downloaded, generated or dev-set image was used. Curation is YOLO11x, CLIP ViT-L-14, a
tflite face localizer, RapidOCR and ImageHash, selecting pool images only. The evaluator's
`classifier.pt` and Inception were never read; the generator trains from scratch.
