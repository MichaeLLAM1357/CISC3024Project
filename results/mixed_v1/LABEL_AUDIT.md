# LABEL_AUDIT.md — who produced which verdict, against which line

This file records the provenance of every label in the three sheets and what the three sheets can
and cannot support. It exists because two of the three were filled in one sitting under time
pressure, and because the audit vocabulary was extended by one token (`humanhead`) that is not in
`starter_kit/audit.py`'s example list, with `ears_cut` and one unclassified cell appearing on the
spot-check sheet. Numbers are machine-copied from
`results/mixed_v1/label_stats.json` (produced by `pipeline/p10_consistency.py`).

**Authorship.** Every `verdict` and every `reason` in `audit/audit.csv`,
`results/mixed_v1/spotcheck_machine.csv` and `audit_legacy_1000row/audit.csv` was written by the
student. No agent, model or classifier produced or suggested a verdict; the project rules
(`LIMITS.md` §1.3, §2) forbid it, and the spot-check sheet was deliberately drawn without printing any model score
next to the cells so that nothing could anchor the judgement.

---

## 1. The vocabulary, and where each token comes from

`audit.py` (docstring, line 11) asks for a `verdict` of good / bad and, for bad ones, a `reason`,
and gives four *examples* ("cartoon, two heads, head too small, not upright"). It does not define a
closed set, so the operational definitions below were taken from the specification the audit is
meant to test — handout p.1 "Data specification" and Briefing slide 5, which are word-for-word the
same four bullets.

| token | n (200) | n (machine arm) | n (spot 100) | line it operationalises | literal? |
|---|---|---|---|---|---|
| `product/print` | 21 | 20 | 8 | "prints on products" | yes |
| `cartoon/drawing` | 10 | 10 | 2 | "no cartoons, drawings, paintings, renders" | yes |
| `statue/toy` | 3 | 2 | 3 | "toys", "renders" | yes |
| `humanhead` | 6 | 6 | 8 | "exactly one **animal** head" | **derived** |
| `two_head` | 2 | 2 | 0 | "exactly one animal head" | yes |
| `head too small` | 5 | 4 | 0 | "the head (with ears) takes about 60–80% of the side" | yes |
| `not upright` | 6 | 6 | 1 | "centred and roughly upright" | yes |
| `blurry/upscaled` | 2 | 2 | 0 | "at least 32 px wide in the pool image (no upscaled blurry heads)" | yes |
| `text/watermark` | 6 | 5 | 1 | "a real photo … of exactly one animal head" | **derived** |
| `wrong_animal` | 1 | 1 | 1 | "label 0 = cat, 1 = dog" | **derived** |
| `ears_cut` | 0 | 0 | 1 | "takes about 60–80% of the side" (ears inside the crop) | yes |
| `unclassified` | 0 | 0 | 1 | — (cell written as `unknow`) | n/a |

The three derived tokens are declared rather than smuggled in:

* `humanhead` — a crop whose centre landed on a person's head. The specification says *animal*
  head, so this is a failure of the same bullet `two_head` uses, but it needed its own word
  because "two heads" would mis-describe it and "wrong animal" would imply a third species. It is
  the only new token the second sitting added, and it accounts for 6 of the 60 machine-arm
  failures — 10%.
* `text/watermark` — an overlaid caption or stock-photo watermark. No bullet names watermarks;
  it is treated as "not a photo of exactly one animal head" because the overlay covers the head.
* `wrong_animal` — the crop shows the other class. Counted as bad because the label is part of the
  row, not only the pixels.

**The spot-check sheet now carries its reasons too** — 26/100 cells are bad and all 26 have one, so
the sheet supports an error mix and not only a good rate. Written as shorthand in the sheet and
counted here as: `product/print` 8, `humanhead` 8, `statue/toy` 3, `cartoon/drawing` 2,
`text/watermark` 1, `wrong_animal` 1, `earcut` 1 (`ears_cut`), `not upright` 1, and one cell written
`unknow` that is counted as *unclassified* and not reinterpreted. Normalising the shorthand
(`earcut` → `ears_cut`, `not upright` → `not_upright`, `text/watermark` → `text`) is a spelling
change only; no verdict and no word was rewritten. Split by kind: **23 of the 26 are semantic**
(print, human head, statue, cartoon, watermark, wrong class), 2 geometric, 1 unclear — the same
inversion the 200-cell audit found (§2), on a draw the audit never saw.

Four bad rows in `audit/audit.csv` **had** an empty `reason` — `s008168`, `s034656`, `s046531`,
`s051499`. That was a gap: handout p.1 says "with a short reason for every bad one", and Briefing
slide 7 repeats it. Closed by filling them; all 66 bad rows and all 26 spot-check bad rows now carry
one, and the column is empty on every good cell. See §4.

---

## 2. What the three sheets say

Wilson 95% intervals. "good rate" = share of cells judged good. This is a *re-label* quantity on
rows that were already in the manifest, not a precision of the pipeline.

| sheet | frame | n | good | good rate | 95% CI |
|---|---|---|---|---|---|
| `audit_legacy_1000row/` | 1,000-row human manifest | 200 | 148 | 0.740 | [0.675, 0.796] |
| `audit/` — human arm | same 1,000 human rows | 59 | 53 | 0.898 | [0.795, 0.953] |
| `audit/` — machine arm | the 2,000 machine-gated rows | 141 | 81 | 0.574 | [0.492, 0.653] |
| `audit/` — pooled | 3,000-row mixed manifest | 200 | 134 | 0.670 | [0.602, 0.731] |
| `spotcheck_machine` | 2,000 machine rows, second salt | 100 | 74 | 0.740 | [0.646, 0.816] |

Two things follow that were not obvious before the labels existed:

1. **The two arms are not the same population, by a wide margin.** 0.574 against 0.898, intervals
   that do not overlap. Whatever the gate was doing, it was admitting crops that a human looking at
   the same specification rejects roughly 36% of the time (0.643 pooled over both machine draws).
   This is consistent with the 0.345
   precision of the fully automatic draft and inconsistent with reading 0.740 as a property of all
   3,000 rows.
2. **The two machine samples disagree with each other.** 74/100 against 81/141 on the same
   2,000-row frame — 0.740 vs 0.574, a difference of 0.166 (SE 0.061, **z = 2.74**), with only 6
   cells in common (they agree 6/6). Either the second sample is a run of easier crops, or the
   standard moved between the two sittings. §3 says which. Pooling the two draws, because they are
   independent samples of one frame: **155/241 = 0.643 [0.581, 0.701]**, and that pooled figure is
   what the data card quotes as the machine frame's precision, with both draws left visible.

---

## 3. The 59 crops that were judged twice

The legacy draw sampled 200 of the 1,000 human rows; the new draw put 59 of those same rows back.
They are the only repeat measurement in the project, and they are not flattering:

| | pass 1 (legacy audit) | pass 2 (mixed audit) |
|---|---|---|
| good rate on the same 59 crops | 0.678 | **0.898** |
| agreement with the other pass | 42/59 = 0.712 | |
| Cohen's κ | **0.196** | |
| flips | 15 `bad → good`, 2 `good → bad` | |

κ = 0.196 is "slight" agreement on the Landis–Koch scale. The 0.712 raw agreement looks acceptable
only because both passes call most cells good; once that is corrected for, the two sittings are
close to unrelated on the cells that matter. And the flips are **one-directional**: 15 of 17 go
towards leniency, moving the good rate from 0.678 to 0.898 on identical pixels.

That has three consequences, in order of how much they cost:

1. **The 0.740 in the report is not reproducible at that margin.** Section 4 of the report calls it
   "re-label consistency". A third pass over 59 of the same crops reproduces it at κ ≈ 0.2. The
   sentence has to be weakened, or the repeat has to be adjudicated (§4).
2. **The machine arm's 0.574 is measured with a softer ruler than the human arm's was.** Pass 2
   moved *towards* good on the cells both passes saw, so 0.574 is if anything an **upper** bound on
   how the pass-1 standard would have judged the machine rows.
3. **Agreement with the teaching team is what is graded** (handout p.1: "your marks depend on how
   well your labels agree with ours"), and a standard that drifts by 15 cells in 59 is the main
   threat to that mark — more than any single verdict.

### Adjudication sheet — filled, and it settles the direction

`consistency_17.png` + `consistency_17.csv` re-render the 17 disagreement cells in `pool_id` order
with **no verdict printed and no arm marker**, so the third pass is a measurement rather than a
reminder of the first two. `pipeline/p10_consistency.py` regenerates both from the two CSVs, and it
refuses to overwrite a sheet that already carries verdicts.

Result: the third pass calls **14 of the 17 good, 3 bad, and sides with pass 2 on 16 of 17**. So the
17 disagreements are not noise around a fixed standard — they are a standard that moved, in one
direction, towards leniency, and the settled standard is **pass 2** (the one used in `audit/`). The
practical consequences:

- 0.670 is measured on the standard the student actually holds; 0.740 was produced by a stricter
  first pass that a later, slower look does not reproduce. Both stay on record; the card uses 0.670.
- Because pass 2 is the *more permissive* ruler, **0.574 for the machine arm is if anything an upper
  bound** on how pass 1 would have judged those rows.
- Adjudication was only possible for the 59 crops that appeared twice, and those are all human-arm
  rows. It says nothing about the 141 machine cells; there is no repeat measurement for them, so the
  machine arm's 0.574 carries sampling error but no measured label error.

## 4. Open items

| # | item | state |
|---|---|---|
| 1 | 4 bad rows missing a `reason` (`s008168`, `s034656`, `s046531`, `s051499`) | **closed** — filled by the human: `head too small`, `not upright`, `product/print`, `cartoon/drawing`. All 66 bad rows now carry a reason, and the column is empty on all 134 good ones |
| 2 | 17 repeat cells unadjudicated | **closed** — third blinded pass sides with pass 2 on 16/17 (§3) |
| 3 | `audit/audit.csv` row order had been re-sorted by an editor, so `index` no longer matched the sheet order | **closed** — restored to `index` order, content untouched (same 200 `(index, pool_id, verdict, reason)` tuples before and after), md5 now `bafda409`; column set is exactly `audit.py`'s nine fields, no extras |
| 4 | audit is drawn from the **final** manifest | **decided** — the 3,000-row manifest ships, so 66/200 bad describes the delivered training set. Written into `DATA_CARD.md` §3 rather than left for a reader to find |
| 5 | spot-check `reason` column was empty on all 100 cells | **closed** — filled by the human: 26 reasons on the 26 bad cells, none on the 74 good ones. Distribution and vocabulary mapping in §1; counted by `p10_consistency.py` into `label_stats.json` → `spotcheck_reason_counts` |
| 6 | `spotcheck_machine.csv` row order was re-sorted by an editor (bad cells first), so `index` no longer followed the sheet | **closed** — restored to `index` order 0–99 once the editor released the file, content untouched (same 100 `(index, pool_id, verdict, reason)` tuples before and after), md5 `67ced284` → `7a6a2ec3`. Column set is `audit.py`'s nine fields, no extras |
| 7 | the machine frame has two draws with different good rates | **recorded** — 0.574 (141 cells) and 0.740 (100 cells), pooled **0.643 [0.581, 0.701]** over 241. Both on record in `DATA_CARD.md` §3; the later draw carries the more lenient standard, so the pooled figure is not a correction upwards |

## 5. Regenerating

```bash
python pipeline/p10_consistency.py     # label_stats.json (both audit arms, the 100-cell spot
                                       # check with its reasons, the pooled machine frame),
                                       # consistency_17.{png,csv}
python pipeline/p9_spotcheck.py        # the 100-cell machine sheet (salt fixed before verdicts)
```

`label_stats.json` keys that back the sentences above: `audit_mixed_by_arm`, `spotcheck_machine`,
`spotcheck_machine_by_class`, `spotcheck_reason_counts`, `machine_frame_two_draws`, `repeat_frame`,
`adjudication`.
