# AGENT_LOG — independent checks by the human

Four checks, one per layer of the project: the data, the manifest and its labels, the model, and the
tooling around the scan. `LIMITS.md` §9 asks for three, at least one about data, written when they
happened rather than back-filled; this file has four, two of them about data.

Each check has the three parts §9 names. **Context** is what I asked for and what the agent reported
it had built, written up from the session record. **Check** is a command I ran myself against the
shipped files, without the agent — it is the part that carries the weight. **Outcome** is what came
back, what I concluded, and what changed in `AGENTS.md` or the tests; a check that found nothing says
that too, rather than inventing a defect.

## Contents

| # | Layer | Claim under test | Outcome |
|---|---|---|---|
| 1 | data | a manifest row rebuilds bit-identically through `catdog_eval.data.crop` | 100 of 100 sampled rows matched, no defect |
| 2 | data / pipeline | `check_manifest` is clean and the labels survive a look | 0 errors, 0 warnings; 9 of 50 sampled cells bad to the eye, all invisible to the checker |
| 3 | model | determinism and batch-composition independence | same seed bit-identical; cat-only vs mixed differs by 7.27e-6, inside the evaluator's 1e-4 |
| 4 | tooling | the scan no longer overwrites an earlier checkpoint's JSON | 24 `*_n1000.json`; the two E2 checkpoints hold different metrics |

---

## Check 1 — data: crops rebuilt with the official crop function

**Context.**

> I asked the agent to build `submission/manifest.csv` from the raw pool and explain what each row
> means. It described the whole chain in `build_manifest.py`: one detection, face localizer, fixed
> geometry `size = max(h/0.70, 40)`, three crop invariants, CLIP exclusion, OCR, and pHash dedup.
> Each row has six fields, and training rebuilds the image as
> `crop(pool_id, x0, y0, size, angle)`. The agent also said `p6_mixed.py` checks that those six
> fields are the same as in `candidates_all.csv`. Every angle in the table is 0, so what this check
> is really testing is crop plus resize, no rotation.

**Check (run this yourself).**
> Preconditions: `submission/manifest.csv` is present and the raw pool is mounted read-only at the
> path in `POOL` (the pool is not in this repository — point it at your own copy of `data/`).

```bash
python - <<'PY'
import csv, random, sys
sys.path.insert(0, "starter_kit")
import numpy as np
from PIL import Image
from catdog_eval.data import Pool, pool_dir, crop

POOL = "D:/CISC3024/PROJECT/data"
rows = list(csv.DictReader(open("submission/manifest.csv", encoding="utf-8")))
sample = random.Random(2026).sample(rows, 100)
pool = Pool(pool_dir(POOL))
bad = 0
for r in sample:
    im = pool.image(r["pool_id"])
    mine = np.asarray(im.crop((int(r["x0"]), int(r["y0"]),
                               int(r["x0"]) + int(r["size"]),
                               int(r["y0"]) + int(r["size"])))
                      .resize((32, 32), Image.Resampling.LANCZOS))
    ref = np.asarray(crop(im, int(r["x0"]), int(r["y0"]), int(r["size"]), float(r["angle"])))
    bad += not np.array_equal(mine, ref)
print("mismatches:", bad, "of", len(sample))
PY
```

**Outcome.**
> I ran this on the shipped manifest with the read-only pool at `D:/CISC3024/PROJECT/data`.
> The 100 sampled rows all rebuilt bit-identically to the arrays from `crop()`, with zero
> mismatches. I found no defect and changed nothing in `AGENTS.md` or the tests.

---

## Check 2 — data/pipeline: manifest validity and label sanity

**Context.**

> The agent reported 1,500 per class, `errors=[]`, no overlaps or duplicate rows, 3,000 distinct
> pool images, and angle 0. It also corrected two earlier statements that did not match the
> delivery: the 1,000 per class figure, and the `AGENTS.md` rule that every row needs
> `human_ok=Y`, while the shipped manifest has 2,000 rows nobody looked at. The cost of that
> amendment is 0.574, 0.898 and 0.643.

**Check (run this yourself).**

```bash
CATDOG_DATA=D:/CISC3024/PROJECT/data python - <<'PY'
import sys
sys.path.insert(0, "starter_kit")
from catdog_eval import check_manifest, default_pool
r = check_manifest("submission/manifest.csv", default_pool())
print("errors:", r["errors"]); print("warnings:", r["warnings"]); print("stats:", r["stats"])
PY
```

> Then draw 50 rows myself and look at the labels at 128×128 — a label flip is invisible to
> `check_manifest`.

**Outcome.**
> `check_manifest` returned no errors and no warnings. It reports 3,000 valid rows,
> 1,500 cats and 1,500 dogs, all from distinct pool images and with angle 0. The shipped
> manifest is 1,500 per class, which includes the machine-selected rows alongside the rows
> I admitted, so the 1,000 per class in the context above is out of date. The script wrote
> `outputs/AGENTLOG_CHECKS/CHECK2_50_labels_128.png` with 50 rows at 128x128, and I looked
> at all of them. Nine cells were bad: 02 bad print, 04 toy/statue, 07 drawing,
> 18 unknown animal, 33 product, 34 print, 42 not upright, 47 drawing, 49 not upright.
> That is 9 of 50 in this sample. `check_manifest` cannot see any of these, which is why I
> did the visual pass. I changed nothing in `AGENTS.md` or the tests.

---

## Check 3 — model: determinism and batch-composition independence

**Context.**

> The agent said `submission/generate.py` loads one model with 5,145,731 parameters, samples with
> 50 DDIM steps at η=0 and guidance 2.5, and uses GroupNorm because BatchNorm would let one
> sample depend on the others in the batch. It claimed three things: same seed gives the same
> image, a different seed gives a different image, and row 0 of a mixed batch equals the same
> row generated on its own. The last one was only tested with two 4-row batches, where it came
> out True. It also quoted the evaluator's 0.96 single class against 0.87 mixed, a 0.09 gap
> under the 0.20 limit.

**Check (run this yourself).**
> Runs as shipped: `submission/weights.pt` is the graded checkpoint, md5 `1e5c9b87…`.

```bash
python - <<'PY'
import sys, torch
sys.path[:0] = ["submission"]
from generate import load_model, generate

m = load_model(torch.device("cpu"))
cats = torch.zeros(4, dtype=torch.long)
mixed = torch.tensor([0, 1, 0, 1, 1, 1])

a = generate(m, mixed, 5); b = generate(m, mixed, 5)
c = generate(m, mixed, 6); d = generate(m, cats, 5)

print("same seed identical:", torch.equal(a, b))
print("different seed differs:", (a - c).abs().max().item() > 1e-3)
print("cat-only vs mixed, same seed, same label:", torch.equal(d[0], a[0]))
PY
```

**Outcome.**
> Same seed with the same batch gave bitwise identical images, and a different seed gave
> different images. The cat-only versus mixed comparison printed False. The largest
> absolute difference between those two outputs was 7.27e-6, and `torch.allclose` with
> atol 1e-4 returned True. Index 0 has the same seeded noise in both calls, the negative
> label is 1 in both, and the UNet uses GroupNorm, so no other sample should enter the
> result. `torch.equal` compares every bit, so a few ULPs are enough to print False. CPU
> kernels pick a different reduction order for a batch of 6 than for a batch of 4, and the
> difference grows over 50 DDIM steps to 7.27e-6. The evaluator checks with a 1e-4
> tolerance and reports 0.96 single class against 0.87 mixed, a 0.09 gap under the 0.20
> limit. I found no batch-composition failure and changed nothing in the model,
> `generate.py`, or the tests.

---

## Check 4 — tooling: does the scan output still overwrite itself?

**Context.**

> The agent told me `scan_posttrain.py` used to write `results/scan/<variant>_n<N>.json`.
> Two checkpoints scanned under the same variant name would land in the same file, and the
> second run would quietly replace the first, leaving the report pointing at overwritten
> numbers. It said the filenames now carry a checkpoint tag and refuse to overwrite.

**Check (run this yourself).**

```bash
# 1. 檔名是否帶 checkpoint 標籤（應為 <variant>__<ckpt>_n<N>.json）
ls -la --time-style=full-iso results/scan/*/*_n1000.json 2>/dev/null | head

# 2. 同名再跑一次，是否拒絕覆寫（summary 應記 refused-to-overwrite）
grep -n "ckpt\|refused-to-overwrite" scan_posttrain.py | head -20

# 3. 同一變體、兩個 checkpoint 的數是否真的不同（不是同一份被蓋掉）
python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("results/scan").rglob("*_n1000.json")):
    d = json.loads(p.read_text()); m = d.get("metrics", {})
    if m: print(p, {k: round(m[k], 6) for k in ("accuracy", "fid", "recall")})
PY
```

**Outcome.**
> The scan folder contains 24 `*_n1000.json` files. The files under `results/scan/E2/`
> carry checkpoint tags in their names, and `scan_posttrain.py:275-280` builds each output
> name from `--ckpt-tag` and returns `refused-to-overwrite` when that file already exists.
> The two E2 checkpoints have different metrics: `contrast200__E2e0995_n1000.json` shows
> accuracy 0.77, FID 106.083651 and recall 0.417035, while
> `contrast200__E2e09995_n1000.json` shows 0.7765, 103.692581 and 0.399336. The overwrite
> fix held, and I made no further change.

---
