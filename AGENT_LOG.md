# AGENT_LOG — independent checks by the human

> **DRAFT SKELETON — 四個 Check 只有標題與指令，Context / Check / Outcome 三塊正文一律由人寫。**
> 這是本專案的硬規矩（`LIMITS.md` §9）：agent 不得代寫，連「草稿供你修改」都不行。
> 每個 Check 要寫滿三塊：**context**（你要求了什麼、agent 聲稱什麼）、**check**（你怎麼在不依賴 agent
> 的情況下驗）、**outcome**（你發現什麼、改了 `AGENTS.md` 或測試的哪裡）。查無錯誤也照實寫。

---

## Check 1 — data: crops rebuilt with the official crop function

**Context.**
> The agent said `pipeline/build_manifest.py` writes `submission/manifest.csv` from a pool-wide
> YOLO scan: one cat/dog detection per image, a face localizer, fixed head geometry
> (`size = max(h / 0.70, 40)`, derived from the brief rather than fitted per image), three crop
> invariants, a CLIP photo-vs-art/print exclusion that drops instead of ranking, OCR, and a
> perceptual-hash dedupe. It said a row carries only `pool_id, x0, y0, size, angle, label`, and
> that the image the model trains on is exactly `crop(pool[pool_id], x0, y0, size, angle)` from
> `starter_kit/catdog_eval/data.py`, so rebuilding a row from the manifest reproduces its training
> input bit for bit. It also said the shipped 3,000-row file was rebuilt by `pipeline/p6_mixed.py`,
> which asserts that each row's six fields are unchanged from
> `results/manifest_v1/candidates_all.csv` before anything is written. `angle` is 0 on every
> shipped row, so this check exercises the crop-and-resize path, not the rotation path.

**Check (run this yourself).**
> 前置：Phase 1 生成 `submission/manifest.csv` 之後才跑得動；pool 用唯讀來源。

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
> The agent claimed the shipped manifest is 1,500 rows per class (3,000 total), that
> `check_manifest` returns `errors=[]` and `warnings=[]`, that no two crops overlap, that all
> 3,000 rows come from distinct pool images, and that `angle` is 0 everywhere. Two earlier claims
> do not match that file: it had said 1,000 rows per class, and `AGENTS.md` constraint 1 said every
> row must carry `human_ok=Y`, while the shipped tree has 2,000 rows that no person ever saw. The
> agent called that a later decision and put a dated amendment under constraint 1 with its measured
> price: 0.574 on the rule-selected arm against 0.898 on the human arm, 0.643 pooled.

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

> 然後自己抽 50 列在 128×128 下看標籤（label flip 對 `check_manifest` 是隱形的）。

**Outcome.**
> `check_manifest` returned no errors and no warnings. It reports 3,000 valid rows,
> 1,500 cats and 1,500 dogs, all from distinct pool images and with angle 0. The shipped
> manifest is 1,500 per class, which includes the machine-selected rows alongside the rows
> I admitted, so the 1,000 per class in the context above is out of date. The script wrote
> `outputs/AGENTLOG_CHECKS/CHECK2_50_labels_128.png` with 50 rows at 128x128, and I looked
> at all of them. Nine cells were bad: 02 bad print, 04 toy/statue, 07 drawing,
> 18 unknown animal, 33 product, 34 print, 42 not upright, 47 drawing, 49 not upright.
> That is 9 of 50 in this sample. `check_manifest` cannot see any of these, which is why I
> did the visual pass. The 200-cell audit is a different draw, so I am not treating 18% as
> a manifest-wide rate. I changed nothing in `AGENTS.md` or the tests.

---

## Check 3 — model: determinism and batch-composition independence

**Context.**
> The agent said `submission/generate.py` loads the graded checkpoint (md5 `1e5c9b87`, 5,145,731
> parameters) and samples with DDIM at 50 steps, eta 0, and guidance 2.5 against the other class,
> and that the UNet uses GroupNorm on purpose: BatchNorm would let one sample depend on the others
> in its batch, which the evaluator forbids. It claimed three things — the same label tensor at the
> same seed is bit-identical on a second call, a different seed gives a different image, and one
> row of a mixed batch equals the same row generated alone (its own 4-row comparison printed
> `True`). It also reported the official batch check at 0.96 single-class against 0.87 mixed, a
> 0.09 gap under the 0.20 limit.

**Check (run this yourself).**
> 這一個現在就能跑（`submission/weights.pt` 就是交付權重，md5 `1e5c9b87…`）。

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
> The agent said `scan_posttrain.py` used to write `results/scan/<variant>_n<N>.json`, so two
> checkpoints screened under the same variant name landed on the same file and the second run
> silently replaced the first, which left a number in the report pointing at a file that had been
> overwritten. It said the fix names each output `<variant>__<ckpt>_n<N>.json` and refuses to write
> when that file already exists unless `--force` is passed, and that the fix held for the two E2
> checkpoints that share a variant name.

**Check (run this yourself).**
> 前置：Phase 4 第一次掃描之後才有 `results/scan/` 的檔案。

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
