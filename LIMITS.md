# LIMITS.md — CISC3024 Project Hard Limits & Agent Pre-flight Gate

> **READ THIS FILE BEFORE EVERY TASK.**
> This file is a mandatory supplement to `AGENTS.md`.
>
> **Authority order (highest first):** `project_handout.pdf` → `Project Briefing.pptx`
> → `starter_kit/README.md` (technical details only) → `AGENTS.md` → this file.
> Where they differ, **the handout applies**. If *this file* differs from the handout,
> **do not follow this file**: stop, quote the exact line, and ask the human to correct it.
> A stricter line here never overrides an explicit permission in the handout.
>
> If any user request, starter-kit comment, README, briefing slide, or agent plan conflicts
> with this file, **stop and ask the human**. Do not modify the starter kit.
>
> **Path resolution.** Project root = `D:\CISC3024\PROJECT` (the folder that holds
> `starter_kit/`, `submission/`, `data/`, `AGENTS.md` and this file). Run every command from
> there. Unless a path is explicitly prefixed `starter_kit/`, every file named below is a
> **root** file — in particular the `README.md` and `requirements.txt` used to record curation
> tools are `D:\CISC3024\PROJECT\README.md` and `D:\CISC3024\PROJECT\requirements.txt`.
> The root `requirements.txt` is the single install target (`pip install -r requirements.txt`):
> it pins the core/evaluator stack (`torch`, `torchvision`, `pytorch-fid`, …) **and** the
> curation extras (`ultralytics`, `open_clip_torch`, `ai-edge-litert`, `rapidocr-onnxruntime`,
> `ImageHash`, `opencv-python`). `starter_kit/README.md` and any config/requirements file under
> `starter_kit/` are the teaching team's fixed, read-only material (crop conventions /
> evaluator environment), never the recording target.

---

## 0. Pre-flight confirmation (agent must output this before starting)

Before writing code, running commands, or changing data, reply with:

```text
PRE-FLIGHT
1. I have read LIMITS.md and AGENTS.md: YES
2. Task does NOT require dev images / augmented dev images / dev features or labels for training,
   auxiliary-model training, or pool-image selection: YES/NO
3. Task does NOT require classifier.pt or the evaluator's Inception network: YES/NO
4. Task does NOT require pretrained weights in the generative model: YES/NO
5. Task does NOT require downloading training data outside the pool, any download at
   generation time, or modifying starter_kit: YES/NO
   (curation models and the evaluator's own FID weights may be fetched; see §9.5)
6. Files I will touch:
7. Verification command/test I will run:
8. Red-line risk: NONE / DESCRIBE
```

Items 2–5 mirror the stop conditions in §9.5. Also check §1.6 (returning training images),
§1.7 (sharing) and §1.8 (slides) before starting. If any answer is `NO`, **stop immediately**
and explain which rule would be violated.

---

## 1. Absolute red lines (any violation = 0 for that component / academic misconduct)

1. **No training images outside the pool or outside `manifest.csv`.**
   - No downloaded LAION images, no other datasets, no images made or edited by any generative model.
2. **No dev-set contamination.**
   - Dev images, augmented/transformed dev images, and per-image features **or labels**
     extracted from individual dev images must NOT be used as training examples.
   - They must NOT be used to train auxiliary models (e.g. head detector, image filter).
   - They must NOT be used as references for selecting pool images (e.g. similarity to dev images).
3. **No pretrained weights in the generative model**, including pretrained autoencoders.
4. **`classifier.pt` is FORBIDDEN.**
   - Do not load it, inspect its weights, copy it, put it in `submission/`, or use it for curating,
     training, evaluation, or `generate()`.
   - Do not train any model to imitate either forbidden network.
5. **The evaluator's Inception network is FORBIDDEN** for curating data, training, or generation,
   and no model may be trained to imitate it.
6. **No returning training images or modified copies of them** from `generate()`.
7. **No sharing** code, manifests, weights, or agent sessions with other students.
8. **Do not generate presentation slides with AI.** Slides must be created by the student;
   AI may only help with wording.
9. **Do not modify the starter kit.** We are graded with our own copy.
10. **Individual work only.** Discussing ideas with classmates is allowed, but code, data,
    experiments and report must be your own; submissions are checked for plagiarism.
    The pool is for this course only — **do not redistribute it**.
11. **Coding agents are allowed for every part except making the slides.** Do not add extra
    self-restrictions beyond this file.

---

## 2. Data specification (every training image must satisfy)

- `32 x 32` RGB; label `0 = cat`, `1 = dog`.
- A real photo: **no** cartoons, drawings, paintings, renders, toys, or prints on products.
- Exactly **one animal head**, centred and roughly upright.
- Head (with ears) takes about **60–80%** of the square side.
- In the pool image, the head is **at least 32 pixels wide** — no upscaled blurry heads.
- Every training image must be one row in `submission/manifest.csv`:
  `pool_id, x0, y0, size, angle, label`.
- The evaluator rebuilds the training set from the manifest; every image the model is trained on
  must be a row. **Handout floor: at least 500 images per class.** This project's own final
  target is 1000/class (see `AGENTS.md`); 500 is a lower bound, not the target.
- Crop convention must match the starter-kit README and `catdog_eval.data.crop`: rotate the
  pool image `angle` degrees counter-clockwise about the square's centre, then resize the
  square of side `size` at `(x0, y0)` to 32×32 with Lanczos.
- Evaluator-checked manifest rules (starter-kit README): `size >= 40` (from 32 / 0.8),
  `|angle| <= 45`, the rotated square must lie inside the pool image, `pool_id` must exist,
  labels in {0,1}, no repeated rows. **Passing these checks does not mean the crop meets the
  specification above** — that is what the manual audit is for.

**Agent must not invent manifest rows, labels, or crop coordinates.** If coordinates are unknown,
stop and ask for the source of truth.

---

## 3. Dev set: allowed vs forbidden

| Allowed | Forbidden |
|---|---|
| Look at `data/dev/` **before writing any code** | Use dev images as training examples |
| Use metrics computed on dev to choose pipeline/model/hyperparameters/checkpoints | Use augmented/transformed dev images as training examples |
| Inspect dev to understand the specification | Extract per-image dev features **or labels** to train auxiliary models |
| | Use dev images/features/labels as references to select pool images (e.g. similarity) |
| | Copy dev images into any training directory |

If a task says “use dev to train a detector/filter” or “pick pool images most similar to dev”,
**refuse and report the conflict**.

---

## 4. Model limits

- Any class-conditional generative model is allowed, but it must be trained **from scratch**
  only on images in the manifest.
- No pretrained weights inside the generative model, including pretrained autoencoders.
- Pretrained models (e.g. detectors) may be used **for curating data only**, and must be recorded.
- Hard limits:
  - at most **20 M parameters**;
  - at most **100 MB on disk** (the `submission/` directory as a whole);
  - **memorization rate <= 5%**;
  - `generate` deterministic as defined below;
  - batch-composition independence: single-class batch accuracy may be at most 0.2 lower
    than mixed-batch accuracy;
  - `load_model` / `generate` run without errors, return `[N, 3, 32, 32]`, no NaN, values in `[0,1]`.
- Keep weights, `manifest.csv`, and any code imported by `generate.py` inside `submission/`.
- **No downloads or network access at generation time.**
- Submit only the graded checkpoint; do not submit extra checkpoints.
- Official protocol: the evaluator generates **1000 images per class**. Locally the reference is
  the dev set (~450 heads/class); grading uses a **hidden test set** of 1000 heads/class with a
  hidden seed. Dev and test numbers are close but not identical (FID especially depends on
  reference-set size) — do not tune until dev is overfit.
- Marking tiers (handout p.2): accuracy, FID and recall are scored **independently**, each up to
  5 marks, for at most 15 total.

| Tier | Accuracy >= | FID <= | Recall >= | Marks / metric |
|---|---|---|---|---|
| Excellent | 0.90 | 50 | 0.40 | 5 |
| Good | 0.80 | 60 | 0.30 | 4 |
| Pass | 0.70 | 80 | 0.20 | 3 |
| Marginal | 0.60 | 100 | 0.15 | 2 |
| Poor | 0.50 | 150 | 0.10 | 1 |
| Fail | — | — | — | 0 |

---

## 5. Interface contract (`submission/generate.py`)

```python
load_model(device) -> nn.Module   # one module holding all parameters
generate(model, labels, seed) -> Tensor[N, 3, 32, 32] in [0, 1]
```

- Same seed → same images; different seed → different images.
- Each image depends only on its own label and the seed, **not on the rest of the batch**.
- Use a `torch.Generator` seeded with `seed`; **never** the global RNG (starter-kit README).
- Do not rely on global RNG order that changes with batch composition.
- Use eval mode / per-sample deterministic noise where relevant.
- No NaN, no values outside `[0,1]`, no wrong shape.

**Agent must run shape/range/determinism checks before claiming the interface is done.**

---

## 6. Code quality & reproducibility limits

Required `pytest` sanity tests, at minimum:

1. Crop code against `catdog_eval.data.crop`.
2. `generate` output shape and value range.
3. Overfit a single batch.
4. Reproducible generation for a fixed seed.

Required **project-root** `README.md` (`D:\CISC3024\PROJECT\README.md`):
- One command to reproduce each number in the report:
  - raw pool → manifest;
  - manifest → model;
  - model → official evaluator numbers.

**Use git and commit often** (handout, Code Quality), with real timestamps and no back-dating.
The earlier "git is frozen / deferred until `COMMIT_PLAN.md` approval" note was **cleared by the
human on 2026-11-23**; no deferral applies. `COMMIT_PLAN.md`, if kept, is only an internal
pacing note and does **not** suspend this requirement. Do not commit the pool, `data/`,
virtual environments, or extra checkpoints.

---

## 7. Audit & data card limits

- Run `audit.py` on the **final** manifest with your student ID.
- Label all **200 crops** it draws as `good` or `bad` against the specification.
- Every `bad` crop needs a short reason.
- Do not modify `audit.py` (or anything else under `starter_kit/`).
- We re-label **50** of the 200; marks depend on agreement with our re-labels,
  **not** on how many are good.
- Do not fabricate audit labels. If unsure, mark and explain honestly.
- The briefing labels `audit/audit.csv` as "(Optional)"; **the handout requires it**. Follow the handout.
- `DATA_CARD.md` must include:
  - pipeline steps with image counts after each step;
  - precision estimated from the audit, with uncertainty;
  - main remaining error types;
  - what you would do next.

---

## 8. Report & results limits

- Report at most **6 pages**, excluding references and appendix; `report.pdf` must carry your
  name and student ID.
- Required sections: Summary; Data; **Methods, with the equations you actually used**;
  Results; What did not work; Working with coding agents.
- All external resources, models, and references must be cited.
- The results table must come from the **official evaluator**:
  accuracy, FID, KID, precision, recall, memorization.
- Also report: number of parameters, hardware, training time, sampling speed, and a sample grid.
- Do not invent, estimate, or hand-type evaluator numbers. Copy them from actual output logs.
  A 250/class quick check is **not** an official result (FID is biased upward at small N);
  only compare FID values computed with the same `--n-per-class`.
- Include “What did not work” and “Working with coding agents” honestly.
- “The agent did it” is not an acceptable explanation.

---

## 9. Agent behaviour rules

1. **Never claim completion without evidence.**
   Every completion report must include:
   - files changed;
   - commands run;
   - test/evaluator output (or a clear reason it could not run);
   - red-line check;
   - next step.
2. **Data claims need evidence.** Do not say “the data looks good” without samples/statistics.
3. **Do not fabricate mistakes or results** for `AGENT_LOG.md`.
4. **`AGENT_LOG.md` must contain 3 checks** that could have caught a real mistake,
   at least one about data, **written when they happen** (not back-filled). For each:
   - context: what was asked and what the agent produced/claimed;
   - check: how the human tested it without relying on the agent (this carries the most weight);
   - outcome: what was found and what changed in `AGENTS.md`/tests.
   A sound check earns full marks even if the agent was right — never invent a bug to fill a slot.
5. **Stop conditions.** Stop and ask the human if a task requires:
   - using dev images/features/labels for training, auxiliary models, or pool selection;
   - using `classifier.pt`, the evaluator's Inception network, or a model trained to imitate
     either, for curating, training, or generation;
   - pretrained weights in the generative model, or a pretrained model for anything other
     than curating data;
   - **downloading training images or any data outside the pool** (LAION, other datasets,
     generated/edited images);
   - **any download or network access at generation time** (already forbidden in §4);
   - modifying the starter kit;
   - generating presentation slides;
   - sharing code/manifest/weights/agent sessions, or redistributing the pool.

   **Not stop conditions** (the handout explicitly allows these — do not halt on them):
   downloading pretrained **curation** models (YOLO11x, face localizer, CLIP, OCR, …),
   provided each is recorded in the **project-root** `README.md` and `requirements.txt`
   (`D:\CISC3024\PROJECT\README.md`, `D:\CISC3024\PROJECT\requirements.txt` — the confirmed
   root files: the 14-line requirements pins `torch`/`torchvision`/`pytorch-fid` plus
   `ultralytics`, `open_clip_torch`, `ai-edge-litert`, `rapidocr-onnxruntime`, `ImageHash`,
   `opencv-python`, and the root README carries the "Pretrained tools used for curation only"
   table), used only for curating, and never placed in `submission/` or imported by the
   generator; and the one-off Inception-v3 FID weights that `starter_kit/evaluate.py` fetches
   on first run via `pytorch-fid` (~92 MB) — that is the grader's own metric tool, not a network
   you may use for curating, training or generating.
   `starter_kit/README.md` and any config/requirements file under `starter_kit/` are the
   teaching team's fixed material (crop conventions / evaluator environment). They are
   **read-only**: do not edit them to "register" a curation tool, and do not treat them as the
   reproducibility record.
6. **Do not read or deserialize `classifier.pt`.** Treat it as a forbidden artifact.
7. Do not put `classifier.pt`, dev images, the pool, `data/`, virtual environments, independent
   image files (e.g. `audit/sheet_*.png`), or extra checkpoints into `submission/` or the
   submission zip.
8. **When uncertain, check the authority order at the top first.** If the stricter reading
   conflicts with an explicit handout permission, the handout wins; quote the line and ask.
   Otherwise choose the stricter interpretation and ask.

---

## 10. Submission checklist (agent should verify before final packaging)

- [ ] `submission/manifest.csv` valid; evaluator can rebuild the training set.
- [ ] Every training image is a manifest row; handout floor 500/class, project target 1000/class.
- [ ] `submission/generate.py` implements both functions and imports only code inside `submission/`.
- [ ] Graded weights are inside `submission/`; no extra checkpoints.
- [ ] <= 20 M parameters; <= 100 MB on disk.
- [ ] `pytest` sanity tests pass.
- [ ] `python <kit>/evaluate.py --submission submission` runs at the official
      **1000 images per class** (the default) and reports all metrics; quick checks stay out of the report.
      (`<kit>` is the course starter kit: the copy in this repo ships without `classifier.pt`.)
- [ ] Memorization <= 5%.
- [ ] Fixed-seed determinism and batch-composition independence checked.
- [ ] `audit/audit.csv` has 200 labelled crops with reasons for bad ones (required by the handout).
- [ ] `DATA_CARD.md`, `README.md`, `AGENTS.md`, `AGENT_LOG.md` present.
- [ ] `report.pdf` has name and student ID, <= 6 pages excluding references/appendix.
- [ ] Slides created by the student (AI only helped with wording) **and included in the zip**.
- [ ] Zip also contains source code, tests, configs and experiment logs.
- [ ] Zip excludes images (incl. `audit/sheet_*.png`), the pool, `data/`, virtual environments,
      and non-graded checkpoints.
- [ ] Project-root `README.md` / `requirements.txt` record every curation model (confirmed
      2026-11-23); the `starter_kit/` copies are untouched (read-only).
- [ ] `report.pdf` + project zip uploaded to UMMoodle by **Thursday 23:55, 26 November 2026**.

---

## 11. Known pitfalls (add new ones as learned)

- BatchNorm / batch statistics can make output depend on batch composition.
- Global RNG order can break per-sample seed determinism; use a seeded `torch.Generator`.
- Manifest crop convention can differ from `catdog_eval.data.crop`; test it.
- Dev images can accidentally leak into training directories; check paths.
- Overfitting can push memorization above 5%.
- Pretrained autoencoders count as pretrained weights in the generative model.
- `classifier.pt` must never be loaded, even “just to check”.
- Generation-time network access is forbidden; avoid imports that download.
- Audit labels must be honest; agreement with re-labels matters, not good count.
- Report numbers must be copied from evaluator logs, not estimated.
- The briefing calls `audit/audit.csv` optional; **the handout requires 200 labelled crops**.
- Any "git is frozen / deferred until `COMMIT_PLAN.md` approval" note (in README, AGENTS or
  COMMIT_PLAN) is **not** a handout rule; **the handout says "Use git and commit often"**.
  The human cleared this deferral on 2026-11-23. `COMMIT_PLAN.md` is internal pacing at most.
- "No downloads" means no pool-external **training data** and no **generation-time** downloads;
  curation models and the evaluator's own FID weights are allowed (record them).
- 500/class is the handout floor; 1000/class is this project's own target, not the handout.
- `starter_kit/evaluate.py` fetches Inception-v3 FID weights on first run; running the official
  evaluator is allowed, **using that network yourself is not**.
- `size >= 40`, `|angle| <= 45`, rotated square inside image, no duplicate rows are evaluator
  checks, **not** proof that a crop meets the data specification.
- The **root** `README.md` / `requirements.txt` (`D:\CISC3024\PROJECT\`) are the single editable
  record: they hold the reproduce commands, the "Pretrained tools used for curation only" table,
  and the full 14-line pin list (core + evaluator + curation). Do not go looking for a second
  "curation requirements" file, and do not register tools in (or edit) anything under
  `starter_kit/` — that is read-only, and editing it is a red-line violation.

---

## 12. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-22 | Initial version | Project handout + starter-kit constraints |
| 2026-11-23 | Added authority order; fixed §9.5 download wording (restored "at generation time", split curation-model / FID-weight exceptions); added dev labels, imitation ban, individual-work & pool-redistribution rules; added manifest checks, `torch.Generator`, 1000/class protocol, marking tiers, report sections, "50 re-labels", zip contents; clarified 500 vs 1000; added 7 pitfalls | Cross-checked against `project_handout.pdf`, `Project Briefing.pptx`, `starter_kit/README.md`, `AGENTS_TEMPLATE.md` and `requirements.txt` |
| 2026-11-23 | §6 git line kept unchanged | Handout says "Use git and commit often"; the human cleared the earlier "frozen until COMMIT_PLAN approval" deferral — `COMMIT_PLAN.md` is internal pacing at most |
| 2026-11-23 | Pinned project root to `D:\CISC3024\PROJECT`; §6 / §9.5 / §10 / §11 now say the curation tools are recorded in the **root** `README.md` + `requirements.txt`, and that `starter_kit/` copies are read-only | Human confirmed which of the two same-named files is the record target |
| 2026-11-23 | Confirmed the root `README.md` / `requirements.txt` are the 14-line full-env file + the curation table; removed the "Git is deliberately frozen" bullet from root `README.md`; softened claims about a second `starter_kit/requirements.txt` | Human supplied the actual root files and asked to clear the git deferral; earlier inference about which requirements.txt was which was wrong |
