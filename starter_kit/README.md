# Starter kit — CISC3024 cat and dog head generation project

Everything here is fixed by the teaching team. Do not modify it; we grade with our own copy.
The full rules are in the assignment handout.

## Contents

| Path | Purpose |
|---|---|
| `evaluate.py` | Official evaluator (CLI and importable API) |
| `audit.py` | Draws the 200 training images you label for the manual audit |
| `catdog_eval/` | Pool reader, crop rule, manifest check, Inception features, metrics, fixed cat/dog classifier (`classifier.pt`) |
| `submission_template/generate.py` | Required interface, with a placeholder that outputs noise |
| `AGENTS_TEMPLATE.md` | Skeleton for your repository's `AGENTS.md` / `CLAUDE.md` |
| `tests/` | Tests for the starter kit and evaluator, provided by the teaching team — run them to check your environment. They **do not replace the tests you must write for your own project** |

## Data

The starter kit (`starter_kit.zip`) and the dev set (`dev_set.zip`) are on UMMoodle; the raw
pool (`student.zip`) is linked from the handout. Put the three zip files in your
project folder and unpack them there:

```bash
unzip starter_kit.zip            # -> starter_kit/
unzip dev_set.zip                # -> data/dev/
unzip student.zip -d data        # -> data/student/
```

On Windows, where `unzip` is not available, use `tar` (built into Windows 10 and later) in
PowerShell:

```powershell
tar -xf starter_kit.zip
tar -xf dev_set.zip              # creates data/, so unpack it before student.zip
tar -xf student.zip -C data
```

Do not use "Extract All" in the Windows Explorer: it adds an extra folder named after the
zip file (`student/student/...`), and the code then cannot find the data. After unpacking,
check that `data/student/index.csv` and `data/dev/dev.npz` exist. If the evaluator prints
`raw pool not found`, its numbers are incomplete: it has not checked your manifest or
measured memorization, while the official evaluation does both.

Then delete the three zip files. Do not commit `data/` to git (add it to your `.gitignore`)
and do not include it in the project zip you submit.

Leave the `laion_pool_student_*.zip` shards inside `data/student/` as they are: the code reads
the images directly from them. The data can also live anywhere else if you set `CATDOG_DATA`
to the folder that contains `dev/` and `student/`. **Run every command in this README from the
project root**, the folder that contains `starter_kit/`, `data/` and `submission/`:

```
data/student/index.csv                  pool_id, width, height
data/student/laion_pool_student_*.zip   images/<pool_id>.jpg (short side 128–320 px)
data/dev/dev.npz                        curated heads: x [N,32,32,3] uint8, y [N] (0 = cat, 1 = dog)
data/dev/images/{cat,dog}/*.png         the same heads as PNG files — look at them
data/dev/manifest.csv                   where they came from (pool ids of a different pool)
```

The pool holds 76,250 unprocessed web images, including cat/dog candidates and other
animals, objects and scenes. You must identify suitable cat/dog images yourself.
Images are downscaled so that the short side is at most 320 px; `width` and `height`
in `index.csv` are the sizes of the pool images, which is what manifest coordinates refer to.
The index contains only `pool_id`, `width` and `height`: no captions or class labels.
IDs are shuffled across the entire pool and do not encode image categories.
The pool is the only allowed source of training images; many images contain no usable
cat/dog head. The dev set is for
evaluation and as an example of the data specification only: never train on it.

## Quick start

```bash
pip install -r starter_kit/requirements.txt
pytest -q starter_kit/tests                 # no data needed
python starter_kit/evaluate.py --submission starter_kit/submission_template --n-per-class 100
```

The first run of the evaluator downloads the Inception-v3 weights used for FID (about
92 MB, via `pytorch-fid`) into `~/.cache/torch/hub/checkpoints/`.

## The manifest

`submission/manifest.csv` lists every training image:

```
pool_id,x0,y0,size,angle,label
s000123,41,17,96,0,0
```

The image is the square of side `size` with top-left corner `(x0, y0)` in pixels of the pool
image, after rotating the pool image by `angle` degrees counter-clockwise about the square's
centre, resized to 32×32 with Lanczos filtering. The code is `catdog_eval.data.crop`; use it
(or test your own crop code against it), because the evaluator rebuilds your training set with
it. Rules checked by the evaluator: the pool id exists, `size ≥ 40`, `|angle| ≤ 45`, the
(rotated) square lies inside the image, labels are 0 or 1, no repeated rows. It warns about
fewer than 500 images per class, unbalanced classes and heavily overlapping crops.

`size ≥ 40` is not an extra rule: it follows from the data specification. A head at least
32 px wide that takes at most 80% of the square needs a square of at least 32 / 0.8 = 40 px.
Passing this check does not mean a crop meets the specification (a 50 px square around a
20 px head passes but is far too loose); that is what the manual audit is for.

```python
import sys; sys.path.insert(0, "starter_kit")
from catdog_eval import check_manifest, default_pool, rebuild
pool = default_pool()                                      # data/student/ (or $CATDOG_DATA)
report = check_manifest("submission/manifest.csv", pool)   # errors, warnings, rows, stats
images, labels = rebuild(report["rows"], pool)             # [N,3,32,32] in [0,1], [N]
```

## The manual audit

```bash
python starter_kit/audit.py --manifest submission/manifest.csv --student-id 1234567
```

writes `audit/audit.csv` (200 rows; fill in `verdict` = good / bad and a `reason` for bad ones)
and `audit/sheet_*.png` (each image at 96×96 and at 32×32). The rows depend only on your
student ID and the set of rows in your manifest, so we draw the same ones. Run it on your final
manifest.

## Evaluating your model

```bash
# Official protocol: 1000 images per class
python starter_kit/evaluate.py --submission submission --out results/eval.json

# Faster while iterating (FID is biased upward for fewer samples:
# only compare FID values computed with the same --n-per-class)
python starter_kit/evaluate.py --submission submission --n-per-class 250

# Score samples you generated yourself (e.g. from your own script)
python starter_kit/evaluate.py --samples my_samples.npz
```

Inside Python:

```python
import sys; sys.path.insert(0, "starter_kit")
from evaluate import compute_metrics, default_reference
metrics = compute_metrics(images, labels, default_reference())  # images [N,3,32,32] in [0,1]
```

The evaluator writes a JSON file with all metrics and a PNG grid (top half cats, bottom half
dogs). **Look at the grid every time.**

## The interface

See `submission_template/generate.py`. In short:

```python
def load_model(device) -> torch.nn.Module: ...
def generate(model, labels, seed) -> torch.Tensor:  # [N,3,32,32] in [0,1]
```

`generate` must be deterministic in `(labels, seed)`, and different seeds must give
different images. Use a `torch.Generator` seeded with `seed`, never the global RNG.
Each image may depend only on its own label and the seed, not on other examples in the batch.
The evaluator checks this property using different batch compositions.

## Metrics at a glance

| Metric | Meaning | Better |
|---|---|---|
| accuracy | the fixed classifier recognises the requested animal | higher |
| FID | distance between Inception feature statistics of your samples and of real heads | lower |
| KID | a related feature-distribution distance; less biased by sample size than FID, but noisy for small samples (see `kid_std`) | lower |
| precision | samples look like real heads (fidelity) | higher |
| recall | samples cover the variety of real heads (diversity) | higher |
| memorization | fraction of samples that are near-copies of your training images | lower; at most 5% (hard limit, see the handout) |

Locally the reference is the dev set (about 450 heads per class); we grade against a hidden
test set of 1000 heads per class curated the same way. Dev and test numbers are close but not
identical: FID in particular depends on the size of the reference set.
