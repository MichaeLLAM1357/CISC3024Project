"""Test 1: our manifest crop is pixel-identical to the official catdog_eval.data.crop.

The manifest stores integer x0, y0, size with angle = 0, so the pipeline crop is
`image.crop((x0, y0, x0 + size, y0 + size)).resize((32, 32), LANCZOS)`.  If this ever drifts
from the evaluator's crop, every training image the evaluator rebuilds is different from the
one we trained on, which is a silent, fatal reproducibility bug.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from catdog_eval.data import Pool, crop, pool_dir

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "submission" / "manifest.csv"
N_ROWS = 25


def pipeline_crop(image: Image.Image, x0: int, y0: int, size: int) -> np.ndarray:
    """The crop implemented in pipeline/build_manifest.py (no rotation)."""
    return np.asarray(image.crop((x0, y0, x0 + size, y0 + size))
                      .resize((32, 32), Image.Resampling.LANCZOS))


@pytest.mark.skipif(not MANIFEST.exists(), reason="submission/manifest.csv not built yet")
def test_manifest_crop_matches_official_crop() -> None:
    import csv

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "manifest is empty"
    pool = Pool(pool_dir(ROOT / "data"))

    step = max(1, len(rows) // N_ROWS)
    checked = 0
    for r in rows[::step][:N_ROWS]:
        im = pool.image(r["pool_id"])
        x0, y0, size = int(r["x0"]), int(r["y0"]), int(r["size"])
        angle = float(r["angle"])
        ours = pipeline_crop(im, x0, y0, size)
        ref = np.asarray(crop(im, x0, y0, size, angle))
        assert ours.shape == (32, 32, 3)
        assert np.array_equal(ours, ref), f"crop mismatch for {r['pool_id']}"
        checked += 1
    assert checked >= min(N_ROWS, len(rows))
