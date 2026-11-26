"""Tests of the pool reader, the manifest check and the crop rule (synthetic pool, no
downloads)."""

import csv
import zipfile

import numpy as np
import pytest
from PIL import Image

import evaluate
from catdog_eval.data import Pool, crop, pool_dir, rebuild
from catdog_eval.manifest import MIN_PER_CLASS, check_manifest


@pytest.fixture()
def pool(tmp_path):
    """Three pool images (two in one zip shard, one in another) with an index."""
    folder = tmp_path / "pool"
    folder.mkdir()
    rng = np.random.default_rng(0)
    sizes = {"s000000": (320, 240), "s000001": (240, 320), "s000002": (400, 320)}
    with open(folder / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool_id", "width", "height"])
        for pid, (wd, ht) in sizes.items():
            w.writerow([pid, wd, ht])
    for shard, ids in (("laion_pool_student_000.zip", ["s000000", "s000001"]),
                       ("laion_pool_student_001.zip", ["s000002"])):
        with zipfile.ZipFile(folder / shard, "w") as z:
            for pid in ids:
                wd, ht = sizes[pid]
                arr = rng.integers(0, 256, (ht, wd, 3), dtype=np.uint8)
                path = tmp_path / f"{pid}.jpg"
                Image.fromarray(arr).save(path, quality=95)
                z.write(path, f"images/{pid}.jpg")
    return Pool(folder)


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool_id", "x0", "y0", "size", "angle", "label"])
        w.writerows(rows)
    return path


def test_pool_reads_images_from_all_shards(pool):
    assert pool.image("s000002").size == (400, 320)
    assert pool.size("s000001") == (240, 320)
    assert "s000003" not in pool


def test_pool_dir_finds_unpacked_student_zip(tmp_path):
    assert pool_dir(tmp_path) == tmp_path / "student"  # default when nothing is unpacked
    (tmp_path / "pool").mkdir()
    (tmp_path / "pool" / "index.csv").write_text("pool_id\n")
    assert pool_dir(tmp_path) == tmp_path / "pool"
    (tmp_path / "student").mkdir()
    (tmp_path / "student" / "index.csv").write_text("pool_id\n")
    assert pool_dir(tmp_path) == tmp_path / "student"


def test_valid_manifest_rebuilds_crops(pool, tmp_path):
    m = write_manifest(tmp_path / "m.csv", [["s000000", 10, 20, 100, 0, 0],
                                            ["s000002", 0, 0, 320, 0, 1],
                                            ["s000001", 50, 60, 80, 12.5, 1]])
    report = check_manifest(m, pool)
    assert report["errors"] == []
    assert report["stats"]["valid_rows"] == 3 and report["stats"]["rotated"] == 1
    images, labels = rebuild(report["rows"], pool)
    assert images.shape == (3, 3, 32, 32) and labels.tolist() == [0, 1, 1]
    expected = np.asarray(crop(pool.image("s000000"), 10, 20, 100))
    assert np.array_equal((images[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                          expected)


@pytest.mark.parametrize("row, message", [
    (["s009999", 0, 0, 100, 0, 0], "not in the raw pool"),
    (["s000000", 0, 0, 100, 0, 2], "label must be"),
    (["s000000", 0, 0, 30, 0, 0], "size 30"),
    (["s000000", 250, 0, 100, 0, 0], "is not inside"),
    (["s000000", 0, 0, 240, 30, 0], "is not inside"),   # rotated corners leave the image
    (["s000000", 0, 0, 100, 60, 0], "angle"),
    (["s000000", "a", 0, 100, 0, 0], "cannot parse"),
])
def test_invalid_rows_are_errors(pool, tmp_path, row, message):
    report = check_manifest(write_manifest(tmp_path / "m.csv", [row]), pool)
    assert len(report["errors"]) == 1 and message in report["errors"][0]


def test_missing_columns_and_repeats(pool, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("pool_id,x,y\ns000000,1,2\n")
    assert "missing columns" in check_manifest(bad, pool)["errors"][0]
    rep = check_manifest(write_manifest(tmp_path / "m.csv", [["s000000", 0, 0, 100, 0, 0]] * 2),
                         pool)
    assert any("repeat" in e for e in rep["errors"])


def test_warnings_for_small_unbalanced_overlapping_data(pool, tmp_path):
    m = write_manifest(tmp_path / "m.csv", [["s000000", 0, 0, 100, 0, 0],
                                            ["s000000", 2, 2, 100, 0, 0]])
    w = " | ".join(check_manifest(m, pool)["warnings"])
    assert f"only 2 cat images (expected at least {MIN_PER_CLASS})" in w
    assert "only 0 dog images" in w
    assert "overlap" in w


def test_dedup_keeps_one_image_per_near_duplicate_group():
    import torch
    g = torch.Generator().manual_seed(0)
    base = torch.rand(20, 3, 32, 32, generator=g)
    noisy = (base[:5] + 0.01 * torch.randn(5, 3, 32, 32, generator=g)).clamp(0, 1)
    unique = evaluate.dedup(torch.cat([base, noisy, base[:3]]))
    assert len(unique) == 20


def test_audit_sample_depends_on_student_id_not_row_order():
    from audit import sample_rows
    rows = [{"pool_id": f"s{i:06d}", "x0": i, "y0": 0, "size": 40, "angle": 0.0, "label": i % 2}
            for i in range(1000)]
    a = sample_rows(rows, "1234567", 200)
    assert len(a) == 200 and len({r["pool_id"] for r in a}) == 200
    assert sample_rows(list(reversed(rows)), "1234567", 200) == a
    assert sample_rows(rows, "7654321", 200) != a
    # Each row's chance depends only on the ID and the row: other rows added to the
    # manifest can only push a drawn row out, never pull an undrawn one in.
    extra = [{**r, "pool_id": "t" + r["pool_id"]} for r in rows]
    b = sample_rows(rows + extra, "1234567", 200)
    assert all(r in a for r in b if r["pool_id"].startswith("s"))


def test_marks_are_per_metric():
    m = {"accuracy": 0.95, "fid": 70.0, "recall": 0.12}
    marks = evaluate.assign_marks(m, [])
    assert {k: v["marks"] for k, v in marks["per_metric"].items()} == \
        {"accuracy": 5, "fid": 3, "recall": 1}
    assert marks["total"] == 9 and marks["max"] == 15
    assert evaluate.assign_marks({"accuracy": 0.4, "fid": 200.0, "recall": 0.0}, [])["total"] == 0
    assert evaluate.assign_marks(m, ["too many parameters"])["total"] == 0
