"""Official evaluator for the CISC3024 cat/dog head generation project.

Evaluate a submission directory (it must contain generate.py, see
submission_template/generate.py for the required interface, and manifest.csv, the
provenance of every training image):

    python starter_kit/evaluate.py --submission path/to/submission --device cpu

Evaluate pre-generated samples (skips the interface, size and manifest checks):

    python starter_kit/evaluate.py --samples samples.npz

Run these from the project root (see README.md), so that ./data is found.

    samples.npz must contain "images" ([N,3,32,32] float in [0,1], or [N,32,32,3] uint8)
    and "labels" ([N] ints, 0 = cat, 1 = dog).

Data: the dev set (data/dev/dev.npz) is the local reference; the teaching team grades
against a hidden test set built the same way (--reference). The raw pool (data/student/) is
needed to check the manifest and to rebuild the training set for the copy check.

Metrics (see README.md for full definitions):
  accuracy      fraction of samples the fixed classifier assigns to the requested class
  fid / kid     Frechet / kernel distance between Inception-v3 features of the samples and
                of the reference set (lower is better)
  precision     fraction of samples inside the reference feature manifold (fidelity)
  recall        fraction of reference images inside the sample manifold (diversity)
  memorization  fraction of samples that look copied from the training set rebuilt from
                manifest.csv: the nearest image is much closer than the second nearest
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catdog_eval.classifier import load_classifier, weights_sha256  # noqa: E402
from catdog_eval.data import (  # noqa: E402
    CLASSES,
    DEFAULT_DATA_ROOT,
    Pool,
    load_image_set,
    pool_dir,
    rebuild,
    to_tensor,
)
from catdog_eval.features import inception_features  # noqa: E402
from catdog_eval.manifest import check_manifest  # noqa: E402
from catdog_eval.metrics import (  # noqa: E402
    frechet_distance,
    gaussian_stats,
    kernel_inception_distance,
    memorization_ratio,
    nearest_neighbor_distances,
    precision_recall,
)

# ---------------------------------------------------------------------------
# Grading constants (fixed by the teaching team; see the assignment handout).
# ---------------------------------------------------------------------------

NUM_CLASSES = len(CLASSES)
IMAGE_SHAPE = (3, 32, 32)

LIMITS = {
    "max_parameters": 20_000_000,
    "max_submission_mb": 100.0,
    "max_memorization_rate": 0.05,
}

# Accuracy, FID and recall are marked independently: each metric earns the marks of the
# highest tier whose threshold it meets (0 below the last tier), so the item is out of
# 3 x 5 = 15 marks. Any violation gives 0 marks. Thresholds only make sense for the official
# protocol (n_per_class=1000) against the hidden test set (2000 heads). Calibrated in
# reference_solution/REPORT.md: real curated heads score accuracy 0.95, FID 23, recall 0.72;
# the reference cGAN and DDPM trained for a few hours on a CPU reach accuracy 0.78 / 0.85,
# FID 69 / 67, recall 0.31 / 0.44 (10 and 12 marks); real heads cropped twice too loosely
# score FID 51. Good and Excellent are extrapolated.
# Must match the tier table in handout/assignment.tex; below the last tier is "Fail" (0).
TIERS = [
    # name,       min accuracy, max FID, min recall, marks
    ("Excellent", 0.90, 50.0, 0.40, 5),
    ("Good",      0.80, 60.0, 0.30, 4),
    ("Pass",      0.70, 80.0, 0.20, 3),
    ("Marginal",  0.60, 100.0, 0.15, 2),
    ("Poor",      0.50, 150.0, 0.10, 1),
]
MAX_MARKS = 5 * 3

# A sample counts as memorized when d1 / d2 < MEMORIZATION_RATIO, where d1 and d2 are the
# pixel-space L2 distances to its nearest and second-nearest training image or mirror image
# of one. This set is first reduced to one image per group of near-duplicates (pixel L2 distance below
# DEDUP_DISTANCE): LAION reposts the same photo many times, and a copy of a photo that is
# in the training set twice would otherwise have d1 ~ d2. PROVISIONAL, see calibration.
MEMORIZATION_RATIO = 0.5
DEDUP_DISTANCE = 4.0

OUTPUT_RANGE_TOLERANCE = 1e-3
DETERMINISM_TOLERANCE = 1e-4
PROBE_LABELS = list(range(NUM_CLASSES)) * 10
PROBE_SEED = 20260929
# Single-class batches (all labels equal) of this size are generated once per class.
# If accuracy on them is much lower than on the mixed batches of the main protocol, the
# output depends on batch composition -- typically BatchNorm left in training mode.
SINGLE_CLASS_BATCH = 50
MAX_SINGLE_CLASS_ACCURACY_DROP = 0.2


class SubmissionError(RuntimeError):
    """The submission violates the required interface."""


# ---------------------------------------------------------------------------
# Reference data (loaded once per process; features cached on disk).
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_reference(data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "dev" / "dev.npz"


@lru_cache(maxsize=None)
def _reference(reference: str, device: str) -> dict:
    path = Path(reference)
    if not path.exists():
        raise FileNotFoundError(f"reference set {path} not found; unpack the dev set into "
                                f"{Path(DEFAULT_DATA_ROOT) / 'dev'} (or set CATDOG_DATA)")
    images, labels = load_image_set(path)
    digest = _sha256(path)
    cache = path.with_name(path.stem + ".inception.pt")
    feats = None
    if cache.exists():
        saved = torch.load(cache, weights_only=True)
        if saved.get("sha256") == digest:
            feats = saved["feats"]
    if feats is None:
        feats = inception_features(images, device)
        try:
            torch.save({"sha256": digest, "feats": feats}, cache)
        except OSError:
            pass
    return {"images": images, "labels": labels, "feats": feats,
            "stats": gaussian_stats(feats), "sha256": digest, "path": str(path)}


# ---------------------------------------------------------------------------
# Metrics on a set of images.
# ---------------------------------------------------------------------------

def dedup(images: torch.Tensor, distance: float = DEDUP_DISTANCE) -> torch.Tensor:
    """Keep one image of every group of near-duplicates (pixel L2 distance < distance)."""
    if len(images) == 0:
        return images
    d = nearest_neighbor_distances(images, images, k=min(len(images), 64))
    flat = images.flatten(1)
    keep = torch.ones(len(images), dtype=torch.bool)
    for i in range(len(images)):
        if not keep[i] or d[i, 1:].numel() == 0 or d[i, 1] >= distance:
            continue
        close = torch.cdist(flat[i:i + 1], flat)[0] < distance
        close[:i + 1] = False
        keep &= ~close
    return images[keep]


def compute_metrics(images: torch.Tensor, labels: torch.Tensor, reference: str | Path,
                    device: str = "cpu", seed: int = 0,
                    training_images: torch.Tensor | None = None) -> dict:
    """All sample-quality metrics for images [N,3,32,32] in [0,1] with requested labels [N]."""
    ref = _reference(str(reference), str(device))
    images = images.detach().float().cpu()
    labels = labels.detach().long().cpu()

    feats = inception_features(images, device)
    logits = load_classifier()(feats)
    correct = logits.argmax(1) == labels
    per_class = [correct[labels == c].float().mean().item() if (labels == c).any() else None
                 for c in range(NUM_CLASSES)]

    mu, sigma = gaussian_stats(feats)
    fid = frechet_distance(mu, sigma, *ref["stats"])
    kid, kid_std = kernel_inception_distance(ref["feats"], feats, seed=seed)

    # Precision/recall are biased by set size, so compare equally sized sets.
    real_feats, fake_feats = ref["feats"], feats
    g = torch.Generator().manual_seed(seed)
    if len(fake_feats) < len(real_feats):
        real_feats = real_feats[torch.randperm(len(real_feats), generator=g)[:len(fake_feats)]]
    elif len(fake_feats) > len(real_feats):
        fake_feats = fake_feats[torch.randperm(len(fake_feats), generator=g)[:len(real_feats)]]
    precision, recall = precision_recall(real_feats, fake_feats)

    metrics = {
        "num_samples": len(images),
        "reference": ref["path"],
        "accuracy": correct.float().mean().item(),
        "per_class_accuracy": dict(zip(CLASSES, per_class)),
        "fid": fid,
        "kid": kid,
        "kid_std": kid_std,
        "precision": precision,
        "recall": recall,
        "memorization_rate": None,
        "nn_distance_median": None,
        "nn_ratio_median": None,
        "training_images_deduplicated": None,
    }
    if training_images is not None and len(training_images) >= 2:
        # Mirror images count as copies too (horizontal flips are a common augmentation).
        unique = dedup(torch.cat([training_images, training_images.flip(3)]))
        nn_dist, nn_ratio = memorization_ratio(images, unique)
        metrics.update({
            "memorization_rate": (nn_ratio < MEMORIZATION_RATIO).float().mean().item(),
            "nn_distance_median": nn_dist.median().item(),
            "nn_ratio_median": nn_ratio.median().item(),
            "training_images_deduplicated": len(unique),
        })
    return metrics


def _metric_tier(value: float, column: int, higher_is_better: bool) -> tuple[str, int]:
    for tier in TIERS:
        threshold = tier[column]
        if (value >= threshold) if higher_is_better else (value <= threshold):
            return tier[0], tier[4]
    return "Fail", 0


def assign_marks(metrics: dict, violations: list[str]) -> dict:
    """Tier and marks of each metric, marked independently, and their total (out of 15)."""
    per_metric = {
        "accuracy": _metric_tier(metrics["accuracy"], 1, higher_is_better=True),
        "fid": _metric_tier(metrics["fid"], 2, higher_is_better=False),
        "recall": _metric_tier(metrics["recall"], 3, higher_is_better=True),
    }
    valid = not violations
    return {
        "per_metric": {k: {"tier": t, "marks": m if valid else 0}
                       for k, (t, m) in per_metric.items()},
        "total": sum(m for _, m in per_metric.values()) if valid else 0,
        "max": MAX_MARKS,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# Submission handling.
# ---------------------------------------------------------------------------

_LOADED_SUBMISSIONS: list[Path] = []


def _forget_previous_submissions() -> None:
    """Drop modules and sys.path entries of submissions loaded earlier in this process, so
    that `from model import ...` in the next submission finds its own model.py."""
    for old in _LOADED_SUBMISSIONS:
        while str(old) in sys.path:
            sys.path.remove(str(old))
        for name, mod in list(sys.modules.items()):
            file = getattr(mod, "__file__", None)
            if file and Path(file).resolve().is_relative_to(old):
                del sys.modules[name]
    _LOADED_SUBMISSIONS.clear()


def load_submission(path: str | Path):
    path = Path(path).resolve()
    gen_file = path / "generate.py"
    if not gen_file.is_file():
        raise SubmissionError(f"{gen_file} not found")
    _forget_previous_submissions()
    _LOADED_SUBMISSIONS.append(path)
    sys.path.insert(0, str(path))  # lets generate.py import sibling modules
    spec = importlib.util.spec_from_file_location("submission_generate", gen_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for fn in ("load_model", "generate"):
        if not callable(getattr(module, fn, None)):
            raise SubmissionError(f"generate.py must define a function `{fn}`")
    return module


def _check_output(out, n: int, where: str) -> torch.Tensor:
    if not isinstance(out, torch.Tensor):
        raise SubmissionError(f"{where}: generate() must return a torch.Tensor, got {type(out)}")
    if tuple(out.shape) != (n, *IMAGE_SHAPE):
        raise SubmissionError(f"{where}: expected shape ({n}, 3, 32, 32), got {tuple(out.shape)}")
    out = out.detach().float().cpu()
    if not torch.isfinite(out).all():
        raise SubmissionError(f"{where}: output contains NaN or Inf")
    lo, hi = out.min().item(), out.max().item()
    if lo < -OUTPUT_RANGE_TOLERANCE or hi > 1 + OUTPUT_RANGE_TOLERANCE:
        raise SubmissionError(
            f"{where}: pixel values must be in [0, 1], got min={lo:.3f} max={hi:.3f}. "
            "If your model works in [-1, 1], map back with (x + 1) / 2.")
    return out.clamp_(0.0, 1.0)


def _call_generate(module, model, labels: torch.Tensor, seed: int, device: str,
                   where: str) -> torch.Tensor:
    with torch.no_grad():
        out = module.generate(model, labels.to(device), seed)
    return _check_output(out, len(labels), where)


def check_interface(module, model, device: str) -> dict:
    labels = torch.tensor(PROBE_LABELS)
    a = _call_generate(module, model, labels, PROBE_SEED, device, "determinism probe")
    b = _call_generate(module, model, labels, PROBE_SEED, device, "determinism probe")
    c = _call_generate(module, model, labels, PROBE_SEED + 1, device, "seed probe")

    classifier = load_classifier()
    mixed = torch.tensor(PROBE_LABELS * (SINGLE_CLASS_BATCH // 10))
    mixed_imgs = _call_generate(module, model, mixed, PROBE_SEED + 100, device, "mixed probe")
    single, single_labels = [], []
    for cls in range(NUM_CLASSES):
        same = torch.full((SINGLE_CLASS_BATCH,), cls)
        single.append(_call_generate(module, model, same, PROBE_SEED + 2 + cls, device,
                                     f"single-class probe ({CLASSES[cls]})"))
        single_labels.append(same)
    feats = inception_features(torch.cat([mixed_imgs, *single]), device)
    pred = classifier(feats).argmax(1)
    n_mixed = len(mixed)
    return {
        "deterministic": (a - b).abs().max().item() <= DETERMINISM_TOLERANCE,
        "seed_sensitive": (a - c).abs().mean().item() > 1e-3,
        "mixed_batch_accuracy": (pred[:n_mixed] == mixed).float().mean().item(),
        "single_class_batch_accuracy":
            (pred[n_mixed:] == torch.cat(single_labels)).float().mean().item(),
    }


def sample_submission(module, model, n_per_class: int, seed: int, batch_size: int,
                      device: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate n_per_class images per class. Batch i uses seed `seed * 10000 + i`."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(n_per_class)
    labels = labels[torch.randperm(len(labels), generator=g)]
    images = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, start in enumerate(range(0, len(labels), batch_size)):
        batch = labels[start:start + batch_size]
        images.append(_call_generate(module, model, batch, seed * 10000 + i, device,
                                     f"batch {i}"))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return torch.cat(images), labels, time.perf_counter() - t0


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*")
                if f.is_file() and "__pycache__" not in f.parts
                and not any(part.startswith(".") for part in f.relative_to(path).parts))
    return total / 2**20


def check_training_data(path: Path, data_root: str | Path) -> tuple[dict, torch.Tensor | None]:
    """Manifest check and the rebuilt training set (None if the pool is not available)."""
    manifest = path / "manifest.csv"
    folder = pool_dir(data_root)
    if not (folder / "index.csv").exists():
        return {"errors": [], "warnings": [f"raw pool not found at {folder}: manifest.csv "
                                           "not checked, memorization not measured"],
                "stats": {}}, None
    pool = Pool(folder)
    report = check_manifest(manifest, pool)
    rows = report.pop("rows")
    training = rebuild(rows, pool)[0] if rows else None
    return report, training


def evaluate_submission(path: str | Path, n_per_class: int = 1000, seed: int = 0,
                        batch_size: int = 500, device: str = "cpu",
                        data_root: str | Path = DEFAULT_DATA_ROOT,
                        reference: str | Path | None = None,
                        grading: bool = False) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Run the full protocol. With grading=True a training set that cannot be rebuilt from
    manifest.csv (e.g. the pool is missing) is a violation instead of a warning."""
    path = Path(path).resolve()
    reference = reference or default_reference(data_root)
    module = load_submission(path)
    model = module.load_model(torch.device(device))
    if not isinstance(model, nn.Module):
        raise SubmissionError("load_model() must return a torch.nn.Module that owns all "
                              "trainable parameters (wrap several networks in one Module)")

    checks = check_interface(module, model, device)
    images, labels, seconds = sample_submission(module, model, n_per_class, seed, batch_size,
                                                device)
    data_report, training = check_training_data(path, data_root)
    metrics = compute_metrics(images, labels, reference, device, seed, training)

    resources = {
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "submission_mb": _dir_size_mb(path),
        "seconds_per_1000": seconds / len(images) * 1000,
    }
    violations, warnings = [], list(data_report["warnings"])
    if not checks["deterministic"]:
        violations.append("generate() is not deterministic for a fixed seed")
    if not checks["seed_sensitive"]:
        violations.append("generate() ignores the seed (outputs identical for different seeds)")
    if resources["num_parameters"] > LIMITS["max_parameters"]:
        violations.append(f"too many parameters: {resources['num_parameters']:,}")
    if resources["submission_mb"] > LIMITS["max_submission_mb"]:
        violations.append(f"submission too large: {resources['submission_mb']:.1f} MB")
    for err in data_report["errors"]:
        violations.append(f"manifest.csv: {err}")
    if grading and training is None and not data_report["errors"]:
        violations.append("training set could not be rebuilt from manifest.csv")
    rate = metrics["memorization_rate"]
    if rate is not None and rate > LIMITS["max_memorization_rate"]:
        violations.append(f"memorization rate {rate:.1%} exceeds "
                          f"{LIMITS['max_memorization_rate']:.0%}: samples copy training images")
    if checks["single_class_batch_accuracy"] < \
            checks["mixed_batch_accuracy"] - MAX_SINGLE_CLASS_ACCURACY_DROP:
        violations.append(
            f"output depends on batch composition: accuracy "
            f"{checks['single_class_batch_accuracy']:.2f} on single-class batches vs "
            f"{checks['mixed_batch_accuracy']:.2f} on mixed batches "
            "(BatchNorm left in training mode?)")

    result = {
        "submission": str(path),
        "protocol": {"n_per_class": n_per_class, "seed": seed, "batch_size": batch_size,
                     "device": device, "grading": grading},
        "checks": checks,
        "training_data": data_report,
        "metrics": metrics,
        "resources": resources,
        "limits": LIMITS,
        "violations": violations,
        "warnings": warnings,
        "marks": assign_marks(metrics, violations),
    }
    return result, images, labels


def evaluate_samples(npz_path: str | Path, device: str = "cpu", seed: int = 0,
                     reference: str | Path | None = None,
                     data_root: str | Path = DEFAULT_DATA_ROOT
                     ) -> tuple[dict, torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)
    images = data["images"]
    if images.dtype == np.uint8:
        images = to_tensor(images) if images.shape[-1] == 3 else \
            torch.from_numpy(images).float() / 255.0
    else:
        images = torch.from_numpy(images).float()
    labels = torch.from_numpy(data["labels"]).long()
    images = _check_output(images, len(labels), str(npz_path))
    metrics = compute_metrics(images, labels, reference or default_reference(data_root),
                              device, seed)
    result = {"samples": str(npz_path), "metrics": metrics, "violations": [],
              "warnings": ["memorization and manifest are only checked with --submission"],
              "marks": assign_marks(metrics, [])}
    return result, images, labels


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def save_grid(images: torch.Tensor, labels: torch.Tensor, path: Path, per_class: int = 16) -> None:
    rows = []
    for c in range(NUM_CLASSES):
        imgs = images[labels == c][:per_class * 4]
        if len(imgs) < per_class * 4:
            imgs = torch.cat([imgs, torch.zeros(per_class * 4 - len(imgs), *IMAGE_SHAPE)])
        rows.append(imgs)
    save_image(torch.cat(rows), str(path), nrow=per_class, padding=2, pad_value=0.5)


def print_summary(result: dict) -> None:
    m = result["metrics"]
    acc = m["per_class_accuracy"]
    lines = [
        "",
        "================ Evaluation summary ================",
        f"samples        {m['num_samples']}   (reference {m['reference']})",
        f"accuracy       {m['accuracy']:.4f}   (cat {acc['cat']:.4f}, dog {acc['dog']:.4f})",
        f"FID            {m['fid']:.2f}",
        f"KID            {m['kid'] * 1000:.2f} x 1e-3   (std {m['kid_std'] * 1000:.2f})",
        f"precision      {m['precision']:.4f}",
        f"recall         {m['recall']:.4f}",
    ]
    if m["memorization_rate"] is not None:
        lines.append(f"memorization   {m['memorization_rate']:.2%}   "
                     f"(median NN distance {m['nn_distance_median']:.2f}, median NN ratio "
                     f"{m['nn_ratio_median']:.2f}, against {m['training_images_deduplicated']} "
                     "training images and their mirror images, near-duplicates removed)")
    if "resources" in result:
        r, lim, c = result["resources"], result["limits"], result["checks"]
        s = result["training_data"].get("stats", {})
        lines += [
            f"parameters     {r['num_parameters']:,}   (limit {lim['max_parameters']:,})",
            f"size           {r['submission_mb']:.2f} MB   (limit {lim['max_submission_mb']:.0f} MB)",
            f"speed          {r['seconds_per_1000']:.2f} s / 1000 images on "
            f"{result['protocol']['device']}",
            f"checks         deterministic={c['deterministic']}  "
            f"seed_sensitive={c['seed_sensitive']}  "
            f"single/mixed batch accuracy={c['single_class_batch_accuracy']:.2f}/"
            f"{c['mixed_batch_accuracy']:.2f}",
        ]
        if s:
            lines.append(f"manifest       {s['valid_rows']}/{s['rows']} valid rows: "
                         f"{s['cat']} cats, {s['dog']} dogs from {s['distinct_pool_images']} "
                         f"pool images ({s['rotated']} rotated)")
    if m["num_samples"] != 2000:
        lines.append("note           tiers are defined for 2000 samples (--n-per-class 1000); "
                     "FID is biased upward for fewer samples (KID is not)")
    for w in result.get("warnings", []):
        lines.append(f"WARNING        {w}")
    for v in result["violations"]:
        lines.append(f"VIOLATION      {v}")
    marks = result["marks"]
    for name, t in marks["per_metric"].items():
        lines.append(f"{name:15s}{t['tier']:10s} {t['marks']} marks")
    lines += [f"marks          {marks['total']} / {marks['max']}"
              + ("" if marks["valid"] else "   (0: see violations)"),
              "===================================================="]
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--submission", help="directory containing generate.py and manifest.csv")
    src.add_argument("--samples", help=".npz file with 'images' and 'labels'")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-per-class", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--reference", default=None,
                   help="reference .npz (default <data-root>/dev/dev.npz)")
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--save-samples", default=None, help="optional .npz path for the samples")
    p.add_argument("--grading", action="store_true",
                   help="official grading run: the training set must be rebuildable from "
                        "manifest.csv")
    args = p.parse_args()

    try:
        if args.submission:
            result, images, labels = evaluate_submission(
                args.submission, args.n_per_class, args.seed, args.batch_size, args.device,
                args.data_root, args.reference, grading=args.grading)
        else:
            result, images, labels = evaluate_samples(args.samples, args.device, args.seed,
                                                      args.reference, args.data_root)
    except SubmissionError as e:
        print(f"SUBMISSION ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    result["environment"] = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "classifier_sha256": weights_sha256(),
        "reference_sha256": _reference(result["metrics"]["reference"], args.device)["sha256"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    grid_path = out.with_suffix(".png")
    save_grid(images, labels, grid_path)
    if args.save_samples:
        np.savez_compressed(args.save_samples, images=images.numpy(), labels=labels.numpy())
    print_summary(result)
    print(f"results -> {out}\ngrid    -> {grid_path}")


if __name__ == "__main__":
    main()
