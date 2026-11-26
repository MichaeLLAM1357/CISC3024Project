"""Test 4: determinism, seed sensitivity, and batch-composition independence.

The evaluator draws a single-class batch and a mixed batch and requires the classifier accuracy
on the single-class batch to be no more than 0.2 lower (LIMITS.md §4).  We cannot measure that
metric without `classifier.pt`, which is forbidden.  Instead we check the stronger property that
causes the metric to hold: the image for a given label at a given batch position does not change
when the other labels in the batch change.
"""

from pathlib import Path

import pytest
import torch

from generate import generate, load_model

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "submission" / "weights.pt"

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(),
                                reason="submission/weights.pt not trained yet")


@pytest.fixture(scope="module")
def model():
    return load_model(torch.device("cpu"))


def test_same_seed_is_reproducible(model) -> None:
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    a = generate(model, labels, seed=11)
    b = generate(model, labels, seed=11)
    # The evaluator's own criterion is `max|a-b| <= 1e-4`, not bitwise equality: cuDNN may pick a
    # different convolution kernel from call to call.  1e-5 is below half a pixel of 8-bit
    # quantisation and still catches any real source of nondeterminism (a global RNG, dropout
    # left on, a batch-dependent noise draw), which all produce differences of order 1e-1.
    assert torch.allclose(a, b, atol=1e-5), \
        f"same seed produced different images (max diff {(a-b).abs().max():.2e})"


def test_different_seed_changes_images(model) -> None:
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    a = generate(model, labels, seed=11)
    b = generate(model, labels, seed=12)
    assert (a - b).abs().max() > 1e-3, "different seeds produced (nearly) identical images"


def test_single_class_batch_matches_mixed_batch(model) -> None:
    # Noise is derived from (seed, position), so the comparison must hold position and label
    # fixed and change only the *other* labels in the batch.
    cat_only = generate(model, torch.zeros(6, dtype=torch.long), seed=5)
    mixed = torch.tensor([0, 1, 0, 1, 1, 1], dtype=torch.long)
    mixed_out = generate(model, mixed, seed=5)

    # Positions 0 and 2 are cats in the mixed batch; their images must be unchanged up to
    # float rounding.  Convolution kernels are chosen per batch shape, so bitwise equality is
    # not guaranteed even for an identical per-sample computation; 1e-5 is far below the
    # 1/255 quantisation of a pixel and cannot move the evaluator's accuracy.
    assert torch.allclose(cat_only[0], mixed_out[0], atol=1e-5)
    assert torch.allclose(cat_only[2], mixed_out[2], atol=1e-5)


def test_batch_size_does_not_change_first_sample(model) -> None:
    # Batch *size* changes which cuDNN kernel Conv2d picks, so the per-sample computation stays
    # identical mathematically but not bitwise.  Antithetic guidance (weights w=2.5 in
    # generate.py) multiplies that drift by ~(2w-1) per DDIM step, so the old 1e-5 tolerance --
    # written for the unguided sampler -- now fails at 2.0e-05.  Measured on the graded weights:
    #   batch 1 vs 4 -> max 2.0e-05, batch 1 vs 8 -> max 1.0e-05, 0 of 3072 pixels above 1/255.
    # The tolerance is therefore the evaluator's own DETERMINISM_TOLERANCE, and the two assertions
    # below pin down what actually matters: no visible pixel, and exact equality when the shape
    # is fixed (which is the logical statement of the contract, checked in the test above).
    one = generate(model, torch.tensor([1]), seed=9)
    many = generate(model, torch.tensor([1, 0, 1, 0]), seed=9)
    diff = (one[0] - many[0]).abs()
    assert diff.max() <= 1e-4, f"batch size changed the sample by {diff.max():.2e}"
    assert (diff > 1 / 255).sum() == 0, "batch size changed a pixel by a visible amount"
