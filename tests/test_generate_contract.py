"""Test 2: the shipped generate() satisfies the interface contract (LIMITS.md §5)."""

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


def test_shape_dtype_range_and_no_nan(model) -> None:
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.long)
    x = generate(model, labels, seed=7)
    assert x.shape == (8, 3, 32, 32)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all(), "NaN or Inf in generated images"
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_single_sample_batch(model) -> None:
    x = generate(model, torch.tensor([0]), seed=3)
    assert x.shape == (1, 3, 32, 32)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_model_is_one_eval_module(model) -> None:
    assert isinstance(model, torch.nn.Module)
    assert model.training is False
    state = model.state_dict()
    assert state, "model has no parameters"
    # one module owns everything: the state_dict reloads into a fresh instance
    clone = type(model)()
    clone.load_state_dict(state)
    assert len(list(clone.parameters())) == len(list(model.parameters()))
