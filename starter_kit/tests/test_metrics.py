"""Unit tests for the metric implementations (no data or weights needed)."""

import numpy as np
import pytest
import torch

from catdog_eval.metrics import (
    frechet_distance,
    gaussian_stats,
    kernel_inception_distance,
    memorization_ratio,
    precision_recall,
)


def test_fid_identical_gaussians_is_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 8))
    mu, sigma = gaussian_stats(a)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-8)


def test_fid_1d_closed_form():
    # N(0, 1) vs N(1, 4): (0-1)^2 + 1 + 4 - 2*sqrt(1*4) = 2
    d = frechet_distance(np.zeros(1), np.eye(1), np.ones(1), 4 * np.eye(1))
    assert d == pytest.approx(2.0)


def test_fid_diagonal_closed_form():
    mu1, mu2 = np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 0.0])
    v1, v2 = np.array([1.0, 2.0, 3.0]), np.array([4.0, 0.5, 3.0])
    expected = ((mu1 - mu2) ** 2).sum() + ((np.sqrt(v1) - np.sqrt(v2)) ** 2).sum()
    assert frechet_distance(mu1, np.diag(v1), mu2, np.diag(v2)) == pytest.approx(expected)


def test_fid_matches_scipy_sqrtm():
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(1)
    a = rng.normal(size=(400, 16)) @ rng.normal(size=(16, 16))
    b = rng.normal(size=(400, 16)) @ rng.normal(size=(16, 16)) + 0.5
    mu1, s1 = gaussian_stats(a)
    mu2, s2 = gaussian_stats(b)
    covmean = linalg.sqrtm(s1 @ s2).real
    expected = ((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(expected, rel=1e-6)


def test_precision_recall_same_distribution_is_high():
    g = torch.Generator().manual_seed(0)
    real = torch.randn(1000, 8, generator=g)
    fake = torch.randn(1000, 8, generator=g)
    p, r = precision_recall(real, fake)
    assert p > 0.8 and r > 0.8


def test_mode_collapse_has_low_recall():
    g = torch.Generator().manual_seed(0)
    real = torch.randn(1000, 8, generator=g)
    fake = 0.01 * torch.randn(1000, 8, generator=g)  # everything at the mode
    p, r = precision_recall(real, fake)
    assert p > 0.9
    assert r < 0.05


def test_disjoint_distributions_score_zero():
    g = torch.Generator().manual_seed(0)
    real = torch.randn(500, 8, generator=g)
    fake = torch.randn(500, 8, generator=g) + 100
    assert precision_recall(real, fake) == (0.0, 0.0)


def test_memorization_ratio_flags_noisy_copies_only():
    g = torch.Generator().manual_seed(0)
    reference = torch.rand(2000, 1, 28, 28, generator=g)
    copies = (reference[:100] + 0.05 * torch.randn(100, 1, 28, 28, generator=g)).clamp(0, 1)
    fresh = torch.rand(100, 1, 28, 28, generator=g)
    _, ratio_copies = memorization_ratio(copies, reference)
    _, ratio_fresh = memorization_ratio(fresh, reference)
    assert (ratio_copies < 0.5).all()
    assert (ratio_fresh > 0.9).all()


def test_memorization_ratio_exact_duplicates_in_reference():
    # A copy of an image that appears twice in the reference set has d1 = d2 = 0;
    # it must still count as memorized (ratio 0), not produce NaN.
    ref = torch.rand(10, 1, 28, 28)
    ref = torch.cat([ref, ref[:1]])
    d1, ratio = memorization_ratio(ref[:1].clone(), ref)
    assert d1.item() == 0.0
    assert ratio.item() == 0.0


def test_kid_same_distribution_is_near_zero_and_detects_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(600, 16)), rng.normal(size=(600, 16))
    same, _ = kernel_inception_distance(a, b, n_subsets=20, subset_size=300)
    shifted, _ = kernel_inception_distance(a, b + 0.5, n_subsets=20, subset_size=300)
    assert abs(same) < 0.01
    assert shifted > 10 * abs(same) + 0.01
