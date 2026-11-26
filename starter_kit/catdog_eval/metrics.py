"""Sample-quality metrics computed in Inception feature space or pixel space.

References
  * FID: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge
    to a Local Nash Equilibrium", NeurIPS 2017.
  * KID: Binkowski et al., "Demystifying MMD GANs", ICLR 2018.
  * Precision / recall: Kynkaanniemi et al., "Improved Precision and Recall
    Metric for Assessing Generative Models", NeurIPS 2019.
"""

import numpy as np
import torch


def gaussian_stats(features: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and (unbiased) covariance of a [N, D] feature matrix, in float64."""
    f = np.asarray(features, dtype=np.float64)
    return f.mean(axis=0), np.cov(f, rowvar=False)


def _psd_sqrt(mat: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(mat)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Squared Frechet (2-Wasserstein) distance between two Gaussians.

    d^2 = ||mu1 - mu2||^2 + Tr(S1) + Tr(S2) - 2 Tr((S1 S2)^{1/2})

    Tr((S1 S2)^{1/2}) is computed as Tr((S1^{1/2} S2 S1^{1/2})^{1/2}); the inner
    matrix is symmetric PSD, so this avoids the complex round-off that a generic
    matrix square root produces.
    """
    diff = mu1 - mu2
    s1_half = _psd_sqrt(sigma1)
    inner = s1_half @ sigma2 @ s1_half
    eig = np.clip(np.linalg.eigvalsh((inner + inner.T) / 2), 0.0, None)
    tr_covmean = np.sqrt(eig).sum()
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _kth_nn_radius(feats: torch.Tensor, k: int, chunk: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest neighbour in the same set (self excluded)."""
    radii = []
    for i in range(0, len(feats), chunk):
        d = torch.cdist(feats[i:i + chunk], feats)
        # k + 1 because the smallest distance is the point to itself.
        radii.append(d.topk(k + 1, dim=1, largest=False).values[:, -1])
    return torch.cat(radii)


def _fraction_inside(query: torch.Tensor, ref: torch.Tensor, ref_radii: torch.Tensor,
                     chunk: int) -> float:
    """Fraction of query points that fall inside at least one reference hypersphere."""
    inside = []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk], ref)
        inside.append((d <= ref_radii.unsqueeze(0)).any(dim=1))
    return torch.cat(inside).float().mean().item()


def precision_recall(real_feats: torch.Tensor, fake_feats: torch.Tensor, k: int = 3,
                     chunk: int = 1000) -> tuple[float, float]:
    """k-NN precision (fidelity) and recall (coverage / diversity).

    precision: fraction of generated samples lying on the real-data manifold.
    recall:    fraction of real samples lying on the generated-data manifold.
    """
    real = real_feats.double()
    fake = fake_feats.double()
    real_r = _kth_nn_radius(real, k, chunk)
    fake_r = _kth_nn_radius(fake, k, chunk)
    precision = _fraction_inside(fake, real, real_r, chunk)
    recall = _fraction_inside(real, fake, fake_r, chunk)
    return precision, recall


def nearest_neighbor_distances(query: torch.Tensor, reference: torch.Tensor, k: int = 2,
                               chunk: int = 500) -> torch.Tensor:
    """Pixel-space L2 distances from each query image to its k nearest reference images.

    Returns [N, k], sorted ascending along dim 1.
    """
    q = query.flatten(1).float()
    r = reference.flatten(1).float()
    out = []
    for i in range(0, len(q), chunk):
        out.append(torch.cdist(q[i:i + chunk], r).topk(k, dim=1, largest=False).values)
    return torch.cat(out)


def memorization_ratio(query: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour ratio test for copied training data.

    Returns (d1, d1 / d2) where d1, d2 are the pixel distances to the nearest and
    second-nearest reference image. A copy (even with small noise added) sits much
    closer to its source than to any other image, giving a ratio far below 1;
    a genuinely new image has a ratio close to 1. An absolute distance threshold
    does not work: plain backgrounds make some images naturally close to many others.
    The reference set must be free of near-duplicates (LAION has many reposts),
    otherwise a copy of an image that appears twice has d1 ~ d2 and passes.
    """
    d = nearest_neighbor_distances(query, reference, k=2)
    return d[:, 0], d[:, 0] / d[:, 1].clamp_min(1e-8)


def kernel_inception_distance(real_feats: torch.Tensor, fake_feats: torch.Tensor,
                              n_subsets: int = 50, subset_size: int = 500,
                              seed: int = 0) -> tuple[float, float]:
    """KID (Binkowski et al., 2018): unbiased MMD^2 with the cubic polynomial kernel
    k(x, y) = (x.y / d + 1)^3, averaged over random subsets. Returns (mean, std).

    Unlike FID it is unbiased, so values from small sample sets can be compared with
    values from the official protocol.
    """
    real = np.asarray(real_feats, dtype=np.float64)
    fake = np.asarray(fake_feats, dtype=np.float64)
    d = real.shape[1]
    m = min(subset_size, len(real), len(fake))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_subsets):
        x = real[rng.choice(len(real), m, replace=False)]
        y = fake[rng.choice(len(fake), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
               + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
        values.append(mmd)
    return float(np.mean(values)), float(np.std(values))
