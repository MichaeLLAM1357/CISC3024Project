"""Evaluation utilities for the CISC3024 cat/dog head generation project.

Everything in this package is fixed by the teaching team. Do not modify it:
your submission is graded with the official copy.
"""

from .classifier import CatDogProbe, load_classifier
from .data import CLASSES, Pool, crop, default_pool, load_dev, load_image_set, pool_dir, rebuild
from .features import inception_features
from .manifest import check_manifest
from .metrics import (
    frechet_distance,
    gaussian_stats,
    kernel_inception_distance,
    memorization_ratio,
    nearest_neighbor_distances,
    precision_recall,
)

__all__ = [
    "CLASSES",
    "CatDogProbe",
    "Pool",
    "check_manifest",
    "crop",
    "default_pool",
    "frechet_distance",
    "gaussian_stats",
    "inception_features",
    "kernel_inception_distance",
    "load_classifier",
    "load_dev",
    "load_image_set",
    "memorization_ratio",
    "nearest_neighbor_distances",
    "pool_dir",
    "precision_recall",
    "rebuild",
]
