"""Shared I/O, seed, and logging helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for numpy, random, and optionally torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def mean_std(values) -> tuple[float, float]:
    """Return (mean, std) for a list of floats, ignoring NaN."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))
