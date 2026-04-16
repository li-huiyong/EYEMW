"""Evaluation utilities: metrics, bootstrap CI, generalization gap."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Standard classification metrics dict.

    Handles edge cases where a test fold contains only one class.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    auc = float("nan")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass

    c0 = report.get("0", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    c1 = report.get("1", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    return {
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "auc": float(auc),
        "class0_precision": float(c0["precision"]),
        "class0_recall": float(c0["recall"]),
        "class0_f1": float(c0["f1-score"]),
        "class1_precision": float(c1["precision"]),
        "class1_recall": float(c1["recall"]),
        "class1_f1": float(c1["f1-score"]),
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (lower is better)."""
    return float(brier_score_loss(y_true, y_prob))


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    metric_fn(y_true, y_pred, y_prob) -> float

    Parameters
    ----------
    groups : participant IDs for cluster bootstrap.  When provided, entire
        participants (clusters) are resampled rather than individual rows,
        yielding valid CIs for clustered data.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_pred, y_prob)
    boot_vals = np.empty(n_boot)

    if groups is not None:
        unique_groups = np.unique(groups)
        group_idx = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)
        for b in range(n_boot):
            sampled = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_idx[g] for g in sampled])
            yt, yp = y_true[idx], y_pred[idx]
            ypr = y_prob[idx] if y_prob is not None else None
            try:
                boot_vals[b] = metric_fn(yt, yp, ypr)
            except (ValueError, ZeroDivisionError):
                boot_vals[b] = float("nan")
    else:
        n = len(y_true)
        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            yt, yp = y_true[idx], y_pred[idx]
            ypr = y_prob[idx] if y_prob is not None else None
            try:
                boot_vals[b] = metric_fn(yt, yp, ypr)
            except (ValueError, ZeroDivisionError):
                boot_vals[b] = float("nan")

    lo = float(np.nanpercentile(boot_vals, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_vals, 100 * (1 - alpha / 2)))
    return point, lo, hi


def generalization_gap(
    within_scores: list[float],
    held_out_scores: list[float],
) -> dict[str, float]:
    """Compute generalization gap statistics."""
    w = np.asarray(within_scores, dtype=float)
    h = np.asarray(held_out_scores, dtype=float)
    return {
        "within_mean": float(np.nanmean(w)),
        "within_std": float(np.nanstd(w)),
        "held_out_mean": float(np.nanmean(h)),
        "held_out_std": float(np.nanstd(h)),
        "gap_mean": float(np.nanmean(w) - np.nanmean(h)),
    }
