"""Cross-validation strategies.

All generators yield (train_idx, test_idx, fold_info) tuples where fold_info
is a dict with metadata about the fold (e.g., held-out study name).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV

from src.config import OUTER_SPLITS, INNER_SPLITS


def grouped_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_splits: int = OUTER_SPLITS,
    inner_splits: int = INNER_SPLITS,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """Participant-grouped nested CV (outer loop only).

    The inner loop (GridSearchCV) is handled at the model level.
    Yields (train_idx, test_idx, fold_info).
    """
    outer_cv = GroupKFold(n_splits=outer_splits)
    for fold_id, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        info = {
            "cv_type": "grouped_nested",
            "fold_id": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        yield train_idx, test_idx, info


def leave_one_study_out(
    X: np.ndarray,
    y: np.ndarray,
    study_labels: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """Leave-one-study-out CV. Each fold holds out one unique study."""
    unique_studies = np.unique(study_labels)
    for study in unique_studies:
        test_mask = study_labels == study
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        info = {
            "cv_type": "leave_one_study_out",
            "held_out": str(study),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        yield train_idx, test_idx, info


def leave_one_window_out(
    X: np.ndarray,
    y: np.ndarray,
    window_labels: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """Leave-one-window-type-out CV."""
    unique_windows = np.unique(window_labels)
    for window in unique_windows:
        test_mask = window_labels == window
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        info = {
            "cv_type": "leave_one_window_out",
            "held_out": str(window),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        yield train_idx, test_idx, info


def leave_one_taskgroup_out(
    X: np.ndarray,
    y: np.ndarray,
    taskgroup_labels: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """Leave-one-task-group-out CV."""
    unique_groups = np.unique(taskgroup_labels)
    for tg in unique_groups:
        test_mask = taskgroup_labels == tg
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        info = {
            "cv_type": "leave_one_taskgroup_out",
            "held_out": str(tg),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        yield train_idx, test_idx, info


def get_inner_cv(groups_train: np.ndarray, n_splits: int = INNER_SPLITS):
    """Return a GroupKFold splitter for the inner CV loop."""
    return GroupKFold(n_splits=n_splits)
