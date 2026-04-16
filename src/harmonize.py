"""Study-harmonization preprocessing.

Loads all study files, adds context variables (WindowType, TaskGroup),
probe-availability masks, and scale-harmonized affect labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    STUDY_FILES, STUDY_META, STUDY_COL, GROUP_COL, PROBE_COL,
    LABEL_COL, GAZE_COLS, AFFECT_RESPONSE_COLS, ALL_CONSTRUCTS,
    TABLES_DIR,
)

# Mapping from construct name to the column prefix used in the EYEMW schema
_CONSTRUCT_PREFIX = {
    "valence": "Valence",
    "arousal": "Arousal",
    "boredom": "Boredom",
    "disengagement": "Disengagement",
}


def _load_study(study_id: str) -> pd.DataFrame:
    """Load one study file and tag rows with a canonical study_id string.

    When a single file contains multiple sub-studies (e.g. study010-011.xlsx
    holds StudyNum 10 and 11), ParticipantNum is made unique by prefixing
    with StudyNum so that different people who happen to share the same
    numeric ID are not collapsed into one CV group.
    """
    path = STUDY_FILES[study_id]
    df = pd.read_excel(path).copy()
    df["study_id"] = study_id
    if "StudyNum" in df.columns and df["StudyNum"].nunique() > 1:
        df["ParticipantNum"] = (
            df["StudyNum"].astype(str) + "_" + df["ParticipantNum"].astype(str)
        )
    else:
        df["ParticipantNum"] = df["ParticipantNum"].astype(str)
    return df


def _add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add WindowType and TaskGroup derived from STUDY_META."""
    df["WindowType"] = df["study_id"].map(
        {sid: m["window_type"] for sid, m in STUDY_META.items()}
    )
    df["TaskGroup"] = df["study_id"].map(
        {sid: m["task_group"] for sid, m in STUDY_META.items()}
    )
    return df


def _add_probe_availability_mask(df: pd.DataFrame) -> pd.DataFrame:
    """One binary column per construct indicating whether the study measures it."""
    avail_map = {sid: set(m["affect_available"]) for sid, m in STUDY_META.items()}
    for construct in ALL_CONSTRUCTS:
        col_name = f"has_{construct}"
        df[col_name] = df["study_id"].map(
            {sid: int(construct in avails) for sid, avails in avail_map.items()}
        )
    return df


def _harmonize_affect(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-harmonize affect labels to [0, 1] and add binarized versions.

    For each affect construct:
    1. Read per-row ScaleDirection / ScaleMin / ScaleMax when available.
    2. Reverse-code when ScaleDirection indicates decreasing (direction != 1).
    3. Min-max normalize to [0, 1] using metadata bounds only.
    4. Add *_bin column (within-study median split, descriptive only --
       not used in the predictive feature matrix).
    """
    for construct, response_col in AFFECT_RESPONSE_COLS.items():
        prefix = _CONSTRUCT_PREFIX[construct]
        dir_col = f"{prefix}ScaleDirection"
        min_col = f"{prefix}ScaleMin"
        max_col = f"{prefix}ScaleMax"

        raw = pd.to_numeric(df[response_col], errors="coerce")

        # Get scale metadata; fill NaN with per-study fallbacks
        if dir_col in df.columns:
            direction = pd.to_numeric(df[dir_col], errors="coerce")
        else:
            direction = pd.Series(np.nan, index=df.index)

        if min_col in df.columns:
            s_min = pd.to_numeric(df[min_col], errors="coerce")
        else:
            s_min = pd.Series(np.nan, index=df.index)

        if max_col in df.columns:
            s_max = pd.to_numeric(df[max_col], errors="coerce")
        else:
            s_max = pd.Series(np.nan, index=df.index)

        # Scale bounds come from metadata only (ScaleMin / ScaleMax columns).
        # If metadata is absent for a study, _norm stays NaN and the
        # pipeline's per-fold imputer handles it downstream.
        for sid in df["study_id"].unique():
            mask = df["study_id"] == sid
            if direction[mask].isna().all():
                direction.loc[mask] = 1  # default: ascending

        # Reverse-code when direction != 1 (i.e., decreasing scale)
        needs_reverse = direction.notna() & (direction != 1)
        reversed_raw = raw.copy()
        if needs_reverse.any():
            reversed_raw[needs_reverse] = (
                s_min[needs_reverse] + s_max[needs_reverse] - raw[needs_reverse]
            )

        # Min-max normalize to [0, 1]
        denom = s_max - s_min
        denom = denom.replace(0, np.nan)
        norm = (reversed_raw - s_min) / denom

        df[f"{construct}_norm"] = norm

        # Within-study median-split binarization
        binarized = pd.Series(np.nan, index=df.index)
        for sid in df["study_id"].unique():
            mask = df["study_id"] == sid
            study_norm = norm[mask].dropna()
            if study_norm.empty:
                continue
            med = study_norm.median()
            binarized.loc[mask] = (norm[mask] >= med).astype(float)
            binarized.loc[mask & norm.isna()] = np.nan
        df[f"{construct}_bin"] = binarized

    return df


def _harmonize_tut(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure TUT label is binary. Studies with continuous TUT get thresholded."""
    tut = pd.to_numeric(df[LABEL_COL], errors="coerce")

    tut_binary_map = {sid: m["tut_is_binary"] for sid, m in STUDY_META.items()}
    needs_binarize = df["study_id"].map(tut_binary_map) == False  # noqa: E712
    if needs_binarize.any():
        tut_bin = tut.copy()
        tut_bin[needs_binarize] = (tut[needs_binarize] >= 0.5).astype(float)
        df["TUT_continuous"] = tut
        df[LABEL_COL] = tut_bin.astype("Int64")
    else:
        df[LABEL_COL] = tut.astype("Int64")

    return df


def harmonize_all(save: bool = True) -> pd.DataFrame:
    """Run the full harmonization pipeline and return the unified DataFrame."""
    frames = []
    for study_id in STUDY_FILES:
        frames.append(_load_study(study_id))

    df = pd.concat(frames, ignore_index=True).copy()

    # Ensure numeric gaze columns
    for col in GAZE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _add_context_columns(df)
    df = _add_probe_availability_mask(df)
    df = _harmonize_affect(df)
    df = _harmonize_tut(df)

    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TABLES_DIR / "harmonized.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return df
