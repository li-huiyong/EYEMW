"""Feature engineering: two explicit feature spaces.

Feature Space A (deployment): gaze proportions/rates + context, no affect.
Feature Space B (coupling):   everything in A + harmonized affect + lagged vars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import (
    GAZE_COLS, GAZE_PROPORTION_COLS, GAZE_COUNT_COLS,
    AFFECT_RESPONSE_COLS, ALL_CONSTRUCTS,
    LABEL_COL, GROUP_COL, STUDY_COL, PROBE_COL,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _participant_z_scores(df: pd.DataFrame, cols: list[str],
                          train_idx: np.ndarray | None = None) -> pd.DataFrame:
    """Within-study, per-participant z-scores for *cols*.

    Groups by (study_id, ParticipantNum). Falls back to study-level, then
    global training statistics when fewer than 2 observations are available.
    When *train_idx* is provided, statistics are computed from those rows
    only (leak-free CV).
    """
    out = pd.DataFrame(index=df.index)
    is_train = pd.Series(True, index=df.index)
    if train_idx is not None:
        is_train = pd.Series(False, index=df.index)
        is_train.iloc[train_idx] = True

    for col in cols:
        raw = pd.to_numeric(df[col], errors="coerce")
        z = pd.Series(np.nan, index=df.index)
        for (sid, pid), idx in df.groupby(["study_id", GROUP_COL]).groups.items():
            vals = raw.loc[idx]
            train_vals = vals[is_train.loc[idx]]
            if train_vals.notna().sum() < 2:
                study_mask = df["study_id"] == sid
                study_train = raw[study_mask & is_train]
                if study_train.notna().sum() < 2:
                    global_train = raw[is_train]
                    m, s = global_train.mean(), global_train.std()
                else:
                    m, s = study_train.mean(), study_train.std()
            else:
                m, s = train_vals.mean(), train_vals.std()
            s = s if (pd.notna(s) and s > 0) else 1.0
            z.loc[idx] = (vals - m) / s
        out[f"{col}_pz"] = z
    return out


def _study_quantiles(df: pd.DataFrame, cols: list[str],
                     train_idx: np.ndarray | None = None) -> pd.DataFrame:
    """Within-study quantile transform to [0, 1] via training ECDF.

    When *train_idx* is provided, the empirical CDF is built from training
    rows only, then applied to all rows (leak-free CV).
    """
    out = pd.DataFrame(index=df.index)
    is_train = pd.Series(True, index=df.index)
    if train_idx is not None:
        is_train = pd.Series(False, index=df.index)
        is_train.iloc[train_idx] = True

    for col in cols:
        raw = pd.to_numeric(df[col], errors="coerce")
        q = pd.Series(np.nan, index=df.index)
        for sid, idx in df.groupby("study_id").groups.items():
            vals = raw.loc[idx]
            train_vals = vals[is_train.loc[idx]].dropna()
            if len(train_vals) < 2:
                global_train = raw[is_train].dropna()
                if len(global_train) < 2:
                    q.loc[idx] = 0.5
                    continue
                sorted_train = np.sort(global_train.values)
            else:
                sorted_train = np.sort(train_vals.values)
            all_vals = vals.values.astype(float)
            positions = np.searchsorted(sorted_train, all_vals, side="right")
            q.loc[idx] = positions / len(sorted_train)
        out[f"{col}_sq"] = q
    return out


def _missingness_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Binary column per feature: 1 if the value is NaN."""
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if df[col].isna().any():
            out[f"{col}_missing"] = df[col].isna().astype(int)
    return out


def _lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create t-1 lagged affect and TUT within each participant.

    Sorts by (study_id, ParticipantNum, ProbeNum) first.
    """
    sort_cols = ["study_id", GROUP_COL, PROBE_COL]
    df_sorted = df.sort_values(sort_cols)

    lag_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS if f"{c}_norm" in df.columns]
    lag_cols.append(LABEL_COL)

    out = pd.DataFrame(index=df_sorted.index)
    grouped = df_sorted.groupby(["study_id", GROUP_COL])
    for col in lag_cols:
        lagged = grouped[col].shift(1)
        prefix = "prev_" + col.replace("_norm", "").replace("Response", "")
        out[prefix] = lagged

    return out.loc[df.index]


def _temporal_variability(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-participant SD of gaze proportions across probes (session-level variability)."""
    out = pd.DataFrame(index=df.index)
    for col in cols:
        raw = pd.to_numeric(df[col], errors="coerce")
        sd = raw.groupby([df["study_id"], df[GROUP_COL]]).transform("std")
        out[f"{col}_sd"] = sd
    return out


def _gaze_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Shannon entropy over the 3-category gaze distribution (AOI, offscreen, other)."""
    aoi = pd.to_numeric(df["AOIGazeProportion"], errors="coerce").clip(0, 1)
    off = pd.to_numeric(df["OffScreenGazeProportion"], errors="coerce").clip(0, 1)
    other = (1 - aoi - off).clip(0, 1)
    probs = np.column_stack([aoi.values, off.values, other.values])
    probs = np.where(probs <= 0, 1e-10, probs)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    return pd.DataFrame({"gaze_entropy": entropy}, index=df.index)


def _interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Cross gaze proportions with one-hot WindowType for context-dependent gaze patterns."""
    ctx = pd.get_dummies(df["WindowType"], prefix="WindowType", dtype=int)
    out = pd.DataFrame(index=df.index)
    for gaze_col in GAZE_PROPORTION_COLS:
        gaze = pd.to_numeric(df[gaze_col], errors="coerce")
        for ctx_col in ctx.columns:
            name = f"{gaze_col}_x_{ctx_col.replace('WindowType_', '')}"
            out[name] = gaze * ctx[ctx_col]
    return out


def _one_hot_context(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode WindowType and TaskGroup."""
    dummies = pd.get_dummies(
        df[["WindowType", "TaskGroup"]], prefix_sep="_", dtype=int
    )
    return dummies


# ── Public API ───────────────────────────────────────────────────────────────

def build_features_A(
    df: pd.DataFrame, train_idx: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Deployment feature space: gaze + context, no contemporaneous affect.

    Parameters
    ----------
    train_idx : array of positional indices for training rows.  When provided,
        study-level statistics (z-scores, quantiles) use only these rows.

    Returns (X, y, groups).
    """
    parts: list[pd.DataFrame] = []

    # 1. Gaze proportions (directly usable across window types)
    gaze_prop = df[GAZE_PROPORTION_COLS].apply(pd.to_numeric, errors="coerce")
    parts.append(gaze_prop)

    # 2. Derived proportions from counts (robust to window differences)
    gazes = pd.to_numeric(df["Gazes"], errors="coerce")
    for cnt_col in GAZE_COUNT_COLS:
        if cnt_col == "Gazes":
            continue
        cnt = pd.to_numeric(df[cnt_col], errors="coerce")
        denom = gazes.replace(0, np.nan)
        parts.append(pd.DataFrame({f"{cnt_col}_rate": cnt / denom}, index=df.index))

    # 3. Participant-centered gaze z-scores (train_idx-aware)
    parts.append(_participant_z_scores(df, GAZE_PROPORTION_COLS, train_idx))

    # 4. Study-normalized quantiles (train_idx-aware)
    parts.append(_study_quantiles(df, GAZE_PROPORTION_COLS, train_idx))

    # 5. Missingness indicators
    parts.append(_missingness_indicators(df, GAZE_COLS))

    # 6. Temporal variability (within-participant SD across probes)
    parts.append(_temporal_variability(df, GAZE_PROPORTION_COLS))

    # 7. Gaze entropy (3-category distribution)
    parts.append(_gaze_entropy(df))

    # 8. Interaction terms (gaze x window type)
    parts.append(_interaction_terms(df))

    # 9. Context one-hot
    parts.append(_one_hot_context(df))

    X = pd.concat(parts, axis=1)
    y = df[LABEL_COL].astype(int)
    groups = df[GROUP_COL]
    return X, y, groups


def build_features_B(
    df: pd.DataFrame, include_lagged: bool = True,
    train_idx: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Coupling feature space: gaze + affect + context (+ optional lags).

    Returns (X, y, groups).
    """
    X_a, y, groups = build_features_A(df, train_idx=train_idx)
    parts = [X_a]

    # Harmonized current affect (continuous)
    affect_norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS
                        if f"{c}_norm" in df.columns]
    if affect_norm_cols:
        parts.append(df[affect_norm_cols])

    # Lagged variables
    if include_lagged:
        lagged = _lagged_variables(df)
        parts.append(lagged)

    X = pd.concat(parts, axis=1)
    return X, y, groups


def get_feature_matrix(
    df: pd.DataFrame,
    space: str = "A",
    impute: bool = False,
    train_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convenience: build features, optionally impute, return numpy arrays.

    Parameters
    ----------
    space : "A" for deployment, "B" for coupling
    impute : if True, median-impute remaining NaN.  **Warning**: when True the
        median is computed from all rows.  For CV pipelines set False
        (default) and let the pipeline's built-in imputer handle it per fold.
    train_idx : forwarded to the feature builder for leak-free statistics.

    Returns (X, y, groups, feature_names)
    """
    if space == "A":
        X_df, y_s, g_s = build_features_A(df, train_idx=train_idx)
    else:
        X_df, y_s, g_s = build_features_B(df, train_idx=train_idx)

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float, na_value=np.nan)
    y = y_s.to_numpy(dtype=float, na_value=np.nan).astype(int)
    groups = g_s.values

    if impute:
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(X)

    return X, y, groups, feature_names
