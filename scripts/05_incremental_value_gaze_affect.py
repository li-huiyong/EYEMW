"""Problem B1: Incremental-value models.

On affect-available studies, compares four feature conditions under
participant-grouped CV (LogisticRegression, fixed C=1.0):
  1. Gaze only
  2. Affect only
  3. Gaze + Affect
  4. Gaze + Affect + Context

Per-study incremental value is tested via LOSO (each study is held out,
model trained on the rest) while still showing whether affect helps equally
across studies. Affect columns exclude TUT and drop constructs that are
all-missing on the training fold (leak-free column selection).

Outputs: results/tables/B1_incremental_value.csv
         results/tables/B1_per_study_incremental.csv
         results/figures/B1_delta_auc.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_COL, GROUP_COL,
    ALL_CONSTRUCTS, TABLES_DIR, FIGURES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.features import build_features_A, build_features_B
from src.cv import grouped_nested_cv, leave_one_study_out
from src.models import get_logreg_pipeline
from src.evaluate import classification_metrics
from src.utils import set_seed


def _select_affect_studies(df):
    """Keep only rows from studies that have at least one affect variable."""
    has_any = False
    for c in ALL_CONSTRUCTS:
        if c != "TUT" and f"has_{c}" in df.columns:
            has_any = has_any | (df[f"has_{c}"] == 1)
    return df[has_any].copy()


def _affect_norm_column_names(df: pd.DataFrame, train_idx: np.ndarray | None) -> list[str]:
    """Harmonized affect norms only (never TUT); drop columns that are all-NaN on train."""
    candidates = [
        f"{c}_norm" for c in ALL_CONSTRUCTS
        if c != "TUT" and f"{c}_norm" in df.columns
    ]
    if not candidates:
        return []
    X = df[candidates].apply(pd.to_numeric, errors="coerce")
    if train_idx is not None:
        nan_frac = X.iloc[train_idx].isna().mean()
    else:
        nan_frac = X.isna().mean()
    return nan_frac[nan_frac <= 0.9].index.tolist()


def _build_gaze_only(df, train_idx=None):
    X_a, y, g = build_features_A(df, train_idx=train_idx)
    gaze_cols = [c for c in X_a.columns if not c.startswith(("WindowType_", "TaskGroup_"))]
    return X_a[gaze_cols], y, g, gaze_cols


def _build_affect_only(df, train_idx=None):
    keep = _affect_norm_column_names(df, train_idx)
    X = df[keep].apply(pd.to_numeric, errors="coerce") if keep else pd.DataFrame(index=df.index)
    y = df[LABEL_COL].astype(int)
    g = df[GROUP_COL]
    return X, y, g, keep


def _build_gaze_affect(df, train_idx=None):
    X_gaze, y, g, gaze_cols = _build_gaze_only(df, train_idx=train_idx)
    keep = _affect_norm_column_names(df, train_idx)
    X_aff = df[keep].apply(pd.to_numeric, errors="coerce") if keep else pd.DataFrame(index=df.index)
    X = pd.concat([X_gaze, X_aff], axis=1)
    return X, y, g, list(X.columns)


def _build_gaze_affect_context(df, train_idx=None):
    X_b, y, g = build_features_B(df, include_lagged=False, train_idx=train_idx)
    return X_b, y, g, list(X_b.columns)


def _make_pipe():
    pipe, _ = get_logreg_pipeline()
    pipe.set_params(lr__C=1.0)
    return pipe


def main() -> None:
    set_seed(SEED)
    df = harmonize_all(save=False)
    df_affect = _select_affect_studies(df).reset_index(drop=True)
    print(f"Affect-available subset: {len(df_affect)} rows, "
          f"{df_affect[GROUP_COL].nunique()} participants")

    y = df_affect[LABEL_COL].astype(int).values
    groups = df_affect[GROUP_COL].values

    conditions = {
        "gaze_only": _build_gaze_only,
        "affect_only": _build_affect_only,
        "gaze_affect": _build_gaze_affect,
        "gaze_affect_context": _build_gaze_affect_context,
    }

    # ── Overall grouped-CV ────────────────────────────────────────────────
    results = []
    folds = list(grouped_nested_cv(np.empty(len(y)), y, groups))

    for cond_name, builder in conditions.items():
        n_feat = len(builder(df_affect)[3])
        print(f"\n--- {cond_name} ({n_feat} features) ---")

        for train_idx, test_idx, fold_info in folds:
            print(f"  fold {fold_info['fold_id']} ...", end=" ", flush=True)
            X_df, _, _, _ = builder(df_affect, train_idx=train_idx)
            X = X_df.values.astype(float)

            pipe = _make_pipe()
            pipe.fit(X[train_idx], y[train_idx])

            y_pred = pipe.predict(X[test_idx])
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]
            metrics = classification_metrics(y[test_idx], y_pred, y_prob)

            row = {"condition": cond_name, "n_features": n_feat,
                   **fold_info, **metrics}
            results.append(row)
            print(f"macro_f1={metrics['macro_f1']:.4f}, auc={metrics['auc']:.4f}")

    out = pd.DataFrame(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "B1_incremental_value.csv", index=False)

    print("\n=== Incremental value summary ===")
    summary = out.groupby("condition").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
    ).round(4)
    print(summary.to_string())

    # Delta AUC bar plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    conds = summary.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(conds, summary["auc_mean"].values,
           yerr=summary["auc_std"].values, capsize=5, alpha=0.8)
    ax.set_ylabel("AUC")
    ax.set_title("Incremental Value: AUC by Feature Condition (LogReg)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "B1_delta_auc.png", dpi=200)
    plt.close(fig)
    print(f"Saved figure to {FIGURES_DIR / 'B1_delta_auc.png'}")

    # ── Per-study incremental value via LOSO ──────────────────────────────
    print("\n=== Per-study incremental value (LOSO) ===")
    study_labels = df_affect["study_id"].values
    loso_folds = list(leave_one_study_out(np.empty(len(y)), y, study_labels))

    per_study_results = []
    for train_idx, test_idx, fold_info in loso_folds:
        study = fold_info["held_out"]
        print(f"\n--- Held out {study} ---")
        for cond_name, builder in conditions.items():
            X_df, _, _, _ = builder(df_affect, train_idx=train_idx)
            X = X_df.values.astype(float)

            pipe = _make_pipe()
            pipe.fit(X[train_idx], y[train_idx])

            y_pred = pipe.predict(X[test_idx])
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]
            metrics = classification_metrics(y[test_idx], y_pred, y_prob)
            per_study_results.append({
                "study": study, "condition": cond_name, **metrics,
            })
            print(f"  {cond_name}: macro_f1={metrics['macro_f1']:.4f}, "
                  f"auc={metrics['auc']:.4f}")

    if per_study_results:
        ps_df = pd.DataFrame(per_study_results)
        ps_path = TABLES_DIR / "B1_per_study_incremental.csv"
        ps_df.to_csv(ps_path, index=False)
        print(f"\nSaved to {ps_path}")

        ps_summary = ps_df.groupby(["study", "condition"]).agg(
            macro_f1=("macro_f1", "first"),
            auc=("auc", "first"),
        ).round(4)
        print("\n=== Per-study LOSO summary ===")
        print(ps_summary.to_string())


if __name__ == "__main__":
    main()
