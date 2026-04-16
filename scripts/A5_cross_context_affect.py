"""Problem A5: Cross-context evaluation with gaze + affect (LOSO).

On the affect-available subset, compares gaze-only vs gaze+affect LogReg
under grouped-CV and LOSO to test whether adding affect improves cross-study
generalization.

LOWO / LOTG are omitted (1:1 study mapping makes them equivalent to LOSO).
Grouped-CV uses inner GridSearchCV; LOSO uses a fixed pipeline (C=1.0)
for the same rationale as A2.

Outputs: results/tables/A5_cross_context_affect.csv
         results/figures/A5_generalization_gap_affect.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_COL, GROUP_COL, ALL_CONSTRUCTS,
    TABLES_DIR, FIGURES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.features import build_features_A, build_features_B
from src.cv import grouped_nested_cv, leave_one_study_out, get_inner_cv
from src.models import get_logreg_pipeline
from src.evaluate import classification_metrics
from src.utils import set_seed


def _select_affect_studies(df):
    has_any = False
    for c in ALL_CONSTRUCTS:
        if c != "TUT" and f"has_{c}" in df.columns:
            has_any = has_any | (df[f"has_{c}"] == 1)
    return df[has_any].copy()


def _run_loso(df, y, groups, study_labels, feature_label, builder):
    """LOSO with fixed C=1.0 pipeline (no inner grid search)."""
    results = []
    for train_idx, test_idx, fold_info in leave_one_study_out(
            np.empty(len(y)), y, study_labels):
        print(f"  {fold_info['held_out']} ...", end=" ", flush=True)
        X_df, _, _ = builder(df, train_idx=train_idx)
        X = X_df.values.astype(float)

        pipe, _ = get_logreg_pipeline()
        pipe.set_params(lr__C=1.0)
        pipe.fit(X[train_idx], y[train_idx])

        y_pred = pipe.predict(X[test_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        metrics = classification_metrics(y[test_idx], y_pred, y_prob)
        results.append({"features": feature_label, "cv_type": "LOSO",
                         **fold_info, **metrics})
        print(f"macro_f1={metrics['macro_f1']:.4f}, auc={metrics['auc']:.4f}")
    return results


def _run_grouped_cv(df, y, groups, feature_label, builder):
    """Participant-grouped CV with inner GridSearchCV."""
    results = []
    for train_idx, test_idx, fold_info in grouped_nested_cv(
            np.empty(len(y)), y, groups):
        print(f"  fold {fold_info['fold_id']} ...", end=" ", flush=True)
        X_df, _, _ = builder(df, train_idx=train_idx)
        X = X_df.values.astype(float)

        pipe, grid = get_logreg_pipeline()
        inner_cv = get_inner_cv(groups[train_idx])
        search = GridSearchCV(pipe, grid, scoring="f1_macro",
                              cv=inner_cv, n_jobs=1, refit=True)
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        best = search.best_estimator_

        y_pred = best.predict(X[test_idx])
        y_prob = best.predict_proba(X[test_idx])[:, 1]
        metrics = classification_metrics(y[test_idx], y_pred, y_prob)
        results.append({"features": feature_label, "cv_type": "grouped_cv",
                         **fold_info, **metrics})
        print(f"macro_f1={metrics['macro_f1']:.4f}, auc={metrics['auc']:.4f}")
    return results


def main() -> None:
    set_seed(SEED)
    print("Loading and harmonizing data ...")
    df = harmonize_all(save=False)
    df_affect = _select_affect_studies(df).reset_index(drop=True)
    print(f"Affect-available subset: {len(df_affect)} rows, "
          f"{df_affect[GROUP_COL].nunique()} participants")

    y = df_affect[LABEL_COL].astype(int).values
    groups = df_affect[GROUP_COL].values
    study_labels = df_affect["study_id"].values

    builders = {
        "gaze_context": lambda df, train_idx=None: build_features_A(df, train_idx=train_idx),
        "gaze_affect_context": lambda df, train_idx=None: build_features_B(
            df, include_lagged=False, train_idx=train_idx),
    }

    all_results = []
    for feat_label, builder in builders.items():
        print(f"\n=== {feat_label} / grouped_cv ===")
        all_results.extend(_run_grouped_cv(
            df_affect, y, groups, feat_label, builder))
        print(f"\n=== {feat_label} / LOSO ===")
        all_results.extend(_run_loso(
            df_affect, y, groups, study_labels, feat_label, builder))

    out = pd.DataFrame(all_results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "A5_cross_context_affect.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")

    # Summary
    print("\n=== Summary (mean macro-F1 by feature set and CV strategy) ===")
    summary = out.groupby(["features", "cv_type"])["macro_f1"].agg(
        ["mean", "std"]).round(4)
    print(summary.to_string())

    # Plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cv_types = ["grouped_cv", "LOSO"]
    feat_labels = list(builders.keys())
    x = np.arange(len(cv_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, fl in enumerate(feat_labels):
        sub = out[out["features"] == fl]
        means = [sub[sub["cv_type"] == c]["macro_f1"].mean() for c in cv_types]
        stds = [sub[sub["cv_type"] == c]["macro_f1"].std() for c in cv_types]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=fl, alpha=0.8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(cv_types)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-Context: Gaze vs Gaze+Affect (LogReg, affect subset)")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "A5_generalization_gap_affect.png", dpi=200)
    plt.close(fig)
    print(f"Saved figure to {FIGURES_DIR / 'A5_generalization_gap_affect.png'}")


if __name__ == "__main__":
    main()
