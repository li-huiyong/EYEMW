"""Paper Analysis 2: Does model complexity help once the right features are present?

Core scientific question: Given the compact emotion + gaze-ratio feature set,
how much residual nonlinearity exists? If LR and SVM are nearly tied, and
RF/XGBoost/MLP do not substantially surpass them, the conclusion is that the
bottleneck is feature signal and dataset structure, not model class.

Runs 5 models on the same feature set (emotion + 2 gaze ratios) under
identical participant-independent 10-fold grouped nested CV on the affect
subset:
  1. LogReg (L2, tuned C)
  2. SVM (RBF)
  3. Random Forest
  4. XGBoost
  5. PyTorch MLP (16-8 hidden, dropout, grouped early stopping, pos_weight)

Complexity ladder:  linear (LR) -> kernel (SVM) -> ensemble (RF/XGB) -> neural (MLP)

Additionally runs all 5 on the full Feature Space B to test whether model
complexity matters more on a richer feature space.

Features with >90% NaN are dropped before fitting.

Reports: per-model macro-F1 and AUC, pairwise Wilcoxon signed-rank tests.

Outputs:
  results/tables/P2_complexity_compact.csv
  results/tables/P2_complexity_fullB.csv
  results/tables/P2_pairwise_tests.csv
  results/figures/P2_model_complexity.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_COL, GROUP_COL,
    ALL_CONSTRUCTS, TABLES_DIR, FIGURES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.features import build_features_B
from src.cv import grouped_nested_cv, get_inner_cv
from src.models import (
    get_logreg_pipeline, get_svm_pipeline,
    get_rf_pipeline, get_xgb_pipeline, get_mlp_pipeline,
)
from src.evaluate import classification_metrics
from src.utils import set_seed

N_OUTER_FOLDS = 10

GAZE_RATIO_FEATURES = [
    "UniqueGazeProportion",
    "OffScreenGazeProportion",
]

MODELS = {
    "LogReg": get_logreg_pipeline,
    "SVM": get_svm_pipeline,
    "RF": get_rf_pipeline,
    "XGB": get_xgb_pipeline,
    "MLP": get_mlp_pipeline,
}


def _select_affect_studies(df):
    has_any = False
    for c in ALL_CONSTRUCTS:
        if c != "TUT" and f"has_{c}" in df.columns:
            has_any = has_any | (df[f"has_{c}"] == 1)
    return df[has_any].copy()


def _drop_high_nan(X: np.ndarray, col_names: list[str],
                   threshold: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Drop columns where >threshold fraction of values are NaN."""
    nan_frac = np.isnan(X).mean(axis=0)
    keep = nan_frac <= threshold
    kept_names = [c for c, k in zip(col_names, keep) if k]
    dropped = [c for c, k in zip(col_names, keep) if not k]
    if dropped:
        print(f"  Dropped >90% NaN features: {dropped}")
    return X[:, keep], kept_names


def _build_compact(df):
    """Emotion + gaze-ratio features (the compact multimodal set)."""
    norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS if f"{c}_norm" in df.columns]
    norm_cols = [c for c in norm_cols if c != "TUT_norm"]
    gaze = df[GAZE_RATIO_FEATURES].apply(pd.to_numeric, errors="coerce")
    affect = df[norm_cols].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([affect, gaze], axis=1)
    y = df[LABEL_COL].astype(int)
    g = df[GROUP_COL]
    return X, y, g


def _run_models(y, groups, feature_label, *, X_static=None,
                X_per_fold=None, folds=None, col_names=None):
    """Run all models under grouped nested CV.

    For MLP, ``groups`` is forwarded to ``fit()`` for participant-grouped
    early stopping.  For other models the pipeline ignores the extra kwarg.
    """
    if folds is None:
        folds = list(grouped_nested_cv(
            np.empty(len(y)), y, groups, outer_splits=N_OUTER_FOLDS))
    results = []
    for model_name, model_fn in MODELS.items():
        print(f"\n  Model: {model_name}")
        for train_idx, test_idx, fold_info in folds:
            X = X_per_fold[fold_info["fold_id"]] if X_per_fold else X_static

            pipe, grid = model_fn()

            if grid:
                inner_cv = get_inner_cv(groups[train_idx])
                search = GridSearchCV(
                    pipe, grid, scoring="f1_macro",
                    cv=inner_cv, n_jobs=1, refit=True,
                )
                search.fit(X[train_idx], y[train_idx],
                           groups=groups[train_idx])
                best = search.best_estimator_
            else:
                pipe.fit(X[train_idx], y[train_idx],
                         mlp__groups=groups[train_idx])
                best = pipe

            y_pred = best.predict(X[test_idx])
            y_prob = best.predict_proba(X[test_idx])[:, 1]
            metrics = classification_metrics(y[test_idx], y_pred, y_prob)

            results.append({
                "features": feature_label,
                "model": model_name,
                **fold_info, **metrics,
            })
            print(f"    Fold {fold_info['fold_id']}: "
                  f"macro_f1={metrics['macro_f1']:.4f}, auc={metrics['auc']:.4f}",
                  flush=True)

    return results


def _pairwise_tests(df_results, feature_label):
    """Wilcoxon signed-rank test across folds for all model pairs."""
    sub = df_results[df_results["features"] == feature_label]
    models = sub["model"].unique()
    rows = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            s1 = sub[sub["model"] == m1].sort_values("fold_id")["macro_f1"].values
            s2 = sub[sub["model"] == m2].sort_values("fold_id")["macro_f1"].values
            if len(s1) == len(s2) and len(s1) >= 3:
                try:
                    stat, p = wilcoxon(s1, s2)
                except ValueError:
                    stat, p = np.nan, np.nan
            else:
                stat, p = np.nan, np.nan
            rows.append({
                "features": feature_label,
                "model_A": m1, "model_B": m2,
                "mean_f1_A": s1.mean(), "mean_f1_B": s2.mean(),
                "delta": s1.mean() - s2.mean(),
                "wilcoxon_stat": stat, "p_value": p,
            })
    return rows


def main() -> None:
    set_seed(SEED)
    print("Loading and harmonizing data ...")
    df = harmonize_all(save=False)
    df_affect = _select_affect_studies(df)
    print(f"Affect-available subset: {len(df_affect)} rows, "
          f"{df_affect[GROUP_COL].nunique()} participants")

    y_arr = df_affect[LABEL_COL].astype(int).values
    g_arr = df_affect[GROUP_COL].values

    # ── Compact feature set ──
    print("\n=== Compact feature set (emotion + gaze ratios) ===")
    X_c, _, _ = _build_compact(df_affect)
    col_names_c = list(X_c.columns)
    X_c_raw = X_c.values.astype(float)
    X_c_raw, col_names_c = _drop_high_nan(X_c_raw, col_names_c)
    print(f"  Features ({len(col_names_c)}): {col_names_c}")

    folds = list(grouped_nested_cv(
        np.empty(len(y_arr)), y_arr, g_arr, outer_splits=N_OUTER_FOLDS))

    compact_results = _run_models(
        y_arr, g_arr, "compact", X_static=X_c_raw, folds=folds,
        col_names=col_names_c)

    # ── Full B feature set (rebuild per fold for leak-free z-scores) ──
    print("\n=== Full Feature Space B ===")
    fold_X_B = {}
    col_names_B = None
    for train_idx, _, fold_info in folds:
        X_df, _, _ = build_features_B(
            df_affect, include_lagged=False, train_idx=train_idx)
        arr = X_df.values.astype(float)
        cnames = list(X_df.columns)
        if col_names_B is None:
            arr, col_names_B = _drop_high_nan(arr, cnames)
            keep_set = set(col_names_B)
        else:
            keep_mask = [c in keep_set for c in cnames]
            arr = arr[:, keep_mask]
        fold_X_B[fold_info["fold_id"]] = arr
    print(f"  Features: {len(col_names_B)}")

    full_results = _run_models(
        y_arr, g_arr, "full_B", X_per_fold=fold_X_B, folds=folds,
        col_names=col_names_B)

    # Save raw results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    compact_df = pd.DataFrame(compact_results)
    compact_df.to_csv(TABLES_DIR / "P2_complexity_compact.csv", index=False)

    full_df = pd.DataFrame(full_results)
    full_df.to_csv(TABLES_DIR / "P2_complexity_fullB.csv", index=False)

    all_df = pd.concat([compact_df, full_df], ignore_index=True)

    # Summary
    print("\n=== Model complexity summary ===")
    for fl in ["compact", "full_B"]:
        sub = all_df[all_df["features"] == fl]
        summary = sub.groupby("model").agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
        ).round(4)
        print(f"\n--- {fl} ---")
        print(summary.to_string())

    # Pairwise tests
    pw_rows = _pairwise_tests(all_df, "compact")
    pw_rows += _pairwise_tests(all_df, "full_B")
    pw_df = pd.DataFrame(pw_rows).round(4)
    pw_df.to_csv(TABLES_DIR / "P2_pairwise_tests.csv", index=False)
    print(f"\nSaved pairwise tests to {TABLES_DIR / 'P2_pairwise_tests.csv'}")

    sig_tests = pw_df[pw_df["p_value"] < 0.05]
    if sig_tests.empty:
        print("  No significant pairwise differences (p < 0.05)")
    else:
        print("  Significant pairwise differences:")
        print(sig_tests.to_string(index=False))

    for fl in ["compact", "full_B"]:
        sub = all_df[all_df["features"] == fl]
        model_means = sub.groupby("model")["macro_f1"].mean()
        spread = model_means.max() - model_means.min()
        print(f"\n  {fl}: model range = {spread:.4f} "
              f"(best={model_means.idxmax()} {model_means.max():.4f}, "
              f"worst={model_means.idxmin()} {model_means.min():.4f})")

    # Figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, fl, title in [
        (axes[0], "compact", "Compact (emotion + 2 gaze ratios)"),
        (axes[1], "full_B", "Full Feature Space B"),
    ]:
        sub = all_df[all_df["features"] == fl]
        model_order = ["LogReg", "SVM", "RF", "XGB", "MLP"]
        model_order = [m for m in model_order if m in sub["model"].unique()]
        x = np.arange(len(model_order))
        means = [sub[sub["model"] == m]["macro_f1"].mean() for m in model_order]
        stds = [sub[sub["model"] == m]["macro_f1"].std() for m in model_order]

        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=15, fontsize=9)
        ax.set_ylabel("Macro-F1")
        ax.set_title(title)
        ax.set_ylim(0.3, 0.85)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

        spread = max(means) - min(means)
        ax.text(0.95, 0.05, f"range = {spread:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.suptitle("Model Complexity Ceiling: Feature Signal vs Model Class",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "P2_model_complexity.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {FIGURES_DIR / 'P2_model_complexity.png'}")


if __name__ == "__main__":
    main()
