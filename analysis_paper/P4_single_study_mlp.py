"""Paper Analysis 4: Single-study MLP on clean, homogeneous data (studies 010-011).

Replicates the old emodata MLP analysis using the updated infrastructure
(TorchMLPClassifier with dropout and grouped early stopping, harmonized data
with correct participant IDs).

Runs all 5 models under 10-fold GroupKFold on two feature conditions:
  1. raw_5: the exact 5 raw features from the old analysis
     (ValenceResponse, ArousalResponse, BoredomResponse,
      AOIGazeProportion, OffScreenGazeProportion) -- zero missingness.
  2. norm_5: the harmonized equivalents
     (valence_norm, arousal_norm, boredom_norm,
      UniqueGazeProportion, OffScreenGazeProportion).

Results are directly comparable to P2 (same TorchMLPClassifier, same
classification_metrics, same CSV schema) so the paper can quantify the
cost of multi-study heterogeneity.

Outputs:
  results/tables/P4_single_study_raw.csv
  results/tables/P4_single_study_norm.csv
  results/tables/P4_single_study_pairwise.csv
  results/figures/P4_single_study.png
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

from src.config import LABEL_COL, GROUP_COL, TABLES_DIR, FIGURES_DIR, SEED
from src.harmonize import harmonize_all
from src.cv import grouped_nested_cv, get_inner_cv
from src.models import (
    get_logreg_pipeline, get_svm_pipeline,
    get_rf_pipeline, get_xgb_pipeline, get_mlp_pipeline,
)
from src.evaluate import classification_metrics
from src.utils import set_seed

N_OUTER_FOLDS = 10

RAW_FEATURES = [
    "ValenceResponse",
    "ArousalResponse",
    "BoredomResponse",
    "AOIGazeProportion",
    "OffScreenGazeProportion",
]

NORM_FEATURES = [
    "valence_norm",
    "arousal_norm",
    "boredom_norm",
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


def _run_models(X, y, groups, feature_label, folds):
    results = []
    for model_name, model_fn in MODELS.items():
        print(f"\n  Model: {model_name}")
        for train_idx, test_idx, fold_info in folds:
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

    df_s = df[df["study_id"] == "010-011"].copy().reset_index(drop=True)
    n_p = df_s[GROUP_COL].nunique()
    tut_rate = df_s[LABEL_COL].mean()
    print(f"Study 010-011: {len(df_s)} rows, {n_p} participants, "
          f"TUT rate={tut_rate:.3f}")

    y = df_s[LABEL_COL].astype(int).values
    groups = df_s[GROUP_COL].values

    folds = list(grouped_nested_cv(
        np.empty(len(y)), y, groups, outer_splits=N_OUTER_FOLDS))

    # ── Raw 5 features (exact old-MLP replication) ──
    print("\n=== Raw 5 features (old MLP replication) ===")
    X_raw = df_s[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    print(f"  Features: {list(X_raw.columns)}")
    print(f"  NaN fraction: {X_raw.isna().mean().to_dict()}")
    X_raw_arr = X_raw.values.astype(float)

    raw_results = _run_models(X_raw_arr, y, groups, "raw_5", folds)

    # ── Normalized 5 features (harmonized pipeline) ──
    print("\n=== Normalized 5 features (harmonized pipeline) ===")
    X_norm = df_s[NORM_FEATURES].apply(pd.to_numeric, errors="coerce")
    print(f"  Features: {list(X_norm.columns)}")
    print(f"  NaN fraction: {X_norm.isna().mean().to_dict()}")
    X_norm_arr = X_norm.values.astype(float)

    norm_results = _run_models(X_norm_arr, y, groups, "norm_5", folds)

    # ── Save ──
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(TABLES_DIR / "P4_single_study_raw.csv", index=False)

    norm_df = pd.DataFrame(norm_results)
    norm_df.to_csv(TABLES_DIR / "P4_single_study_norm.csv", index=False)

    all_df = pd.concat([raw_df, norm_df], ignore_index=True)

    # ── Summary ──
    print("\n=== Single-study model comparison ===")
    for fl in ["raw_5", "norm_5"]:
        sub = all_df[all_df["features"] == fl]
        summary = sub.groupby("model").agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
        ).round(4)
        print(f"\n--- {fl} ---")
        print(summary.to_string())

    # ── Pairwise tests ──
    pw_rows = _pairwise_tests(all_df, "raw_5")
    pw_rows += _pairwise_tests(all_df, "norm_5")
    pw_df = pd.DataFrame(pw_rows).round(4)
    pw_df.to_csv(TABLES_DIR / "P4_single_study_pairwise.csv", index=False)

    sig = pw_df[pw_df["p_value"] < 0.05]
    if sig.empty:
        print("\n  No significant pairwise differences (p < 0.05)")
    else:
        print("\n  Significant pairwise differences:")
        print(sig.to_string(index=False))

    # ── Figure ──
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, fl, title in [
        (axes[0], "raw_5", "Raw 5 features (old replication)"),
        (axes[1], "norm_5", "Harmonized 5 features"),
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

    plt.suptitle("Study 010-011: Single-Study Model Comparison",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "P4_single_study.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {FIGURES_DIR / 'P4_single_study.png'}")


if __name__ == "__main__":
    main()
