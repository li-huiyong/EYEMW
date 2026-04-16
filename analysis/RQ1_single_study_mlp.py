"""Paper Analysis RQ1: Single-study models on clean, homogeneous data (study 010-011).

Uses TorchMLPClassifier (dropout, grouped early stopping), harmonized data, and
10-fold grouped nested CV. Compares five feature regimes (coherent ladder):

  1. gaze_ratio     -- UniqueGazeProportion, OffScreenGazeProportion
  2. gaze_rich      -- engineered gaze (Feature Space A minus context dummies)
  3. emotion        -- harmonized affect norms (no TUT_norm)
  4. emotion_gaze_ratio -- emotion + two gaze-ratio columns
  5. emotion_gaze_rich  -- gaze_rich + emotion (no context dummies)

Runs all 5 classifiers (LogReg, SVM, RF, XGB, MLP) on each regime.

Outputs:
  results/tables/RQ1_single_study.csv
  results/tables/RQ1_single_study_pairwise.csv
  results/figures/RQ1_single_study.png
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LABEL_COL, GROUP_COL, ALL_CONSTRUCTS, TABLES_DIR, FIGURES_DIR, SEED
from src.harmonize import harmonize_all
from src.features import build_features_A
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

# Order for tables and figure panels
REGIME_ORDER = [
    "gaze_ratio",
    "gaze_rich",
    "emotion",
    "emotion_gaze_ratio",
    "emotion_gaze_rich",
]

MODELS = {
    "LogReg": get_logreg_pipeline,
    "SVM": get_svm_pipeline,
    "RF": get_rf_pipeline,
    "XGB": get_xgb_pipeline,
    "MLP": get_mlp_pipeline,
}


def _norm_cols_emotion(df: pd.DataFrame) -> list[str]:
    cols = [f"{c}_norm" for c in ALL_CONSTRUCTS if f"{c}_norm" in df.columns]
    cols = [c for c in cols if c != "TUT_norm"]
    # Omit norms with no observed values in this subset (avoids empty-column impute warnings)
    return [c for c in cols if df[c].notna().sum() > 0]


def _build_gaze_ratio(df: pd.DataFrame, train_idx=None) -> pd.DataFrame:
    return df[GAZE_RATIO_FEATURES].apply(pd.to_numeric, errors="coerce").copy()


def _build_gaze_rich(df: pd.DataFrame, train_idx=None) -> pd.DataFrame:
    X_a, _, _ = build_features_A(df, train_idx=train_idx)
    gaze_cols = [c for c in X_a.columns
                 if not c.startswith(("WindowType_", "TaskGroup_"))]
    return X_a[gaze_cols].copy()


def _build_emotion(df: pd.DataFrame, train_idx=None) -> pd.DataFrame:
    cols = _norm_cols_emotion(df)
    return df[cols].apply(pd.to_numeric, errors="coerce").copy()


def _build_emotion_gaze_ratio(df: pd.DataFrame, train_idx=None) -> pd.DataFrame:
    emo = _build_emotion(df)
    gaze = df[GAZE_RATIO_FEATURES].apply(pd.to_numeric, errors="coerce")
    return pd.concat([emo, gaze], axis=1)


def _build_emotion_gaze_rich(df: pd.DataFrame, train_idx=None) -> pd.DataFrame:
    rich = _build_gaze_rich(df, train_idx=train_idx)
    emo = _build_emotion(df)
    return pd.concat([emo, rich], axis=1)


REGIME_BUILDERS = {
    "gaze_ratio": _build_gaze_ratio,
    "gaze_rich": _build_gaze_rich,
    "emotion": _build_emotion,
    "emotion_gaze_ratio": _build_emotion_gaze_ratio,
    "emotion_gaze_rich": _build_emotion_gaze_rich,
}


def _run_models(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_label: str,
    folds: list,
    builder,
) -> list:
    """Fit each model on each outer fold; builder(df, train_idx) returns X DataFrame."""
    results = []
    for model_name, model_fn in MODELS.items():
        print(f"\n  Model: {model_name}")
        for train_idx, test_idx, fold_info in folds:
            X_df = builder(df, train_idx=train_idx)
            X = X_df.values.astype(float)
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
                "n_features": X.shape[1],
                **fold_info, **metrics,
            })
            print(f"    Fold {fold_info['fold_id']}: "
                  f"macro_f1={metrics['macro_f1']:.4f}, auc={metrics['auc']:.4f}",
                  flush=True)
    return results


def _pairwise_tests(df_results: pd.DataFrame, feature_label: str) -> list:
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
    warnings.filterwarnings(
        "ignore",
        message="Skipping features without any observed values",
        category=UserWarning,
        module="sklearn.impute._base",
    )
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

    train0 = folds[0][0]
    all_results = []

    for regime in REGIME_ORDER:
        builder = REGIME_BUILDERS[regime]
        X_ref = builder(df_s, train_idx=train0)
        print(f"\n=== {regime} ({X_ref.shape[1]} features) ===")
        print(f"  Columns: {list(X_ref.columns)}")
        print(f"  NaN fraction: {X_ref.isna().mean().to_dict()}")

        rows = _run_models(
            df_s, y, groups, regime, folds, builder,
        )
        all_results.extend(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(TABLES_DIR / "RQ1_single_study.csv", index=False)

    # ── Summary ──
    print("\n=== Single-study model comparison (five regimes) ===")
    for fl in REGIME_ORDER:
        sub = all_df[all_df["features"] == fl]
        summary = sub.groupby("model").agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
        ).round(4)
        print(f"\n--- {fl} ---")
        print(summary.to_string())

    # ── Pairwise tests (per regime) ──
    pw_rows = []
    for fl in REGIME_ORDER:
        pw_rows += _pairwise_tests(all_df, fl)
    pw_df = pd.DataFrame(pw_rows).round(4)
    pw_df.to_csv(TABLES_DIR / "RQ1_single_study_pairwise.csv", index=False)

    sig = pw_df[pw_df["p_value"] < 0.05]
    if sig.empty:
        print("\n  No significant pairwise model differences (p < 0.05)")
    else:
        print("\n  Significant pairwise model differences:")
        print(sig.to_string(index=False))

    # ── Figure: one panel per regime ──
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_reg = len(REGIME_ORDER)
    fig, axes = plt.subplots(1, n_reg, figsize=(3.2 * n_reg, 5), sharey=True)
    if n_reg == 1:
        axes = [axes]

    titles = {
        "gaze_ratio": "Gaze ratio (2)",
        "gaze_rich": "Gaze rich",
        "emotion": "Emotion",
        "emotion_gaze_ratio": "Emotion + gaze ratio",
        "emotion_gaze_rich": "Emotion + gaze rich",
    }

    for ax, fl in zip(axes, REGIME_ORDER):
        sub = all_df[all_df["features"] == fl]
        model_order = ["LogReg", "SVM", "RF", "XGB", "MLP"]
        model_order = [m for m in model_order if m in sub["model"].unique()]
        x = np.arange(len(model_order))
        means = [sub[sub["model"] == m]["macro_f1"].mean() for m in model_order]
        stds = [sub[sub["model"] == m]["macro_f1"].std() for m in model_order]

        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Macro-F1")
        ax.set_title(titles.get(fl, fl), fontsize=9)
        ax.set_ylim(0.3, 0.85)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

        spread = max(means) - min(means)
        ax.text(0.95, 0.05, f"range={spread:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, style="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="wheat", alpha=0.5))

    plt.suptitle("Study 010-011: Model comparison (five feature regimes)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "RQ1_single_study.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {FIGURES_DIR / 'RQ1_single_study.png'}")
    print(f"Saved tables to {TABLES_DIR / 'RQ1_single_study.csv'}, "
          f"{TABLES_DIR / 'RQ1_single_study_pairwise.csv'}")


if __name__ == "__main__":
    main()
