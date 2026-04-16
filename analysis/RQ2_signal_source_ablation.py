"""Paper Analysis RQ2: Where does the TUT prediction signal reside?

Core scientific question: Is the performance gain from the emotion-labeled
subset driven by (a) the emotion features themselves, (b) the gaze-ratio
features, or (c) their interaction?

Compares five feature regimes (names aligned with RQ1 single-study) under identical
participant-independent grouped nested CV on the affect-available subset:

  1. gaze_ratio     -- UniqueGazeProportion, OffScreenGazeProportion only
  2. gaze_rich      -- engineered gaze (Feature Space A minus context dummies)
  3. emotion        -- harmonized affect norms only (no TUT_norm)
  4. emotion_gaze_ratio -- emotion + two gaze-ratio columns (compact multimodal)
  5. emotion_gaze_task  -- Feature Space B: gaze + affect + context, no lags
     (RQ1 single-study emotion_gaze_rich omits context dummies; RQ2 pools
     multi-study and uses emotion_gaze_task with WindowType/TaskGroup one-hot.)

Reports per-condition macro-F1, AUC, kappa with bootstrap 95% CIs.
Computes pairwise delta (emotion_gaze_ratio minus each simpler set) to isolate
incremental contributions.

Uses LogReg as the reference model (near-tied with SVM; keeps the
comparison about features, not model choice).

Outputs:
  results/tables/RQ2_signal_source.csv
  results/tables/RQ2_signal_deltas.csv
  results/figures/RQ2_signal_source.png
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
    LABEL_COL, GROUP_COL, ALL_CONSTRUCTS, TABLES_DIR, FIGURES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.features import build_features_A, build_features_B
from src.cv import grouped_nested_cv, get_inner_cv
from src.models import get_logreg_pipeline
from src.evaluate import classification_metrics, bootstrap_ci
from src.utils import set_seed


GAZE_RATIO_FEATURES = [
    "UniqueGazeProportion",
    "OffScreenGazeProportion",
]

# Display / CSV labels (match RQ1 where applicable)
REGIME_ORDER = [
    "gaze_ratio",
    "gaze_rich",
    "emotion",
    "emotion_gaze_ratio",
    "emotion_gaze_task",
]


def _select_affect_studies(df):
    has_any = False
    for c in ALL_CONSTRUCTS:
        if c != "TUT" and f"has_{c}" in df.columns:
            has_any = has_any | (df[f"has_{c}"] == 1)
    return df[has_any].copy()


def _build_gaze_ratio(df, train_idx=None):
    """Two gaze proportion features only."""
    X = df[GAZE_RATIO_FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    y = df[LABEL_COL].astype(int)
    g = df[GROUP_COL]
    return X, y, g, "gaze_ratio"


def _build_gaze_rich(df, train_idx=None):
    """Engineered gaze (Space A) without context one-hot columns."""
    X_a, y, g = build_features_A(df, train_idx=train_idx)
    gaze_cols = [c for c in X_a.columns
                 if not c.startswith(("WindowType_", "TaskGroup_"))]
    return X_a[gaze_cols], y, g, "gaze_rich"


def _build_emotion(df, train_idx=None):
    """Harmonized affect features only."""
    norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS if f"{c}_norm" in df.columns]
    norm_cols = [c for c in norm_cols if c != "TUT_norm"]
    X = df[norm_cols].apply(pd.to_numeric, errors="coerce").copy()
    y = df[LABEL_COL].astype(int)
    g = df[GROUP_COL]
    return X, y, g, "emotion"


def _build_emotion_gaze_ratio(df, train_idx=None):
    """Emotion + two gaze-ratio columns (compact multimodal)."""
    norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS if f"{c}_norm" in df.columns]
    norm_cols = [c for c in norm_cols if c != "TUT_norm"]
    gaze = df[GAZE_RATIO_FEATURES].apply(pd.to_numeric, errors="coerce")
    affect = df[norm_cols].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([affect, gaze], axis=1)
    y = df[LABEL_COL].astype(int)
    g = df[GROUP_COL]
    return X, y, g, "emotion_gaze_ratio"


def _build_emotion_gaze_task(df, train_idx=None):
    """Feature Space B: gaze + affect + context one-hot, no lags."""
    X, y, g = build_features_B(df, include_lagged=False, train_idx=train_idx)
    return X, y, g, "emotion_gaze_task"


def _macro_f1_metric(y_true, y_pred, _y_prob):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _auc_metric(y_true, _y_pred, y_prob):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float("nan")


def main() -> None:
    set_seed(SEED)
    print("Loading and harmonizing data ...")
    df = harmonize_all(save=False)
    df_affect = _select_affect_studies(df)
    n_part = df_affect[GROUP_COL].nunique()
    print(f"Affect-available subset: {len(df_affect)} rows, {n_part} participants")
    print(f"Studies: {sorted(df_affect['study_id'].unique())}")

    builders = [
        _build_gaze_ratio,
        _build_gaze_rich,
        _build_emotion,
        _build_emotion_gaze_ratio,
        _build_emotion_gaze_task,
    ]

    y = df_affect[LABEL_COL].astype(int).values
    groups = df_affect[GROUP_COL].values
    n = len(y)

    all_results = []
    condition_preds = {}

    for builder in builders:
        X_ref, _, _, cond_name = builder(df_affect)
        feat_names = list(X_ref.columns)
        print(f"\n--- {cond_name} ({len(feat_names)} features: {feat_names}) ---")

        cond_y_true_all, cond_y_pred_all, cond_y_prob_all = [], [], []
        cond_groups_all = []

        for train_idx, test_idx, fold_info in grouped_nested_cv(
                np.empty(n), y, groups):
            X_df, _, _, _ = builder(df_affect, train_idx=train_idx)
            X = X_df.values.astype(float)

            pipe, grid = get_logreg_pipeline()
            inner_cv = get_inner_cv(groups[train_idx])
            search = GridSearchCV(
                pipe, grid, scoring="f1_macro",
                cv=inner_cv, n_jobs=1, refit=True,
            )
            search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
            best = search.best_estimator_

            y_pred = best.predict(X[test_idx])
            y_prob = best.predict_proba(X[test_idx])[:, 1]
            metrics = classification_metrics(y[test_idx], y_pred, y_prob)

            row = {"condition": cond_name, "n_features": len(feat_names),
                   **fold_info, **metrics}
            all_results.append(row)

            cond_y_true_all.append(y[test_idx])
            cond_y_pred_all.append(y_pred)
            cond_y_prob_all.append(y_prob)
            cond_groups_all.append(groups[test_idx])

            print(f"  Fold {fold_info['fold_id']}: "
                  f"macro_f1={metrics['macro_f1']:.4f}, auc={metrics['auc']:.4f}")

        condition_preds[cond_name] = {
            "y_true": np.concatenate(cond_y_true_all),
            "y_pred": np.concatenate(cond_y_pred_all),
            "y_prob": np.concatenate(cond_y_prob_all),
            "groups": np.concatenate(cond_groups_all),
        }

    out = pd.DataFrame(all_results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "RQ2_signal_source.csv", index=False)

    # Summary
    print("\n=== Signal source ablation summary ===")
    summary = out.groupby("condition").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        kappa_mean=("kappa", "mean"),
    ).round(4)
    print(summary.to_string())

    # Bootstrap CIs for pooled OOF predictions (cluster bootstrap)
    print("\n=== Bootstrap 95% CIs (pooled out-of-fold, cluster bootstrap) ===")
    ci_rows = []
    for cond_name, preds in condition_preds.items():
        yt, yp, ypr = preds["y_true"], preds["y_pred"], preds["y_prob"]
        grp = preds["groups"]
        f1_pt, f1_lo, f1_hi = bootstrap_ci(
            _macro_f1_metric, yt, yp, ypr,
            n_boot=2000, seed=SEED, groups=grp)
        auc_pt, auc_lo, auc_hi = bootstrap_ci(
            _auc_metric, yt, yp, ypr,
            n_boot=2000, seed=SEED, groups=grp)
        ci_rows.append({
            "condition": cond_name,
            "macro_f1": f1_pt, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
            "auc": auc_pt, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        })
        print(f"  {cond_name}: F1={f1_pt:.4f} [{f1_lo:.4f}, {f1_hi:.4f}], "
              f"AUC={auc_pt:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")

    # Pairwise deltas relative to emotion_gaze_ratio
    print("\n=== Incremental deltas (emotion_gaze_ratio - X) ===")
    delta_rows = []
    ref = condition_preds["emotion_gaze_ratio"]
    for cond_name, preds in condition_preds.items():
        if cond_name == "emotion_gaze_ratio":
            continue

        def _delta_f1(y_true, y_pred, y_prob):
            from sklearn.metrics import f1_score
            ref_f1 = f1_score(ref["y_true"], ref["y_pred"], average="macro", zero_division=0)
            alt_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            return ref_f1 - alt_f1

        pt, lo, hi = bootstrap_ci(
            _delta_f1, preds["y_true"], preds["y_pred"],
            preds["y_prob"], n_boot=2000, seed=SEED,
            groups=preds["groups"])
        delta_rows.append({
            "comparison": f"emotion_gaze_ratio - {cond_name}",
            "delta_f1": pt, "delta_ci_lo": lo, "delta_ci_hi": hi,
        })
        print(f"  emotion_gaze_ratio - {cond_name}: "
              f"delta_F1={pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(TABLES_DIR / "RQ2_signal_deltas.csv", index=False)
    print(f"\nSaved to {TABLES_DIR / 'RQ2_signal_deltas.csv'}")

    # Figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ci_df = pd.DataFrame(ci_rows)
    ci_df = ci_df.set_index("condition").loc[REGIME_ORDER].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(ci_df))
    for ax, metric, lo_col, hi_col, label in [
        (axes[0], "macro_f1", "f1_ci_lo", "f1_ci_hi", "Macro-F1"),
        (axes[1], "auc", "auc_ci_lo", "auc_ci_hi", "AUC"),
    ]:
        vals = ci_df[metric].values
        errs = np.array([vals - ci_df[lo_col].values,
                         ci_df[hi_col].values - vals])
        ax.bar(x, vals, yerr=errs, capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ci_df["condition"], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(0.3, max(0.85, vals.max() + 0.1))
        ax.set_title(f"Signal Source Ablation: {label}")
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "RQ2_signal_source.png", dpi=200)
    plt.close(fig)
    print(f"Saved figure to {FIGURES_DIR / 'RQ2_signal_source.png'}")


if __name__ == "__main__":
    main()
