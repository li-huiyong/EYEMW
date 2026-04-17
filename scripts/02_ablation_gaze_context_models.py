"""Problem A1: Ablation ladder -- Gaze only / Context only / Gaze+Context.

Predicts TUT from deployment features (no contemporaneous affect).
Runs 4 sklearn models under participant-grouped nested CV.
Outputs: results/tables/A1_ablation.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LABEL_COL, GROUP_COL, TABLES_DIR, SEED
from src.harmonize import harmonize_all
from src.features import build_features_A
from src.cv import grouped_nested_cv, get_inner_cv
from src.models import ALL_SKLEARN_MODELS, get_logreg_pipeline
from src.evaluate import classification_metrics
from src.utils import set_seed

GAZE_PREFIX_PATTERNS = [
    "UniqueGazeProportion", "OffScreenGazeProportion", "AOIGazeProportion",
    "Gazes", "UniqueGazes", "OffscreenGazes", "AOIGazes",
    "_pz", "_sq", "_rate", "_missing", "_sd", "gaze_entropy",
    "_x_",  # interaction terms (gaze x window)
]
CONTEXT_PREFIX_PATTERNS = ["WindowType_", "TaskGroup_"]


def _select_columns(feature_names: list[str], patterns: list[str]) -> list[int]:
    """Return column indices whose names match any of the given patterns."""
    idxs = []
    for i, name in enumerate(feature_names):
        if any(p in name for p in patterns):
            idxs.append(i)
    return idxs


def main() -> None:
    set_seed(SEED)
    print("Loading and harmonizing data ...")
    df = harmonize_all(save=False)

    y = df[LABEL_COL].astype(int).values
    groups = df[GROUP_COL].values

    # Column names are stable across folds; get them from a reference call
    X_ref, _, _ = build_features_A(df)
    feat_names = list(X_ref.columns)

    gaze_idx = _select_columns(feat_names, GAZE_PREFIX_PATTERNS)
    ctx_idx = _select_columns(feat_names, CONTEXT_PREFIX_PATTERNS)
    all_idx = list(range(len(feat_names)))

    conditions = {
        "gaze_only": gaze_idx,
        "context_only": ctx_idx,
        "gaze_context": all_idx,
    }

    # Pre-compute per-fold feature matrices (avoids rebuilding per model)
    folds = list(grouped_nested_cv(np.empty(len(y)), y, groups))
    fold_X = {}
    for train_idx, _, fold_info in folds:
        X_df, _, _ = build_features_A(df, train_idx=train_idx)
        fold_X[fold_info["fold_id"]] = X_df.values.astype(float)

    results = []

    for cond_name, col_idx in conditions.items():
        if not col_idx:
            print(f"  Skipping {cond_name}: no columns selected")
            continue
        print(f"\n--- Condition: {cond_name} ({len(col_idx)} features) ---")

        if cond_name == "context_only":
            model_name = "LogReg"
            print(f"  Model: {model_name} (direct fit, no grid search)")
            for train_idx, test_idx, fold_info in folds:
                X_full = fold_X[fold_info["fold_id"]]
                X_cond = X_full[:, col_idx]
                pipe, _ = get_logreg_pipeline()
                pipe.set_params(lr__C=1.0)
                pipe.fit(X_cond[train_idx], y[train_idx])

                y_pred = pipe.predict(X_cond[test_idx])
                y_prob = pipe.predict_proba(X_cond[test_idx])[:, 1]
                metrics = classification_metrics(y[test_idx], y_pred, y_prob)

                row = {
                    "condition": cond_name,
                    "model": model_name,
                    "n_features": len(col_idx),
                    **fold_info,
                    **metrics,
                }
                results.append(row)
                print(f"    Fold {fold_info['fold_id']}: macro_f1={metrics['macro_f1']:.4f}")
            continue

        for model_name, model_fn in ALL_SKLEARN_MODELS.items():
            print(f"  Model: {model_name}")
            for train_idx, test_idx, fold_info in folds:
                X_full = fold_X[fold_info["fold_id"]]
                X_cond = X_full[:, col_idx]

                pipe, grid = model_fn()
                inner_cv = get_inner_cv(groups[train_idx])
                search = GridSearchCV(
                    pipe, grid, scoring="f1_macro",
                    cv=inner_cv, n_jobs=1, refit=True,
                )
                search.fit(X_cond[train_idx], y[train_idx], groups=groups[train_idx])
                best = search.best_estimator_

                y_pred = best.predict(X_cond[test_idx])
                y_prob = best.predict_proba(X_cond[test_idx])[:, 1]
                metrics = classification_metrics(y[test_idx], y_pred, y_prob)

                row = {
                    "condition": cond_name,
                    "model": model_name,
                    "n_features": len(col_idx),
                    **fold_info,
                    **metrics,
                }
                results.append(row)
                print(f"    Fold {fold_info['fold_id']}: macro_f1={metrics['macro_f1']:.4f}")

    out = pd.DataFrame(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "A1_ablation.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")

    # Print summary
    print("\n=== Summary (mean +/- std across folds) ===")
    summary = out.groupby(["condition", "model"]).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
    ).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
