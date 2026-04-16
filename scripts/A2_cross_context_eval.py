"""Problem A2: Cross-context evaluation (LOSO).

Evaluates gaze+context LogReg under Leave-One-Study-Out (LOSO) and compares
against the within-study grouped-CV baseline from A1.

LOWO and LOTG are omitted: in EYEMW every WindowType and TaskGroup maps 1:1
to a study, so those strategies produce identical splits to LOSO.
The grouped-CV baseline is loaded from A1 to avoid re-running the same
experiment (A1 gaze_context+LogReg = A2 grouped_cv exactly).

LOSO uses a fixed pipeline (C=1.0) rather than inner GridSearchCV because
the scientific question is study-level generalization and tuning on the
remaining studies introduces unwarranted adaptation to the specific study mix.

Outputs: results/tables/A2_cross_context.csv
         results/figures/A2_generalization_gap.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LABEL_COL, GROUP_COL, TABLES_DIR, FIGURES_DIR, SEED
from src.harmonize import harmonize_all
from src.features import build_features_A
from src.cv import leave_one_study_out
from src.models import get_logreg_pipeline
from src.evaluate import classification_metrics, generalization_gap
from src.utils import set_seed


def main() -> None:
    set_seed(SEED)
    print("Loading and harmonizing data ...")
    df = harmonize_all(save=False)

    y = df[LABEL_COL].astype(int).values
    groups = df[GROUP_COL].values
    study_labels = df["study_id"].values

    # ── LOSO evaluation (fixed pipeline, no inner grid search) ────────────
    print("\n=== Leave-One-Study-Out (LOSO) ===")
    loso_results = []
    pipe_template, _ = get_logreg_pipeline()
    pipe_template.set_params(lr__C=1.0)

    for train_idx, test_idx, fold_info in leave_one_study_out(
            np.empty(len(y)), y, study_labels):
        print(f"  Held out {fold_info['held_out']} "
              f"(train={fold_info['n_train']}, test={fold_info['n_test']}) ...",
              end=" ", flush=True)

        X_df, _, _ = build_features_A(df, train_idx=train_idx)
        X = X_df.values.astype(float)

        pipe, _ = get_logreg_pipeline()
        pipe.set_params(lr__C=1.0)
        pipe.fit(X[train_idx], y[train_idx])

        y_pred = pipe.predict(X[test_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        metrics = classification_metrics(y[test_idx], y_pred, y_prob)

        row = {"cv_type": "LOSO", **fold_info, **metrics}
        loso_results.append(row)
        print(f"macro_f1={metrics['macro_f1']:.4f}, auc={metrics['auc']:.4f}")

    loso_df = pd.DataFrame(loso_results)

    # ── Load grouped-CV baseline from A1 (gaze_context + LogReg) ──────────
    a1_path = TABLES_DIR / "A1_ablation.csv"
    if a1_path.exists():
        a1 = pd.read_csv(a1_path)
        gcv = a1[(a1["condition"] == "gaze_context") &
                  (a1["model"] == "LogReg")].copy()
        gcv["cv_type"] = "grouped_cv"
        print(f"\nLoaded {len(gcv)} grouped-CV rows from A1")
    else:
        print("\nA1 results not found — skipping grouped-CV baseline")
        gcv = pd.DataFrame()

    # ── Combine and save ──────────────────────────────────────────────────
    out = pd.concat([gcv, loso_df], ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "A2_cross_context.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")

    # ── Generalization gap ────────────────────────────────────────────────
    if not gcv.empty:
        within = gcv["macro_f1"].tolist()
        held = loso_df["macro_f1"].tolist()
        gap = generalization_gap(within, held)
        print(f"\nLOSO gap: within={gap['within_mean']:.4f}, "
              f"held_out={gap['held_out_mean']:.4f}, gap={gap['gap_mean']:.4f}")

    # ── Bar plot ──────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cv_types = out["cv_type"].unique()
    means = [out[out["cv_type"] == c]["macro_f1"].mean() for c in cv_types]
    stds = [out[out["cv_type"] == c]["macro_f1"].std() for c in cv_types]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cv_types, means, yerr=stds, capsize=5, alpha=0.8,
           color=["steelblue", "coral"][:len(cv_types)])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Within-Study vs Cross-Study Generalization\n(LogReg, gaze+context)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "A2_generalization_gap.png", dpi=200)
    plt.close(fig)
    print(f"Saved figure to {FIGURES_DIR / 'A2_generalization_gap.png'}")


if __name__ == "__main__":
    main()
