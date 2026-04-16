"""Paper Analysis 5 (RQ3): Does emotion-aware TUT prediction generalize
across learning contexts?

Leave-one-study-out evaluation on the affect-available subset with three
feature conditions (gaze_only, emotion_only, emotion_gaze).  Within-study
baselines are loaded from P1 results.

LOWO / LOTG are omitted because WindowType and TaskGroup map 1:1 to
study_id in EYEMW.  Each held-out row is annotated with TaskGroup and
WindowType so the table can be read as task-family transfer.

Outputs:
  results/tables/P5_cross_context.csv
  results/tables/P5_generalization_gap.csv
  results/figures/P5_cross_context.png
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
    LABEL_COL, GROUP_COL, ALL_CONSTRUCTS, STUDY_META,
    GAZE_PROPORTION_COLS, TABLES_DIR, FIGURES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.features import build_features_A
from src.cv import leave_one_study_out
from src.models import get_logreg_pipeline
from src.evaluate import classification_metrics, generalization_gap
from src.utils import set_seed


def _select_affect_studies(df):
    has_any = False
    for c in ALL_CONSTRUCTS:
        if c != "TUT" and f"has_{c}" in df.columns:
            has_any = has_any | (df[f"has_{c}"] == 1)
    return df[has_any].copy()


def _build_gaze_only(df, train_idx=None):
    X_a, _, _ = build_features_A(df, train_idx=train_idx)
    gaze_cols = [c for c in X_a.columns
                 if not c.startswith(("WindowType_", "TaskGroup_"))]
    return X_a[gaze_cols].values.astype(float)


def _build_emotion_only(df, train_idx=None):
    norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS
                 if f"{c}_norm" in df.columns and c != "TUT"]
    X = df[norm_cols].apply(pd.to_numeric, errors="coerce")
    nan_frac = X.isna().mean()
    keep = nan_frac[nan_frac <= 0.9].index.tolist()
    return X[keep].values.astype(float)


def _build_emotion_gaze(df, train_idx=None):
    gaze = _build_gaze_only(df, train_idx=train_idx)
    emo = _build_emotion_only(df, train_idx=train_idx)
    return np.column_stack([emo, gaze])


def _make_pipe():
    pipe, _ = get_logreg_pipeline()
    pipe.set_params(lr__C=1.0)
    return pipe


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
        "gaze_only": _build_gaze_only,
        "emotion_only": _build_emotion_only,
        "emotion_gaze": _build_emotion_gaze,
    }

    # ── LOSO evaluation ──────────────────────────────────────────────────
    print("\n=== Leave-One-Study-Out ===")
    loso_rows = []
    for train_idx, test_idx, fold_info in leave_one_study_out(
            np.empty(len(y)), y, study_labels):
        study = fold_info["held_out"]
        meta = STUDY_META.get(study, {})
        print(f"\n  Held out: {study} ({meta.get('task_group', '?')}, "
              f"{meta.get('window_type', '?')})", flush=True)

        for cond, builder in builders.items():
            X = builder(df_affect, train_idx=train_idx)
            pipe = _make_pipe()
            pipe.fit(X[train_idx], y[train_idx])

            y_pred = pipe.predict(X[test_idx])
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]
            metrics = classification_metrics(y[test_idx], y_pred, y_prob)

            row = {
                "held_out_study": study,
                "task_group": meta.get("task_group", ""),
                "window_type": meta.get("window_type", ""),
                "condition": cond,
                "n_test": len(test_idx),
                **metrics,
            }
            loso_rows.append(row)
            print(f"    {cond}: macro_f1={metrics['macro_f1']:.4f}, "
                  f"auc={metrics['auc']:.4f}", flush=True)

    loso_df = pd.DataFrame(loso_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    loso_df.to_csv(TABLES_DIR / "P5_cross_context.csv", index=False)
    print(f"\nSaved {len(loso_df)} rows to P5_cross_context.csv")

    # ── Within-study baseline from P1 ────────────────────────────────────
    p1_path = TABLES_DIR / "P1_signal_source.csv"
    gap_rows = []
    if p1_path.exists():
        p1 = pd.read_csv(p1_path)
        print("\n=== Generalization Gap (within from P1 vs LOSO) ===")
        for cond in builders:
            within_vals = p1[p1["condition"] == cond]["macro_f1"].tolist()
            loso_vals = loso_df[loso_df["condition"] == cond]["macro_f1"].tolist()
            if within_vals and loso_vals:
                gap = generalization_gap(within_vals, loso_vals)
                gap_rows.append({"condition": cond, **gap})
                print(f"  {cond}: within={gap['within_mean']:.4f}, "
                      f"LOSO={gap['held_out_mean']:.4f}, "
                      f"gap={gap['gap_mean']:.4f}")
    else:
        print("\nP1 results not found; skipping generalization gap.")

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        gap_df.to_csv(TABLES_DIR / "P5_generalization_gap.csv", index=False)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n=== LOSO summary by condition ===")
    summary = loso_df.groupby("condition").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
    ).round(4)
    print(summary.to_string())

    # ── Figure ───────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    studies = loso_df["held_out_study"].unique()
    conds = list(builders.keys())
    x = np.arange(len(studies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(conds):
        sub = loso_df[loso_df["condition"] == cond]
        vals = [sub[sub["held_out_study"] == s]["macro_f1"].values[0]
                for s in studies]
        labels_text = [f"{s}\n({STUDY_META.get(s, {}).get('task_group', '')})"
                       for s in studies]
        ax.bar(x + i * width, vals, width, label=cond, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels_text, fontsize=8)
    ax.set_ylabel("Macro-F1 (held-out study)")
    ax.set_title("RQ3: Cross-Context Generalization (LOSO)")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", ls="--", alpha=0.4, lw=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "P5_cross_context.png", dpi=200)
    plt.close(fig)
    print(f"Saved figure to {FIGURES_DIR / 'P5_cross_context.png'}")


if __name__ == "__main__":
    main()
