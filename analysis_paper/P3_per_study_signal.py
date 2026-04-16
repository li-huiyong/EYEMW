"""Paper Analysis 3: Per-study signal decomposition.

Core scientific question: Where does the TUT prediction signal reside
across the EYEMW studies? Does emotion help equally everywhere, or is
the aggregate gain dominated by specific study contexts?

For each affect-available study, runs the same 3 conditions from P1:
  gaze_only, emotion_only, emotion_gaze
under within-study participant-grouped CV (using LogReg).

Also runs all studies (including 001-002 which has no emotion) on
gaze_only to establish the baseline landscape.

This directly tests whether:
  (a) gaze-only is near-random in some studies but useful in others,
  (b) emotion explains most of the gain in specific study contexts, and
  (c) gaze adds incremental value on top of emotion uniformly or selectively.

Outputs:
  results/tables/P3_per_study_signal.csv
  results/figures/P3_per_study_signal.png
  results/figures/P3_incremental_gain.png
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
    LABEL_COL, GROUP_COL, GAZE_PROPORTION_COLS,
    ALL_CONSTRUCTS, STUDY_META, TABLES_DIR, FIGURES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.features import build_features_A
from src.cv import grouped_nested_cv, get_inner_cv
from src.models import get_logreg_pipeline
from src.evaluate import classification_metrics
from src.utils import set_seed


GAZE_RATIO_FEATURES = [
    "UniqueGazeProportion",
    "OffScreenGazeProportion",
]


def _build_gaze_only(df, train_idx=None):
    X_a, y, g = build_features_A(df, train_idx=train_idx)
    gaze_cols = [c for c in X_a.columns
                 if not c.startswith(("WindowType_", "TaskGroup_"))]
    return X_a[gaze_cols], y, g


def _build_emotion_only(df, train_idx=None):
    norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS if f"{c}_norm" in df.columns]
    norm_cols = [c for c in norm_cols if c != "TUT_norm"]
    available = [c for c in norm_cols if df[c].notna().sum() > 5]
    if not available:
        return None, None, None
    X = df[available].apply(pd.to_numeric, errors="coerce").copy()
    y = df[LABEL_COL].astype(int)
    g = df[GROUP_COL]
    return X, y, g


def _build_emotion_gaze(df, train_idx=None):
    norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS if f"{c}_norm" in df.columns]
    norm_cols = [c for c in norm_cols if c != "TUT_norm"]
    available = [c for c in norm_cols if df[c].notna().sum() > 5]
    if not available:
        return None, None, None
    gaze = df[GAZE_RATIO_FEATURES].apply(pd.to_numeric, errors="coerce")
    affect = df[available].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([affect, gaze], axis=1)
    y = df[LABEL_COL].astype(int)
    g = df[GROUP_COL]
    return X, y, g


def _run_cv_for_study(df_s, builder, study_id, cond_name):
    """Run grouped CV for one study/condition. Return list of result dicts."""
    y = df_s[LABEL_COL].astype(int).values
    groups = df_s[GROUP_COL].values

    n_participants = len(np.unique(groups))
    n_splits = min(5, n_participants)
    if n_splits < 2:
        return []
    if len(np.unique(y)) < 2:
        return []

    X_ref, _, _ = builder(df_s)
    if X_ref is None:
        return []
    n_features = X_ref.shape[1]

    results = []
    for train_idx, test_idx, fold_info in grouped_nested_cv(
            np.empty(len(y)), y, groups, outer_splits=n_splits):
        X_df, _, _ = builder(df_s, train_idx=train_idx)
        X = X_df.values.astype(float)

        pipe, grid = get_logreg_pipeline()

        n_inner = min(3, len(np.unique(groups[train_idx])))
        if n_inner < 2:
            pipe.set_params(lr__C=1.0)
            pipe.fit(X[train_idx], y[train_idx])
            best = pipe
        else:
            inner_cv = get_inner_cv(groups[train_idx], n_splits=n_inner)
            try:
                search = GridSearchCV(
                    pipe, grid, scoring="f1_macro",
                    cv=inner_cv, n_jobs=1, refit=True,
                )
                search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
                best = search.best_estimator_
            except ValueError:
                pipe.set_params(lr__C=1.0)
                pipe.fit(X[train_idx], y[train_idx])
                best = pipe

        y_pred = best.predict(X[test_idx])
        y_prob = best.predict_proba(X[test_idx])[:, 1]
        metrics = classification_metrics(y[test_idx], y_pred, y_prob)
        results.append({
            "study": study_id,
            "condition": cond_name,
            "n_features": n_features,
            **fold_info,
            **metrics,
        })

    return results


def main() -> None:
    set_seed(SEED)
    print("Loading and harmonizing data ...")
    df = harmonize_all(save=False)

    all_results = []

    for study_id in sorted(df["study_id"].unique()):
        df_s = df[df["study_id"] == study_id].copy().reset_index(drop=True)
        meta = STUDY_META.get(study_id, {})
        n_part = df_s[GROUP_COL].nunique()
        tut_rate = df_s[LABEL_COL].dropna().astype(float).mean()

        print(f"\n=== Study {study_id} ({meta.get('task_group', '?')}, "
              f"{len(df_s)} rows, {n_part} participants, "
              f"TUT rate={tut_rate:.3f}) ===")

        if n_part < 5:
            print(f"  Skipping: too few participants ({n_part})")
            continue

        # Gaze-only (available for all studies)
        gaze_results = _run_cv_for_study(
            df_s, _build_gaze_only, study_id, "gaze_only")
        all_results.extend(gaze_results)
        if gaze_results:
            mean_f1 = np.mean([r["macro_f1"] for r in gaze_results])
            print(f"  gaze_only:    F1={mean_f1:.4f}")

        # Emotion-only and Emotion+Gaze (only for affect-available studies)
        has_affect = len(meta.get("affect_available", [])) > 1
        if has_affect:
            emo_results = _run_cv_for_study(
                df_s, _build_emotion_only, study_id, "emotion_only")
            all_results.extend(emo_results)
            if emo_results:
                mean_f1 = np.mean([r["macro_f1"] for r in emo_results])
                print(f"  emotion_only: F1={mean_f1:.4f}")

            eg_results = _run_cv_for_study(
                df_s, _build_emotion_gaze, study_id, "emotion_gaze")
            all_results.extend(eg_results)
            if eg_results:
                mean_f1 = np.mean([r["macro_f1"] for r in eg_results])
                print(f"  emotion_gaze: F1={mean_f1:.4f}")

    out = pd.DataFrame(all_results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "P3_per_study_signal.csv", index=False)
    print(f"\nSaved {len(out)} rows to {TABLES_DIR / 'P3_per_study_signal.csv'}")

    # Summary table
    print("\n=== Per-study summary (mean macro-F1) ===")
    summary = out.groupby(["study", "condition"])["macro_f1"].agg(
        ["mean", "std"]).round(4)
    print(summary.to_string())

    # Compute incremental gains
    print("\n=== Incremental gains ===")
    studies = out["study"].unique()
    gain_rows = []
    for study in studies:
        sub = out[out["study"] == study]
        gaze_f1 = sub[sub["condition"] == "gaze_only"]["macro_f1"].mean()
        emo_f1 = sub[sub["condition"] == "emotion_only"]["macro_f1"].mean() \
            if "emotion_only" in sub["condition"].values else np.nan
        eg_f1 = sub[sub["condition"] == "emotion_gaze"]["macro_f1"].mean() \
            if "emotion_gaze" in sub["condition"].values else np.nan
        gain_rows.append({
            "study": study,
            "task": STUDY_META.get(study, {}).get("task_group", "?"),
            "gaze_only_f1": gaze_f1,
            "emotion_only_f1": emo_f1,
            "emotion_gaze_f1": eg_f1,
            "emo_over_gaze": emo_f1 - gaze_f1 if not np.isnan(emo_f1) else np.nan,
            "gaze_incr_over_emo": eg_f1 - emo_f1 if not (np.isnan(eg_f1) or np.isnan(emo_f1)) else np.nan,
        })
        row = gain_rows[-1]
        emo_gain = f"{row['emo_over_gaze']:+.4f}" if not np.isnan(row.get("emo_over_gaze", np.nan)) else "N/A"
        gaze_incr = f"{row['gaze_incr_over_emo']:+.4f}" if not np.isnan(row.get("gaze_incr_over_emo", np.nan)) else "N/A"
        print(f"  {study} ({row['task']}): gaze={gaze_f1:.4f}, "
              f"emo={emo_f1:.4f}, emo+gaze={eg_f1:.4f}, "
              f"emo_gain={emo_gain}, gaze_incr={gaze_incr}")

    # Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Grouped bar chart per study
    gain_df = pd.DataFrame(gain_rows).set_index("study")
    studies_plot = gain_df.index.tolist()
    x = np.arange(len(studies_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    conditions_to_plot = ["gaze_only_f1", "emotion_only_f1", "emotion_gaze_f1"]
    labels = ["Gaze only", "Emotion only", "Emotion + Gaze"]

    for i, (col, label) in enumerate(zip(conditions_to_plot, labels)):
        vals = gain_df[col].values
        mask = ~np.isnan(vals)
        positions = x[mask] + i * width
        ax.bar(positions, vals[mask], width, label=label, alpha=0.85)

    ax.set_xticks(x + width)
    task_labels = [f"{s}\n({gain_df.loc[s, 'task']})" for s in studies_plot]
    ax.set_xticklabels(task_labels, fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Per-Study Signal Decomposition: Where Does TUT Prediction Signal Reside?")
    ax.set_ylim(0.3, 0.85)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "P3_per_study_signal.png", dpi=200)
    plt.close(fig)

    # Figure 2: Incremental gain waterfall
    affect_studies = gain_df.dropna(subset=["emotion_only_f1"])
    if not affect_studies.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x2 = np.arange(len(affect_studies))
        w2 = 0.35

        ax.bar(x2 - w2 / 2, affect_studies["emo_over_gaze"].values, w2,
               label="Emotion over Gaze", alpha=0.85)
        ax.bar(x2 + w2 / 2, affect_studies["gaze_incr_over_emo"].values, w2,
               label="Gaze incremental over Emotion", alpha=0.85)

        ax.set_xticks(x2)
        task_labels2 = [f"{s}\n({affect_studies.loc[s, 'task']})"
                        for s in affect_studies.index]
        ax.set_xticklabels(task_labels2, fontsize=8)
        ax.set_ylabel("Delta Macro-F1")
        ax.set_title("Incremental Gains: Emotion and Gaze Contributions by Study")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "P3_incremental_gain.png", dpi=200)
        plt.close(fig)

    print(f"\nSaved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
