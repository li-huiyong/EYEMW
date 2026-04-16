"""Paper Analysis 6 (RQ4): Does gaze add incremental value beyond affect,
and if so, in which contexts?

Part A — Incremental gains with bootstrap CIs
  Four nested feature conditions under participant-grouped 5-fold CV:
    gaze_only, emotion_only, emotion_simple_gaze, emotion_rich
  Per-study breakdown and cluster-bootstrap CIs on the deltas
  (emotion_simple_gaze − emotion_only, emotion_rich − emotion_only).

Part B — Mixed-effects interaction model
  GEE logistic regression with valence_norm (the one affect variable common
  to all four affect studies), gaze proportions, TaskGroup dummies, and
  valence_norm × TaskGroup interactions.

Outputs:
  results/tables/P6_incremental_gains.csv
  results/tables/P6_per_study_gains.csv
  results/tables/P6_bootstrap_deltas.csv
  results/tables/P6_gee_interaction.csv
  results/figures/P6_incremental_gains.png
  results/figures/P6_per_study_delta.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_COL, GROUP_COL, ALL_CONSTRUCTS, STUDY_META,
    GAZE_PROPORTION_COLS, TABLES_DIR, FIGURES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.features import build_features_A
from src.models import get_logreg_pipeline
from src.evaluate import classification_metrics, bootstrap_ci
from src.utils import set_seed


N_FOLDS = 5
N_BOOT = 1000


# ── Feature builders ─────────────────────────────────────────────────────────

def _select_affect(df):
    has_any = False
    for c in ALL_CONSTRUCTS:
        if c != "TUT" and f"has_{c}" in df.columns:
            has_any = has_any | (df[f"has_{c}"] == 1)
    return df[has_any].copy()


def _gaze_only(df, train_idx=None):
    X_a, _, _ = build_features_A(df, train_idx=train_idx)
    gaze_cols = [c for c in X_a.columns
                 if not c.startswith(("WindowType_", "TaskGroup_"))]
    return X_a[gaze_cols].values.astype(float)


def _emotion_only(df, train_idx=None):
    norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS
                 if f"{c}_norm" in df.columns and c != "TUT"]
    X = df[norm_cols].apply(pd.to_numeric, errors="coerce")
    nan_frac = X.isna().mean()
    keep = nan_frac[nan_frac <= 0.9].index.tolist()
    return X[keep].values.astype(float)


def _emotion_simple_gaze(df, train_idx=None):
    emo = _emotion_only(df, train_idx=train_idx)
    simple = df[["UniqueGazeProportion", "OffScreenGazeProportion"]].apply(
        pd.to_numeric, errors="coerce"
    ).values.astype(float)
    return np.column_stack([emo, simple])


def _emotion_rich(df, train_idx=None):
    emo = _emotion_only(df, train_idx=train_idx)
    gaze = _gaze_only(df, train_idx=train_idx)
    return np.column_stack([emo, gaze])


CONDITIONS = {
    "gaze_only": _gaze_only,
    "emotion_only": _emotion_only,
    "emotion_simple_gaze": _emotion_simple_gaze,
    "emotion_rich": _emotion_rich,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_pipe():
    pipe, _ = get_logreg_pipeline()
    pipe.set_params(lr__C=1.0)
    return pipe


def _run_grouped_cv(df, y, groups, builders, n_folds=N_FOLDS):
    """Run grouped CV for each condition; return per-fold results DataFrame."""
    gkf = GroupKFold(n_splits=n_folds)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(
            gkf.split(np.zeros(len(y)), y, groups), start=1):
        for cond, builder in builders.items():
            X = builder(df, train_idx=train_idx)
            pipe = _make_pipe()
            pipe.fit(X[train_idx], y[train_idx])

            y_pred = pipe.predict(X[test_idx])
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]
            m = classification_metrics(y[test_idx], y_pred, y_prob)
            rows.append({"fold": fold_id, "condition": cond, **m})
    return pd.DataFrame(rows)


def _summarize(fold_df):
    return fold_df.groupby("condition").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
    ).round(4)


# ── Part A: Incremental gains ───────────────────────────────────────────────

def part_a(df_affect):
    y = df_affect[LABEL_COL].astype(int).values
    groups = df_affect[GROUP_COL].values

    # 1. Overall grouped CV
    print("\n=== Part A: 5-fold Grouped CV (overall) ===")
    fold_df = _run_grouped_cv(df_affect, y, groups, CONDITIONS)
    summary = _summarize(fold_df)
    print(summary.to_string())
    fold_df.to_csv(TABLES_DIR / "P6_incremental_gains.csv", index=False)

    # 2. Per-study breakdown
    print("\n=== Per-study breakdown ===")
    per_study_rows = []
    for sid in sorted(df_affect["study_id"].unique()):
        mask = df_affect["study_id"] == sid
        df_s = df_affect[mask].reset_index(drop=True)
        y_s = df_s[LABEL_COL].astype(int).values
        g_s = df_s[GROUP_COL].values
        n_groups = len(np.unique(g_s))
        nf = min(N_FOLDS, n_groups)
        if nf < 2:
            print(f"  {sid}: too few participants ({n_groups}), skipping")
            continue
        meta = STUDY_META.get(sid, {})
        sdf = _run_grouped_cv(df_s, y_s, g_s, CONDITIONS, n_folds=nf)
        for cond in CONDITIONS:
            sub = sdf[sdf["condition"] == cond]
            per_study_rows.append({
                "study_id": sid,
                "task_group": meta.get("task_group", ""),
                "condition": cond,
                "macro_f1": sub["macro_f1"].mean(),
                "auc": sub["auc"].mean(),
            })
        print(f"  {sid} ({meta.get('task_group', '')}): done "
              f"({len(df_s)} rows, {n_groups} participants)")

    per_study_df = pd.DataFrame(per_study_rows)
    per_study_df.to_csv(TABLES_DIR / "P6_per_study_gains.csv", index=False)

    # 3. Cluster-bootstrap CIs on incremental deltas
    print("\n=== Bootstrap CIs on incremental deltas ===")
    delta_rows = []
    for delta_name, cond_a, cond_b in [
        ("simple_minus_emo", "emotion_simple_gaze", "emotion_only"),
        ("rich_minus_emo", "emotion_rich", "emotion_only"),
    ]:
        f1_a = fold_df[fold_df["condition"] == cond_a]["macro_f1"].values
        f1_b = fold_df[fold_df["condition"] == cond_b]["macro_f1"].values
        diffs = f1_a - f1_b

        rng = np.random.default_rng(SEED)
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot[i] = np.mean(diffs[idx])
        lo = float(np.percentile(boot, 2.5))
        hi = float(np.percentile(boot, 97.5))
        row = {
            "delta": delta_name,
            "mean": float(np.mean(diffs)),
            "ci_lower": lo,
            "ci_upper": hi,
        }
        delta_rows.append(row)
        print(f"  {delta_name}: mean={row['mean']:.4f} "
              f"[{lo:.4f}, {hi:.4f}]")

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(TABLES_DIR / "P6_bootstrap_deltas.csv", index=False)

    return fold_df, per_study_df


# ── Part B: GEE interaction model ───────────────────────────────────────────

def part_b(df_affect):
    print("\n=== Part B: GEE Interaction Model ===")
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
    except ImportError:
        print("  statsmodels not installed; skipping GEE.")
        return

    tmp = df_affect.copy()
    tmp["valence_norm_f"] = pd.to_numeric(tmp["valence_norm"], errors="coerce")

    for col in GAZE_PROPORTION_COLS:
        tmp[col + "_f"] = pd.to_numeric(tmp[col], errors="coerce")

    tg_dummies = pd.get_dummies(tmp["TaskGroup"], prefix="TG", dtype=int)
    tg_ref = tg_dummies.columns[0]
    tg_dummies = tg_dummies.drop(columns=[tg_ref])
    tmp = pd.concat([tmp, tg_dummies], axis=1)

    for tg_col in tg_dummies.columns:
        tmp[f"val_x_{tg_col}"] = tmp["valence_norm_f"] * tmp[tg_col]

    pred_cols = (
        ["valence_norm_f"]
        + [c + "_f" for c in GAZE_PROPORTION_COLS]
        + list(tg_dummies.columns)
        + [f"val_x_{c}" for c in tg_dummies.columns]
    )

    tmp_clean = tmp.dropna(subset=pred_cols + [LABEL_COL, GROUP_COL])
    if len(tmp_clean) < 50:
        print("  Too few complete rows for GEE; skipping.")
        return

    y_gee = tmp_clean[LABEL_COL].astype(float)
    X_gee = sm.add_constant(tmp_clean[pred_cols].astype(float))
    groups_gee = tmp_clean[GROUP_COL]

    try:
        model = GEE(
            y_gee, X_gee, groups=groups_gee,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=100)
        print(result.summary())
        coef_df = pd.DataFrame({
            "coef": result.params,
            "std_err": result.bse,
            "z": result.tvalues,
            "p_value": result.pvalues,
        })
        coef_df.to_csv(TABLES_DIR / "P6_gee_interaction.csv")
        print(f"\n  Saved GEE coefficients to P6_gee_interaction.csv")
    except Exception as e:
        print(f"  GEE fitting failed: {e}")


# ── Figures ──────────────────────────────────────────────────────────────────

def _plot_overall(fold_df):
    summary = fold_df.groupby("condition")["macro_f1"].agg(["mean", "std"])
    order = ["gaze_only", "emotion_only", "emotion_simple_gaze", "emotion_rich"]
    summary = summary.loc[[c for c in order if c in summary.index]]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["std"], capsize=4,
           color=["#5b9bd5", "#ed7d31", "#a5a5a5", "#70ad47"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("RQ4: Incremental Feature Ablation (5-fold Grouped CV)")
    ax.set_ylim(0, 0.8)
    ax.axhline(0.5, color="grey", ls="--", alpha=0.4, lw=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "P6_incremental_gains.png", dpi=200)
    plt.close(fig)
    print(f"Saved figure to P6_incremental_gains.png")


def _plot_per_study_delta(per_study_df):
    if per_study_df.empty:
        return
    studies = sorted(per_study_df["study_id"].unique())
    conds = ["emotion_only", "emotion_simple_gaze", "emotion_rich"]
    x = np.arange(len(studies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(conds):
        vals = []
        for sid in studies:
            row = per_study_df[(per_study_df["study_id"] == sid) &
                               (per_study_df["condition"] == cond)]
            vals.append(row["macro_f1"].values[0] if len(row) else 0)
        ax.bar(x + i * width, vals, width, label=cond, alpha=0.85)

    labels = [f"{s}\n({STUDY_META.get(s, {}).get('task_group', '')})"
              for s in studies]
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("RQ4: Per-Study Incremental Gaze Value")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="grey", ls="--", alpha=0.4, lw=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "P6_per_study_delta.png", dpi=200)
    plt.close(fig)
    print(f"Saved figure to P6_per_study_delta.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    print("Loading and harmonizing data ...")
    df = harmonize_all(save=False)
    df_affect = _select_affect(df).reset_index(drop=True)
    print(f"Affect-available subset: {len(df_affect)} rows, "
          f"{df_affect[GROUP_COL].nunique()} participants")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fold_df, per_study_df = part_a(df_affect)
    part_b(df_affect)

    _plot_overall(fold_df)
    _plot_per_study_delta(per_study_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
