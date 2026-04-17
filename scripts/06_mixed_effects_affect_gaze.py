"""Problem B2: Hierarchical mixed-effects logistic regression.

Model 1 (main): Fixed effects for harmonized affect + gaze proportions,
  participant-clustered GEE with exchangeable correlation.
Model 2 (interaction): Adds boredom_norm * study_id interaction terms to
  test whether boredom's effect on TUT varies across studies.

Outputs: results/tables/B2_mixed_effects.csv
         results/tables/B2_boredom_by_study.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_COL, GROUP_COL, GAZE_PROPORTION_COLS,
    ALL_CONSTRUCTS, TABLES_DIR, SEED,
)
from src.harmonize import harmonize_all
from src.utils import set_seed


def main() -> None:
    set_seed(SEED)
    df = harmonize_all(save=False)

    # Keep only studies with at least one affect variable
    has_affect = False
    for c in ALL_CONSTRUCTS:
        if c != "TUT" and f"has_{c}" in df.columns:
            has_affect = has_affect | (df[f"has_{c}"] == 1)
    df_sub = df[has_affect].copy()

    print(f"Using {len(df_sub)} rows from affect-available studies")

    # Build the model matrix
    gaze_cols = [c for c in GAZE_PROPORTION_COLS if c in df_sub.columns and df_sub[c].notna().sum() > 10]
    affect_norm_cols = [f"{c}_norm" for c in ALL_CONSTRUCTS
                        if f"{c}_norm" in df_sub.columns and df_sub[f"{c}_norm"].notna().sum() > 10]

    predictors = gaze_cols + affect_norm_cols
    print(f"Fixed effects ({len(predictors)}): {predictors}")

    # Prepare clean data
    model_df = df_sub[predictors + [LABEL_COL, GROUP_COL, "study_id"]].dropna(subset=[LABEL_COL])
    for col in predictors:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=predictors)

    y = model_df[LABEL_COL].astype(int).values
    if len(np.unique(y)) < 2:
        print("Only one class present. Aborting.")
        return

    print(f"Clean rows for model: {len(model_df)}")

    # Fit mixed-effects logistic using statsmodels
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
    except ImportError:
        print("statsmodels not installed. Falling back to basic logistic regression.")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        X = model_df[predictors].values
        X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
        lr = LogisticRegression(solver="saga", l1_ratio=0, class_weight="balanced", max_iter=5000, random_state=SEED)
        lr.fit(X, y)
        coef_df = pd.DataFrame({
            "predictor": predictors,
            "coefficient": lr.coef_[0],
        }).sort_values("coefficient", ascending=False, key=abs)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        coef_df.to_csv(TABLES_DIR / "B2_mixed_effects.csv", index=False)
        print(coef_df.to_string(index=False))
        return

    # GEE with participant as cluster (approximates mixed effects)
    X_df = model_df[predictors].copy()
    X_df = (X_df - X_df.mean()) / X_df.std().replace(0, 1)
    X_df = sm.add_constant(X_df)

    groups_gee = model_df[GROUP_COL].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = GEE(
                y, X_df, groups=groups_gee,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            result = model.fit(maxiter=100)
            print(result.summary())

            coef_df = pd.DataFrame({
                "predictor": result.params.index,
                "coefficient": result.params.values,
                "std_err": result.bse.values,
                "z": result.tvalues.values,
                "p_value": result.pvalues.values,
            })
        except Exception as e:
            print(f"GEE failed ({e}), using basic logistic as fallback")
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            X = StandardScaler().fit_transform(model_df[predictors].values)
            lr = LogisticRegression(solver="saga", l1_ratio=0, class_weight="balanced", max_iter=5000, random_state=SEED)
            lr.fit(X, y)
            coef_df = pd.DataFrame({
                "predictor": ["const"] + predictors,
                "coefficient": np.concatenate([[lr.intercept_[0]], lr.coef_[0]]),
            })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "B2_mixed_effects.csv"
    coef_df.to_csv(out_path, index=False)
    print(f"\nSaved coefficients to {out_path}")
    print(coef_df.to_string(index=False))

    # --- Model 2: Boredom x Study interaction ---
    print("\n=== Boredom x Study interaction model ===")
    if "boredom_norm" not in predictors:
        print("boredom_norm not in predictors; skipping interaction model.")
        return

    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
    except ImportError:
        print("statsmodels not installed; skipping interaction model.")
        return

    X_int = model_df[predictors].copy()
    X_int = (X_int - X_int.mean()) / X_int.std().replace(0, 1)

    study_dummies = pd.get_dummies(model_df["study_id"], prefix="study", dtype=int)
    ref_study = study_dummies.columns[0]
    study_dummies = study_dummies.drop(columns=[ref_study])

    for col in study_dummies.columns:
        X_int[f"boredom_x_{col}"] = X_int["boredom_norm"] * study_dummies[col].values

    X_int = sm.add_constant(X_int)
    groups_gee = model_df[GROUP_COL].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model2 = GEE(
                y, X_int, groups=groups_gee,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            result2 = model2.fit(maxiter=100)

            interaction_cols = [c for c in result2.params.index if c.startswith("boredom_x_")]
            int_df = pd.DataFrame({
                "term": interaction_cols,
                "coefficient": result2.params[interaction_cols].values,
                "std_err": result2.bse[interaction_cols].values,
                "z": result2.tvalues[interaction_cols].values,
                "p_value": result2.pvalues[interaction_cols].values,
            })

            boredom_main = result2.params.get("boredom_norm", np.nan)
            int_df = pd.concat([
                pd.DataFrame([{
                    "term": f"boredom_norm (ref={ref_study})",
                    "coefficient": boredom_main,
                    "std_err": result2.bse.get("boredom_norm", np.nan),
                    "z": result2.tvalues.get("boredom_norm", np.nan),
                    "p_value": result2.pvalues.get("boredom_norm", np.nan),
                }]),
                int_df,
            ], ignore_index=True)

            int_path = TABLES_DIR / "B2_boredom_by_study.csv"
            int_df.to_csv(int_path, index=False)
            print(int_df.to_string(index=False))
            print(f"\nSaved to {int_path}")

            any_sig = (int_df["p_value"].iloc[1:] < 0.05).any()
            print(f"\nAny significant boredom x study interaction (p<0.05): {any_sig}")

        except Exception as e:
            print(f"Interaction model failed: {e}")


if __name__ == "__main__":
    main()
