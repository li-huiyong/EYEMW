# EYEMW Project Code Structure

## 1. Project Layout

```
EYEMW/
├── data/                            # Raw study xlsx files (gitignored)
│   ├── study001-002.xlsx            # Reading/listening, TUT only
│   ├── study006.xlsx                # Reading, TUT + valence + arousal
│   ├── study010-011.xlsx            # Video, TUT + all affect
│   ├── study016.xlsx                # Problem-solving, TUT + valence + disengagement
│   ├── study019.xlsx                # Math, TUT + valence
│   ├── eyedata.xlsx                 # Legacy (study001-002 predecessor)
│   └── emodata.xlsx                 # Legacy (study010-011 predecessor)
├── src/                             # Reusable library modules
│   ├── __init__.py
│   ├── config.py                    # Paths, column constants, per-study metadata
│   ├── harmonize.py                 # Study harmonization + scale normalization
│   ├── features.py                  # Two feature spaces (A: deploy, B: coupling)
│   ├── cv.py                        # CV strategies (grouped, LOSO, LOWO, LOTG)
│   ├── models.py                    # Model definitions (LR, SVM, RF, XGB)
│   ├── evaluate.py                  # Metrics, bootstrap CI
│   └── utils.py                     # Seed, mean_std helpers
├── scripts/                         # Runnable analysis scripts
│   ├── 00_inspect_data.py           # Schema validation
│   ├── 01_harmonize.py              # Harmonization pipeline runner
│   ├── A1_ablation_ladder.py        # Gaze/Context/Both ablation
│   ├── A2_cross_context_eval.py     # LOSO / LOWO / LOTG + base-rate diagnostic
│   ├── A3_minimal_realtime.py       # Minimal deployable model
│   ├── A5_cross_context_affect.py   # Cross-context with gaze+affect vs gaze-only
│   ├── B1_incremental_value.py      # Affect incremental value + per-study breakdown
│   ├── B2_mixed_effects.py          # GEE + boredom x study interaction model
│   ├── B3_lagged_antecedent.py      # Panel-data lagged antecedent analysis
│   └── B5_shap_stability.py         # SHAP rank + sign stability
├── results/
│   ├── tables/                      # CSV outputs
│   └── figures/                     # PNG plots
├── analysis_paper/                  # Paper-focused analyses (signal source + complexity)
│   ├── P1_signal_source_ablation.py # Gaze vs Emotion vs Combined ablation
│   ├── P2_model_complexity_ceiling.py # 5 models on same features
│   ├── P3_per_study_signal.py       # Per-study signal decomposition
│   └── README.md                    # Scientific rationale and how to run
├── old/                             # Archived original scripts
│   ├── eyedata-groupcv.py
│   ├── emodata-groupcv-emo5-*.py
│   ├── *-split715.py
│   └── image/                       # Old visualization scripts + outputs
├── requirements.txt
├── README.md
└── PROJECT_CODE_STRUCTURE.md
```

---

## 2. Library Modules (`src/`)

### `src/config.py`
Central source of truth for paths, column constants, and per-study metadata.

| Study | TaskGroup | WindowType | Affect Available |
|---|---|---|---|
| 001-002 | reading_listening | prior_sentence | TUT only |
| 006 | reading | paragraph_before_probe | TUT, valence, arousal |
| 010-011 | video | fixed_20s | TUT, valence, arousal, boredom, disengagement |
| 016 | problem_solving | averaged_over_problem | TUT, valence, disengagement |
| 019 | math | preceding_problem | TUT, valence |

### `src/harmonize.py`
`harmonize_all()` loads all 5 study files, concatenates, and adds:
- **WindowType** and **TaskGroup** columns
- **ProbeAvailabilityMask**: `has_TUT`, `has_valence`, `has_arousal`, `has_boredom`, `has_disengagement`
- **Scale-harmonized affect**: reverse-codes by ScaleDirection, min-max normalizes to [0,1] (`*_norm`), and adds within-study median-split binarized versions (`*_bin`)
- **Binary TUT**: thresholds continuous TUT (study 016) at 0.5

### `src/features.py`
Two feature spaces:

- **Feature Space A** (deployment): gaze proportions, derived rates, participant-centered z-scores, study-normalized quantiles, missingness indicators, temporal variability (within-participant SD), gaze entropy (3-category Shannon entropy), gaze x window interaction terms, and context one-hot (WindowType + TaskGroup). No contemporaneous affect.
- **Feature Space B** (coupling): everything in A + harmonized current affect (`*_norm`) + lagged affect/TUT from ProbeNum ordering.

### `src/cv.py`
Cross-validation generators yielding `(train_idx, test_idx, fold_info)`:
- `grouped_nested_cv` -- participant-grouped 5x3 nested CV
- `leave_one_study_out` -- LOSO
- `leave_one_window_out` -- LOWO
- `leave_one_taskgroup_out` -- LOTG

### `src/models.py`
Model pipelines returning `(pipeline, param_grid)`:
- `get_svm_pipeline()` -- Imputer + Scaler + SVC(RBF, balanced)
- `get_logreg_pipeline()` -- Imputer + Scaler + LogReg(L2 via l1_ratio=0, saga)
- `get_rf_pipeline()` -- Imputer + RF(balanced)
- `get_xgb_pipeline()` -- Imputer + ScalePosWeightedXGB
- `get_mlp_pipeline()` -- Imputer + Scaler + shallow MLP (sklearn, 16-8 hidden units)
- `get_minimal_pipeline()` -- Imputer + Scaler + shallow LogReg

### `src/evaluate.py`
- `classification_metrics()` -- macro-F1, per-class P/R/F1, kappa, AUC
- `brier_score()` -- Brier score
- `bootstrap_ci()` -- bootstrap confidence intervals for any metric
- `generalization_gap()` -- within vs held-out performance comparison

---

## 3. Analysis Scripts (`scripts/`)

### Problem A: Behavioral Detection (gaze + context, no affect)

| Script | Analysis | Output |
|---|---|---|
| `A1_ablation_ladder.py` | 3 conditions (gaze/context/both); context-only uses single LogReg (no grid search); others use 4 models with grouped nested CV | `A1_ablation.csv` |
| `A2_cross_context_eval.py` | 4 CV strategies (grouped, LOSO, LOWO, LOTG) with LogReg; TUT base-rate diagnostic with Spearman correlation | `A2_cross_context.csv`, `A2_baserate_diagnostic.csv`, `A2_generalization_gap.png`, `A2_baserate_vs_performance.png` |
| `A3_minimal_realtime.py` | Top-6 stable features + shallow LogReg | `A3_minimal_model.csv` |
| `A5_cross_context_affect.py` | LOSO/LOWO/LOTG with gaze-only vs gaze+affect (on affect-available studies), LogReg | `A5_cross_context_affect.csv`, `A5_generalization_gap_affect.png` |

### Problem B: Affective-Cognitive Coupling (gaze + affect + context)

| Script | Analysis | Output |
|---|---|---|
| `B1_incremental_value.py` | 4 conditions (gaze/affect/both/all) with RF + per-study breakdown | `B1_incremental_value.csv`, `B1_per_study_incremental.csv`, `B1_delta_auc.png` |
| `B2_mixed_effects.py` | GEE with participant clustering + boredom x study interaction model | `B2_mixed_effects.csv`, `B2_boredom_by_study.csv` |
| `B3_lagged_antecedent.py` | Affect-available studies only; fixed-effects logistic with participant dummies (panel model) + RF CV | `B3_lagged_coefficients.csv`, `B3_lagged_performance.csv` |
| `B5_shap_stability.py` | Bootstrap SHAP rank + per-study sign consistency | `B5_shap_stability.csv`, `B5_shap_rank_stability.png`, `B5_shap_sign_consistency.png` |

### Paper-Focused Analyses (`analysis_paper/`)

| Script | Scientific Question | Output |
|---|---|---|
| `P1_signal_source_ablation.py` | Where does the TUT prediction signal reside? Compares gaze-only, emotion-only, gaze-ratio-only, emotion+gaze, and full B under participant-independent CV with bootstrap CIs and pairwise deltas. | `P1_signal_source.csv`, `P1_signal_deltas.csv`, `P1_signal_source.png` |
| `P2_model_complexity_ceiling.py` | Does model class matter once the right features are present? Runs 6 models (LR, SVM, RF, XGB, MLP, baseline LR) on both the compact and full feature sets with Wilcoxon pairwise tests. | `P2_complexity_compact.csv`, `P2_complexity_fullB.csv`, `P2_pairwise_tests.csv`, `P2_model_complexity.png` |
| `P3_per_study_signal.py` | Is the emotion gain uniform or study-specific? Runs gaze-only / emotion-only / emotion+gaze per study to identify which contexts drive the aggregate finding. | `P3_per_study_signal.csv`, `P3_per_study_signal.png`, `P3_incremental_gain.png` |

---

## 4. Method Index

### By analysis goal

- **TUT detection (deployment)**: A1, A2, A3
- **Affect-TUT coupling**: B1, B2, B3
- **Feature explanation stability**: B5
- **Cross-context generalization**: A2 (LOSO/LOWO/LOTG), A5 (with affect)
- **Antecedent analysis**: B3 (lagged variables, panel model)
- **Study-varying coupling**: B2 (boredom x study interaction)
- **Signal source and model ceiling (paper)**: P1, P2, P3

### By model type

- **SVM**: A1
- **Logistic Regression**: A1, A2, A3, A5, B3 (fixed-effects panel)
- **Random Forest**: A1, B1, B3, B5
- **XGBoost**: A1
- **GEE (mixed-effects)**: B2

---

## 5. Dependencies

See `requirements.txt`. Key packages:

| Package | Used by |
|---|---|
| numpy, pandas, openpyxl | All |
| scikit-learn | All modeling scripts |
| matplotlib | All figure-producing scripts |
| scipy | A2 (Spearman correlation) |
| xgboost | A1 (XGB ablation) |
| shap | B5 (SHAP stability) |
| statsmodels | B2 (GEE) |

---

## 6. Execution Order

```bash
python scripts/00_inspect_data.py           # Verify schema
python scripts/01_harmonize.py              # Build harmonized dataset

# Problem A (independent, any order)
python scripts/A1_ablation_ladder.py
python scripts/A2_cross_context_eval.py
python scripts/A3_minimal_realtime.py
python scripts/A5_cross_context_affect.py

# Problem B (independent, any order)
python scripts/B1_incremental_value.py
python scripts/B2_mixed_effects.py
python scripts/B3_lagged_antecedent.py
python scripts/B5_shap_stability.py

# Paper analyses (independent, any order)
python analysis_paper/P1_signal_source_ablation.py
python analysis_paper/P2_model_complexity_ceiling.py
python analysis_paper/P3_per_study_signal.py
```

All scripts run from the repository root. A/B series and paper analyses are independent after `01_harmonize.py`.
