# Paper-Focused Analysis: EYEMW TUT Prediction

This folder contains six analysis scripts that produce the core empirical
findings for the EYEMW competition paper. They address four research questions:

1. **RQ1: Where does the TUT prediction signal reside?** (Feature signal — P1)
2. **RQ2: Does model complexity help once the right features are present?** (Model ceiling — P2, P4)
3. **RQ3: Does emotion-aware TUT prediction generalize across learning contexts?** (Cross-context — P5)
4. **RQ4: Does gaze add incremental value beyond affect, and in which contexts?** (Mechanism — P6)

## Scientific Contribution

The central claim is that **emotion changes the TUT detection problem
qualitatively**, and that the main bottleneck for EYEMW-based mind-wandering
prediction is feature signal and dataset structure, not model sophistication.

### Claim 1: Emotion is the primary signal source

Moving from gaze-only to emotion + gaze-ratio features produces the largest
performance jump observed in the entire analysis. The P1 analysis isolates this
by comparing gaze-only, emotion-only, gaze-ratio-only, and the combined set
under identical participant-independent cross-validation with bootstrap CIs.
If emotion-only already captures most of the gain, the contribution is a clear
statement about where the available EYEMW signal resides.

### Claim 2: Model complexity helps only weakly

LogReg, SVM, RF, XGBoost, and a baseline LR are compared on the same compact
feature set. If the model range (best minus worst) is small relative to the
feature ablation range, the conclusion is: *the main bottleneck is feature
signal and dataset structure, with only weak residual nonlinearity.*

### Claim 3: The gain is not uniform across studies

Per-study decomposition reveals which study contexts drive the aggregate
finding and whether gaze adds incremental value everywhere or only in specific
window/task configurations.

### Claim 4: Emotion-aware prediction generalizes across contexts (RQ3)

Leave-one-study-out evaluation shows whether affect's predictive contribution
persists when tested on unseen learning contexts (task types, window
definitions).  The generalization gap (within-study minus LOSO performance)
quantifies how much context-dependence remains.

### Claim 5: Gaze provides complementary information in specific contexts (RQ4)

Nested ablations (gaze-only → emotion-only → emotion+simple gaze →
emotion+rich) with bootstrap CIs on incremental deltas test whether gaze is
redundant once affect is known, or complements it.  Per-study breakdowns and a
GEE interaction model with `valence_norm × TaskGroup` reveal where gaze
matters most.

## Scripts

| Script | RQ | Question | Key Output |
|---|---|---|---|
| `P1_signal_source_ablation.py` | RQ1 | Where does the signal reside? | `P1_signal_source.csv`, `P1_signal_deltas.csv`, `P1_signal_source.png` |
| `P2_model_complexity_ceiling.py` | RQ2 | Does model class matter? | `P2_complexity_compact.csv`, `P2_pairwise_tests.csv`, `P2_model_complexity.png` |
| `P3_per_study_signal.py` | RQ1 | Which studies drive the finding? | `P3_per_study_signal.csv`, `P3_per_study_signal.png`, `P3_incremental_gain.png` |
| `P4_single_study_mlp.py` | RQ2 | MLP on homogeneous 010-011 data | `P4_single_study_raw.csv`, `P4_single_study_norm.csv`, `P4_single_study.png` |
| `P5_cross_context_generalization.py` | RQ3 | Does emotion generalize across contexts? | `P5_cross_context.csv`, `P5_generalization_gap.csv`, `P5_cross_context.png` |
| `P6_incremental_gaze_value.py` | RQ4 | Does gaze add value beyond affect? | `P6_incremental_gains.csv`, `P6_per_study_gains.csv`, `P6_bootstrap_deltas.csv`, `P6_gee_interaction.csv` |

## How to Run

All scripts should be run from the repository root:

```bash
python analysis_paper/P1_signal_source_ablation.py
python analysis_paper/P2_model_complexity_ceiling.py
python analysis_paper/P3_per_study_signal.py
python analysis_paper/P4_single_study_mlp.py
python analysis_paper/P5_cross_context_generalization.py
python analysis_paper/P6_incremental_gaze_value.py
```

Prerequisites: run `python scripts/01_harmonize.py` first (or let the scripts
auto-harmonize on first call).

Results are written to `results/tables/` and `results/figures/`.

## Feature Sets Used

- **gaze_only**: All gaze features from Feature Space A (proportions, derived
  rates, z-scores, quantiles, SD, entropy, interactions), excluding context
  one-hot columns.
- **gaze_ratio_only**: `UniqueGazeProportion` and `OffScreenGazeProportion`
  only (the two most robust gaze features identified across studies).
- **emotion_only**: Harmonized normalized affect features (`valence_norm`,
  `arousal_norm`, `boredom_norm`, `disengagement_norm`), excluding TUT.
- **emotion_gaze**: Emotion + the 2 gaze-ratio features (compact multimodal).
- **full_B**: Full Feature Space B (all gaze + all affect + context, no lags).

## Relationship to Existing Analyses

These scripts complement the existing `scripts/` pipeline:
- P1 extends B1 (`B1_incremental_value.py`) by adding the missing
  emotion-only ablation and bootstrap CIs for pairwise deltas.
- P2 extends A1 (`A1_ablation_ladder.py`) by holding the feature set constant
  and varying only the model class on the affect subset.
- P3 extends the B1 per-study breakdown by adding the emotion-only condition
  and computing per-study incremental gains explicitly.
- P4 bridges the old single-study pipeline with the new harmonized approach
  by comparing raw vs normalized features on study 010-011.
- P5 extends A2/A5 LOSO evaluation with a clean 3-condition design
  (gaze/emotion/emotion+gaze) plus generalization gap reporting.
- P6 extends B1 per-study incremental analysis with cluster-bootstrap CIs
  on incremental deltas and a GEE interaction model testing whether gaze's
  contribution varies by task type.
