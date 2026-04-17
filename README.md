# EYEMW

[EYEMW Dataset Competition](https://www.eyemindwander.com/competition/)

## Project structure

```
EYEMW/
├── analysis/          # Paper analyses (RQ1–RQ4)
├── data/              # Competition Excel files (not in git)
├── results/           # Generated tables and figures (not in git)
├── scripts/           # Harmonization and exploratory benchmarks
├── src/               # Harmonize, features, models, CV, evaluation
├── requirements.txt
└── README.md
```

| Path | Role |
|------|------|
| `src/` | Shared preprocessing, feature builders, models, cross-validation |
| `analysis/` | `RQ1_single_study_mlp.py`, `RQ2_signal_source_ablation.py`, `RQ3_incremental_gaze_value.py`, `RQ4_cross_context_generalization.py` |
| `scripts/` | Ordered analysis pipeline scripts (01 to 06) and runner |

## Get started

Create a virtual environment and install dependencies ([venv](https://docs.python.org/3/library/venv.html), [pip](https://packaging.python.org/)):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put the competition spreadsheets in `data/` using the filenames expected in `src/config.py` (`STUDY_FILES`).

From the repository root, build the harmonized table:

```bash
python scripts/01_prepare_harmonized_data.py
```

Run harmonization plus the exploratory `scripts/` pipeline in one step:

```bash
python scripts/00_run_analysis_pipeline.py
```

Or run steps individually in this order:
`01_prepare_harmonized_data.py` ->
`02_ablation_gaze_context_models.py` ->
`03_cross_study_generalization_gaze_context.py` ->
`04_cross_study_generalization_with_affect.py` ->
`05_incremental_value_gaze_affect.py` ->
`06_mixed_effects_affect_gaze.py`.
Outputs go under `results/tables/` and `results/figures/`.

Paper-style analyses (writes the same `results/` tree):

```bash
python analysis/RQ1_single_study_mlp.py
python analysis/RQ2_signal_source_ablation.py
python analysis/RQ3_incremental_gaze_value.py
python analysis/RQ4_cross_context_generalization.py
```

## Notes

- Run commands from the **repository root** so imports (`src`, `analysis`) resolve.
- `data/` and `results/` are **gitignored**; regenerate outputs locally after pulling.
- Analysis scripts may call `harmonize_all(save=False)` instead of relying on a saved `harmonized.csv`.
- Optional packages (e.g. `statsmodels` for GEE in `RQ3`) should match `requirements.txt` for full runs.
