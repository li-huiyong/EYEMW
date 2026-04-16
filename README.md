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
| `scripts/` | `01_harmonize.py` and other utilities; see each file’s docstring |

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
python scripts/01_harmonize.py
```

Run the analyses (writes under `results/tables/` and `results/figures/`):

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
