"""Run the study-harmonization pipeline and print summary diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    LABEL_COL, GROUP_COL, GAZE_COLS, GAZE_PROPORTION_COLS,
    AFFECT_RESPONSE_COLS, ALL_CONSTRUCTS, TABLES_DIR,
)
from src.harmonize import harmonize_all


def main() -> None:
    print("Running harmonization pipeline ...")
    df = harmonize_all(save=True)
    print(f"Total rows: {len(df)}, Total participants: {df[GROUP_COL].nunique()}")

    # Per-study summary
    print("\n=== Per-study summary ===")
    for sid, grp in df.groupby("study_id", sort=False):
        n_part = grp[GROUP_COL].nunique()
        label_vc = grp[LABEL_COL].value_counts(dropna=False).to_dict()
        print(f"\nStudy {sid}: {len(grp)} rows, {n_part} participants")
        print(f"  TaskGroup={grp['TaskGroup'].iloc[0]}, WindowType={grp['WindowType'].iloc[0]}")
        print(f"  TUT label dist: {label_vc}")

        # Affect availability
        avail = [c for c in ALL_CONSTRUCTS if grp[f"has_{c}"].iloc[0] == 1]
        print(f"  Affect available: {avail}")

        # Harmonized affect stats
        for construct in ALL_CONSTRUCTS:
            norm_col = f"{construct}_norm"
            if norm_col in df.columns:
                vals = grp[norm_col].dropna()
                if len(vals) > 0:
                    print(f"    {construct}_norm: mean={vals.mean():.3f}, "
                          f"std={vals.std():.3f}, n={len(vals)}")

    # Gaze missingness summary
    print("\n=== Gaze column missingness by study ===")
    for col in GAZE_COLS:
        rates = df.groupby("study_id")[col].apply(lambda s: s.isna().mean() * 100)
        nonzero = rates[rates > 0]
        if len(nonzero) > 0:
            print(f"  {col}: " + ", ".join(
                f"{sid}={r:.0f}%" for sid, r in nonzero.items()
            ))

    out_path = TABLES_DIR / "harmonized.csv"
    print(f"\nSaved harmonized data ({len(df)} rows, {len(df.columns)} cols) to {out_path}")


if __name__ == "__main__":
    main()
