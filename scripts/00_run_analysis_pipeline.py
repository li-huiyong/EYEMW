"""Run the main scripts pipeline in dependency order (from repo root).

Order:
  1. 01_prepare_harmonized_data.py
  2. 02_ablation_gaze_context_models.py
  3. 03_cross_study_generalization_gaze_context.py
  4. 04_cross_study_generalization_with_affect.py
  5. 05_incremental_value_gaze_affect.py
  6. 06_mixed_effects_affect_gaze.py

Paper-facing scripts live under analysis/; run them separately if needed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "scripts/01_prepare_harmonized_data.py",
    "scripts/02_ablation_gaze_context_models.py",
    "scripts/03_cross_study_generalization_gaze_context.py",
    "scripts/04_cross_study_generalization_with_affect.py",
    "scripts/05_incremental_value_gaze_affect.py",
    "scripts/06_mixed_effects_affect_gaze.py",
]


def main() -> None:
    for rel in SCRIPTS:
        path = ROOT / rel
        print(f"\n{'=' * 60}\n>>> python {rel}\n{'=' * 60}", flush=True)
        rc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            check=False,
        ).returncode
        if rc != 0:
            print(f"\nFAILED: {rel} (exit {rc})", flush=True)
            sys.exit(rc)
    print("\nAll scripts finished successfully.", flush=True)


if __name__ == "__main__":
    main()
