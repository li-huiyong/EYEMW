# EYEMW

Analysis pipeline for the [EYEMW Dataset Competition](https://www.eyemindwander.com/competition/).

## Overview

This project separates two scientific questions that the EYEMW multi-study collection poses:

**Problem A -- Behavioral Detection.**
Can we predict task-unrelated thought (TUT) from webcam gaze features and task context alone, without contemporaneous self-report affect? How well does such a detector generalize across studies with different pre-probe window definitions?

**Problem B -- Affective-Cognitive Coupling.**
How do self-reported affect (valence, arousal, boredom, disengagement) relate to TUT? Does gaze add incremental predictive value beyond affect? Do lagged affective states predict future TUT episodes? Does the boredom-TUT coupling vary by study context?

## Studies Included

| Study | Task | Window | Affect |
|---|---|---|---|
| 001-002 | Reading/listening | Prior sentence | TUT only |
| 006 | Reading | Paragraph before probe | TUT, valence, arousal |
| 010-011 | Video | Fixed 20s | TUT, valence, arousal, boredom, disengagement |
| 016 | Problem-solving | Averaged over problem | TUT, valence, disengagement |
| 019 | Math | Preceding problem | TUT, valence |

## Quick Start

```bash
pip install -r requirements.txt

# 1. Inspect data schema
python scripts/00_inspect_data.py

# 2. Harmonize studies
python scripts/01_harmonize.py

# 3. Problem A: Behavioral detection
python scripts/A1_ablation_ladder.py
python scripts/A2_cross_context_eval.py
python scripts/A3_minimal_realtime.py
python scripts/A5_cross_context_affect.py

# 4. Problem B: Affective-cognitive coupling
python scripts/B1_incremental_value.py
python scripts/B2_mixed_effects.py
python scripts/B3_lagged_antecedent.py
python scripts/B5_shap_stability.py
```

See [PROJECT_CODE_STRUCTURE.md](PROJECT_CODE_STRUCTURE.md) for the full code map.

## Key Claims

1. Gaze-only TUT detection is weak under strict participant-independent evaluation in sentence-based reading/listening settings.
2. Affect-rich settings produce stronger TUT prediction; the jump comes mainly from affect, not model complexity.
3. EYEMW's heterogeneity (different pre-probe windows across studies) is the core scientific challenge, not nuisance.
4. Affective-cognitive coupling should be studied separately from behavioral detection.
5. The benchmark should include cross-study, cross-window, and antecedent-sensitive generalization -- not only within-study accuracy.
