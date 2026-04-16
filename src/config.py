"""Central configuration: paths, column constants, and per-study metadata."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Study files ──────────────────────────────────────────────────────────────
STUDY_FILES = {
    "001-002": DATA_DIR / "study001-002.xlsx",
    "006":     DATA_DIR / "study006.xlsx",
    "010-011": DATA_DIR / "study010-011.xlsx",
    "016":     DATA_DIR / "study016.xlsx",
    "019":     DATA_DIR / "study019.xlsx",
}

# ── Column names (EYEMW full 109-column schema) ─────────────────────────────
# Gaze features (7 core columns present in all studies)
GAZE_COLS = [
    "Gazes",
    "UniqueGazes",
    "UniqueGazeProportion",
    "OffscreenGazes",
    "OffScreenGazeProportion",
    "AOIGazes",
    "AOIGazeProportion",
]

GAZE_PROPORTION_COLS = [
    "UniqueGazeProportion",
    "OffScreenGazeProportion",
    "AOIGazeProportion",
]

GAZE_COUNT_COLS = [
    "Gazes",
    "UniqueGazes",
    "OffscreenGazes",
    "AOIGazes",
]

# Affect response columns
AFFECT_RESPONSE_COLS = {
    "valence": "ValenceResponse",
    "arousal": "ArousalResponse",
    "boredom": "BoredomResponse",
    "disengagement": "DisengagementResponse",
}

LABEL_COL = "TUTProbeResponse"
GROUP_COL = "ParticipantNum"
STUDY_COL = "StudyNum"
PROBE_COL = "ProbeNum"

# Per-construct metadata columns follow the pattern: {Construct}{Suffix}
# e.g. ValenceScaleDirection, ValenceScaleMin, ValenceScaleMax,
#      ValenceResponseType, ValenceProbeType
CONSTRUCT_PREFIXES = [
    "TUT", "Valence", "Arousal", "Boredom", "Disengagement",
    "Intentionality", "Awareness", "FMT",
]
SCALE_SUFFIXES = ["ScaleDirection", "ScaleMin", "ScaleMax"]
META_SUFFIXES = ["ResponseType", "ProbeType"] + SCALE_SUFFIXES

# Context columns available in the dataset
TASK_TYPE_COL = "TaskType"
PERFORMANCE_BINARY_COL = "PerformanceBinary"
PERFORMANCE_DIR_COL = "PerformanceDirection"
TASK_PERFORMANCE_COL = "TaskPerformance"
EXP_SETTING_COL = "ExpSetting"
DEVICE_TYPE_COL = "DeviceType"

# ── Per-study metadata ───────────────────────────────────────────────────────
# Affect availability validated against 00_inspect_data.py output.
# Study 016 has continuous/aggregated TUT (0, 0.333, 0.667, 1.0); all others
# are binary. The harmonize layer binarises at threshold >= 0.5.

STUDY_META = {
    "001-002": {
        "task_group": "reading_listening",
        "window_type": "prior_sentence",
        "affect_available": ["TUT"],
        "tut_is_binary": True,
    },
    "006": {
        "task_group": "reading",
        "window_type": "paragraph_before_probe",
        "affect_available": ["TUT", "valence", "arousal"],
        "tut_is_binary": True,
    },
    "010-011": {
        "task_group": "video",
        "window_type": "fixed_20s",
        "affect_available": ["TUT", "valence", "arousal", "boredom", "disengagement"],
        "tut_is_binary": True,
    },
    "016": {
        "task_group": "problem_solving",
        "window_type": "averaged_over_problem",
        "affect_available": ["TUT", "valence", "disengagement"],
        "tut_is_binary": False,
    },
    "019": {
        "task_group": "math",
        "window_type": "preceding_problem",
        "affect_available": ["TUT", "valence"],
        "tut_is_binary": True,
    },
}

ALL_CONSTRUCTS = ["TUT", "valence", "arousal", "boredom", "disengagement"]

# ── CV defaults ──────────────────────────────────────────────────────────────
OUTER_SPLITS = 5
INNER_SPLITS = 3
SEED = 42
