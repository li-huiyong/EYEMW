"""
eyedata：按被试固定三分 train:val:test = 7:1.5:1.5（被试人数约 70% / 15% / 15%），
同一被试只落在一个子集。预处理（imputer、scaler）仅在训练行上 fit；
在验证集上网格选参，再在 train+val 上重训，最后在测试集上报告一次指标。
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

_REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = _REPO_ROOT / "data" / "eyedata.xlsx"

FEATURE_COLS = [
    "Gazes",
    "UniqueGazes",
    "UniqueGazeProportion",
    "OffscreenGazes",
    "OffScreenGazeProportion",
    "AOIGazes",
    "AOIGazeProportion",
]
LABEL_COL = "TUTProbeResponse"
GROUP_COL = "ParticipantNum"

# 被试级分层标签：该被试样本中 TUT 的众数（用于划分时近似保持类别比例）
def participant_strat_labels(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby(GROUP_COL)[LABEL_COL]
        .agg(lambda s: int(s.mode(dropna=True).iloc[0]) if len(s.mode(dropna=True)) else int(s.iloc[0]))
    )


def split_participants_715(pids: np.ndarray, y_strat: np.ndarray, random_state: int = 42):
    """按 7:1.5:1.5 划分被试；stratify 不可行时退回无分层。"""
    test_size = 0.15 / (0.7 + 0.15 + 0.15)  # 0.15
    val_of_trainval = 0.15 / (0.7 + 0.15)  # 15% / 85%

    def _split(a, b, ts, stratify):
        try:
            return train_test_split(
                a, b, test_size=ts, stratify=stratify, random_state=random_state
            )
        except ValueError:
            return train_test_split(
                a, b, test_size=ts, stratify=None, random_state=random_state
            )

    p_tv, p_te, y_tv, y_te = _split(pids, y_strat, test_size, y_strat)
    p_tr, p_va, _, _ = _split(p_tv, y_tv, val_of_trainval, y_tv)
    return p_tr, p_va, p_te


def main():
    df = pd.read_excel(DATA_PATH)
    required = FEATURE_COLS + [LABEL_COL, GROUP_COL]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"缺少列: {c}")

    pids = df[GROUP_COL].unique()
    strat = participant_strat_labels(df)
    y_strat = np.array([strat[pid] for pid in pids])

    p_tr, p_va, p_te = split_participants_715(pids, y_strat, random_state=42)
    set_tr, set_va, set_te = set(p_tr), set(p_va), set(p_te)

    train_df = df[df[GROUP_COL].isin(set_tr)]
    val_df = df[df[GROUP_COL].isin(set_va)]
    test_df = df[df[GROUP_COL].isin(set_te)]

    print("数据:", DATA_PATH)
    print(
        "被试数 — train:", len(p_tr), "| val:", len(p_va), "| test:", len(p_te),
        "| 合计:", len(pids),
    )
    print(
        "行数 — train:", len(train_df), "| val:", len(val_df), "| test:", len(test_df),
    )
    for name, d in ("train", train_df), ("val", val_df), ("test", test_df):
        vc = d[LABEL_COL].value_counts().sort_index()
        print(f"  {name} TUTProbeResponse:\n{vc.to_string()}")

    X_train = train_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_train = train_df[LABEL_COL].astype(int).values
    X_val = val_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_val = val_df[LABEL_COL].astype(int).values
    X_test = test_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_test = test_df[LABEL_COL].astype(int).values

    param_grid = {
        "svm__C": [1, 10],
        "svm__gamma": ["scale", 0.1],
    }

    best_f1 = -1.0
    best_params = None
    for C in param_grid["svm__C"]:
        for gamma in param_grid["svm__gamma"]:
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "svm",
                        SVC(
                            kernel="rbf",
                            C=C,
                            gamma=gamma,
                            class_weight="balanced",
                            probability=True,
                            random_state=42,
                        ),
                    ),
                ]
            )
            pipe.fit(X_train, y_train)
            y_pred_v = pipe.predict(X_val)
            f1 = f1_score(y_val, y_pred_v, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_params = {"C": C, "gamma": gamma}

    print("\n验证集 macro-F1 最优:", best_f1, "| 参数:", best_params)

    trainval_df = pd.concat([train_df, val_df], axis=0)
    X_tv = trainval_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_tv = trainval_df[LABEL_COL].astype(int).values

    final_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=best_params["C"],
                    gamma=best_params["gamma"],
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    final_pipe.fit(X_tv, y_tv)

    y_pred = final_pipe.predict(X_test)
    proba = final_pipe.predict_proba(X_test)[:, 1]
    print("\n=== 测试集（仅评一次）===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Cohen kappa:", round(cohen_kappa_score(y_test, y_pred), 4))
    try:
        print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    except ValueError:
        print("ROC-AUC: n/a（单类标签）")


if __name__ == "__main__":
    main()
