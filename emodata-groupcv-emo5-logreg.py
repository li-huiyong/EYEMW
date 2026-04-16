"""
emodata：3 情感 + AOIGazeProportion + OffScreenGazeProportion（5 维），逻辑回归分类。
外层 / 内层均为 GroupKFold(groups=ParticipantNum)，保证同一受试者不同时出现在训练与测试。
Pipeline：Imputer（仅训练折 fit）→ StandardScaler（仅训练折 fit）→ LogisticRegression。
内层 GridSearchCV 以 f1_macro 选 C，外层测试折仅用于最终评估。
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

_REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = _REPO_ROOT / "data" / "emodata.xlsx"

FEATURE_COLS = [
    "ValenceResponse",
    "ArousalResponse",
    "BoredomResponse",
    "AOIGazeProportion",
    "OffScreenGazeProportion",
]
LABEL_COL = "TUTProbeResponse"
GROUP_COL = "ParticipantNum"

OUTER_SPLITS = 10
INNER_SPLITS = 5


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main():
    df = pd.read_excel(DATA_PATH)
    required = FEATURE_COLS + [LABEL_COL, GROUP_COL]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"缺少列: {c}")

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = df[LABEL_COL].astype(int)
    groups = df[GROUP_COL]

    n_participants = groups.nunique()
    print("数据:", DATA_PATH, "| 行数:", len(df), "| 被试数:", n_participants)
    print(
        f"模型: LogisticRegression(L2, lbfgs) | 特征: 5 维（3 情感 + 2 注视比例）| "
        f"外层 GroupKFold n_splits={OUTER_SPLITS}，内层 n_splits={INNER_SPLITS} | 内层评分: f1_macro"
    )

    if n_participants < OUTER_SPLITS:
        raise SystemExit(
            f"被试数 {n_participants} < 外层折数 {OUTER_SPLITS}，请减小 OUTER_SPLITS"
        )

    outer_cv = GroupKFold(n_splits=OUTER_SPLITS)
    inner_cv = GroupKFold(n_splits=INNER_SPLITS)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=10000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_grid = {
        "lr__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }

    fold_kappa = []
    fold_auc = []
    per_class_precision_0 = []
    per_class_recall_0 = []
    per_class_f1_0 = []
    per_class_precision_1 = []
    per_class_recall_1 = []
    per_class_f1_1 = []
    fold_macro_f1 = []

    for fold_id, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train, groups=groups_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        fold_kappa.append(kappa)
        fold_auc.append(auc)
        per_class_precision_0.append(report["0"]["precision"])
        per_class_recall_0.append(report["0"]["recall"])
        per_class_f1_0.append(report["0"]["f1-score"])
        per_class_precision_1.append(report["1"]["precision"])
        per_class_recall_1.append(report["1"]["recall"])
        per_class_f1_1.append(report["1"]["f1-score"])
        fold_macro_f1.append(macro_f1)

        print(f"\n================ Fold {fold_id} ================")
        print("Best Params:", search.best_params_)
        print("Best Inner-CV f1_macro:", round(search.best_score_, 4))
        print(
            f"Class 0 -> P:{report['0']['precision']:.4f} "
            f"R:{report['0']['recall']:.4f} F1:{report['0']['f1-score']:.4f}"
        )
        print(
            f"Class 1 -> P:{report['1']['precision']:.4f} "
            f"R:{report['1']['recall']:.4f} F1:{report['1']['f1-score']:.4f}"
        )
        print("Macro F1:", round(macro_f1, 4))
        print("Kappa:", round(kappa, 4))
        print("AUC:", round(auc, 4) if not np.isnan(auc) else "n/a")

    p0_m, p0_s = mean_std(per_class_precision_0)
    r0_m, r0_s = mean_std(per_class_recall_0)
    f0_m, f0_s = mean_std(per_class_f1_0)
    p1_m, p1_s = mean_std(per_class_precision_1)
    r1_m, r1_s = mean_std(per_class_recall_1)
    f1_m, f1_s = mean_std(per_class_f1_1)
    macro_f1_m, macro_f1_s = mean_std(fold_macro_f1)
    k_m, k_s = mean_std(fold_kappa)
    auc_m, auc_s = mean_std(fold_auc)

    print("\n===============================================")
    print(
        f"{OUTER_SPLITS}-Fold GroupCV Summary (emodata, 5 features + LogisticRegression)"
    )
    print("Class 0 Precision: {:.4f} ± {:.4f}".format(p0_m, p0_s))
    print("Class 0 Recall:    {:.4f} ± {:.4f}".format(r0_m, r0_s))
    print("Class 0 F1:        {:.4f} ± {:.4f}".format(f0_m, f0_s))
    print("Class 1 Precision: {:.4f} ± {:.4f}".format(p1_m, p1_s))
    print("Class 1 Recall:    {:.4f} ± {:.4f}".format(r1_m, r1_s))
    print("Class 1 F1:        {:.4f} ± {:.4f}".format(f1_m, f1_s))
    print("Macro F1:          {:.4f} ± {:.4f}".format(macro_f1_m, macro_f1_s))
    print("Kappa:             {:.4f} ± {:.4f}".format(k_m, k_s))
    print("AUC:               {:.4f} ± {:.4f}".format(auc_m, auc_s))


if __name__ == "__main__":
    main()
