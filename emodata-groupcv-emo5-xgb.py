"""
emodata：3 情感 + AOIGazeProportion + OffScreenGazeProportion（5 维），XGBoost 二分类。
外层 / 内层 GroupKFold(groups=ParticipantNum)。Pipeline：Imputer → XGB（每次 fit 仅用当前训练 y
计算 scale_pos_weight = n_neg/n_pos，避免用全表统计泄漏）。

依赖：pip install xgboost
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("缺少依赖 xgboost，请先安装：pip install xgboost") from e

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


class ScalePosWeightedXGB(BaseEstimator, ClassifierMixin):
    """
    包装 XGBClassifier：在 fit 时根据传入的训练标签计算 scale_pos_weight，
    使每个 CV 折、每次 refit 仅使用该折训练子集的类频。
    """

    def __init__(
        self,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y_arr = np.asarray(y).astype(int)
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        spw = float(n_neg) / max(float(n_pos), 1.0)

        self.model_ = XGBClassifier(
            objective="binary:logistic",
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=spw,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            eval_metric="logloss",
        )
        self.model_.fit(X, y_arr)
        self.classes_ = getattr(self.model_, "classes_", np.array([0, 1]))
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


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
        f"模型: XGBClassifier（训练折内 scale_pos_weight）| 特征: 5 维 | "
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
            (
                "xgb",
                ScalePosWeightedXGB(
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    param_grid = {
        "xgb__max_depth": [4, 6],
        "xgb__learning_rate": [0.05, 0.1],
        "xgb__n_estimators": [200, 400],
        "xgb__min_child_weight": [1, 3],
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
    print(f"{OUTER_SPLITS}-Fold GroupCV Summary (emodata, 5 features + XGBoost)")
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
