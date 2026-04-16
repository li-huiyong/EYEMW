"""
emodata：3 维情感（Valence/Arousal/Boredom）+ AOI 比例 + OffScreen 比例（共 5 维），
使用 PyTorch MLP 进行走神(1)/不走神(0)二分类。

评估策略：
- 外层 GroupKFold（按 ParticipantNum 分组，防止被试泄漏）
- 每个外层训练折中再做一次 GroupShuffleSplit 得到验证集（用于早停）
- 预处理（缺失值填补/标准化）仅在训练子集 fit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
VAL_SIZE_IN_TRAIN = 0.2
SEED = 42

BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class FoldResult:
    macro_f1: float
    kappa: float
    auc: float
    p0: float
    r0: float
    f10: float
    p1: float
    r1: float
    f11: float


def mean_std(values: list[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    y_pred = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y, y_pred)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")
    return {
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "auc": float(auc),
    }


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> nn.Module:
    model = MLPBinaryClassifier(input_dim=x_train.shape[1]).to(device)
    train_loader = make_loader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

    pos_count = float(np.sum(y_train == 1))
    neg_count = float(np.sum(y_train == 0))
    if pos_count > 0:
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val_f1 = -1.0
    no_improve = 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, x_val, y_val, device=device)
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_excel(DATA_PATH)
    required = FEATURE_COLS + [LABEL_COL, GROUP_COL]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"缺少列: {col}")

    x_all = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_all = df[LABEL_COL].astype(int).to_numpy()
    groups_all = df[GROUP_COL].to_numpy()

    n_participants = df[GROUP_COL].nunique()
    print(f"数据: {DATA_PATH} | 行数: {len(df)} | 被试数: {n_participants}")
    print(f"特征: 5 维（3 情感 + 2 比例）| 外层 GroupKFold n_splits={OUTER_SPLITS}")

    if n_participants < OUTER_SPLITS:
        raise SystemExit(f"被试数 {n_participants} < 外层折数 {OUTER_SPLITS}，请减小 OUTER_SPLITS")

    outer_cv = GroupKFold(n_splits=OUTER_SPLITS)
    fold_results: list[FoldResult] = []

    for fold_id, (train_idx, test_idx) in enumerate(
        outer_cv.split(x_all, y_all, groups_all), start=1
    ):
        x_train_full = x_all.iloc[train_idx]
        y_train_full = y_all[train_idx]
        g_train_full = groups_all[train_idx]

        x_test = x_all.iloc[test_idx]
        y_test = y_all[test_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE_IN_TRAIN, random_state=SEED + fold_id)
        tr_sub_idx, val_sub_idx = next(gss.split(x_train_full, y_train_full, g_train_full))

        x_train_df = x_train_full.iloc[tr_sub_idx]
        y_train = y_train_full[tr_sub_idx]
        x_val_df = x_train_full.iloc[val_sub_idx]
        y_val = y_train_full[val_sub_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        x_train = scaler.fit_transform(imputer.fit_transform(x_train_df))
        x_val = scaler.transform(imputer.transform(x_val_df))
        x_test_np = scaler.transform(imputer.transform(x_test))

        model = train_one_fold(x_train, y_train, x_val, y_val, device=device)

        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(x_test_np, dtype=torch.float32, device=device))
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_probs >= 0.5).astype(int)

        report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, test_pred)
        try:
            auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auc = float("nan")

        fold_results.append(
            FoldResult(
                macro_f1=float(macro_f1),
                kappa=float(kappa),
                auc=float(auc),
                p0=float(report["0"]["precision"]),
                r0=float(report["0"]["recall"]),
                f10=float(report["0"]["f1-score"]),
                p1=float(report["1"]["precision"]),
                r1=float(report["1"]["recall"]),
                f11=float(report["1"]["f1-score"]),
            )
        )

        print(f"\n================ Fold {fold_id} ================")
        print(f"Class 0 -> P:{report['0']['precision']:.4f} R:{report['0']['recall']:.4f} F1:{report['0']['f1-score']:.4f}")
        print(f"Class 1 -> P:{report['1']['precision']:.4f} R:{report['1']['recall']:.4f} F1:{report['1']['f1-score']:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Kappa:    {kappa:.4f}")
        print(f"AUC:      {auc:.4f}" if not np.isnan(auc) else "AUC:      n/a")

    p0_m, p0_s = mean_std([r.p0 for r in fold_results])
    r0_m, r0_s = mean_std([r.r0 for r in fold_results])
    f10_m, f10_s = mean_std([r.f10 for r in fold_results])
    p1_m, p1_s = mean_std([r.p1 for r in fold_results])
    r1_m, r1_s = mean_std([r.r1 for r in fold_results])
    f11_m, f11_s = mean_std([r.f11 for r in fold_results])
    macro_m, macro_s = mean_std([r.macro_f1 for r in fold_results])
    kappa_m, kappa_s = mean_std([r.kappa for r in fold_results])
    auc_m, auc_s = mean_std([r.auc for r in fold_results])

    print("\n===============================================")
    print(f"{OUTER_SPLITS}-Fold GroupCV Summary (emodata, MLP 5 features)")
    print(f"Class 0 Precision: {p0_m:.4f} ± {p0_s:.4f}")
    print(f"Class 0 Recall:    {r0_m:.4f} ± {r0_s:.4f}")
    print(f"Class 0 F1:        {f10_m:.4f} ± {f10_s:.4f}")
    print(f"Class 1 Precision: {p1_m:.4f} ± {p1_s:.4f}")
    print(f"Class 1 Recall:    {r1_m:.4f} ± {r1_s:.4f}")
    print(f"Class 1 F1:        {f11_m:.4f} ± {f11_s:.4f}")
    print(f"Macro F1:          {macro_m:.4f} ± {macro_s:.4f}")
    print(f"Kappa:             {kappa_m:.4f} ± {kappa_s:.4f}")
    print(f"AUC:               {auc_m:.4f} ± {auc_s:.4f}")


if __name__ == "__main__":
    main()
