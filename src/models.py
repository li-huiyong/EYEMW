"""Model definitions with consistent interface.

Each `get_*_pipeline` returns (pipeline, param_grid) ready for GridSearchCV.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import SEED


# ── Sklearn pipelines ────────────────────────────────────────────────────────

def get_svm_pipeline():
    """SVC(RBF, balanced) with imputer + scaler."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=SEED, class_weight="balanced")),
    ])
    grid = {
        "svm__kernel": ["rbf"],
        "svm__C": [1],
        "svm__gamma": ["scale"],
    }
    return pipe, grid


def get_logreg_pipeline():
    """L2-penalized logistic regression with imputer + scaler."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="saga", l1_ratio=0, max_iter=10000,
            random_state=SEED, class_weight="balanced",
        )),
    ])
    grid = {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    return pipe, grid


def get_rf_pipeline():
    """Random forest (no scaler needed) with imputer."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            random_state=SEED, class_weight="balanced", n_jobs=1,
        )),
    ])
    grid = {
        "rf__n_estimators": [200],
        "rf__max_depth": [None],
        "rf__min_samples_leaf": [1],
    }
    return pipe, grid


class ScalePosWeightedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper that sets scale_pos_weight from training labels."""

    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=200,
                 min_child_weight=1, random_state=SEED):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier
        n_pos = max(np.sum(y == 1), 1)
        n_neg = np.sum(y == 0)
        self.model_ = XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=n_neg / n_pos,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def get_xgb_pipeline():
    """XGBoost with dynamic scale_pos_weight and imputer."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", ScalePosWeightedXGB()),
    ])
    grid = {
        "xgb__max_depth": [6],
        "xgb__learning_rate": [0.1],
        "xgb__n_estimators": [200],
        "xgb__min_child_weight": [1],
    }
    return pipe, grid


class _TorchMLP(object):
    """Minimal PyTorch network: input -> 16 -> ReLU -> Dropout -> 8 -> ReLU -> 1."""

    def __init__(self, input_dim, dropout=0.2):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def __call__(self, x):
        return self.net(x).squeeze(1)

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible PyTorch MLP with dropout and grouped early stopping.

    Matches the architecture of the old ``emodata-groupcv-emo5-mlp.py``:
    input -> 16 -> ReLU -> Dropout(0.2) -> 8 -> ReLU -> 1.

    Class imbalance is handled via ``BCEWithLogitsLoss(pos_weight=neg/pos)``.
    Early stopping monitors macro-F1 on a participant-grouped validation set
    when ``groups`` is passed to ``fit()``.
    """

    def __init__(self, max_epochs=150, lr=1e-3, weight_decay=1e-4,
                 batch_size=32, patience=15, val_size=0.2,
                 dropout=0.2, random_state=SEED):
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.val_size = val_size
        self.dropout = dropout
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
        from sklearn.metrics import f1_score

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)
        device = torch.device("cpu")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=self.val_size,
                                   random_state=self.random_state)
            tr, va = next(gss.split(X, y, groups))
        else:
            ss = ShuffleSplit(n_splits=1, test_size=self.val_size,
                              random_state=self.random_state)
            tr, va = next(ss.split(X, y))

        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        n_pos = max(float(np.sum(y_tr == 1)), 1.0)
        n_neg = float(np.sum(y_tr == 0))
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32,
                                  device=device)

        model = _TorchMLP(X_tr.shape[1], dropout=self.dropout).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)

        ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(
                                self.random_state))

        best_f1, no_improve = -1.0, 0
        best_state = None

        for _ in range(self.max_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                va_logits = model(torch.tensor(X_va, device=device))
                va_probs = torch.sigmoid(va_logits).cpu().numpy()
            va_pred = (va_probs >= 0.5).astype(int)
            val_f1 = f1_score(y_va.astype(int), va_pred, average="macro",
                              zero_division=0)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._device = device
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X, device=self._device))
            p1 = torch.sigmoid(logits).cpu().numpy()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def get_mlp_pipeline():
    """PyTorch MLP with dropout, grouped early stopping, and pos_weight."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", TorchMLPClassifier(random_state=SEED)),
    ])
    grid: dict = {}
    return pipe, grid


ALL_SKLEARN_MODELS = {
    "SVM": get_svm_pipeline,
    "LogReg": get_logreg_pipeline,
    "RF": get_rf_pipeline,
    "XGB": get_xgb_pipeline,
}
