from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaselineModel:
    def __init__(self, random_state: int = 42, C: float = 1.0, solver: str = 'liblinear') -> None:
        self.pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=random_state,
                    C=C,
                    solver=solver,
                    max_iter=1000,
                )),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> Pipeline:
        return joblib.load(path)


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred),
    }
