from __future__ import annotations

import pandas as pd

from src.models.baseline import BaselineModel, compute_classification_metrics
from src.models.train import generate_synthetic_data
from src.features.feature_engineering import compute_features


def test_generate_synthetic_data():
    data = generate_synthetic_data(seed=123, n_samples=10)
    assert 'target' in data.columns
    assert data.shape == (10, 3)


def test_baseline_fit_predict():
    data = generate_synthetic_data(seed=42, n_samples=50)
    X = compute_features(data.drop(columns=['target']))
    y = data['target']
    model = BaselineModel(random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    metrics = compute_classification_metrics(y, preds)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 'f1' in metrics
