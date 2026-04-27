from __future__ import annotations

import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature transformations for model training.

    Args:
        df: Input data with numeric features.

    Returns:
        DataFrame with transformed numeric features and interactions.
    """
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not numeric_columns:
        raise ValueError('Input dataframe must contain numeric features.')

    features = df[numeric_columns].copy()
    features['feature_sum'] = features.sum(axis=1)
    
    if len(numeric_columns) >= 2:
        features['feature_1_x_feature_2'] = features[numeric_columns[0]] * features[numeric_columns[1]]
    else:
        features['feature_1_x_feature_2'] = features[numeric_columns[0]]
    
    features['feature_mean'] = features.mean(axis=1)
    features = features.fillna(0.0)

    return features
