from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'feature_1': [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01],
            'feature_2': [0.05, 0.07, 0.03, 0.08, 0.06, 0.02, 0.09, 0.04],
            'feature_3': [1e6, 2e6, 1.5e6, 3e6, 2.5e6, 1.2e6, 2.8e6, 1.8e6],
            'target': [0, 1, 0, 1, 1, 0, 1, 1],
        }
    )
