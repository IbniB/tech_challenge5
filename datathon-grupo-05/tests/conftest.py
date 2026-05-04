import pytest
import pandas as pd

@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Mock do raw data importado pelo yfinance para testes MLOps offline."""
    return pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "Open": [10.0, 11.0, 10.5, 12.0],
        "High": [11.5, 11.5, 12.5, 13.0],
        "Low": [9.5, 10.0, 10.0, 11.0],
        "Close": [11.0, 10.5, 12.0, 12.5],
        "Volume": [1000, 1500, 1200, 2000]
    })
