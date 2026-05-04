import pytest
import numpy as np
import pandas as pd
import pandera as pa

from src.features.feature_engineering import (
    FINANCIAL_SCHEMA,
    add_technical_indicators,
    create_sequences,
    validate_and_load_data,
)

def test_schema_validation_success(sample_raw_data):
    """Garante que o contrato de MLOps permita passagem de dados saudáveis."""
    validated = FINANCIAL_SCHEMA.validate(sample_raw_data)
    assert not validated.empty
    assert list(validated.columns) == list(sample_raw_data.columns)

def test_schema_validation_failure(sample_raw_data):
    """Garante que nulls ou features ausentes quebram o pipeline imediatamente (Datalake defense)."""
    corrupted_data = sample_raw_data.copy()
    corrupted_data.loc[0, "Close"] = None

    with pytest.raises(pa.errors.SchemaError):
        FINANCIAL_SCHEMA.validate(corrupted_data)

def test_create_sequences():
    """Garante o janelamento matemático do LSTM perfeitamente."""
    dummy_features = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    dummy_target = dummy_features.copy()
    window = 3

    x, y = create_sequences(dummy_features, dummy_target, window)

    assert x.shape == (7, 3, 1)
    assert y.shape == (7,)
    np.testing.assert_array_equal(x[0], np.array([[1], [2], [3]]))
    assert y[0] == 4

def test_add_technical_indicators():
    """Verifica que indicadores técnicos são calculados e colunas esperadas estão presentes."""
    n = 60
    df = pd.DataFrame({
        "Close": np.linspace(10.0, 20.0, n) + np.random.default_rng(42).normal(0, 0.1, n),
        "Volume": [100000] * n,
    })
    result = add_technical_indicators(df)
    assert not result.empty
    for col in ("RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "SMA_20", "EMA_20"):
        assert col in result.columns, f"Coluna {col} não encontrada"

def test_validate_missing_file():
    """Garante que FileNotFoundError é lançado para arquivo de ticker inexistente."""
    with pytest.raises(FileNotFoundError):
        validate_and_load_data("ticker_que_nao_existe_xpto")
