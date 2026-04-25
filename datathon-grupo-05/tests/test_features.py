import pytest
import numpy as np
import pandas as pd
import pandera as pa

from src.features.feature_engineering import FINANCIAL_SCHEMA, create_sequences

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
    # Vetor de 10 dias com 1 feature de preço
    dummy_data = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    window = 3
    
    x, y = create_sequences(dummy_data, window)
    
    # Se window é 3, perco os primeiros 3 índices pra formar a 1a janela
    # Tamanho esperado das instâncias previsoras = 10 - 3 = 7
    assert x.shape == (7, 3)
    assert y.shape == (7,)
    
    # Primeira janela deve ter [1, 2, 3] e tentar prever o valor 4
    np.testing.assert_array_equal(x[0], np.array([1, 2, 3]))
    assert y[0] == 4
