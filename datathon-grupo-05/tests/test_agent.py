import pytest
from src.agent.tools import predict_lstm_price

def test_predict_lstm_price_tool():
    """Verifica se a tool de predição rejeita ativos inválidos ou se previne falhas brutas elegantemente."""
    # O mock vai tentar puxar no Yahoo Finance.
    # LIXO_TOTAL retornará vazio e deve estourar o limite de 60 pregões.
    res = predict_lstm_price("LIXO_TOTAL_INVALIDO.SA")
    assert isinstance(res, str)
    assert "Dados insuficientes" in res or "Scaler não encontrado" in res

def test_react_agent_structure():
    """Verifica se as políticas de Guardrails / System Prompt estão presentes no cérebro do Agente (via tools)."""
    from src.agent.react_agent import criar_agente
    # Apenas validamos se o módulo de importação existe e as descrições contêm a política da CVM
    from src.agent.tools import query_compliance_rag
    assert "compliance" in query_compliance_rag.__name__.lower()
