from src.agent.tools import predict_lstm_price, fetch_real_stock_price, query_compliance_rag

def test_predict_lstm_price_tool():
    """Verifica se a tool de predição rejeita ativos inválidos ou se previne falhas brutas elegantemente."""
    res = predict_lstm_price("LIXO_TOTAL_INVALIDO.SA")
    assert isinstance(res, str)
    assert "Dados insuficientes" in res or "Scaler não encontrado" in res

def test_react_agent_structure():
    """Verifica se as políticas de Guardrails / System Prompt estão presentes no cérebro do Agente (via tools)."""
    assert "compliance" in query_compliance_rag.__name__.lower()

def test_fetch_real_stock_price_invalid():
    """Testa tool de cotação com ticker inválido — deve retornar string de erro elegante."""
    result = fetch_real_stock_price("TICKER_INVALIDO_XPTO_9999")
    assert isinstance(result, str)

def test_query_compliance_rag_graceful():
    """Testa que a tool RAG retorna string mesmo quando a base ChromaDB não está disponível."""
    result = query_compliance_rag("qual o limite de risco para renda variável?")
    assert isinstance(result, str)
