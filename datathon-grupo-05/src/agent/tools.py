"""Tools do Agente ReAct — Datathon Fase 05.

Três ferramentas obrigatórias (≥ 3 exigidas pela banca).
LLM: Qwen2.5-0.5B-Instruct (local, zero custo, sem API key).
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, zero custo).

    1. predict_lstm_price    — Consulta a API FastAPI interna (modelo LSTM treinado)
    2. fetch_real_stock_price — Cotação real via yfinance (âncora anti-alucinação)
    3. query_compliance_rag  — Consulta semântica na base de conformidade CVM/Risco
"""
import logging
import os

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


# ─── Tool 1: Predição LSTM via API Interna ──────────────────────────────────────

def predict_lstm_price(ticker: str) -> str:
    """Consulta a API FastAPI interna com o modelo LSTM treinado da tesouraria.

    Busca os últimos 60 dias de preço via yfinance, normaliza com o scaler
    e envia para o endpoint /predict da API de serving.

    Args:
        ticker: Símbolo do ativo (ex: PETR4.SA ou NVDC34.SA).

    Returns:
        String com a predição de preço normalizado ou mensagem de erro.
    """
    import joblib
    import numpy as np
    from pathlib import Path

    try:
        # Normaliza o ticker para o padrão esperado
        if not ticker.endswith(".SA") and ticker.upper() in ("PETR4", "NVDC34"):
            ticker = f"{ticker.upper()}.SA"

        # Determina o ticker_id para carregar o scaler correto
        ticker_id = ticker.lower().replace(".", "_").replace("-", "_")

        # Busca os últimos 100 dias (precisamos de 60 pregões + margem para as features técnicas de 26 dias)
        import yfinance as yf
        df = yf.download(ticker, period="100d", progress=False)
        if isinstance(df.columns, object) and hasattr(df.columns, 'levels'):
            df.columns = [c[0] for c in df.columns]
            
        from src.features.feature_engineering import add_technical_indicators
        df = add_technical_indicators(df)
        
        feature_cols = ['Close', 'Volume', 'Daily_Return', 'Log_Return', 'SMA_20', 
                        'EMA_20', 'Volatility_20', 'BB_Upper', 'BB_Lower', 'RSI_14', 
                        'MACD', 'MACD_Signal']
        
        features = df[feature_cols].values[-60:]

        if len(features) < 60:
            return f"Dados insuficientes para {ticker} após indicadores: {len(features)} pregões disponíveis."

        # Normaliza usando o scaler do treino
        scaler_path = Path(f"data/processed/{ticker_id}_feature_scaler.pkl")
        if not scaler_path.exists():
            return (
                f"Scaler não encontrado para {ticker_id}. "
                "Execute o data pipeline primeiro."
            )
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(features)
        
        # Envia o payload no formato correto: batch=1, timesteps=60, features=12
        payload = {"data": [scaled_data.tolist()]}

        # Chama a API de serving
        base_url = os.environ.get("PREDICT_API_URL", "http://localhost:8000")
        response = requests.post(f"{base_url}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            pred_scaled = data["prediction_scaled"]
            # Desnormaliza para R$
            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
            return (
                f"O modelo LSTM interno prevê que {ticker} fechará próximo de "
                f"R$ {pred_real:.2f} no próximo pregão. "
                f"(Valor normalizado: {pred_scaled:.5f} | Run MLflow: {data.get('model_run_id', 'N/A')})"
            )
        elif response.status_code == 503:
            return (
                f"API de predição indisponível (503): modelo não carregado. "
                "Verifique se o serviço de serving está rodando e se há modelos no MLflow."
            )
        else:
            return f"Erro da API ({response.status_code}): {response.text[:200]}"

    except requests.exceptions.ConnectionError:
        return (
            "API de serving inacessível. Certifique-se de que a FastAPI está rodando "
            "em http://localhost:8000 ou defina PREDICT_API_URL corretamente."
        )
    except Exception as exc:
        logger.error("Erro na tool de predição: %s", exc)
        return f"Erro ao consultar o modelo LSTM: {exc}"


# ─── Tool 2: Cotação Real via yfinance ─────────────────────────────────────────

def fetch_real_stock_price(ticker: str) -> str:
    """Busca o preço de fechamento mais recente via yfinance.

    Âncora de realidade: garante que o agente trabalha com dados reais
    de mercado, não com valores internos do modelo.

    Args:
        ticker: Símbolo do ativo (ex: PETR4.SA, NVDC34.SA, PETR4).

    Returns:
        String com o preço de fechamento mais recente.
    """
    try:
        if not ticker.endswith(".SA") and not any(c.isdigit() for c in ticker[-2:]):
            pass  # ticker internacional (ex: NVDA)
        elif not ticker.endswith(".SA"):
            ticker = f"{ticker.upper()}.SA"

        stk = yf.Ticker(ticker)
        hist = stk.history(period="2d")

        if hist.empty:
            return f"Dados de mercado indisponíveis para {ticker}. Verifique se o ticker está correto."

        close = hist["Close"].iloc[-1]
        data  = hist.index[-1].strftime("%d/%m/%Y")
        return (
            f"O ativo {ticker} registrou fechamento de R$ {close:.2f} em {data}. "
            "Dado extraído em tempo real via yfinance."
        )

    except Exception as exc:
        logger.error("Erro na tool de cotação: %s", exc)
        return f"Erro ao buscar cotação de {ticker}: {exc}"


# ─── Tool 3: Consulta Semântica de Compliance (RAG) ────────────────────────────

def query_compliance_rag(query: str) -> str:
    """Consulta a base de conformidade corporativa (CVM + Políticas de Risco).

    Busca semântica no índice FAISS construído com os documentos de compliance.
    Toda resposta envolvendo recomendação de ativo deve passar por esta tool.

    Args:
        query: Pergunta ou contexto a validar contra as diretrizes corporativas.

    Returns:
        Trechos relevantes das diretrizes de conformidade.
    """
    try:
        # Import local para evitar erro se OPENAI_API_KEY não estiver definida
        # no momento do carregamento do módulo
        from src.agent.rag_pipeline import consultar_rag
        return consultar_rag(query, k=3)
    except Exception as exc:
        logger.error("Erro na tool RAG: %s", exc)
        return f"Base de compliance temporariamente indisponível: {exc}"
