"""API de Serving — Predição LSTM com carregamento seguro de modelo (GAP 03).

Resolve:
    - GAP 03: Elimina o placeholder perigoso ("Awaiting Model Bind") que retornava
              predições falsas silenciosamente quando o MLflow estava indisponível.
              O modelo é carregado de forma segura no startup, e a API rejeita
              requisições explicitamente se não houver modelo disponível.
    - GAP 01: Prometheus para observabilidade operacional (latência, contagem, drift warnings).
"""
import logging
import os
import time

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Métricas Prometheus ────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("predict_requests_total", "Total de predições requisitadas")
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Latência das predições (s)")
DRIFT_WARNINGS  = Counter("data_drift_warnings", "Sinalizações de Data Drift na entrada")
MODEL_LOAD_FAILURES = Counter("model_load_failures_total", "Falhas ao carregar modelo do MLflow")

app = FastAPI(
    title="Mesa de Operações — Previsão Analítica LSTM",
    version="2.0.0",
    description="API de inferência LSTM para previsão de fechamento de ativos financeiros.",
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ─── Schema de Entrada ──────────────────────────────────────────────────────────
class TimeSeriesInput(BaseModel):
    data: list[list[list[float]]] = Field(
        ...,
        description="Tensor 3D das features normalizadas [batch, timesteps, features]. Ex: [1, 60, 12]"
    )
    asset_id: str = Field(default="PETR4.SA", description="Ticker do ativo")


# ─── Estado Global do Modelo ────────────────────────────────────────────────────
# Nunca usar placeholder ou valor hardcoded — isso é o anti-padrão do GAP 03.
# Se o modelo não carregou, a API falha explicitamente com 503.
_model = None
_model_run_id: str | None = None


def _carregar_modelo_mlflow(ticker_id: str = "petr4_sa") -> bool:
    """Carrega o modelo mais recente do experimento MLflow para o ticker informado.

    Retorna True se o carregamento for bem-sucedido, False caso contrário.
    Em caso de falha, a API opera sem modelo e rejeita predições (503).
    Isso elimina o anti-padrão de responder predições falsas silenciosamente (GAP 03).
    """
    global _model, _model_run_id

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("stock_prediction_lstm")
        if not experiment:
            logger.error("Experimento 'stock_prediction_lstm' não encontrado no MLflow.")
            MODEL_LOAD_FAILURES.inc()
            return False

        ticker_upper = ticker_id.upper().replace("_", ".")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.ticker = '{ticker_upper}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            logger.error("Nenhum run encontrado para %s no MLflow.", ticker_upper)
            MODEL_LOAD_FAILURES.inc()
            return False

        run = runs[0]
        model_uri = f"runs:/{run.info.run_id}/lstm_model"
        _model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
        _model.eval()
        _model_run_id = run.info.run_id[:8]

        logger.info("Modelo carregado com sucesso. Run: %s | Ticker: %s", _model_run_id, ticker_upper)
        return True

    except Exception as exc:
        logger.error("Falha ao carregar modelo do MLflow: %s", exc)
        MODEL_LOAD_FAILURES.inc()
        return False


@app.on_event("startup")
def startup_event():
    """Carrega o modelo no startup. Se falhar, a API sobe mas rejeita predições."""
    ticker_id = os.getenv("SERVING_TICKER_ID", "petr4_sa")
    sucesso = _carregar_modelo_mlflow(ticker_id)
    if not sucesso:
        logger.warning(
            "API iniciada SEM modelo carregado. "
            "Endpoint /predict retornará 503 até o modelo estar disponível."
        )


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/infer")
async def infer_asset_price(payload: TimeSeriesInput):
    """Realiza inferência de preço de fechamento normalizado para o ativo informado."""
    REQUEST_COUNT.inc()
    start_time = time.time()

    # Rejeição explícita se modelo não disponível — elimina o anti-padrão GAP 03
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo não disponível. O serviço está aguardando o carregamento do MLflow. "
                "Verifique o experimento 'stock_prediction_lstm' e tente novamente."
            ),
        )

    try:
        sequence = np.array(payload.data)
        tensor_seq = torch.tensor(sequence, dtype=torch.float32)

        with torch.no_grad():
            prediction = _model(tensor_seq).item()

        # Watchdog de Data Drift na entrada (GAP 06 na camada de serving)
        mean_input = float(np.mean(payload.data))
        std_input  = float(np.std(payload.data))
        if mean_input < 0.0 or mean_input > 1.0 or std_input > 0.5:
            DRIFT_WARNINGS.inc()
            logger.warning(
                "Possível Data Drift na entrada de %s: mean=%.3f, std=%.3f",
                payload.asset_id, mean_input, std_input,
            )

        logger.info("Inferência OK — %s | scaled=%.5f | run=%s",
                    payload.asset_id, prediction, _model_run_id)

    except Exception as exc:
        logger.error("Erro na inferência: %s", exc)
        raise HTTPException(status_code=500, detail=f"Erro interno na inferência: {exc}")
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "asset":              payload.asset_id,
        "prediction_scaled":  round(prediction, 6),
        "model_run_id":       _model_run_id,
        "status":             "ok",
    }


@app.post("/reload-model")
async def reload_model():
    """Força o recarregamento do modelo do MLflow sem derrubar a API.
    
    Isso implementa o padrão de atualização incremental (GAP 03):
    o modelo antigo permanece em memória até o novo ser carregado com sucesso.
    """
    ticker_id = os.getenv("SERVING_TICKER_ID", "petr4_sa")
    sucesso = _carregar_modelo_mlflow(ticker_id)
    if not sucesso:
        raise HTTPException(status_code=503, detail="Falha ao recarregar modelo do MLflow.")
    return {"status": "modelo recarregado", "run_id": _model_run_id}


@app.get("/")
async def liveness_probe():
    """Liveness probe para Kubernetes/Docker."""
    return {"status": "alive"}

@app.get("/ready")
async def health_check():
    """Health check com estado real do modelo — nunca retorna OK com modelo ausente."""
    return {
        "status":        "ok" if _model is not None else "degraded",
        "model_loaded":  _model is not None,
        "model_run_id":  _model_run_id,
        "version":       "2.0.0",
    }


@app.get("/startup")
async def startup_probe():
    """Startup probe para Kubernetes/Docker — Garante que dependências pesadas carregaram."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado")
    return {"status": "started"}


from pydantic import BaseModel

class TrainPayload(BaseModel):
    ticker: str = "PETR4.SA"

from fastapi import BackgroundTasks

@app.post("/train")
async def schedule_train(payload: TrainPayload, background_tasks: BackgroundTasks):
    """Agenda treinamento LSTM assíncrono para não travar a API."""
    import subprocess
    def run_training(ticker):
        logger.info(f"Iniciando treinamento assíncrono para {ticker}")
        subprocess.run(["poetry", "run", "python", "src/models/train.py", "--ticker_id", ticker])
        
    background_tasks.add_task(run_training, payload.ticker)
    return {"status": "Treinamento agendado", "ticker": payload.ticker}


class AgentPayload(BaseModel):
    query: str

@app.post("/agent")
async def agent_query(payload: AgentPayload):
    """Aciona o Agente ReAct Financeiro via API."""
    try:
        # A chamada ao LangChain Agent (Import assíncrono para não pesar o startup)
        from src.agent.react_agent import _carregar_llm_local
        # Mocking or calling the real agent depending on memory 
        return {"response": "Agente acionado. (Para inferência real via API, instanciar a chain)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_quality")
async def evaluate_quality():
    """Executa o pipeline de avaliação (LLM-as-a-judge / Ragas) do Golden Set."""
    import subprocess
    logger.info("Iniciando Quality Gate...")
    subprocess.Popen(["poetry", "run", "python", "evaluation/llm_judge.py"])
    return {"status": "Avaliação assíncrona iniciada. Acompanhe os logs."}
