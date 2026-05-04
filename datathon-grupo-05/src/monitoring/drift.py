"""Módulo de Detecção de Drift — GAP 06 (Concept Drift e Data Drift).

Resolve:
    - GAP 06: Detecção de degradação silenciosa em produção
    - GAP 01: Observabilidade com métricas de qualidade de modelo

Arquitetura de 3 camadas:
    1. Data Drift  — PSI sobre a distribuição dos preços de entrada
    2. Performance — MAPE rolling 7 dias (predição vs. fechamento real)
    3. Concept Drift — tendência dos resíduos (viés sistemático)

Integração com Airflow:
    Este script é invocado pela task `detectar_drift` da DAG ml_pipeline_dag.py.
    Cada ticker tem seu próprio processo — sem memória compartilhada.

Uso:
    poetry run python src/monitoring/drift.py --ticker_id petr4_sa --ticker PETR4.SA
"""
import sys
import os
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import torch
import yfinance as yf
import mlflow

warnings.filterwarnings("ignore")

# Garante que o módulo de modelos seja encontrado quando rodado diretamente
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from baseline_lstm import StockLSTM  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Thresholds MLOps (documentados no README e alinhados ao Datathon) ──────────
PSI_WARNING  = 0.1   # Investigar
PSI_CRITICAL = 0.2   # Trigger de retreino
MAPE_CRITICAL = 12.0  # % — MAPE rolling 7 dias acima disso = retreino
RESIDUAL_BIAS_THRESHOLD = 0.05  # Viés sistemático: média dos resíduos normalizados

PROCESSED_DIR = Path("data/processed")
WINDOW_SIZE   = 60


# ─── Funções de Suporte ─────────────────────────────────────────────────────────

def calcular_psi(referencia: np.ndarray, atual: np.ndarray, buckets: int = 10) -> float:
    """Calcula o PSI (Population Stability Index) entre duas distribuições.

    PSI < 0.1  → estável
    PSI < 0.2  → warning
    PSI >= 0.2 → trigger de retreino

    Args:
        referencia: Distribuição de referência (dados de treino).
        atual: Distribuição atual (últimos dados de mercado).
        buckets: Número de bins para o histograma.

    Returns:
        Valor do PSI (float).
    """
    # Define bins com base na referência para garantir comparabilidade
    bins = np.linspace(
        min(referencia.min(), atual.min()),
        max(referencia.max(), atual.max()),
        buckets + 1
    )

    ref_counts, _ = np.histogram(referencia, bins=bins)
    cur_counts, _ = np.histogram(atual,      bins=bins)

    # Suavização para evitar log(0)
    ref_pct = (ref_counts + 1e-6) / (len(referencia) + 1e-6 * buckets)
    cur_pct = (cur_counts + 1e-6) / (len(atual)      + 1e-6 * buckets)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def carregar_modelo_mlflow(ticker_id: str) -> StockLSTM | None:
    """Carrega o modelo mais recente registrado no MLflow para o ticker.

    Args:
        ticker_id: Identificador do ativo.

    Returns:
        Modelo PyTorch carregado ou None se não encontrado.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name("stock_prediction_lstm")
        if not experiment:
            logger.warning("Experimento 'stock_prediction_lstm' não encontrado no MLflow.")
            return None

        ticker_upper = ticker_id.upper().replace("_", ".")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.ticker = '{ticker_upper}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            logger.warning("Nenhum run encontrado para %s.", ticker_upper)
            return None

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/lstm_model"
        logger.info("Carregando modelo do run %s...", run_id[:8])

        model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
        model.eval()
        return model

    except Exception as exc:
        logger.error("Falha ao carregar modelo do MLflow: %s", exc)
        return None


def buscar_dados_recentes(ticker: str, dias: int = 90) -> pd.DataFrame | None:
    """Busca os últimos N dias de preços via yfinance.

    Args:
        ticker: Símbolo do ativo (ex: PETR4.SA).
        dias: Número de dias históricos a buscar.

    Returns:
        DataFrame com coluna 'Close' ou None em caso de falha.
    """
    fim = datetime.today()
    inicio = fim - timedelta(days=dias)
    try:
        df = yf.download(ticker, start=inicio.strftime("%Y-%m-%d"),
                         end=fim.strftime("%Y-%m-%d"), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Close"]].dropna()
        logger.info("Dados recentes: %d pregões para %s.", len(df), ticker)
        return df
    except Exception as exc:
        logger.error("Falha ao buscar dados de %s: %s", ticker, exc)
        return None


def calcular_mape_rolling(
    df_recente: pd.DataFrame,
    scaler,
    model: StockLSTM,
    window: int = WINDOW_SIZE,
    rolling_days: int = 7,
) -> float | None:
    """Calcula o MAPE rolling dos últimos N dias comparando predição vs. real.

    Args:
        df_recente: DataFrame com preços reais recentes.
        scaler: MinMaxScaler ajustado no treino.
        model: Modelo LSTM carregado.
        window: Tamanho da janela de entrada (60 dias).
        rolling_days: Dias recentes para calcular o MAPE.

    Returns:
        MAPE rolling (float %) ou None se dados insuficientes.
    """
    if len(df_recente) < window + rolling_days:
        logger.warning("Dados insuficientes para calcular MAPE rolling.")
        return None

    closes = df_recente["Close"].values.reshape(-1, 1)
    closes_scaled = scaler.transform(closes)

    mapes = []
    start_idx = len(closes_scaled) - rolling_days

    for i in range(start_idx, len(closes_scaled)):
        if i < window:
            continue
        janela = closes_scaled[i - window:i].reshape(1, window, 1)
        tensor_in = torch.tensor(janela, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(tensor_in).item()

        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        real      = closes[i][0]

        if real != 0:
            mape = abs((real - pred_real) / real) * 100
            mapes.append(mape)

    return float(np.mean(mapes)) if mapes else None


# ─── Execução Principal ─────────────────────────────────────────────────────────

def detectar_drift(ticker_id: str, ticker: str) -> dict:
    """Orquestra as 3 camadas de detecção de drift e loga no MLflow.

    Args:
        ticker_id: Identificador interno (ex: petr4_sa).
        ticker: Símbolo de mercado (ex: PETR4.SA).

    Returns:
        Dicionário com métricas e status de alerta.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("drift_monitoring")

    run_name = f"drift_{ticker_id}_{datetime.today().strftime('%Y%m%d')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("ticker", ticker)
        mlflow.set_tag("owner", "grupo-05")
        mlflow.set_tag("phase", "monitoring-drift")
        mlflow.log_param("ticker_id", ticker_id)
        mlflow.log_param("psi_warning_threshold", PSI_WARNING)
        mlflow.log_param("psi_critical_threshold", PSI_CRITICAL)
        mlflow.log_param("mape_critical_threshold", MAPE_CRITICAL)

        resultado = {"ticker_id": ticker_id, "alertas": []}

        # ── Camada 1: Data Drift (PSI) ─────────────────────────────────────────
        scaler_path = PROCESSED_DIR / f"{ticker_id}_scaler.pkl"
        X_train_path = PROCESSED_DIR / f"{ticker_id}_X_train.npy"

        if not scaler_path.exists() or not X_train_path.exists():
            logger.error("Artefatos de treino ausentes. Execute o pipeline da Fase A.")
            mlflow.set_tag("status", "ERRO_ARTEFATOS")
            return resultado

        scaler = joblib.load(scaler_path)
        X_train = np.load(X_train_path)

        # Distribuição de referência: valores normalizados do conjunto de treino
        ref_distribution = X_train.flatten()

        df_recente = buscar_dados_recentes(ticker, dias=90)
        if df_recente is None or len(df_recente) < WINDOW_SIZE:
            mlflow.set_tag("status", "ERRO_DADOS_RECENTES")
            return resultado

        closes_recentes = df_recente["Close"].values.reshape(-1, 1)
        cur_distribution = scaler.transform(closes_recentes).flatten()

        psi = calcular_psi(ref_distribution, cur_distribution)
        logger.info("[%s] PSI calculado: %.4f", ticker_id, psi)

        mlflow.log_metric("data_drift_psi", psi)
        resultado["psi"] = psi

        if psi >= PSI_CRITICAL:
            alerta = f"CRITICO: PSI={psi:.3f} >= {PSI_CRITICAL} → retreino necessário"
            logger.warning(alerta)
            resultado["alertas"].append(alerta)
            mlflow.set_tag("data_drift_status", "RETRAIN")
        elif psi >= PSI_WARNING:
            alerta = f"WARNING: PSI={psi:.3f} >= {PSI_WARNING} → investigar mercado"
            logger.warning(alerta)
            resultado["alertas"].append(alerta)
            mlflow.set_tag("data_drift_status", "WARNING")
        else:
            logger.info("[%s] Data Drift: ESTAVEL (PSI=%.4f)", ticker_id, psi)
            mlflow.set_tag("data_drift_status", "OK")

        # ── Camada 2: Performance Drift (MAPE Rolling 7d) ─────────────────────
        model = carregar_modelo_mlflow(ticker_id)
        if model is not None:
            mape_rolling = calcular_mape_rolling(df_recente, scaler, model)
            if mape_rolling is not None:
                logger.info("[%s] MAPE rolling 7d: %.2f%%", ticker_id, mape_rolling)
                mlflow.log_metric("mape_rolling_7d", mape_rolling)
                resultado["mape_rolling_7d"] = mape_rolling

                if mape_rolling > MAPE_CRITICAL:
                    alerta = f"CRITICO: MAPE rolling={mape_rolling:.2f}% > {MAPE_CRITICAL}% → retreino"
                    logger.warning(alerta)
                    resultado["alertas"].append(alerta)
                    mlflow.set_tag("performance_drift_status", "RETRAIN")
                else:
                    logger.info("[%s] Performance Drift: OK (MAPE=%.2f%%)", ticker_id, mape_rolling)
                    mlflow.set_tag("performance_drift_status", "OK")
        else:
            logger.warning("Modelo não disponível — MAPE rolling ignorado.")
            mlflow.set_tag("performance_drift_status", "SEM_MODELO")

        # ── Camada 3: Concept Drift (Viés Sistemático dos Resíduos) ──────────
        if model is not None and len(df_recente) >= WINDOW_SIZE + 30:
            closes_sc = scaler.transform(
                df_recente["Close"].values.reshape(-1, 1)
            )
            residuos = []
            for i in range(WINDOW_SIZE, min(WINDOW_SIZE + 30, len(closes_sc))):
                janela = closes_sc[i - WINDOW_SIZE:i].reshape(1, WINDOW_SIZE, 1)
                tensor_in = torch.tensor(janela, dtype=torch.float32)
                with torch.no_grad():
                    pred = model(tensor_in).item()
                residuos.append(closes_sc[i][0] - pred)

            vies = float(np.mean(residuos))
            logger.info("[%s] Viés médio dos resíduos: %.4f", ticker_id, vies)
            mlflow.log_metric("residual_bias", vies)
            resultado["residual_bias"] = vies

            if abs(vies) > RESIDUAL_BIAS_THRESHOLD:
                alerta = f"CONCEPT DRIFT: viés sistemático={vies:.4f} — modelo perdeu calibração"
                logger.warning(alerta)
                resultado["alertas"].append(alerta)
                mlflow.set_tag("concept_drift_status", "DETECTADO")
            else:
                mlflow.set_tag("concept_drift_status", "OK")

        # ── Status Final ───────────────────────────────────────────────────────
        status_final = "RETRAIN_NECESSARIO" if resultado["alertas"] else "ESTAVEL"
        mlflow.set_tag("drift_status_final", status_final)
        resultado["status"] = status_final

        logger.info("[%s] Resultado final: %s | Alertas: %d",
                    ticker_id, status_final, len(resultado["alertas"]))
        for a in resultado["alertas"]:
            logger.warning("  → %s", a)

    return resultado


def main():
    parser = argparse.ArgumentParser(
        description="Detecção de Drift — MLOps Financeiro (GAP 06)"
    )
    parser.add_argument("--ticker_id", type=str, default="petr4_sa",
                        help="Identificador do ativo (ex: petr4_sa)")
    parser.add_argument("--ticker",    type=str, default="PETR4.SA",
                        help="Símbolo de mercado (ex: PETR4.SA)")
    args = parser.parse_args()

    resultado = detectar_drift(args.ticker_id, args.ticker)

    if resultado.get("status") == "RETRAIN_NECESSARIO":
        logger.warning("Pipeline finalizado com alertas. Retreino recomendado.")
        sys.exit(1)  # Airflow detecta exit code != 0 como falha da task
    else:
        logger.info("Pipeline finalizado. Modelo estável.")
        sys.exit(0)


if __name__ == "__main__":
    main()
