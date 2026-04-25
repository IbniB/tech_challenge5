"""Módulo de processamento de pipelines de dados para alimentar a rede LSTM obedecendo os DVC inputs."""
import logging
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
DEFAULT_TICKER = "petr4_sa"
DEFAULT_WINDOW_SIZE = 60

# Data Contract na Engenharia MLOps (Gap 04 diz para validarmos os dados com Pandera para não aceitar lixo)
FINANCIAL_SCHEMA = DataFrameSchema({
    "Date": Column(str, nullable=False),
    "Open": Column(float, nullable=False),
    "High": Column(float, nullable=False),
    "Low": Column(float, nullable=False),
    "Close": Column(float, nullable=False),
    "Volume": Column(int, nullable=False, coerce=True), # Volume pode no schema ser string/float -> force int
})

def validate_and_load_data(ticker_id: str) -> pd.DataFrame:
    """Carrega sob o contract schema para evitar poison data em produção."""
    file_path = RAW_DATA_PATH / f"{ticker_id}_raw.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Erro no pipeline: Faltando {file_path}. Rode o módulo coletor via DVC antes.")
    
    logger.info("Carregando snapshot de mercado %s validando contra esquema corporativo pandera...", file_path)
    # skiprows or header adjust may be needed if yfinance saved multiindex, usually reset_index flattens it properly unless Ticker was an axis.
    df = pd.read_csv(file_path, header=[0, 1] if "Price" in str(pd.read_csv(file_path, nrows=2).columns) else 0)
    
    # Fallback to flatten nasty multi-indices from yfinance 0.2+ formats
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df.columns = df.columns.str.strip() # limpando brancos de header se existirem

    try:
        validated_df = FINANCIAL_SCHEMA.validate(df)
        logger.info("Dataset financeiro higienizado com sucesso. Nenhuma quebra no Contract.")
        return validated_df
    except pa.errors.SchemaError as exc:
        logger.error("DANGER! Data Drift detectado na injeção crua! Valores/Schema corrompidos da Internet. Detalhes: %s", exc)
        raise

def create_sequences(data: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Corta as janelas temporais contínuas e pareia com o próximo valor target (Close)."""
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def process_and_window_data(ticker_id: str, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
    """Rotina de transformação End-To-End convertendo de CSV bruto para Array 3D na escala da LSTM."""
    df = validate_and_load_data(ticker_id)
    
    # A métrica financeira focada em fechamentos
    target_data = df[["Close"]].values
    
    logger.info("Equilibrando amplitude dos valores usando Min-Max Scaler do Scikit-Learn...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target_data)
    
    logger.info("Produzindo agrupamento das janelas fatiadas contendo o tamanho t de %d dias...", window_size)
    x_windows, y_targets = create_sequences(scaled_data, window_size)
    
    # Redimensionando para [samples, time steps, features] que as redes PyTorch/Keras LSTM exigem
    x_windows = np.reshape(x_windows, (x_windows.shape[0], x_windows.shape[1], 1))
    
    logger.info("Shape X gerado com sucesso - Matriz de Previsores Base: %s", x_windows.shape)
    logger.info("Shape Y gerado com sucesso - Vetor de Target Histórico: %s", y_targets.shape)
    
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    np.save(PROCESSED_DATA_PATH / f"{ticker_id}_X_train.npy", x_windows)
    np.save(PROCESSED_DATA_PATH / f"{ticker_id}_y_train.npy", y_targets)
    joblib.dump(scaler, PROCESSED_DATA_PATH / f"{ticker_id}_scaler.pkl")
    
    logger.info("Toda a arquitetura transacional salva. Caminho liberado para iniciar os fits.")

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering e Windowing (Data Preparation para Deep Learning)")
    parser.add_argument("--ticker_id", type=str, default=DEFAULT_TICKER, help="ID contido no nome do arquivo dataset raw")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE, help="Comprimento/Delay de Lags do Time-Series")
    args = parser.parse_args()
    
    process_and_window_data(args.ticker_id, args.window)

if __name__ == "__main__":
    main()
