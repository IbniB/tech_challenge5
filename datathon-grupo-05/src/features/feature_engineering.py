"""Módulo de processamento de pipelines de dados para alimentar a rede LSTM obedecendo os DVC inputs."""
import logging
import argparse
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

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features técnicas avançadas de mercado para o modelo LSTM."""
    # 1. Retornos
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Médias Móveis
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 3. Volatilidade e Bollinger Bands
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['Volatility_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Volatility_20'] * 2)
    
    # 4. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 5. MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Dropa os dias iniciais que não têm histórico suficiente para a Média Móvel de 26 dias
    return df.dropna().copy()


def create_sequences(features: np.ndarray, target: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Corta as janelas temporais contínuas e pareia com o próximo valor target."""
    x, y = [], []
    for i in range(window_size, len(features)):
        x.append(features[i-window_size:i, :])
        y.append(target[i, 0])
    return np.array(x), np.array(y)

def process_and_window_data(ticker_id: str, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
    """Rotina de transformação End-To-End convertendo de CSV bruto para Array 3D na escala da LSTM."""
    df = validate_and_load_data(ticker_id)
    df = add_technical_indicators(df)
    
    # Features e Target separados para evitar vazamento e facilitar a desnormalização
    feature_cols = ['Close', 'Volume', 'Daily_Return', 'Log_Return', 'SMA_20', 
                    'EMA_20', 'Volatility_20', 'BB_Upper', 'BB_Lower', 'RSI_14', 
                    'MACD', 'MACD_Signal']
    
    logger.info("Criadas %d features técnicas no pipeline.", len(feature_cols))
    
    features_data = df[feature_cols].values
    target_data = df[['Close']].values
    
    logger.info("Equilibrando amplitude dos valores usando Min-Max Scaler do Scikit-Learn...")
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features_data)
    scaled_target = target_scaler.fit_transform(target_data)
    
    logger.info("Produzindo agrupamento das janelas fatiadas contendo o tamanho t de %d dias...", window_size)
    x_windows, y_targets = create_sequences(scaled_features, scaled_target, window_size)
    
    # Shape garantido: [samples, time steps, features]
    logger.info("Shape do tensor tridimensional: %s", x_windows.shape)
    
    # MLOps Temporal Split (80% Treino / 20% Validação Cega)
    split_idx = int(len(x_windows) * 0.8)
    X_train, X_test = x_windows[:split_idx], x_windows[split_idx:]
    y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]
    
    logger.info("Treino: %s (X), %s (y) | Teste: %s (X), %s (y)", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Escrita atômica: salva em arquivo temporário e só renomeia quando concluído.
    artefatos = {
        f"{ticker_id}_X_train.npy": X_train,
        f"{ticker_id}_y_train.npy": y_train,
        f"{ticker_id}_X_test.npy":  X_test,
        f"{ticker_id}_y_test.npy":  y_test,
    }
    for nome, dado in artefatos.items():
        destino = PROCESSED_DATA_PATH / nome
        tmp = PROCESSED_DATA_PATH / f".tmp_{nome}"
        np.save(tmp, dado)
        tmp.replace(destino)  # replace é atômico no Windows
        logger.info("Artefato persistido: %s (shape=%s)", nome, dado.shape)

    joblib.dump(feature_scaler, PROCESSED_DATA_PATH / f"{ticker_id}_feature_scaler.pkl")
    joblib.dump(target_scaler, PROCESSED_DATA_PATH / f"{ticker_id}_target_scaler.pkl")
    
    logger.info("Toda a arquitetura transacional salva. Caminho liberado para iniciar os fits.")

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering e Windowing (Data Preparation para Deep Learning)")
    parser.add_argument("--ticker_id", type=str, default=DEFAULT_TICKER, help="ID contido no nome do arquivo dataset raw")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE, help="Comprimento/Delay de Lags do Time-Series")
    args = parser.parse_args()
    
    process_and_window_data(args.ticker_id, args.window)

if __name__ == "__main__":
    main()
