"""Script automatizado para coleta de dados financeiros focados na geração do Dataset Raw do LSTM."""
import logging
import argparse
from pathlib import Path
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TICKER = "PETR4.SA"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-07-20"
RAW_DATA_PATH = Path("data/raw")

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Extrai os dados da bolsa via yahoo finance.

    Args:
        ticker: Sigla do ativo financeiro.
        start_date: Data de partida (YYYY-MM-DD).
        end_date: Data de fim (YYYY-MM-DD).

    Returns:
        Um dataset pandas com o histórico diário do ativo.
    """
    logger.info("Buscando dados históricos de mercado para %s entre %s e %s", ticker, start_date, end_date)
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"O Dataset retornado para o arquivo {ticker} est vazio. Verifique os parmetros.")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        logger.info("Extração do sistema conectada perfeitamente: %d instâncias/linhas de datas recuperadas com sucesso.", df.shape[0])
        return df
    except Exception as e:
        logger.error("Falha estrutural ou de network na extração do ativo: %s", e)
        raise

def save_raw_data(df: pd.DataFrame, ticker: str) -> Path:
    """Garante que a escrita ocorra sob a jurisdição do isolamento DVC para uso nos treinamentos ML.
    
    Args:
        df: Dataset pandas previamente testado e validado.
        ticker: O qualificador do nome do ativo extraído.
    """
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filename = f"{ticker.replace('.', '_').lower()}_raw.csv"
    file_path = RAW_DATA_PATH / filename
    
    logger.info("Protegendo o dado via Data Storage (Raw Route): %s", file_path)
    df.to_csv(file_path, index=False)
    
    logger.info("Atenção ao MLOps: Rastreie essa carga APENAS usando `dvc add %s`", file_path)
    return file_path

def main():
    parser = argparse.ArgumentParser(description="Extrator temporal Datalake Financeiro - Fase 04 (LSTM) e 05 (Governança MLOps)")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Cód do Ativo CVM/Bovespa (Ex: PETR4.SA)")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="Data limite esquerda de extração (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=DEFAULT_END, help="Data mais recente da busca (YYYY-MM-DD)")
    args = parser.parse_args()

    df = fetch_stock_data(args.ticker, args.start, args.end)
    save_raw_data(df, args.ticker)

if __name__ == "__main__":
    main()
