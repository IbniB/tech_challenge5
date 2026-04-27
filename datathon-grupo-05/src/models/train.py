from __future__ import annotations

import logging
import sys
from pathlib import Path

# Adicionar diretório raiz ao path para permitir imports de 'src'
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_ARTIFACT_PATH, load_model_config
from src.features.feature_engineering import compute_features
from src.models.baseline import BaselineModel, compute_classification_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def Load_Data(symbol: str = "NVDA", period: str = "1y") -> pd.DataFrame:
    """Carrega dados históricos de mercado usando yfinance.

    Args:
        symbol: Símbolo da ação (padrão: NVDA).
        period: Período de dados (padrão: 1y para 1 ano).

    Returns:
        DataFrame com features de preço e target binário.
    """
    logger.info("Carregando dados de %s para período %s", symbol, period)
    data = yf.download(symbol, period=period, progress=False)

    if data.empty:
        raise ValueError(f"Nenhum dado encontrado para {symbol}")

    data = data.reset_index()
    data = data.rename(columns={"Date": "date"})

    data["daily_return"] = data["Close"].pct_change()
    data["price_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volume_MA"] = data["Volume"].rolling(window=5).mean()

    data["target"] = (data["daily_return"].shift(-1) > 0).astype(int)
    data = data.dropna()

    feature_cols = ["daily_return", "price_range", "volume_MA"]
    result = data[feature_cols + ["target"]].copy()
    result.columns = ["feature_1", "feature_2", "feature_3", "target"]

    logger.info("Dados carregados: %d registros, colunas %s", len(result), list(result.columns))
    return result


def Generate_Synthetic_Data(seed: int = 42, n_samples: int = 500) -> pd.DataFrame:
    """Gera dados sintéticos para testes locais.

    Args:
        seed: Semente para reprodutibilidade.
        n_samples: Número de amostras.

    Returns:
        DataFrame com 3 features e target binário.
    """
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "feature_1": rng.uniform(-0.05, 0.05, n_samples),
            "feature_2": rng.uniform(0, 0.1, n_samples),
            "feature_3": rng.uniform(0, 1e7, n_samples),
        }
    )
    data["target"] = (data["feature_1"] + data["feature_2"] * 0.5 + rng.normal(0, 0.01, n_samples) > 0).astype(int)
    return data


def Train_Data_Model(data: pd.DataFrame, config: object) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X = compute_features(data.drop(columns=['target']))
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    with mlflow.start_run(run_name='baseline_training') as run:
        mlflow.log_param('random_state', config.random_state)
        mlflow.log_param('test_size', config.test_size)
        mlflow.log_param('solver', config.solver)
        mlflow.log_param('penalty', config.penalty)
        mlflow.log_param('C', config.C)
        mlflow.log_param('n_features', X_train.shape[1])
        mlflow.log_param('n_samples_train', X_train.shape[0])

        mlflow.set_tag('model_name', 'datathon_baseline')
        mlflow.set_tag('model_version', '0.1.0')
        mlflow.set_tag('model_type', 'classification')
        mlflow.set_tag('training_data_version', 'synthetic-v1')
        mlflow.set_tag('owner', 'grupo-05')
        mlflow.set_tag('phase', 'datathon-fase05')
        mlflow.set_tag("framework", BaselineModel.__module__.split(".")[0])

        model = BaselineModel(
            random_state=config.random_state,
            C=config.C,
            solver=config.solver,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        model.save(MODEL_ARTIFACT_PATH)
        mlflow.sklearn.log_model(model.pipeline, artifact_path='model')

        logger.info('Treinamento concluído: %s', metrics)
        return run.info.run_id


def main() -> None:
    config = load_model_config()

    try:
        data = Load_Data(symbol="NVDA")
        logger.info("Usando dados reais do NVDA")
    except Exception as exc:
        logger.warning("Falha ao carregar dados do yfinance: %s. Usando dados sintéticos.", exc)
        data = Generate_Synthetic_Data(seed=config.random_state)

    run_id = Train_Data_Model(data, config)
    logger.info('MLflow run_id=%s', run_id)
    print(f"\n✅ Treinamento concluído!")
    print(f"🏃 MLflow run_id: {run_id}")
    print(f"📊 Visualize em: http://127.0.0.1:5000/  (se MLflow server estiver rodando)\n")


if __name__ == '__main__':
    main()
