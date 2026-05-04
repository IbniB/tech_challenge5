"""Pipeline de treinamento LSTM com MLflow tracking padronizado (Datathon Fase 05).

Resolve:
    - GAP 05: Governança de versionamento de modelos (tags obrigatórias + lineage)
    - GAP 01: Métricas financeiras rastreatéis (RMSE, MAE, MAPE, σ-tolerance) por epoch
    - GAP 04: Logging estruturado, type hints e docstrings (padrão Datathon)

Métrica de negócio (critério de aceite da banca):
    σ-tolerance: ≥ 70% das predições dentro de 0.5 desvios-padrão do preço real observado.
"""
import os
import subprocess
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

from baseline_lstm import StockLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir: str, ticker_id: str) -> tuple:
    """Carrega tensores pré-processados da Fase A e converte para PyTorch.

    Args:
        data_dir: Diretório onde os arquivos .npy estão salvos.
        ticker_id: Identificador do ativo (ex: petr4_sa, nvdc34_sa).

    Returns:
        Tupla de tensores (X_train, y_train, X_test, y_test).
    """
    logger.info("Carregando tensores do ativo '%s' em %s...", ticker_id, data_dir)
    X_train = np.load(os.path.join(data_dir, f"{ticker_id}_X_train.npy"))
    y_train = np.load(os.path.join(data_dir, f"{ticker_id}_y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, f"{ticker_id}_X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, f"{ticker_id}_y_test.npy"))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

    return X_train_t, y_train_t, X_test_t, y_test_t


def compute_financial_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas financeiras interpretáveis para séries temporais.

    Métrica de negócio principal (σ-tolerance):
        Porcentagem de predições que ficaram dentro de 0.5 desvios-padrão
        do preço observado. Critério de aceite da banca: ≥ 70%.

    Args:
        y_true: Valores reais (numpy array).
        y_pred: Predições do modelo (numpy array).

    Returns:
        Dicionário com RMSE, MAE, MAPE e sigma_tolerance_pct.
    """
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

    # σ-tolerance: critério de aceite do Datathon
    # Conta quantas predições ficaram dentro de 0.5 * std(y_true)
    sigma      = np.std(y_true)
    within_sigma = np.mean(np.abs(y_true - y_pred) <= 0.5 * sigma) * 100  # em %
    meets_criterion = within_sigma >= 70.0  # critério: ≥ 70%

    return {
        "mae":                  float(mae),
        "rmse":                 float(rmse),
        "mape":                 float(mape),
        "sigma_tolerance_pct": float(within_sigma),
        "sigma":               float(sigma),
        "meets_70pct_criterion": int(meets_criterion),  # 1 = aprovado, 0 = reprovado
    }


def get_git_sha() -> str:
    """Retorna o hash curto do commit atual para rastreabilidade (GAP 05)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def train_model(args: argparse.Namespace) -> None:
    """Executa o loop de treinamento LSTM com rastreamento completo via MLflow.

    Args:
        args: Argumentos CLI com hiperparâmetros e configurações do ativo.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("stock_prediction_lstm")

    # Run name derivado dinamicamente do ticker — sem hardcode (corrige bug de PETR4 fixo)
    ticker_upper = args.ticker_id.upper().replace("_", ".")
    run_name = f"Baseline_LSTM_{ticker_upper}"

    with mlflow.start_run(run_name=run_name):

        # === Tags Obrigatórias do Datathon (GAP 05) ===
        mlflow.set_tag("model_type", "regression_timeseries")
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("ticker", ticker_upper)
        mlflow.set_tag("owner", "grupo-05")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("risk_level", "high")  # Predição financeira = alto risco regulatório
        mlflow.set_tag("git_sha", get_git_sha())

        # Log de parâmetros
        mlflow.log_params(vars(args))

        X_train, y_train, X_test, y_test = load_data(args.data_dir, args.ticker_id)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset  = TensorDataset(X_test,  y_test)
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Usando hardware: %s", device)

        model = StockLSTM(
            input_size=X_train.shape[2],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        logger.info("Iniciando treinamento LSTM — %d epochs", args.epochs)

        for epoch in range(args.epochs):
            # --- Fase de Treino ---
            model.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # --- Fase de Validação com métricas financeiras ---
            model.eval()
            val_losses = []
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(batch_y.cpu().numpy().flatten())

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss   = float(np.mean(val_losses))
            fin_metrics    = compute_financial_metrics(
                np.array(all_targets), np.array(all_preds)
            )

            # === Log temporal por epoch (curva de aprendizagem visível no MLflow) ===
            mlflow.log_metric("train_mse",            avg_train_loss,                    step=epoch)
            mlflow.log_metric("val_mse",              avg_val_loss,                      step=epoch)
            mlflow.log_metric("val_rmse",             fin_metrics["rmse"],               step=epoch)
            mlflow.log_metric("val_mae",              fin_metrics["mae"],                step=epoch)
            mlflow.log_metric("val_mape",             fin_metrics["mape"],               step=epoch)
            mlflow.log_metric("sigma_tolerance_pct",  fin_metrics["sigma_tolerance_pct"], step=epoch)
            mlflow.log_metric("meets_70pct_criterion", fin_metrics["meets_70pct_criterion"], step=epoch)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                criterion_status = "✅ APROVADO" if fin_metrics["meets_70pct_criterion"] else "❌ ABAIXO"
                logger.info(
                    "Epoch [%d/%d] | Train MSE: %.6f | Val RMSE: %.6f | MAPE: %.2f%% | "
                    "σ-tolerance: %.1f%% %s",
                    epoch + 1, args.epochs,
                    avg_train_loss, fin_metrics["rmse"],
                    fin_metrics["mape"],
                    fin_metrics["sigma_tolerance_pct"], criterion_status,
                )

        logger.info("Treinamento concluído.")

        # === Sumário Final das Métricas de Negócio ===
        final_metrics = compute_financial_metrics(
            np.array(all_targets), np.array(all_preds)
        )
        mlflow.log_metrics({
            "final_rmse":                final_metrics["rmse"],
            "final_mae":                 final_metrics["mae"],
            "final_mape":                final_metrics["mape"],
            "final_sigma_tolerance_pct": final_metrics["sigma_tolerance_pct"],
            "final_sigma":               final_metrics["sigma"],
            "final_meets_70pct":         final_metrics["meets_70pct_criterion"],
        })
        status = "✅ APROVADO" if final_metrics["meets_70pct_criterion"] else "❌ NÃO APROVADO"
        logger.info(
            "=== RESULTADO FINAL [%s] === | σ-tolerance: %.1f%% (critério: ≥70%%) | "
            "RMSE: %.6f | MAPE: %.2f%%",
            status, final_metrics["sigma_tolerance_pct"],
            final_metrics["rmse"], final_metrics["mape"],
        )

        # === Persistência com assinatura automática (elimina WARNING do MLflow) ===
        input_example = X_train[:1].numpy()
        mlflow.pytorch.log_model(
            model,
            artifact_path="lstm_model",
            input_example=input_example,
        )
        logger.info("Modelo persistido no MLflow com signature e artefatos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Treinamento Corporativo LSTM — Datathon Fase 05")
    parser.add_argument("--data_dir",      type=str,   default="data/processed")
    parser.add_argument("--ticker_id",     type=str,   default="petr4_sa",
                        help="Identificador do ativo (ex: petr4_sa, nvdc34_sa)")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--hidden_size",   type=int,   default=50)
    parser.add_argument("--num_layers",    type=int,   default=2)
    parser.add_argument("--dropout",       type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()
    train_model(args)
