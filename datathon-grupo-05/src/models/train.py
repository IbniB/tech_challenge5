import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# MLOps Governança
import mlflow
import mlflow.pytorch

from baseline_lstm import StockLSTM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_dir):
    """
    Restaura os Data Contracts armazenados na Fase A (Engenharia) e traciona 
    na memória da Rede convertendo diretamente do pipeline NumPy pra tensores imutáveis.
    """
    logger.info(f"Carregando tensores de {data_dir}...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Converter de forma hard-typed para evitar Type Drifts (Gap Governança)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train_t, y_train_t, X_test_t, y_test_t

def train_model(args):
    # Setup MLflow - Resolução do "Gap 05" (Ausência de rastreamento de Pesos e Treinamento)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("stock_prediction_lstm")
    
    with mlflow.start_run(run_name=f"Baseline_LSTM_{args.run_id}"):
        
        # Enfileirando MLOps Logs - Sem "Modelos Órfãos" a partir de hoje
        mlflow.log_params(vars(args))
        
        X_train, y_train, X_test, y_test = load_data(args.data_dir)
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando hardware nativo: {device}")
        
        model = StockLSTM(
            input_size=1, 
            hidden_size=args.hidden_size, 
            num_layers=args.num_layers, 
            dropout=args.dropout
        ).to(device)
        
        # Função de Perda Clássica para Séries (Mean Squared Error)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        logger.info(f"Acionando Loop Deep Learning (LSTM) - Epocas: {args.epochs}")
        for epoch in range(args.epochs):
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
                
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
                    
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Watchdog do Treino
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
            # Log de Observabilidade temporal - Permite a banca ver as curvas de aprendizagem no MLflow UI
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            
        logger.info("Fase Computacional Concluída.")
        
        # Acoplamento de Artefatos no MLflow Central Repository
        mlflow.pytorch.log_model(model, "lstm_model")
        logger.info("Modelo de Inferência persistido via MLflow de forma transparente.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Treinamento Corporativo LSTM da Mesa Financeira")
    parser.add_argument("--data_dir", type=str, default="../../data/processed")
    parser.add_argument("--run_id", type=str, default="PETR4")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    args = parser.parse_args()
    train_model(args)
