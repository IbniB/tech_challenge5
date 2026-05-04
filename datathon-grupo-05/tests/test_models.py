import torch
from src.models.baseline_lstm import StockLSTM

def test_stock_lstm_initialization():
    """Verifica se o modelo LSTM corporativo inicializa corretamente com N features."""
    model = StockLSTM(input_size=12, hidden_size=64, num_layers=2)
    assert model.hidden_size == 64
    assert model.num_layers == 2

def test_stock_lstm_forward_pass():
    """Verifica se o formato de saída do forward pass bate com (batch, 1) para um batch simulado."""
    model = StockLSTM(input_size=12, hidden_size=32, num_layers=1)
    
    # Simula um Tensor MLOps com [batch=2, seq_len=60, features=12]
    batch_size = 2
    dummy_input = torch.rand((batch_size, 60, 12))
    
    output = model(dummy_input)
    
    # A saída da predição para cada amostra no batch deve ser um escalar (tamanho 1)
    assert output.shape == (batch_size, 1), "Erro crítico no Output Shape do LSTM. Isso quebrará a API de Inferência."
