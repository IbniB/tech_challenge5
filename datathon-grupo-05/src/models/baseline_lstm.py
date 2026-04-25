import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """
    Núcleo de Deep Learning corporativo para Séries Temporais Financeiras.
    Esta rede é designada para tratar o sumário de transações do Datathon (Janelamentos).
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # O batch_first=True é mandatório pois nossos tensores entram na ordem MLOps de (Batch, Timesteps, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Decodificador linear para achatar os pesos de volta para o valor Scaled do Pregão
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Injeção no formato (batch_size, sequence_length, input_size)
        # Inicialização protegida contra sujeira na memória usando instanciamento limpo
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Propagação interna no cérebro
        out, _ = self.lstm(x, (h0, c0))
        
        # MLOps: Queremos prever apenas o futuro, logo pegamos APENAS o frame final da série temporal processada (-1)
        out = out[:, -1, :]
        
        # Ajuste de Dimensionalidade Final
        out = self.fc(out)
        
        return out
