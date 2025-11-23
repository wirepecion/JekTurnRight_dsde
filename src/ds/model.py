import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class FloodLSTM(nn.Module, PyTorchModelHubMixin):
    """
    Standard LSTM for Time Series Forecasting.
    Decoupled from training logic for portability.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the LAST time step
        return self.fc(out[:, -1, :]).squeeze()