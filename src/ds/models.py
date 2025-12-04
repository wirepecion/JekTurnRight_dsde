import torch
import torch.nn as nn
import pickle
import json
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

class FloodLSTM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).squeeze()
    
    @property
    def hub_model_id(self) -> str:
        return "sirasira/flood-lstm-v1"
    
    @classmethod
    def load_from_hub(cls, repo_id: str | None = None, device: str = "cpu"):
        repo_id = repo_id or "sirasira/flood-lstm-v1"

        # 1) model via mixin
        model = cls.from_pretrained(repo_id, map_location=device)
        model.to(device)
        model.eval()

        # 2) scaler
        scaler_path = hf_hub_download(repo_id, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # 3) thresholds
        thresholds_path = hf_hub_download(repo_id, "thresholds.json")
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)

        return model, scaler, thresholds

    


