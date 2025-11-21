import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from huggingface_hub import PyTorchModelHubMixin
import pickle
import json
import os

# --- CONFIGURATION ---
CONFIG = {
    "SEQ_LEN": 30,
    "BATCH_SIZE": 64,
    "HIDDEN_DIM": 64,
    "LAYERS": 2,
    "DROPOUT": 0.4,
    "EPOCHS": 50,
    "PATIENCE": 7,
    "LR": 1e-3,
    "WD": 1e-5,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- UTILITIES ---
class EarlyStopping:
    def __init__(self, patience=7, path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# --- MODEL ---
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

# --- ETL ---
def process_data(df):
    df = df.copy()
    df['date'] = pd.to_datetime({'year': df['year_timestamp'], 'month': df['month_timestamp'], 'day': df['days_timestamp']})
    df = df.sort_values(['subdistrict', 'date']).reset_index(drop=True)
    
    # Physics Features
    for w in [30, 60, 90]:
        col = f'API_{w}d'
        df[col] = df.groupby('subdistrict')['water'].transform(lambda x: x.rolling(w, min_periods=1).mean()).bfill()
    
    # Seasonality Features
    df['month_sin'] = np.sin(2 * np.pi * df['month_timestamp'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_timestamp'] / 12)
    
    df['target'] = (df['number_of_report_flood'] > 0).astype(int)
    return df

def create_tensors(df):
    cutoff = pd.Timestamp("2024-01-01")
    train_df = df[df['date'] < cutoff].copy()
    test_df = df[df['date'] >= cutoff].copy()
    
    # Export Test Set for Optuna
    test_df.to_csv("test_set.csv", index=False)
    
    features = ['water', 'total_report', 'API_30d', 'API_60d', 'API_90d', 'month_sin', 'month_cos', 'latitude', 'longitude']
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)

    def _slide(sub_df):
        X, y = [], []
        for _, g in sub_df.groupby('subdistrict'):
            v = g[features].values
            t = g['target'].values
            if len(v) <= CONFIG["SEQ_LEN"]: continue
            for i in range(len(v) - CONFIG["SEQ_LEN"]):
                X.append(v[i : i+CONFIG["SEQ_LEN"]])
                y.append(t[i+CONFIG["SEQ_LEN"]])
        return np.array(X), np.array(y)

    X_tr, y_tr = _slide(train_df)
    X_te, y_te = _slide(test_df)
    pos_weight = (len(y_tr) - sum(y_tr)) / (sum(y_tr) + 1e-5)
    return (X_tr, y_tr), (X_te, y_te), pos_weight, len(features)

# --- EXECUTION ---
if __name__ == "__main__":
    print(">>> 🏋️ Phase 1: Training Started...")
    # LOAD YOUR DATA HERE
    df_full = pd.read_csv("water_and_sun_report.csv") 
    df_clean = process_data(df_full)
    (X_tr, y_tr), (X_te, y_te), p_weight, input_dim = create_tensors(df_clean)

    # Dataset Wrapper
    class DS(Dataset):
        def __init__(self, X, y): self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    tr_loader = DataLoader(DS(X_tr, y_tr), batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    te_loader = DataLoader(DS(X_te, y_te), batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    model = FloodLSTM(input_dim, CONFIG["HIDDEN_DIM"], CONFIG["LAYERS"], CONFIG["DROPOUT"]).to(CONFIG["DEVICE"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WD"])
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p_weight).to(CONFIG["DEVICE"]))
    stop = EarlyStopping(patience=CONFIG["PATIENCE"])

    for ep in range(CONFIG["EPOCHS"]):
        model.train()
        t_loss = 0
        for X, y in tr_loader:
            X, y = X.to(CONFIG["DEVICE"]), y.to(CONFIG["DEVICE"])
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            t_loss += loss.item()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X, y in te_loader:
                X, y = X.to(CONFIG["DEVICE"]), y.to(CONFIG["DEVICE"])
                v_loss += crit(model(X), y).item()
        
        print(f"Epoch {ep+1:02d}: Train {t_loss/len(tr_loader):.4f} | Val {v_loss/len(te_loader):.4f}")
        stop(v_loss/len(te_loader), model)
        if stop.early_stop: break

    # Save
    config = {"input_dim": input_dim, "hidden_dim": CONFIG["HIDDEN_DIM"], "num_layers": CONFIG["LAYERS"], "dropout": CONFIG["DROPOUT"]}
    with open("config.json", "w") as f: json.dump(config, f)
    torch.save(model.state_dict(), "pytorch_model.bin")
    print("✅ Phase 1 Complete.")