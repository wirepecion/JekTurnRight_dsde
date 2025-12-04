import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.setting.config import PROCESSED_DIR, MODEL_DIR
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
    def __init__(self, patience=7, path=MODEL_DIR / 'best_model.pth'):
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

def create_tensors(df):
    cutoff = pd.Timestamp("2024-01-01")
    train_df = df[df['date'] < cutoff].copy()
    test_df = df[df['date'] >= cutoff].copy()

    # Export Test Set for Optuna
    test_df.to_csv(PROCESSED_DIR / "test_set.csv", index=False)

    features = ['rainfall', 'total_report', 'API_30d', 'API_60d', 'API_90d', 'month_sin', 'month_cos', 'latitude', 'longitude']
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    with open(MODEL_DIR / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)

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

# Dataset Wrapper
class DS(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]
