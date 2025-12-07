import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# --- Import your specific modules ---
from src.ds.models import FloodLSTM
from src.ds.utils import create_tensors, DS, CONFIG # Assuming your utils are here
from src.setting.config import PROCESSED_DIR, MODEL_DIR

# --- 1. Global Setup (Load Data Once) ---
# We load data outside the objective function to avoid reading CSVs 100 times.
def prepare_tuning_data():
    print(">>> 📥 Preparing Data for Optuna...")
    
    # 1. Load Raw Data
    df = pd.read_csv(PROCESSED_DIR / "clean_flood_data.csv") # Check your filename
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['subdistrict', 'date']).reset_index(drop=True)
    
    # 2. Use your existing util to process features/scaling
    # Note: We ignore the (X_te, y_te) returned here. 
    # We NEVER tune on the Test set.
    (X_full, y_full), _, pos_weight, input_dim = create_tensors(df)
    
    # 3. Create a Validation Split for Tuning
    # We take 20% of the training data to be the "Judge" for Optuna
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"    - Training Set: {X_train.shape}")
    print(f"    - Tuning Val Set: {X_val.shape}")
    print(f"    - Positive Weight: {pos_weight:.2f}")
    
    return X_train, y_train, X_val, y_val, pos_weight, input_dim

# Load data into memory immediately
try:
    X_TR, Y_TR, X_VAL, Y_VAL, POS_WEIGHT, INPUT_DIM = prepare_tuning_data()
    # Move static tensors to device if small enough, otherwise keep in CPU and move batches
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    POS_WEIGHT_TENSOR = torch.tensor(POS_WEIGHT).to(DEVICE)
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# --- 2. The Optuna Objective ---
def objective(trial):
    # --- A. Suggest Hyperparameters ---
    # Model Architecture
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2) # 3 layers is usually unstable for small data
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    
    # Training Config
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # --- B. Setup DataLoaders for this Trial ---
    train_ds = DS(X_TR, Y_TR)
    val_ds = DS(X_VAL, Y_VAL)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # --- C. Initialize Model ---
    model = FloodLSTM(INPUT_DIM, hidden_dim, num_layers, dropout).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Use BCEWithLogitsLoss because it's numerically more stable than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_TENSOR)
    
    # --- D. Training Loop (Shortened for Tuning) ---
    # We don't need 50 epochs to know if a config is trash. 10-15 is enough.
    TUNING_EPOCHS = 15
    
    for epoch in range(TUNING_EPOCHS):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(X_b)
            # NEW (Fix) - Force model output to be [Batch, 1]
            loss = criterion(out.view(-1, 1), y_b.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
        # --- E. Validation & Pruning ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                out = model(X_v)
                loss = criterion(out, y_v.unsqueeze(1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Report to Optuna
        trial.report(avg_val_loss, epoch)
        
        # ✂️ PRUNING: Kill bad trials early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_val_loss

# --- 3. Execution ---
if __name__ == "__main__":
    print(f">>> 🚀 Starting Hyperparameter Tuning on {DEVICE}...")
    
    # Use MedianPruner (Stops a trial if it's worse than the median of previous trials)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    # Run 20-50 trials depending on your GPU time
    study.optimize(objective, n_trials=30)
    
    print("\n>>> 🏆 Tuning Complete!")
    print(f"    Best Validation Loss: {study.best_value:.4f}")
    print(f"    Best Params: {study.best_params}")
    
    # Save best params to JSON so train.py can use them
    out_path = MODEL_DIR / "config.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f)
    
    print(f"    ✅ Saved configuration to {out_path}")