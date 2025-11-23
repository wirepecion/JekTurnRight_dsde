"""
src/ds/train.py
---------------
Trains the FloodLSTM model using Spark-generated Parquet files.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import json
import pickle
import logging
import sys

# Custom Imports
from src.ds.dataset import FloodLSTMDataset
from src.ds.model import FloodLSTM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FloodTrainer")

# --- CONFIGURATION ---
CONFIG = {
    "SEQ_LEN": 30,
    "BATCH_SIZE": 64,
    "HIDDEN_DIM": 64,
    "LAYERS": 2,
    "DROPOUT": 0.4,
    "EPOCHS": 30,       # Reduced epochs for faster testing
    "PATIENCE": 5,
    "LR": 1e-3,
    "WD": 1e-5,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

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

def train():
    logger.info(">>> Phase 1: Loading Data...")
    
    # Point to the FOLDER created by Spark
    parquet_path = "data/processed/flood_training_data_spark"
    
    try:
        # Initialize Dataset
        full_dataset = FloodLSTMDataset(parquet_path, sequence_length=CONFIG["SEQ_LEN"])
        
        # Save Scaler (Critical for Inference!)
        scaler = full_dataset.get_scaler()
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        logger.info("✅ Saved scaler.pkl")
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Split Train/Test (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    tr_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    te_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    # Init Model
    input_dim = full_dataset.get_input_dim()
    logger.info(f"Model Input Dimension: {input_dim}")
    
    model = FloodLSTM(input_dim, CONFIG["HIDDEN_DIM"], CONFIG["LAYERS"], CONFIG["DROPOUT"]).to(CONFIG["DEVICE"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WD"])
    
    # Loss Function
    # Use 'pos_weight' here if floods are extremely rare (e.g., < 5%)
    # pos_weight = torch.tensor([5.0]).to(CONFIG["DEVICE"])
    # crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit = nn.BCEWithLogitsLoss()
    
    stop = EarlyStopping(patience=CONFIG["PATIENCE"])

    logger.info(">>> Phase 2: Training Loop...")
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

        avg_t = t_loss / len(tr_loader)
        avg_v = v_loss / len(te_loader)
        print(f"Epoch {ep+1:02d}: Train {avg_t:.4f} | Val {avg_v:.4f}")
        
        stop(avg_v, model)
        if stop.early_stop:
            logger.info("Early stopping triggered.")
            break

    # Save Configuration
    model_config = {
        "input_dim": input_dim, 
        "hidden_dim": CONFIG["HIDDEN_DIM"], 
        "num_layers": CONFIG["LAYERS"], 
        "dropout": CONFIG["DROPOUT"]
    }
    with open("model_config.json", "w") as f: json.dump(model_config, f)
    logger.info("✅ Training Complete. Artifacts saved.")

if __name__ == "__main__":
    train()