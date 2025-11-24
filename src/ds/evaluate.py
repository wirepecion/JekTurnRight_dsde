"""
src/ds/evaluate.py
------------------
Loads the trained model and evaluates it against the Test Set.
Generates a Classification Report and Confusion Matrix.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import json
import logging
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Import custom modules
from src.ds.dataset import FloodLSTMDataset
from src.ds.model import FloodLSTM

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Evaluator")

def evaluate():
    logger.info(">>> Phase 1: Loading Artifacts...")
    
    # 1. Load Configuration
    try:
        with open("model_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("model_config.json not found. Did you train the model?")
        sys.exit(1)

    # 2. Load Data (Same logic as training, but we focus on Test Set)
    parquet_path = "data/processed/flood_training_data_spark"
    full_dataset = FloodLSTMDataset(parquet_path, sequence_length=30, target_col='number_of_report_flood')
    
    # Re-create the split (Must match training split logic!)
    # In production, we usually save the test indices to a file to ensure exact match.
    # Here we assume random_split with fixed seed works for POC.
    torch.manual_seed(42) # Ensure reproducibility
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_ds = random_split(full_dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FloodLSTM(
        config["input_dim"], 
        config["hidden_dim"], 
        config["num_layers"], 
        config["dropout"]
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        logger.error("best_model.pth not found.")
        sys.exit(1)

    # 4. Inference Loop
    logger.info(">>> Phase 2: Running Inference on Test Set...")
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            
            # Forward Pass
            logits = model(X)
            probs = torch.sigmoid(logits) # Convert logits to probability (0-1)
            
            # Binarize (Flood if prob > 0.5)
            # Senior Note: In real life, we