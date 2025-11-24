"""
src/ds/inference_hf.py
----------------------
Hybrid Inference Engine.
1. LOADS data from local Spark Output.
2. DOWNLOADS model from Hugging Face Hub.
3. PREDICTS using Optuna Thresholds.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import pickle
import logging
import sys
from huggingface_hub import hf_hub_download
from src.ds.dataset import FloodLSTMDataset # Reuse your dataset logic

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("HF_Inference")

# --- 1. MODEL ARCHITECTURE (Must match Colab exactly) ---
class FloodLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).squeeze()

def run_inference(repo_id="sirasira/flood-lstm-v1"):
    logger.info(f">>> ☁️ Syncing with Hugging Face: {repo_id}...")
    
    try:
        # A. Download Artifacts
        scaler_path = hf_hub_download(repo_id=repo_id, filename="scaler.pkl")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        thresh_path = hf_hub_download(repo_id=repo_id, filename="thresholds.json")
        
        # B. Load Artifacts
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        with open(config_path, "r") as f: config = json.load(f)
        with open(thresh_path, "r") as f: thresholds = json.load(f)
        
        logger.info("✅ Artifacts Loaded.")
        
    except Exception as e:
        logger.error(f"Failed to download from HF: {e}")
        return

    # C. Load Data (From Spark)
    parquet_path = "data/processed/flood_training_data_spark"
    logger.info(f">>> 📂 Loading Spark Data from {parquet_path}...")
    
    # We use your Dataset class to handle the sliding window & scaling
    # CRITICAL: We pass the HF scaler to ensure consistency with Colab training
    dataset = FloodLSTMDataset(parquet_path, sequence_length=30, scaler=scaler)
    
    # D. Init Model
    device = torch.device("cpu")
    model = FloodLSTM(config["input_dim"], config["hidden_dim"], config["num_layers"], config["dropout"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # E. Prediction Loop
    logger.info(">>> 🔮 Running Inference...")
    
    # We need to map predictions back to dates/stations.
    # The Dataset class shuffles/groups data, so we iterate manually here for control.
    # (Re-implementing simplified sliding window on the dataframe directly for inference transparency)
    
    df = dataset.df # The dataframe inside the dataset class (already sorted/scaled)
    seq_len = 30
    results = []
    
    # Group by Station
    for station, g in df.groupby('station_code'):
        if len(g) <= seq_len: continue
        
        # We need raw values for prediction (already scaled in dataset.df)
        feature_cols = dataset.feature_cols
        vals = g[feature_cols].values
        dates = g['date'].values
        months = g['month_timestamp'].values # Ensure this column exists from Spark
        
        # Predict for the LAST available day (Forecasting tomorrow)
        # Or predict for the whole sequence if backtesting
        
        # Let's predict the latest day for this station
        last_seq = vals[-seq_len:]
        target_date = dates[-1]
        target_month = months[-1]
        
        X = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob = torch.sigmoid(model(X)).item()
            
        # Dynamic Threshold
        # Note: Your Colab uses month 5-10 as Wet Season
        # Spark 'month_timestamp' is 1-12.
        is_wet = 5 <= target_month <= 10
        limit = thresholds["wet"] if is_wet else thresholds["dry"]
        
        status = "🚨 FLOOD" if prob > limit else "✅ Safe"
        
        results.append({
            "date": str(target_date),
            "station": station,
            "risk_score": prob,
            "threshold": limit,
            "status": status
        })
        
    # F. Save Results
    res_df = pd.DataFrame(results)
    output_file = "latest_forecast.csv"
    res_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(f"📊 FORECAST REPORT ({len(res_df)} Stations)")
    print("="*60)
    print(res_df.sort_values("risk_score", ascending=False).head(10))
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    run_inference()