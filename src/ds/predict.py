import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import os
from huggingface_hub import hf_hub_download
import numpy as np

from model import FloodLSTM
from data_utils import CONFIG

def run_forecast(csv_path, repo_id="sirasira/flood-lstm-v1", burn_in=90):
    print(f""">>> \u2728 Starting Forecast on {csv_path}...""")

    # --- A. Download Resources ---
    print("     >>> Syncing with Hugging Face Hub...")
    try:
        files = ["scaler.pkl", "config.json", "pytorch_model.bin", "thresholds.json"]
        paths = {}
        for f in files:
            paths[f] = hf_hub_download(repo_id=repo_id, filename=f)
    except Exception as e:
        return f"\u274C Error downloading resources: {e}"

    # --- B. Load Artifacts ---
    with open(paths["scaler.pkl"], "rb") as f: scaler = pickle.load(f)
    with open(paths["config.json"], "r") as f: conf = json.load(f)
    with open(paths["thresholds.json"], "r") as f: thresh = json.load(f)

    # Load Model
    device = torch.device("cpu")
    model = FloodLSTM(conf["input_dim"], conf["hidden_dim"], conf["num_layers"], conf["dropout"])
    model.load_state_dict(torch.load(paths["pytorch_model.bin"], map_location=device))
    model.eval()

    # --- C. Load & ETL Data ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return f"\u274C Error: Input file '{csv_path}' not found."

    # Date & Sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['subdistrict', 'date']).reset_index(drop=True)

    # Physics Features (API)
    print("     >>> Calculating Physics Features...")
    for w in [30, 60, 90]:
        col = f'API_{w}d'
        df[col] = df.groupby('subdistrict')['rainfall'].transform(lambda x: x.rolling(w, min_periods=1).mean()).bfill()

    # Seasonality Features
    df['month_timestamp'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_timestamp'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_timestamp'] / 12)

    # --- D. Prediction Loop ---
    features = ['rainfall', 'total_report', 'API_30d', 'API_60d', 'API_90d', 'month_sin', 'month_cos', 'latitude', 'longitude']
    SEQ_LEN = CONFIG["SEQ_LEN"]
    results = []

    print("    🔮 Running Inference...")
    for sub, g in df.groupby('subdistrict'):
        if len(g) <= burn_in:
            print(f"    \u26A0\ufe0f Skipping {sub}: Not enough data for burn-in ({len(g)} rows).")
            continue

        # Scale Group
        g_scaled = g.copy()
        g_scaled[features] = scaler.transform(g[features])
        vals = g_scaled[features].values
        dates = g['date'].values
        months = g['month_timestamp'].values

        # Start predicting AFTER the burn-in period
        for i in range(burn_in, len(g)):
            seq_start = i - SEQ_LEN
            if seq_start < 0: continue

            # Create Tensor
            X_ts = torch.FloatTensor(vals[seq_start:i]).unsqueeze(0).to(device)

            # Forward Pass
            with torch.no_grad():
                prob = torch.sigmoid(model(X_ts)).item()

            # Apply Dynamic Threshold
            current_month = months[i]
            is_wet = 5 <= current_month <= 10
            limit = thresh["wet"] if is_wet else thresh["dry"]

            results.append({
                "date": str(pd.Timestamp(dates[i]).date()),
                "location": sub,
                "risk_score": f"{prob:.2%}",
                "status": "\u26A0\ufe0f FLOOD" if prob > limit else "\u2705 Safe",
                "threshold_used": f"{limit:.3f}"
            })

    return pd.DataFrame(results)

if __name__ == "__main__":

    # Change this to your input filename
    INPUT_FILE = "/content/test_set.csv"
    OUTPUT_FILE = "2024_forecast.csv"

    df_result = run_forecast(INPUT_FILE)

    if isinstance(df_result, pd.DataFrame):
        if not df_result.empty:
            print(f"\n>>> \u2705 Forecast Complete! Saving to {OUTPUT_FILE}")
            print(df_result.tail()) 
            df_result.to_csv(OUTPUT_FILE, index=False)
        else:
            print(">>> \u26A0\ufe0f No predictions made. Check input data length.")
    else:
        print(df_result) 
