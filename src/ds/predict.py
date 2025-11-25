import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import os
from huggingface_hub import hf_hub_download
import numpy as np

from src.ds.models import FloodLSTM
from src.ds.utils import CONFIG
from src.setting.config import PROCESSED_DIR, MODEL_DIR

def run_forecast(csv_path, repo_id="sirasira/flood-lstm-v1", burn_in=90):
    print(f""">>> \u2728 Starting Forecast on {csv_path}...""")

    # --- A. Load Model ---
    print("     >>> Load model from Hugging Face Hub")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, thresh = FloodLSTM.load_from_hub(repo_id=repo_id, device=device)

    # --- B. Load & ETL Data ---
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

    # --- C. Prediction Loop ---
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
    INPUT_FILE = PROCESSED_DIR / "test_set.csv"
    OUTPUT_FILE = MODEL_DIR / "2024_forecast.csv"

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
