import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

# --- Project Imports ---
# Ensure these paths exist in your project structure
from src.ds.models import FloodLSTM
from src.ds.utils import CONFIG
from src.setting.config import PROCESSED_DIR, MODEL_DIR

# --- 1. Metrics & Artifact Generation ---
def save_performance_artifacts(y_true, y_pred, output_dir):
    """
    Generates F2 Score, Classification Report, and Confusion Matrix Heatmap.
    Saves them to the specified output_dir.
    """
    print("    >>> 📊 Generating Performance Artifacts (Focus: F2 Score)...")
    
    # A. Calculate F2 Score (The "Safety First" Metric)
    # beta=2 weights Recall higher than Precision. 
    # We want to minimize False Negatives (missing a flood).
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    print(f"        - 🏆 Global F2 Score: {f2:.4f}")

    # B. Classification Report (Text)
    report = classification_report(y_true, y_pred, target_names=['Safe', 'Flood'])
    
    # Custom Header for the report
    full_report = (
        f"Model Evaluation Report\n"
        f"=======================\n"
        f"F2 Score (Recall-Weighted): {f2:.4f}\n"
        f"-----------------------\n\n"
        f"{report}"
    )

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"        - Report saved to: {report_path}")

    # C. Confusion Matrix (Image)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # 'd' format is for integers (counts).
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Safe', 'Pred Flood'], 
                yticklabels=['Actual Safe', 'Actual Flood'])
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Flood Prediction Matrix (F2={f2:.2f})')
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close() # Good practice: close plot to free memory
    print(f"        - Matrix saved to: {cm_path}")

# --- 2. Main Execution Loop ---
def run_forecast_and_evaluate(csv_path, repo_id="sirasira/flood-lstm-v1", burn_in=90):
    print(f""">>> ✨ Starting Forecast & Evaluation on {csv_path}...""")

    # --- A. Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    >>> Device: {device}")
    model, scaler, thresh = FloodLSTM.load_from_hub(repo_id=repo_id, device=device)

    # --- B. Load & ETL Data ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return f"❌ Error: Input file '{csv_path}' not found."

    # Validate Ground Truth existence
    # Note: Ensure your CSV has this column name!
    TARGET_COL = 'target' 
    has_ground_truth = TARGET_COL in df.columns

    # Date Sorting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['subdistrict', 'date']).reset_index(drop=True)

    # Feature Engineering (Vectorized for speed)
    print("    >>> Calculating Physics Features...")
    for w in [30, 60, 90]:
        col = f'API_{w}d'
        # Groupby transform is much faster than iterating
        df[col] = df.groupby('subdistrict')['rainfall'].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        ).bfill()

    # Seasonality
    df['month_timestamp'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_timestamp'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_timestamp'] / 12)

    # --- C. Prediction Loop ---
    features = ['rainfall', 'total_report', 'API_30d', 'API_60d', 'API_90d', 'month_sin', 'month_cos', 'latitude', 'longitude']
    SEQ_LEN = CONFIG["SEQ_LEN"]
    
    results = []
    y_true_all = [] # Accumulate all actuals
    y_pred_all = [] # Accumulate all predictions

    print("    🔮 Running Inference...")
    
    # Group by location
    for sub, g in df.groupby('subdistrict'):
        if len(g) <= burn_in:
            continue

        # Scale data
        g_scaled = g.copy()
        g_scaled[features] = scaler.transform(g[features])
        vals = g_scaled[features].values
        dates = g['date'].values
        months = g['month_timestamp'].values
        
        # Ground Truth extraction
        actuals = g[TARGET_COL].values if has_ground_truth else None

        # Time-step iteration
        # (Optimization Note: In V2, we should use a DataLoader for batching)
        for i in range(burn_in, len(g)):
            seq_start = i - SEQ_LEN
            if seq_start < 0: continue

            # Create Input Tensor
            X_ts = torch.FloatTensor(vals[seq_start:i]).unsqueeze(0).to(device)
            
            # Forward Pass
            with torch.no_grad():
                prob = torch.sigmoid(model(X_ts)).item()

            # Dynamic Threshold Logic
            current_month = months[i]
            is_wet = 5 <= current_month <= 10
            limit = thresh["wet"] if is_wet else thresh["dry"]
            
            # Determine Prediction Class
            pred_class = 1 if prob > limit else 0

            # Store Metrics
            if has_ground_truth:
                y_true_all.append(actuals[i])
                y_pred_all.append(pred_class)

            results.append({
                "date": str(pd.Timestamp(dates[i]).date()),
                "location": sub,
                "risk_score": f"{prob:.2%}",
                "status": "⚠️ FLOOD" if pred_class == 1 else "✅ Safe",
                "actual": actuals[i] if has_ground_truth else "N/A"
            })

    # --- D. Export Metrics ---
    if has_ground_truth and len(y_true_all) > 0:
        save_performance_artifacts(y_true_all, y_pred_all, MODEL_DIR)
    else:
        print("    ⚠️ No ground truth column found or empty predictions. Skipping metrics.")

    return pd.DataFrame(results)

# --- 3. Entry Point ---
if __name__ == "__main__":
    # Ensure this matches your file system
    INPUT_FILE = PROCESSED_DIR / "test_set.csv"
    OUTPUT_FILE = MODEL_DIR / "2024_forecast.csv"

    # Run the pipeline
    df_result = run_forecast_and_evaluate(INPUT_FILE)

    # Save CSV results
    if isinstance(df_result, pd.DataFrame) and not df_result.empty:
        print(f"\n>>> ✅ Forecast Complete! Saving to {OUTPUT_FILE}")
        df_result.to_csv(OUTPUT_FILE, index=False)
    else:
        print(">>> ⚠️ No predictions made.")