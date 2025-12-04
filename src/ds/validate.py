import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from src.setting.config import PROCESSED_DIR

# Import model class
from src.ds.models import FloodLSTM

# ---  VALIDATION ENGINE ---

def run_validation(test_csv_path= PROCESSED_DIR / "test_set.csv", repo_id="sirasira/flood-lstm-v1"):
    print(f""">>>  Starting Backtest Validation on {test_csv_path}""")

    # A. Load Model & Artifacts
    print("     >>> Syncing artifacts from Hugging Face Hub")
    model, scaler, thresh_config = FloodLSTM.load_from_hub(repo_id=repo_id, device="cuda" if torch.cuda.is_available() else "cpu")

    # B. Load Data
    df = pd.read_csv(test_csv_path)

    # --- SENIOR FIX: Handle "Processed" vs "Raw" Data ---
    # 1. Date Processing
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Reconstruct from timestamp parts if 'date' doesn't exist
        df['date'] = pd.to_datetime({'year': df['year_timestamp'], 'month': df['month_timestamp'], 'day': df['days_timestamp']})

    df = df.sort_values(['subdistrict', 'date']).reset_index(drop=True)

    # 2. Feature Engineering (Idempotent - safe to run twice)
    # If API columns exist, this just overwrites them (safe). If missing, it creates them.
    for w in [30, 60, 90]:
        col = f'API_{w}d'
        df[col] = df.groupby('subdistrict')['rainfall'].transform(lambda x: x.rolling(w, min_periods=1).mean()).bfill()

    df['month_timestamp'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_timestamp'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_timestamp'] / 12)

    # 3. TARGET RESOLUTION (The Fix)
    if 'target' in df.columns:
        print("     >>> Found pre-calculated 'target' column.")
        # Ensure it's int
        df['target'] = df['target'].astype(int)
    elif 'number_of_report_flood' in df.columns:
        print("     >>> Calculating 'target' from 'number_of_report_flood'...")
        df['target'] = (df['number_of_report_flood'] > 0).astype(int)
    else:
        print("❌ CRITICAL ERROR: No Ground Truth found.")
        print(f"   Columns found: {list(df.columns)}")
        print("   Please verify your input CSV contains 'target' or 'number_of_report_flood'.")
        return

    # C. Prediction Loop
    features = ['rainfall', 'total_report', 'API_30d', 'API_60d', 'API_90d', 'month_sin', 'month_cos', 'latitude', 'longitude']
    SEQ_LEN = 30 # Hardcoded as in the original script, or can be imported from data_utils.CONFIG["SEQ_LEN"]

    y_true_all = []
    y_pred_all = []

    print("     >>> Running Inference...")
    for sub, g in df.groupby('subdistrict'):
        if len(g) <= SEQ_LEN: continue

        # Scale
        g_scaled = g.copy()
        g_scaled[features] = scaler.transform(g[features])
        vals = g_scaled[features].values
        targets = g['target'].values
        months = g['month_timestamp'].values

        for i in range(len(g) - SEQ_LEN):
            # Input Tensor
            X = torch.FloatTensor(vals[i : i+SEQ_LEN]).unsqueeze(0)

            # Model Prediction
            with torch.no_grad():
                prob = torch.sigmoid(model(X)).item()

            # Dynamic Threshold Application
            # We look at the MONTH of the prediction target (i + SEQ_LEN)
            target_month = months[i+SEQ_LEN]
            is_wet = 5 <= target_month <= 10
            limit = thresh_config["wet"] if is_wet else thresh_config["dry"]

            pred_label = 1 if prob > limit else 0

            y_pred_all.append(pred_label)
            y_true_all.append(targets[i+SEQ_LEN])

    # D. Generate Report
    print("\n" + "="*60)
    print("     >>> FINAL BACKTEST REPORT (2024 Data)")
    print("="*60)

    if len(y_true_all) == 0:
        print("⚠️ No predictions made. Check sequence length vs data size.")
        return

    print(classification_report(y_true_all, y_pred_all, target_names=["No Flood", "Flood"])) # Removed Thai characters to avoid font issues in stdout.

    cm = confusion_matrix(y_true_all, y_pred_all)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"   True Negatives (Safe):  {tn}")
    print(f"   False Positives (Alarm): {fp}")
    print(f"   False Negatives (Missed): {fn}  <-- Must be low")
    print(f"   True Positives (Caught):  {tp}")

    f2 = fbeta_score(y_true_all, y_pred_all, beta=2)
    print(f"\n📰 Safety Score (F2): {f2:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_validation()
