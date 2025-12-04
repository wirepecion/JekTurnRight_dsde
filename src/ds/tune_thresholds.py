import optuna
import pandas as pd
import numpy as np
import torch
import pickle
import json
from sklearn.metrics import fbeta_score

# Import from local modules
from src.ds.models import FloodLSTM
from src.ds.utils import CONFIG
from src.setting.config import MODEL_DIR, PROCESSED_DIR

def run_tuning():
    print(">>> Phase 2: Optuna Tuning")
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Artifacts
    with open(MODEL_DIR / "config.json", "r") as f: conf = json.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f: scaler = pickle.load(f)
    model = FloodLSTM(conf["input_dim"], conf["hidden_dim"], conf["num_layers"], conf["dropout"])
    model.load_state_dict(torch.load(MODEL_DIR /"best_model.pth", map_location=device))
    model.eval()

    # Load Test Data
    df = pd.read_csv(PROCESSED_DIR / "test_set.csv")
    # The test_set.csv is already preprocessed by create_tensors, so no need to call process_data again

    features = ['rainfall', 'total_report', 'API_30d', 'API_60d', 'API_90d', 'month_sin', 'month_cos', 'latitude', 'longitude']
    SEQ_LEN = CONFIG["SEQ_LEN"]

    # Cache Predictions
    print("     >>> Caching Predictions")
    y_true, y_probs, months = [], [], []

    for sub, g in df.groupby('subdistrict'):
        if len(g) <= SEQ_LEN: continue

        # SCALE DATAFRAME FIRST (Fixes Warning)
        g_scaled = g.copy()
        g_scaled[features] = scaler.transform(g[features])
        v_scaled = g_scaled[features].values

        t = g['target'].values
        m = g['month_timestamp'].values

        for i in range(len(g) - SEQ_LEN):
            X_ts = torch.FloatTensor(v_scaled[i : i+SEQ_LEN]).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(X_ts)).item()
            y_probs.append(prob)
            y_true.append(t[i+SEQ_LEN])
            months.append(m[i+SEQ_LEN])

    y_true, y_probs, months = np.array(y_true), np.array(y_probs), np.array(months)

    def objective(trial):
        th_wet = trial.suggest_float("wet", 0.2, 0.6)
        th_dry = trial.suggest_float("dry", 0.5, 0.9)
        thresholds = np.where((months >= 5) & (months <= 10), th_wet, th_dry)
        preds = (y_probs > thresholds).astype(int)
        return fbeta_score(y_true, preds, beta=2)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    with open(MODEL_DIR / "thresholds.json", "w") as f: json.dump(study.best_params, f)
    print("\u2705 Phase 2 Complete. Thresholds saved.")

if __name__ == "__main__":
    run_tuning()
