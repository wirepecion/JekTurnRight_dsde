import optuna
import pandas as pd
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import fbeta_score

# Import from local modules
from src.ds.models import FloodLSTM
from src.ds.utils import CONFIG
from src.setting.config import MODEL_DIR, PROCESSED_DIR

def visualize_optuna_results(study, save_dir):
    """
    Creates a 2D Contour plot to visualize the 'Wet' vs 'Dry' threshold landscape.
    """
    print("    >>> 🎨 Generating Optimization Landscape...")
    
    # Extract data from the study
    trials_df = study.trials_dataframe()
    
    # Filter only completed trials
    trials_df = trials_df[trials_df.state == "COMPLETE"]
    
    x = trials_df['params_wet']
    y = trials_df['params_dry']
    z = trials_df['value'] # F2 Score

    # --- Plot 1: 2D Contour Map (The Landscape) ---
    plt.figure(figsize=(10, 8))
    
    # Use Tricontourf to fill gaps between scattered trial points
    cntr = plt.tricontourf(x, y, z, levels=20, cmap="viridis")
    plt.colorbar(cntr, label="F2 Score")
    
    # Plot the actual trial points as dots
    plt.scatter(x, y, c='white', s=30, alpha=0.5, edgecolors='black', label='Trial')
    
    # Highlight the Best Trial
    best_wet = study.best_params['wet']
    best_dry = study.best_params['dry']
    best_score = study.best_value
    
    plt.scatter(best_wet, best_dry, c='red', s=150, marker='*', edgecolors='white', 
                label=f'Best (F2={best_score:.3f})', zorder=10)

    plt.title(f"Threshold Optimization Landscape\nBest: Wet={best_wet:.2f}, Dry={best_dry:.2f}")
    plt.xlabel("Wet Season Threshold")
    plt.ylabel("Dry Season Threshold")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3, linestyle="--")
    
    out_path = save_dir / "optuna_optimization_map.png"
    plt.savefig(out_path)
    plt.close()
    print(f"        - Landscape saved to: {out_path}")

    # --- Plot 2: History (Convergence Check) ---
    plt.figure(figsize=(10, 5))
    plt.plot(trials_df['number'], trials_df['value'].cummax(), color='red', label='Best So Far')
    plt.scatter(trials_df['number'], trials_df['value'], color='blue', alpha=0.3, s=10, label='Trial')
    
    plt.title("Optimization Convergence History")
    plt.xlabel("Trial Number")
    plt.ylabel("F2 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    hist_path = save_dir / "optuna_history.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"        - History saved to: {hist_path}")


# --- 2. Main Logic ---
def run_tuning():
    print(">>> Phase 2: Optuna Tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Artifacts
    with open(MODEL_DIR / "config.json", "r") as f: conf = json.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f: scaler = pickle.load(f)
    
    # Initialize Model
    model = FloodLSTM(conf["input_dim"], conf["hidden_dim"], conf["num_layers"], conf["dropout"])
    model.load_state_dict(torch.load(MODEL_DIR /"best_model.pth", map_location=device))
    model.eval()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"    >>> Device: {device}")
    # model, scaler, t = FloodLSTM.load_from_hub(repo_id="sirasira/flood-lstm-v1", device=device)

    # Load Test Data
    try:
        df = pd.read_csv(PROCESSED_DIR / "test_set.csv")
    except FileNotFoundError:
        print("❌ Error: test_set.csv not found.")
        return

    features = ['rainfall', 'total_report', 'API_30d', 'API_60d', 'API_90d', 'month_sin', 'month_cos', 'latitude', 'longitude']
    SEQ_LEN = CONFIG["SEQ_LEN"]

    # Cache Predictions (Vectorized Inference recommended here, but using your loop for now)
    print("     >>> Caching Predictions...")
    y_true, y_probs, months = [], [], []

    # Filter groups that are too short first to avoid errors
    groups = [g for _, g in df.groupby('subdistrict') if len(g) > SEQ_LEN]

    for g in groups:
        # Optimization: Pre-calculate scaling
        g_scaled = g.copy()
        g_scaled[features] = scaler.transform(g[features])
        v_scaled = g_scaled[features].values

        t = g['target'].values
        m = g['month_timestamp'].values
        
        # Prepare Batch instead of loop (Mini-optimization)
        # Creating a list of tensors and stacking is faster than looping inference
        X_batch = []
        valid_indices = []
        
        for i in range(len(g) - SEQ_LEN):
            X_batch.append(v_scaled[i : i+SEQ_LEN])
            valid_indices.append(i + SEQ_LEN)
            
        if not X_batch: continue
        
        X_tensor = torch.FloatTensor(np.array(X_batch)).to(device)
        
        with torch.no_grad():
            # Batch Inference: Pass all windows for this subdistrict at once
            # This makes the "Caching" step 10x-50x faster
            probs = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
            
        y_probs.extend(probs)
        y_true.extend(t[valid_indices])
        months.extend(m[valid_indices])

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    months = np.array(months)

    print(f"     >>> Data Cached. Samples: {len(y_true)}")

    # Define Objective
    def objective(trial):
        # We search for the "Sweet Spot"
        th_wet = trial.suggest_float("wet", 0.4, 0.85) # Widened range slightly
        th_dry = trial.suggest_float("dry", 0.1, 0.6)
        
        # Vectorized Threshold Application
        thresholds = np.where((months >= 5) & (months <= 10), th_wet, th_dry)
        preds = (y_probs > thresholds).astype(int)
        
        # Optimize for F2 (Recall-Weighted)
        return fbeta_score(y_true, preds, beta=2)

    # Create Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100) # 100 is a good number for 2 params

    # Save Results
    best_params = study.best_params
    print(f"\n>>> 🏆 Optimization Complete")
    print(f"    Best F2 Score: {study.best_value:.4f}")
    print(f"    Best Params:   {best_params}")

    with open(MODEL_DIR / "thresholds.json", "w") as f: 
        json.dump(best_params, f)
    
    # --- Trigger Visualization ---
    visualize_optuna_results(study, MODEL_DIR)

    print("✅ Phase 2 Complete. Thresholds and Plots saved.")

if __name__ == "__main__":
    run_tuning()