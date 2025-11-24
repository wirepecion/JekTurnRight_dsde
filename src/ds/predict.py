"""
src/ds/predict.py
-----------------
Inference Engine for User Uploads.
Handles:
1. Column Mapping (rain_fall -> rainfall)
2. Feature Engineering on-the-fly (Seasonality, API)
3. Model Prediction
"""
import torch
import pandas as pd
import numpy as np
import pickle
import json
import logging
from src.ds.model import FloodLSTM

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Predictor")

class FloodPredictor:
    def __init__(self, model_dir="."):
        """
        Loads Model, Config, and Scaler from the directory.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Config
        try:
            with open(f"{model_dir}/model_config.json", "r") as f:
                self.config = json.load(f)
            
            # 2. Load Scaler
            with open(f"{model_dir}/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
                
            # 3. Load Model
            self.model = FloodLSTM(
                self.config["input_dim"], 
                self.config["hidden_dim"], 
                self.config["num_layers"], 
                self.config["dropout"]
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(f"{model_dir}/best_model.pth", map_location=self.device))
            self.model.eval()
            logger.info("✅ Model loaded successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"Missing artifact: {e}. Did you train the model?")
            raise e

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms User CSV format into Model Features.
        """
        df = df.copy()
        
        # A. Column Mapping (User Input -> Model Name)
        rename_map = {
            'rain_fall': 'rainfall',
            # Add others if names differ, e.g. 'sub_district' -> 'subdistrict'
        }
        df = df.rename(columns=rename_map)
        
        # B. Date Construction
        try:
            df['date'] = pd.to_datetime(dict(
                year=df.year_timestamp, 
                month=df.month_timestamp, 
                day=df.days_timestamp
            ))
        except Exception as e:
            logger.error(f"Date parsing failed: {e}")
            raise ValueError("Invalid Date Columns")

        # C. Sort (Critical for rolling windows)
        df = df.sort_values(['subdistrict', 'date']).reset_index(drop=True)

        # D. Handle Missing "History" Features
        # The model expects 'number_of_report_flood' (Autoregression).
        # If user doesn't provide it, we assume 0 (No current flood).
        if 'number_of_report_flood' not in df.columns:
            df['number_of_report_flood'] = 0

        # E. Feature Engineering 1: Seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month_timestamp'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_timestamp'] / 12)

        # F. Feature Engineering 2: Soil Memory (API)
        # Logic: If user uploads 1 row, rolling window fails. We fill with 0.
        for w in [30, 60, 90]:
            col_name = f'API_{w}d'
            # We use min_periods=1 so it works even with just 1 day of data
            df[col_name] = df.groupby('subdistrict')['rainfall'].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            ).fillna(0.0)

        return df

    def predict(self, csv_path: str):
        logger.info(f"Processing file: {csv_path}")
        
        # 1. Read & Clean
        raw_df = pd.read_csv(csv_path)
        clean_df = self.preprocess(raw_df)
        
        # 2. Define Features (Must match Training EXACTLY)
        feature_cols = [
            'rainfall', 'total_report', 'number_of_report_flood',
            'API_30d', 'API_60d', 'API_90d',
            'month_sin', 'month_cos',
            'latitude', 'longitude'
        ]
        
        # 3. Scale
        # Warning: If scaler expects columns in specific order, we must enforce it
        X_scaled = self.scaler.transform(clean_df[feature_cols])
        
        # 4. Inference Loop
        results = []
        seq_len = self.config["SEQ_LEN"] # usually 30
        
        # We group by subdistrict to handle multiple locations in one file
        for name, group in clean_df.groupby('subdistrict'):
            # Check if we have enough data for a full sequence
            data = group[feature_cols].values # Get scaled values? NO! scaler returns numpy array
            
            # Correct way: Grab the slice from X_scaled corresponding to this group
            indices = group.index
            group_scaled = X_scaled[indices]
            
            if len(group) < seq_len:
                # Cold Start Strategy: Pad with zeros if user provided < 30 days
                # Create a buffer of zeros
                padding = np.zeros((seq_len - len(group), len(feature_cols)))
                # Stack padding + actual data
                sequence = np.vstack([padding, group_scaled])
            else:
                # Take the LAST 'seq_len' days (Forecast based on latest data)
                sequence = group_scaled[-seq_len:]
            
            # Convert to Tensor (Batch Size = 1)
            X_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(X_tensor)
                prob = torch.sigmoid(logits).item()
            
            # Decision (Using generic 0.5 threshold or your Optuna one)
            status = "🚨 FLOOD RISK" if prob > 0.5 else "✅ Safe"
            
            results.append({
                "subdistrict": name,
                "date": str(group['date'].iloc[-1].date()),
                "rain_today": group['rainfall'].iloc[-1],
                "risk_score": f"{prob:.1%}",
                "prediction": status
            })
            
        return pd.DataFrame(results)

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Generate a dummy file to test
    dummy_data = {
        "year_timestamp": [2025, 2025],
        "month_timestamp": [10, 10],
        "days_timestamp": [2, 3],
        "subdistrict": ["Lat Krabang", "Lat Krabang"],
        "rain_fall": [1.5, 50.2],
        "total_report": [5.0, 12.0],
        "latitude": [13.72, 13.72],
        "longitude": [100.75, 100.75] # Corrected Lon for Lat Krabang
    }
    pd.DataFrame(dummy_data).to_csv("user_upload_test.csv", index=False)
    
    # Run Prediction
    predictor = FloodPredictor()
    result = predictor.predict("user_upload_test.csv")
    
    print("\n=== 🔮 FORECAST RESULTS ===")
    print(result)