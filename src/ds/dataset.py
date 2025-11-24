"""
src/ds/dataset.py
-----------------
PyTorch Dataset.
Consumes Densified Spark Data (grouped by Subdistrict).
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FloodLSTMDataset(Dataset):
    def __init__(self, parquet_path: str, sequence_length: int = 30, scaler=None):
        self.seq_len = sequence_length
        
        # 1. Load Data
        logger.info(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        
        # 2. Sort by Location & Time (Crucial for LSTM sliding window)
        # We must group by 'subdistrict' now, not 'station_code'
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(by=['subdistrict', 'date']).reset_index(drop=True)
        
        # 3. Define Input Features
        # Must match the columns in your Spark Output
        self.feature_cols = [
            'rainfall',               # Physical Driver 1
            'total_report',           # Social Sensor
            'number_of_report_flood', # Autoregression (Past history)
            'API_30d', 'API_60d', 'API_90d', # Soil Memory
            'month_sin', 'month_cos', # Seasonality
            'latitude', 'longitude'   # Spatial Context
        ]
        
        # 4. Scaling
        # LSTMs fail if 'rainfall' is 50.0 and 'latitude' is 13.0. We scale all to ~0-1 range.
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        else:
            self.scaler = scaler
            self.df[self.feature_cols] = self.scaler.transform(self.df[self.feature_cols])

        # 5. Create Sliding Windows
        self.sequences = []
        self.targets = []
        
        # Group by SUBDISTRICT to prevent windows from sliding across different locations
        grouped = self.df.groupby('subdistrict')
        
        for name, group in grouped:
            data = group[self.feature_cols].values
            # We use the pre-calculated binary target from Spark
            labels = group['target'].values
            
            if len(data) <= self.seq_len:
                continue
            
            # Logic: Use [t-30 ... t-1] to predict [t]
            for i in range(len(data) - self.seq_len):
                seq = data[i : i + self.seq_len]
                label = labels[i + self.seq_len]
                
                self.sequences.append(seq)
                self.targets.append(label)
        
        # Convert list to PyTorch Tensors (Float32 is standard for GPUs)
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
        logger.info(f"Dataset Ready. Input Shape: {self.sequences.shape} (Samples, TimeSteps, Features)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
        
    def get_input_dim(self):
        return self.sequences.shape[2]
        
    def get_scaler(self):
        return self.scaler