"""
test_ds_loading.py
------------------
Verify that the Spark data can be loaded into PyTorch format.
"""
import logging
from src.ds.dataset import FloodLSTMDataset
from torch.utils.data import DataLoader

# Configure Logging
logging.basicConfig(level=logging.INFO)

def main():
    # 1. Path to your Spark Output
    # Note: Spark creates a folder. Pandas read_parquet can read the folder directly.
    parquet_path = "data/processed/flood_training_data_spark"
    
    print(">>> 1. Initializing Dataset...")
    try:
        # Look back 7 days
        dataset = FloodLSTMDataset(parquet_path, sequence_length=7)
        
        print(f"\n--- Dataset Statistics ---")
        print(f"Total Samples: {len(dataset)}")
        print(f"Input Features: {dataset.get_input_dim()}") # Should be 4 (Rain, Lat, Lon, Flood)
        
        # 2. Test DataLoader (Batching)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Get one batch
        X_batch, y_batch = next(iter(loader))
        
        print(f"\n--- Batch Shape Verification ---")
        print(f"X (Input) Shape: {X_batch.shape}  Expected: [32, 7, 4]")
        print(f"y (Label) Shape: {y_batch.shape}  Expected: [32]")
        
        print("\n>>> SUCCESS. Data is ready for training.")
        
    except Exception as e:
        print(f"\n!!! FAIL: {e}")
        # Hint: If this fails, check if the parquet folder exists!

if __name__ == "__main__":
    main()