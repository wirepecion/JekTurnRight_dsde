import pandas as pd

# 1. Point to the FOLDER (not a specific file)
parquet_path = "data/processed/flood_training_data_spark"

print(f">>> Reading from: {parquet_path}")
try:
    # Pandas automatically stitches all the small files together
    df = pd.read_parquet(parquet_path)
    
    print("\n--- ✅ SUCCESS: Data Loaded ---")
    print(f"Total Days of Data: {len(df)}")
    print(f"Columns Found: {list(df.columns)}")
    
    print("\n--- 🔍 Sample Row (What the model sees) ---")
    # We look at one specific station to check if features make sense
    sample = df.head(5)
    print(sample)
    
    print(f"Date:          {sample['date_join']}")
    print(f"Station:       {sample['station_code']}")
    print(f"Rainfall:      {sample['rainfall']} mm")
    print(f"Floods Reported: {sample['number_of_report_flood']} (Target)")
    print(f"Soil Memory (30d): {sample['API_30d']:.2f}")
    
    if 'number_of_report_flood' in df.columns:
        print("\n--- 🎯 Target Check ---")
        flood_days = df[df['number_of_report_flood'] > 0]
        print(f"Days with Floods: {len(flood_days)} ({len(flood_days)/len(df):.1%} of data)")
        
except Exception as e:
    print(f"\n❌ FAIL: {e}")