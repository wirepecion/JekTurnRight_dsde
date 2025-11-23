"""
run_pipeline.py
---------------
Entry point script.
"""
import logging
from src.dataprep.pipeline import FloodDataPipeline, PipelineConfig

# Configure logging to show up in terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # 1. Setup Configuration
    config = PipelineConfig(
        traffy_path="data/raw/bangkok_traffy.csv",
        shape_path="data/raw/BMA/BMA_ADMIN_SUB_DISTRICT.shp",  # Ensure this points to your .shp file
        station_path="data/external/station.csv",
        rain_path="data/external/rainfall.csv",
        output_path="data/processed/clean_flood_data.parquet"
    )

    # 2. Instantiate the Pipeline Class
    pipeline = FloodDataPipeline(config)

    # 3. Run it
    final_df = pipeline.run()

    # 4. Verify
    print("\n--- Final Data Preview ---")
    print(final_df.head())
    print(f"Total Rows: {len(final_df)}")

if __name__ == "__main__":
    main()