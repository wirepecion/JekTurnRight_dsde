from src.dataprep.pipeline import FloodDataPipeline, PipelineConfig

def main():
    # Setup Config
    config = PipelineConfig(
        traffy_path="data/raw/bangkok_traffy.csv",
        shape_path="data/raw/BMA/BMA_ADMIN_SUB_DISTRICT.shp",
        station_path="data/external/station.csv",
        rain_path="data/external/rainfall.csv",
        output_path="data/processed/clean_flood_data.csv"
    )

    # Run Pipeline
    pipeline = FloodDataPipeline(config)
    final_df = pipeline.run()

    print(final_df.head())

if __name__ == "__main__":
    main()