from pyspark.sql import SparkSession
from traffy_flood_etl import run_traffy_flood_etl

if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("TraffyFloodETL")
        .getOrCreate()
    )

    INPUT = "data/raw/bangkok_traffy.csv"
    CLEANED_OUT = "data/processed/traffy_clean.parquet"
    FLOOD_TS_OUT = "data/processed/flood_daily_by_district.parquet"

    run_traffy_flood_etl(
        spark,
        input_path=INPUT,
        cleaned_output_path=CLEANED_OUT,
        flood_ts_output_path=FLOOD_TS_OUT,
    )

    spark.stop()


