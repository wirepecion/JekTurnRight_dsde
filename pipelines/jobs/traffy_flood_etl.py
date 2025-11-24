# pipelines/jobs/traffy_flood_etl.py
from pyspark.sql import SparkSession
from src.setting.config import RAW_DIR, CLEANED_TRAFFY_PATH, FLOOD_TS_PATH
from src.de.spark_jobs.traffy_flood_etl import run_traffy_flood_etl


def main():
    spark = (
        SparkSession.builder
        .appName("TraffyFloodETL")
        .getOrCreate()
    )

    try:
        run_traffy_flood_etl(
            spark,
            input_path=str(RAW_DIR / "bangkok_traffy.csv"),
            cleaned_output_path=str(CLEANED_TRAFFY_PATH),
            flood_ts_output_path=str(FLOOD_TS_PATH),
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()


# uv run python -m pipelines.jobs.traffy_flood_etl
