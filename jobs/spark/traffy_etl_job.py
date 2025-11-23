"""
jobs/spark/traffy_etl_job.py
----------------------------
The "Boss" ETL Job.
Orchestrates data cleaning, spatial mapping, and rainfall enrichment.

Usage:
  spark-submit \
    --master local[*] \
    --py-files jobs/spark/utils/geo_spark.py \
    --files conf/log4j2.properties \
    jobs/spark/traffy_etl_job.py \
    --input data/raw/bangkok_traffy.csv \
    --station data/external/station.csv \
    --rainfall data/external/rainfall.csv \
    --output data/processed/flood_training_data
"""

import sys
import argparse
import pandas as pd # Used only on Driver for small reference data
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, DateType, TimestampType

# Import Custom Spatial Logic (Must be passed via --py-files)
try:
    from utils.geo_spark import set_broadcast_variable, build_station_index, find_nearest_station
except ImportError:
    print("CRITICAL ERROR: utils.geo_spark not found. Did you forget '--py-files jobs/spark/utils/geo_spark.py'?")
    sys.exit(1)

def get_logger(spark):
    """Bridge Python logging to Spark's Log4j"""
    log4j = spark._jvm.org.apache.log4j
    return log4j.LogManager.getLogger("TraffyFloodETL")

def parse_args():
    parser = argparse.ArgumentParser(description="Traffy Flood ETL")
    parser.add_argument("--input", required=True, help="Path to raw Traffy CSV")
    parser.add_argument("--station", required=True, help="Path to station metadata CSV")
    parser.add_argument("--rainfall", required=True, help="Path to rainfall CSV")
    parser.add_argument("--output", required=True, help="Path to save Parquet output")
    return parser.parse_args()

def main():
    # 1. Parse Arguments
    args = parse_args()

    # 2. Initialize Spark
    spark = SparkSession.builder \
        .appName("TraffyFloodETL_Production") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:conf/log4j2.properties") \
        .getOrCreate()

    logger = get_logger(spark)
    logger.info(f"Starting ETL Job. Input: {args.input}")

    try:
        # ====================================================
        # PHASE 1: BROADCAST STATION LOOKUP (Driver Side)
        # ====================================================
        logger.info(">>> Phase 1: Building Spatial Index for Stations...")
        
        # [cite_start]We use Pandas here because station data is tiny (< 1MB) [cite: 3]
        station_pd = pd.read_csv(args.station)
        
        # [cite_start]Clean Station Data (Logic ported from mergers.py) [cite: 3]
        station_pd = station_pd.rename(columns={
            "StationCode": "station_code",
            "DistrictName": "district",
            "Latitude": "latitude",
            "Longitude": "longitude"
        })
        
        # Fix District Names (Standardize spelling)
        replacements = {
            "ป้อมปราบฯ": "ป้อมปราบศัตรูพ่าย",
            "ราษฏร์บูรณะ": "ราษฎร์บูรณะ"
        }
        station_pd["district"] = station_pd["district"].replace(replacements)
        
        # Remove prefix 'BKK' if exists (Logic from services.py)
        station_pd["station_code"] = station_pd["station_code"].apply(
            lambda x: x[3:] if isinstance(x, str) and len(x) == 9 else x
        )

        # Build & Broadcast the Spatial Tree
        station_index = build_station_index(station_pd)
        broadcast_var = spark.sparkContext.broadcast(station_index)
        set_broadcast_variable(broadcast_var) # Register on Workers
        
        logger.info(f"Broadcasted {len(station_pd)} stations to workers.")

        # ====================================================
        # PHASE 2: INGEST & CLEAN TRAFFY DATA (Distributed)
        # ====================================================
        logger.info(">>> Phase 2: Processing Traffy Flood Reports...")
        
        df_raw = spark.read.option("header", "true").csv(args.input)
        
        # [cite_start]Select & Cast basic columns [cite: 2]
        df_clean = df_raw.select(
            F.col("ticket_id"),
            F.col("type"),
            F.col("timestamp"),
            F.col("coords"),
            F.col("province"),
            F.col("district").alias("district_traffy"), # Rename to avoid join collision later
            F.col("subdistrict")
        )

        # [cite_start]Apply Filters (Logic from pipeline.py) [cite: 3]
        df_clean = df_clean \
            .filter(F.col("province").contains("กรุงเทพ")) \
            .withColumn("timestamp_dt", F.to_timestamp("timestamp")) \
            .filter(
                (F.year("timestamp_dt") >= 2022) & 
                (F.year("timestamp_dt") <= 2024)
            )

        # Parse Coordinates (Split "100.5,13.7")
        df_clean = df_clean \
            .withColumn("longitude", F.split("coords", ",").getItem(0).cast(DoubleType())) \
            .withColumn("latitude", F.split("coords", ",").getItem(1).cast(DoubleType())) \
            .filter(F.col("latitude").isNotNull() & F.col("longitude").isNotNull())

        # ====================================================
        # PHASE 3: SPATIAL MAPPING (The Heavy Lifting)
        # ====================================================
        logger.info(">>> Phase 3: Executing Vectorized Spatial UDF...")
        
        # This uses the Broadcast variable to find nearest station for each row
        # using the 'district_traffy' to narrow the search scope
        df_mapped = df_clean.withColumn(
            "station_code",
            find_nearest_station(F.col("district_traffy"), F.col("latitude"), F.col("longitude"))
        )
        
        # Cache this result because we might use it twice (debug count + join)
        df_mapped.cache()
        # Trigger an action to force the UDF to run and catch errors early
        mapped_count = df_mapped.count()
        logger.info(f"Successfully mapped {mapped_count} flood reports.")

        # ====================================================
        # PHASE 4: ENRICH WITH RAINFALL
        # ====================================================
        logger.info(">>> Phase 4: Merging Rainfall Data...")
        
        df_rain = spark.read.option("header", "true").csv(args.rainfall)
        
        # [cite_start]Dynamic Stack (Unpivot) - "melt" in Spark SQL [cite: 3]
        # Identifies all columns that look like station codes (S01, S02...)
        station_cols = [c for c in df_rain.columns if c not in ["Date", "time"]]
        
        if not station_cols:
            logger.warn("No station columns found in rainfall data!")
        
        # Generate the SQL stack expression string
        stack_expr = f"stack({len(station_cols)}, " + \
                     ", ".join([f"'{c}', `{c}`" for c in station_cols]) + \
                     ") as (station_code, rainfall)"
        
        df_rain_long = df_rain.select(
            F.col("Date").alias("date_rain"), # Keep string for now
            F.expr(stack_expr)
        )
        
        # Convert date strings to DateType for joining
        # Assuming Rainfall Date format matches specific pattern, usually yyyy-MM-dd
        df_rain_long = df_rain_long.withColumn("date_rain", F.to_date("date_rain"))
        
        # Prepare Traffy side for join
        df_traffy_final = df_mapped.withColumn("date_report", F.to_date("timestamp_dt"))

        # LEFT JOIN: We want all flood reports, even if no rain data found
        df_result = df_traffy_final.join(
            df_rain_long,
            (df_traffy_final.station_code == df_rain_long.station_code) & 
            (df_traffy_final.date_report == df_rain_long.date_rain),
            "left"
        ).select(
            df_traffy_final["*"],
            df_rain_long["rainfall"]
        )
        
        # Fill missing rainfall with 0.0 (Assumption: No data = No rain)
        df_result = df_result.na.fill(0.0, subset=["rainfall"])

        # ====================================================
        # PHASE 5: WRITE OUTPUT
        # ====================================================
        logger.info(f">>> Phase 5: Writing Output to {args.output}")
        
        (df_result
         .write
         .mode("overwrite")
         .partitionBy("district_traffy") # Partitioning optimizes downstream queries
         .parquet(args.output))
        
        logger.info(">>> JOB COMPLETED SUCCESSFULLY. I AM THE BOSS.")

    except Exception as e:
        logger.error(f"!!! JOB FAILED: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()