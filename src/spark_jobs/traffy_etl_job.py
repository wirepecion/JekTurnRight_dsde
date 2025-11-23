"""
src/spark_jobs/traffy_etl_job.py
--------------------------------
The Main Spark Application.
Orchestrates the End-to-End ETL Pipeline with Densification.

Usage:
  export PYTHONPATH=$PYTHONPATH:.
  spark-submit --master local[2] --driver-memory 4g src/spark_jobs/traffy_etl_job.py
"""
import sys
import logging
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# --- IMPORT WORKER MODULES ---
from src.dataprep import io, mergers
from src.spark_jobs import cleaning, geo, features

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Initialize Spark (Optimized for WSL/Local)
    logger.info(">>> [1/7] Initializing Spark...")
    spark = (SparkSession.builder
             .appName("TraffyFloodETL_Production")
             .master("local[2]")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")

    try:
        # 2. Prepare & Broadcast Reference Data
        logger.info(">>> [2/7] Broadcasting Reference Data...")
        
        # A. Shapefile
        shape_gdf = io.load_shapefile("data/raw/BMA/BMA_ADMIN_SUB_DISTRICT.shp")
        shape_gdf = shape_gdf.to_crs("EPSG:4326")
        bc_shape = spark.sparkContext.broadcast(shape_gdf)
        geo.set_shape_broadcast(bc_shape)
        
        # B. Stations
        station_df = io.load_csv("data/external/station.csv")
        station_df = mergers.clean_station_metadata(station_df)
        station_lookup = geo.prepare_station_lookup(station_df)
        bc_stations = spark.sparkContext.broadcast(station_lookup)
        geo.set_station_broadcast(bc_stations)

        # 3. Process Main Data
        logger.info(">>> [3/7] Reading & Cleaning Traffy Data...")
        df_raw = spark.read.option("header", "true").csv("data/raw/bangkok_traffy.csv")
        
        df_clean = (df_raw
            .transform(lambda df: cleaning.drop_nan_rows(df, ["coords", "timestamp"]))
            .transform(cleaning.convert_types)
            .transform(cleaning.extract_time_features)
            .filter("year_timestamp >= 2022 AND year_timestamp <= 2024")
            .transform(cleaning.parse_coordinates)
            .transform(cleaning.clean_province_name)
            .transform(cleaning.parse_type_column)
        )

        # 4. Spatial Operations
        logger.info(">>> [4/7] Running Spatial UDFs...")
        df_mapped = df_clean.withColumn(
            "station_code", 
            geo.get_nearest_station_udf(F.col("district"), F.col("latitude"), F.col("longitude"))
        )
        df_mapped.cache()
        logger.info(f"    Mapped {df_mapped.count()} rows successfully.")

        # 5. Merge Rainfall
        logger.info(">>> [5/7] Merging Rainfall Data...")
        rain_pd = io.load_csv("data/external/rainfall.csv")
        rain_pd_clean = mergers.clean_rainfall_data(rain_pd)
        df_rain_clean = spark.createDataFrame(rain_pd_clean)

        df_traffy_ready = df_mapped.withColumn("date_join", F.to_date("timestamp"))

        df_joined = df_traffy_ready.join(
            df_rain_clean,
            (df_traffy_ready.station_code == df_rain_clean.station_code) & 
            (df_traffy_ready.date_join == df_rain_clean.date),
            "left"
        ).select(
            df_traffy_ready["*"],
            df_rain_clean["rainfall"]
        )
        
        df_joined = df_joined.na.fill(0.0, subset=["rainfall"])

        # 6. Feature Engineering
        logger.info(">>> [6/7] Generating Physics (API) & Seasonality...")
        df_feats = features.add_seasonality(df_joined)
        df_feats = features.add_soil_memory(df_feats)
        df_feats = df_feats.na.fill(0.0, subset=["API_30d", "API_60d", "API_90d"])

        # =========================================================
        # 6.5. AGGREGATE (Sparse)
        # =========================================================
        logger.info(">>> [6.5/7] Aggregating from Tickets to Daily Stats...")

        df_feats = df_feats.withColumn("is_flood", F.array_contains(F.col("type_list"), "น้ำท่วม"))

        df_daily_sparse = df_feats.groupBy("subdistrict", "date_join") \
            .agg(
                F.first("latitude").alias("lat_real"),
                F.first("longitude").alias("lon_real"),
                F.first("rainfall").alias("rain_real"),
                F.count("ticket_id").alias("total_report_real"),
                F.sum(F.when(F.col("is_flood") == True, 1).otherwise(0)).alias("flood_real"),
                F.first("API_30d").alias("api30_real"),
                F.first("API_60d").alias("api60_real"),
                F.first("API_90d").alias("api90_real"),
                F.first("month_sin").alias("sin_real"),
                F.first("month_cos").alias("cos_real"),
                F.first("year_timestamp").alias("year_real"),
                F.first("month_timestamp").alias("month_real")
            )

        # =========================================================
        # 6.6. DENSIFICATION (Filling missing dates)
        # =========================================================
        logger.info(">>> [6.6/7] Densifying Data...")

        # A. Generate Date Skeleton
        df_dates = spark.sql("""
            SELECT explode(sequence(
                to_date('2022-01-01'), 
                to_date('2024-12-31'), 
                interval 1 day
            )) as date
        """)

        # B. Get Unique Subdistricts
        df_subs = df_feats.select("subdistrict").distinct()

        # C. Cross Join
        df_skeleton = df_dates.crossJoin(df_subs)

        # D. Join Skeleton with Actual Data
        # THE FIX IS HERE: We drop df_daily_sparse.subdistrict to avoid ambiguity
        df_dense = df_skeleton.join(
            df_daily_sparse,
            (df_skeleton.date == df_daily_sparse.date_join) & 
            (df_skeleton.subdistrict == df_daily_sparse.subdistrict),
            "left"
        ).drop(df_daily_sparse.subdistrict)

        # =========================================================
        # 6.7. FILL MISSING VALUES
        # =========================================================
        # 1. Fill Metrics with 0
        df_final = df_dense \
            .withColumn("total_report", F.coalesce(F.col("total_report_real"), F.lit(0))) \
            .withColumn("number_of_report_flood", F.coalesce(F.col("flood_real"), F.lit(0))) \
            .withColumn("rainfall", F.coalesce(F.col("rain_real"), F.lit(0.0))) \
            .withColumn("API_30d", F.coalesce(F.col("api30_real"), F.lit(0.0))) \
            .withColumn("API_60d", F.coalesce(F.col("api60_real"), F.lit(0.0))) \
            .withColumn("API_90d", F.coalesce(F.col("api90_real"), F.lit(0.0))) \
            .withColumn("target", F.when(F.col("flood_real") > 0, 1).otherwise(0))

        # 2. Backfill/ForwardFill Coordinates
        # Use Window to spread valid Lat/Lon to the empty days
        w_sub = Window.partitionBy("subdistrict").orderBy("date")
        
        df_final = df_final \
            .withColumn("latitude", F.last("lat_real", ignorenulls=True).over(w_sub)) \
            .withColumn("longitude", F.last("lon_real", ignorenulls=True).over(w_sub)) \
            .withColumn("latitude", F.first("latitude", ignorenulls=True).over(w_sub.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))) \
            .withColumn("longitude", F.first("longitude", ignorenulls=True).over(w_sub.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))

        # 3. Re-calculate Date Features
        df_final = df_final \
            .withColumn("year_timestamp", F.year("date")) \
            .withColumn("month_timestamp", F.month("date")) \
            .withColumn("month_sin", F.sin(2 * np.pi * F.col("month_timestamp") / 12)) \
            .withColumn("month_cos", F.cos(2 * np.pi * F.col("month_timestamp") / 12))

        # 4. Final Selection
        df_final = df_final.select(
            "subdistrict", "date", "year_timestamp", "month_timestamp",
            "latitude", "longitude", 
            "total_report", "number_of_report_flood", "target",
            "rainfall", "API_30d", "API_60d", "API_90d", 
            "month_sin", "month_cos"
        )

        # 7. Write Output
        output_path = "data/processed/flood_training_data_spark"
        logger.info(f">>> [7/7] Writing Result to {output_path}...")
        
        (df_final
         .write
         .mode("overwrite")
         .partitionBy("year_timestamp", "month_timestamp")
         .parquet(output_path))
         
        # CSV for checking
        csv_path = "data/processed/flood_training_data_csv"
        logger.info(f">>> [Extra] Writing CSV to {csv_path}...")
        (df_final.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_path))

        print("\n--- FINAL SCHEMA (Densified) ---")
        df_final.printSchema()
        
        logger.info(">>> JOB SUCCESS. I AM THE BOSS.")

    except Exception as e:
        logger.error(f"!!! JOB FAILED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
