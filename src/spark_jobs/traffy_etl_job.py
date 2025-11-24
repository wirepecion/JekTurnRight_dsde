"""
src/spark_jobs/traffy_etl_job.py
--------------------------------
The Main Spark Application.
Refactored to ensure COMPLETE Time Series (Every Subdistrict x Every Date).

Logic:
1. Master Location Table: Map every Subdistrict Centroid -> Nearest Station.
2. Rain Expansion: Broadcast rain from Station -> Subdistrict.
3. Flood Aggregation: Count reports per Subdistrict/Date.
4. Densification: Cross Join Dates x Locations to fill gaps.
5. Feature Engineering: Calculate API on the dense data.

Usage:
  export PYTHONPATH=$PYTHONPATH:.
  spark-submit --master local[2] --driver-memory 4g src/spark_jobs/traffy_etl_job.py
"""
import sys
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# --- IMPORT WORKER MODULES ---
from src.dataprep import io, mergers
from src.spark_jobs import cleaning

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_master_location_map(shape_path, station_path):
    """
    Runs on Driver (Pandas/Geopandas).
    Creates a master mapping: Subdistrict -> Lat/Lon (Centroid) -> Nearest Station.
    """
    logger.info(">>> [Driver] Building Master Location Map (Subdistrict -> Station)...")
    
    # 1. Load Shapefile & Get Centroids
    gdf = gpd.read_file(shape_path)
    # Rename cols to match standard
    gdf = gdf.rename(columns={'SUBDISTR_1': 'subdistrict', 'DISTRICT_N': 'district'})
    gdf = gdf.to_crs("EPSG:4326")
    centroids = gdf.geometry.centroid
    
    df_loc = pd.DataFrame({
        'subdistrict': gdf['subdistrict'],
        'district': gdf['district'],
        'latitude': centroids.y,
        'longitude': centroids.x
    })
    
    # 2. Load Stations
    station_df = pd.read_csv(station_path)
    station_df = mergers.clean_station_metadata(station_df)
    
    # 3. Map Subdistrict Centroids to Nearest Station using BallTree
    # (Same logic as your original python code, but run once for metadata)
    station_coords = np.radians(station_df[['latitude', 'longitude']].values)
    loc_coords = np.radians(df_loc[['latitude', 'longitude']].values)
    
    tree = BallTree(station_coords, metric='haversine')
    dist, idx = tree.query(loc_coords, k=1)
    
    df_loc['station_code'] = station_df.iloc[idx.flatten()]['station_code'].values
    
    # Returns: [subdistrict, district, latitude, longitude, station_code]
    return df_loc

def main():
    # 1. Initialize Spark
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
        # =========================================================
        # STEP 2: PREPARE BACKBONE (Date x Location)
        # =========================================================
        logger.info(">>> [2/7] Preparing Backbone (Locations & Dates)...")
        
        # A. Create Master Location DataFrame (Subdistrict -> Station)
        # We do this on driver because logic is complex but data is small
        pdf_master = get_master_location_map(
            "data/raw/BMA/BMA_ADMIN_SUB_DISTRICT.shp", 
            "data/external/station.csv"
        )
        df_master_loc = spark.createDataFrame(pdf_master)
        
        # B. Create Date Sequence (2022-2024)
        df_dates = spark.sql("""
            SELECT explode(sequence(
                to_date('2022-01-01'), 
                to_date('2024-12-31'), 
                interval 1 day
            )) as date
        """)
        
        # C. Cross Join: Every Subdistrict on Every Day
        # This is our "Skeleton". It ensures NO day is missing.
        df_skeleton = df_dates.crossJoin(df_master_loc)
        
        # =========================================================
        # STEP 3: EXPAND RAINFALL
        # =========================================================
        logger.info(">>> [3/7] Expanding Rainfall to Subdistricts...")
        
        # Load Rain
        rain_pd = io.load_csv("data/external/rainfall.csv")
        rain_pd_clean = mergers.clean_rainfall_data(rain_pd)
        df_rain = spark.createDataFrame(rain_pd_clean) # Cols: date, station_code, rainfall
        
        # Join Rain onto Skeleton (via station_code)
        # Now every subdistrict has rain data for every day
        df_backbone = df_skeleton.join(
            df_rain,
            (df_skeleton.station_code == df_rain.station_code) & 
            (df_skeleton.date == df_rain.date),
            "left"
        ).select(
            df_skeleton.date,
            df_skeleton.subdistrict,
            df_skeleton.district,
            df_skeleton.latitude,
            df_skeleton.longitude,
            df_rain.rainfall
        )
        
        # Fill Missing Rain (e.g., station error) with 0.0
        df_backbone = df_backbone.na.fill(0.0, subset=["rainfall"])

        # =========================================================
        # STEP 4: PROCESS FLOOD REPORTS
        # =========================================================
        logger.info(">>> [4/7] Processing Flood Reports...")
        
        df_raw = spark.read.option("header", "true").csv("data/raw/bangkok_traffy.csv")
        
        df_clean = (df_raw
            .transform(lambda df: cleaning.drop_nan_rows(df, ["coords", "timestamp"]))
            .transform(cleaning.convert_types)
            .transform(cleaning.clean_province_name)
            .transform(cleaning.parse_type_column)
        )
        
        # Aggregate: Count Reports per Subdistrict/Date
        df_clean = df_clean.withColumn("is_flood", F.array_contains(F.col("type_list"), "น้ำท่วม"))
        df_clean = df_clean.withColumn("date_report", F.to_date("timestamp"))
        
        df_flood_agg = df_clean.groupBy("subdistrict", "date_report") \
            .agg(
                F.count("ticket_id").alias("total_report_real"),
                F.sum(F.when(F.col("is_flood") == True, 1).otherwise(0)).alias("flood_real")
            )

        # =========================================================
        # STEP 5: FINAL MERGE (Backbone + Flood)
        # =========================================================
        logger.info(">>> [5/7] Merging Flood Reports into Backbone...")
        
        # Left Join: Keep all Backbone rows (dates), attach flood counts if they exist
        # NOTE: We explicitly drop the right side columns to avoid ambiguity
        df_final = df_backbone.join(
            df_flood_agg,
            (df_backbone.date == df_flood_agg.date_report) & 
            (df_backbone.subdistrict == df_flood_agg.subdistrict),
            "left"
        ).drop(df_flood_agg.subdistrict).drop(df_flood_agg.date_report)
        
        # Fill Missing Counts with 0
        df_final = df_final \
            .withColumn("total_report", F.coalesce(F.col("total_report_real"), F.lit(0))) \
            .withColumn("number_of_report_flood", F.coalesce(F.col("flood_real"), F.lit(0))) \
            .withColumn("target", F.when(F.col("flood_real") > 0, 1).otherwise(0)) \
            .drop("total_report_real", "flood_real")

        # =========================================================
        # STEP 6: FEATURE ENGINEERING (Physics)
        # =========================================================
        logger.info(">>> [6/7] Calculating Physics (API) & Seasonality...")
        
        # 1. Seasonality
        df_final = df_final \
            .withColumn("year_timestamp", F.year("date")) \
            .withColumn("month_timestamp", F.month("date")) \
            .withColumn("month_sin", F.sin(2 * np.pi * F.col("month_timestamp") / 12)) \
            .withColumn("month_cos", F.cos(2 * np.pi * F.col("month_timestamp") / 12))
            
        # 2. Soil Memory (API)
        # CRITICAL: Partition by 'subdistrict' now, not 'station_code'
        w_spec = Window.partitionBy("subdistrict").orderBy("date")
        
        df_final = df_final \
            .withColumn("API_30d", F.avg("rainfall").over(w_spec.rowsBetween(-30, 0))) \
            .withColumn("API_60d", F.avg("rainfall").over(w_spec.rowsBetween(-60, 0))) \
            .withColumn("API_90d", F.avg("rainfall").over(w_spec.rowsBetween(-90, 0))) \
            .na.fill(0.0, subset=["API_30d", "API_60d", "API_90d"])

        # =========================================================
        # STEP 7: WRITE OUTPUT
        # =========================================================
        output_path = "data/processed/flood_training_data_spark"
        logger.info(f">>> [7/7] Writing Result to {output_path}...")
        
        (df_final
         .write
         .mode("overwrite")
         .partitionBy("year_timestamp", "month_timestamp")
         .parquet(output_path))
         
        # Check CSV
        csv_path = "data/processed/flood_training_data_csv"
        (df_final.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_path))

        print("\n--- FINAL SCHEMA ---")
        df_final.printSchema()
        
        logger.info(">>> JOB SUCCESS. I AM THE BOSS.")

    except Exception as e:
        logger.error(f"!!! JOB FAILED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()