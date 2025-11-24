"""
test_spark_geo.py
-----------------
Tests the Spatial UDFs.
"""
import pandas as pd
import geopandas as gpd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.dataprep import io, mergers # Reuse your existing IO logic
from src.spark_jobs import geo

def main():
    print(">>> 1. Starting Spark...")
    spark = SparkSession.builder \
        .appName("TestGeo") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .master("local[*]") \
        .getOrCreate()

    # --- PREPARE BROADCAST DATA (DRIVER SIDE) ---
    print(">>> 2. Loading & Broadcasting Reference Data...")
    
    # A. Shapefile
    shape_gdf = io.load_shapefile("data/raw/BMA/BMA_ADMIN_SUB_DISTRICT.shp")
    shape_gdf = shape_gdf.to_crs("EPSG:4326") # Ensure CRS
    # Broadcast it
    bc_shape = spark.sparkContext.broadcast(shape_gdf)
    geo.set_shape_broadcast(bc_shape)

    # B. Stations
    station_df = io.load_csv("data/external/station.csv")
    station_df = mergers.clean_station_metadata(station_df)
    # Convert to Lookup Dict
    station_lookup = geo.prepare_station_lookup(station_df)
    # Broadcast it
    bc_stations = spark.sparkContext.broadcast(station_lookup)
    geo.set_station_broadcast(bc_stations)

    # --- CREATE DUMMY DATA ---
    print(">>> 3. Creating Test Data...")
    # Real coordinate in Pathum Wan (Siam Paragonish area)
    data = [
        ("Test1", "เขตปทุมวัน", 13.746, 100.535), # Center of Bangkok
        ("Test2", "BadDistrict", 0.0, 0.0)       # Null Island
    ]
    df = spark.createDataFrame(data, ["id", "district", "latitude", "longitude"])

    # --- RUN UDFS ---
    print(">>> 4. Running Spatial Verification...")
    df = df.withColumn("is_in_bma", geo.verify_in_bma_udf(F.col("latitude"), F.col("longitude")))

    print(">>> 5. Running Nearest Station Search...")
    df = df.withColumn("nearest_station", geo.get_nearest_station_udf(F.col("district"), F.col("latitude"), F.col("longitude")))

    # --- SHOW RESULTS ---
    print(">>> RESULTS:")
    df.show(truncate=False)
    
    spark.stop()

if __name__ == "__main__":
    main()