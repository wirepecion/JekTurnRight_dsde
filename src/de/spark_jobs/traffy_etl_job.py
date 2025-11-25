# src/de/spark_jobs/traffy_etl_job.py
"""
Main Spark ETL for flood training data.

Steps:
1. Build Master Location Map (Subdistrict -> Station) on driver.
2. Create dense backbone: every date x subdistrict.
3. Expand rainfall from station to subdistricts.
4. Process Traffy flood reports & aggregate.
5. Join backbone + floods, fill gaps.
6. Feature engineering (seasonality, API).
7. Write partitioned Parquet + a single CSV export.
"""
from __future__ import annotations

import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
from pyspark.sql import Window
from pyspark.sql import functions as F

from src.dataprep import io, mergers  # driver-side Pandas helpers
from src.de import settings
from src.de.spark_session import create_spark
from src.de.spark_jobs import cleaning  # your Spark cleaning module

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_master_location_map(shape_path, station_path):
    """
    Runs on Driver (Pandas/Geopandas).
    Creates: Subdistrict -> Lat/Lon (Centroid) -> Nearest Station.

    Uses BallTree once on driver (small data) then broadcast to Spark.
    """
    logger.info(">>> [Driver] Building Master Location Map (Subdistrict -> Station)...")

    # 1. Load Shapefile & Get Centroids
    gdf = gpd.read_file(shape_path)
    gdf = gdf.rename(columns={"SUBDISTR_1": "subdistrict", "DISTRICT_N": "district"})
    gdf = gdf.to_crs("EPSG:4326")
    centroids = gdf.geometry.centroid

    df_loc = pd.DataFrame({
        "subdistrict": gdf["subdistrict"],
        "district": gdf["district"],
        "latitude": centroids.y,
        "longitude": centroids.x,
    })

    # 2. Load Stations (Pandas-level cleaning)
    station_df = pd.read_csv(station_path)
    station_df = mergers.clean_station_metadata(station_df)

    # 3. Map Subdistrict Centroids to Nearest Station using BallTree
    station_coords = np.radians(station_df[["latitude", "longitude"]].values)
    loc_coords = np.radians(df_loc[["latitude", "longitude"]].values)

    tree = BallTree(station_coords, metric="haversine")
    dist, idx = tree.query(loc_coords, k=1)

    df_loc["station_code"] = station_df.iloc[idx.flatten()]["station_code"].values
    return df_loc


def main() -> None:
    logger.info(">>> [1/7] Initializing Spark...")
    spark = create_spark()

    try:
        # =========================================================
        # STEP 2: PREPARE BACKBONE (Date x Location)
        # =========================================================
        logger.info(">>> [2/7] Preparing Backbone (Locations & Dates)...")

        pdf_master = get_master_location_map(
            settings.BMA_SHAPE,
            settings.STATION_CSV,
        )
        df_master_loc = spark.createDataFrame(pdf_master)

        df_dates = spark.sql("""
            SELECT explode(sequence(
                to_date('2022-01-01'),
                to_date('2024-12-31'),
                interval 1 day
            )) AS date
        """)

        df_skeleton = df_dates.crossJoin(df_master_loc)

        # =========================================================
        # STEP 3: EXPAND RAINFALL
        # =========================================================
        logger.info(">>> [3/7] Expanding Rainfall to Subdistricts...")

        rain_pd = io.load_csv(settings.RAINFALL_CSV)
        rain_pd_clean = mergers.clean_rainfall_data(rain_pd)
        df_rain = spark.createDataFrame(rain_pd_clean)  # date, station_code, rainfall

        df_backbone = (
            df_skeleton.join(
                df_rain,
                (df_skeleton.station_code == df_rain.station_code)
                & (df_skeleton.date == df_rain.date),
                "left",
            )
            .select(
                df_skeleton.date,
                df_skeleton.subdistrict,
                df_skeleton.district,
                df_skeleton.latitude,
                df_skeleton.longitude,
                df_rain.rainfall,
            )
        )

        df_backbone = df_backbone.na.fill(0.0, subset=["rainfall"])

        # =========================================================
        # STEP 4: PROCESS FLOOD REPORTS
        # =========================================================
        logger.info(">>> [4/7] Processing Flood Reports...")

        df_raw = (
            spark.read
            .option("header", "true")
            .csv(settings.TRAFFY_RAW_CSV.as_posix())
        )

        df_clean = (
            df_raw
            .transform(lambda df: cleaning.drop_nan_rows(df, ['ticket_id', 'type', 'organization', 'coords', 'province', 'timestamp', 'last_activity']))
            .transform(cleaning.convert_types)
            .transform(cleaning.clean_province_name)
            .transform(cleaning.parse_type_column)
        )

        df_clean = df_clean.withColumn(
            "is_flood", F.array_contains(F.col("type_list"), "น้ำท่วม")
        )
        df_clean = df_clean.withColumn("date_report", F.to_date("timestamp"))

        df_flood_agg = (
            df_clean.groupBy("subdistrict", "date_report")
            .agg(
                F.count("ticket_id").alias("total_report_real"),
                F.sum(F.when(F.col("is_flood") == True, 1).otherwise(0)).alias(
                    "flood_real"
                ),
            )
        )

        # =========================================================
        # STEP 5: FINAL MERGE (Backbone + Flood)
        # =========================================================
        logger.info(">>> [5/7] Merging Flood Reports into Backbone...")

        df_final = (
            df_backbone.join(
                df_flood_agg,
                (df_backbone.date == df_flood_agg.date_report)
                & (df_backbone.subdistrict == df_flood_agg.subdistrict),
                "left",
            )
            .drop(df_flood_agg.subdistrict)
            .drop(df_flood_agg.date_report)
        )

        df_final = (
            df_final
            .withColumn("total_report",
                        F.coalesce(F.col("total_report_real"), F.lit(0)))
            .withColumn("number_of_report_flood",
                        F.coalesce(F.col("flood_real"), F.lit(0)))
            .withColumn("target",
                        F.when(F.col("flood_real") > 0, 1).otherwise(0))
            .drop("total_report_real", "flood_real")
        )

        # =========================================================
        # STEP 6: FEATURE ENGINEERING
        # =========================================================
        logger.info(">>> [6/7] Calculating Seasonality & API...")

        df_final = (
            df_final
            .withColumn("year_timestamp", F.year("date"))
            .withColumn("month_timestamp", F.month("date"))
            .withColumn(
                "month_sin",
                F.sin(2 * np.pi * F.col("month_timestamp") / 12),
            )
            .withColumn(
                "month_cos",
                F.cos(2 * np.pi * F.col("month_timestamp") / 12),
            )
        )

        w_spec = Window.partitionBy("subdistrict").orderBy("date")

        df_final = (
            df_final
            .withColumn("API_30d",
                        F.avg("rainfall").over(w_spec.rowsBetween(-30, 0)))
            .withColumn("API_60d",
                        F.avg("rainfall").over(w_spec.rowsBetween(-60, 0)))
            .withColumn("API_90d",
                        F.avg("rainfall").over(w_spec.rowsBetween(-90, 0)))
            .na.fill(0.0, subset=["API_30d", "API_60d", "API_90d"])
        )

        # =========================================================
        # STEP 7: WRITE OUTPUT
        # =========================================================
        logger.info(">>> [7/7] Writing outputs...")

        parquet_path = settings.FLOOD_TRAIN_PARQUET.as_posix()
        csv_dir = settings.FLOOD_TRAIN_CSV.as_posix()

        (
            df_final
            .write
            .mode("overwrite")
            .partitionBy("year_timestamp", "month_timestamp")
            .parquet(parquet_path)
        )

        (
            df_final
            .coalesce(1)
            .write
            .mode("overwrite")
            .option("header", "true")
            .csv(csv_dir)
        )

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
