from __future__ import annotations

from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


# -----------------------------
# Config / constants
# -----------------------------

# You can tweak this list anytime
FLOOD_KEYWORDS: List[str] = [
    "น้ำท่วม",
    "น้ำขัง",
    "น้ำล้น",
    "น้ำท่วมขัง",
    "ท่อระบายน้ำ",
    "ท่ออุดตัน",
    "ท่อระบาย",
    "ท่อตัน",
    "น้ำไหลย้อน",
    "ระบายน้ำไม่ทัน",
    "flood",
    "waterlogging",
    "drain",
    "drainage",
]


# -----------------------------
# Core ETL functions
# -----------------------------

def load_raw_traffy(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    """
    Load raw Traffy CSV.
    Designed so it can be called from Spark or an Airflow task.

    :param spark: SparkSession
    :param input_path: Path to raw Traffy CSV file (local, HDFS, GCS, S3, etc.)
    :return: Spark DataFrame
    """
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(input_path)
    )
    return df


def clean_and_filter_bangkok(df: DataFrame) -> DataFrame:
    """
    Basic cleaning:
    - parse timestamp
    - filter valid timestamps
    - filter province to Bangkok (Thai or English)
    - drop duplicate ticket_id if exists
    """
    # Parse timestamp
    df = df.withColumn(
        "timestamp",
        F.to_timestamp("timestamp")
    ).filter(F.col("timestamp").isNotNull())

    # Normalize text columns
    for col in ["province", "district", "subdistrict"]:
        if col in df.columns:
            df = df.withColumn(
                col,
                F.trim(F.col(col).cast("string"))
            )

    # Filter Bangkok if province exists
    if "province" in df.columns:
        df = df.filter(
            (F.col("province").contains("กรุงเทพ")) |
            (F.lower(F.col("province")).contains("bangkok"))
        )

    # Drop duplicate tickets if ticket_id exists
    if "ticket_id" in df.columns:
        window_cols = ["ticket_id"]
        df = df.dropDuplicates(window_cols)
    else:
        df = df.dropDuplicates()

    return df


def add_time_columns(df: DataFrame) -> DataFrame:
    """
    Add date and time-based columns used later for aggregation and modeling.
    """
    df = df.withColumn("date", F.to_date("timestamp"))
    df = df.withColumn("year", F.year("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    df = df.withColumn("day", F.dayofmonth("timestamp"))
    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("weekday", F.dayofweek("timestamp"))  # Sunday=1 ... Saturday=7
    return df


def add_flood_flag(df: DataFrame) -> DataFrame:
    """
    Add a boolean 'is_flood' column based on keywords in comment/type.
    Uses a simple OR of contains(...) over FLOOD_KEYWORDS.
    """

    # Safe lowercase string columns
    comment_col = F.lower(F.coalesce(F.col("comment").cast("string"), F.lit("")))
    type_col = F.lower(F.coalesce(F.col("type").cast("string"), F.lit("")))
    merged_text = F.concat_ws(" ", comment_col, type_col)

    flood_cond = F.lit(False)
    for kw in FLOOD_KEYWORDS:
        flood_cond = flood_cond | merged_text.contains(kw.lower())

    df = df.withColumn("is_flood", flood_cond.cast(T.BooleanType()))
    return df


def aggregate_flood_daily_by_district(df: DataFrame) -> DataFrame:
    """
    Filter flood-related tickets and aggregate to daily counts per district.
    Returns a DataFrame with:
        date, district, flood_complaint_count
    and filled missing counts with 0 for existing (date, district) combos.
    """
    # Filter flood tickets only
    df_flood = df.filter(F.col("is_flood") == True)

    # Drop rows without district or date
    df_flood = df_flood.filter(
        F.col("district").isNotNull() & F.col("date").isNotNull()
    )

    # Aggregate
    agg = (
        df_flood
        .groupBy("date", "district")
        .agg(
            F.count("ticket_id").alias("flood_complaint_count")
        )
    )

    # Build full (date, district) grid to fill missing days with 0
    date_min, date_max = agg.agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date")
    ).first()

    districts_df = agg.select("district").distinct()

    date_df = (
        df_flood
        .select("date")
        .where(F.col("date").between(date_min, date_max))
        .distinct()
        .orderBy("date")
    )

    # If you want strict full range:
    # date_df = (
    #     spark.createDataFrame(
    #         [(date_min + F.expr(f"INTERVAL {i} DAYS"),) for i in range((date_max - date_min).days + 1)],
    #         schema=T.StructType([T.StructField("date", T.DateType(), False)])
    #     )
    # )

    full_grid = (
        date_df.crossJoin(districts_df)
        .withColumnRenamed("district", "district_full")
    )

    full_agg = (
        full_grid
        .join(
            agg,
            (full_grid["date"] == agg["date"]) &
            (full_grid["district_full"] == agg["district"]),
            how="left"
        )
        .select(
            full_grid["date"],
            full_grid["district_full"].alias("district"),
            F.coalesce(agg["flood_complaint_count"], F.lit(0)).cast("int").alias("flood_complaint_count")
        )
    )

    return full_agg


def run_traffy_flood_etl(
    spark: SparkSession,
    input_path: str,
    cleaned_output_path: str,
    flood_ts_output_path: str,
) -> None:
    """
    Main ETL entrypoint.

    Steps:
      1. Load raw Traffy CSV
      2. Clean & filter to Bangkok
      3. Add time columns
      4. Add flood flag
      5. Aggregate daily flood counts per district
      6. Write outputs to Parquet

    This function is designed to be called from:
      - a Spark job (spark-submit)
      - an Airflow DAG via PythonOperator or SparkSubmitOperator
    """
    # 1) Load
    df_raw = load_raw_traffy(spark, input_path)

    # 2) Clean / filter
    df_clean = clean_and_filter_bangkok(df_raw)

    # 3) Time columns
    df_clean = add_time_columns(df_clean)

    # 4) Flood flag
    df_clean = add_flood_flag(df_clean)

    # 5) Aggregate to daily × district
    df_flood_ts = aggregate_flood_daily_by_district(df_clean)

    # 6) Write outputs
    (
        df_clean
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(cleaned_output_path)
    )

    (
        df_flood_ts
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(flood_ts_output_path)
    )

    print(f"[ETL] Wrote cleaned tickets to: {cleaned_output_path}")
    print(f"[ETL] Wrote flood daily TS to: {flood_ts_output_path}")