from __future__ import annotations

from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
import os
import findspark

# -----------------------------
# Config / constants
# -----------------------------
os.environ["SPARK_HOME"] = "/opt/spark"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

findspark.init()
spark = SparkSession.builder \
    .appName("JekTurnRight Analysis") \
    .getOrCreate()

FLOOD_KEYWORDS: List[str] = [
    "น้ำท่วม", "น้ำขัง", "น้ำล้น", "น้ำท่วมขัง",
    "ท่อระบายน้ำ", "ท่ออุดตัน", "ท่อระบาย", "ท่อตัน",
    "น้ำไหลย้อน", "ระบายน้ำไม่ทัน",
    "flood", "waterlogging", "drain", "drainage",
]

# -----------------------------
# Core ETL functions
# -----------------------------
def load_raw_traffy(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    print("[STEP] Loading raw Traffy CSV...")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "DROPMALFORMED")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .csv(input_path)
    )

    df = df.drop("_corrupt_record")
    return df


def clean_and_filter_bangkok(df: DataFrame) -> DataFrame:
    print("[STEP] Cleaning + schema validation + Bangkok filter...")

    df = df.withColumn(
        "timestamp_ts",
        F.expr("try_cast(timestamp as timestamp)")
    )

    df = df.filter(F.col("timestamp_ts").isNotNull())

    df = df.drop("timestamp").withColumnRenamed("timestamp_ts", "timestamp")

    text_cols = ["province", "district", "subdistrict", "comment", "type"]
    for col in text_cols:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("string"))
            df = df.filter(F.col(col).isNotNull())

    if "province" in df.columns:
        df = df.filter(
            (F.col("province").contains("กรุงเทพ")) |
            (F.lower(F.col("province")).contains("bangkok"))
        )

    if "ticket_id" in df.columns:
        df = df.dropDuplicates(["ticket_id"])

    return df


def add_time_columns(df: DataFrame) -> DataFrame:
    print("[STEP] Adding time columns...")

    df = df.withColumn("date", F.to_date("timestamp"))
    df = df.withColumn("year", F.year("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    df = df.withColumn("day", F.dayofmonth("timestamp"))
    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("weekday", F.dayofweek("timestamp"))
    return df


def add_flood_flag(df: DataFrame) -> DataFrame:
    print("[STEP] Adding flood_flag column...")

    comment_col = F.lower(F.coalesce(F.col("comment").cast("string"), F.lit("")))
    type_col = F.lower(F.coalesce(F.col("type").cast("string"), F.lit("")))
    merged_text = F.concat_ws(" ", comment_col, type_col)

    flood_cond = F.lit(False)
    for kw in FLOOD_KEYWORDS:
        flood_cond = flood_cond | merged_text.contains(kw.lower())

    df = df.withColumn("is_flood", flood_cond.cast(T.BooleanType()))
    return df


def aggregate_flood_daily_by_district(df: DataFrame) -> DataFrame:
    print("[STEP] Aggregating daily flood counts per district...")

    df_flood = df.filter(F.col("is_flood") == True)

    df_flood = df_flood.filter(
        F.col("district").isNotNull() & F.col("date").isNotNull()
    )

    # if df_flood.rdd.isEmpty():
    #     print("[WARNING] No flood rows found after filtering.")
    #     empty_schema = T.StructType([
    #         T.StructField("date", T.DateType(), True),
    #         T.StructField("district", T.StringType(), True),
    #         T.StructField("flood_complaint_count", T.IntegerType(), True),
    #     ])
    #     return df_flood.sparkSession.createDataFrame(
    #         df_flood.sparkSession.sparkContext.emptyRDD(), empty_schema
    #     )

    agg = (
        df_flood
        .groupBy("date", "district")
        .agg(F.count("ticket_id").alias("flood_complaint_count"))
    )

    return agg


def run_traffy_flood_etl(
    spark: SparkSession,
    input_path: str,
    cleaned_output_path: str,
    flood_ts_output_path: str,
) -> None:
    print("============================================")
    print("🚀 RUNNING TRAFFY FLOOD ETL PIPELINE")
    print("============================================")

    # 1) Load
    df_raw = load_raw_traffy(spark, input_path)

    # 2) Clean / filter
    df_clean = clean_and_filter_bangkok(df_raw)

    # 3) Time columns
    df_clean = add_time_columns(df_clean)

    # 4) Flood flag
    df_clean = add_flood_flag(df_clean)

    # 5) Aggregate TS
    df_flood_ts = aggregate_flood_daily_by_district(df_clean)

    # 6) Write outputs
    print("[STEP] Writing cleaned tickets...")
    (
        df_clean
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(cleaned_output_path)
    )

    print("[STEP] Writing daily flood time series...")
    (
        df_flood_ts
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(flood_ts_output_path)
    )

    print("============================================")
    print("✅ ETL COMPLETED")
    print("   Cleaned data:", cleaned_output_path)
    print("   Flood TS:", flood_ts_output_path)
    print("============================================")


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("TraffyFloodETL")
        .getOrCreate()
    )

    RAW_DIR = "data/raw/bangkok_traffy.csv"
    CLEANED_OUT = "data/processed/traffy_clean.parquet"
    FLOOD_TS_OUT = "data/processed/flood_daily_by_district.parquet"

    run_traffy_flood_etl(
        spark,
        input_path=RAW_DIR,
        cleaned_output_path=CLEANED_OUT,
        flood_ts_output_path=FLOOD_TS_OUT,
    )

    spark.stop()

# uv run python pipelines/jobs/traffy_flood_etl.py