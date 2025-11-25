"""
src/spark_jobs/cleaning.py
--------------------------
Spark-native implementation of cleaning logic.
Includes defensive coding for dirty data (bad dates, missing coords).
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType

def drop_nan_rows(df: DataFrame, subset_cols: list) -> DataFrame:
    """
    Pandas: df.dropna(subset=...)
    Spark: df.na.drop(subset=...)
    """
    return df.na.drop(subset=subset_cols)

def convert_types(df: DataFrame) -> DataFrame:
    """
    Pandas: pd.to_datetime(..., errors='coerce')
    Spark: try_cast(col as timestamp)
    
    Fixes: [CAST_INVALID_INPUT] errors where timestamp is '0' or garbage.
    """
    return df \
        .withColumn("timestamp", F.expr("try_cast(timestamp as timestamp)")) \
        .withColumn("last_activity", F.expr("try_cast(last_activity as timestamp)"))

def extract_time_features(df: DataFrame) -> DataFrame:
    """
    Extracts Year, Month, Day from timestamp.
    """
    return df \
        .withColumn("year_timestamp", F.year("timestamp")) \
        .withColumn("month_timestamp", F.month("timestamp")) \
        .withColumn("days_timestamp", F.dayofmonth("timestamp"))

def parse_coordinates(df: DataFrame) -> DataFrame:
    """
    Pandas: str.split(expand=True)
    Spark: F.split().getItem() with SAFETY CHECK
    
    Fixes: [INVALID_ARRAY_INDEX] errors where coords string is empty or incomplete.
    """
    # 1. Split '100.5,13.7' into an array
    coords = F.split(F.col("coords"), ",")
    
    # 2. Extract with Safety Checks
    # Logic: If array has fewer than 2 items, return NULL instead of Crashing.
    return df \
        .withColumn("longitude", 
                    F.when(F.size(coords) >= 2, coords.getItem(0).cast(DoubleType()))
                     .otherwise(F.lit(None))) \
        .withColumn("latitude", 
                    F.when(F.size(coords) >= 2, coords.getItem(1).cast(DoubleType()))
                     .otherwise(F.lit(None)))

def clean_province_name(df: DataFrame) -> DataFrame:
    """
    Normalizes 'Bangkok' variations and filters for it.
    """
    # 1. Update name where it contains 'กรุงเทพ'
    df = df.withColumn(
        "province",
        F.when(F.col("province").contains("กรุงเทพ"), "กรุงเทพมหานคร")
         .otherwise(F.col("province"))
    )
    # 2. Filter only Bangkok
    return df.filter(F.col("province") == "กรุงเทพมหานคร")

def parse_type_column(df: DataFrame) -> DataFrame:
    """
    Cleans the 'type' column (e.g., remove brackets, quotes).
    """
    # Remove '{}', quotes, and spaces using Regex
    clean_str = F.regexp_replace(F.col("type"), r"[\{\}\'\"\s]", "")
    
    # Split by comma into an ARRAY
    # Filter out empty strings from the array
    return df \
        .filter(F.col("type") != "{}") \
        .withColumn("type_list", F.split(clean_str, ",")) \
        .withColumn("type_list", F.expr("filter(type_list, x -> x != '')"))