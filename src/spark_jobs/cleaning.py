"""
src/spark_jobs/cleaning.py
--------------------------
Spark-native implementation of cleaning logic.
Note: We use Lazy Evaluation (Expressions), not immediate execution.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType

# drop cols ['photo', 'photo_after', 'star']

def drop_not_used_columns(df: DataFrame, subset_cols: list) -> DataFrame:
    '''
    Drops columns not in subset_cols.
    Pandas: df.drop(columns=...) -> Spark: df.drop(*columns)
    '''
    return df.drop(*subset_cols)

def drop_nan_rows(df: DataFrame, subset_cols: list) -> DataFrame:
    """Pandas: df.dropna(subset=...) -> Spark: df.na.drop(subset=...)"""
    return df.na.drop(subset=subset_cols)

def convert_types(df: DataFrame) -> DataFrame:
    """
    Pandas: pd.to_datetime(..., errors='coerce')
    Spark: try_cast(col as timestamp)
    
    Why: The CSV contains '0' or garbage strings. 'to_timestamp' might crash 
    depending on Spark ANSI settings. 'try_cast' safely returns NULL.
    """
    return df \
        .withColumn("timestamp", F.expr("try_cast(timestamp as timestamp)")) \
        .withColumn("last_activity", F.expr("try_cast(last_activity as timestamp)"))

def extract_time_features(df: DataFrame) -> DataFrame:
    """
    Pandas: df['dt'].dt.year
    Spark: F.year(F.col('dt'))
    """
    return df \
        .withColumn("year_timestamp", F.year("timestamp")) \
        .withColumn("month_timestamp", F.month("timestamp")) \
        .withColumn("days_timestamp", F.dayofmonth("timestamp"))

def parse_coordinates(df: DataFrame) -> DataFrame:
    """
    Pandas: str.split(expand=True)
    Spark: F.split().getItem() with SAFETY CHECK
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
    Pandas: df[df['province'].str.contains("...")]
    Spark: F.col().contains()
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
    Pandas: apply(lambda x: x.replace(...)) -> Slow!
    Spark: regexp_replace -> Fast (C++ speed)
    """
    # Remove '{}', quotes, and spaces
    clean_str = F.regexp_replace(F.col("type"), r"[\{\}\'\"\s]", "")
    
    # Split by comma into an ARRAY
    # Filter out empty strings from the array
    return df \
        .filter(F.col("type") != "{}") \
        .withColumn("type_list", F.split(clean_str, ",")) \
        .withColumn("type_list", F.expr("filter(type_list, x -> x != '')"))

