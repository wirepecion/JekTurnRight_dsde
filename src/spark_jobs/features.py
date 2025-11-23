"""
src/spark_jobs/features.py
--------------------------
Feature Engineering on Spark DataFrames.
Calculates:
1. Seasonality (Sine/Cosine of Month)
2. Soil Memory (Antecedent Precipitation Index - API) using Window Functions.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
import numpy as np

def add_seasonality(df: DataFrame) -> DataFrame:
    """
    Adds cyclical time features so the model understands 
    that "December" is close to "January".
    """
    return df \
        .withColumn("month_sin", F.sin(2 * np.pi * F.col("month_timestamp") / 12)) \
        .withColumn("month_cos", F.cos(2 * np.pi * F.col("month_timestamp") / 12))

def add_soil_memory(df: DataFrame) -> DataFrame:
    """
    Calculates API (Antecedent Precipitation Index).
    Logic: Rolling average of rainfall over 30, 60, 90 days.
    
    Why: Floods happen when ground is already saturated. 
    Rain today matters more if it rained yesterday.
    """
    # Cast to double to ensure math precision
    df = df.withColumn("rain_val", F.col("rainfall").cast("double"))

    # Define the Sliding Windows
    # Partition by Station (PHYSICAL location), Order by Time
    w_spec = Window.partitionBy("station_code").orderBy("timestamp")

    # rowsBetween(-30, 0) includes current row + 30 previous rows
    return df \
        .withColumn("API_30d", F.avg("rain_val").over(w_spec.rowsBetween(-30, 0))) \
        .withColumn("API_60d", F.avg("rain_val").over(w_spec.rowsBetween(-60, 0))) \
        .withColumn("API_90d", F.avg("rain_val").over(w_spec.rowsBetween(-90, 0))) \
        .drop("rain_val") # Clean up temp column