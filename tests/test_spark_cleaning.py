"""
test_spark_cleaning.py
----------------------
Quick test to see if Spark Logic matches our expectations.
"""
from pyspark.sql import SparkSession
from src.spark_jobs import cleaning


def main():
    # 1. Start Spark
    print(">>> Starting Spark...")
    spark = SparkSession.builder \
        .appName("TestCleaning") \
        .master("local[*]") \
        .getOrCreate()

    # 2. Load Raw CSV
    print(">>> Reading CSV...")
    df_raw = spark.read.option("header", "true").csv("data/raw/bangkok_traffy.csv")

    # 3. Apply Transformations (Chaining)
    print(">>> Applying Cleaning Logic...")
    df_clean = (df_raw
        .transform(lambda df: cleaning.drop_nan_rows(df, ["coords", "timestamp"]))
        .transform(cleaning.convert_types)
        .transform(cleaning.extract_time_features)
        .filter("year_timestamp >= 2022 AND year_timestamp <= 2024")
        .transform(cleaning.parse_coordinates)
        .transform(cleaning.clean_province_name)
        .transform(cleaning.parse_type_column)
    )

    # 4. Action: Show results
    print(">>> Results:")
    df_clean.select("timestamp", "province", "latitude", "longitude", "type_list").show(5, truncate=False)
    
    print(f"Total Cleaned Rows: {df_clean.count()}")
    spark.stop()

if __name__ == "__main__":
    main()