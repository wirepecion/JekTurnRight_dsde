from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, lower, regexp_replace

from src.common.config import RAW_DIR, PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPARK_LOCAL_DIR = PROJECT_ROOT / "spark_tmp"
SPARK_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

WAREHOUSE_DIR = PROJECT_ROOT / "spark_warehouse"
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "bangkok_traffy.csv"
OUTPUT_DIR = PROCESSED_DIR / "traffy_cleaned_parquet"
OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)


def build_spark():
    spark = (
        SparkSession.builder
        .appName("TraffyBasicETL")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.local.dir", str(SPARK_LOCAL_DIR))
        .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
        .getOrCreate()
    )
    return spark



def standardize_column_names(df):
    """
    Make all column names lower_snake_case:
    - lowercase
    - spaces and weird chars -> underscore
    """
    new_cols = [
        regexp_replace(lower(col_name), r"[^a-z0-9]+", "_").alias(col_name)
        for col_name in df.columns
    ]

    # new_cols is just expressions; we need mapping manually:
    mapping = {}
    for old in df.columns:
        new = old.lower()
        # Replace anything not a-z0-9 with underscore
        import re
        new = re.sub(r"[^a-z0-9]+", "_", new).strip("_")
        mapping[old] = new

    for old, new in mapping.items():
        df = df.withColumnRenamed(old, new)

    return df


def basic_profiling(df):
    """
    Print out some basic info: schema, counts, sample rows, etc.
    """
    print("=== Schema ===")
    df.printSchema()

    print("=== Row count ===")
    print(df.count())

    print("=== Sample rows ===")
    df.show(5, truncate=False)

    print("=== Null counts (first 20 columns) ===")
    for c in df.columns[:20]:
        null_count = df.filter(col(c).isNull()).count()
        print(f"{c}: {null_count} nulls")

    # If there is a 'category' or 'problem_type'-like column, show top values
    candidate_cols = [c for c in df.columns if "type" in c or "category" in c]
    if candidate_cols:
        col_name = candidate_cols[0]
        print(f"=== Top categories by count ({col_name}) ===")
        (
            df.groupBy(col_name)
            .count()
            .orderBy(col("count").desc())
            .show(10, truncate=False)
        )


def simple_clean(df):
    """
    Example cleaning:
    - standardize column names
    - try to parse timestamp column (if present)
    - cast lat/lon to double if present
    - drop exact duplicates
    """
    df = standardize_column_names(df)

    # Try to find a timestamp-like column
    # ts_candidates = [c for c in df.columns if "time" in c or "date" in c]
    # if ts_candidates:
    #     ts_col = ts_candidates[0]
    #     print(f"[Clean] Parsing timestamp column: {ts_col}")
    #     # Change this format string once you see the real data
    #     df = df.withColumn(ts_col, to_timestamp(col(ts_col)))
    ts_candidates = [c for c in df.columns if "time" in c or "date" in c]
    if ts_candidates:   
        ts_col = ts_candidates[0]
        print(f"[Clean] Parsing timestamp column (tolerant): {ts_col}")
        df = df.withColumn(ts_col, expr(f"try_cast({ts_col} AS TIMESTAMP)"))

    # Try to find lat/lon columns
    lat_candidates = [c for c in df.columns if "lat" in c]
    lon_candidates = [c for c in df.columns if "lon" in c or "lng" in c]

    if lat_candidates and lon_candidates:
        lat_col = lat_candidates[0]
        lon_col = lon_candidates[0]
        print(f"[Clean] Casting lat/lon columns: {lat_col}, {lon_col}")
        df = df.withColumn(lat_col, col(lat_col).cast("double"))
        df = df.withColumn(lon_col, col(lon_col).cast("double"))

    # Drop exact duplicate rows
    print("[Clean] Dropping exact duplicates")
    before = df.count()
    df = df.dropDuplicates()
    after = df.count()
    print(f"[Clean] Dropped {before - after} duplicate rows")

    return df


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    spark = build_spark()

    print(f"[ETL] Reading raw Traffy data from: {RAW_FILE}")
    df = (
        spark.read
        .option("header", True)
        # .option("inferSchema", True)
        .csv(str(RAW_FILE))
    )

    print("[ETL] Basic profiling on raw data")
    basic_profiling(df)

    print("[ETL] Cleaning data...")
    df_clean = simple_clean(df)

    print("[ETL] Basic profiling on cleaned data")
    basic_profiling(df_clean)

    # Save as Parquet (good for Spark)
    output_path = str(OUTPUT_DIR)
    print(f"[ETL] Writing cleaned data to: {output_path}")
    (
        df_clean
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    print("[ETL] Done.")
    spark.stop()


if __name__ == "__main__":
    main()
