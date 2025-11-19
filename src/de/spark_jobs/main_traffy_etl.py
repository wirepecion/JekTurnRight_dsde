from pyspark.sql import SparkSession
from traffy_flood_etl import run_traffy_flood_etl

if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("TraffyFloodETL")
        .getOrCreate()
    )

    INPUT = "data/raw/bangkok_traffy.csv"
    CLEANED_OUT = "data/processed/traffy_clean.parquet"
    FLOOD_TS_OUT = "data/processed/flood_daily_by_district.parquet"

    run_traffy_flood_etl(
        spark,
        input_path=INPUT,
        cleaned_output_path=CLEANED_OUT,
        flood_ts_output_path=FLOOD_TS_OUT,
    )

    spark.stop()

    # spark-submit main_traffy_etl.py
    
    # certutil -hashfile spark-4.0.1-bin-hadoop3.tgz SHA512

    # curl --ssl-no-revoke -L -o C:\hadoop\bin\winutils.exe https://github.com/cdarlint/winutils/raw/master/hadoop-4.0.1/bin/winutils.exe

    # setx SPARK_HOME "C:\Spark\spark-4.0.1-bin-hadoop3"

    # "C:\Program Files\7-Zip\7z.exe" x "C:\Users\sirav\Downloads\spark-4.0.1-bin-hadoop3.tgz" -oC:\Spark
    
    # C:\Program Files\Java\jdk-21\bin\java.exe

    # $jdkPath = "C:\Program Files\Java\jdk-21\bin" # Replace with your actual   path
    # [System.Environment]::SetEnvironmentVariable("JAVA_HOME", $jdkPath, "Machine")
    # [System.Environment]::SetEnvironmentVariable("PATH", "$env:Path;$jdkPath\bin", "Machine")

