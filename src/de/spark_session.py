# src/de/spark_session.py
from __future__ import annotations

from pyspark.sql import SparkSession
from . import settings


def create_spark(app_name: str | None = None,
                 master: str | None = None) -> SparkSession:
    """
    Central place to configure Spark for all DE jobs.
    """
    app_name = app_name or settings.SPARK_APP_NAME
    master = master or settings.SPARK_MASTER

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .config("spark.driver.memory", settings.SPARK_DRIVER_MEMORY)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
