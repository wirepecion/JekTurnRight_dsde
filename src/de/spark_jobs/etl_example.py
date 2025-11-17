"""
Example Spark ETL job for data processing.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, trim, when
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkETLJob:
    """
    Example Spark ETL job for extracting, transforming, and loading data.
    """

    def __init__(self, app_name: str = "ETL_Job"):
        """
        Initialize Spark session.

        Args:
            app_name: Name of the Spark application
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        logger.info(f"Spark session created: {app_name}")

    def extract_csv(self, input_path: str):
        """
        Extract data from CSV file.

        Args:
            input_path: Path to input CSV file

        Returns:
            Spark DataFrame
        """
        logger.info(f"Extracting data from: {input_path}")
        df = self.spark.read.csv(input_path, header=True, inferSchema=True)
        logger.info(f"Extracted {df.count()} rows")
        return df

    def transform_data(self, df):
        """
        Transform the data.

        Args:
            df: Input Spark DataFrame

        Returns:
            Transformed Spark DataFrame
        """
        logger.info("Applying transformations...")
        
        # Example transformations
        transformed_df = df \
            .withColumn("name_upper", upper(trim(col("name")))) \
            .withColumn("is_active", when(col("status") == "active", True).otherwise(False)) \
            .filter(col("value").isNotNull())
        
        logger.info("Transformations applied")
        return transformed_df

    def load_parquet(self, df, output_path: str, mode: str = "overwrite"):
        """
        Load data to Parquet format.

        Args:
            df: Spark DataFrame to save
            output_path: Path to output directory
            mode: Write mode (overwrite, append, etc.)
        """
        logger.info(f"Loading data to: {output_path}")
        df.write.mode(mode).parquet(output_path)
        logger.info("Data loaded successfully")

    def run_etl(self, input_path: str, output_path: str):
        """
        Run the complete ETL pipeline.

        Args:
            input_path: Path to input data
            output_path: Path to output location
        """
        try:
            # Extract
            df = self.extract_csv(input_path)
            
            # Transform
            transformed_df = self.transform_data(df)
            
            # Load
            self.load_parquet(transformed_df, output_path)
            
            logger.info("ETL job completed successfully")
        except Exception as e:
            logger.error(f"ETL job failed: {e}")
            raise
        finally:
            self.stop()

    def stop(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def example_etl():
    """
    Example ETL job execution.
    """
    etl_job = SparkETLJob("Example_ETL")
    
    # Example paths (adjust as needed)
    input_path = "data/raw/sample_data.csv"
    output_path = "data/processed/transformed_data"
    
    try:
        etl_job.run_etl(input_path, output_path)
    except Exception as e:
        print(f"ETL job failed: {e}")


if __name__ == "__main__":
    example_etl()
