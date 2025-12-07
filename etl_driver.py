import argparse
import sys
import logging
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_spark_etl():
    logger.info(">>> ENGINE SELECTED: APACHE SPARK")
    try:
        # Lazy import: Only load Spark if actually requested
        from src.de.spark_jobs.traffy_etl_job import main as spark_main
        spark_main()
    except ImportError as e:
        logger.error("Failed to import Spark modules. Is PySpark installed?")
        raise e

def run_python_etl():
    logger.info(">>> ENGINE SELECTED: PURE PYTHON (PANDAS)")
    try:
        # Lazy import
        from jobs.python_data_pipeline import main as python_main
        python_main()
    except ImportError as e:
        logger.error("Failed to import Python pipeline modules.")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Flood Data ETL Driver")
    
    # The 'switch' you asked for
    parser.add_argument(
        "--engine", 
        choices=["spark", "python"], 
        default="python",
        help="Choose the processing engine: 'spark' for big data, 'python' for small/local."
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.engine == "spark":
        run_spark_etl()
    elif args.engine == "python":
        run_python_etl()
        
    duration = time.time() - start_time
    logger.info(f">>> JOB FINISHED. Duration: {duration:.2f} seconds.")

if __name__ == "__main__":
    main()