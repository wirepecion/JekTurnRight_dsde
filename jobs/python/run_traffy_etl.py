# jobs/python/run_traffy_etl.py
from src.de.spark_jobs.traffy_etl_job import main

if __name__ == "__main__":
    main()

# in project root
# python -m jobs.python.run_traffy_etl
# or
# python jobs/python/run_traffy_etl.py
