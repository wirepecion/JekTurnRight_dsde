# 1. Set the Project Root (Critical Step)
export PYTHONPATH=$PYTHONPATH:/home/sirav/JekTurnRight_dsde

# 2. Run the Job
spark-submit \
  --master local[2] \
  --driver-memory 4g \
  src/spark_jobs/traffy_etl_job.py