# We set PYTHONPATH to current directory (.) so imports work
export PYTHONPATH=$PYTHONPATH:/home/sirav/JekTurnRight_dsde

# Run the job
spark-submit \
  --master local[*] \
  --driver-memory 4g \
  jobs/spark/traffy_etl_job.py