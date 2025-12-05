#!/bin/bash
set -e

# Default to python if no argument is provided
# Usage: ./submit_job.sh spark  OR  ./submit_job.sh python
ENGINE=${1:-python}

echo ">>> [ORCHESTRATOR] Starting Flood Pipeline using engine: $ENGINE"

# 1. Setup Directories
echo ">>> [1/4] Creating Directory Structure..."
mkdir -p data/raw/BMA data/external data/processed

# 2. Download Data
echo ">>> [2/4] Downloading Raw Data..."
python3 -m jobs.download_raw_data

# 3. Scrape Meta Data
echo ">>> [3/4] Scraping Station Metadata..."
python3 -m jobs.station_name_scraping
# 4. Run ETL (The Switch)
echo ">>> [4/4] Running ETL with $ENGINE..."
export PYTHONPATH=$PYTHONPATH:.

# Pass the flag to the Python driver
python3 etl_driver.py --engine "$ENGINE"

echo ">>> [SUCCESS] Pipeline Finished."

# pip install -r requirements.txt --force-reinstall
# pip freeze > requirements.txt