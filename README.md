# JekTurnRight_dsde

**External Resources:**
* **Visualization Dashboard:** [JekTurnRight_dsde_visualize](https://github.com/ahpu9158/JekTurnRight_dsde_visualize)
* **HuggingFace Model:** [sirasira/flood-lstm-v1](https://huggingface.co/sirasira/flood-lstm-v1/tree/main)
* **FASTAPI (HF Space):** [bangkok-flood-api](https://huggingface.co/spaces/sirasira/bangkok-flood-api/blob/main/app.py)

---

## Project Structure

```text
.
├── config/               # Configuration files (e.g., logging)
├── data/
│   ├── external/         # Rainfall and station reference data
│   ├── model/            # Trained model artifacts (.pth, .bin, metrics)
│   ├── processed/        # Cleaned data and Spark outputs (parquet/csv)
│   └── raw/              # Raw Shapefiles (BMA) and scraped CSVs
├── jobs/                 # Orchestration scripts (Download, Pipeline, Scrapers)
├── notebooks/            # Jupyter notebooks for Analysis & Experiments
├── scrapers/             # Specific scraper logic
│   ├── traffy/           # Traffy Fondue data downloader
│   └── water_level/      # Water level station scrapers
├── src/                  # Main Source Code Package
│   ├── dataprep/         # Data cleaning, geo-processing, and I/O
│   ├── de/               # Data Engineering (Spark jobs & sessions)
│   ├── ds/               # Data Science (Training, Prediction, Deployment)
│   └── setting/          # Project-wide settings
├── tests/                # Unit tests
├── etl_driver.py         # Main entry point for ETL operations
├── init.sh               # Setup script (venv creation & installation)
├── submit_job.sh         # Pipeline orchestrator script
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project configuration
```

## Setup Environment

We provide a convenience script to set up the environment automatically.

### 1\. Run Initialization Script

This script creates a virtual environment (`venv`) and installs all required dependencies.

```bash
# Make the script executable (if needed)
chmod +x init.sh

# Run setup
./init.sh
```

### 2\. Activate Environment

**Important:** You must activate the virtual environment manually every time you open a new terminal.

```bash
# On macOS and Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3\. Setup Jupyter Kernel

To use this environment inside Jupyter Notebooks:

```bash
python -m ipykernel install --user --name traffy-dsde --display-name "Python (traffy-dsde)"
```

## Usage (Automated Pipeline)

The `submit_job.sh` script orchestrates the entire pipeline:

1.  Creates necessary directories (`data/raw`, `data/processed`).
2.  Downloads raw data.
3.  Scrapes station metadata.
4.  Runs the ETL process (using either Python or Spark).

### Run with Python Engine (Default)

Best for local development or small datasets.

```bash
chmod +x submit_job.sh
./submit_job.sh
```

### Run with Spark Engine

Best for processing large historical datasets.

```bash
./submit_job.sh spark
```

-----

## Manual Execution (For Developers)

If you need to run specific parts of the pipeline individually for debugging:

**1. Download Raw Data**

```bash
python -m jobs.download_raw_data
```

**2. Scrape Metadata**

```bash
python -m jobs.station_name_scraping
```

**3. Run ETL Driver Manually**

```bash
# Run with Python pandas
python etl_driver.py --engine python

# Run with PySpark
python etl_driver.py --engine spark
```

## Development

### Running Tests

```bash
pytest tests/
```

### Data Management

  * **Raw Data:** Placed in `data/raw/` (Git ignored)
  * **Processed Data:** Saved to `data/processed/` (Git ignored)
  * **Model Artifacts:** Saved to `data/model/`

## Notes

  * **Spark:** This project uses PySpark. Ensure you have Java (JDK 8 or 11) installed if running Spark jobs locally.
  * **Visualization:** The dashboard logic is separated into an external repository (linked above).