# 🌊 Bangkok Flood Prediction Pipeline (JekTurnRight)

> **An End-to-End Data Engineering & Machine Learning Pipeline predicting flood risks in Bangkok using Citizen Reports (Traffy Fondue) and Rainfall Data.**

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![Spark](https://img.shields.io/badge/Apache_Spark-ETL-orange?style=for-the-badge&logo=apachespark)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi)

## 📖 About The Project

This repository contains the **Core Pipeline** for the "Integrating Traffy Fondue Reports for Flooding Prediction" project (DSDE Class, Chulalongkorn University).

**The Problem:** Flooding in Bangkok is a critical issue. Traditional sensors are limited, but citizens actively report issues via the **Traffy Fondue** platform.
**The Solution:** We built a system that combines these citizen reports with official rainfall data to predict flood risks at the subdistrict level using Deep Learning (LSTM).

### 🔗 Ecosystem & Links
This repository handles Data Engineering and Modeling. The full system includes:
* **📊 Visualization Dashboard:** [JekTurnRight_dsde_visualize](https://github.com/ahpu9158/JekTurnRight_dsde_visualize) - (Frontend Streamlit).
* **🧠 Trained Model:** [HuggingFace Model Hub](https://huggingface.co/sirasira/flood-lstm-v1/tree/main) - (Artifacts).
* **🚀 Inference API:** [Bangkok Flood API](https://huggingface.co/spaces/sirasira/bangkok-flood-api/blob/main/app.py) - (FastAPI on HF Spaces).

---

## ⚙️ The Pipeline Architecture

This repository implements the following 5-phase pipeline:

1.  **Data Ingestion (Scraping):**
    * **Traffy Fondue:** Scrapes citizen complaint data (specifically "Flood" and "Road" categories).
    * **Rainfall Data:** Scrapes daily rainfall records from the Bangkok Drainage and Sewerage Department (DDS).
    * **Water Stations:** Scrapes metadata for 100+ water measurement stations.

2.  **Data Engineering (Spark ETL):**
    * Uses **Apache Spark** to handle large historical datasets (2022-2024).
    * **Spatial Joins:** Maps report coordinates and rain station locations to specific Bangkok subdistricts.
    * **Feature Engineering:** Calculates rolling window statistics (30d, 60d rainfall averages) and seasonality (sine/cosine time features).
      
3.  **Model Training (LSTM):**
    * Utilizes a **Long Short-Term Memory (LSTM)** network to capture temporal dependencies (e.g., saturated soil from previous days' rain).
    * **Loss Function:** Optimized using **F2 Score** to prioritize *Recall* (minimizing missed flood warnings) over Precision.
    * **Tuning:** Hyperparameter optimization using **Optuna**.

4.  **Deployment:**
    * Serves the trained model via **FastAPI** for real-time inference.

---

## 📂 Project Structure

```text
.
├── config/               # Configuration files (logging, paths)
├── data/
│   ├── external/         # Rainfall and station reference data
│   ├── model/            # Trained model artifacts (.pth, .bin, metrics)
│   ├── processed/        # Cleaned data and Spark outputs (parquet/csv)
│   └── raw/              # Raw Shapefiles (BMA) and scraped CSVs
├── jobs/                 # Orchestration scripts (Download, Pipeline, Scrapers)
├── notebooks/            # Jupyter notebooks for EDA & Experiments
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
````

-----

## 💻 Setup & Installation

We provide a convenience script to set up the environment automatically.

### 1\. Run Initialization Script

This script creates a virtual environment (`venv`) and installs all required dependencies.

```bash
# Make the script executable
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

-----

## 🚀 Usage (Automated Pipeline)

The `submit_job.sh` script orchestrates the entire pipeline:

1.  Creates necessary directories (`data/raw`, `data/processed`).
2.  Downloads raw data.
3.  Scrapes station metadata.
4.  Runs the ETL process.

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

## 🛠 Manual Execution (For Developers)

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

-----

## 📊 Model Performance

  * **Algorithm:** LSTM (Long Short-Term Memory)
  * **Metric:** F2 Score (Beta=2)
  * **Performance:** The model achieved a Best F2 Score of **0.48**.
  * **Strategy:** We utilized dynamic thresholds for Wet vs. Dry seasons to maximize safety (Recall), ensuring flood events are rarely missed even if it generates some false alarms.

-----

## 👨‍💻 Contributors

**Team Jek TurnRight**

  * Patcharapon Srisuwan
  * Jedsada Meesuk
  * Siravut Chunu
  * Titiporn Somboon

*Submitted for 2110403 Data Science and Data Engineering (DSDE-CEDT), Chulalongkorn University.*
