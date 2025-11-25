# src/de/settings.py
from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Try to find project root in a robust way.

    Priority:
    1) PROJECT_ROOT env var (works in notebooks, Windows, WSL).
    2) Walk up from this file until we see pyproject.toml or .git.
    3) Fallback: two levels up from this file.
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # Fallback: src/de/settings.py -> src/de -> src -> project_root
    return here.parents[2]


PROJECT_ROOT = get_project_root()

# --- Base dirs ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Raw inputs ---
BMA_SHAPE = RAW_DIR / "BMA" / "BMA_ADMIN_SUB_DISTRICT.shp"
TRAFFY_RAW_CSV = RAW_DIR / "bangkok_traffy.csv"

# --- External (metadata, rain) ---
STATION_CSV = EXTERNAL_DIR / "station.csv"
RAINFALL_CSV = EXTERNAL_DIR / "rainfall.csv"

# --- Outputs ---
FLOOD_TRAIN_PARQUET = PROCESSED_DIR / "flood_training_data_spark"
FLOOD_TRAIN_CSV = PROCESSED_DIR / "flood_training_data_csv"

# --- Spark defaults ---
SPARK_APP_NAME = "TraffyFloodETL_Production"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
