"""
Data loading utilities for various formats.
"""
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Utilities for loading data from various sources and formats.
    """

    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory to load/save data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading CSV from: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from {filename}")
        return df

    def save_csv(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            filename: Name of the CSV file
            **kwargs: Additional arguments for df.to_csv
        """
        filepath = self.data_dir / filename
        logger.info(f"Saving CSV to: {filepath}")
        df.to_csv(filepath, index=False, **kwargs)
        logger.info(f"Saved {len(df)} rows to {filename}")

    def load_parquet(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from Parquet file.

        Args:
            filename: Name of the Parquet file
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading Parquet from: {filepath}")
        df = pd.read_parquet(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from {filename}")
        return df

    def save_parquet(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        """
        Save DataFrame to Parquet file.

        Args:
            df: DataFrame to save
            filename: Name of the Parquet file
            **kwargs: Additional arguments for df.to_parquet
        """
        filepath = self.data_dir / filename
        logger.info(f"Saving Parquet to: {filepath}")
        df.to_parquet(filepath, index=False, **kwargs)
        logger.info(f"Saved {len(df)} rows to {filename}")

    def load_json(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from JSON file.

        Args:
            filename: Name of the JSON file
            **kwargs: Additional arguments for pd.read_json

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading JSON from: {filepath}")
        df = pd.read_json(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from {filename}")
        return df

    def save_json(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        """
        Save DataFrame to JSON file.

        Args:
            df: DataFrame to save
            filename: Name of the JSON file
            **kwargs: Additional arguments for df.to_json
        """
        filepath = self.data_dir / filename
        logger.info(f"Saving JSON to: {filepath}")
        df.to_json(filepath, orient='records', **kwargs)
        logger.info(f"Saved {len(df)} rows to {filename}")
