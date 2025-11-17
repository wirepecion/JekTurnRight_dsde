"""
Data transformation utilities.
"""
import pandas as pd
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Utilities for data transformation and cleaning.
    """

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names (lowercase, replace spaces with underscores).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned column names
        """
        df = df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        logger.info("Column names cleaned")
        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates

        Returns:
            DataFrame without duplicates
        """
        before = len(df)
        df = df.drop_duplicates(subset=subset)
        after = len(df)
        logger.info(f"Removed {before - after} duplicate rows")
        return df

    @staticmethod
    def fill_missing(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Fill missing values.

        Args:
            df: Input DataFrame
            strategy: Strategy for filling ('mean', 'median', 'mode', 'zero')

        Returns:
            DataFrame with filled missing values
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns

        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        logger.info(f"Filled missing values using strategy: {strategy}")
        return df

    @staticmethod
    def normalize_dates(df: pd.DataFrame, date_columns: List[str], format: str = "%Y-%m-%d") -> pd.DataFrame:
        """
        Normalize date columns to a specific format.

        Args:
            df: Input DataFrame
            date_columns: List of date column names
            format: Target date format

        Returns:
            DataFrame with normalized dates
        """
        df = df.copy()
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime(format)
        logger.info(f"Normalized {len(date_columns)} date columns")
        return df
