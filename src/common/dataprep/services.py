# src/common/dataprep/services.py
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class TraffyDataCleaner:
    """
    Stateless cleaning utilities for Traffy Fondue data:
    - drop NaNs in crucial columns
    - convert dates
    - extract coords
    - filter by year
    - Bangkok-only
    - parse 'type' column
    """

    REQUIRED_COLS: List[str] = [
        "ticket_id",
        "type",
        "organization",
        "coords",
        "province",
        "timestamp",
        "last_activity",
    ]

    @staticmethod
    def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Dropping rows with NaN in required columns")
        return df.dropna(subset=TraffyDataCleaner.REQUIRED_COLS)

    @staticmethod
    def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp and last_activity to datetime.
        """
        df = df.copy()
        logger.info("Converting timestamp and last_activity to datetime")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
        df["last_activity"] = pd.to_datetime(df["last_activity"], format="ISO8601", errors="coerce")
        df = df.dropna(subset=["timestamp"])
        return df

    @staticmethod
    def convert_to_date(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp to date column 'date', keep last_activity as date.
        """
        df = df.copy()
        logger.info("Converting timestamp & last_activity to pure dates")
        df["timestamp"] = df["timestamp"].dt.date
        df.rename(columns={"timestamp": "date"}, inplace=True)
        df["last_activity"] = df["last_activity"].dt.date
        return df

    @staticmethod
    def add_crucial_cols(df: pd.DataFrame) -> pd.DataFrame:
        """
        - year_timestamp, month_timestamp, days_timestamp
        - extract longitude/latitude from 'coords'
        """
        df = df.copy()
        logger.info("Adding time columns and splitting coords")
        df["year_timestamp"] = df["timestamp"].dt.year
        df["month_timestamp"] = df["timestamp"].dt.month
        df["days_timestamp"] = df["timestamp"].dt.day
        # also add alias 'day_timestamp' to be safe with old code
        df["day_timestamp"] = df["days_timestamp"]

        df[["longitude", "latitude"]] = df["coords"].str.split(",", expand=True).astype(float)
        return df

    @staticmethod
    def drop_not_used_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Dropping columns: {cols}")
        return df.drop(columns=cols, axis="columns", errors="ignore")

    @staticmethod
    def filter_year(df: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Filtering by year in [{start}, {stop}]")
        return df[(df["year_timestamp"] >= start) & (df["year_timestamp"] <= stop)]

    @staticmethod
    def drop_not_use_province(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only Bangkok rows.
        """
        df = df.copy()
        logger.info("Normalizing Bangkok province names and filtering")
        df.loc[df["province"].str.contains("กรุงเทพ"), "province"] = "กรุงเทพมหานคร"
        df = df[df["province"] == "กรุงเทพมหานคร"]
        return df

    @staticmethod
    def parse_type_string(text):
        if not isinstance(text, str) or pd.isna(text):
            return []
        text = text.replace("{", "").replace("}", "").replace("'", "").replace('"', "")
        items = text.split(",")
        cleaned_items = [item.strip() for item in items if item.strip()]
        return cleaned_items

    @staticmethod
    def clean_type_columns(df: pd.DataFrame, explode: bool = False) -> pd.DataFrame:
        """
        - remove rows where type == '{}'
        - add 'type_list' (list of tags)
        - optional: explode to 1 row per tag
        """
        df = df.copy()
        logger.info("Cleaning 'type' column and creating 'type_list'")
        df = df[df["type"] != "{}"]
        df["type_list"] = df["type"].apply(TraffyDataCleaner.parse_type_string)

        if explode:
            df = df.explode("type_list")

        return df

    @staticmethod
    def use_simple_imputer(
        strategy: str,
        data: pd.DataFrame,
        column: list,
        fill_value=None,
    ) -> pd.DataFrame:
        df = data.copy()
        logger.info(f"Imputing columns {column} with strategy={strategy}")
        if strategy == "constant":
            imp = SimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            imp = SimpleImputer(strategy=strategy)
        df[column] = imp.fit_transform(df[column])
        return df

    @staticmethod
    def get_report_detail(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily flood-related metrics per (district, subdistrict).
        """
        data = df.copy()
        logger.info("Building daily flood report counts per subdistrict")
        data["flood"] = data["type"].str.contains("น้ำท่วม")

        data = (
            data.groupby(
                [
                    "date",
                    "year_timestamp",
                    "month_timestamp",
                    "day_timestamp",
                    "district",
                    "subdistrict",
                ]
            )
            .agg(
                number_of_report_flood=("flood", "sum"),
                total_report=("flood", "count"),
            )
            .reset_index()
        )
        return data


class AnalyticsService:
    """
    Insight utilities: co-occurrence, exploding tags, heatmap.
    """

    @staticmethod
    def co_occurrence_analysis(df: pd.DataFrame, correlation_check__tag: str) -> pd.Series:
        """
        Compute co-occurrence counts of tags that appear together
        with `correlation_check__tag` inside the 'type' string.
        """
        logger.info(f"Running co-occurrence analysis for tag: {correlation_check__tag}")
        s = df["type"].value_counts()
        s_tag = s[s.index.str.contains(correlation_check__tag)]
        co_occurrence_counts = {}

        for tag_set_str, count in s_tag.items():
            tags = tag_set_str.strip("{}").split(",")
            if len(tags) == 1:
                co_occurrence_counts[correlation_check__tag] = (
                    co_occurrence_counts.get(correlation_check__tag, 0) + count
                )
            else:
                for tag in tags:
                    tag = tag.strip()
                    if tag and tag != correlation_check__tag:
                        co_occurrence_counts[tag] = co_occurrence_counts.get(tag, 0) + count

        return pd.Series(co_occurrence_counts).sort_values(ascending=False)

    @staticmethod
    def explode_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Explode 'type' column into rows of 'type_list'.
        """
        logger.info("Exploding 'type' column into 'type_list'")
        df_cleaned = df.dropna(subset=["type"]).copy()
        df_cleaned["type_list"] = df_cleaned["type"].apply(TraffyDataCleaner.parse_type_string)
        df_cleaned = df_cleaned.explode("type_list")
        return df_cleaned
