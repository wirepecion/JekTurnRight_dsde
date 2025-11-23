# src/common/dataprep/mergers.py
from __future__ import annotations

import logging

import pandas as pd

from .services import TraffyDataCleaner
from .geo_utils import GeoMatcher

logger = logging.getLogger(__name__)


class StationRainMerger:
    """
    Cleaning & merging for station info and rainfall.
    """

    @staticmethod
    def clean_station(station_df: pd.DataFrame) -> pd.DataFrame:
        df = station_df.copy()
        logger.info("Cleaning station metadata")
        df = df.rename(
            columns={
                "StationCode": "station_code",
                "StationName_Short": "station_name",
                "DistrictName": "district",
                "Latitude": "latitude",
                "Longitude": "longitude",
                "LastReadingTime": "last_reading_time",
                "DistrictCode_from_JS": "districtCode_from_js",
            }
        )

        district_replacements = {
            "ป้อมปราบฯ": "ป้อมปราบศัตรูพ่าย",
            "ราษฏร์บูรณะ": "ราษฎร์บูรณะ",
        }
        df["district"] = df["district"].replace(district_replacements)
        df = df[df["district"] != "อำเภอเมืองสมุทรปราการ"].copy()

        if df["station_code"].str.len().eq(9).all():
            df.loc[:, "station_code"] = df["station_code"].str[3:]

        df = df[df["station_code"] != "PYT.02"].copy()
        return df

    @staticmethod
    def clean_rainfall(rain_df: pd.DataFrame) -> pd.DataFrame:
        df = rain_df.copy()
        logger.info("Cleaning rainfall pivot table")
        not_use_station = ["NKM.03", "LSI.02", "MBR.03", "NJK.04", "SPK.01"]

        df = df.drop(columns=not_use_station, errors="ignore")
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            df = TraffyDataCleaner.use_simple_imputer(
                strategy="most_frequent", data=df, column=nan_cols
            )

        df = df.rename(columns={"Date": "date"})
        rain_expand_df = df.melt(
            id_vars="date", var_name="station_code", value_name="rainfall"
        )
        rain_expand_df["date"] = pd.to_datetime(
            rain_expand_df["date"], format="mixed", errors="coerce"
        )
        rain_expand_df["date"] = rain_expand_df["date"].dt.date
        return rain_expand_df

    @staticmethod
    def get_station_code_for_rain(rain_df: pd.DataFrame, shape_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Attaching station_code with subdistrict/district for rainfall")
        data = rain_df.merge(
            shape_df[["subdistrict", "district", "station_code", "latitude", "longitude"]],
            how="left",
            on="station_code",
        )
        data = data.dropna(subset=["subdistrict", "district"])
        return data

    @staticmethod
    def get_result_of_merge(report_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging rainfall with aggregated flood reports")
        data = rain_df.merge(
            report_df[
                [
                    "date",
                    "station_code",
                    "number_of_report_flood",
                    "total_report",
                    "district",
                    "subdistrict",
                ]
            ],
            on=["date", "station_code", "district", "subdistrict"],
            how="left",
        )
        data = TraffyDataCleaner.use_simple_imputer(
            strategy="constant",
            data=data,
            column=["number_of_report_flood", "total_report"],
            fill_value=0,
        )
        data = data[
            [
                "date",
                "district",
                "subdistrict",
                "station_code",
                "latitude",
                "longitude",
                "number_of_report_flood",
                "total_report",
                "rainfall",
            ]
        ]
        return data
