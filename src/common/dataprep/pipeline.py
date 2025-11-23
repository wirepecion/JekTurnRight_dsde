from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .io import DataIO
from .services import TraffyDataCleaner
from .geo_utils import GeoMatcher
from .mergers import StationRainMerger

logger = logging.getLogger(__name__)


class FloodDataPipeline:
    """
    High-level pipeline to:
      1) Clean raw Traffy data
      2) Attach spatial labels (district/subdistrict)
      3) Merge with scraped station & rainfall data
    """

    def __init__(
        self,
        shape_path: str,
        csv_to_check_shape_path: Optional[str] = None,
        year_start: int = 2022,
        year_stop: int = 2024,
    ) -> None:
        self.shape_path = str(shape_path)
        self.csv_to_check_shape_path = csv_to_check_shape_path
        self.year_start = year_start
        self.year_stop = year_stop

        logger.info(f"Initializing FloodDataPipeline with shape_path={shape_path}")
        self.shape_gdf = GeoMatcher.get_shape_file(self.shape_path)

        self.check_df: Optional[pd.DataFrame] = None
        if csv_to_check_shape_path is not None:
            self.check_df = DataIO.read_csv(csv_to_check_shape_path)
            self.shape_gdf = GeoMatcher.verify_geopandas(self.shape_gdf, self.check_df)

    # ------------------------------------------------------------------
    # BASE CLEANING
    # ------------------------------------------------------------------

    def clean_base(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply:
          - drop rows with missing crucial fields
          - convert datetimes
          - extract coords
          - add time columns
          - filter by year
          - drop some columns (photo, photo_after, star)
          - keep only Bangkok
          - verify coordinates with shapefile
          - clean type column (add 'type_list')
        """
        logger.info("Starting base cleaning pipeline for Traffy data")

        df = TraffyDataCleaner.drop_nan_rows(df_raw)
        df = TraffyDataCleaner.convert_data_types(df)
        df = TraffyDataCleaner.add_crucial_cols(df)
        df = TraffyDataCleaner.filter_year(df, start=self.year_start, stop=self.year_stop)
        df = TraffyDataCleaner.drop_not_used_columns(df, cols=["photo", "photo_after", "star"])
        df = TraffyDataCleaner.drop_not_use_province(df)

        df = GeoMatcher.verify_coordination_with_address(
            df=df, verify_df=self.shape_gdf, check=self.check_df
        )

        df = TraffyDataCleaner.clean_type_columns(df, explode=False)
        df = TraffyDataCleaner.convert_to_date(df)
        return df

    # ------------------------------------------------------------------
    # MERGING WITH SCRAPED DATA (STATION + RAINFALL)
    # ------------------------------------------------------------------

    def merge_with_scraped_data(
        self,
        data: pd.DataFrame,
        station_path: str,
        rainfall_path: str,
    ) -> pd.DataFrame:
        """
        Reimplementation of your original merge_with_scraped_data()
        using clean, modular components.
        """
        logger.info("Merging cleaned Traffy data with station and rainfall scraped data")

        df = data.copy()

        # ---- load and clean station metadata
        station_df = DataIO.read_csv(station_path)
        station_df = StationRainMerger.clean_station(station_df)

        # ---- load and clean rainfall
        rainfall_df = DataIO.read_csv(rainfall_path)
        rainfall_df = StationRainMerger.clean_rainfall(rainfall_df)

        # ---- shapefile centroids
        shape_df = GeoMatcher.get_shape_file(self.shape_path)
        shape_df = GeoMatcher.get_centroid(shape_df)

        # ---- aggregate daily flood report metrics
        df_with_date = df.copy()
        # after convert_to_date, 'date' exists; ensure it's present
        if "date" not in df_with_date.columns and "timestamp" in df_with_date.columns:
            df_with_date = TraffyDataCleaner.convert_to_date(df_with_date)
        report_df = TraffyDataCleaner.get_report_detail(df_with_date)

        # ---- attach centroids to report df
        report_with_centroid = GeoMatcher.get_centroid_of_subdistrict(report_df, shape_df)

        # ---- nearest station for each report row
        report_with_station = GeoMatcher.make_station_info_nearest(
            report_with_centroid, station_df
        )

        # ---- also compute centroids + station mapping at subdistrict-level
        shape_with_station = GeoMatcher.make_station_info_nearest(shape_df, station_df)

        # ---- attach station_code to rainfall
        rain_and_station = StationRainMerger.get_station_code_for_rain(
            rainfall_df, shape_with_station
        )

        # ---- final merge rainfall + flood reports
        result = StationRainMerger.get_result_of_merge(
            report_df=report_with_station,
            rain_df=rain_and_station,
        )
        return result


# ----------------------------------------------------------------------
# Backwards-compatible functional API
# ----------------------------------------------------------------------


def get_cleaned_data(
    file_path: str,
    shape_path: str,
    csv_to_check_shape_path: str | None = None,
) -> pd.DataFrame:
    """
    Functional wrapper simulating your original get_cleaned_data().
    """
    df_raw = DataIO.read_csv(file_path)
    pipeline = FloodDataPipeline(
        shape_path=shape_path,
        csv_to_check_shape_path=csv_to_check_shape_path,
    )
    return pipeline.clean_base(df_raw)


def merge_with_scraped_data(
    data: pd.DataFrame,
    station_path: str,
    rainfall_path: str,
    shape_path: str,
    csv_to_check_shape_path: str | None = None,
) -> pd.DataFrame:
    """
    Functional wrapper simulating your original merge_with_scraped_data().
    """
    pipeline = FloodDataPipeline(
        shape_path=shape_path,
        csv_to_check_shape_path=csv_to_check_shape_path,
    )
    return pipeline.merge_with_scraped_data(
        data=data,
        station_path=station_path,
        rainfall_path=rainfall_path,
    )
