# src/common/dataprep/geo_utils.py
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)


class GeoMatcher:
    """
    Spatial helpers:
    - load and normalize BMA shapefile
    - verify subdistrict/district names
    - spatial join with Traffy coordinates
    - compute subdistrict centroids
    - assign nearest station via BallTree
    """

    @staticmethod
    def get_shape_file(file_path: str) -> gpd.GeoDataFrame:
        logger.info(f"Loading shape file: {file_path}")
        gdf = gpd.read_file(file_path)
        gdf = gdf.rename(
            columns={
                "SUBDISTRIC": "subdistrict_id",
                "SUBDISTR_1": "subdistrict",
                "DISTRICT_I": "district_id",
                "DISTRICT_N": "district",
                "CHANGWAT_N": "changwat",
                "Shape_Area": "shape_area",
            }
        )
        gdf = gdf.drop(
            columns=[
                "OBJECTID",
                "AREA_CAL",
                "AREA_BMA",
                "PERIMETER",
                "ADMIN_ID",
                "CHANGWAT_I",
                "Shape_Leng",
            ],
            errors="ignore",
        )
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        if gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf

    @staticmethod
    def verify_geopandas(shape_gdf: gpd.GeoDataFrame, check_df: pd.DataFrame | None = None):
        """
        Align subdistrict/district names with reference df (if provided).
        """
        if check_df is None:
            return shape_gdf

        logger.info("Verifying/aligning subdistrict & district names with reference CSV")
        shape_gdf = GeoMatcher.check_subdistrict(shape_gdf, check_df)
        shape_gdf = GeoMatcher.check_district(shape_gdf, check_df)
        return shape_gdf

    @staticmethod
    def check_subdistrict(shape_gdf: gpd.GeoDataFrame, subdistrict: pd.DataFrame) -> gpd.GeoDataFrame:
        shape_gdf = shape_gdf.copy()
        shape_gdf["subdistrict_id"] = shape_gdf["subdistrict_id"].astype("int")
        if len(shape_gdf.loc[~shape_gdf["subdistrict"].isin(subdistrict["sname"]), "subdistrict"]) != 0:
            subdistrict_dict = dict(zip(subdistrict["scode"], subdistrict["sname"]))
            shape_gdf["subdistrict"] = shape_gdf["subdistrict_id"].map(subdistrict_dict)
        return shape_gdf

    @staticmethod
    def check_district(shape_gdf: gpd.GeoDataFrame, district: pd.DataFrame) -> gpd.GeoDataFrame:
        shape_gdf = shape_gdf.copy()
        shape_gdf["district_id"] = shape_gdf["district_id"].astype("int")
        if len(shape_gdf.loc[~shape_gdf["district"].isin(district["dname"]), "district"]) != 0:
            district_dict = dict(zip(district["dcode"], district["dname"]))
            shape_gdf["district"] = shape_gdf["district_id"].map(district_dict)
        return shape_gdf

    @staticmethod
    def verify_coordination_with_address(
        df: pd.DataFrame,
        verify_df: gpd.GeoDataFrame,
        check: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Spatial join between Traffy coords and BMA polygons.
        Filters rows where district/subdistrict matches.
        """
        logger.info("Verifying coordinates against BMA shapefile")
        df_points = gpd.GeoDataFrame(
            df.copy(),
            geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
            crs="EPSG:4326",
        )

        if check is not None:
            verify_df = GeoMatcher.verify_geopandas(verify_df, check)

        if verify_df.crs.to_string() != "EPSG:4326":
            verify_df = verify_df.to_crs("EPSG:4326")

        joined = gpd.sjoin(df_points, verify_df, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]

        joined["check_dis"] = joined["district_left"] == joined["district_right"]
        joined["check_sub"] = joined["subdistrict_left"] == joined["subdistrict_right"]

        joined = joined[(joined["check_dis"]) & (joined["check_sub"])]

        drop_cols = [
            "index_right",
            "subdistrict_id",
            "district_id",
            "changwat",
            "geometry",
            "subdistrict_right",
            "district_right",
            "shape_area",
            "check_dis",
            "check_sub",
        ]
        joined = joined.drop(columns=drop_cols, errors="ignore")
        joined = joined.rename(
            columns={
                "district_left": "district",
                "subdistrict_left": "subdistrict",
            }
        )
        return joined

    @staticmethod
    def get_centroid(shape_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        shape_centers = shape_df.copy()
        logger.info("Computing centroids for subdistrict polygons")
        shape_centers["centroid"] = shape_centers.geometry.centroid.to_crs(epsg=4326)
        shape_centers["latitude"] = shape_centers["centroid"].y
        shape_centers["longitude"] = shape_centers["centroid"].x
        return shape_centers

    @staticmethod
    def get_centroid_of_subdistrict(df: pd.DataFrame, shape_df: gpd.GeoDataFrame) -> pd.DataFrame:
        logger.info("Merging report df with subdistrict centroids")
        data = df.merge(
            shape_df[["subdistrict", "district", "latitude", "longitude"]],
            how="left",
            on=["subdistrict", "district"],
        )
        return data

    @staticmethod
    def make_station_info_nearest(df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in df, assign nearest station in SAME district using BallTree.
        """
        df = df.copy()
        logger.info("Assigning nearest station per row using BallTree (by district)")

        n_rows = len(df)
        station_map = (
            station_df.groupby("district")
            .apply(lambda g: g[["station_code", "latitude", "longitude"]].to_dict("records"))
            .to_dict()
        )

        df["station_list"] = df["district"].map(station_map)
        df["station_code"] = np.nan
        df_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
        positions = np.arange(n_rows)

        for district, stations in station_map.items():
            idx = positions[df["district"].to_numpy() == district]
            if len(idx) == 0:
                continue

            if len(stations) == 1:
                df.loc[df.index[idx], "station_code"] = stations[0]["station_code"]
            else:
                st_coords = np.radians(
                    np.array([[s["latitude"], s["longitude"]] for s in stations])
                )
                tree = BallTree(st_coords, metric="haversine")
                dist, nearest_idx = tree.query(df_rad[idx], k=1)
                nearest_idx = nearest_idx.flatten()
                df.loc[df.index[idx], "station_code"] = [
                    stations[i]["station_code"] for i in nearest_idx
                ]

        df = df.drop(columns=["station_list"], axis="columns")
        return df
