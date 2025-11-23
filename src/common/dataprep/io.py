# src/common/dataprep/io.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from .geo_utils import GeoMatcher

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


class DataIO:
    """
    Small wrapper around reading CSV / shapefiles so the rest of the
    code doesn't care about paths and formats.
    """

    @staticmethod
    def read_csv(path: PathLike, **kwargs) -> pd.DataFrame:
        path = Path(path)
        logger.info(f"Reading CSV: {path}")
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def read_shape(path: PathLike) -> gpd.GeoDataFrame:
        path = Path(path)
        logger.info(f"Reading shapefile: {path}")
        return GeoMatcher.get_shape_file(str(path))
