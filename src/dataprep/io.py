import pandas as pd
import geopandas as gpd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_csv(file_path: str) -> pd.DataFrame:
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        logger.info(f"Loading CSV: {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

def load_shapefile(file_path: str) -> gpd.GeoDataFrame:
    """Loads and standardizes the BMA Shapefile."""
    logger.info(f"Loading Shapefile: {file_path}")
    gdf = gpd.read_file(file_path)
    
    # Standardize Column Names immediately upon load
    rename_map = {
        'SUBDISTRIC': 'subdistrict_id',
        'SUBDISTR_1': 'subdistrict',
        'DISTRICT_I': 'district_id',
        'DISTRICT_N': 'district',
        'CHANGWAT_N': 'changwat',
        'Shape_Area': 'shape_area'
    }
    gdf.rename(columns=rename_map, inplace=True)
    
    # Drop useless columns immediately
    drop_cols = ['OBJECTID', 'AREA_CAL', 'AREA_BMA', 'PERIMETER', 'ADMIN_ID', 'CHANGWAT_I', 'Shape_Leng']
    gdf.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return gdf