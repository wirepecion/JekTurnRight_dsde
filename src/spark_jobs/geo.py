"""
src/spark_jobs/geo.py
---------------------
Spark-native Geospatial Logic.
Uses Broadcast Variables to perform 'Point-in-Polygon' and 'Nearest Neighbor'
checks in parallel on workers.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType, BooleanType

# --- GLOBAL BROADCAST VARIABLES (Worker Side) ---
# These are None on the Driver. They get populated on Workers.
_BROADCAST_SHAPE = None
_BROADCAST_STATIONS = None

def set_shape_broadcast(bc_shape):
    """Called by Driver to register shapefile broadcast."""
    global _BROADCAST_SHAPE
    _BROADCAST_SHAPE = bc_shape

def set_station_broadcast(bc_stations):
    """Called by Driver to register station broadcast."""
    global _BROADCAST_STATIONS
    _BROADCAST_STATIONS = bc_stations

# --- 1. SPATIAL VERIFICATION UDF ---

@pandas_udf(BooleanType())
def verify_in_bma_udf(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """
    Checks if lat/lon is inside ANY subdistrict in the broadcasted shapefile.
    Returns True/False.
    """
    # 1. Get Data from Broadcast
    if _BROADCAST_SHAPE is None:
        raise RuntimeError("Broadcast shapefile not set! Call set_shape_broadcast first.")
    
    gdf_shape = _BROADCAST_SHAPE.value # This is the GeoDataFrame
    
    # 2. Convert inputs to Points
    # Note: We create a GeoDataFrame on the fly for this BATCH of data
    points = [Point(xy) for xy in zip(lon, lat)]
    gdf_points = gpd.GeoDataFrame({'geometry': points}, crs="EPSG:4326")
    
    # 3. Spatial Join (Fast for small shapefiles like BMA)
    # We just want to know if it intersects ANY polygon
    joined = gpd.sjoin(gdf_points, gdf_shape, how="inner", predicate="within")
    
    # 4. Return Boolean Series (aligned with input index)
    # If the index exists in 'joined', it was found.
    return pd.Series(gdf_points.index.isin(joined.index))

# --- 2. NEAREST STATION UDF ---

@pandas_udf(StringType())
def get_nearest_station_udf(district: pd.Series, lat: pd.Series, lon: pd.Series) -> pd.Series:
    """
    Finds nearest station code. Uses PRE-BUILT BallTree.
    """
    if _BROADCAST_STATIONS is None:
        raise RuntimeError("Broadcast stations not set!")
    
    lookup_map = _BROADCAST_STATIONS.value 
    results = []
    
    for d, la, lo in zip(district, lat, lon):
        # 1. Validation
        if pd.isna(d) or pd.isna(la) or pd.isna(lo):
            results.append(None)
            continue
            
        # 2. Smart Lookup (Fuzzy Match)
        target_data = lookup_map.get(d)
        if target_data is None and d.startswith("เขต"):
            target_data = lookup_map.get(d.replace("เขต", ""))
        if target_data is None and not d.startswith("เขต"):
            target_data = lookup_map.get(f"เขต{d}")

        if target_data is None:
            results.append(None)
            continue
            
        # 3. FAST Query (No building, just querying)
        if len(target_data['codes']) == 1:
            results.append(target_data['codes'][0])
        elif target_data['tree'] is not None:
            # Reuse the pre-built tree
            p_rad = np.radians([[la, lo]])
            dist, idx = target_data['tree'].query(p_rad, k=1)
            results.append(target_data['codes'][idx[0][0]])
        else:
            results.append(None)
        
    return pd.Series(results)# --- HELPER: PREPARE DATA FOR BROADCASTING ---

def prepare_station_lookup(station_df: pd.DataFrame) -> dict:
    """
    Converts Station DataFrame into a fast Lookup Dictionary.
    NOW PRE-BUILDS THE BALLTREE to save RAM.
    """
    lookup = {}
    for district, group in station_df.groupby('district'):
        coords = group[['latitude', 'longitude']].values
        codes = group['station_code'].values
        
        # Pre-build the tree here!
        tree = None
        if len(codes) > 1:
            tree = BallTree(np.radians(coords), metric='haversine')
            
        lookup[district] = {
            'coords': coords,
            'codes': codes,
            'tree': tree  # <--- Storing the heavy object
        }
    return lookup
    """
    Converts Station DataFrame into a fast Lookup Dictionary.
    Output: { 'DistrictName': {'coords': [[lat, lon], ...], 'codes': ['S01', ...]} }
    """
    lookup = {}
    for district, group in station_df.groupby('district'):
        lookup[district] = {
            'coords': group[['latitude', 'longitude']].values,
            'codes': group['station_code'].values
        }
    return lookup