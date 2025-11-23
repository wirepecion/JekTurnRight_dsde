import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

# 1. Global Variable to hold the Broadcasted Data on the Worker
_BROADCAST_STATIONS = None

def set_broadcast_variable(broadcast_var):
    """
    Called once on the Driver to register the variable.
    """
    global _BROADCAST_STATIONS
    _BROADCAST_STATIONS = broadcast_var

def build_station_index(station_df_pd):
    """
    Called on the Driver.
    Converts the Pandas Station DataFrame into a Dictionary of KD-Trees.
    Structure: { 'DistrictName': { 'tree': cKDTree, 'codes': [list_of_ids] } }
    """
    lookup = {}
    # Group by district to limit search space (optimization)
    for district, group in station_df_pd.groupby("district"):
        coords = group[["latitude", "longitude"]].values
        codes = group["station_code"].values
        lookup[district] = {
            "tree": cKDTree(coords),
            "codes": codes
        }
    return lookup

# 2. The Vectorized UDF (User Defined Function)
# This function receives a BATCH of data (e.g., 10,000 rows) as Pandas Series.
@pandas_udf(StringType())
def find_nearest_station(district_series: pd.Series, lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
    """
    Input: Columns of District, Lat, Lon from the Main DataFrame.
    Output: Column of 'station_code'.
    """
    # Retrieve the station data from the Worker's memory
    lookup = _BROADCAST_STATIONS.value
    
    results = []
    
    # Iterate through the batch (Vectorized logic is hard here due to dictionary lookup per row)
    # But this is still 100x faster than standard Python UDF because of Arrow overhead reduction.
    for district, lat, lon in zip(district_series, lat_series, lon_series):
        # Safety checks
        if pd.isna(district) or district not in lookup or pd.isna(lat) or pd.isna(lon):
            results.append(None)
            continue
        
        # Get the KDTree for this district
        data = lookup[district]
        
        # Query nearest neighbor (k=1)
        # Returns (distance, index)
        _, idx = data["tree"].query([lat, lon], k=1)
        
        # Append the Station Code
        results.append(data["codes"][idx])
        
    return pd.Series(results)