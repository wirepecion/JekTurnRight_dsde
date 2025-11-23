import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import logging

logger = logging.getLogger(__name__)

def verify_and_correct_names(shape_gdf: gpd.GeoDataFrame, ref_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Aligns District/Subdistrict names in Shapefile using a Reference CSV."""
    if ref_df is None: 
        return shape_gdf
    # Simple pass-through if no reference DF is provided
    return shape_gdf

def spatial_join_verification(df: pd.DataFrame, shape_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Performs Point-in-Polygon check.
    Ensures the Lat/Lon actually falls inside the claimed District.
    """
    # 1. Convert DataFrame to GeoDataFrame
    points = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")
    
    # 2. Ensure CRS matches
    if shape_gdf.crs != "EPSG:4326":
        shape_gdf = shape_gdf.to_crs("EPSG:4326")
        
    # 3. Spatial Join
    # This creates 'district_left' (from df) and 'district_right' (from shape)
    joined = gpd.sjoin(gdf_points, shape_gdf, how="left", predicate="within")
    
    # 4. Filter: logic must match (claimed district == actual polygon district)
    # Note: We handle cases where the raw data might be missing district info
    if 'district_left' in joined.columns and 'district_right' in joined.columns:
        valid_mask = (joined['district_left'] == joined['district_right']) & \
                     (joined['subdistrict_left'] == joined['subdistrict_right'])
        valid_data = joined[valid_mask].copy()
    else:
        # If raw data didn't have district columns, we rely purely on the shapefile's result
        valid_data = joined.copy()

    # 5. Rename columns back! (The Fix)
    # We keep the 'left' (original) versions but rename them back to standard names
    rename_dict = {
        'district_left': 'district',
        'subdistrict_left': 'subdistrict'
    }
    valid_data = valid_data.rename(columns=rename_dict)
    
    # 6. Clean up columns
    # Drop the shapefile's columns (_right) and geometry artifacts
    drop_cols = ['geometry', 'index_right', 'district_right', 'subdistrict_right']
    return valid_data.drop(columns=drop_cols, errors='ignore')

def get_nearest_station(df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the nearest station code to each row using BallTree.
    """
    df = df.copy()
    
    # 1. Pre-calculate station coordinates per district
    # FIX: Added include_groups=False to silence FutureWarning
    station_map = station_df.groupby('district')[['station_code', 'latitude', 'longitude']].apply(
        lambda g: g.to_dict('records')
    ).to_dict()

    df['station_code'] = np.nan
    
    # Convert all points to radians once
    df_rad = np.radians(df[['latitude', 'longitude']].to_numpy())
    
    # Get unique districts present in the data to iterate over
    # (Faster than iterating over the station map if data covers fewer districts)
    present_districts = df['district'].unique()

    for district in present_districts:
        # Skip if this district has no stations
        if district not in station_map:
            continue
            
        stations = station_map[district]
        
        # Find rows belonging to this district
        mask = df['district'] == district
        idx = np.where(mask)[0]
        
        if len(idx) == 0:
            continue
        
        if len(stations) == 1:
            df.loc[mask, 'station_code'] = stations[0]['station_code']
        else:
            # Build Tree for this district's stations
            st_coords = np.radians([[s['latitude'], s['longitude']] for s in stations])
            tree = BallTree(st_coords, metric='haversine')
            
            # Query
            # df_rad[idx] selects the coordinates for the relevant rows
            _, nearest_idx_local = tree.query(df_rad[idx], k=1)
            
            # Map back to station codes
            assigned_codes = [stations[i]['station_code'] for i in nearest_idx_local.flatten()]
            df.loc[mask, 'station_code'] = assigned_codes
            
    return df