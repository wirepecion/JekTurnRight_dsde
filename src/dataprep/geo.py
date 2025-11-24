import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import logging

logger = logging.getLogger(__name__)

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
    if 'district_left' in joined.columns and 'district_right' in joined.columns:
        valid_mask = (joined['district_left'] == joined['district_right']) & \
                     (joined['subdistrict_left'] == joined['subdistrict_right'])
        valid_data = joined[valid_mask].copy()
    else:
        valid_data = joined.copy()

    # 5. Rename columns back
    rename_dict = {
        'district_left': 'district',
        'subdistrict_left': 'subdistrict'
    }
    valid_data = valid_data.rename(columns=rename_dict)
    
    # 6. Clean up columns
    drop_cols = ['geometry', 'index_right', 'district_right', 'subdistrict_right']
    return valid_data.drop(columns=drop_cols, errors='ignore')

# ... existing imports ...

def get_subdistrict_centroids(shape_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Converts polygon shapefile to a DataFrame of centroids (lat/lon).
    Used to map every subdistrict to a weather station.
    """
    gdf = shape_gdf.copy()
    # Convert to centroid points
    # Warning: Ideally project to meters (UTM) before centroid, but EPSG:4326 is acceptable for small areas like BKK
    centroids = gdf.geometry.centroid.to_crs(epsg=4326)
    
    return pd.DataFrame({
        'subdistrict': gdf['subdistrict'],
        'district': gdf['district'],
        'latitude': centroids.y,
        'longitude': centroids.x
    })

def get_nearest_station(df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the nearest station code to each row using BallTree.
    Optimized: Reduced object creation inside loops.
    """
    df = df.copy()
    
    # Pre-calculate station coordinates per district
    # include_groups=False silences Pandas FutureWarnings
    station_map = station_df.groupby('district')[['station_code', 'latitude', 'longitude']].apply(
        lambda g: g.to_dict('records'), 
        include_groups=False
    ).to_dict()

    df['station_code'] = np.nan
    
    # Convert all points to radians once
    df_rad = np.radians(df[['latitude', 'longitude']].to_numpy())
    
    # Iterate only over districts present in the data
    present_districts = df['district'].unique()

    for district in present_districts:
        if district not in station_map:
            continue
            
        stations = station_map[district]
        mask = df['district'] == district
        idx = np.where(mask)[0]
        
        if len(idx) == 0: continue
        
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