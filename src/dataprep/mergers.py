import pandas as pd
from .cleaning import impute_missing

def clean_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'StationCode': 'station_code', 'StationName_Short': 'station_name',
        'DistrictName': 'district', 'Latitude': 'latitude', 'Longitude': 'longitude'
    })
    # Hardcoded fixes (Technical Debt, but necessary for now)
    df['district'] = df['district'].replace({
        'ป้อมปราบฯ': 'ป้อมปราบศัตรูพ่าย',
        'ราษฏร์บูรณะ': 'ราษฎร์บูรณะ'
    })
    # Remove prefix logic
    mask = df['station_code'].str.len() == 9
    df.loc[mask, 'station_code'] = df.loc[mask, 'station_code'].str[3:]
    return df

def clean_rainfall_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop bad columns
    df = df.drop(columns=['NKM.03','LSI.02'], errors='ignore')
    
    # 2. Impute
    df = impute_missing(df, df.columns[df.isna().any()].tolist(), 'most_frequent')
    
    # 3. Melt (Wide to Long)
    df = df.rename(columns={'Date': 'date'})
    df_long = df.melt(id_vars='date', var_name='station_code', value_name='rainfall')
    
    # 4. Fix Dates
    df_long['date'] = pd.to_datetime(df_long['date'], format='mixed').dt.date
    return df_long

def merge_rainfall_with_reports(
    report_df: pd.DataFrame, 
    rain_df: pd.DataFrame, 
    station_df: pd.DataFrame  # <--- NEW ARGUMENT
) -> pd.DataFrame:
    
    # 1. Aggregate Reports (Counts per day/station)
    report_agg = report_df.groupby(['date', 'station_code']).agg(
        flood_count=('type_list', lambda x: x.apply(lambda l: 'น้ำท่วม' in l).sum()),
        total_report=('ticket_id', 'count')
    ).reset_index()
    
    # 2. Merge Rain + Flood Counts
    merged = rain_df.merge(report_agg, on=['date', 'station_code'], how='left')
    
    # 3. Fill NaN (No report = 0 floods)
    merged[['flood_count', 'total_report']] = merged[['flood_count', 'total_report']].fillna(0)
    
    # 4. Attach Location Info (Lat/Lon) from Station Data
    # We only need the coordinates and station_code
    station_coords = station_df[['station_code', 'latitude', 'longitude']].drop_duplicates()
    
    final_df = merged.merge(station_coords, on='station_code', how='left')
    
    return final_df